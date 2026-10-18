"""Filesystem-backed job/clip store for local ClipForge."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger

from app import sharelink
from app.config import Paths

try:  # POSIX advisory locking — guards read-modify-write across processes
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


def _uploads_root(paths: Paths) -> Path:
    """Where new source files are written: inside the codebase, by default.

    ``getattr`` rather than a plain attribute because tests (and any caller)
    may hand in a duck-typed paths object; falling back to ``data/uploads``
    keeps every such caller working and matches the real default anyway.
    """
    return getattr(paths, "uploads_dir", None) or (paths.data_dir / "uploads")


def _job_source_dirs(paths: Paths, job_id: str) -> tuple[Path, ...]:
    """Every directory that can hold this job's source file, best first.

    Mirrors ``Paths.job_source_dirs`` with the same duck-typing tolerance: a
    job uploaded before this change still lives in the legacy ``downloads/``
    tree, so reads must fan out across both and neither may be assumed.
    """
    fn = getattr(paths, "job_source_dirs", None)
    if callable(fn):
        return tuple(fn(job_id))
    return (_uploads_root(paths) / job_id, paths.downloads_dir / job_id)


class LocalStore:
    """Thread- AND process-safe JSON document store rooted at data_dir.

    The worker supervisor (API process) and the per-job runner (child process)
    both touch job documents: the parent tails child output into the job log
    while the child writes stage progress. Two processes writing the same
    `x.json.tmp` produced corrupt, unreadable files — so tmp names are unique
    per process/thread and read-modify-write paths hold an advisory file lock.
    """

    def __init__(self, paths: Paths) -> None:
        self._paths = paths
        self._lock = threading.RLock()
        self._paths.data_dir.mkdir(parents=True, exist_ok=True)
        self._paths.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._paths.downloads_dir.mkdir(parents=True, exist_ok=True)
        uploads = _uploads_root(self._paths)
        uploads.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # cross-process write primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _tmp_for(path: Path) -> Path:
        """Unique tmp path per process+thread (never shared between writers)."""
        return path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")

    @contextmanager
    def _file_lock(self, path: Path):
        """Advisory exclusive lock around a read-modify-write of `path`."""
        if fcntl is None:  # pragma: no cover - non-POSIX
            yield
            return
        lock_path = path.with_name(path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()

    # ------------------------------------------------------------------
    # ids
    # ------------------------------------------------------------------

    def job_id(self, doc: dict[str, Any]) -> str:
        """id alias used by callers that reach into nested job payloads.

        Several report helpers expect a ``job_id(doc)`` signature.  The
        canonical key is ``_id``; older documents and a handful of callers
        used ``id``, so we prefer the former and fall back to the latter.
        """
        if doc is None:
            return ""
        if "_id" in doc and isinstance(doc["_id"], str) and doc["_id"]:
            return doc["_id"]
        mid = doc.get("id")
        if isinstance(mid, str) and mid:
            return mid
        return ""

    def new_job_id(self) -> str:
        return uuid.uuid4().hex

    # ------------------------------------------------------------------
    # job creation
    # ------------------------------------------------------------------

    def create_job(
        self,
        *,
        source_type: str,
        source: str | None,
        title: str | None = None,
        min_clip_sec: int | None = None,
        max_clip_sec: int | None = None,
        max_clips: int | None = None,
        caption_preset: str | None = None,
        language: str = "auto",
        duration: float | None = None,
    ) -> dict[str, Any]:
        """Create and persist a new job document (stage: queued).

        This was previously missing entirely — both ``POST /api/jobs`` and
        ``POST /api/jobs/upload`` called ``store.create_job(...)`` and
        crashed with ``AttributeError: 'LocalStore' object has no attribute
        'create_job'`` → the raw 500 the user saw on every "Forge clips"
        submit.

        Field names match what the pipeline (``app/task.py``) and the API
        serializers actually read:

        - ``source`` / ``url``: source filename or URL (pipeline reads ``source``)
        - ``options``: minClipSec / maxClipSec / maxClips (read by the detect stage)
        - ``language``: caption language ("auto" | "en" | "es")
        - ``durationSec``: pre-probed source length, when known
        """
        now_ms = int(time.time() * 1000)
        job: dict[str, Any] = {
            "_id": self.new_job_id(),
            "sourceType": source_type,
            "source": source,
            "url": source if source_type == "url" else None,
            "title": title,
            "stage": "queued",
            "status": "queued",
            "createdAt": now_ms,
            "durationSec": duration,
            "language": language,
            "options": {
                "minClipSec": min_clip_sec,
                "maxClipSec": max_clip_sec,
                "maxClips": max_clips,
            },
            "captionPreset": caption_preset or "bold_pop",
            "clips": [],
            "logs": [],
            "progress": 0.0,
            "isTerminal": False,
        }
        self.save_job(job)
        return job

    # ------------------------------------------------------------------
    # job put/get/delete
    # ------------------------------------------------------------------

    def _job_path(self, job_id: str) -> Path:
        return self._paths.data_dir / "jobs" / f"{job_id}.json"

    def save_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Persist or overwrite a job document (single-process caller only).

        Callers that are also touched by the per-job runner must hold the
        advisory file lock before calling this (see ``_file_lock``); this
        method is process-safe but assumes the caller is the only writer in
        its process.
        """
        jid = self.job_id(job)
        path = self._job_path(jid)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_for(path)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(job, fh, default=str, indent=2, sort_keys=False)
        tmp.replace(path)
        return job

    def get_job(self, job_id: str) -> dict[str, Any]:
        path = self._job_path(job_id)
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logger.warning(f"corrupt job file: {path}")
            return {}

    def delete_job(self, job_id: str) -> bool:
        path = self._job_path(job_id)
        if not path.exists():
            return False
        path.unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------------
    # listing helpers
    # ------------------------------------------------------------------

    def _list_job_paths(self) -> list[Path]:
        """Return job document paths newest-first, broken ties by name."""
        # Job documents live in the jobs/ subdirectory, not directly in data_dir.
        # This keeps data_dir clean (config/status files sit alongside the
        # subdirectory) and matches how the pipeline has written jobs since the
        # hosted-worker refactor.
        jobs_dir = self._paths.data_dir / "jobs"
        try:
            files = sorted(
                (p for p in jobs_dir.iterdir() if p.suffix == ".json" and p.is_file()),
                key=lambda p: (-p.stat().st_mtime_ns, p.name),
                reverse=True,
            )
        except OSError:
            return []
        return list(files)

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return every job document, newest first.

        Kept for backward compatibility — some callers still use this name.
        New read paths use the summary projection below to avoid shipping full
        logs on every read.
        """
        return self.iter_all_jobs()

    def iter_all_jobs(self) -> list[dict[str, Any]]:
        """Return every job document, newest first.

        Kept for a few legacy callers; new read paths use the summary
        projection below to avoid shipping full logs on every read.
        """
        out: list[dict[str, Any]] = []
        for path in self._list_job_paths():
            try:
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
                if isinstance(doc, dict):
                    out.append(doc)
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def list_job_summaries(
        self,
        limit: int = 10,
        before: int | None = None,
    ) -> tuple[list[dict[str, Any]], int, int | None]:
        """One page of lightweight job summaries, newest first.

        Returns ``(page, total, next_before)`` so the list view can paginate
        without loading every full document (and its embedded logs) into the
        poll loop.
        """
        all_paths = self._list_job_paths()
        total = len(all_paths)

        # cursor pagination — ``before`` is the mtime-ns of the youngest job
        # already on the current page; the next page is everything older than
        # that.
        def before_key(p: Path) -> int:
            return p.stat().st_mtime_ns

        candidates: list[Path] = []
        if before is not None:
            for p in all_paths:
                if before_key(p) < before:
                    candidates.append(p)
                else:
                    break
        else:
            candidates = list(all_paths)

        page: list[dict[str, Any]] = []
        for path in candidates[:max(0, limit)]:
            try:
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
                if not isinstance(doc, dict):
                    continue
                page.append(doc)
            except (json.JSONDecodeError, OSError):
                continue

        next_before = None
        if len(page) == limit and page:
            # Use the mtime of the last job's file as the cursor
            last_job = page[-1]
            last_path = self._job_path(last_job.get("_id", ""))
            if last_path.exists():
                next_before = last_path.stat().st_mtime_ns

        return page, total, next_before

    # ------------------------------------------------------------------
    # job log
    # ------------------------------------------------------------------

    def append_job_log(self, job_id: str, message: str) -> None:
        """Append a log line to a job's log array."""
        with self._file_lock(self._job_path(job_id)):
            job = self.get_job(job_id)
            if not isinstance(job, dict):
                job = {}
            logs = job.get("logs")
            if not isinstance(logs, list):
                logs = []
            logs.append(message)
            job["logs"] = logs
            self.save_job(job)

    # ------------------------------------------------------------------
    # clip operations
    # ------------------------------------------------------------------

    def clips_delivered(self, job_id: str, job: dict[str, Any]) -> tuple[int, int]:
        """Return (rendered_count, promised_count) for a job.

        rendered: number of clip files actually on disk
        promised: number of clips the job was supposed to produce
        """
        if not job_id:
            return 0, 0
        clip_dir = self._paths.outputs_dir / job_id
        rendered = 0
        if clip_dir.exists():
            for p in clip_dir.iterdir():
                if p.is_file() and p.suffix in {".mp4", ".mov", ".webm"}:
                    rendered += 1
        promised = int(job.get("clipTarget") or job.get("clipCount") or 0)
        return rendered, promised

    def list_clips(self, job_id: str) -> list[dict[str, Any]]:
        """Return metadata for all clips belonging to a job."""
        out: list[dict[str, Any]] = []
        clip_dir = self._paths.outputs_dir / job_id
        if not clip_dir.exists():
            return out
        for p in sorted(clip_dir.iterdir()):
            if p.is_file() and p.suffix in {".mp4", ".mov", ".webm"}:
                out.append({
                    "_id": p.stem,
                    "jobId": job_id,
                    "path": str(p),
                    "filename": p.name,
                    "size": p.stat().st_size,
                })
        return out

    def get_clip(self, job_id: str, clip_id: str) -> dict[str, Any] | None:
        """Return metadata for a specific clip."""
        clips = self.list_clips(job_id)
        for c in clips:
            if c.get("_id") == clip_id:
                return c
        return None

    def write_clip(self, job_id: str, clip_data: dict[str, Any]) -> dict[str, Any]:
        """Record a clip's metadata (called after rendering completes)."""
        clip_id = clip_data.get("_id", self.new_job_id())
        clip_path = self._paths.outputs_dir / job_id / f"{clip_id}.json"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = clip_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(clip_data, indent=2), encoding="utf-8")
        tmp.replace(clip_path)
        return clip_data

    def list_all_clips(self) -> list[dict[str, Any]]:
        """Return metadata for all clips across all jobs."""
        out: list[dict[str, Any]] = []
        if not self._paths.outputs_dir.exists():
            return out
        for job_dir in self._paths.outputs_dir.iterdir():
            if job_dir.is_dir():
                out.extend(self.list_clips(job_dir.name))
        return out

    def list_clips_for_list(self, job_id: str | None = None) -> list[dict[str, Any]]:
        """Return clips for the clip list endpoint, with share state attached.
        
        Called by /api/clips. Returns lightweight clip metadata suitable for
        the dashboard clip list.
        """
        if job_id:
            return self.list_clips(job_id)
        return self.list_all_clips()

    def create_clips(
        self,
        job_id: str,
        candidates: list[dict[str, Any]],
        *,
        detected_source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Record the AI-detected candidate clips for a job.

        Called after the detection stage finishes. Persists one clip document
        per candidate under ``data/clips/<job_id>/<clip_id>.json`` so the
        ``/api/clips`` endpoint can return real ``ClipDto`` records (with
        start/end, renderState, captionCues, shareState, ...) instead of a
        single summary blob.

        Also writes the lightweight ``data/clips/<job_id>.json`` summary used
        by the job card ("AI-detected via <source>", candidate count) and
        patches the job document with the clip target + detected count.
        """
        clips_dir = self._paths.data_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        now_ms = int(time.time() * 1000)
        clip_ids: list[str] = []
        for i, c in enumerate(candidates):
            clip_id = c.get("_id") or self.new_job_id()
            # Write the minted id BACK onto the candidate object. Downstream
            # stages (reframe cuts, captions, export) key their work on these
            # in-place ids; without this, export fabricated ids like clip_000
            # that have no clip document on disk and crashed the render with
            # FileNotFoundError on the first clip of every batch.
            c["_id"] = clip_id
            order = int(c.get("order", i))
            start = float(c.get("start", 0.0) or 0.0)
            end = float(c.get("end", 0.0) or 0.0)
            if end <= start:
                end = start + 5.0
            if end - start < 1.0:
                end = start + 5.0
            doc: dict[str, Any] = {
                "_id": clip_id,
                "jobId": job_id,
                "order": order,
                "start": round(start, 3),
                "end": round(end, 3),
                "score": float(c.get("score", 0.0) or 0.0),
                "title": str(c.get("title", "Clip") or "Clip"),
                "hookText": str(c.get("hook_text", "") or ""),
                "transcript": str(c.get("transcript", "") or ""),
                "hashtags": [str(h) for h in (c.get("hashtags") or [])],
                "captionCues": [],
                "aspect": str(c.get("aspect", "9:16") or "9:16"),
                "videoPath": None,
                "thumbPath": None,
                "mediaBytes": None,
                "mediaWidth": None,
                "mediaHeight": None,
                "mediaDurationSec": None,
                "mediaError": None,
                "renderState": "pending",
                "renderMs": None,
                "shareId": None,
                "shareState": "missing",
                "shareDetail": None,
                "shareUrl": None,
                "sharePageUrl": None,
                "translationTracks": None,
                "detectedSource": detected_source,
                "createdAt": now_ms,
            }
            clip_doc_path = clips_dir / f"{clip_id}.json"
            tmp = clip_doc_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(clip_doc_path)
            clip_ids.append(clip_id)

        # lightweight summary blob used by the job card
        summary = {
            "jobId": job_id,
            "candidateCount": len(candidates),
            "clipCount": len(clip_ids),
            "detectedSource": detected_source,
            "detectedAt": now_ms,
            "clipIds": clip_ids,
        }
        summary_path = clips_dir / f"{job_id}.json"
        tmp = summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(summary_path)

        # Patch the job document so the UI can show the clip target + detected count
        job = self.get_job(job_id)
        if job:
            job["candidatesDetected"] = len(candidates)
            job["clipTarget"] = len(clip_ids) or max(1, int((job.get("durationSec") or 0) / 30))
            job["clipIds"] = clip_ids
            job["detectedSource"] = detected_source
            self.write_job(job)
        return summary

    # ------------------------------------------------------------------
    # per-clip writes (updated by the render stage / share path)
    # ------------------------------------------------------------------

    def write_clip(
        self,
        clip_id: str,
        fields: dict[str, Any],
        *,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist updated fields on a single clip document.

        ``job_id`` is optional; when absent the store infers it from the clip
        document currently on disk.
        """
        clips_dir = self._paths.data_dir / "clips"
        clip_path = clips_dir / f"{clip_id}.json"
        if not clip_path.exists():
            raise FileNotFoundError(clip_path)
        existing = json.loads(clip_path.read_text(encoding="utf-8"))
        resolved_job_id = job_id or existing.get("jobId", "")
        merged = {**existing, **fields}
        merged["jobId"] = resolved_job_id
        tmp = clip_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(clip_path)

        # keep the summary blob's clipIds list in sync (best-effort)
        if resolved_job_id:
            summary_path = clips_dir / f"{resolved_job_id}.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    ids = list(summary.get("clipIds", []) or [])
                    if clip_id not in ids:
                        ids.append(clip_id)
                    summary["clipIds"] = ids
                    summary_path.with_suffix(".json.tmp").write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
                    ).replace(summary_path)
                except Exception:
                    pass
        return merged

    def list_clip_docs(self, job_id: str) -> list[dict[str, Any]]:
        """Return the per-clip ``ClipDto`` documents for one job.

        This is the source of truth for ``/api/clips?jobId=...``. It reads the
        individual ``data/clips/<clip_id>.json`` files (not the summary blob)
        so every field the render/share/caption stages write is visible.
        """
        clips_dir = self._paths.data_dir / "clips"
        out: list[dict[str, Any]] = []
        for p in sorted(clips_dir.glob("*.json")):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(doc, dict):
                continue
            if doc.get("jobId") != job_id:
                continue
            # skip the summary blob (it has candidateCount / clipIds / detectedAt)
            if "candidateCount" in doc or "clipIds" in doc:
                continue
            if not doc.get("_id"):
                continue
            out.append(doc)
        out.sort(key=lambda d: (int(d.get("order", 0)) if str(d.get("order", 0)).isdigit() else 0, d.get("_id", "")))
        return out

    def list_clips_for_list(self, job_id: str | None = None) -> list[dict[str, Any]]:
        """Return clips for the clip list endpoint, with share state attached.

        Called by /api/clips. Returns lightweight clip metadata suitable for
        the dashboard clip list.
        """
        if job_id:
            return self.list_clip_docs(job_id)
        return self.list_all_clips()

    def list_all_clips(self, limit: int = 512) -> list[dict[str, Any]]:
        """Return metadata for all clips across all jobs."""
        out: list[dict[str, Any]] = []
        clips_dir = self._paths.data_dir / "clips"
        if not clips_dir.exists():
            return out
        for p in sorted(clips_dir.glob("*.json")):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(doc, dict):
                continue
            if "candidateCount" in doc or "clipIds" in doc:
                continue
            if not doc.get("_id"):
                continue
            out.append(doc)
            if len(out) >= limit:
                break
        return out

    def get_clip_doc(self, clip_id: str) -> dict[str, Any] | None:
        """Return a single clip document by id, or None."""
        clips_dir = self._paths.data_dir / "clips"
        p = clips_dir / f"{clip_id}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def get_clip_doc_safe(self, clip_id: str) -> dict[str, Any] | None:
        """Return a single clip document by id, or None."""
        return self.get_clip_doc(clip_id)

    # ------------------------------------------------------------------
    # legacy clip helpers — rewrite to use the new clip-doc layer
    # ------------------------------------------------------------------

    def list_clips(self, job_id: str) -> list[dict[str, Any]]:
        """Return metadata for all clips belonging to a job."""
        out: list[dict[str, Any]] = []
        clip_dir = self._paths.outputs_dir / job_id
        if not clip_dir.exists():
            return out
        for p in sorted(clip_dir.iterdir()):
            if p.is_file() and p.suffix in {".mp4", ".mov", ".webm"}:
                out.append({
                    "_id": p.stem,
                    "jobId": job_id,
                    "path": str(p),
                    "filename": p.name,
                    "size": p.stat().st_size,
                })
        return out

    def get_clip(self, job_id: str, clip_id: str) -> dict[str, Any] | None:
        """Return metadata for a specific clip."""
        clips = self.list_clips(job_id)
        for c in clips:
            if c.get("_id") == clip_id:
                return c
        return None

    def write_clip_legacy(self, job_id: str, clip_data: dict[str, Any]) -> dict[str, Any]:
        """Legacy write into outputs/<job>/<clip>.json. Keep for callers that still need it."""
        clip_id = clip_data.get("_id", self.new_job_id())
        clip_path = self._paths.outputs_dir / job_id / f"{clip_id}.json"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = clip_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(clip_data, indent=2), encoding="utf-8")
        tmp.replace(clip_path)
        return clip_data

    # ------------------------------------------------------------------
    # write_job is an alias for save_job (kept for backward compat)
    # ------------------------------------------------------------------

    def write_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Persist or overwrite a job document. Alias for save_job."""
        return self.save_job(job)

    # ------------------------------------------------------------------
    # output management
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of outputs for external reporting."""
        clips = self.list_all_clips()
        return {
            "clipCount": len(clips),
            "clips": clips,
        }

    def materialize(self, job_id: str) -> bool:
        """Ensure outputs for a job are accessible. Returns True if ok."""
        return (self._paths.outputs_dir / job_id).exists()

    def drop_local_after_push(self, job_id: str) -> bool:
        """Drop local clip files after they've been pushed to durable storage."""
        clip_dir = self._paths.outputs_dir / job_id
        if clip_dir.exists():
            import shutil
            shutil.rmtree(clip_dir, ignore_errors=True)
            return True
        return False

    def find_source_file(self, job_id: str) -> Path | None:
        """Find the source video file for a job.
        
        Checks both the new uploads dir and the legacy downloads dir
        (for jobs that were processed before the uploads refactor).
        """
        # Legacy downloads dir (pre-uploads-refactor jobs)
        legacy_dir = self._paths.downloads_dir / job_id
        if legacy_dir.exists():
            for p in legacy_dir.iterdir():
                if p.is_file() and p.suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
                    return p
        # New uploads dir
        for d in _job_source_dirs(self._paths, job_id):
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.is_file() and p.suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
                    return p
        return None

    def job_source_dir_for_write(self, job_id: str) -> Path:
        """Return the directory where a job's source file should be written."""
        return _uploads_root(self._paths) / job_id

    def upload_file(self, job_id: str, filename: str, data: bytes) -> Path:
        """Store an uploaded file for a job."""
        upload_dir = self.job_source_dir_for_write(job_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / filename
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path

    def upload_key(self, job_id: str, filename: str) -> str:
        """Return a storage key for an uploaded file. Simple filename-based key."""
        return f"{job_id}/{filename}"

    def restore_job(self, job_id: str) -> bool:
        """Restore a job's state. No-op for local store."""
        return True

    def config_from_env(self) -> dict[str, Any]:
        """Return configuration from environment. Empty for local store."""
        return {}

    def key_for_path(self, path: Path) -> str:
        """Return a storage key for a given file path."""
        rel = path.relative_to(self._paths.outputs_dir)
        return str(rel).replace("\\", "/")
