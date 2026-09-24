"""Filesystem-backed job/clip store for local ClipForge.

Jobs and clips live as JSON documents under ``data/`` so state survives
restarts without any external database. All writes are atomic (tmp + rename).
"""

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

from app.config import Paths

try:  # POSIX advisory locking — guards read-modify-write across processes
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


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

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    # jobs
    # ------------------------------------------------------------------

    def _job_path(self, job_id: str) -> Path:
        return self._paths.data_dir / "jobs" / f"{job_id}.json"

    def create_job(
        self,
        source_type: str,
        source: str,
        title: str | None = None,
        min_clip_sec: int | None = None,
        max_clip_sec: int | None = None,
        max_clips: int | None = None,
        caption_preset: str | None = None,
    ) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        job: dict[str, Any] = {
            "_id": self.new_id(),
            "sourceType": source_type,
            "source": source,
            "title": title,
            "isUpload": source_type == "upload",
            "stage": "queued",
            "progress": 0.0,
            "progressDetail": "Queued — ClipForge's server worker will pick it up.",
            "unitLabel": None,
            "unitsDone": None,
            "unitsTotal": None,
            "stageEtaSec": None,
            "totalEtaSec": None,
            "lastProgressAt": now_ms,
            "slow": False,
            "stageStartedAt": now_ms,
            "stageTimings": [],
            "logs": [],
            "error": None,
            "errorDetail": None,
            "durationSec": None,
            "clipCount": None,
            "totalClipsSec": None,
            "detectedSource": None,
            "downloadSource": None,
            "captionsTxt": None,
            "options": {
                "minClipSec": min_clip_sec,
                "maxClipSec": max_clip_sec,
                "maxClips": max_clips,
                "captionPreset": caption_preset,
            },
            "createdAt": now_ms,
            "completedAt": None,
        }
        self.write_job(job)
        return job

    def _write_job_unlocked(self, job: dict[str, Any]) -> None:
        """Write a job while the caller's cross-process lock is held."""
        path = self._job_path(job["_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_for(path)
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def write_job(self, job: dict[str, Any]) -> None:
        # The API supervisor tails runner output while the child updates stage
        # state. Without the advisory lock, a stale read-modify-write can erase
        # the child's terminal `failed` state and make the parent report a
        # misleading "worker exited unexpectedly" error.
        path = self._job_path(job["_id"])
        with self._file_lock(path):
            with self._lock:
                self._write_job_unlocked(job)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        for attempt in (1, 2):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except Exception as exc:  # unreadable job docs are surfaced, not swallowed
                if attempt == 1:
                    # a writer may have been mid-rename on older builds; retry
                    # once before declaring the document unreadable
                    logger.warning(f"[store] job {job_id} unreadable, retrying: {exc}")
                    continue
                logger.error(f"[store] unreadable job {job_id}: {exc}")
                return None
        return None

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for path in sorted((self._paths.data_dir / "jobs").glob("*.json")):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.error(f"[store] unreadable job file {path.name}: {exc}")
        jobs.sort(key=lambda j: j.get("createdAt", 0), reverse=True)
        return jobs

    def delete_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False
        with self._lock:
            # remove rendered artifacts + transcripts for this job
            for clip in self.list_clips(job_id):
                for key in ("videoPath", "thumbPath"):
                    p = clip.get(key)
                    if p:
                        Path(p).unlink(missing_ok=True)
                cpath = self._paths.data_dir / "clips" / f"{clip['_id']}.json"
                cpath.unlink(missing_ok=True)
            self._job_path(job_id).unlink(missing_ok=True)
            logger.info(f"[store] deleted job {job_id} and its clips")
        return True

    # ------------------------------------------------------------------
    # clips
    # ------------------------------------------------------------------

    def _clip_path(self, clip_id: str) -> Path:
        return self._paths.data_dir / "clips" / f"{clip_id}.json"

    def create_clips(
        self,
        job_id: str,
        candidates: list[dict[str, Any]],
        detected_source: str,
    ) -> list[dict[str, Any]]:
        """Create (or REFRESH) clip docs for a job.

        On retry/resume existing docs are reused so rendered clips survive;
        only the detection fields are refreshed.
        """
        docs: list[dict[str, Any]] = []
        existing = self.list_clips(job_id)
        for i, c in enumerate(candidates):
            base = existing[i] if i < len(existing) else None
            if base is None:
                doc: dict[str, Any] = {
                    "_id": self.new_id(),
                    "videoPath": None,
                    "thumbPath": None,
                    "mediaBytes": None,
                    "mediaWidth": None,
                    "mediaHeight": None,
                    "mediaDurationSec": None,
                    "mediaError": None,
                    "renderState": "pending",
                    "renderMs": None,
                    "captionCues": [],
                    "silenceCuts": [],
                    "cropBox": None,
                    "aspect": "9:16",
                }
            else:
                # resume: keep the id, rendered file paths and renderState so
                # already-rendered clips are skipped, not re-rendered
                doc = dict(base)
                if doc.get("renderState") == "running":
                    doc["renderState"] = "pending"
            doc.update({
                "jobId": job_id,
                "order": i,
                "start": float(c["start"]),
                "end": float(c["end"]),
                "score": float(c.get("score", 0)),
                "title": str(c.get("title", "Untitled clip"))[:120],
                "hookText": str(c.get("hook_text", ""))[:200],
                "transcript": str(c.get("transcript", "")),
                "hashtags": [str(h) for h in c.get("hashtags", [])][:12],
            })
            docs.append(doc)
            self.write_clip(doc)
        if docs:
            job = self.get_job(job_id) or {}
            job["detectedSource"] = detected_source
            self.write_job(job)
        return docs

    def write_clip(self, clip: dict[str, Any]) -> None:
        path = self._clip_path(clip["_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_for(path)
        with self._lock:
            tmp.write_text(json.dumps(clip, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    def get_clip(self, clip_id: str) -> dict[str, Any] | None:
        path = self._clip_path(clip_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[store] unreadable clip {clip_id}: {exc}")
            return None

    def list_clips(self, job_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted((self._paths.data_dir / "clips").glob("*.json")):
            try:
                clip = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error(f"[store] unreadable clip file {path.name}: {exc}")
                continue
            if clip.get("jobId") == job_id:
                out.append(clip)
        out.sort(key=lambda c: c.get("order", 0))
        return out

    def list_all_clips(self, limit: int = 96) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted((self._paths.data_dir / "clips").glob("*.json")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        out.sort(key=lambda c: c.get("order", 0))
        return out[:limit]

    def clips_ready_count(self, job_id: str) -> int:
        return sum(1 for c in self.list_clips(job_id) if c.get("videoPath"))

    # ------------------------------------------------------------------
    # logs
    # ------------------------------------------------------------------

    def append_job_log(self, job_id: str, line: str) -> None:
        """Append a log line atomically across processes (parent tailer +
        child pipeline both log)."""
        path = self._job_path(job_id)
        if not path.exists():
            return
        with self._file_lock(path):
            job = self.get_job(job_id)
            if job is None:
                return
            logs: list[str] = list(job.get("logs", []))
            logs.append(f"{time.strftime('%H:%M:%S')} {line}")
            job["logs"] = logs[-300:]
            # write_job() would try to acquire the same advisory lock again;
            # use the explicitly locked primitive while this read-modify-write
            # is in progress.
            with self._lock:
                self._write_job_unlocked(job)

    # ------------------------------------------------------------------
    # misc helpers
    # ------------------------------------------------------------------

    def find_source_file(self, job_id: str) -> Path | None:
        d = self._paths.downloads_dir / job_id
        if not d.exists():
            return None
        for p in sorted(d.glob("*")):
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".m4a"}:
                return p
        return None
