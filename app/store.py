"""Filesystem-backed job/clip store for local ClipForge.

Jobs and clips live as JSON documents under ``data/`` so state survives
restarts without any external database. All writes are atomic (tmp + rename).
"""

from __future__ import annotations

from app import sharelink

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
        job_id = job["_id"]
        # A job deleted mid-run stays deleted. The child process cannot be
        # interrupted mid-render, so without this it would faithfully write its
        # progress and terminal state back and the job the user deleted would
        # reappear, now pointing at files that were unlinked underneath it.
        if self.is_deleted(job_id):
            logger.info(f"[store] ignoring write for deleted job {job_id}")
            return
        path = self._job_path(job_id)
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

    #: Fields the batch LIST renders. Everything else — most importantly
    #: `logs`, which is up to 300 lines per job — is dead weight on a payload
    #: the dashboard re-fetches every two seconds. With 17 jobs that was
    #: 285 KB per poll, which is what made "Clips are loading..." outlast the
    #: clip request it was waiting behind. The full document is still served,
    #: unchanged, by GET /api/jobs/{id}.
    LIST_FIELDS = (
        "_id", "title", "source", "sourceType", "isUpload", "stage", "progress",
        "progressDetail", "createdAt", "completedAt", "durationSec",
        "clipCount", "totalClipsSec", "transcriptWordCount", "detectedSource",
        "downloadSource", "error", "errorDetail", "failedStage", "interrupted",
        "interruptReason", "mediaError", "options", "renderState", "unitsTotal",
        "unitsDone", "unitLabel", "clipTarget", "clipTargetReason",
    )

    def list_job_summaries(
        self, *, limit: int = 10, before: int | None = None
    ) -> tuple[list[dict[str, Any]], int, int | None]:
        """Newest-first batch summaries, ONE page at a time.

        Returns `(page, total, next_before)`. `next_before` is the
        `createdAt` of the last item on this page, or None at the end of the
        list — the cursor the dashboard asks for when the user scrolls.

        The projection is applied while the document is in hand, so the cost of
        a page of 10 is 10 file reads regardless of how large the full
        documents are. There is no database here to index, so the sort that
        stands in for "indexes on batch_id and created_at" is done once over
        the parsed headers rather than by the caller re-sorting every slice.
        """
        jobs = self.list_jobs()
        total = len(jobs)
        if before is not None:
            jobs = [j for j in jobs if int(j.get("createdAt") or 0) < int(before)]
        page = jobs[: max(1, int(limit))] if limit > 0 else jobs
        next_before = (
            int(page[-1].get("createdAt") or 0) if page and len(jobs) > len(page) else None
        )
        return [{k: j[k] for k in self.LIST_FIELDS if k in j} for j in page], total, next_before

    def list_clips_for_list(self, job_id: str) -> list[dict[str, Any]]:
        """Clip documents trimmed to what the grid and the share dialog read.

        The one real cut is the per-word timings inside `captionCues`. Each cue
        carries a `words` array of every word with its own start/end, which the
        card, the SRT builder and the lightbox never read — they use only
        start/end/text — and it made up about 85% of the clip payload: 7 KB per
        clip, 91 KB for a 13-clip batch. Same captions on screen, one seventh
        of the bytes.

        The trim is applied to the returned documents only; `list_clips` still
        hands every caller the full record.
        """
        out: list[dict[str, Any]] = []
        for clip in self.list_clips(job_id):
            cues = clip.get("captionCues")
            if isinstance(cues, list):
                trimmed = []
                for c in cues:
                    if not isinstance(c, dict):
                        trimmed.append(c)
                        continue
                    c = {k: v for k, v in c.items() if k not in ("w", "words")}
                    # Whisper timings arrive as accumulated float error —
                    # 0.2400000000000091 — and every one of those digits is
                    # serialised for every cue of every clip on screen. Millis
                    # is what SRT itself uses, so nothing observable changes.
                    for field in ("start", "end"):
                        val = c.get(field)
                        if isinstance(val, (int, float)):
                            c[field] = round(float(val), 3)
                    trimmed.append(c)
                clip["captionCues"] = trimmed
            out.append(clip)
        return out

    def delete_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False
        with self._lock:
            # A tombstone FIRST. Deleting a job that a child process is still
            # working on used to resurrect it: the runner kept going and its
            # final write_job/write_clip calls recreated the very files we had
            # just removed, so a deleted job reappeared as "complete" with
            # zombie clips attached to no job. The marker makes those late
            # writes no-ops instead.
            self._tombstone_path(job_id).parent.mkdir(parents=True, exist_ok=True)
            self._tombstone_path(job_id).write_text(
                json.dumps({"jobId": job_id, "at": int(time.time() * 1000)}),
                encoding="utf-8",
            )
            # Move the documents and rendered media to data/trash/ instead of
            # unlinking them. Deleting used to be IRREVERSIBLE: the retention
            # sweep calls this on every job older than retention_hours (24h by
            # default), so a batch finished on Monday was unrecoverable by
            # Tuesday — the job record, the clip records and every rendered
            # MP4 were gone with no backup and no undo, which is exactly how a
            # user's whole library vanished overnight. Trash keeps the bytes and
            # restore_job() puts them back; the job is still invisible to the
            # app either way, because the tombstone still blocks list_jobs().
            trash = self._trash_dir(job_id)
            moved = 0
            for clip in self.list_clips(job_id):
                for key in ("videoPath", "thumbPath"):
                    p = clip.get(key)
                    if p:
                        moved += self._move_to_trash(Path(p), trash / "media")
                cpath = self._clip_path(clip["_id"])
                moved += self._move_to_trash(cpath, trash / "clips")
            moved += self._move_to_trash(self._job_path(job_id), trash)
            (trash / "meta.json").write_text(
                json.dumps({"jobId": job_id, "at": int(time.time() * 1000), "moved": moved}),
                encoding="utf-8",
            )
            logger.info(
                f"[store] deleted job {job_id} and its clips (tombstoned); "
                f"{moved} file(s) moved to {trash} — restorable"
            )
        return True

    # ------------------------------------------------------------------
    # trash / restore
    # ------------------------------------------------------------------

    def _trash_dir(self, job_id: str) -> Path:
        """`data/trash/<jobId>-<timestamp>` — one dir per deletion."""
        return self._paths.data_dir / "trash" / f"{job_id}-{int(time.time())}"

    @staticmethod
    def _move_to_trash(src: Path, dest_dir: Path) -> int:
        """Move one file into `dest_dir`, keeping its name. Returns 1 on move."""
        if not src.exists():
            return 0
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / src.name
            # never clobber a same-named file from an earlier deletion
            if target.exists():
                target = dest_dir / f"{src.stem}.{int(time.time())}{src.suffix}"
            src.replace(target)
            return 1
        except OSError as exc:
            # A locked/unreadable artifact must not abort the deletion — the
            # tombstone is what makes the job disappear, and a leftover file in
            # trash is strictly better than a resurrected job.
            logger.warning(f"[store] could not move {src} to trash: {exc}")
            return 0

    def restore_job(self, job_id: str) -> bool:
        """Undo the most recent delete of `job_id` from the trash.

        Restores the job document, its clip documents and any rendered media
        that was still on disk, and clears the tombstone so the job (and its
        late writes, if a child is still running) become live again.
        """
        tomb = self._tombstone_path(job_id)
        if not tomb.exists():
            return False
        with self._lock:
            candidates = sorted(
                (self._paths.data_dir / "trash").glob(f"{job_id}-*"),
                key=lambda p: p.name,
            )
            if not candidates:
                logger.warning(f"[store] restore {job_id}: tombstone but no trash dir")
                return False
            trash = candidates[-1]
            restored = 0
            # Rendered media is restored to the absolute path recorded in the
            # clip doc, not to a guessed directory — that is the only place the
            # app will look for it.
            media_dir = trash / "media"
            clip_dir = trash / "clips"
            trashed_clips = sorted(clip_dir.glob("*.json")) if clip_dir.exists() else []
            for cpath in trashed_clips:
                try:
                    clip = json.loads(cpath.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning(f"[store] restore {job_id}: unreadable {cpath.name}: {exc}")
                    clip = {}
                for key in ("videoPath", "thumbPath"):
                    p = clip.get(key)
                    if not p:
                        continue
                    candidate = media_dir / Path(str(p)).name
                    if not candidate.exists():
                        continue
                    try:
                        target = Path(str(p))
                        target.parent.mkdir(parents=True, exist_ok=True)
                        candidate.replace(target)
                        restored += 1
                    except OSError as exc:
                        logger.warning(f"[store] restore {job_id}: cannot place {p}: {exc}")
                restored += self._move_to_trash(cpath, self._paths.data_dir / "clips")
            src_job = trash / f"{job_id}.json"
            if src_job.exists():
                restored += self._move_to_trash(src_job, self._paths.data_dir / "jobs")
            tomb.unlink(missing_ok=True)
            logger.info(f"[store] restored job {job_id} from {trash} ({restored} file(s))")
        return True

    def _tombstone_path(self, job_id: str) -> Path:
        return self._paths.data_dir / "jobs" / f"{job_id}.deleted"

    def is_deleted(self, job_id: str) -> bool:
        """True once a job has been deleted; late writes must not revive it."""
        return self._tombstone_path(job_id).exists()

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

        Reuse is only safe while the cut is UNCHANGED. A re-detection can
        return a different window at the same index (clip 3 is no longer the
        same three seconds), and blindly keeping renderState="done" would
        leave the previous encode on disk under the new metadata: the job
        reports success while shipping the old video. So any material move of
        the cut invalidates the render and the clip is encoded again.
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
                    # Minted at CREATION, not at share time: the public link
                    # then exists from the moment the clip does, so a user who
                    # copies it while the batch is still rendering gets a link
                    # that starts working the moment the copy finishes rather
                    # than a 404 that never resolves.
                    "shareId": sharelink.new_id(),
                    "shareState": sharelink.STATE_PREPARING,
                }
            else:
                # resume: keep the id, rendered file paths and renderState so
                # already-rendered clips are skipped, not re-rendered
                doc = dict(base)
                # "running" means a worker died mid-encode; "failed" means the
                # encode raised. Both are unfinished work, not results: put them
                # back in the queue and drop the stale error so the next attempt
                # starts clean.
                if doc.get("renderState") in ("running", "failed"):
                    doc["renderState"] = "pending"
                    doc["mediaError"] = None
                # ...but only while the cut still matches what was encoded.
                if self._cut_moved(base, c):
                    doc.update({
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
                    })
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
            # The job record can legitimately be gone: the user deleted the job
            # while this child process was mid-render. Recording the detection
            # source is a nice-to-have; crashing the render loop with a
            # KeyError on a job that no longer exists is not.
            job = self.get_job(job_id)
            if job is not None:
                job["detectedSource"] = detected_source
                self.write_job(job)
            else:
                logger.info(f"[store] job {job_id} vanished mid-render — not recording detectedSource")
        return docs

    @staticmethod
    def _cut_moved(base: dict[str, Any], cand: dict[str, Any]) -> bool:
        """True when a new candidate no longer covers the same window.

        Half a frame of slack absorbs float noise from word-index rounding;
        anything larger is a genuinely different clip.
        """
        try:
            return (
                abs(float(base["start"]) - float(cand["start"])) > 0.02
                or abs(float(base["end"]) - float(cand["end"])) > 0.02
            )
        except (KeyError, TypeError, ValueError):
            return True

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

    def clips_delivered(
        self, job_id: str, job: dict[str, Any] | None = None
    ) -> tuple[int, int]:
        """``(clips actually on disk, clips the job promised)`` for one job.

        "Promised" is the job's own ``clipCount`` when it recorded one, else the
        number of clip records — a job that got as far as counting knows what a
        full batch looked like, and a job that died before counting is judged
        against what it managed to write.

        Rendered means the MP4 exists on disk, not merely that a ``videoPath``
        string was recorded: a container recycle can leave a job pointing at
        files that are gone, and treating those as delivered would restore
        ``stage=complete`` for a batch the user cannot download.
        """
        clips = self.list_clips(job_id)
        rendered = sum(
            1
            for c in clips
            if c.get("videoPath") and Path(str(c["videoPath"])).exists()
        )
        promised = int((job or {}).get("clipCount") or 0) or len(clips)
        return rendered, promised

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
