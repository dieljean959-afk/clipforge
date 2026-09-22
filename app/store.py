"""Filesystem-backed job/clip store for local ClipForge.

Jobs and clips live as JSON documents under ``data/`` so state survives
restarts without any external database. All writes are atomic (tmp + rename).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import Paths


class LocalStore:
    """Thread-safe JSON document store rooted at data_dir."""

    def __init__(self, paths: Paths) -> None:
        self._paths = paths
        self._lock = threading.RLock()
        self._paths.data_dir.mkdir(parents=True, exist_ok=True)
        self._paths.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._paths.downloads_dir.mkdir(parents=True, exist_ok=True)

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
            "progressDetail": "Queued — the local worker will pick it up.",
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
            },
            "createdAt": now_ms,
            "completedAt": None,
        }
        self.write_job(job)
        return job

    def write_job(self, job: dict[str, Any]) -> None:
        path = self._job_path(job["_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # unreadable job docs are surfaced, not swallowed
            logger.error(f"[store] unreadable job {job_id}: {exc}")
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
        docs: list[dict[str, Any]] = []
        for i, c in enumerate(candidates):
            doc = {
                "_id": self.new_id(),
                "jobId": job_id,
                "order": i,
                "start": float(c["start"]),
                "end": float(c["end"]),
                "score": float(c.get("score", 0)),
                "title": str(c.get("title", "Untitled clip"))[:120],
                "hookText": str(c.get("hook_text", ""))[:200],
                "transcript": str(c.get("transcript", "")),
                "hashtags": [str(h) for h in c.get("hashtags", [])][:12],
                "captionCues": c.get("caption_cues", []),
                "silenceCuts": c.get("silence_cuts", []),
                "cropBox": c.get("crop_box"),
                "aspect": "9:16",
                "videoPath": None,
                "thumbPath": None,
                "mediaBytes": None,
                "mediaWidth": None,
                "mediaHeight": None,
                "mediaDurationSec": None,
                "mediaError": None,
                "renderState": "pending",
                "renderMs": None,
            }
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
        tmp = path.with_suffix(".json.tmp")
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
        job = self.get_job(job_id)
        if job is None:
            return
        logs: list[str] = list(job.get("logs", []))
        logs.append(f"{time.strftime('%H:%M:%S')} {line}")
        job["logs"] = logs[-300:]
        self.write_job(job)

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
