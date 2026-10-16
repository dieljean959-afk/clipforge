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
    # job put/get/delete
    # ------------------------------------------------------------------

    def _job_path(self, job_id: str) -> Path:
        return self._paths.data_dir / f"{job_id}.json"

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
        try:
            files = sorted(
                (p for p in self._paths.data_dir.iterdir() if p.suffix == ".json"),
                key=lambda p: (-p.stat().st_mtime_ns, p.name),
                reverse=True,
            )
        except OSError:
            return []
        return list(files)

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
            next_before = before_key(page[-1].get("_id", ""))  # type: ignore[arg-type]
            # Recompute from path since we keyed by mtime above and the
            # doc itself does not carry a stable cursor value worth reusing.
            last_path = all_paths[total - 1 - (total - 1 - candidates.index(page[-1]))]  # type: ignore[arg-type]
            next_before = before_key(last_path)

        return page, total, next_before
