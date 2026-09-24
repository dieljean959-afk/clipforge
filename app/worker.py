"""In-process worker: a daemon thread that drains the local job queue.

FastAPI and the worker live in the SAME process — "worker offline" is
structurally impossible. Jobs are picked FIFO; there is no broker and no
second process to forget to start.

A stall watchdog runs beside the queue: a job that writes no progress for
STALL_TIMEOUT_SEC while the worker is alive is failed EXPLICITLY with a
readable error instead of hanging forever (the pre-watchdog behavior).
"""

from __future__ import annotations

import queue
import threading
import time

from loguru import logger

from app.config import Settings
from app.store import LocalStore
from app.task import Pipeline

# No-progress window before a running job is declared stalled. Generous on
# purpose: the medium Whisper model on CPU can spend minutes INSIDE one decode
# window (intra-window progress events keep this honest — see subtitle.py).
STALL_TIMEOUT_SEC = 20 * 60

class InProcessWorker:
    def __init__(self, settings: Settings, store: LocalStore) -> None:
        self.settings = settings
        self.store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._current_job: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="clipforge-worker", daemon=True)
            self._thread.start()
            logger.info("In-process worker online, consuming queue: clipforge-local")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def enqueue(self, job_id: str) -> int:
        self._queue.put(job_id)
        depth = self._queue.qsize()
        logger.info(f"[worker] job {job_id} queued (depth {depth})")
        return depth

    def retry(self, job_id: str) -> int:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        job.update({
            "stage": "queued",
            "progress": 0,
            "progressDetail": "Re-queued — the local worker will pick it up.",
            "error": None,
            "errorDetail": None,
        })
        self.store.write_job(job)
        return self.enqueue(job_id)

    # ------------------------------------------------------------------

    def _run(self) -> None:
        pipeline = Pipeline(self.settings, self.store)
        logger.info("[worker] consuming queue: clipforge-local (single-threaded, FIFO)")
        watchdog = threading.Thread(target=self._watchdog, name="clipforge-watchdog", daemon=True)
        watchdog.start()
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._current_job = job_id
            try:
                pipeline.run(job_id)
            except Exception as exc:  # pipeline.run catches its own, but be safe
                logger.exception(f"[worker] job {job_id} crashed the worker loop: {exc}")
            finally:
                self._current_job = None
                self._queue.task_done()

    def _watchdog(self) -> None:
        """Fail running jobs that stopped reporting progress.

        A hard-hung whisper/ffmpeg call cannot be interrupted from Python, so
        the honest outcome is an explicit "stalled" failure the UI can show —
        the job stops pretending to progress, and the user can retry. The
        worker loop itself keeps running (it is a separate thread).
        """
        while not self._stop.wait(30.0):
            job_id = self._current_job
            if job_id is None:
                continue
            job = self.store.get_job(job_id)
            if job is None or job.get("stage") in {"complete", "failed"}:
                continue
            last = job.get("lastProgressAt") or job.get("stageStartedAt") or job.get("createdAt") or 0
            silent_for = time.time() * 1000 - last
            if silent_for > STALL_TIMEOUT_SEC * 1000:
                logger.error(
                    f"[worker] job {job_id} stalled: no progress for {silent_for / 1000:.0f}s "
                    f"at stage {job.get('stage')} — failing explicitly"
                )
                self.store.append_job_log(
                    job_id,
                    f"stall watchdog: no progress for {silent_for / 1000:.0f}s at stage {job.get('stage')} — failing the job",
                )
                job.update({
                    "stage": "failed",
                    "error": (
                        f"Pipeline stalled — no progress for {int(silent_for / 1000)}s "
                        f"at stage {job.get('stage')}. The worker may be out of memory or the "
                        "source is unreadable. Retry, or try a shorter/lower-resolution source."
                    ),
                    "errorDetail": f"stall watchdog fired after {int(silent_for / 1000)}s of silence",
                    "completedAt": int(time.time() * 1000),
                })
                self.store.write_job(job)

    # status for the dashboard card
    def status(self) -> dict:
        return {
            "online": self._thread is not None and self._thread.is_alive(),
            "currentJob": self._current_job,
            "queueDepth": self._queue.qsize(),
        }
