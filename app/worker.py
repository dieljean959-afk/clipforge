"""In-process worker: a daemon thread that drains the local job queue.

FastAPI and the worker live in the SAME process — "worker offline" is
structurally impossible. Jobs are picked FIFO; there is no broker and no
second process to forget to start.
"""

from __future__ import annotations

import queue
import threading
import time

from loguru import logger

from app.config import Settings
from app.store import LocalStore
from app.task import Pipeline


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

    # status for the dashboard card
    def status(self) -> dict:
        return {
            "online": self._thread is not None and self._thread.is_alive(),
            "currentJob": self._current_job,
            "queueDepth": self._queue.qsize(),
        }
