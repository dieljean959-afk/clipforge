"""The queue must never hold a job that is already running, or a finished one.

Both shapes were live on this host, on one job:

* Every API boot treated any non-terminal `stage` as "the process died", reset
  the job to `stage=queued / progress=0` and re-queued it — while its runner was
  still exporting clips. The card then read "Queued… 0%" beside a batch that was
  actively encoding, for as long as the export went between progress writes.
* That re-queue also added a queue entry that could not start until the single
  worker slot freed (it was busy with that same job), and once it did free, it
  would have run the whole batch again from stage one.

The guards: `enqueue()` refuses a job whose child is alive, `_run()` drops an
entry whose job already reached a terminal stage, and boot recovery leaves any
job the daemon reports as active completely alone.
"""

from __future__ import annotations

import queue
import threading
import time

from app.worker import InProcessWorker


class _Proc:
    """Just enough of subprocess.Popen for the liveness check."""

    def __init__(self, pid: int, alive: bool = True) -> None:
        self.pid = pid
        self._alive = alive

    def poll(self):
        return None if self._alive else -9


class _Store:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)


def _worker() -> tuple[InProcessWorker, _Store]:
    """A worker with real queue state and no store/process side effects."""
    worker = InProcessWorker.__new__(InProcessWorker)
    worker._lock = threading.Lock()
    worker._pending = []
    worker._queue = queue.Queue()
    worker._active = {}
    worker._current_job = None
    worker._current_proc = None
    worker._stop = threading.Event()
    worker._threads = []
    store = _Store()
    worker.store = store
    # Queue-position bookkeeping is not what this test is about.
    worker._write_queue_position = lambda *a, **k: None  # type: ignore[method-assign]
    return worker, store


def test_enqueue_refuses_a_job_whose_runner_is_alive():
    worker, store = _worker()
    store.jobs["job_1"] = {"_id": "job_1", "stage": "exporting"}
    worker._active["job_1"] = _Proc(4321)

    assert worker.enqueue("job_1") == 0
    assert worker._queue.empty(), "a running job must not get a queue entry"
    assert worker._pending == []


def test_enqueue_refuses_a_job_the_legacy_slot_is_still_supervising():
    """The tiny window where the queue item is popped but `_active` is unset."""
    worker, store = _worker()
    store.jobs["job_1"] = {"_id": "job_1", "stage": "transcribing"}
    worker._current_job = "job_1"
    worker._current_proc = _Proc(999)

    assert worker.enqueue("job_1") == 0
    assert worker._queue.empty()


def test_enqueue_still_accepts_a_job_whose_process_was_killed():
    """`retry()` kills first and then re-queues, so a dead pid must get through."""
    worker, store = _worker()
    store.jobs["job_1"] = {"_id": "job_1", "stage": "transcribing"}
    worker._active["job_1"] = _Proc(4321, alive=False)

    assert worker.enqueue("job_1") == 1
    assert worker._pending == ["job_1"]
    assert not worker._queue.empty()


def test_a_queued_entry_for_a_finished_job_is_dropped(monkeypatch):
    worker, store = _worker()
    store.jobs["job_1"] = {"_id": "job_1", "stage": "complete"}
    started: list[str] = []
    monkeypatch.setattr(worker, "_run_job_process", started.append)

    worker._queue.put("job_1")
    thread = threading.Thread(target=worker._run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if not worker._pending and worker._queue.empty():
                break
            time.sleep(0.01)
    finally:
        worker._stop.set()
        thread.join(timeout=2)

    assert started == [], "a finished batch must not be run a second time"
    assert worker._current_job is None


def test_a_running_job_is_not_reset_by_a_second_enqueue():
    """The exact live shape: re-queue while exporting must change nothing."""
    worker, store = _worker()
    job = {"_id": "job_1", "stage": "exporting", "progress": 13.0}
    store.jobs["job_1"] = job
    proc = _Proc(25551)
    worker._active["job_1"] = proc

    depth = worker.enqueue("job_1")

    assert depth == 0
    assert store.jobs["job_1"] == {"_id": "job_1", "stage": "exporting", "progress": 13.0}
    assert proc.poll() is None, "queuing must never kill the live runner"
