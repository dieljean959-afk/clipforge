"""A restarted worker must pick its queue back up from the job documents.

The run queue lived only in the daemon's memory. Every deploy, crash or
"Restart sandbox" emptied it while the job documents still said `queued` — and
nothing put them back, so the dashboard showed "Queued 0%" next to an idle
worker until someone pressed Retry. The documents are the durable state; the
adoption pass is what makes a restart a resume instead of a stop.
"""

from __future__ import annotations

import threading

import pytest

from app import jobstatus
from app.worker import InProcessWorker


class _Store:
    def __init__(self, jobs: list[dict]) -> None:
        self.jobs = jobs
        self.logs: dict[str, list[str]] = {}

    def list_jobs(self) -> list[dict]:
        return self.jobs

    def append_job_log(self, job_id: str, line: str) -> None:
        self.logs.setdefault(job_id, []).append(line)


def _worker(jobs: list[dict]) -> tuple[InProcessWorker, _Store, list[str]]:
    worker = InProcessWorker.__new__(InProcessWorker)
    store = _Store(jobs)
    queued: list[str] = []
    worker.store = store
    worker._pending = set()
    worker._active = {}
    worker._lock = threading.RLock()
    worker.enqueue = lambda job_id: queued.append(job_id) or len(queued)  # type: ignore[method-assign]
    return worker, store, queued


def test_a_queued_job_is_adopted_after_the_queue_is_lost():
    worker, store, queued = _worker(
        [{"_id": "job-q", "status": "queued", "stage": "queued"}]
    )

    adopted = worker.adopt_pending()

    assert adopted == ["job-q"] and queued == ["job-q"]
    assert any("RESUMED" in line for line in store.logs["job-q"])


def test_a_job_interrupted_mid_stage_is_adopted():
    worker, _store, queued = _worker(
        [{"_id": "job-t", "status": "queued", "stage": "transcribing", "progress": 84.0}]
    )
    assert worker.adopt_pending() == ["job-t"]
    assert queued == ["job-t"]


@pytest.mark.parametrize(
    "job",
    [
        {"_id": "done", "status": "complete", "stage": "complete"},
        {"_id": "dead", "status": "failed", "stage": "failed"},
        {"_id": "paused", "status": "paused", "stage": "paused"},
        {"_id": "del", "status": "deleting", "stage": "deleting", "deleting": {"phase": "cancel"}},
    ],
)
def test_settled_jobs_are_never_resurrected(job):
    worker, _store, queued = _worker([job])
    assert worker.adopt_pending() == []
    assert queued == []


def test_a_job_out_of_resume_budget_is_left_for_reconcile_to_fail():
    worker, _store, queued = _worker(
        [{"_id": "loop", "status": "queued", "stage": "grabbing", "interruptCount": 4}]
    )
    assert worker.adopt_pending() == []
    assert queued == []


def test_adoption_is_idempotent_across_repeated_startups():
    worker, _store, queued = _worker(
        [{"_id": "job-q", "status": "queued", "stage": "queued"}]
    )
    assert worker.adopt_pending() == ["job-q"]
    # The first enqueue put it in the in-memory queue: a second pass must not
    # deliver it twice.
    worker._pending.add("job-q")
    assert worker.adopt_pending() == []
    assert queued == ["job-q"]


def test_an_unreadable_store_adopts_nothing_instead_of_crashing():
    worker, _store, queued = _worker([])
    worker.store = type("S", (), {"list_jobs": staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("boom")))})()
    assert worker.adopt_pending() == []
    assert queued == []


# --- the stage label the card renders --------------------------------------


def test_a_queued_job_is_labelled_queued_not_lowercase():
    job = {"status": "queued", "stage": "queued"}
    assert jobstatus.stage_label(job) == "Queued"
    assert jobstatus.active_stage(job) is None


def test_pipeline_aliases_map_onto_the_four_visible_stages():
    assert jobstatus.active_stage({"stage": "fetch"}) == "grabbing"
    assert jobstatus.active_stage({"stage": "transcribe"}) == "transcribing"
    assert jobstatus.active_stage({"stage": "detect"}) == "finding"
    assert jobstatus.active_stage({"stage": "export"}) == "rendering"


# --- "queued" must not hide a job that is actually running ----------------


def test_a_waiting_job_reports_queued_even_with_a_resume_stage():
    job = {
        "status": "queued", "stage": "transcribing", "queuedAt": 2000, "lastProgressAt": 1000,
    }
    assert jobstatus.status_of(job) == "queued"
    assert jobstatus.active_stage(job) is None


def test_a_job_that_ticked_after_the_enqueue_is_running_again():
    # An older child process may never write status=running; the progress
    # timestamp is enough to tell that it started.
    job = {
        "status": "queued", "stage": "transcribing", "queuedAt": 1000, "lastProgressAt": 2000,
    }
    assert jobstatus.status_of(job) == "running"
    assert jobstatus.active_stage(job) == "transcribing"


def test_a_job_never_enqueued_is_queued():
    assert jobstatus.status_of({"status": "queued", "stage": "queued"}) == "queued"
