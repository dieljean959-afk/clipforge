"""One host restart, two failure messages: `-15` and `143` are the same event.

`Popen` reports a signalled child as `-15`. This app's runner installs a SIGTERM
handler that raises `SystemExit(143)` so its `finally` blocks run, and then the
child exits with a POSITIVE 143. Only the negative form was treated as "an
external kill, so resume", so the same restart could resume one job and
hard-fail another with "exited unexpectedly (code 143)" while its transcript sat
finished on disk.

The second half of the bug: the resume budget counted every interruption
forever, so three host bounces in an hour killed a batch that was making
progress on each attempt.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.worker import InProcessWorker, death_signal


# --- both conventions mean the same thing --------------------------------


@pytest.mark.parametrize(
    "rc, sig",
    [
        (-15, 15),   # Popen: SIGTERM
        (143, 15),   # shell / SystemExit(143): SIGTERM
        (-9, 9),     # Popen: SIGKILL
        (137, 9),    # shell: SIGKILL
        (-2, 2),     # Popen: SIGINT
        (130, 2),    # shell: SIGINT
        (-1, 1),     # Popen: SIGHUP
        (129, 1),    # shell: SIGHUP
    ],
)
def test_a_signalled_child_is_recognised_from_either_reporting_convention(rc, sig):
    assert death_signal(rc) == sig


@pytest.mark.parametrize("rc", [0, 1, 2, 100, 127, 200, -11, 139, None])
def test_an_ordinary_exit_or_a_crash_is_not_an_external_kill(rc):
    assert death_signal(rc) is None


def test_the_failure_banner_accepts_both_codes_and_rejects_ordinary_ones():
    base = {"stage": "failed"}

    assert InProcessWorker.has_signal_failure(
        {**base, "error": "Worker process was terminated by SIGTERM during transcribing (code 143)"}
    )
    assert InProcessWorker.has_signal_failure(
        {**base, "error": "Worker process exited unexpectedly (code -15) before the job finished"}
    )
    assert not InProcessWorker.has_signal_failure(
        {**base, "error": "Transcription failed: cannot probe duration (code 1)"}
    )
    assert not InProcessWorker.has_signal_failure({**base, "error": "quantize: code 200"})
    assert not InProcessWorker.has_signal_failure({"stage": "complete", "error": "(code 143)"})


# --- the resume budget counts CONSECUTIVE interruptions ------------------


class _Store:
    def __init__(self, job: dict) -> None:
        self.job = job
        self.logs: list[str] = []

    def write_job(self, job: dict) -> None:
        # Never clear: the worker mutates this dict in place and the test reads
        # it back. The real store serialises it to disk.
        self.snapshot = dict(job)

    def append_job_log(self, job_id: str, line: str) -> None:
        self.logs.append(line)


def _worker(job: dict) -> tuple[InProcessWorker, _Store, list[str]]:
    worker = InProcessWorker.__new__(InProcessWorker)
    store = _Store(job)
    queued: list[str] = []
    worker.store = store
    worker.enqueue = lambda job_id: queued.append(job_id) or 1  # type: ignore[method-assign]
    return worker, store, queued


def test_a_job_that_moved_on_gets_a_fresh_resume_budget():
    job = {
        "_id": "j", "stage": "finding", "progress": 62.0,
        "interruptCount": 3, "interruptStage": "transcribing", "interruptProgress": 40.0,
    }
    worker, store, queued = _worker(job)

    assert worker._note_interrupted(job, "j", -15, "finding", "") is True
    assert job["interruptCount"] == 1, "progress through a stage must reset the budget"
    assert job["interruptStage"] == "finding" and job["interruptProgress"] == 62.0
    assert queued == ["j"]


def test_a_job_dying_in_the_same_place_still_runs_out_of_budget():
    job = {
        "_id": "j", "stage": "transcribing", "progress": 40.0,
        "interruptCount": 3, "interruptStage": "transcribing", "interruptProgress": 40.0,
    }
    worker, store, queued = _worker(job)

    assert worker._note_interrupted(job, "j", 143, "transcribing", "") is False
    assert queued == [], "a job that keeps dying at the same point must stop looping"


def test_the_interrupt_reason_survives_both_code_conventions():
    for rc in (-15, 143):
        job = {"_id": "j", "stage": "transcribing", "progress": 10.0}
        worker, store, queued = _worker(job)
        assert worker._note_interrupted(job, "j", rc, "transcribing", "") is True
        assert "SIGTERM" in job["interruptReason"], job["interruptReason"]
        assert any("SIGTERM" in line for line in store.logs)


# --- reconcile must not re-fail a rescued job with an older reason --------


def test_an_interrupted_attempt_beats_an_earlier_failure_in_the_log():
    from app.worker import _reason_from_logs

    job = {
        "logs": [
            "[job] FAILED at grabbing: Could not fetch video: sign in to confirm",
            "[job] INTERRUPTED: worker killed by SIGTERM during 'grabbing' — auto-resuming (attempt 1 of 3)",
            "[job] [stream] section 1/9: 0s–150s",
        ]
    }
    assert _reason_from_logs(job) == "", "an already-retried failure must not become the reason"


def test_the_newest_failure_is_still_reported():
    from app.worker import _reason_from_logs

    job = {
        "logs": [
            "[job] INTERRUPTED: worker killed by SIGTERM during 'grabbing' — auto-resuming (attempt 1 of 3)",
            "[job] FAILED: Worker process was terminated by SIGTERM (code 143)",
        ]
    }
    assert "SIGTERM" in _reason_from_logs(job)


# --- a finished batch must never be flipped to failed ---------------------


def test_exit0_with_the_childs_completion_line_is_a_success():
    logs = ["Rendering clip 15", "clip 9f2: 5.1 MB stored", "COMPLETE in 1751.8s"]
    assert InProcessWorker.completed_despite_exit_code(0, logs)


def test_exit0_without_a_completion_line_is_still_incomplete():
    assert not InProcessWorker.completed_despite_exit_code(0, ["Rendering clip 4"])


def test_a_killed_child_is_not_rescued_by_an_old_completion_line():
    # An interrupted retry can carry COMPLETE from an earlier attempt; a signal
    # death must still go through the resume path, not be called a success.
    assert not InProcessWorker.completed_despite_exit_code(143, ["COMPLETE in 90.0s"])


def test_the_all_clips_ready_detail_also_counts():
    assert InProcessWorker.completed_despite_exit_code(0, ["All clips ready"])
