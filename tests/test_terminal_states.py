"""A settled batch can never move back to queued/fetching — and the ETA clamps.

These pin the two worst live incidents on job ``6ce9abe11d1c`` (23:36 source):

* every API boot re-queued a job whose document already said
  ``isTerminal: true, retryable: false`` because ``_recover_jobs`` only looked at
  a leftover ``interrupted`` flag — so the batch restarted at "Fetching video /
  Downloading 40%" on every restart;
* ``runner.main`` wrote ``stage="fetch", progress=0`` unconditionally before the
  pipeline had read a checkpoint, which is the same symptom from the other side;
* the card read ``~2693 min left`` because the ETA divided the CUMULATIVE stage
  clock (which survives resumes on purpose) by the new attempt's fraction.
"""

from __future__ import annotations

from pathlib import Path

from app import eta, jobstatus

ROOT = Path(__file__).resolve().parents[1]


def _load_source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# terminal states never reopen
# --------------------------------------------------------------------------- #

def test_complete_batch_is_not_requeueable() -> None:
    job = {"stage": "complete", "status": "complete", "isTerminal": True}
    assert jobstatus.is_requeueable(job) is False


def test_failed_batch_marked_terminal_is_not_requeueable() -> None:
    # The real shape: failed, but a stale `interrupted: true` from an earlier
    # attempt was still on the document.
    job = {
        "stage": "failed",
        "status": "failed",
        "isTerminal": True,
        "retryable": False,
        "interrupted": True,
    }
    assert jobstatus.is_requeueable(job) is False


def test_finished_stamp_alone_settles_a_job() -> None:
    job = {"stage": "failed", "status": "failed", "finishedAt": 1_700_000_000_000}
    assert jobstatus.is_requeueable(job) is False


def test_running_job_is_requeueable() -> None:
    job = {"stage": "transcribing", "status": "running", "interrupted": True}
    assert jobstatus.is_requeueable(job) is True


def test_user_retry_reopens_a_settled_job() -> None:
    job = {"stage": "failed", "status": "failed", "isTerminal": True, "retryable": False,
           "finishedAt": 123}
    jobstatus.allow_retry(job)
    assert jobstatus.is_requeueable(job) is True
    assert job["finishedAt"] is None


def test_recovery_path_checks_requeueable_before_enqueueing() -> None:
    """The guard must be in the recovery path itself, not only in jobstatus."""
    src = _load_source("app/main.py")
    assert "jobstatus.is_requeueable(job)" in src, (
        "_recover_jobs must refuse to re-queue a settled job"
    )
    # It has to run BEFORE the stage is reset to queued.
    guard = src.index("jobstatus.is_requeueable(job)")
    reset = src.index('"stage": "queued"', guard)
    assert guard < reset


def test_runner_does_not_relabel_a_captured_source_as_fetching() -> None:
    src = _load_source("app/runner.py")
    assert "is_requeueable(job)" in src, "runner must refuse a settled job"
    assert "has_source" in src, (
        "runner must not reset stage/progress to fetch for a source already captured"
    )


def test_progress_write_refuses_a_terminal_document() -> None:
    src = _load_source("app/task.py")
    marker = "A TERMINAL BATCH IS IMMUTABLE"
    assert marker in src
    i = src.index(marker)
    window = src[i:i + 900]
    assert 'job.get("isTerminal") is True' in window
    assert "return" in window


# --------------------------------------------------------------------------- #
# ETA: never above 24 h, never a division by a near-zero rate
# --------------------------------------------------------------------------- #

def test_eta_clamps_to_a_day() -> None:
    assert eta.clamp_eta(50 * 3600) == eta.MAX_ETA_SEC
    assert eta.clamp_eta(-5) is None
    assert eta.clamp_eta(float("nan")) is None


def test_progress_eta_rejects_an_absurd_extrapolation() -> None:
    # 48,853 s of a previous attempt divided by 4% of this one: the exact shape
    # that produced "~2693 min left".
    assert eta.eta_from_progress(48_853.0, 0.04) is None


def test_progress_eta_is_linear_on_the_attempt_clock() -> None:
    # Half done after 100 s of THIS attempt: 100 s remaining.
    assert eta.eta_from_progress(100.0, 0.5) == 100.0


def test_plan_total_is_clamped() -> None:
    plan = [{"stage": "fetch", "sec": 50 * 3600.0}]
    assert eta.plan_total_sec(plan) == float(eta.MAX_ETA_SEC)


def test_task_uses_attempt_clock_not_the_display_clock() -> None:
    src = _load_source("app/task.py")
    assert "stageRunStartedAt" in src
    # The measured division must use the attempt clock.
    i = src.index("elapsed = max(0.0, (now_ms - run_start) / 1000)")
    assert "eta.estimate" in src[i:i + 400]


def test_retry_clears_terminal_stamps_before_queueing() -> None:
    """A retry that left isTerminal=True could never start: the runner's
    settled-job guard exits immediately and the card sits on \"queued\"."""
    src = _load_source("app/worker.py")
    i = src.index("def retry(self, job_id: str) -> int:")
    window = src[i:i + 3000]
    assert "jobstatus.allow_retry(job)" in window
    assert '"isTerminal": False' in window
    assert '"finishedAt": None' in window
    assert "if not job:" in window
