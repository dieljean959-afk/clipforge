"""`reconcile_stale_running`: a job claiming "running" with no worker behind it.

A job's `stage` is written by a child process. If that process is SIGKILLed,
OOM-reaped or lost to a host restart, nothing overwrites it — the document keeps
saying "finding, 61%" forever with no process and no queue entry. That is the
state behind "Stage 3 of 4, running" on a job that had already stopped.

The worker knows which jobs it is running and which are queued, so it is the
authority.

WHAT RECONCILING DOES NOW, and why the assertions below changed: it RE-QUEUES
the orphan instead of failing it. An orphan is by construction work the HOST
took away — the web process is restarted periodically and the pipeline child
dies with it — and the stage markers on disk say exactly where to pick it up.
The old behaviour wrote a generic "the worker stopped and never reported back"
failure, which restated the symptom, named no cause, burned a resume budget
slot and erased the interrupt evidence, leaving a job that was 99% cached
looking permanently dead.

A terminal state is reached only when re-queueing stops being useful: attempts
that make no further progress, or an upload whose file is gone — which fails
immediately with a re-upload message instead of re-queueing into the
"resuming (attempt 2)" loop.

    python tools/stale_reconcile_test.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from app.store import LocalStore  # noqa: E402
from app.worker import STALLED_RESUME_LIMIT, InProcessWorker  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


class _Paths:
    def __init__(self, root: Path) -> None:
        self.data_dir = root
        self.downloads_dir = root / "downloads"
        self.outputs_dir = root / "outputs"
        self.uploads_dir = root / "uploads"
        for d in (self.data_dir, self.downloads_dir, self.outputs_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)

    def job_source_dirs(self, job_id: str) -> tuple[Path, ...]:
        return (self.uploads_dir / job_id, self.downloads_dir / job_id)


def _make_worker(tmp: Path) -> tuple[InProcessWorker, LocalStore]:
    from app.config import load_config

    settings = load_config(quiet=True)
    settings.app = type(settings.app)(
        paths=_Paths(tmp), host="127.0.0.1", port=8000,
    )
    store = LocalStore(settings.app.paths)
    return InProcessWorker(settings, store), store


def _job(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "_id": "job1", "source": "https://youtu.be/abc12345678",
        "stage": "finding", "progress": 61.0, "createdAt": int(time.time() * 1000),
        "lastProgressAt": int(time.time() * 1000), "logs": [],
    }
    base.update(kw)
    return base


def test_stale_running_is_requeued() -> None:
    print("\n[1] a running stage with no worker is re-queued, not failed")
    with tempfile.TemporaryDirectory() as td:
        worker, store = _make_worker(Path(td))
        store.write_job(_job())
        fixed = worker.reconcile_stale_running()
        check("the job was reconciled", "job1" in fixed, str(fixed))
        job = store.get_job("job1") or {}
        check("it is queued again", job.get("stage") == "queued", str(job.get("stage")))
        check("marked as interrupted (the resumable path)",
              job.get("interrupted") is True, str(job.get("interrupted")))
        check("the reason names a restart, not a mystery",
              "restart" in str(job.get("interruptReason")).lower(),
              str(job.get("interruptReason"))[:80])
        check("the old error is cleared (it is not failing any more)",
              job.get("error") is None, str(job.get("error"))[:60])
        check("progress is not left at 61%", job.get("progress") == 0,
              str(job.get("progress")))
        check("the unit counters of the dead attempt are cleared",
              job.get("unitsDone") is None and job.get("unitsTotal") is None,
              f"{job.get('unitsDone')}/{job.get('unitsTotal')}")
        check("it is actually back in the queue", "job1" in worker.queue_snapshot(),
              str(worker.queue_snapshot()))
        # It is live again, so a second pass must leave it alone.
        again = worker.reconcile_stale_running()
        check("idempotent — a job it just re-queued is left alone",
              again == [], str(again))


def test_stalled_resumes_end_in_an_honest_failure() -> None:
    print("\n[1b] attempts that make no progress stop instead of looping")
    with tempfile.TemporaryDirectory() as td:
        worker, store = _make_worker(Path(td))
        store.write_job(_job())
        sig = worker._progress_signature(store.get_job("job1") or {})
        for attempt in range(STALLED_RESUME_LIMIT + 2):
            job = store.get_job("job1") or {}
            # Re-arm it as an orphan exactly as a host restart would, and
            # never let it get any further than it already is.
            job.update({
                "stage": "finding", "interrupted": True,
                "resumeSignature": sig, "stalledResumes": attempt,
                "interruptCount": attempt + 1,
            })
            store.write_job(job)
            worker._pending.clear()
            worker.reconcile_stale_running(force=True)
        job = store.get_job("job1") or {}
        check("it ends failed, not looping",
              job.get("stage") == "failed", str(job.get("stage")))
        check("the failure keeps the stage it died in",
              job.get("failedStage") == "finding", str(job.get("failedStage")))
        check("the failure says the server restarted",
              "restart" in str(job.get("error")).lower(), str(job.get("error"))[:90])
        check("no longer claims to be waiting to be resumed",
              job.get("interrupted") is False, str(job.get("interrupted")))
        for field in ("stagePlan", "unitsDone", "unitsTotal", "unitLabel",
                      "stageEtaSec", "totalEtaSec", "queueEtaSec", "queuePosition"):
            check(f"{field} cleared on the terminal state", field not in job,
                  str(job.get(field)))


def test_lost_upload_is_not_requeued() -> None:
    print("\n[1c] an upload that is gone fails once instead of looping")
    with tempfile.TemporaryDirectory() as td:
        worker, store = _make_worker(Path(td))
        store.write_job(_job(_id="lost", sourceType="upload", isUpload=True,
                             source="source.mp4"))
        worker.reconcile_stale_running(force=True)
        job = store.get_job("lost") or {}
        check("failed", job.get("stage") == "failed", str(job.get("stage")))
        check("says re-upload", "re-upload" in str(job.get("error")).lower(),
              str(job.get("error"))[:90])
        check("not queued", "lost" not in worker.queue_snapshot(),
              str(worker.queue_snapshot()))
        check("not marked interrupted", job.get("interrupted") is False,
              str(job.get("interrupted")))


def test_queued_job_is_reconciled() -> None:
    print("\n[2] a queued job that is not in the queue is equally stale")
    with tempfile.TemporaryDirectory() as td:
        worker, store = _make_worker(Path(td))
        store.write_job(_job(_id="q1", stage="queued", progress=0))
        fixed = worker.reconcile_stale_running()
        check("reconciled", "q1" in fixed, str(fixed))
        job = store.get_job("q1") or {}
        check("it is back in the queue", job.get("stage") == "queued",
              str(job.get("stage")))


def test_live_jobs_are_untouched() -> None:
    print("\n[3] jobs the worker is actually running or about to run are left alone")
    with tempfile.TemporaryDirectory() as td:
        worker, store = _make_worker(Path(td))
        store.write_job(_job(_id="active", stage="rendering"))
        store.write_job(_job(_id="pending", stage="queued"))
        store.write_job(_job(_id="done", stage="complete", progress=100))
        store.write_job(_job(_id="bad", stage="failed", error="boom"))
        store.write_job(_job(_id="held", stage="paused"))
        # Pretend the worker holds these.
        worker._active["active"] = object()  # type: ignore[assignment]
        worker._pending.append("pending")
        fixed = worker.reconcile_stale_running()
        check("nothing reconciled", fixed == [], str(fixed))
        check("the running job kept its stage",
              (store.get_job("active") or {}).get("stage") == "rendering")
        check("the queued job kept its stage",
              (store.get_job("pending") or {}).get("stage") == "queued")
        check("a complete job is untouched",
              (store.get_job("done") or {}).get("error") is None)
        check("a failed job keeps its original error",
              (store.get_job("bad") or {}).get("error") == "boom")
        check("a paused job keeps its stage",
              (store.get_job("held") or {}).get("stage") == "paused")


def test_no_jobs_is_safe() -> None:
    print("\n[4] an empty store is a no-op, not a crash")
    with tempfile.TemporaryDirectory() as td:
        worker, store = _make_worker(Path(td))
        check("no jobs -> no fixes", worker.reconcile_stale_running() == [])
        check("store really is empty", store.list_jobs() == [])


def main() -> int:
    test_stale_running_is_requeued()
    test_stalled_resumes_end_in_an_honest_failure()
    test_lost_upload_is_not_requeued()
    test_queued_job_is_reconciled()
    test_live_jobs_are_untouched()
    test_no_jobs_is_safe()
    print(f"\nstale_reconcile: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
