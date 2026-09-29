"""`reconcile_stale_running`: a job claiming "running" with no worker behind it.

A job's `stage` is written by a child process. If that process is SIGKILLed,
OOM-reaped or lost to a host restart, nothing overwrites it — the document keeps
saying "finding, 61%" forever with no process and no queue entry. That is the
state behind "Stage 3 of 4, running" on a job that had already stopped.

The worker knows which jobs it is running and which are queued, so it is the
authority. These tests pin that it fixes exactly the stale ones and leaves
everything else completely alone.

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
from app.worker import InProcessWorker  # noqa: E402

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
        for d in (self.data_dir, self.downloads_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)


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


def test_stale_running_is_cleared() -> None:
    print("\n[1] a running stage with no worker is reconciled to failed")
    with tempfile.TemporaryDirectory() as td:
        worker, store = _make_worker(Path(td))
        store.write_job(_job())
        fixed = worker.reconcile_stale_running()
        check("the job was reconciled", "job1" in fixed, str(fixed))
        job = store.get_job("job1") or {}
        check("stage is now failed", job.get("stage") == "failed", str(job.get("stage")))
        check("keeps the stage it died in", job.get("failedStage") == "finding",
              str(job.get("failedStage")))
        check("explains what happened",
              "never reported back" in str(job.get("error")), str(job.get("error"))[:60])
        check("progress is not left at 61%", job.get("progress") == 0,
              str(job.get("progress")))
        check("not marked interrupted (that is a different, resumable path)",
              job.get("interrupted") is False, str(job.get("interrupted")))
        for field in ("stagePlan", "unitsDone", "unitsTotal", "unitLabel",
                      "totalEtaSec", "queuePosition", "progressDetail"):
            check(f"{field} cleared", field not in job, str(job.get(field)))
        # Idempotent: a second pass must not keep rewriting the document.
        again = worker.reconcile_stale_running()
        check("idempotent — a settled job is left alone", again == [], str(again))


def test_queued_job_is_reconciled() -> None:
    print("\n[2] a queued job that is not in the queue is equally stale")
    with tempfile.TemporaryDirectory() as td:
        worker, store = _make_worker(Path(td))
        store.write_job(_job(_id="q1", stage="queued", progress=0))
        fixed = worker.reconcile_stale_running()
        check("reconciled", "q1" in fixed, str(fixed))
        job = store.get_job("q1") or {}
        check("stage is failed", job.get("stage") == "failed", str(job.get("stage")))


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
    test_stale_running_is_cleared()
    test_queued_job_is_reconciled()
    test_live_jobs_are_untouched()
    test_no_jobs_is_safe()
    print(f"\nstale_reconcile: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
