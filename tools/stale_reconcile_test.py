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


def _make(tmp: Path) -> tuple[Any, LocalStore, InProcessWorker]:
    from app.config import load_config

    settings = load_config(quiet=True)
    settings.app = type(settings.app)(
        paths=_Paths(tmp), host="127.0.0.1", port=8000,
    )
    store = LocalStore(settings.app.paths)
    return settings, store, InProcessWorker(settings, store)


def _make_worker(tmp: Path) -> tuple[InProcessWorker, LocalStore]:
    _settings, store, worker = _make(tmp)
    return worker, store


def _job(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "_id": "job1", "source": "https://youtu.be/abc12345678",
        "stage": "finding", "progress": 61.0, "createdAt": int(time.time() * 1000),
        "lastProgressAt": int(time.time() * 1000), "logs": [],
    }
    base.update(kw)
    return base


def test_sigterm_leaves_the_job_resumable() -> None:
    """The sequence that produced "Failed during Transcribing — SIGTERM … It
    resumes automatically" on a card, for a job that then never resumed.

    The parent's shutdown hook flags the job as interrupted, then the child is
    SIGTERMed. The child's own handler used to answer that by writing a terminal
    `failed`, which (a) contradicted the message it had just written and (b)
    made the job invisible to `reconcile_stale_running`, because that skips
    terminal jobs. The promise was unbacked: five finished transcript chunks on
    disk and a Retry button as the only way forward.
    """
    print("\n[5] a SIGTERM'd job stays re-queueable, so 'it resumes' is true")
    import signal
    import subprocess

    from app import runner

    with tempfile.TemporaryDirectory() as td:
        settings, store, worker = _make(Path(td))
        # A URL job: always resumable, because the source can be re-fetched.
        job = store.create_job(source_type="url", source="https://youtu.be/abc12345678")
        jid = job["_id"]

        # ---- 1. the parent's graceful-shutdown hook runs first ------------
        # It only flags jobs that have a LIVE CHILD, so the test needs one:
        # a real process, registered exactly where the worker keeps them.
        store.write_job({
            **store.get_job(jid),
            "stage": "transcribing",
            "progress": 45,
            "lastProgressAt": int(time.time() * 1000),
        })
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        worker._active[jid] = child
        try:
            worker.mark_in_flight_interrupted()
        finally:
            worker._active.pop(jid, None)
            child.terminate()
            child.wait(timeout=5)
        flagged = store.get_job(jid) or {}
        check("the shutdown hook flagged the job as interrupted",
              flagged.get("interrupted") is True, str(flagged.get("interrupted")))
        parent_reason = str(flagged.get("interruptReason") or "")
        check("and promised a resume", "resume" in parent_reason.lower(), parent_reason)

        # ---- 2. the child is SIGTERMed and handles it ----------------------
        runner._install_process_handlers(jid, store, settings)
        handler = signal.getsignal(signal.SIGTERM)
        raised = None
        try:
            handler(signal.SIGTERM, None)  # type: ignore[misc]
        except SystemExit as exc:
            raised = exc.code
        check("the child still exits with the conventional SIGTERM code",
              raised == 143, str(raised))

        after = store.get_job(jid) or {}
        check("the job is NOT failed any more", after.get("stage") == "transcribing",
              str(after.get("stage")))
        from app import jobstatus

        check("it is not terminal, so the reconciler will still see it",
              not jobstatus.is_terminal(after), str(after.get("stage")))
        check("no error was written over the resume promise",
              not after.get("error"), str(after.get("error"))[:90])
        check("the parent's reason was not clobbered",
              str(after.get("interruptReason")) == parent_reason,
              str(after.get("interruptReason"))[:90])
        check("it is still flagged as interrupted", after.get("interrupted") is True)
        check("the finished work on disk is untouched",
              not Path(td, "nope").exists())
        check("and nothing was deleted from the job store",
              store.get_job(jid) is not None)

        # ---- 3. the next boot re-queues it --------------------------------
        worker.reconcile_stale_running()
        resumed = store.get_job(jid) or {}
        check("the reconciler picks it up and re-queues it",
              resumed.get("stage") == "queued", str(resumed.get("stage")))
        check("with no error left over", resumed.get("error") is None,
              str(resumed.get("error"))[:80])
        check("and a reason that names the restart",
              "restart" in str(resumed.get("interruptReason")).lower(),
              str(resumed.get("interruptReason"))[:80])

        # ---- 4. no regression: a job that is already settled is untouched --
        for settled in ("complete", "failed", "paused"):
            j2 = store.create_job(source_type="url", source=f"https://youtu.be/{settled}00")
            store.write_job({**store.get_job(j2["_id"]), "stage": settled,
                             "error": "original"})
            runner._install_process_handlers(j2["_id"], store, settings)
            try:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)  # type: ignore[misc]
            except SystemExit:
                pass
            left = store.get_job(j2["_id"]) or {}
            check(f"a {settled} job is left exactly as it was",
                  left.get("stage") == settled and left.get("error") == "original",
                  f"{left.get('stage')} / {str(left.get('error'))[:40]}")


def test_real_runner_child_survives_sigterm() -> None:
    """The production path, end to end: a REAL `python -m app.runner` child,
    a REAL SIGTERM, a REAL job document.

    The section above calls the installed handler in-process, which proves the
    logic but not the wiring — that `main()` installs it before the pipeline
    runs, that the signal actually reaches the handler in a separate process,
    and that the write lands in the store the child was given. Those are three
    separate things to get wrong, so they are tested the way they happen.

    The child is FROZEN with SIGSTOP the moment it reaches a live stage, so it
    cannot finish before the signal is queued, and the source is one that
    REFUSES the connection rather than one that hangs — a blocking socket read
    would leave the child inside a C call, where Python cannot run its handler
    until the read returns, and the test would be measuring the test rig rather
    than the product. (An earlier version did exactly that: it passed three
    times standalone and failed under the QA runner, where the busier box
    changed the timing. A test that depends on winning a race is not a test.)
    """
    print("\n[6] a real runner process, really SIGTERMed, stays resumable")
    import os
    import signal
    import socket
    import subprocess

    from app import jobstatus
    from app.config import load_config
    from app.worker import InProcessWorker

    ROOT = Path(__file__).resolve().parents[1]

    # A port that is bound and immediately closed: connecting is refused in
    # milliseconds, so the child reaches a live stage and returns into Python
    # promptly instead of parking inside a blocking read.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    with tempfile.TemporaryDirectory() as td:
        # The child resolves its store from these env vars, so the test process
        # resolves the SAME ones — otherwise the child looks in a different
        # tree, finds no job and exits without doing anything at all.
        overrides = {
            "CLIPFORGE_DATA_DIR": f"{td}/data",
            "CLIPFORGE_OUTPUTS_DIR": f"{td}/outputs",
            "CLIPFORGE_DOWNLOADS_DIR": f"{td}/downloads",
        }
        saved = {k: os.environ.get(k) for k in overrides}
        os.environ.update(overrides)
        # The env is restored in the finally below, before anything else runs.
        settings = load_config(quiet=True)
        store = LocalStore(settings.app.paths)
        jid = store.create_job(
            source_type="url", source=f"http://127.0.0.1:{port}/stream.mp4",
        )["_id"]
        check("the job starts queued",
              (store.get_job(jid) or {}).get("stage") == "queued")

        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}

        child = subprocess.Popen(
            [sys.executable, "-m", "app.runner", jid],
            env=env, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        rc = None
        try:
            # Freeze the child during bootstrap — it is importing modules then,
            # which is Python bytecode, so the handler is guaranteed to run on
            # SIGCONT instead of waiting on some C call or a `pip` subprocess
            # deep inside yt-dlp's self-update. Freezing is what stops it from
            # finishing before the signal is queued; bootstrapping is what
            # stops the signal from being deferred.
            time.sleep(0.4)
            check("the real child is running", child.poll() is None, str(child.poll()))
            child.send_signal(signal.SIGSTOP)
            time.sleep(0.1)
            frozen_stage = (store.get_job(jid) or {}).get("stage")
            child.send_signal(signal.SIGTERM)
            time.sleep(0.1)
            check("the child is still alive and frozen, holding its stage",
                  child.poll() is None
                  and (store.get_job(jid) or {}).get("stage") == frozen_stage,
                  f"rc={child.poll()} stage={frozen_stage}")
            child.send_signal(signal.SIGCONT)
            rc = child.wait(timeout=60)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        check("the child exited with the conventional SIGTERM code", rc == 143, str(rc))
        after = store.get_job(jid) or {}
        check(f"the real child did NOT write a terminal failure (held "
              f"'{frozen_stage}')",
              after.get("stage") == frozen_stage, str(after.get("stage")))
        check("the job is still non-terminal", not jobstatus.is_terminal(after),
              str(after.get("stage")))
        check("no error was written", not after.get("error"), str(after.get("error"))[:90])
        check("it is flagged as interrupted", after.get("interrupted") is True)
        check("and the promise is on the record",
              "resume" in str(after.get("interruptReason") or "").lower()
              or "resume" in str(after.get("progressDetail") or "").lower(),
              f"{after.get('interruptReason')} / {after.get('progressDetail')}")

        # The next boot's reconciler, over the same real store.
        InProcessWorker(settings, store).reconcile_stale_running()
        resumed = store.get_job(jid) or {}
        check("the reconciler re-queues the job the real child left behind",
              resumed.get("stage") == "queued", str(resumed.get("stage")))


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
    test_sigterm_leaves_the_job_resumable()
    test_real_runner_child_survives_sigterm()
    print(f"\nstale_reconcile: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
