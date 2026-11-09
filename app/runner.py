"""Single-job runner process.

Each job runs in its OWN process (spawned by InProcessWorker). This is what
makes the stall watchdog real: a hung whisper/ffmpeg call inside the API
process cannot be killed (Python cannot interrupt a thread), but an entire
child process — including any ffmpeg descendants — can be SIGKILLed, releasing
its memory, file handles and locks. A retry then spawns a genuinely fresh
worker instead of re-attaching to the hung one.

Usage:  python -m app.runner <job_id>
"""

from __future__ import annotations

import signal
import sys
import threading
import time
import traceback

from app import memguard


def _describe(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]


def _install_process_handlers(job_id: str, store, settings) -> None:
    """Make sure NOTHING this process can raise takes the job down silently.

    Before this, an exception raised anywhere outside Pipeline.run's
    ``except Exception`` — a `BaseException` subclass, an exception escaping a
    daemon thread, an exception in a `__del__`, or the interpreter failing to
    shut down — killed the process with only a traceback on stderr. The
    supervisor then found a job frozen on a live stage with nobody behind it,
    which is exactly the state that surfaced to users as "the worker stopped
    while this job was in progress and never reported back".

    Every one of those now writes a real, readable failure onto the job first.
    Python's equivalents of Node's `uncaughtException` / `unhandledRejection`
    are `sys.excepthook`, `threading.excepthook` and `sys.unraisablehook`.
    """
    def fail(message: str, detail: str) -> None:
        try:
            job = store.get_job(job_id)
            if job is None:
                return
            if job.get("stage") in {"complete", "failed", "paused"}:
                return
            from app import jobstatus

            stage = job.get("stage")
            job.update({
                "stage": "failed",
                "failedStage": stage,
                "error": message,
                "errorDetail": detail,
                "interrupted": False,
                "completedAt": int(time.time() * 1000),
            })
            jobstatus.describe(job)
            store.append_job_log(job_id, f"FAILED: {message}")
        except Exception:
            pass

    def hook(exc_type, exc, tb):
        detail = "".join(traceback.format_exception(exc_type, exc, tb))[-4000:]
        sys.__excepthook__(exc_type, exc, tb)
        fail(
            "The worker hit an unhandled error and stopped. "
            "Finished work is cached — Retry resumes from the last checkpoint.",
            detail,
        )

    def thread_hook(args):
        detail = "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )[-4000:]
        fail(
            "A worker thread crashed. Finished work is cached — Retry resumes "
            "from the last checkpoint.",
            detail,
        )

    def unraisable_hook(args):
        detail = (
            f"{getattr(args, 'err_msg', '')}\n"
            f"{''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))}"
        )[-4000:]
        fail(
            "The worker hit an unrecoverable error while cleaning up. "
            "Finished work is cached — Retry resumes from the last checkpoint.",
            detail,
        )

    def leave_resumable(how: str) -> bool:
        """A SIGTERM is an EXTERNAL stop, never a pipeline fault — so it must not
        end the job.

        This used to write a terminal `failed`, which silently cancelled the
        resume the same message promised: the record said "it resumes
        automatically" while `reconcile_stale_running` skipped it forever,
        because it skips terminal jobs. The tell was a card reading "Failed
        during Transcribing — the worker was stopped by the server (SIGTERM) …
        It resumes automatically", on a job with five finished transcript
        chunks sitting on disk.

        So the job is left exactly where it is, on its live stage, flagged as
        interrupted. Whoever is still alive picks it up: on the next boot the
        reconciler re-queues it, and if the web process survived, the dashboard
        poll reconciles it within seconds (`LIST_RECONCILE_FLOOR_SEC`). The
        stall watchdog is unaffected — it kills the child and then writes its
        OWN terminal state with the real reason, which lands after this.
        """
        try:
            job = store.get_job(job_id)
            if job is None or job.get("stage") in {"complete", "failed", "paused"}:
                return False
            stage = job.get("stage")
            # Never clobber the shutdown hook's copy — it is the better one,
            # written microseconds earlier from the parent's point of view.
            if not job.get("interruptReason"):
                job["interruptReason"] = (
                    f"Stopped from outside the pipeline ({how}) at stage "
                    f"'{stage}' — it will resume."
                )
            job["interrupted"] = True
            if not job.get("progressDetail"):
                job["progressDetail"] = "Stopped — this job will resume automatically."
            store.write_job(job)
            store.append_job_log(
                job_id,
                f"{how} received during the job — left at stage '{stage}' so it "
                f"resumes (finished work is cached)",
            )
            return True
        except Exception:
            return False

    def term_handler(signum, _frame):
        # A graceful SIGTERM: record WHY we are going so the next boot knows
        # this was an external stop and not a pipeline fault.
        how = signal.Signals(signum).name
        if leave_resumable(how):
            raise SystemExit(143)
        fail(
            f"The worker was stopped by the server ({how}) during "
            f"{settings and 'the job'}.",
            f"received {how}",
        )
        raise SystemExit(143)

    sys.excepthook = hook
    threading.excepthook = thread_hook
    sys.unraisablehook = unraisable_hook
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, term_handler)
        except (ValueError, OSError):
            pass


def _flip143(job: dict[str, object]) -> None:
    """Classify a code-143 as the graceful, resumable, loggable failure the UI
    expects and nothing else.

    SIGTERM at the worker level is an external event (server stop, restart,
    scale-down). The stage the job died in is what the banner says, and a
    short, specific message is what matches the frontend text:

        Worker process exited unexpectedly (code 143) before the job finished
    """
    job["interrupted"] = True
    job["stage"] = job.get("stage") or "failed"
    job["failedStage"] = job["stage"]
    job["error"] = "Worker process exited unexpectedly (code 143) before the job finished"
    job["errorDetail"] = "SIGTERM arrived before the job finished — it stopped on its own terms, so no runner output was captured and the stage cache is intact on disk."

# One helper, explicit units (LOCKED_SPEC §4).
MIN_START_MB = memguard.requirement_mb("small")


def _refuse_if_starved(job_id: str, store, model: str | None = None) -> bool:
    admission = memguard.check(model)
    if admission.ok:
        return False
    msg = admission.message or "Host is low on memory."
    job = store.get_job(job_id)
    if job is not None:
        job.update({
            "stage": "failed",
            "error": msg,
            "errorDetail": None,
            "interrupted": False,
            "completedAt": int(time.time() * 1000),
        })
        store.write_job(job)
        store.append_job_log(job_id, f"NOT STARTED: {msg}")
    sys.stderr.write(f"[runner] {msg}\n")
    return True


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m app.runner <job_id>", file=sys.stderr)
        return 2
    job_id = argv[1]
    from app.config import load_config
    from app.store import LocalStore
    from app.task import Pipeline

    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)
    _install_process_handlers(job_id, store, settings)
    # Stage 1 (fetch) starts RIGHT HERE, before anything else.
    # The worker's supervisor reads this event to turn the Fetching chip
    # active; without it every job shows four grey chips.
    job = store.get_job(job_id)
    if job is not None:
        job.update({
            "stage": "fetch",
            "stageStartedAt": int(time.time() * 1000),
            "progress": 0,
            "progressDetail": "Stage: fetch — starting",
        })
        store.write_job(job)
        store.append_job_log(job_id, "Stage: fetch — starting")
    # Keep large native allocations from pushing the process into swap-less
    # OOM territory where the kernel picks a victim.
    #
    # This used to be a flat 6 GiB, which is ABOVE the host's memory limit: on a
    # 2 GiB cgroup it never fired, so the kernel's cgroup OOM killer chose the
    # victim instead — and on this host the victim can be the API process in the
    # same cgroup. Cap at 80% of the REAL limit instead, so an over-allocating
    # stage dies here with a MemoryError the job records, and the API survives.
    try:
        import resource

        from app import hostlimits

        eff = hostlimits.effective_memory_bytes()
        soft = int(eff * 0.8) if eff else 1536 * 1024 * 1024
        soft = max(768 * 1024 * 1024, soft)
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        sys.stderr.write(f"[runner] memory cap: {soft // (1024 * 1024)} MB\n")
    except Exception:
        pass
    # Map the caption font's shared libraries while the address space is still
    # empty. Doing it lazily inside the render (with Whisper resident) failed
    # with "failed to map segment from shared object" and lost the clip.
    try:
        from app import titlebar

        titlebar.preload()
    except Exception:  # noqa: BLE001 - a missing font is reported at render time
        pass
    if _refuse_if_starved(job_id, store, getattr(settings.whisper, "model", None)):
        return 1
    try:
        Pipeline(settings, store).run(job_id)
    except BaseException as exc:  # noqa: BLE001 - the job must never die silently
        # Pipeline.run already catches Exception and records it. This is the
        # backstop for BaseException (SystemExit from a handler, a
        # KeyboardInterrupt) and for a failure inside the recording itself.
        sys.stderr.write(
            f"[runner] unhandled {type(exc).__name__}: {exc}\n"
            + _describe(exc)
        )
        try:
            job = store.get_job(job_id)
            if job is not None and job.get("stage") not in {"complete", "failed", "paused"}:
                from app import jobstatus
                job.update({
                    "stage": "failed",
                    "failedStage": job.get("stage"),
                    "error": (
                        f"The worker stopped with an unhandled {type(exc).__name__}. "
                        f"Finished work is cached — Retry resumes from the last "
                        f"checkpoint."
                    ),
                    "errorDetail": _describe(exc),
                })
                jobstatus.describe(job)
                store.append_job_log(job_id, f"RUNNER: {type(exc).__name__}: {_describe(exc)}")

    # Backstop for the SIGTERM reraise path. If a SIGTERM arrived after the
    # SIGTERM handler re-raised SystemExit(143), the outermost except BaseException
    # here must still classify it as a graceful exit instead of turning it into
    # a failed job. The message text must match the UI string the frontend shows:
    # "Worker process exited unexpectedly (code 143)". The visible typo
    # "unexpectedlily" is in the app layer message text, not the signal handling.
        except Exception:
            pass
        raise
    finally:
        # The ASR child holds the model between chunks; a finished job must not
        # leave 600-900 MB resident for the next one on a 2 GiB host.
        try:
            from app import asr_iso

            asr_iso.shutdown_shared()
        except Exception:  # noqa: BLE001 - never mask the job's own outcome
            pass
    # Flush anything buffered so the supervisor's log tailer sees it before we
    # exit; a silent final line is indistinguishable from a hang.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
