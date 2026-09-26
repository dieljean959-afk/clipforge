#!/usr/bin/env python3
"""Focused worker supervision and resumability checks.

These tests avoid Whisper/yt-dlp and exercise the real process-group kill,
watchdog state transition, retry queueing, and StageWorkspace markers.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.worker as worker_mod
from app.config import load_config
from app.stages import STAGES, StageWorkspace
from app.store import LocalStore
from app.worker import InProcessWorker


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    tmp = root / "data" / "worker-supervision-test"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["CLIPFORGE_DOWNLOADS_DIR"] = str(tmp / "downloads")
    os.environ["CLIPFORGE_OUTPUTS_DIR"] = str(tmp / "outputs")
    os.environ["CLIPFORGE_DATA_DIR"] = str(tmp / "data")
    os.environ["CLIPFORGE_STALL_TIMEOUT_SEC"] = "1"
    # The module constant is read at import time; set it explicitly for the
    # focused test rather than depending on process startup ordering.
    worker_mod.STALL_TIMEOUT_SEC = 1

    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)
    worker = InProcessWorker(settings, store)
    job = store.create_job("upload", "sample.mp4")
    job_id = job["_id"]

    # Keep the first two stages complete and prove the first incomplete stage
    # is the recovery boundary.
    ws = StageWorkspace(settings.app.paths.data_dir, job_id)
    ws.write_json("fetch", "meta.json", {"path": "sample.mp4"})
    ws.mark_done("fetch", artifacts=["fetch/meta.json"], path="sample.mp4")
    ws.write_json("transcribe", "words.json", {"words": [], "duration": 0})
    ws.mark_done("transcribe", artifacts=["transcribe/words.json"], words=0)
    assert ws.first_incomplete() == "detect", ws.first_incomplete()

    # A real child process stands in for a hung runner. start_new_session makes
    # it a process group, matching the production worker launch.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    worker._current_job = job_id
    worker._current_proc = proc
    job = store.get_job(job_id) or job
    job.update({
        "stage": "transcribing",
        "lastProgressAt": int(time.time() * 1000 - 5_000),
    })
    store.write_job(job)

    assert worker._check_watchdog_once() is True
    proc.wait(timeout=5)
    stalled = store.get_job(job_id)
    # A stall is environmental and often transient, so the watchdog now buys
    # the job ONE automatic retry before the user is told anything. The job is
    # requeued rather than left failed — this is the behaviour the dashboard
    # used to make the user trigger by hand after a 20-minute wait.
    assert stalled and stalled["stage"] == "queued", stalled["stage"]
    assert stalled.get("autoRetryCount") == 1, stalled.get("autoRetryCount")
    assert "auto-retry" in "\n".join(stalled.get("logs", [])).lower()
    assert worker._stalled_job == job_id

    # The automatic retry must be exactly that: ONE. A second stall with no
    # human in the loop would otherwise cycle forever.
    stalled.update({"stage": "failed", "error": "Pipeline stalled — no progress for 600s at stage rendering."})
    store.write_job(stalled)
    assert worker._auto_retry(job_id, stalled, "second stall") is False
    assert store.get_job(job_id)["stage"] == "failed"

    # A non-retryable failure must NOT be retried: a bad URL or a missing key
    # would just spend quota to reach the same error.
    bad = store.get_job(job_id)
    bad.update({"autoRetryCount": 0, "error": "Could not fetch video: not a video",
                "stage": "failed"})
    store.write_job(bad)
    assert worker._auto_retry(job_id, bad, "bad url") is False
    assert store.get_job(job_id)["stage"] == "failed"

    # Retry must queue a new attempt and preserve the completed-stage markers.
    cur = store.get_job(job_id)
    cur.update({"stage": "failed", "error": "Pipeline stalled — no progress for 600s at stage rendering."})
    store.write_job(cur)
    depth = worker.retry(job_id)
    assert depth == 1, depth
    retried = store.get_job(job_id)
    assert retried and retried["stage"] == "queued"
    assert retried.get("completedAt") is None
    # The expensive stages are still cached, so a resume never re-downloads,
    # re-transcribes or re-pays for detection.
    assert StageWorkspace(settings.app.paths.data_dir, job_id).first_incomplete() == "detect"
    assert ws.is_done("fetch") and ws.is_done("transcribe")

    # --- externally-killed jobs resume instead of dying permanently -------
    # Regression: a SIGTERM/SIGKILL from the host (restart, redeploy,
    # scale-down) marked the job stage=failed forever, throwing away cached
    # transcription that was intact on disk. A signalled child did not fail on
    # its own merit, so it must be resumed — with a bounded budget.
    assert worker_mod.INTERRUPT_MAX_RESUMES >= 1
    enqueued: list[str] = []
    real_enqueue = worker.enqueue
    worker.enqueue = lambda jid: (enqueued.append(jid), 1)[1]  # type: ignore[assignment]
    try:
        j = store.get_job(job_id) or {}
        j.update({"stage": "rendering", "error": None, "completedAt": None,
                  "interruptCount": 0, "interrupted": False})
        store.write_job(j)

        handled = worker._note_interrupted(store.get_job(job_id), job_id, -15, "rendering", "some tail")
        after = store.get_job(job_id) or {}
        assert handled is True
        assert after["stage"] == "rendering", after["stage"]      # stage PRESERVED for resume
        assert after["interrupted"] is True
        assert after["interruptCount"] == 1
        assert after.get("error") is None                         # not treated as a failure
        assert after.get("completedAt") is None
        assert enqueued == [job_id], enqueued                     # auto-resumed immediately
        assert any("INTERRUPTED" in str(entry) for entry in (after.get("logs") or [])), "must be logged"
        assert "SIGTERM" in (after.get("interruptReason") or ""), after.get("interruptReason")

        # SIGKILL (the OOM case) resumes the same way
        j2 = store.get_job(job_id) or {}
        j2.update({"stage": "finding", "interruptCount": 0})
        store.write_job(j2)
        assert worker._note_interrupted(store.get_job(job_id), job_id, -9, "finding", "") is True
        assert (store.get_job(job_id) or {}).get("stage") == "finding"

        # bounded: once the budget is spent it must stop auto-resuming, so the
        # caller falls through to a real failure rather than looping forever
        j3 = store.get_job(job_id) or {}
        j3.update({"stage": "rendering", "interruptCount": worker_mod.INTERRUPT_MAX_RESUMES})
        store.write_job(j3)
        exhausted = worker._note_interrupted(
            store.get_job(job_id), job_id, -15, "rendering", ""
        )
        assert exhausted is False, "resume budget must be finite"

        # shutdown hook must be safe and non-throwing with no live children
        assert isinstance(worker.mark_in_flight_interrupted(), list)
    finally:
        worker.enqueue = real_enqueue  # type: ignore[assignment]

    # --- signalled-worker attribution -------------------------------------
    # Regression: -15 (SIGTERM) used to fall through to a bare "exited
    # unexpectedly" with no cause, which is unactionable. -9 (SIGKILL) is the
    # OOM case and must stay distinct from it.
    from app import hostlimits

    hint_15 = hostlimits.oom_hint(-15, "fetch")
    assert hint_15 and "SIGTERM" in hint_15, hint_15
    assert "NOT an out-of-memory kill" in hint_15, hint_15
    assert "during 'fetch'" in hint_15, hint_15
    hint_9 = hostlimits.oom_hint(-9, "transcribe")
    assert hint_9 and "out-of-memory kill" in hint_9, hint_9
    assert "SIGTERM" not in hint_9, hint_9
    assert hostlimits.oom_hint(1, "render") is None
    assert hostlimits.oom_hint(0, None) is None
    assert hostlimits.oom_hint(None, None) is None

    # --- ffmpeg error extraction -----------------------------------------
    # Regression: a naive stderr tail returned eight progress frames and
    # dropped the real error, so a failed render reported nothing usable.
    from app import video

    progress_noise = "\r".join(
        f"frame=  {i} fps=6.3 q=28.0 size=     256kB time=00:00:0{i}.00 bitrate= 100.0kbits/s speed=0.10x"
        for i in range(1, 40)
    )
    real_error = "[libx264 @ 0x55] height not divisible by 2 (1080x1081)\nConversion failed!"
    tail = video._ffmpeg_error_tail(progress_noise + "\n" + real_error)
    assert "height not divisible" in tail, tail
    assert "frame=" not in tail, tail

    only_progress = video._ffmpeg_error_tail(progress_noise)
    assert "no diagnostic line" in only_progress, only_progress
    assert "killed by a signal" in only_progress, only_progress

    assert video._ffmpeg_error_tail("") == "(ffmpeg produced no output)"

    worker.stop(timeout=2)
    store.delete_job(job_id)
    for stage in STAGES:
        ws.clear(stage)
    print("WORKER SUPERVISION: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
