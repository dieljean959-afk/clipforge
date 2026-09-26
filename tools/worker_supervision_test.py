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
    failed = store.get_job(job_id)
    assert failed and failed["stage"] == "failed"
    assert "killed" in (failed.get("error") or "").lower()
    assert "stall watchdog" in "\n".join(failed.get("logs", [])).lower()
    assert worker._stalled_job == job_id

    # Retry must queue a new attempt and preserve the completed-stage markers.
    depth = worker.retry(job_id)
    assert depth == 1
    retried = store.get_job(job_id)
    assert retried and retried["stage"] == "queued"
    assert retried.get("completedAt") is None
    assert StageWorkspace(settings.app.paths.data_dir, job_id).first_incomplete() == "detect"
    assert ws.is_done("fetch") and ws.is_done("transcribe")

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
