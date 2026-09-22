#!/usr/bin/env python3
"""Backend smoke test (no whisper/yt-dlp needed): config validation, fail-fast
ffmpeg check, job creation, worker consumption, and the honest failure path."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use throwaway folders so the real ./data stays clean.
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CLIPFORGE_SMOKE", "1")


def main() -> int:
    print("== 1. config load + boot banner")
    from app.config import boot_banner, load_config

    settings = load_config()
    boot_banner(settings)
    assert settings.video.width == 1080 and settings.video.height == 1920

    print("== 2. fail-fast: missing ffmpeg must raise ConfigError")
    from app import config as config_mod

    original_which = config_mod.shutil.which

    def no_ffmpeg(name):
        return None if name in ("ffmpeg", "ffprobe") else original_which(name)

    config_mod.shutil.which = no_ffmpeg
    try:
        config_mod.ffmpeg_bin()
        print("   FAIL: expected ConfigError for missing ffmpeg")
        return 1
    except config_mod.ConfigError as exc:
        print(f"   OK: {str(exc)[:90]}...")
    finally:
        config_mod.shutil.which = original_which

    print("== 3. API smoke: health, job create, worker consumes, honest fail")
    from fastapi.testclient import TestClient

    # Don't run real warmup (whisper not installed here); patch it out.
    import app.main as main_mod

    main_mod._do_warmup = lambda: None

    # context-manager form so FastAPI startup events (worker.start) actually fire
    with TestClient(main_mod.app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200, r.text
        health = r.json()
        print(f"   health: worker={health['worker']['online']}, llm={health['llm']}")
        assert health["ok"] is True
        assert health["worker"]["online"] is True, "in-process worker must be online"

        r = client.post(
            "/api/jobs",
            json={"sourceType": "url", "source": "https://example.com/not-a-video.mp4"},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["jobId"]
        print(f"   job created: {job_id}")

        # wait for the worker thread to attempt the job (it must FAIL honestly —
        # no fake stages, no heuristics)
        deadline = time.time() + 25
        stage, error = None, None
        while time.time() < deadline:
            job = main_mod.store.get_job(job_id)
            stage, error = job["stage"], job.get("error")
            if stage in ("failed", "complete"):
                break
            time.sleep(0.5)
        job = main_mod.store.get_job(job_id)
        print(f"   final stage: {stage}, error: {(error or '')[:110]}")
        if stage != "failed" or not error:
            print("   FAIL: job should fail honestly (yt-dlp/network unavailable)")
            return 1

        logs = client.get(f"/api/jobs/{job_id}/logs").json()
        print(f"   job log lines: {len(logs)}")

        # -- 3b. new/changed endpoints ---------------------------------
        # verify-clips on an empty store
        r = client.get("/api/dev/verify-clips")
        body = r.json() if r.status_code == 200 else {}
        if r.status_code != 200 or body.get("total") != 0:
            print(f"   FAIL: verify-clips empty store: {r.status_code} {body}")
            return 1
        print("   verify-clips empty store: OK")

        # redetect on a missing job -> readable 404, not a crash
        r = client.post("/api/jobs/doesnotexist/redetect")
        if r.status_code != 404 or "not found" not in r.json().get("detail", ""):
            print(f"   FAIL: redetect missing job: {r.status_code} {r.text[:120]}")
            return 1
        print("   redetect missing job: OK")

        # export/zip with an unknown clip id -> readable 404 (regression:
        # this used to raise TypeError on a None clip)
        r = client.post("/api/export/zip", json={"clipIds": ["fake-id"]})
        detail = r.json().get("detail", "") if r.status_code == 404 else ""
        if r.status_code != 404 or "no rendered clips" not in detail:
            print(f"   FAIL: export zip empty: {r.status_code} {detail}")
            return 1
        print("   export zip empty: OK")

        # video/download endpoints: inline for <video>, attachment for MP4 DL
        import subprocess
        import tempfile

        from app.config import ffmpeg_bin

        tmp = Path(tempfile.mkdtemp(prefix="cf-smoke-"))
        fake_mp4 = tmp / "clip.mp4"
        subprocess.run(
            [ffmpeg_bin(), "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
             "-c:v", "libx264", "-preset", "ultrafast", str(fake_mp4)],
            check=True, capture_output=True,
        )
        main_mod.store.write_clip({
            "_id": "smokeclip1", "jobId": "smokejob1", "order": 0,
            "start": 0.0, "end": 1.0, "score": 90, "title": "Smoke test clip",
            "hookText": "", "transcript": "", "hashtags": [], "captionCues": [],
            "aspect": "9:16", "videoPath": str(fake_mp4), "thumbPath": None,
            "renderState": "done",
        })
        r = client.get("/api/clips/smokeclip1/video")
        if r.status_code != 200 or "attachment" in r.headers.get("content-disposition", ""):
            print(f"   FAIL: video endpoint should stream inline: {r.status_code}")
            return 1
        r = client.get("/api/clips/smokeclip1/download")
        if r.status_code != 200 or "attachment" not in r.headers.get("content-disposition", ""):
            print(f"   FAIL: download endpoint should be attachment: {r.status_code}")
            return 1
        r = client.get("/api/dev/verify-clips?jobId=smokejob1")
        entry = (r.json().get("clips") or [{}])[0]
        if r.status_code != 200 or entry.get("valid") is not True:
            print(f"   FAIL: verify-clips on real file: {r.status_code} {entry}")
            return 1
        print("   video inline / download attachment / verify real file: OK")
        Path("data", "clips", "smokeclip1.json").unlink(missing_ok=True)

    # cleanup the throwaway job
    main_mod.store.delete_job(job_id)
    if Path("downloads", job_id).exists():
        print("   note: downloads dir left for inspection")

    # The worker thread is a daemon; TestClient teardown can otherwise race it
    # and abort the interpreter ("terminate called without an active exception").
    main_mod.worker.stop(timeout=5)

    print("== 4. llm guard: detection without key raises LlmNotConfigured")
    from app import llm as llm_mod

    try:
        llm_mod.detect_clips([{"start": 0, "end": 1, "word": "x"}], settings)
        print("   FAIL: expected LlmNotConfigured")
        return 1
    except llm_mod.LlmNotConfigured as exc:
        print(f"   OK: {exc}")

    print("== 5. silence-trim + reframe filters are pure functions")
    from app import video as video_mod

    graph, new_dur = video_mod.silence_trim_filter(
        [{"start": 0.0, "end": 0.8}, {"start": 3.0, "end": 4.0}], 5.0
    )
    assert graph and abs(new_dur - (5.0 - 0.8 - 1.0)) < 0.01, (graph, new_dur)
    print(f"   trim graph ok ({new_dur:.1f}s kept)")
    pan = video_mod.crop_pan_filter(
        [{"t": 0, "x": 0, "w": 100, "h": 100}, {"t": 2, "x": 50, "w": 100, "h": 100}],
        200, 100, 56, 100,
    )
    assert pan.startswith("crop="), pan[:60]
    print("   crop-pan graph ok")

    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Hard-exit: daemon threads (worker, anyio pool) can race interpreter
    # teardown and abort the C runtime AFTER all checks passed. os._exit skips
    # that teardown, so the exit code always reflects the real test result.
    os._exit(rc)
