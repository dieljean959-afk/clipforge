#!/usr/bin/env python3
# ClipForge worker VM agent (Oracle Cloud Always Free).
#
# Runs as a systemd service on the worker VM. Loop:
#   1. heartbeat  -> POST /api/worker/heartbeat   (proves liveness; may be
#                    told to drain-and-stop by the control plane)
#   2. claim      -> GET  /api/worker/claim       (oldest queued URL job)
#   3. download   -> yt-dlp PYTHON API (import yt_dlp, NOT subprocess):
#                      format "bv*[height<=1080]+ba/b[height<=1080]",
#                      merged to MP4, progress_hooks report real download %
#                      + MB downloaded. On ImportError/failure the agent
#                      self-updates yt-dlp IN-PROCESS (pip) and retries once.
#   4. verify     -> ffprobe: duration > 0 and playable, else fail the job
#   5. upload     -> POST the MP4 to the Convex storage upload URL
#   6. progress   -> POST /api/worker/progress   (bytes done/total)
#   7. fail       -> POST /api/worker/fetch-failed (exact yt-dlp/stderr error)
#
# Env (see deploy.sh): CLIPFORGE_BASE_URL, CLIPFORGE_WORKER_TOKEN,
# YT_DLP_AUTOUPDATE (default 1), WORKER_HOME.

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

BASE_URL = os.environ.get("CLIPFORGE_BASE_URL", "").rstrip("/")
TOKEN = os.environ.get("CLIPFORGE_WORKER_TOKEN", "")
HOME = Path(os.environ.get("WORKER_HOME", "/opt/clipforge-worker"))
DOWNLOADS = HOME / "downloads"
DOWNLOADS.mkdir(parents=True, exist_ok=True)

MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB per job
HEARTBEAT_EVERY = 20  # seconds
CLAIM_EVERY = 5  # seconds
PROGRESS_EVERY = 2.0  # seconds between progress posts

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# yt_dlp is imported lazily (see load_yt_dlp) so the agent can boot even if
# the package needs a refresh first.
_yt_dlp = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def api_call(method: str, path: str, **kwargs):
    return requests.request(
        method, f"{BASE_URL}{path}", headers=HEADERS, timeout=30, **kwargs
    )


def load_yt_dlp():
    """Import yt_dlp; on any failure pip-install/upgrade it in-process."""
    global _yt_dlp
    try:
        import yt_dlp  # noqa: PLC0415

        _yt_dlp = yt_dlp
        return _yt_dlp
    except Exception as exc:  # ImportError or a broken install
        log(f"yt-dlp import failed ({exc}) — self-updating in-process...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
            check=False,
            timeout=300,
        )
        import yt_dlp  # noqa: PLC0415

        _yt_dlp = yt_dlp
        return _yt_dlp


def self_update_yt_dlp() -> bool:
    """yt-dlp rots fast (YouTube breaks it without warning) — refresh it."""
    if os.environ.get("YT_DLP_AUTOUPDATE", "1") != "1":
        return False
    log("auto-updating yt-dlp (pip install -U yt-dlp)...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    ok = r.returncode == 0
    log(f"yt-dlp self-update {'OK' if ok else 'FAILED'}: {(r.stderr or r.stdout)[-300:]}")
    return ok


def ffprobe(path: Path) -> dict | None:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_entries", "format=duration,size:stream=width,height,codec_name",
                str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            log(f"ffprobe failed: {out.stderr[-300:]}")
            return None
        data = json.loads(out.stdout or "{}")
        duration = float((data.get("format") or {}).get("duration") or 0)
        width = height = 0
        for s in data.get("streams") or []:
            if s.get("width"):
                width, height = int(s["width"]), int(s["height"])
                break
        if duration <= 0:
            return None
        return {"durationSec": duration, "width": width, "height": height}
    except Exception as exc:
        log(f"ffprobe error: {exc}")
        return None


def upload_to_convex(upload_url: str, path: Path) -> str | None:
    with path.open("rb") as f:
        put = requests.post(
            upload_url, data=f,
            headers={"Content-Type": "video/mp4"}, timeout=1800,
        )
    if not put.ok:
        log(f"storage upload failed: HTTP {put.status_code}")
        return None
    storage_id = (put.json() or {}).get("storageId")
    return storage_id


def get_upload_url(job_id: str) -> str | None:
    # The storage upload URL comes from a Convex action the worker can call
    # through the same HTTP gateway: /api/worker/upload-url (see http.ts).
    r = api_call("POST", "/api/worker/upload-url", json={"jobId": job_id})
    if not r.ok:
        log(f"upload-url fetch failed: HTTP {r.status_code}")
        return None
    return (r.json() or {}).get("uploadUrl")


def run_fetch(job: dict) -> None:
    job_id = job["jobId"]
    source = job["source"]
    ydl = load_yt_dlp()
    target = DOWNLOADS / f"{job_id}.mp4"
    last_progress = [0.0]

    def progress_hook(d):
        if d.get("status") == "downloading":
            done = int(d.get("downloaded_bytes") or 0)
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            now = time.time()
            if now - last_progress[0] >= PROGRESS_EVERY:
                last_progress[0] = now
                try:
                    api_call(
                        "POST", "/api/worker/progress",
                        json={"jobId": job_id, "bytesDone": done,
                              "bytesTotal": total},
                        timeout=10,
                    )
                except Exception:
                    pass
        elif d.get("status") == "finished":
            log(f"download complete ({d.get('total_bytes', 0) / (1024 * 1024):.1f} MB)")

    ydl_opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]",
        "merge_output_format": "mp4",
        "outtmpl": str(target),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "progress_hooks": [progress_hook],
        # Windows-safe + headless-safe defaults
        "concurrent_fragment_downloads": 4,
    }

    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            log(f"yt-dlp attempt {attempt}: {source[:70]}")
            with ydl.YoutubeDL(ydl_opts) as y:
                y.extract_info(source, download=True)
            break
        except Exception as exc:
            err = str(exc)[:1500]
            log(f"yt-dlp attempt {attempt} failed: {err}")
            api_call(
                "POST", "/api/worker/fetch-failed",
                json={"jobId": job_id, "error": err}, timeout=10,
            )
            if attempt == 1 and self_update_yt_dlp():
                ydl = load_yt_dlp()
                continue
            return  # chain exhausted; server falls through to next fetcher

    if not target.exists() or target.stat().st_size < 1024:
        api_call(
            "POST", "/api/worker/fetch-failed",
            json={"jobId": job_id, "error": "yt-dlp produced no file"}, timeout=10,
        )
        return

    size_mb = target.stat().st_size / (1024 * 1024)
    if target.stat().st_size > MAX_BYTES:
        api_call(
            "POST", "/api/worker/fetch-failed",
            json={"jobId": job_id,
                  "error": f"video is {size_mb:.0f} MB — over the 2 GB limit"},
            timeout=10,
        )
        target.unlink(missing_ok=True)
        return

    meta = ffprobe(target)
    if meta is None:
        api_call(
            "POST", "/api/worker/fetch-failed",
            json={"jobId": job_id,
                  "error": "ffprobe could not verify the download (corrupt file?)"},
            timeout=10,
        )
        target.unlink(missing_ok=True)
        return

    log(f"download complete ({size_mb:.1f} MB) — ffprobe verified "
        f"{meta['durationSec']:.0f}s {meta['width']}x{meta['height']}")

    upload_url = get_upload_url(job_id)
    if upload_url is None:
        return
    storage_id = upload_to_convex(upload_url, target)
    target.unlink(missing_ok=True)
    if storage_id is None:
        return

    # The browser decode-verifies the stored file and advances the pipeline
    # via reportFetch — the VM's part is done once the bytes are stored.
    log(f"stored {storage_id} — handing off to the pipeline")


def pending_jobs_count() -> int:
    try:
        r = api_call("GET", "/api/worker/claim")
        if r.ok and (r.json() or {}).get("job") is not None:
            return 1
    except Exception:
        pass
    return 0


def main() -> None:
    if not BASE_URL or not TOKEN:
        log("FATAL: CLIPFORGE_BASE_URL and CLIPFORGE_WORKER_TOKEN must be set")
        sys.exit(1)
    log(f"ClipForge worker online — {BASE_URL} — yt-dlp ready")
    load_yt_dlp()

    last_beat = 0.0
    while True:
        now = time.time()
        if now - last_beat >= HEARTBEAT_EVERY:
            last_beat = now
            try:
                r = api_call(
                    "POST", "/api/worker/heartbeat",
                    json={"modelLoaded": True, "pendingJobs": pending_jobs_count()},
                    timeout=10,
                )
                if r.ok and (r.json() or {}).get("drainAndStop"):
                    log("drain-and-stop requested — queue is empty, exiting "
                        "(systemd Restart=no on the stop unit will power off)")
                    sys.exit(0)
            except Exception as exc:
                log(f"heartbeat failed: {exc}")

        try:
            r = api_call("GET", "/api/worker/claim")
            if r.ok:
                job = (r.json() or {}).get("job")
                if job:
                    log(f"claimed job {job['jobId']}")
                    run_fetch(job)
        except Exception as exc:
            log(f"claim loop error: {exc}")

        time.sleep(CLAIM_EVERY)


if __name__ == "__main__":
    main()
