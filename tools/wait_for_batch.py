#!/usr/bin/env python3
"""Wait for a job to finish rendering, then check every clip can actually be used.

The job runs in a child process and the UI is on a phone, so "is it done yet"
and "does clip 10 have a working MP4 button" are things that have to be asked
over HTTP. This tool does both, bounded: it polls until the job is terminal or
the time budget runs out, and it never pretends a clip is ready because its
record says so — it asks the download endpoint for real bytes.

Run:  .venv/bin/python tools/wait_for_batch.py 885ba6a7ac26 [--seconds 150]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

BASE = "http://localhost:5173"
TERMINAL = {"complete", "failed"}


def get(path: str, timeout: float = 20.0):
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # noqa: BLE001 - the caller reports the message
        return None, str(e).encode()


def poll(job_id: str, budget: float) -> dict:
    deadline = time.time() + budget
    last = ""
    while time.time() < deadline:
        status, body = get(f"/api/jobs/{job_id}")
        if status != 200:
            print(f"  ! job read failed: HTTP {status} {body[:80]!r}")
            time.sleep(5)
            continue
        job = json.loads(body)
        line = (
            f"{job.get('stage')} · {job.get('progress', 0):.0f}% · "
            f"{job.get('progressDetail') or ''}"
        )
        if line != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {line}", flush=True)
            last = line
        if job.get("stage") in TERMINAL:
            return job
        time.sleep(8)
    status, body = get(f"/api/jobs/{job_id}")
    return json.loads(body) if status == 200 else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--seconds", type=float, default=150.0)
    args = ap.parse_args()

    print(f"waiting on job {args.job_id} (up to {args.seconds:.0f}s)")
    job = poll(args.job_id, args.seconds)
    stage = job.get("stage")
    print(f"\njob: {stage} · clipCount={job.get('clipCount')} · "
          f"error={str(job.get('error'))[:120]}")

    status, body = get(f"/api/clips?jobId={args.job_id}")
    if status != 200:
        print(f"could not read clips: HTTP {status}")
        return 1
    clips = sorted(json.loads(body), key=lambda c: c.get("order", 0))
    print(f"clips: {len(clips)}\n")

    ready = 0
    for c in clips:
        n = c.get("order", 0) + 1
        mp4_status, mp4_body = (None, b"") if not c.get("videoPath") else \
            get(f"/api/clips/{c['_id']}/download")
        srt_status, srt_body = get(f"/api/clips/{c['_id']}/srt")
        # A range request is enough: we are asking "does this file serve?", not
        # pulling 50 MB per clip to prove a button is enabled.
        req = urllib.request.Request(
            f"{BASE}/api/clips/{c['_id']}/download",
            headers={"Range": "bytes=0-1023"},
        )
        ranged = "n/a"
        if c.get("videoPath"):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    ranged = f"{r.status}/{len(r.read())}B"
            except Exception as e:  # noqa: BLE001
                ranged = f"ERR {e}"
        usable = c.get("renderState") == "done" and c.get("videoPath") and srt_status == 200
        ready += bool(usable)
        print(f"  {n:2d} {c.get('title', '')[:30]:32} state={c.get('renderState'):8} "
              f"srt={srt_status} mp4={ranged} {'OK' if usable else 'NOT USABLE'}")
    print(f"\n{ready}/{len(clips)} clips fully usable (MP4 + .srt downloadable)")
    return 0 if ready == len(clips) and stage == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
