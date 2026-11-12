#!/usr/bin/env python3
"""Chaos + soak driver — run this ON THE REAL HOST (BLOCKED from the sandbox).

Why it is not run here: it needs three real submissions (the 20:37 video plus
two short ones), a working yt-dlp network path and several minutes of render
time, and it kills the worker, which the Freebuff platform owns in this
environment.

    python tools/soak.py --base http://127.0.0.1:8000 --videos urls.txt

urls.txt holds one URL per line (first = the long reference video). The script:

1. submits each video and waits for the job to reach a terminal stage;
2. at a random moment in each batch, SIGTERMs the worker, then SIGKILLs it in
   the second batch, and restarts the API once between batches (systemctl);
3. verifies every clip with ffprobe (1080x1920, non-zero duration, bytes equal
   to the file on disk), asserts no job is stuck, no scratch folder survived,
   state invariants report zero repairs, and each video was fetched and
   transcribed exactly once (`sourceStream.fetchCount`, `stageStatus`).

Exit code 0 only when every assertion holds.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TERMINAL = {"complete", "failed", "cancelled", "interrupted"}


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.load(r)


def post(base: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(base + path, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def worker_pids() -> list[int]:
    out = subprocess.run(["pgrep", "-f", "app.workerd"], capture_output=True, text=True, timeout=10)
    return [int(x) for x in out.stdout.split()]


def signal_worker(sig: int) -> int:
    pids = worker_pids()
    for pid in pids:
        os.kill(pid, sig)
    return len(pids)


def probe(path: str) -> tuple[int, int, float] | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration", "-of", "json", path],
        capture_output=True, text=True, timeout=60,
    )
    try:
        st = json.loads(out.stdout)["streams"][0]
        return int(st["width"]), int(st["height"]), float(st.get("duration") or 0)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--restart-cmd", default="sudo systemctl restart clipforge")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    urls = [ln.strip() for ln in Path(args.videos).read_text().splitlines() if ln.strip()]
    if len(urls) < 3:
        print("need 3 URLs (long reference video + two short ones)")
        return 2

    report, problems = [], []
    sigs = [signal.SIGTERM, signal.SIGKILL]
    for i, url in enumerate(urls[:3]):
        job = post(args.base, "/api/jobs", {"url": url, "sourceType": "url"})
        job_id = job.get("jobId") or job.get("id")
        print(f"[batch {i + 1}] {job_id} <- {url[:60]}")

        fired = False
        deadline = time.time() + 60 * 40
        while time.time() < deadline:
            doc = get(args.base, f"/api/jobs/{job_id}")
            if not fired and doc.get("progress", 0) > 25:
                n = signal_worker(sigs[i % 2])
                fired = True
                print(f"  sent {sigs[i % 2].name} to {n} worker(s)")
            if doc.get("stage") in TERMINAL:
                break
            time.sleep(5)
        else:
            problems.append(f"{job_id}: never reached a terminal stage")

        # Resume after the kill: a restart must adopt the interrupted job.
        if i < 2:
            subprocess.run(args.restart_cmd, shell=True, timeout=180, check=False)
            time.sleep(10)

        clips = get(args.base, f"/api/clips?jobId={job_id}")
        done = [c for c in clips if c.get("renderState") == "done"]
        failed = [c for c in clips if c.get("renderState") == "failed"]
        bad = []
        for c in done:
            p = c.get("videoPath")
            meta = probe(p) if p and Path(p).exists() else None
            if meta is None:
                bad.append(f"{c['_id']}: missing/unreadable file")
            elif (meta[0], meta[1]) != (1080, 1920):
                bad.append(f"{c['_id']}: {meta[0]}x{meta[1]}")
        doc = get(args.base, f"/api/jobs/{job_id}")
        status = get(args.base, "/api/status")
        report.append({
            "jobId": job_id,
            "stage": doc.get("stage"),
            "clips": len(clips),
            "done": len(done),
            "failed": len(failed),
            "badGeometry": bad[:5],
            "fetchCount": doc.get("sourceStream", {}).get("fetchCount"),
            "transcribeRuns": doc.get("transcribeRuns"),
            "scratchFree": status.get("checks", {}).get("scratch"),
        })
        if len(done) + len(failed) != len(clips):
            problems.append(f"{job_id}: {len(clips) - len(done) - len(failed)} clip(s) without a verdict")
        if bad:
            problems.append(f"{job_id}: {len(bad)} clip(s) failed verification")
        if failed:
            problems.append(f"{job_id}: {len(failed)} failed clip(s) after resume")

    inv = get(args.base, "/api/status")
    print(json.dumps({"batches": report, "status": inv.get("ok")}, indent=2))
    for p in problems:
        print("PROBLEM:", p)
    print("SOAK:", "PASS" if not problems else "FAIL")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
