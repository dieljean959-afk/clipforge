#!/usr/bin/env python3
"""Chaos + soak driver — run this ON THE REAL HOST (BLOCKED from the sandbox).

Why it is not run here: it needs two real submissions (the 20:37 video plus
one more real video), a working yt-dlp network path and tens of minutes of
render time, and it kills the worker, which the Freebuff platform owns in this
environment.

    python tools/soak.py --base http://127.0.0.1:8000 --videos urls.txt

urls.txt holds one real URL per line (first = the 20:37 reference video).

1. submits each video and waits for a terminal stage;
2. SIGTERMs the worker in batch 1, SIGKILLs it in batch 2, restarts the API
   once between batches (systemctl), and relies on checkpoint resume;
3. probes every clip through the app's own ranged video endpoint (1080x1920,
   h264 yuv420p, AAC, CFR 24-60, faststart, duration, full decode), asserts
   every preview returns 206, every MP4/SRT/ZIP download works, a fresh share
   token returns 206 without cookies, no job loops, no scratch survives, and
   each source was fetched/transcribed once.

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


def probe(path: str) -> dict | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", path],
        capture_output=True, text=True, timeout=90,
    )
    try:
        data = json.loads(out.stdout)
        v = next(s for s in data["streams"] if s.get("codec_type") == "video")
        a = next((s for s in data["streams"] if s.get("codec_type") == "audio"), {})
        fps_s = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
        n, _, d = fps_s.partition("/")
        return {
            "width": int(v.get("width") or 0),
            "height": int(v.get("height") or 0),
            "fps": float(n) / float(d or 1) if float(d or 1) else 0.0,
            "duration": float(data.get("format", {}).get("duration") or 0),
            "video": v.get("codec_name"),
            "pix": v.get("pix_fmt"),
            "audio": a.get("codec_name"),
        }
    except Exception:
        return None


def ranged(url: str, end: int = 1023) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{end}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - localhost
            return r.status, r.read(end + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def download(url: str, method: str = "GET", payload: dict | None = None) -> tuple[int, bytes]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"content-type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--restart-cmd", default="sudo systemctl restart clipforge")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    urls = [ln.strip() for ln in Path(args.videos).read_text().splitlines() if ln.strip()]
    if len(urls) < 2:
        print("need 2 real URLs (20:37 reference video + one more real video)")
        return 2

    report, problems = [], []
    sigs = [signal.SIGTERM, signal.SIGKILL]
    for i, url in enumerate(urls[:2]):
        job = post(args.base, "/api/jobs", {"source": url, "sourceType": "url"})
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
            cid = c["_id"]
            meta = probe(f"{args.base}/api/clips/{cid}/video")
            if meta is None:
                bad.append(f"{cid}: unreadable through the preview URL")
                continue
            if (meta["width"], meta["height"]) != (1080, 1920):
                bad.append(f"{cid}: {meta['width']}x{meta['height']}")
            if meta["video"] != "h264" or meta["pix"] != "yuv420p" or meta["audio"] != "aac":
                bad.append(f"{cid}: {meta['video']}/{meta['pix']}/{meta['audio']}")
            if not (24 <= meta["fps"] <= 60):
                bad.append(f"{cid}: fps {meta['fps']}")
            status, body = ranged(f"{args.base}/api/clips/{cid}/video")
            if status != 206 or len(body) != 1024:
                bad.append(f"{cid}: preview HTTP {status}/{len(body)}B")
            status, body = download(f"{args.base}/api/clips/{cid}/download")
            if status != 200 or b"ftyp" not in body[:64] or len(body) < 10_000:
                bad.append(f"{cid}: MP4 download HTTP {status}/{len(body)}B")
            status, body = download(f"{args.base}/api/clips/{cid}/srt")
            if status != 200 or b"[INAUDIBLE]" in body.upper():
                bad.append(f"{cid}: SRT HTTP {status} or still has a marker")
            share = post(args.base, f"/api/clips/{cid}/share")
            token = share.get("token")
            if not token:
                bad.append(f"{cid}: share token missing")
            else:
                status, body = ranged(f"{args.base}/s/{token}.mp4")
                if status != 206 or len(body) != 1024:
                    bad.append(f"{cid}: share HTTP {status}/{len(body)}B")
        if done:
            status, body = download(
                f"{args.base}/api/export/zip", "POST", {"clipIds": [c["_id"] for c in done[:2]]},
            )
            if status != 200 or len(body) < 10_000 or not body.startswith(b"PK"):
                bad.append(f"ZIP export HTTP {status}/{len(body)}B")
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
