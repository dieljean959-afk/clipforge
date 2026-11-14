#!/usr/bin/env python3
"""Host-only real-media run: two 10-clip batches + repair of an existing batch.

The sandbox cannot pass YouTube's bot check and has no working DeepSeek
balance, so run this on the machine that has your cookies and keys configured
in the app (never paste a key into this script or its output).

    YT_DLP_COOKIES_FROM_BROWSER=chrome \
    python tools/host_real_batches.py \
      --url 'https://youtu.be/627Fvn2fEjU' \
      --url 'https://youtu.be/-UZJGTjG3dc' \
      --repair-job c84094fb317f46c2afa8a63b7eb0bc24

It prints requested/ready/failed counts, every clip's audioSync result, and
retries each failed clip of --repair-job through the real Retry route (which
re-fetches an expired bucket source from the original URL).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

BASE = os.getenv("CLIPFORGE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"detail": raw[:500]}


def create(url: str, clips: int) -> str:
    status, body = call(
        "POST",
        "/api/jobs",
        {"sourceType": "url", "source": url, "maxClips": clips},
    )
    job_id = body.get("jobId") or body.get("id")
    if status != 200 or not job_id:
        raise SystemExit(f"create failed HTTP {status}: {body.get('detail') or body}")
    return str(job_id)


def report(job_id: str) -> dict:
    status, job = call("GET", f"/api/jobs/{job_id}")
    if status != 200:
        raise SystemExit(f"job read failed HTTP {status}: {job.get('detail') or job}")
    print(
        f"\njob {job_id}: stage={job.get('stage')} "
        f"requested={job.get('requestedClips', job.get('clipTarget'))} "
        f"ready={job.get('readyClips', 0)} failed={job.get('failedClips', 0)} "
        f"candidates={job.get('candidatesDetected', '?')}"
    )
    if job.get("unmetReason"):
        print("  unmet:", job["unmetReason"])
    for clip in job.get("clips") or []:
        sync = clip.get("audioSync") or {}
        place = sync.get("placement") or {}
        speech = sync.get("speech") or {}
        print(
            f"  order {clip.get('order')}: {clip.get('renderState')} "
            f"lag={place.get('lagMs', '?')}ms corr={place.get('correlation', '?')} "
            f"speech={speech.get('ok', '?')} "
            f"reason={str(clip.get('mediaError') or clip.get('error') or '')[:120]}"
        )
    return job


def wait(job_id: str, poll: int = 30) -> dict:
    while True:
        job = report(job_id)
        if job.get("isTerminal") or job.get("stage") in {"done", "failed", "ready", "complete", "error"}:
            return job
        time.sleep(poll)


def repair(job_id: str) -> None:
    print(f"\nrepairing failed clips of {job_id} …")
    status, job = call("GET", f"/api/jobs/{job_id}")
    if status != 200:
        print("  cannot read job:", job.get("detail") or job)
        return
    for clip in job.get("clips") or []:
        if clip.get("renderState") != "failed":
            continue
        cid = clip.get("_id") or clip.get("id")
        status, result = call("POST", f"/api/clips/{cid}/retry-render")
        print(f"  order {clip.get('order')} retry HTTP {status}: {str(result)[:240]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", default=[], help="real source URL")
    parser.add_argument("--clips", type=int, default=10)
    parser.add_argument("--repair-job", default="")
    parser.add_argument("--poll", type=int, default=30)
    args = parser.parse_args()
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("warning: DEEPSEEK_API_KEY is not in this shell; the app may still load it from its own env")
    if not args.url:
        args.url = [
            "https://youtu.be/627Fvn2fEjU",
            "https://youtu.be/-UZJGTjG3dc",
        ]
    for url in args.url:
        job_id = create(url, args.clips)
        print(f"created {job_id} for {url}")
        wait(job_id, args.poll)
    if args.repair_job:
        repair(args.repair_job)


if __name__ == "__main__":
    main()
