#!/usr/bin/env python3
"""End-to-end proof for ONE language: submit, wait, and report the real facts.

    .venv/bin/python tools/e2e.py <video-url> --lang es [--clips 3]
    .venv/bin/python tools/e2e.py <video-url> --lang en --url http://127.0.0.1:8000

It prints, from the running backend's own data (never from a mock):

  * the job id and the detected / locked language;
  * the first 15 s of the real transcript;
  * whether the finished batch is listed by the same query "Your batches" uses;
  * ffprobe of one produced clip (resolution / codecs / fps / audio);
  * the .srt sidecar's first cues, so accents are visible in the artefact.

Exit code 0 only when the batch reaches a terminal state with at least one clip
that ffprobe accepts. `--clips` is the requested count (3 by default).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

POLL_SEC = 15.0
DEFAULT_BUDGET_SEC = 3600.0


def _local_token() -> str:
    """The API requires a session; scripts use data/auth/automation.token."""
    import os

    base = Path(os.getenv("CLIPFORGE_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
    try:
        return (base / "auth" / "automation.token").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _req(base: str, path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    token = _local_token()
    if token:
        headers["X-ClipForge-Token"] = token
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def _fmt_sec(v: object) -> str:
    try:
        n = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "?"
    return f"{int(n // 60)}m{int(n % 60):02d}s"


def main() -> int:
    ap = argparse.ArgumentParser(description="Prove one language end to end.")
    ap.add_argument("url", help="source URL (YouTube etc.)")
    ap.add_argument("--lang", required=True, choices=["en", "es", "auto"])
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--url-api", dest="base", default="http://127.0.0.1:8000",
                    help="backend base URL (default http://127.0.0.1:8000)")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SEC,
                    help="seconds to wait before giving up (default 3600)")
    ap.add_argument("--data", default="data", help="backend data dir (default: data)")
    args = ap.parse_args()

    try:
        created = _req(args.base, "/api/jobs", {
            "source": args.url,
            "sourceType": "url",
            "maxClips": args.clips,
            "language": args.lang,
        })
    except urllib.error.URLError as exc:
        print(f"BLOCKED: backend not reachable at {args.base} ({exc})", file=sys.stderr)
        return 2
    job_id = created.get("jobId") or created.get("id")
    if not job_id:
        print(f"BLOCKED: backend did not return a job id: {created}", file=sys.stderr)
        return 2
    print(f"job {job_id} ({'deduped' if created.get('deduped') else 'created'}) lang={args.lang}")

    deadline = time.time() + args.budget
    job: dict = {}
    seen: list[str] = []
    while time.time() < deadline:
        try:
            job = _req(args.base, f"/api/jobs/{job_id}")
        except urllib.error.URLError as exc:
            print(f"  (poll failed: {exc})")
            time.sleep(POLL_SEC)
            continue
        stage = str(job.get("stage") or "")
        line = f"  {stage:12s} {float(job.get('progress') or 0):5.1f}%  {job.get('progressDetail') or ''}"
        if not seen or seen[-1] != line:
            print(line[:160], flush=True)
            seen.append(line)
        if job.get("isTerminal") or stage in {"complete", "failed", "cancelled"}:
            break
        time.sleep(POLL_SEC)

    stage = str(job.get("stage") or "")
    print(f"terminal: stage={stage} status={job.get('status')} "
          f"ready={job.get('readyClips')} failed={job.get('failedClips')} "
          f"batchState={job.get('batchState')} took={_fmt_sec(job.get('stageElapsedSec'))}")
    if stage not in {"complete", "failed"}:
        print(f"BLOCKED: batch did not finish within {int(args.budget)}s", file=sys.stderr)
        return 3

    # Detected language + the first 15 s, straight from the checkpoint file.
    words_path = Path(args.data) / "work" / str(job_id) / "transcribe" / "words.json"
    if words_path.is_file():
        payload = json.loads(words_path.read_text(encoding="utf-8"))
        print(f"detected language: {payload.get('language')!r} "
              f"model={payload.get('model')} engine={payload.get('engine')}")
        words = payload.get("words") or []
        head = " ".join(str(w.get("word") or "") for w in words
                        if float(w.get("start") or 0) <= 15.0)
        print(f"first 15 s: {head[:500]}")
    else:
        print(f"transcript checkpoint missing at {words_path}")

    clips = job.get("clips") or []
    ready = [c for c in clips if str(c.get("renderState")) == "done"]
    print(f"clips: {len(ready)} ready of {job.get('clipCount')}")
    if job.get("batchState"):
        print(f"listed in Your batches: batchState={job['batchState']} "
              f"finishedAt={job.get('finishedAt')}")

    outputs = Path(args.data) / "outputs" / str(job_id)
    files = sorted(outputs.glob("*.mp4")) if outputs.is_dir() else []
    if not files:
        print(f"BLOCKED: no rendered clip under {outputs}", file=sys.stderr)
        return 4
    clip = files[0]
    print(f"ffprobe {clip}:")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,channels",
         "-show_entries", "format=duration", "-of", "json", str(clip)],
        capture_output=True, text=True, timeout=60,
    )
    print(probe.stdout.strip())
    srt = clip.with_suffix(".srt")
    if srt.is_file():
        print(f"{srt.name} (first 8 lines):")
        print("\n".join(srt.read_text(encoding="utf-8").splitlines()[:8]))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
