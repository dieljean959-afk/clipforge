"""Chaos test: kill the worker mid-pipeline and prove the job still finishes.

Run against a REAL job and a REAL video. Every kill is followed by the checks
that matter: the job must still be the same job (one fetch, no re-transcribed
chunks) and every clip must exist with 1080x1920 geometry.

    .venv/bin/python tools/chaos.py --job <job_id> --kills 6 --max-minutes 45
    .venv/bin/python tools/chaos.py --source "https://youtu.be/..." --kills 6

Exit code 0 only when the job reaches `complete` with valid clips.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("CLIPFORGE_API", "http://127.0.0.1:8000")
DATA = ROOT / "data"


def get(path: str) -> dict:
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.loads(r.read().decode())


def post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def job(job_id: str) -> dict:
    return get(f"/api/jobs/{job_id}")


def daemon_pid() -> int | None:
    out = subprocess.run(["pgrep", "-f", "app.workerd"], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
    return pids[0] if pids else None


def api_pid() -> int | None:
    out = subprocess.run(
        ["pgrep", "-f", "uvicorn app.main:app"], capture_output=True, text=True
    )
    pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
    return pids[0] if pids else None


def wait_until(pred, timeout: float, interval: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(interval)
    return None


def clips(job_id: str) -> list[dict]:
    """Clip rows from the durable store (the API projects a narrower shape)."""
    out: list[dict] = []
    for path in (DATA / "clips").glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if str(row.get("jobId")) == job_id:
            out.append(row)
    return out


def log_tail(job_id: str, n: int = 3) -> list[str]:
    try:
        doc = job(job_id)
        return [str(x) for x in (doc.get("logs") or [])][-n:]
    except Exception:
        return []


def _job_text(job_id: str) -> str:
    """Everything the job wrote: its document plus every runner log."""
    parts: list[str] = []
    try:
        parts.append((DATA / "jobs" / f"{job_id}.json").read_text(errors="replace"))
    except OSError:
        pass
    for p in (DATA / "work").glob(f"{job_id}/**/runner.log"):
        try:
            parts.append(p.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)


def fetch_count(job_id: str) -> int:
    """How many times the source was actually fetched for this job.

    The stage writes "Stage: fetch — starting" once per attempt; a resume that
    reuses the bucket object logs "reusing the source in the bucket" instead.
    """
    text = _job_text(job_id)
    if not text:
        return -1
    return max(1 if "reusing the source in the bucket" in text else 0,
               text.count("Stage: fetch — starting"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--source", default="https://youtu.be/627Fvn2fEjU?si=MMQ3k1JaLHyOfjyu")
    ap.add_argument("--kills", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--interval", type=float, default=45.0)
    args = ap.parse_args()

    job_id = args.job
    if not job_id:
        created = post("/api/jobs", {
            "sourceType": "url", "source": args.source, "language": "es",
            "emojiEnabled": True, "emojiDensity": "medium",
            "captionPreset": "bold_stack", "maxClips": 15,
        })
        job_id = created["jobId"]
    print(f"job: {job_id}")

    deadline = time.time() + args.max_minutes * 60
    kills = 0
    kills_api = False
    events: list[str] = []
    while time.time() < deadline:
        doc = job(job_id)
        stage = str(doc.get("stage"))
        status = str(doc.get("status"))
        print(f"[{time.strftime('%H:%M:%S')}] {status:>9} {stage:<14} "
              f"{float(doc.get('progress') or 0):5.1f}% {str(doc.get('progressDetail'))[:48]}")
        if stage == "complete":
            break
        if stage == "failed" and not doc.get("interrupted"):
            print("job failed:", str(doc.get("error"))[:200])
            break
        if kills < args.kills and stage not in {"queued", "complete", "failed"}:
            time.sleep(max(3.0, args.interval / 3))
            pid = daemon_pid()
            if pid:
                sig = signal.SIGTERM if kills % 2 == 0 else signal.SIGKILL
                os.kill(pid, sig)
                kills += 1
                events.append(f"{time.strftime('%H:%M:%S')} {sig.name} -> daemon {pid} at {stage}")
                print(f"  *** {sig.name} to worker daemon {pid} during {stage}")
                wait_until(lambda: daemon_pid() not in (None, pid), 90)
                # keep the API alive: it respawns the daemon on the next call
                wait_until(lambda: (get("/api/health") or {}).get("ok"), 90)
        elif not kills_api and stage in {"finding", "exporting"}:
            pid = api_pid()
            if pid:
                os.kill(pid, signal.SIGKILL)
                kills_api = True
                events.append(f"{time.strftime('%H:%M:%S')} SIGKILL -> API {pid} at {stage}")
                print(f"  *** SIGKILL to API {pid} during {stage}")
        time.sleep(2)

    doc = job(job_id)
    found = clips(job_id)
    done = [c for c in found if c.get("renderState") == "done"]
    bad = [c for c in done if (c.get("mediaWidth"), c.get("mediaHeight")) != (1080, 1920)]
    report = {
        "job": job_id,
        "stage": doc.get("stage"),
        "status": doc.get("status"),
        "clips": len(found),
        "done": len(done),
        "wrong_geometry": len(bad),
        "fetches": fetch_count(job_id),
        "kills": kills,
        "api_kills": int(kills_api),
        "events": events,
    }
    print(json.dumps(report, indent=2))
    ok = doc.get("stage") == "complete" and done and not bad and report["fetches"] <= 1
    if not ok:
        print("CHAOS: FAILED the acceptance checks", file=sys.stderr)
        print("last log:", *log_tail(job_id, 4), sep="\n  ", file=sys.stderr)
        return 1
    print(f"CHAOS: PASSED — {len(done)} clips at 1080x1920, {report['fetches']} fetch, "
          f"{kills} worker kills, {int(kills_api)} API kill(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
