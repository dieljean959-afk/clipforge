"""Soak proof for LOCKED_SPEC §4: five jobs back to back, one worker kill, one API restart.

Terminal commands here are short-lived, so this driver is RESUMABLE: it writes
its state to outputs/soak-state.json and each invocation watches for a bounded
number of seconds before returning. Run it repeatedly until it reports a final
result.

  python3 tools/soak_locked.py --fresh      # upload 5 jobs, start watching
  python3 tools/soak_locked.py              # continue watching (drills fire here)

Drills, exactly the two the user hit:
  * job 3 in flight  -> os.kill(pid, 9) on the `python -m app.runner` child
  * job 4 in flight  -> touch app/worker.py, the host restarts the API

Per job it prints status, duration, observed peak RSS (VmHWM) of the runner,
and whether the clip file exists on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8000"
SRC_DIR = ROOT / "outputs" / "soak-src"
SPEECH = ROOT / "tools" / "fixtures" / "speech-jfk-11s.wav"
STATE = ROOT / "outputs" / "soak-state.json"


def api(path: str) -> dict:
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.loads(r.read().decode())


def make_sources(count: int) -> list[Path]:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(1, count + 1):
        p = SRC_DIR / f"soaki{i}.mp4"
        if not p.exists():
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=10",
                 "-i", str(SPEECH), "-t", "10", "-c:v", "libx264",
                 "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-shortest", str(p)],
                check=True,
            )
        out.append(p)
    return out


def upload(path: Path) -> str:
    boundary = "----soakboundary"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; filename="' + path.name.encode() + b'"\r\n',
        b"Content-Type: video/mp4\r\n\r\n",
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        API + "/api/jobs/upload?maxClips=1", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())["jobId"]


def runner_pids() -> list[int]:
    out = subprocess.run(["pgrep", "-f", "app.runner"], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def hwm(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--watch", type=float, default=95.0)
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()

    state: dict
    if args.fresh or not STATE.exists():
        ids = [upload(p) for p in make_sources(args.jobs)]
        state = {"ids": ids, "began": {}, "ended": {}, "peak": {},
                 "killed": False, "restarted": False, "t0": time.time()}
        save(state)
        print(f"uploaded {len(ids)} jobs: {ids}", flush=True)
    else:
        state = json.loads(STATE.read_text())

    deadline = time.time() + args.watch
    while time.time() < deadline:
        live = runner_pids()
        for pid in live:
            pass
        for idx, jid in enumerate(state["ids"], start=1):
            try:
                job = api(f"/api/jobs/{jid}")
            except Exception:
                time.sleep(2)
                continue
            stage = str(job.get("stage") or "")
            status = str(job.get("status") or "")
            inflight = stage not in ("queued", "waiting-upload", "done", "failed",
                                     "complete", "cancelled", "expired")
            if inflight and jid not in state["began"]:
                state["began"][jid] = time.time()
            for pid in live:
                state["peak"][jid] = max(state["peak"].get(jid, 0), hwm(pid))
            if idx == 3 and inflight and live and not state["killed"]:
                victim = live[0]
                os.kill(victim, 9)
                state["killed"] = True
                state["killed_at"] = time.time()
                print(f"DRILL: kill -9 runner pid {victim} during job 3", flush=True)
            # Only bounce the API once the kill has been handled, so the two
            # drills stay separate events instead of one restart.
            kill_settled = state["killed"] and time.time() - state.get("killed_at", 0) > 25
            if idx == 4 and inflight and kill_settled and not state["restarted"]:
                (ROOT / "app" / "worker.py").touch()
                state["restarted"] = True
                print("DRILL: touched app/worker.py during job 4 (host restarts API)", flush=True)
            if status in ("done", "failed", "error", "complete") and jid not in state["ended"]:
                state["ended"][jid] = time.time()
        save(state)
        pending = [j for j in state["ids"] if j not in state["ended"]]
        print(f"pending={len(pending)} killed={state['killed']} restarted={state['restarted']}",
              flush=True)
        if not pending:
            break
        time.sleep(5)

    state["done"] = all(j in state["ended"] for j in state["ids"])
    save(state)
    if not state["done"]:
        print("not finished yet — run this command again to continue", flush=True)
        return 2

    print("\n=== soak result ===", flush=True)
    ok = 0
    for idx, jid in enumerate(state["ids"], start=1):
        job = api(f"/api/jobs/{jid}")
        clips = job.get("clips") or []
        served = [
            c for c in clips
            if str(c.get("outputPath") or c.get("path") or "")
            and (ROOT / str(c.get("outputPath") or c.get("path")).lstrip("/")).exists()
        ]
        dur = state["ended"].get(jid, 0) - state["began"].get(jid, state["ended"].get(jid, 0))
        good = job.get("status") in ("done", "complete") and served
        ok += 1 if good else 0
        print(f"job{idx} {jid} status={job.get('status')} stage={job.get('stage')} "
              f"attempts={job.get('attempts')} duration={dur:.0f}s "
              f"peak_rss={state['peak'].get(jid, 0)}kB clips={len(served)}/{len(clips)} "
              f"reason={str(job.get('error') or '')[:80]}", flush=True)
    print(f"\n{ok}/{len(state['ids'])} jobs finished with a clip, no manual retry", flush=True)
    return 0 if ok == len(state["ids"]) else 1


if __name__ == "__main__":
    sys.exit(main())
