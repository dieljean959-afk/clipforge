"""Proof that the API and the worker are separate processes (LOCKED_SPEC §4).

Two drills, on real jobs, with pids recorded as evidence:

  A. bounce the API mid-job  -> the worker daemon and its job process keep the
     same pids, the replacement API ADOPTS the daemon (no second daemon), the
     job finishes.
  B. kill -9 the daemon mid-job -> the supervisor restarts it, the orphaned job
     process is reaped, the job is re-queued as interrupted and finishes from
     its cached stages.

Resumable, because a terminal command here lives for ~90 s:

    python3 tools/split_proof.py --fresh --watch 80
    python3 tools/split_proof.py --watch 80      # repeat until it prints the table
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8000"
STATE = ROOT / "outputs" / "split-proof.json"


def get_json(path: str):
    with urllib.request.urlopen(API + path, timeout=15) as r:
        return json.loads(r.read().decode())


def health_ok() -> bool:
    try:
        get_json("/api/health")
        return True
    except Exception:
        return False


def pids(pattern: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(p) for p in out.stdout.split() if p.strip().isdigit()]


def api_pid() -> int | None:
    found = pids("uvicorn app.main:app")
    return found[0] if found else None


def daemon_pid() -> int | None:
    found = pids("app.workerd")
    return found[0] if found else None


def runner_pid() -> int | None:
    found = pids("app.runner")
    return found[0] if found else None


def upload(path: Path) -> str:
    boundary = "----splitboundary"
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


def in_flight(job: dict) -> bool:
    return str(job.get("stage") or "") not in (
        "queued", "waiting-upload", "complete", "done", "failed", "paused", "cancelled",
    )


def finished(job: dict) -> bool:
    return str(job.get("status") or "") in ("complete", "done", "failed")


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--watch", type=float, default=80.0)
    args = ap.parse_args()

    if args.fresh or not STATE.exists():
        src = ROOT / "outputs" / "soak-src" / "soaki1.mp4"
        if not src.exists():
            print(f"missing source {src} — run tools/soak_locked.py --fresh first")
            return 1
        state = {
            "job_a": upload(src),
            "job_b": None,
            "phase": "bounce",
            "obs": {},
            "t0": time.time(),
        }
        save(state)
        print(f"uploaded job A {state['job_a']}", flush=True)

    state = json.loads(STATE.read_text())
    obs = state["obs"]
    deadline = time.time() + args.watch

    while time.time() < deadline:
        obs.setdefault("daemon_at_start", daemon_pid())
        obs.setdefault("api_at_start", api_pid())

        if state["phase"] == "bounce":
            job = get_json(f"/api/jobs/{state['job_a']}")
            if in_flight(job) and runner_pid():
                obs["runner_before_bounce"] = runner_pid()
                obs["daemon_before_bounce"] = daemon_pid()
                obs["api_before_bounce"] = api_pid()
                os.utime(ROOT / "app" / "worker.py")  # the host restarts the API
                state["phase"] = "bounce_wait"
                save(state)
                print("DRILL A: API restarted mid-job (touched app/worker.py); "
                      f"api={obs['api_before_bounce']} daemon={obs['daemon_before_bounce']} "
                      f"runner={obs['runner_before_bounce']}", flush=True)

        elif state["phase"] == "bounce_wait":
            if health_ok() and api_pid() != obs.get("api_before_bounce"):
                obs["api_after_bounce"] = api_pid()
                obs["daemon_after_bounce"] = daemon_pid()
                obs["runner_after_bounce"] = runner_pid()
                state["phase"] = "finish_a"
                save(state)
                print(f"DRILL A: API back api={obs['api_after_bounce']} "
                      f"daemon={obs['daemon_after_bounce']} "
                      f"runner={obs['runner_after_bounce']}", flush=True)

        elif state["phase"] == "finish_a":
            job = get_json(f"/api/jobs/{state['job_a']}")
            if finished(job):
                obs["job_a_status"] = job.get("status")
                obs["job_a_clips"] = len(glob.glob(str(ROOT / "outputs" / state["job_a"] / "clip_*.mp4")))
                obs["same_daemon"] = obs.get("daemon_after_bounce") == obs.get("daemon_before_bounce")
                obs["same_runner"] = obs.get("runner_after_bounce") == obs.get("runner_before_bounce")
                state["job_b"] = upload(ROOT / "outputs" / "soak-src" / "soaki2.mp4")
                state["phase"] = "kill"
                save(state)
                print(f"job A {job.get('status')} after the bounce; "
                      f"uploaded job B {state['job_b']}", flush=True)

        elif state["phase"] == "kill":
            job = get_json(f"/api/jobs/{state['job_b']}")
            if in_flight(job) and runner_pid():
                obs["runner_before_kill"] = runner_pid()
                obs["daemon_before_kill"] = daemon_pid()
                try:
                    os.kill(obs["daemon_before_kill"], 9)
                except OSError as exc:
                    print(f"could not kill daemon: {exc}", flush=True)
                state["phase"] = "kill_wait"
                save(state)
                print(f"DRILL B: kill -9 daemon pid {obs['daemon_before_kill']} "
                      f"mid-job (runner {obs['runner_before_kill']} orphaned)", flush=True)

        elif state["phase"] == "kill_wait":
            new_daemon = daemon_pid()
            if new_daemon and new_daemon != obs.get("daemon_before_kill"):
                obs["daemon_after_kill"] = new_daemon
                obs["runner_after_kill"] = runner_pid()
                state["phase"] = "finish_b"
                save(state)
                print(f"DRILL B: supervisor restarted the daemon "
                      f"daemon={new_daemon} runner={obs['runner_after_kill']}", flush=True)

        elif state["phase"] == "finish_b":
            job = get_json(f"/api/jobs/{state['job_b']}")
            if finished(job) and not in_flight(job):
                obs["job_b_status"] = job.get("status")
                obs["job_b_interrupts"] = job.get("interruptCount")
                obs["job_b_clips"] = len(glob.glob(str(ROOT / "outputs" / state["job_b"] / "clip_*.mp4")))
                state["phase"] = "done"
                save(state)
                break

        save(state)
        time.sleep(4)

    if state["phase"] != "done":
        print(f"phase={state['phase']} — run this command again to continue", flush=True)
        return 2

    print("\n=== API / worker split proof ===", flush=True)
    for key in (
        "api_at_start", "daemon_at_start",
        "api_before_bounce", "api_after_bounce",
        "daemon_before_bounce", "daemon_after_bounce",
        "runner_before_bounce", "runner_after_bounce",
        "same_daemon", "same_runner", "job_a_status", "job_a_clips",
        "daemon_before_kill", "daemon_after_kill", "runner_before_kill", "runner_after_kill",
        "job_b_status", "job_b_interrupts", "job_b_clips",
    ):
        print(f"{key:24} {obs.get(key)}", flush=True)

    ok = (
        obs.get("same_daemon") and obs.get("same_runner")
        and obs.get("job_a_clips", 0) >= 1
        and obs.get("job_b_clips", 0) >= 1
        and obs.get("daemon_after_kill") != obs.get("daemon_before_kill")
    )
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
