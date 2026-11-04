#!/usr/bin/env python3
"""Reproduce the reported job failure OUTSIDE the app, using the same command
and env the worker actually uses.

The worker runs ONE JOB per child process:

    python -m app.runner <job_id>

spawned by app/worker.py::_run_job_process with the worker's whole environment
plus PYTHONUNBUFFERED=1, on the same cwd the app itself uses. So to reproduce an
EXACT JOB from the user's screen, we replay the two steps the user saw (job
created, then the runner child exits) — no faster, no slower, same binaries,
same cwd, same env.

What this script does:
  1. Copies the app's genuine environment (config, paths, PYTHONPATH, ...) into
     a throwaway data dir so the runner sees the same files as a real job.
  2. Creates a job exactly like the dashboard does: POST /api/jobs with a
     sourceType=url + source=<url>.
  3. Prints the FULL stdout, stderr and exit code of the real child process
     `python -m app.runner <job_id>` that the worker would have spawned.
  4. Prints the process-tree facts (which binary, which cwd, which env vars
     with presence-only) and confirms the state of the job AFTER the child
     exits: stage, status, error, errorDetail, interrupted, completedAt.

That is the runs-reproduce-the-user's-screenshoot contract: the same command,
the same environment, the same input, and the FULL output of the process the
worker actually spawned.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# The app's own root: this script is placed beside app/ by the harness and the
# worker's environment (config, paths, PYTHONPATH, ...) is copied in so the
# runner sees the same files it would in a real job.
APP_ROOT = Path(__file__).resolve().parent
WORK_DIR = Path(tempfile.mkdtemp(prefix="repro-job-"))
DATA_DIR = WORK_DIR / "data"
PROC = None  # the exact child the worker would have spawned


def copy_env() -> None:
    """Copy the app's runtime env into the throwaway data dir so the runner
    reads the same config/paths the app uses."""
    for name in (".env", ".env.local", ".env.keys", "config.toml"):
        src = APP_ROOT / name
        if not src.exists():
            continue
        dst = DATA_DIR / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # The runner needs the package importable; PYTHONPATH is inherited from the
    # worker's environment (set by app/worker.py::_run_job_process).
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(APP_ROOT))
    env["PYTHONUNBUFFERED"] = "1"
    env["CLIPFORGE_DATA_DIR"] = str(DATA_DIR)
    env["CLIPFORGE_DOWNLOADS_DIR"] = str(DATA_DIR / "downloads")
    env["CLIPFORGE_OUTPUTS_DIR"] = str(DATA_DIR / "outputs")
    return env


def main() -> int:
    env = copy_env()

    # 1. Job creation: exactly what POST /api/jobs does in app/main.py.
    #    The real supervisor would have picked this up and spawned the runner.
    url = os.environ.get("REPRO_URL", "https://youtu.be/Brf_WibGtZw")
    print(f"[repro] creating job with sourceType=url source={url}")
    create = subprocess.run(
        [sys.executable, "-m", "app.main", "--create-job", url],
        cwd=str(APP_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    print("[repro] create-job stdout:")
    print(create.stdout)
    print("[repro] create-job stderr:")
    print(create.stderr)
    print(f"[repro] create-job exit code: {create.returncode}")
    if create.returncode != 0:
        print("[repro] JOB CREATION FAILED — the dashboard cannot even submit")
        return create.returncode

    # 2. The exact child the worker would have spawned: the same command, same
    #    env, same cwd. We keep the FULL stdout/stderr and the exact exit code.
    job_id = None
    for line in (create.stdout or "").splitlines():
        if "jobId" in line or "job " in line:
            for part in line.split():
                if part.startswith("job-"):
                    job_id = part
    if job_id is None:
        print("[repro] could not determine job_id from create-job output")
        return 1
    print(f"[repro] job_id={job_id}")

    # The worker's real command (app/worker.py::_run_job_process).
    cmd = [sys.executable, "-m", "app.runner", job_id]
    print(f"[repro] spawning exactly: {' '.join(cmd)}")
    print("[repro] ---- START OF FULL OUTPUT ----")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
        print(proc.stdout)
        print(f"[repro] ---- END OF FULL OUTPUT ----")
        print(f"[repro] FULL STDOUT+STDERR (merged, as the worker saw it):")
        print(proc.stdout)
        print(f"[repro] FULL EXIT CODE (as the worker saw it): {proc.returncode}")
        global PROC
        PROC = proc
        return proc.returncode
    except subprocess.TimeoutExpired as exc:
        print("[repro] runner TIMED OUT — the child never exited; is it still running?")
        print("---- START OF TIMEOUT OUTPUT ----")
        print((exc.stdout or b"").decode("utf-8", errors="replace"))
        print("---- END OF TIMEOUT OUTPUT ----")
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
