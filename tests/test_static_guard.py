"""Permanent guard against the crash class that took the API down.

The outage was not a memory problem: `app/main.py` imported fine and then
raised `NameError: name 'worker_mod' is not defined` inside the module-level
`_boot()` call, so uvicorn exited before it ever bound the port and the browser
saw an empty 500 from the proxy.

Three separate `NameError`s were live in this tree at the same time —
`worker_mod` (boot), `on_log` (clip detection), `video_svc` (thumbnails) — and
each one was invisible until the exact code path ran. `ruff`'s F821 rule finds
that whole class statically, so it runs here rather than in someone's browser.

Two checks:
  * F821 over `app/` — no undefined name may be introduced again.
  * `import app.main` in a subprocess — the module-level boot path must import.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _ruff() -> str | None:
    exe = shutil.which("ruff") or shutil.which("ruff", path=str(ROOT / ".venv" / "bin"))
    return exe


def test_no_undefined_names_in_app() -> None:
    """Every name referenced in app/ must resolve.

    Skipped (loudly, never silently passed) when ruff is unavailable, because a
    missing linter must not look like a green build.
    """
    ruff = _ruff()
    if ruff is None:
        pytest.skip("ruff not installed — undefined-name check could not run")
    result = subprocess.run(
        [ruff, "check", str(ROOT / "app"), "--select", "F821", "--output-format", "concise"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "Undefined names found in app/ — each one is a NameError waiting for "
        "its code path to run:\n" + (result.stdout or result.stderr)
    )


def test_app_main_imports() -> None:
    """`import app.main` must succeed — this is the boot path uvicorn uses.

    Run in a subprocess so a failure here cannot poison this test process's
    module state (and so the assertion is about a cold import, exactly like the
    supervisor's).
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import app.main; print(len(app.main.app.routes))"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        "import app.main failed — uvicorn would exit before binding the port:\n"
        + (proc.stderr or proc.stdout)[-3000:]
    )
    assert int(proc.stdout.strip().splitlines()[-1]) > 20


def test_boot_survives_a_midflight_job(tmp_path: Path) -> None:
    """The exact outage: boot recovery must not kill the module.

    `_boot()` runs at import time and its recovery pass is what raised the
    NameError — but only when a job was mid-flight, which is why the API worked
    on a clean start and died after every restart during a batch. Seeding a
    mid-flight job makes that code path execute on a plain import.
    """
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True)
    job_id = "deadbeefdeadbeefdeadbeefdeadbeef"
    (jobs / f"{job_id}.json").write_text(
        json.dumps(
            {
                "_id": job_id,
                "stage": "exporting",
                "progress": 60,
                "interrupted": True,
                "interruptCount": 1,
                "sourceType": "url",
                "source": "https://example.invalid/gone.mp4",
                "durationSec": 30,
                "createdAt": 0,
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ, CLIPFORGE_DATA_DIR=str(tmp_path))
    proc = subprocess.run(
        [sys.executable, "-c", "import app.main; print('boot-ok')"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        "importing app.main with a mid-flight job crashed — this is the outage:\n"
        + output[-3000:]
    )
    assert "boot-ok" in output
    # Recovery must have MADE A DECISION about the seeded job (resume or fail it),
    # not skipped it and not crashed on it.
    assert job_id in output, output[-2000:]
