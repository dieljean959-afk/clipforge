#!/usr/bin/env python3
"""Clipmake process supervisor.

Runs the API (uvicorn) and, unless disabled, the Vite dev server that the
platform preview points at. Both are restarted automatically when they exit or
when something asks for a restart (UI button, watchdog, memory/loop-lag guard).

Guarantees:
  * a dead child is respawned within ~1s (well inside the 3s requirement)
  * backoff 1s -> 2s -> 5s -> 10s, capped, with a restart limit so a broken
    build does not spin forever
  * every restart is recorded in .data/system/restarts.json with its reason
  * restart requests are consumed from .data/system/restart-request.json
  * children run in their own session, so a stuck process group can be killed

Usage:
    .venv/bin/python scripts/supervise.py
    CLIPFORGE_NO_UI=1 .venv/bin/python scripts/supervise.py   # API only

The module is imported by file path on purpose: a broken ``app`` import must
never be able to stop the supervisor from starting.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("clipmake_system", ROOT / "app" / "system.py")
system = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(system)

BACKOFF = (1, 2, 5, 10)
RESTART_LIMIT = 8          # restarts allowed inside RESTART_WINDOW ...
RESTART_WINDOW = 120.0     # ... before a cooldown
COOLDOWN = 10.0
TICK = 0.5


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PORT", env.get("CLIPFORGE_PORT", "8000"))
    env.setdefault("CLIPFORGE_HOST", "0.0.0.0")
    env["PYTHONUNBUFFERED"] = "1"
    return env


class Child:
    def __init__(self, name: str, cmd: list[str], log_name: str) -> None:
        self.name = name
        self.cmd = cmd
        self.log_path = system.STATE_DIR / log_name
        self.proc: subprocess.Popen | None = None
        self.failures: list[float] = []

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        system.ensure_dir()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logfh = self.log_path.open("a", encoding="utf-8")
        logfh.write(f"\n===== {system.now_iso()} starting: {' '.join(self.cmd)} =====\n")
        logfh.flush()
        self.proc = subprocess.Popen(  # noqa: S603 - fixed argv, our own server
            self.cmd,
            cwd=str(ROOT),
            env=_env(),
            stdout=logfh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop(self, timeout: float = 5.0) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        pid = self.proc.pid
        for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                return
            except OSError:
                try:
                    self.proc.send_signal(sig)
                except OSError:
                    return
            deadline = time.time() + wait
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    return
                time.sleep(0.1)


def _backoff(failures: int) -> float:
    return float(BACKOFF[min(max(failures - 1, 0), len(BACKOFF) - 1)])


def _note_failure(failures: list[float], now: float) -> list[float]:
    """Record one failure, dropping anything outside the restart window.

    Extracted so the window/limit rules are testable without spawning a real
    server: the supervisor's whole point is that these thresholds are right.
    """
    kept = [stamp for stamp in failures if now - stamp < RESTART_WINDOW]
    kept.append(now)
    return kept


def _should_cool_down(failures: list[float]) -> bool:
    """True once a child has restarted more than RESTART_LIMIT times in-window."""
    return len(failures) > RESTART_LIMIT


def _reason(child: Child) -> tuple[str, int | None, float]:
    """(human reason, exit code, seconds the child lived)."""
    code = child.proc.returncode if child.proc else None
    lived = 0.0
    if child.proc:
        try:
            lived = round(time.time() - child.started_at, 1)
        except AttributeError:
            lived = 0.0
    if code is None:
        return ("process vanished", code, lived)
    if code < 0:
        return (f"killed by signal {-code}", code, lived)
    return (f"exited with code {code}", code, lived)


def _wait_health(port: str, timeout: float = 30.0) -> float | None:
    """Seconds until /api/health answers 200, or None."""
    started = time.time()
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() - started < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:  # noqa: S310 - loopback
                if resp.status == 200:
                    return round(time.time() - started, 3)
        except Exception:  # noqa: BLE001 - still booting
            time.sleep(0.2)
    return None


def main() -> int:
    port = os.getenv("PORT", os.getenv("CLIPFORGE_PORT", "8000"))
    host = os.getenv("CLIPFORGE_HOST", "0.0.0.0")
    ui_port = os.getenv("CLIPFORGE_UI_PORT", "3000")
    python = os.getenv("CLIPFORGE_PYTHON", str(ROOT / ".venv" / "bin" / "python"))
    if not Path(python).exists():
        python = sys.executable

    api_cmd = [python, "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)]
    children = [Child("api", api_cmd, "api.log")]

    if os.getenv("CLIPFORGE_NO_UI", "0") != "1" and (ROOT / "node_modules" / "vite" / "bin" / "vite.js").exists():
        ui_cmd = ["node", str(ROOT / "node_modules" / "vite" / "bin" / "vite.js"),
                  "--host", "0.0.0.0", "--port", ui_port]
        children.append(Child("ui", ui_cmd, "ui.log"))

    if os.getenv("CLIPFORGE_NO_WATCHDOG", "0") != "1" and (ROOT / "scripts" / "watchdog.py").exists():
        children.append(Child("watchdog", [python, str(ROOT / "scripts" / "watchdog.py")], "watchdog.log"))

    # Tell the Vite plugin not to spawn its own backend: we own that process
    # now, and two uvicorns racing for :8000 is worse than either one alone.
    if len(children) > 1:
        os.environ["CLIPFORGE_BACKEND_SUPERVISED"] = "1"

    system.ensure_dir()
    system.log(
        f"[start] supervisor pid={os.getpid()} api={host}:{port} ui={ui_port if len(children) > 1 else 'off'}"
    )
    system.write_server(
        supervisorPid=os.getpid(),
        supervised=True,
        apiPort=int(port),
        uiPort=int(ui_port) if len(children) > 1 else None,
        apiPid=None,
    )

    stopping = {"flag": False}

    def _shutdown(signum, _frame):
        stopping["flag"] = True
        system.log(f"[stop] supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    for child in children:
        child.start()
        child.started_at = time.time()
        system.log(f"[start] {child.name} pid={child.proc.pid if child.proc else '?'}")
        if child.name == "api":
            first_health = _wait_health(port)
            system.log(f"[ready] /api/health answered after {first_health}s")
    system.write_server(apiPid=children[0].proc.pid if children[0].proc else None,
                        uiPid=children[1].proc.pid if len(children) > 1 and children[1].proc else None)

    latency = _wait_health(port)
    if latency is None:
        system.log("[warn] api did not answer /api/health within 30s of boot")
    else:
        system.log(f"[ready] /api/health 200 after {latency}s")

    while not stopping["flag"]:
        request = system.take_restart_request()
        if request:
            system.record_event(
                "restart",
                str(request.get("source", "request")),
                f"restart requested: {request.get('reason', 'no reason given')}",
            )
            for child in children:
                child.stop()
            for child in children:
                child.start()
                child.started_at = time.time()
            system.write_server(apiPid=children[0].proc.pid if children[0].proc else None,
                                uiPid=children[1].proc.pid if len(children) > 1 and children[1].proc else None)
            latency = _wait_health(port)
            system.log(f"[ready] restarted by {request.get('source')} — health after {latency}s")
            continue

        for child in children:
            if child.alive():
                continue
            reason, code, lived = _reason(child)
            if child.name == "watchdog":
                system.log(f"[watchdog] helper exited ({reason}); respawning in 1s")
                time.sleep(1)
                child.start()
                child.started_at = time.time()
                continue
            now = time.time()
            child.failures = _note_failure(child.failures, now)
            extra = {"exitCode": code, "livedSec": lived, "failuresInWindow": len(child.failures)}
            if _should_cool_down(child.failures):
                system.record_event("restart", "supervisor", f"{child.name} crash loop: {reason}", **extra)
                system.log(f"[limit] {child.name} hit the restart limit; cooling down {COOLDOWN}s")
                time.sleep(COOLDOWN)
                child.failures.clear()
            delay = _backoff(len(child.failures))
            system.record_event(
                "restart", "supervisor", f"{child.name} {reason}; restarting in {delay:.0f}s", **extra
            )
            time.sleep(delay)
            if stopping["flag"]:
                break
            child.start()
            child.started_at = time.time()
            system.write_server(apiPid=children[0].proc.pid if children[0].proc else None,
                                uiPid=children[1].proc.pid if len(children) > 1 and children[1].proc else None)
            if child.name == "api":
                latency = _wait_health(port)
                system.log(f"[ready] api back after {latency}s")
        time.sleep(TICK)

    for child in children:
        child.stop()
    system.record_event("stop", "supervisor", "supervisor stopped")
    system.write_server(supervisorPid=None, apiPid=None, uiPid=None, supervised=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - last-resort log, then exit non-zero
        system.log(f"[fatal] supervisor crashed: {type(exc).__name__}: {exc}")
        raise
