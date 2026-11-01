#!/usr/bin/env python3
"""Clipmake watchdog — the automatic version of the reset button.

Runs as a separate process, so it can still act when the API is hung, wedged in
a C call, or serving 500s:

  * GET /api/health every 15s with a 3s timeout
  * 3 consecutive failures  -> request a restart
  * memory above 85% of RAM -> request a restart
  * event loop blocked >10s -> request a restart
  * cooldown between automatic restarts so a slow boot is not punished

A restart request is written to .data/system/restart-request.json; the
supervisor picks it up within ~0.5s and does the actual killing/restarting.
That split is deliberate: the watchdog has no business owning process handles.

Usage:
    .venv/bin/python scripts/watchdog.py
    .venv/bin/python scripts/watchdog.py --once        # one check, then exit
    .venv/bin/python scripts/watchdog.py --failures 2 --interval 5
"""

from __future__ import annotations

import argparse
import importlib.util
import os
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

MEMORY_LIMIT = 0.85       # fraction of total RAM
LOOP_LAG_LIMIT_MS = 10_000
RESTART_COOLDOWN = 120.0  # seconds between automatic restarts


def _get_health(port: str, timeout: float) -> tuple[bool, str]:
    url = f"http://127.0.0.1:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - loopback
            if resp.status == 200:
                return True, "ok"
            return False, f"HTTP {resp.status}"
    except Exception as exc:  # noqa: BLE001 - any failure to answer is a failure
        return False, f"{type(exc).__name__}: {exc}"


def _restart(source: str, reason: str, last: list[float]) -> None:
    now = time.time()
    recent = [t for t in last if now - t < RESTART_COOLDOWN]
    if recent:
        system.log(f"[watchdog] restart suppressed (cooldown): {reason}")
        return
    last.append(now)
    system.request_restart(source, reason)
    system.log(f"[watchdog] RESTART requested: {reason}")


LAG_FRESH_SEC = 15.0


def _lag_from_payload(payload: dict, now: float) -> float:
    """Event-loop lag from a published sample, or 0.0 if it is stale/corrupt.

    A stale sample must never trigger a restart: if the server stopped
    publishing, the health check is the authority, not an old number.
    """
    try:
        stamp = time.mktime(time.strptime(str(payload.get("time")), "%Y-%m-%dT%H:%M:%SZ"))
        if now - stamp > LAG_FRESH_SEC:
            return 0.0
        return float(payload.get("ms", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _loop_lag_ms() -> float:
    return _lag_from_payload(
        system._read_json(system.LOOP) or {},  # noqa: SLF001 - same module family
        time.time(),
    )


def _decide(
    *,
    served: bool,
    failures: int,
    failures_needed: int,
    memory_ratio: float | None,
    loop_lag_ms: float,
) -> tuple[str, str] | None:
    """Which restart trigger fires, if any — the watchdog's whole policy.

    Pure so it can be tested at the exact thresholds without a live server.
    Health failures come first: if the server is not answering, the memory and
    lag numbers are not trustworthy anyway.
    """
    if failures >= failures_needed:
        return ("health", f"{failures} consecutive health failures")
    if memory_ratio is not None and memory_ratio > MEMORY_LIMIT:
        return ("memory", f"memory {memory_ratio * 100:.1f}% of RAM above {MEMORY_LIMIT * 100:.0f}%")
    if loop_lag_ms > LOOP_LAG_LIMIT_MS:
        return ("loop", f"event loop blocked {loop_lag_ms:.0f}ms (> {LOOP_LAG_LIMIT_MS}ms)")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Clipmake watchdog")
    parser.add_argument("--port", default=os.getenv("PORT", os.getenv("CLIPFORGE_PORT", "8000")))
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--failures", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    failures = 0
    restarts: list[float] = []
    system.log(f"[watchdog] online pid={os.getpid()} port={args.port} every {args.interval}s")

    while True:
        served, detail = _get_health(args.port, args.timeout)
        pid = system.server_pid()
        ratio = system.memory_ratio(pid)
        lag = _loop_lag_ms()

        if served:
            failures = 0
        else:
            failures += 1

        system.write_watchdog(
            ok=served, detail=detail, failures=failures, pid=pid,
            memoryRatio=round(ratio, 3) if ratio is not None else None,
            loopLagMs=lag,
        )
        system.log(
            f"[watchdog] health={'ok' if served else 'FAIL'} ({detail}) "
            f"failures={failures}/{args.failures} rss={round(ratio * 100, 1) if ratio is not None else '?'}% "
            f"lag={lag}ms"
        )

        trigger = _decide(
            served=served,
            failures=failures,
            failures_needed=args.failures,
            memory_ratio=ratio,
            loop_lag_ms=lag,
        )
        if trigger:
            kind, reason = trigger
            if kind == "health":
                reason = f"{reason} ({detail})"
            _restart("watchdog", reason, restarts)
            if kind == "health":
                failures = 0

        if args.once:
            return 0 if served else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        system.log("[watchdog] stopped")
        raise SystemExit(0)
