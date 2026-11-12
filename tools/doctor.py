#!/usr/bin/env python
"""One pass/fail report for this host: boot, health, routes, model, bucket, LLM.

Usage::

    .venv/bin/python tools/doctor.py            # human report, exit 1 on FAIL
    .venv/bin/python tools/doctor.py --json     # the exact /api/doctor payload

The rule the report follows: a check that finds a problem WARNS and, where it
can, cleans up. It never turns "this host is short on disk" into "this batch
failed" — a batch with verified clips stays successful.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _api(path: str, timeout: float = 5.0):
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:8000{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:
            return exc.code, {}
    except Exception as exc:  # noqa: BLE001 - reported as a check, not a crash
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


def collect() -> dict:
    """Prefer the live API (it owns the worker/model state); fall back locally."""
    from app.config import load_config
    from app import doctor, invariants
    from app.store import LocalStore

    settings = load_config(quiet=True)
    code, live = _api("/api/doctor")
    if code == 200 and isinstance(live, dict) and live.get("checks"):
        report = live
        report["source"] = "live-api"
    else:
        report = doctor.run_doctor(settings)
        report["dependencies"] = doctor.dependency_status(settings)
        report["source"] = f"offline ({code or 'no answer'})"

    store = LocalStore(settings.app.paths)
    try:
        report["invariants"] = invariants.check_and_repair(store, settings.app.paths)
    except Exception as exc:  # noqa: BLE001
        report["invariants"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Route smoke test: a 5xx or an empty body is a FAIL; 404 on a route we
    # wrote is a FAIL too (that is a missing endpoint, not a missing page).
    routes = [
        "/api/health", "/api/status", "/api/boot", "/api/jobs",
        "/api/jobs/fd714625e79a4db895aeafc43931b3e0", "/api/clips",
    ]
    smoke = []
    for route in routes:
        status, body = _api(route)
        smoke.append({
            "route": route,
            "status": status,
            "ok": status == 200,
            "error": (body or {}).get("error") if isinstance(body, dict) else None,
        })
    report["routes"] = smoke
    report["ok"] = bool(report.get("ok")) and all(r["ok"] for r in smoke)
    return report


def render(report: dict) -> str:
    lines: list[str] = []
    summary = report.get("summary") or {}
    lines.append(
        f"ClipForge doctor — {report.get('source', 'local')} — "
        f"{'OK' if report.get('ok') else 'FAIL'} "
        f"({summary.get('ok', 0)} ok, {summary.get('warn', 0)} warn, "
        f"{summary.get('fail', 0)} fail)"
    )
    for check in report.get("checks", []):
        mark = {"ok": "PASS", "warn": "WARN", "fail": "FAIL"}.get(
            check.get("status"), "INFO"
        )
        lines.append(f"  [{mark}] {check.get('name')}: {check.get('detail')}")
    deps = report.get("dependencies") or []
    if deps:
        lines.append("  dependencies:")
        for dep in deps:
            lines.append(
                f"    [{'PASS' if dep.get('ok') else 'FAIL'}] {dep.get('name')}: "
                f"{dep.get('detail')}"
            )
    routes = report.get("routes") or []
    if routes:
        lines.append("  route smoke:")
        for r in routes:
            lines.append(f"    [{'PASS' if r['ok'] else 'FAIL'}] {r['route']} -> {r['status']}"
                         + (f" ({r['error']})" if r.get("error") else ""))
    boot = report.get("boot") or {}
    if boot:
        lines.append(
            f"  boot: listen={boot.get('listenMs')}ms firstHealth={boot.get('firstHealthMs')}ms "
            f"failed={boot.get('failedPhases') or []}"
        )
    inv = report.get("invariants") or {}
    if inv:
        lines.append(
            f"  invariants: checked={inv.get('checked')} repaired={inv.get('repaired')}"
            + (f" error={inv['error']}" if inv.get("error") else "")
        )
    return "\n".join(lines)


def main() -> int:
    as_json = "--json" in sys.argv
    report = collect()
    print(json.dumps(report, indent=1, default=str) if as_json else render(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
