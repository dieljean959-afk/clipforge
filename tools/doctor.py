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


def _local_token() -> str:
    """The API needs a session; local tooling uses data/auth/automation.token."""
    import os

    base = Path(os.getenv("CLIPFORGE_DATA_DIR", str(ROOT / "data"))) / "auth"
    try:
        return (base / "automation.token").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _api(path: str, timeout: float = 5.0):
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:8000{path}"
    token = _local_token()
    headers = {"X-ClipForge-Token": token} if token else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    # The live doctor includes a 1-token provider call and a bucket round-trip;
    # five seconds was shorter than a cold DeepSeek response, so the tool fell
    # back to an offline run and lost the bucket/clip/share checks.
    code, live = _api("/api/doctor", timeout=45.0)
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

    # ---- the four checks the loop/ETA/model incidents asked for -----------
    report["stateMachine"] = _state_machine(store)
    report["batchesVsBucket"] = _batches_vs_bucket(store)
    report["modelFolder"] = _model_folder()
    report["etaSanity"] = _eta_sanity()

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


def _state_machine(store) -> dict:
    """A settled batch must never sit on a non-terminal stage, and vice versa.

    This is the check that would have caught the QqdCXWFobi8 loop on day one:
    the document said finished, the recovery path said queued.
    """
    from app import jobstatus

    bad: list[dict] = []
    checked = 0
    for job in store.list_jobs():
        checked += 1
        stage = str(job.get("stage") or "")
        terminal = jobstatus.is_terminal(job)
        if terminal and stage not in jobstatus.TERMINAL_STAGES:
            bad.append({"jobId": job.get("_id"), "why": f"terminal on stage {stage!r}"})
        if not terminal and stage in jobstatus.TERMINAL_STAGES:
            bad.append({"jobId": job.get("_id"), "why": f"stage {stage!r} without a terminal stamp"})
        if job.get("finishedAt") and jobstatus.is_requeueable(job):
            bad.append({"jobId": job.get("_id"), "why": "finished but still re-queueable"})
    return {"checked": checked, "bad": bad[:20], "ok": not bad}


def _batches_vs_bucket(store) -> dict:
    """Every finished batch must be listed; every listed clip must exist.

    "Your batches" reads clip rows, so the failure mode to catch is clips on
    disk/in the bucket that no batch names — exactly the 9-clip batch that
    rendered into the bucket and appeared nowhere.
    """
    listed = 0
    orphans: list[dict] = []
    for job in store.list_jobs():
        jid = str(job.get("_id") or "")
        docs = store.list_clip_docs_any_authority(jid) if jid else []
        ready = [d for d in docs if str(d.get("renderState")) == "done"]
        if ready:
            listed += 1
        if ready and not (job.get("isTerminal") or str(job.get("stage")) == "complete"):
            orphans.append({"jobId": jid, "readyClips": len(ready),
                            "stage": job.get("stage")})
    return {"batchesWithClips": listed, "notFinished": orphans[:20], "ok": not orphans}


def _model_folder() -> dict:
    """Report every model copy with its size and whether a release would free it."""
    from app import modelstore

    try:
        root = modelstore._root()
        total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        active, note = modelstore.active_model_name()
        plan = modelstore.releasable(active, keep_active_local=True)
        return {
            "path": str(root),
            "mb": round(total / 1048576, 1),
            "active": active,
            "note": note,
            "freeableMb": plan.get("mb"),
            "freeable": plan.get("items"),
            "kept": plan.get("kept"),
            "ok": total < 1200 * 1048576,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "ok": False}


def _eta_sanity() -> dict:
    """The ETA the UI would show must be finite, positive and <= 24 h."""
    from app import eta

    samples = [eta.eta_from_progress(48_853.0, 0.04), eta.clamp_eta(50 * 3600),
               eta.eta_from_progress(100.0, 0.5)]
    worst = max((s for s in samples if s is not None), default=None)
    return {
        "samples": samples,
        "maxSec": worst,
        "capSec": eta.MAX_ETA_SEC,
        "ok": worst is None or worst <= eta.MAX_ETA_SEC,
    }


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
    sm = report.get("stateMachine") or {}
    if sm:
        lines.append(f"  [{'PASS' if sm.get('ok') else 'FAIL'}] state machine: "
                     f"{sm.get('checked')} jobs checked, {len(sm.get('bad') or [])} bad")
        for item in (sm.get("bad") or [])[:5]:
            lines.append(f"    {item['jobId']}: {item['why']}")
    bb = report.get("batchesVsBucket") or {}
    if bb:
        lines.append(f"  [{'PASS' if bb.get('ok') else 'FAIL'}] your batches vs clips: "
                     f"{bb.get('batchesWithClips')} batches have clips, "
                     f"{len(bb.get('notFinished') or [])} with clips but not finished")
    mf = report.get("modelFolder") or {}
    if mf:
        lines.append(f"  [{'PASS' if mf.get('ok') else 'WARN'}] model folder: "
                     f"{mf.get('mb')} MB at {mf.get('path')}, active={mf.get('active')}, "
                     f"freeable now={mf.get('freeableMb')} MB")
        for kept in (mf.get("kept") or [])[:3]:
            lines.append(f"    kept {kept.get('mb')} MB — {kept.get('why')}")
    es = report.get("etaSanity") or {}
    if es:
        lines.append(f"  [{'PASS' if es.get('ok') else 'FAIL'}] ETA sanity: "
                     f"max={es.get('maxSec')}s cap={es.get('capSec')}s")
    return "\n".join(lines)


def main() -> int:
    as_json = "--json" in sys.argv
    report = collect()
    print(json.dumps(report, indent=1, default=str) if as_json else render(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
