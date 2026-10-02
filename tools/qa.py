#!/usr/bin/env python3
"""One command for the whole ClipForge QA suite.

    python tools/qa.py                 # every suite, grouped by area
    python tools/qa.py --area share    # one area
    python tools/qa.py --suite share_endpoint_test
    python tools/qa.py --shard 1/4     # for terminals with a wall-clock limit
    python tools/qa.py --list          # the matrix, no execution

Prints PASS/FAIL per suite and a summary, and writes data/qa-report.json so two
runs can be compared (`--record baseline` appends to data/qa-history.jsonl).

Why a runner instead of a shell loop: the suites print their own per-check
lines, but several of them predate the convention of exiting non-zero on
failure, so a shell loop that trusted exit codes alone would report a green run
for a suite that printed failures. A suite is PASS here only if it exited 0 AND
printed no failure marker. Nothing is weakened: this only ever adds a second
opinion, never removes one.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")

# --------------------------------------------------------------------------- #
# The matrix. One row per suite, tagged with the area it is evidence for.
# --------------------------------------------------------------------------- #

AREAS: dict[str, list[tuple[str, list[str], int]]] = {
    "pipeline": [
        ("transcribe_unit_test", ["--fast"], 150),
        ("encode_profile_test", [], 90),
        ("e2e_render_test", [], 150),
        ("render_loop_test", [], 120),
        ("playback_test", [], 90),
        ("job_status_test", [], 90),
    ],
    "reliability": [
        ("reliability_test", ["--seconds", "20"], 170),
        ("stale_reconcile_test", [], 120),
        ("heartbeat_test", [], 90),
        ("worker_supervision_test", [], 90),
        ("resume_invalidation_test", [], 90),
        ("pause_resume_test", [], 120),
        ("boot_guard_test", [], 90),
        ("stability_test", [], 170),
        ("delete_cancels_test", [], 90),
        ("queue_truthfulness_test", [], 90),
        ("health_resilience_test", [], 90),
        ("export_memory_test", [], 120),
        ("pipeline_resilience_test", [], 120),
    ],
    "ai": [
        ("jsonfix_test", [], 60),
        ("detect_json_test", [], 90),
        ("deepseek_provider_test", [], 90),
        ("detect_backend_test", [], 90),
        ("llm_provider_test", [], 90),
        ("nvidia_provider_test", [], 90),
    ],
    "loading": [
        ("preview_layout_test", [], 90),
        ("mobile_overflow_test", [], 90),
        ("export_memory_test", [], 120),
    ],
    "share": [
        ("share_public_test", [], 150),
        ("share_instant_test", [], 150),
        ("share_endpoint_test", [], 120),
        ("share_intents_test", [], 120),
    ],
    "storage": [
        ("backend_supervisor_test", [], 90),
        ("storage_test", [], 90),
        ("stability_test", [], 170),
    ],
    "ui": [
        ("ui_bugfix_test", [], 90),
        ("caption_social_test", [], 120),
        ("caption_preset_test", [], 120),
        ("preview_layout_test", [], 90),
    ],
}

#: A suite name can appear under more than one area; run it once.
SEEN: dict[str, tuple[str, list[str], int]] = {}

#: Suites that are not standalone scripts would need a stub file each; there
#: are none left. Kept as an explicit empty map so a future addition has an
#: obvious home rather than a magic branch in `run_suite`.
ALIASES: dict[str, str] = {}


def flatten(areas: list[str]) -> list[tuple[str, str, list[str], int]]:
    rows: list[tuple[str, str, list[str], int]] = []
    for area in areas:
        for name, args, timeout in AREAS[area]:
            if name not in SEEN and (ROOT / "tools" / f"{name}.py").exists():
                SEEN[name] = (name, args, timeout)
            if name in SEEN:
                rows.append((area, name, args, timeout))
    return rows

#: A line that means "this suite reported a failure", whatever its exit code.
FAIL_LINE = re.compile(r"\bFAIL\b|FAILED:|\d+ failed", re.IGNORECASE)
SUMMARY_LINE = re.compile(r"(\d+)\s+passed,\s*(\d+)\s+failed", re.IGNORECASE)
ALL_PASSED = re.compile(r"ALL (\d+) (?:CHECKS )?PASSED", re.IGNORECASE)


def run_suite(name: str, args: list[str], timeout: int) -> dict:
    """Run one suite. Returns a result row; never raises."""
    started = time.monotonic()
    if name in ALIASES:
        cmd = [PY, "-c", ALIASES[name]]
    else:
        cmd = [PY, "-u", str(ROOT / "tools" / f"{name}.py"), *args]
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout
        )
        out = proc.stdout + proc.stderr
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(
            exc.stdout, bytes
        ) else (exc.stdout or "")
        out += f"\nTIMEOUT after {timeout}s"
        code = 124

    took = time.monotonic() - started
    passed = failed = None
    m = SUMMARY_LINE.search(out)
    if m:
        passed, failed = int(m.group(1)), int(m.group(2))
    else:
        m = ALL_PASSED.search(out)
        if m:
            passed, failed = int(m.group(1)), 0

    # "0 failed" is not a failure, so the number decides and the word only
    # locates the line to read.
    real_fail = [
        ln for ln in out.splitlines()
        if re.search(r"\bFAIL\b|FAILED:", ln) and not re.search(r"0 failed", ln)
    ]
    ok = code == 0 and not real_fail and (failed is None or failed == 0)
    return {
        "suite": name,
        "exit": code,
        "passed": passed,
        "failed": failed,
        "seconds": round(took, 1),
        "ok": ok,
        "tail": out.strip().splitlines()[-1][:160] if out.strip() else "",
        "detail": real_fail[:3],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", action="append", choices=sorted(AREAS),
                    help="only these areas (repeatable)")
    ap.add_argument("--suite", action="append", help="only these suites")
    ap.add_argument("--shard", help="1/4 — run a slice, for a time-limited shell")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--record", help="append this run to data/qa-history.jsonl under a label")
    args = ap.parse_args()

    if args.list:
        for area, rows in AREAS.items():
            print(f"{area}:")
            for name, a, t in rows:
                print(f"  {name:28s} {' '.join(a):16s} timeout={t}s")
        return 0

    rows = flatten(args.area or sorted(AREAS))
    if args.suite:
        wanted = set(args.suite)
        rows = [r for r in rows if r[1] in wanted]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        # Positional, after de-duplication, so shards are stable and never
        # empty. A hash would scatter related suites and leave a shard with
        # nothing at all to run.
        unique: list[tuple[str, str, list[str], int]] = []
        seen: set[str] = set()
        for row in rows:
            if row[1] in seen:
                continue
            seen.add(row[1])
            unique.append(row)
        rows = [r for j, r in enumerate(unique) if j % n == i % n]

    print("ClipForge QA")
    print(f"  python : {PY}")
    print(f"  suites : {len(rows)}")
    print()

    results: list[dict] = []
    by_area: dict[str, list[dict]] = {}
    seen_done: set[str] = set()
    for area, name, sargs, timeout in rows:
        if name in seen_done:
            by_area.setdefault(area, []).append({"suite": name, "ok": True,
                                                 "skipped": "already run"})
            continue
        seen_done.add(name)
        print(f"  {name} ...", flush=True, end=" ")
        res = run_suite(name, sargs, timeout)
        res["area"] = area
        results.append(res)
        by_area.setdefault(area, []).append(res)
        print(f"{'PASS' if res['ok'] else 'FAIL'}  {res['seconds']}s"
              + (f"  ({res['passed']} checks)" if res["passed"] is not None else ""))

    print()
    print("=" * 74)
    for area in sorted(by_area):
        rs = [r for r in by_area[area] if not r.get("skipped")]
        bad = [r for r in rs if not r["ok"]]
        print(f"{area.upper():12s} {len(rs) - len(bad)}/{len(rs)} suites pass")
        for r in bad:
            print(f"   FAIL {r['suite']} (exit {r['exit']}, {r['seconds']}s)")
            for d in r.get("detail", []):
                print(f"        {d}")
    total_ok = sum(1 for r in results if r["ok"])
    print("=" * 74)
    print(f"TOTAL: {total_ok}/{len(results)} suites pass, "
          f"{sum(r['seconds'] for r in results):.0f}s wall clock")
    print("=" * 74)

    report = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "suites": results,
        "total": len(results),
        "passed": total_ok,
    }
    out_path = ROOT / "data" / "qa-report.json"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.record:
            with (ROOT / "data" / "qa-history.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**report, "label": args.record}) + "\n")
            print(f"recorded as '{args.record}' in data/qa-history.jsonl")
    except OSError as exc:
        print(f"(could not write data/qa-report.json: {exc})")

    return 0 if total_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
