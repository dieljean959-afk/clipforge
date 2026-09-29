"""The job read as "Failed" in one view and "running, stage 3 of 4" in another.

Both readings were true of the same document and neither was a fabrication:
the failure banner keyed off `stage`/`error`, while the progress block keyed
off `progress`, `progressDetail`, `stagePlan` and `queuePosition` — all of
which a failed job still carries, because the pipeline stops writing them the
moment it fails and nothing ever cleared them.

These tests pin the fix: one derivation (`app.jobstatus.describe`) used by
every read route, and a reconciliation that clears a job whose worker died
without reporting.

    python tools/job_status_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import jobstatus  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


# The real document from the reported bug: failed at the detection stage, but
# still carrying a full set of running-looking fields from its last tick.
FAILED_JOB: dict[str, Any] = {
    "_id": "f3b81a1dd7e6",
    "source": "https://youtu.be/O0yDqRwmMSE",
    "stage": "failed",
    "failedStage": "finding",
    "error": "AI detection failed: Groq rejected the request permanently (400)",
    "errorDetail": "Traceback (most recent call last): ...",
    "progress": 42.857142857142854,
    "progressDetail": "Detection: chunk 3/7 (cached)",
    "stagePlan": [
        {"stage": "Finding highlights", "sec": 50, "current": True},
        {"stage": "Framing 9:16", "sec": 21},
        {"stage": "Rendering clips", "sec": 420},
    ],
    "unitsDone": None, "unitsTotal": None, "unitLabel": "chunk",
    "totalEtaSec": 478.6, "stageEtaSec": None, "queuePosition": None,
    "lastProgressAt": 1790709059562, "stageStartedAt": 1790709059536,
    "clipCount": 10, "createdAt": 1790446529987,
}

# A job whose worker died without ever writing a terminal state.
STALE_RUNNING_JOB: dict[str, Any] = {
    "_id": "8d2dd9608be0",
    "source": "https://youtu.be/x",
    "stage": "finding",
    "progress": 61.0,
    "progressDetail": "Detection: chunk 3/7 (cached)",
    "stagePlan": [{"stage": "Finding highlights", "sec": 50, "current": True}],
    "unitsDone": 3, "unitsTotal": 7, "unitLabel": "chunk",
    "totalEtaSec": 478.6, "lastProgressAt": 1790709059562, "createdAt": 1,
}


def test_failed_job_is_not_running() -> None:
    print("\n[1] a failed job cannot read as running")
    d = jobstatus.describe(FAILED_JOB)
    check("status is failed", d["status"] == "failed", d["status"])
    check("is terminal", d["isTerminal"] is True)
    check("no active stage", d["activeStage"] is None, str(d["activeStage"]))
    check("names the stage it died in", d["stageLabel"] == "AI clip detection",
          str(d["stageLabel"]))
    # The fields that made a view show a spinner must be GONE, not just unused.
    for field in ("stagePlan", "unitsDone", "unitsTotal", "unitLabel",
                  "totalEtaSec", "queuePosition", "queueEtaSec"):
        check(f"{field} cleared on a failed job", field not in d, str(d.get(field)))
    check("progress is no longer a partial 42.9%", d["progress"] == 0, str(d["progress"]))
    check("progressDetail is not the stale chunk line",
          "chunk 3/7" not in str(d.get("progressDetail")), str(d.get("progressDetail")))


def test_running_job_is_clamped() -> None:
    print("\n[2] a running job is never 'done'")
    d = jobstatus.describe({"stage": "rendering", "progress": 100.0})
    check("clamped below 100", d["progress"] == 99.0, str(d["progress"]))
    check("not terminal", d["isTerminal"] is False)
    check("active stage mapped", d["activeStage"] == "rendering", str(d["activeStage"]))

    d = jobstatus.describe({"stage": "queued", "progress": 0})
    check("queued stays queued", d["status"] == "queued", d["status"])
    check("queued is not terminal", d["isTerminal"] is False)


def test_every_status_agrees() -> None:
    print("\n[3] the two fields that used to disagree can no longer disagree")
    for job, want_status, want_terminal in [
        (FAILED_JOB, "failed", True),
        (STALE_RUNNING_JOB, "running", False),
        ({"stage": "complete"}, "complete", True),
        ({"stage": "paused"}, "paused", True),
        ({"stage": "transcribing", "progress": 30}, "running", False),
        ({"stage": "queued"}, "queued", False),
        ({"stage": "grabbing"}, "running", False),
        ({}, "queued", False),
    ]:
        d = jobstatus.describe(job)
        check(f"{job.get('stage', '<none>')!r} -> {want_status}",
              d["status"] == want_status, d["status"])
        check(f"{job.get('stage', '<none>')!r} terminal={want_terminal}",
              d["isTerminal"] is want_terminal, str(d["isTerminal"]))
        # The invariant that fixes the bug: a terminal job never carries a
        # live stage, and a live job never carries terminal progress.
        if d["isTerminal"]:
            check(f"{job.get('stage', '<none>')!r} has no active stage",
                  d["activeStage"] is None, str(d["activeStage"]))
        else:
            check(f"{job.get('stage', '<none>')!r} is below 100%",
                  d["progress"] < 100, str(d["progress"]))


def test_status_of_aliases() -> None:
    print("\n[4] internal stage aliases resolve to user-facing stages")
    for raw, want in [
        ("grabbing", "grabbing"), ("fetch", "grabbing"), ("fetching", "grabbing"),
        ("transcribing", "transcribing"), ("transcribe", "transcribing"),
        ("finding", "finding"), ("detecting", "finding"), ("detect", "finding"),
        ("rendering", "rendering"), ("reframe", "rendering"),
    ]:
        check(f"{raw} -> {want}", jobstatus.plan_stage(raw) == want,
              str(jobstatus.plan_stage(raw)))


def test_failed_uses_failed_stage() -> None:
    print("\n[5] a failed job names the stage it actually died in")
    d = jobstatus.describe({"stage": "failed", "failedStage": "transcribing"})
    check("uses failedStage, not 'failed'", d["stageLabel"] == "Transcribing",
          str(d["stageLabel"]))
    # Older jobs have no failedStage; must not render a bare "Failed" label.
    d = jobstatus.describe({"stage": "failed"})
    check("no failedStage -> no misleading label", d["stageLabel"] is None,
          str(d["stageLabel"]))


def test_paused_is_not_running() -> None:
    print("\n[6] a paused job is idle, not running")
    d = jobstatus.describe({"stage": "paused", "pausedStage": "finding", "progress": 40})
    check("status is paused", d["status"] == "paused", d["status"])
    check("terminal", d["isTerminal"] is True)
    check("no active stage", d["activeStage"] is None, str(d["activeStage"]))
    check("pausedStage not echoed as running", "pausedStage" not in d)


def test_input_not_mutated() -> None:
    print("\n[7] describe() never mutates the stored document")
    original = dict(FAILED_JOB)
    jobstatus.describe(FAILED_JOB)
    check("input untouched", FAILED_JOB == original,
          "describe() mutated the caller's dict")
    # A copy is essential: the store's dict is what gets written back, and
    # clearing fields on it would silently destroy the record on the next save.
    check("a copy is returned", jobstatus.describe(FAILED_JOB) is not FAILED_JOB)


def test_hostile_input() -> None:
    print("\n[8] odd documents still produce a coherent record")
    for name, job in [
        ("progress as a string", {"stage": "finding", "progress": "42.5"}),
        ("progress as junk", {"stage": "finding", "progress": "abc"}),
        ("progress None", {"stage": "finding", "progress": None}),
        ("negative progress", {"stage": "finding", "progress": -5}),
        ("unknown stage", {"stage": "wat", "progress": 10}),
        ("stage is a number", {"stage": 7}),
    ]:
        try:
            d = jobstatus.describe(job)
            ok = isinstance(d.get("status"), str) and isinstance(d["progress"], (int, float))
            check(f"{name} -> coherent", ok, str(d.get("status")))
            check(f"{name} progress in range", 0 <= d["progress"] <= 100, str(d["progress"]))
        except Exception as exc:  # noqa: BLE001
            check(f"{name} -> coherent", False, f"raised {type(exc).__name__}: {exc}")


def main() -> int:
    test_failed_job_is_not_running()
    test_running_job_is_clamped()
    test_every_status_agrees()
    test_status_of_aliases()
    test_failed_uses_failed_stage()
    test_paused_is_not_running()
    test_input_not_mutated()
    test_hostile_input()
    print(f"\njob_status: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
