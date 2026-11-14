"""Does the real failed job now read consistently through the API?

The reported symptom was one job showing "Failed during AI clip detection"
and "Stage 3 of 4, 61% overall, Detection: chunk 3/7 (cached)" at the same
time. This boots the real app over the real data dir and prints what each
route returns, so the fix is checked against the actual document rather than a
fixture.

    python tools/job_status_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import jobstatus  # noqa: E402
from app.config import load_config  # noqa: E402
from app.store import LocalStore  # noqa: E402

#: Fields whose presence on a terminal job is what let a view render a spinner.
RUNNING_SHAPED = (
    "stagePlan", "unitsDone", "unitsTotal", "unitLabel",
    "stageEtaSec", "totalEtaSec", "queueEtaSec", "queuePosition",
)


def main() -> int:
    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)
    problems = 0
    for job in store.list_jobs():
        raw_id = job.get("_id", "?")
        d = jobstatus.describe(job)
        status = d["status"]
        terminal = d["isTerminal"]
        print(f"\n{raw_id[:12]}  stage={job.get('stage')}  -> status={status} terminal={terminal}")
        print(f"  stored progress : {job.get('progress')}")
        print(f"  served progress : {d['progress']}")
        print(f"  served detail   : {str(d.get('progressDetail'))[:60]!r}")
        print(f"  stageLabel      : {d['stageLabel']}")

        leaked = [f for f in RUNNING_SHAPED if f in d]
        if terminal and leaked:
            print(f"  LEAK            : running-only fields still served: {leaked}")
            problems += 1
        else:
            print(f"  running fields  : {leaked or 'none (correct for terminal)'}")

        if terminal and d["activeStage"] is not None:
            print(f"  LEAK            : activeStage set on a {status} job")
            problems += 1
        if not terminal and d["progress"] >= 100:
            print("  LEAK            : a live job reports >=100%")
            problems += 1
        if terminal and "chunk" in str(d.get("progressDetail", "")).lower():
            print("  LEAK            : a dead job still shows a chunk progress line")
            problems += 1

    print(f"\n{'FAIL' if problems else 'OK'}: {problems} inconsistency(ies)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
