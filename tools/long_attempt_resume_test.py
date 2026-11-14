#!/usr/bin/env python3
"""A long attempt killed twice at one checkpoint must NOT become a wedge.

The user's requirement: start a job, walk away, and it finishes. On this host
that is not a formality. Measured stage times for this project are `fetch`
6167s and `detect` 1243s, and the sandbox recycles the web process every
15-60 minutes. So a multi-hour job is routinely SIGKILLed partway through a
single render unit, repeatedly, while doing exactly what it was asked to do.

Before this was fixed, two such kills at the same checkpoint exhausted
STALLED_RESUME_LIMIT and the job was failed with "the server restarted 3 times
... it cannot be resumed automatically" -- which is the opposite of the truth,
and was the failure the user actually hit.

The rule under test: an attempt that lived at least MIN_USEFUL_ATTEMPT_SEC is
forward progress even when its signature is unchanged, because time was spent
doing real work. An attempt that dies almost immediately, over and over, at the
same point, is a real wedge and still stops.

Run: .venv/bin/python tools/long_attempt_resume_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures: list[str] = []
checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))
        failures.append(label)


class _Store:
    """Minimal job store: _note_interrupted only reads/writes one document."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def get_job(self, jid: str):
        return self.jobs.get(jid)

    def write_job(self, job: dict) -> None:
        self.jobs[job["_id"]] = dict(job)

    def append_job_log(self, jid: str, msg: str) -> None:
        self.jobs[jid].setdefault("logs", []).append(msg)


def attempt(store: _Store, lived_sec: float, signature: str, label: str):
    """Drive one simulated host kill and report (resumed, stalledResumes)."""
    import threading

    from app.worker import InProcessWorker

    worker = object.__new__(InProcessWorker)
    worker.store = store
    worker._lock = threading.RLock()
    worker._progress_signature = lambda _jid: signature  # type: ignore[method-assign]

    # Record re-queues instead of performing them. What is under test is the
    # DECISION to resume (the stall counter), not the queue plumbing behind
    # enqueue(), which reaches into worker state this harness does not build.
    requeued: list[str] = []

    def _fake_enqueue(job_id: str) -> int:
        requeued.append(job_id)
        return len(requeued)

    worker.enqueue = _fake_enqueue  # type: ignore[method-assign]
    attempt.requeued = requeued  # type: ignore[attr-defined]

    jid = "longattempt"
    # Carry the counters FORWARD between attempts, the way the real re-queue
    # path does. Resetting them per call would make a wedge look like a first
    # attempt every time and the limit could never be reached -- a harness bug
    # that would have hidden a real one.
    prior = store.jobs.get(jid, {})
    store.jobs[jid] = {
        "_id": jid,
        "stage": "rendering",
        # The attempt began `lived_sec` ago.
        "stageStartedAt": (time.time() - lived_sec) * 1000,
        "resumeSignature": prior.get("resumeSignature", signature),
        "stalledResumes": int(prior.get("stalledResumes") or 0),
        "interruptCount": int(prior.get("interruptCount") or 0),
        "logs": list(prior.get("logs") or []),
    }
    resumed = worker._note_interrupted(store.jobs[jid], jid, -9, "rendering", "")
    # Re-read from the store: _note_interrupted writes the updated document
    # back, and that is what the next attempt would see.
    stored = store.jobs[jid]
    stalled = stored.get("stalledResumes")
    print(f"       {label}: lived={lived_sec:.0f}s resumed={resumed} stalled={stalled}")
    return resumed, stalled


def main() -> int:
    print("long attempt resume")
    from app.worker import MIN_USEFUL_ATTEMPT_SEC, STALLED_RESUME_LIMIT

    check(
        "MIN_USEFUL_ATTEMPT_SEC is 10 minutes",
        MIN_USEFUL_ATTEMPT_SEC == 600.0,
        f"got {MIN_USEFUL_ATTEMPT_SEC!r}",
    )
    check(
        "MIN_USEFUL_ATTEMPT_SEC exceeds the stall limit's intent",
        MIN_USEFUL_ATTEMPT_SEC > 60,
        "a wedge could outlive the useful-attempt window",
    )

    # --- the reported failure: two long kills, same checkpoint --------------
    sig = "4/6 stages, 0 clip(s) rendered, 8 chunk(s) transcribed"
    store = _Store()
    _, first = attempt(store, 3700, sig, "kill 1 (1h01m into the render)")
    resumed2, second = attempt(store, 3400, sig, "kill 2 (56m into the same render)")
    check("first long kill resumes", first == 0, f"stalled={first}")
    check("second long kill STILL resumes", resumed2 is True, f"resumed={resumed2}")
    check("stall counter stayed at zero", second == 0, f"stalled={second}")

    # --- a third one, for the host that recycles every 15-60 min ------------
    resumed3, third = attempt(store, 2900, sig, "kill 3 (48m, same checkpoint)")
    check("third long kill still resumes", resumed3 is True, f"resumed={resumed3}")
    check("still not counted as stalled", third == 0, f"stalled={third}")

    # --- a genuine wedge must still be stopped ------------------------------
    wedge = _Store()
    results = [
        attempt(wedge, 3, "4/6 stages, 0 clips", f"wedge attempt {i}")[0]
        for i in range(1, 5)
    ]
    check(
        "an instantly-repeating failure is eventually stopped",
        results[-1] is False,
        f"results={results}",
    )
    check(
        "the wedge is stopped exactly at the limit",
        results[: STALLED_RESUME_LIMIT - 1].count(True) == STALLED_RESUME_LIMIT - 1,
        f"results={results}",
    )

    print()
    if failures:
        print(f"FAILED {len(failures)}/{checks}: {failures}")
        return 1
    print(f"OK {checks}/{checks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
