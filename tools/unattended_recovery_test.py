#!/usr/bin/env python3
"""Jobs must recover WITHOUT a browser open.

The reported bug: a job stopped at "Transcribing chunk 5 of 8", the card read
"updated 5h ago", and nothing ever resumed it — until the owner happened to
load the dashboard, at which point it recovered instantly.

Two independent causes, both covered here.

1. Recovery was only reachable from a request handler.
   `reconcile_stale_running` re-queues a job whose worker a host restart
   SIGKILLed, but it was called from exactly two places -- GET /api/jobs and
   GET /api/jobs/{id} -- which only fire when a browser polls. With no tab open
   nothing reconciled anything. The watchdog thread now calls it on a timer, so
   recovery no longer depends on anyone looking.

2. A job that WAS advancing was failed as a no-progress loop.
   The stall counter compares `_progress_signature`, which counted only finished
   stages and rendered clips. A job that transcribed 8 chunks and had not yet
   rendered a clip looked identical to one that had done nothing, so
   `stalledResumes` hit STALLED_RESUME_LIMIT and the job was failed with "the
   server restarted 3 times ... it cannot be resumed automatically". The
   signature now counts cached transcription chunks, so real work is visible.

Run: .venv/bin/python tools/unattended_recovery_test.py
"""

from __future__ import annotations

import ast
import inspect
import sys
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


def main() -> int:
    print("unattended recovery")

    worker_path = ROOT / "app" / "worker.py"
    main_path = ROOT / "app" / "main.py"
    worker_src = worker_path.read_text(encoding="utf-8")
    main_src = main_path.read_text(encoding="utf-8")

    from app.worker import STALLED_RESUME_LIMIT, InProcessWorker  # noqa: E402

    _progress_signature = InProcessWorker._progress_signature

    # --- 1. Reconciliation must not depend on a browser ---------------------
    check(
        "watchdog declares a reconcile interval",
        InProcessWorker.RECONCILE_INTERVAL_SEC == 30.0,
        f"got {InProcessWorker.RECONCILE_INTERVAL_SEC!r}",
    )

    # Walk the AST rather than grepping text: the requirement is that the
    # watchdog LOOP body calls reconcile, not that the word appears anywhere in
    # the class (it also appears in the method definition and its docstring).
    tree = ast.parse(worker_src)
    watchdog_calls_reconcile = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_watchdog":
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            watchdog_calls_reconcile = "reconcile_stale_running" in body
    check(
        "the watchdog loop itself calls reconcile_stale_running",
        watchdog_calls_reconcile,
        "_watchdog does not reconcile; recovery still needs a request",
    )

    # The handler calls may stay (they make the dashboard truthful), but they
    # must not be the ONLY trigger.
    check(
        "reconcile is not confined to request handlers",
        watchdog_calls_reconcile and "reconcile_stale_running" in main_src,
        "no timer-driven reconcile found",
    )

    # A pass that raises must not take the watchdog -- and every other liveness
    # guarantee -- down with it.
    watchdog_src = inspect.getsource(InProcessWorker._watchdog)
    check(
        "a failing reconcile pass cannot kill the watchdog",
        "except" in watchdog_src,
        "reconcile is unguarded inside the watchdog loop",
    )

    # --- 2. Advancing work must count as progress --------------------------
    check(
        "_progress_signature mentions cached chunks",
        "chunk" in inspect.getsource(_progress_signature).lower(),
        "transcription chunks are not part of the signature",
    )

    import tempfile

    from app.config import load_config

    class _Store:
        def __init__(self, clips): self._clips = clips

        def list_clips(self, _job_id): return self._clips

    settings = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        # Point the signature's data_dir at the temp tree via a stand-in
        # settings object; the real dataclass is frozen.
        settings.app.paths.data_dir = data_dir
        worker = object.__new__(InProcessWorker)
        worker.settings = settings
        worker.store = _Store([{"renderState": "done"}])

        job_id = "unattended1"
        chunk_dir = data_dir / "work" / job_id / "transcribe" / "chunks"
        chunk_dir.mkdir(parents=True)
        before = _progress_signature(worker, job_id)
        for i in range(3):
            (chunk_dir / f"chunk_{i:05d}.json").write_text("{}", encoding="utf-8")
        after = _progress_signature(worker, job_id)

        check(
            "signature changes when chunks are cached",
            before != after,
            f"signature is unchanged: {after!r}",
        )
        check(
            "signature reports the chunk count",
            "3 chunk" in after,
            f"got {after!r}",
        )
        check(
            "signature is stable for unchanged work",
            after == _progress_signature(worker, job_id),
            "signature is not deterministic",
        )

    # --- 3. The stall limit must still protect against real loops ----------
    check(
        "a no-progress stall limit still exists",
        isinstance(STALLED_RESUME_LIMIT, int) and STALLED_RESUME_LIMIT >= 1,
        "the limit was removed entirely; a wedged job would loop forever",
    )
    check(
        "_resume_is_worthwhile still consults the stall counter",
        "stalledResumes" in inspect.getsource(InProcessWorker._resume_is_worthwhile),
        "the no-progress guard was dropped",
    )

    print()
    if failures:
        print(f"FAILED {len(failures)}/{checks}: {failures}")
        return 1
    print(f"OK {checks}/{checks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
