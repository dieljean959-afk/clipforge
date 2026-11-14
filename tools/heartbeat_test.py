"""The liveness heartbeat must prove life, and must never fake progress.

Two failure modes matter and they pull in opposite directions:

  * too quiet  -- a long ffmpeg leaves `lastProgressAt` stale, the UI says "no
    recent activity", and the watchdog kills a render that was working.
  * too loud   -- a heartbeat that keeps ticking after the work it was
    covering has died turns a real stall into an eternal "working" job that
    never fails and never recovers.

So: it bumps liveness during the block, stops on the exception path, touches
`lastProgressAt` and NOTHING else, and works on a job that does not exist.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import task as task_mod  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print(f"\n[{title}]")


class FakeStore:
    """Just enough store for the heartbeat: a job document and a write log."""

    def __init__(self) -> None:
        self.job = {
            "_id": "hbtest",
            "stage": "rendering",
            "progress": 40,
            "lastProgressAt": 1_000_000,
        }
        self.patches: list[dict] = []

    def get_job(self, job_id: str):
        return dict(self.job) if self.job else None

    def write_job(self, job) -> None:
        self.job = dict(job)


def make_pipeline(store: FakeStore):
    """A Pipeline with only the attributes _heartbeat touches.

    `_patch_job` is the real write path the heartbeat uses, so the stub counts
    PATCHES rather than whole-document writes -- that is the unit the heartbeat
    actually produces.
    """
    pipe = task_mod.Pipeline.__new__(task_mod.Pipeline)
    pipe.store = store

    def patch(job_id: str, **fields):
        store.patches.append(dict(fields))
        if store.job:
            store.job.update(fields)
        return store.job

    pipe._patch_job = patch
    return pipe


def test_beats_while_working() -> None:
    section("a heartbeat proves liveness during a long, silent step")
    store = FakeStore()
    pipe = make_pipeline(store)
    before = store.job["lastProgressAt"]

    # A real 5s beat would make this test slow; shorten the interval so the
    # behaviour is exercised for real rather than mocked away.
    original = task_mod.HEARTBEAT_SEC
    task_mod.HEARTBEAT_SEC = 0.05
    try:
        with pipe._heartbeat("hbtest", "render clip 1"):
            deadline = time.monotonic() + 0.4
            while time.monotonic() < deadline:
                time.sleep(0.02)
    finally:
        task_mod.HEARTBEAT_SEC = original

    after = store.job["lastProgressAt"]
    check("lastProgressAt advances while the block is open", after > before,
          f"{before} -> {after}")
    check("it beat several times, not once", len(store.patches) >= 2,
          f"{len(store.patches)} patches")
    check("every beat carries ONLY the liveness timestamp",
          all(set(p) == {"lastProgressAt"} for p in store.patches),
          str([sorted(p) for p in store.patches[:3]]))


def test_stops_on_failure() -> None:
    section("it stops when the step fails, so a real stall still fails")
    store = FakeStore()
    pipe = make_pipeline(store)

    original = task_mod.HEARTBEAT_SEC
    task_mod.HEARTBEAT_SEC = 0.05
    try:
        raised = False
        try:
            with pipe._heartbeat("hbtest", "render clip 2"):
                time.sleep(0.12)
                raise video_render_error()
        except RuntimeError:
            raised = True
        check("the exception still propagates", raised)
        writes_at_exit = len(store.patches)
        time.sleep(0.25)  # several intervals: the thread must be gone
    finally:
        task_mod.HEARTBEAT_SEC = original

    check("no beats arrive after the block exits",
          len(store.patches) == writes_at_exit,
          f"{len(store.patches) - writes_at_exit} late patches")
    check("no heartbeat thread is left running",
          not any(t.name.startswith("heartbeat-hbtest") and t.is_alive()
                  for t in threading.enumerate()))


def video_render_error() -> RuntimeError:
    return RuntimeError("simulated render failure")


def test_never_invents_progress() -> None:
    section("it reports liveness only, never progress")
    store = FakeStore()
    pipe = make_pipeline(store)
    original_progress = store.job["progress"]

    original = task_mod.HEARTBEAT_SEC
    task_mod.HEARTBEAT_SEC = 0.05
    try:
        with pipe._heartbeat("hbtest", "transcribe"):
            time.sleep(0.2)
    finally:
        task_mod.HEARTBEAT_SEC = original

    check("the percentage never moves on its own",
          store.job["progress"] == original_progress,
          f"{original_progress} -> {store.job['progress']}")
    check("no ETA is invented",
          store.job.get("stageEtaSec") is None
          and store.job.get("totalEtaSec") is None)
    check("the stage is never changed by a heartbeat",
          store.job["stage"] == "rendering")


def test_tolerates_a_missing_job() -> None:
    section("a heartbeat on a deleted job must not raise into the render")
    store = FakeStore()
    store.job = None
    pipe = make_pipeline(store)

    original = task_mod.HEARTBEAT_SEC
    task_mod.HEARTBEAT_SEC = 0.05
    try:
        with pipe._heartbeat("gone", "render clip 9"):
            time.sleep(0.2)
        check("the block completes cleanly with no job on disk", True)
    except Exception as exc:
        check("the block completes cleanly with no job on disk", False, str(exc))
    finally:
        task_mod.HEARTBEAT_SEC = original


def test_wired_into_the_long_steps() -> None:
    section("it actually wraps the steps that go silent")
    src = (ROOT / "app" / "task.py").read_text(encoding="utf-8")
    check("the render encode is wrapped", "with self._heartbeat(job_id, f\"render clip" in src)
    check("transcription is wrapped", 'with self._heartbeat(job_id, "transcribe")' in src)
    check("the interval is the requested 5s",
          "HEARTBEAT_SEC = 5.0" in src)
    check("it is inside a try/finally, so it always stops",
          "finally:\n            stop.set()" in src)


def main() -> int:
    test_beats_while_working()
    test_stops_on_failure()
    test_never_invents_progress()
    test_tolerates_a_missing_job()
    test_wired_into_the_long_steps()
    print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
