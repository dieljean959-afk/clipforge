"""Regression tests for the failures this pass fixed.

Each test names the observed symptom, so a future change that reintroduces it
fails with the reason instead of a bare assertion.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app import boot as bootmod
from app import diskwatch, exits, invariants
from app.config import load_config
from app.store import LocalStore


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPFORGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLIPFORGE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("CLIPFORGE_DOWNLOADS_DIR", str(tmp_path / "downloads"))
    settings = load_config(quiet=True)
    for p in (settings.app.paths.data_dir, settings.app.paths.outputs_dir,
              settings.app.paths.downloads_dir):
        p.mkdir(parents=True, exist_ok=True)
    return settings, LocalStore(settings.app.paths)


def _job(store, *, stage="complete", **extra):
    job = store.create_job(source_type="url", source="https://example.invalid/x")
    job.update({"stage": stage, "status": stage, **extra})
    store.write_job(job)
    return job


def _clip(store, job_id, order, *, state="done", video=True, start=0.0, end=30.0):
    cid = f"clip{order:02d}" + "0" * 26
    store.write_clip(
        cid,
        {
            "jobId": job_id,
            "order": order,
            "start": start,
            "end": end,
            "renderState": state,
            "videoPath": f"/tmp/{cid}.mp4" if video else None,
        },
        job_id=job_id,
    )
    return cid


# --------------------------------------------------------------------------- #
# the 93.4 MB disk self-check failure
# --------------------------------------------------------------------------- #

def test_global_disk_delta_is_not_a_verdict(env, tmp_path):
    """A batch whose clips are delivered must not fail because free space moved.

    The observed failure: "left 93.4 MB ... tolerance 50 MB" while all 14 clips
    were verified and in the bucket. The delta belongs to logs/model caches, not
    to the render.
    """
    settings, store = env
    job = _job(store, clipCount=14)
    ledger = diskwatch.JobLedger("j", [tmp_path / "nope"], data_dir=settings.app.paths.data_dir)
    ledger.start_free = diskwatch.free_bytes(settings.app.paths.data_dir)
    ledger.start_free += 93 * 1024 * 1024  # pretend the box had 93 MB more
    ok, why = ledger.global_delta()
    assert ok is False, "an over-tolerance delta is still reported"
    assert "informational" in why
    # ... but it is not the hard check: that one only looks at the job's scratch.
    scratch_ok, _ = ledger.scratch_check()
    assert scratch_ok is True
    assert job["clipCount"] == 14


def test_scratch_check_names_the_leftover(tmp_path):
    scratch = tmp_path / "work" / "job"
    scratch.mkdir(parents=True)
    (scratch / "big.bin").write_bytes(b"x" * 4096)
    ledger = diskwatch.JobLedger("j", [scratch], data_dir=tmp_path)
    ok, why = ledger.scratch_check()
    assert ok is False and str(scratch) in why
    diskwatch.cleanup([scratch])
    ok, why = ledger.scratch_check()
    assert ok is True and "gone" in why


def test_disk_full_is_retry_after_cleanup_not_fail_fast():
    verdict = exits.classify(None, stage="exporting", disk_full=True)
    assert verdict.kind == exits.DISK_FULL
    assert verdict.action == exits.RETRY_AFTER_CLEANUP
    assert verdict.retryable is True
    assert exits.is_disk_full(OSError(28, "No space left on device"))
    assert exits.is_disk_full("ffmpeg: No space left on device")


def test_signal_conventions_are_one_event():
    assert exits.classify(143).kind == exits.classify(-15).kind == exits.EXTERNAL_INTERRUPTION
    assert exits.classify(137).kind == exits.classify(-9).kind == exits.EXTERNAL_KILL
    assert exits.classify(139).kind == exits.NATIVE_CRASH   # SIGSEGV
    assert exits.classify(134).kind == exits.NATIVE_CRASH   # SIGABRT
    assert exits.classify(0).kind == exits.INCOMPLETE_EXIT


# --------------------------------------------------------------------------- #
# atomic / consistent state
# --------------------------------------------------------------------------- #

def test_a_delivered_batch_cannot_be_saved_as_an_empty_failure(env):
    """clips exist, record says clips=[] progress=0 isTerminal=false → repaired."""
    settings, store = env
    job = _job(store, stage="failed", clipCount=0, progress=0, isTerminal=False,
               errorKind="fail_fast")
    for i in range(14):
        _clip(store, job["_id"], i)
    _clip(store, job["_id"], 14, state="failed", video=False)

    report = invariants.check_and_repair(store, settings.app.paths)
    assert report["repaired"] == 1
    fixed = store.get_job(job["_id"])
    assert fixed["stage"] == "complete"
    assert fixed["isTerminal"] is True
    assert fixed["clipCount"] == 14
    assert "14 clips ready, 1 failed" in fixed["progressDetail"]
    assert fixed["errorKind"] is None


def test_partial_success_label_is_honest():
    counts = {"done": 14, "failed": 1, "pending": 0, "total": 15}
    assert invariants.partial_success_label(counts) == "14 clips ready, 1 failed"
    assert invariants.partial_success_label(
        {"done": 0, "failed": 1, "pending": 0, "total": 1}
    ) == "1 clip failed"


def test_invariants_restore_clip_docs_from_the_projection(env):
    """A re-detect deleted the documents; the job projection is the last copy."""
    settings, store = env
    job = _job(store, stage="complete", progress=100,
               clips=[{"_id": "a" * 32, "order": 0, "renderState": "done",
                       "videoPath": "/tmp/a.mp4", "jobId": None}],
               clipIds=["a" * 32])
    assert store.get_clip_doc("a" * 32) is None
    report = invariants.check_and_repair(store, settings.app.paths)
    assert report["restoredDocs"] == 1
    assert store.get_clip_doc("a" * 32)["renderState"] == "done"


def test_a_stale_partial_warning_is_cleared_after_the_retry(env):
    settings, store = env
    job = _job(store, stage="complete", clipCount=15, progress=100, isTerminal=True,
               warning="14 clips ready, 1 failed — the failed clip can be retried")
    for i in range(15):
        _clip(store, job["_id"], i)
    invariants.check_and_repair(store, settings.app.paths)
    assert store.get_job(job["_id"]).get("warning") is None


def test_list_clip_docs_uses_the_job_not_a_stale_summary(env):
    """The batch listed 4 clips because a summary blob outvoted 15 documents."""
    settings, store = env
    job = _job(store, stage="complete")
    ids = [_clip(store, job["_id"], i) for i in range(15)]
    summary = settings.app.paths.data_dir / "clips" / f"{job['_id']}.json"
    summary.write_text(json.dumps({"jobId": job["_id"], "clipIds": ids[:4]}))
    assert len(store.list_clip_docs(job["_id"])) == 15


def test_job_state_transitions_are_atomic(tmp_path):
    """save_job is tmp+rename: a reader never sees a half-written document."""
    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)
    path = store._job_path("x" * 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"_id": "x" * 32, "stage": "queued", "big": "y" * 200_000}
    store.save_job(doc)
    assert json.loads(path.read_text())["stage"] == "queued"
    leftovers = list(path.parent.glob("*.tmp"))
    assert not leftovers, f"atomic write left temp files: {leftovers}"


# --------------------------------------------------------------------------- #
# boot readiness
# --------------------------------------------------------------------------- #

def test_boot_phases_have_timeouts_and_never_raise():
    b = bootmod.Boot()
    def boom():
        raise RuntimeError("dependency down")
    phase = b.run_phase("worker_spawn", boom, timeout_ms=500)
    assert phase.status == "failed" and "dependency down" in phase.error
    # the boot continues: a later phase still runs
    assert b.run_phase("model_preload", lambda: {"detail": "ok"}, timeout_ms=500).status == "ok"
    assert b.summary()["failedPhases"] == ["worker_spawn"]


def test_a_hung_phase_times_out_instead_of_blocking():
    b = bootmod.Boot()
    t0 = time.time()
    phase = b.run_phase("invariants", lambda: time.sleep(5), timeout_ms=200)
    assert time.time() - t0 < 2.0
    assert phase.status == "timeout"


def test_health_route_touches_no_disk(monkeypatch):
    """`/api/health` must answer from memory: 1-2 ms, never a disk walk."""
    import inspect

    from app import main

    src = inspect.getsource(main.health)
    for forbidden in ("disk_stats", "list_jobs", "objectstore", "status()"):
        assert forbidden not in src, f"health() must not call {forbidden}"


def test_instance_lock_refuses_a_live_holder_and_reclaims_a_dead_one(tmp_path):
    lock_path = tmp_path / "api.lock"
    first = bootmod.InstanceLock(lock_path, "api", pid=999001)
    assert first.acquire() is True
    second = bootmod.InstanceLock(lock_path, "api", pid=999002)
    # pid 999001 is not running → stale → reclaimed
    assert second.acquire() is True
    assert second.stale_reclaimed == 999001
    second.release()
    assert not lock_path.exists()


def test_single_instance_lock_blocks_a_second_server(tmp_path):
    import os

    lock_path = tmp_path / "api.lock"
    mine = bootmod.InstanceLock(lock_path, "api", pid=os.getpid())
    assert mine.acquire() is True
    other = bootmod.InstanceLock(lock_path, "api", pid=os.getpid() + 1)
    assert other.acquire() is False
    assert other.holder == os.getpid()
    mine.release()


def test_boot_failure_is_recorded_loudly(tmp_path):
    payload = bootmod.failure_payload(
        "config", "CLIPFORGE_PORT=abc is not a number",
        "Set a numeric port in config.toml or unset CLIPFORGE_PORT.",
    )
    bootmod.record_fatal(payload, tmp_path)
    stored = json.loads((tmp_path / "boot_failure.json").read_text())
    assert stored["cause"].startswith("CLIPFORGE_PORT")
    assert stored["fix"]
    assert bootmod.last_failure(tmp_path)["where"] == "config"
    bootmod.clear_failure(tmp_path)
    assert not (tmp_path / "boot_failure.json").exists()


def test_clip_media_never_recurses_into_the_share_copy(tmp_path, monkeypatch):
    """`/api/clips/<id>/thumb` answered 500 RecursionError for every shared clip.

    `_clip_media(clip, "video")` returned `_share_copy_for(clip)`, which asked
    `_clip_media(clip, "video")` for the master. One path, one meaning.
    """
    from app import main

    master = tmp_path / "clip_012.mp4"
    master.write_bytes(b"x")
    clip = {"_id": "c" * 32, "shareId": "s1", "videoPath": str(master),
            "thumbPath": str(tmp_path / "missing.jpg")}
    assert main._clip_media(clip, "video") == master
    assert main._clip_media(clip, "thumb") is None
    assert main._master_path(clip) == master
