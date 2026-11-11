"""Deleting a batch is a background purge that must stay inside its batch.

The delete used to run entirely inside the request (worker cancel over a unix
socket, record delete, full media walk, bucket listing). On this host that
request could outlive the proxy timeout, the edge answered with its own HTML
502 page, and the batch was still there. So the endpoint now marks the batch
`deleting` and answers 202, and `app/purge.py` removes the bytes in small
steps.

What these tests pin down is the part that would be a disaster if wrong:

  * the mark deletes NOTHING (the 202 is not a claim that it is gone);
  * a purge never touches another batch's files;
  * a job id that tries to escape the media folder is refused;
  * a run interrupted halfway resumes, and re-running it is harmless;
  * a step that fails leaves the batch visible in `deleting` with the reason,
    instead of half-deleted and invisible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import purge
from app.config import Paths
from app.store import LocalStore


def _paths(tmp_path: Path) -> Paths:
    return Paths(
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs",
        downloads_dir=tmp_path / "downloads",
        static_dir=tmp_path / "static",
    )


def _mkjob(store: LocalStore, paths: Paths, job_id: str, *, files: int = 3) -> list[Path]:
    """A finished batch with work, model caches, a source and rendered clips."""
    store.write_job(
        {"_id": job_id, "source": "s", "title": "t", "stage": "complete", "clipCount": files}
    )
    made: list[Path] = []
    for base in (
        paths.data_dir / "work" / job_id,
        paths.data_dir / "models" / job_id,
        paths.data_dir / "layers" / job_id,
        paths.downloads_dir / job_id,
        paths.outputs_dir / job_id,
    ):
        base.mkdir(parents=True, exist_ok=True)
        for i in range(files):
            p = base / f"{base.name}_{i}.bin"
            p.write_bytes(b"x" * 16)
            made.append(p)
    clip = paths.data_dir / "clips" / f"{job_id}-clip0.json"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_text(json.dumps({"_id": f"{job_id}-clip0", "jobId": job_id, "order": 0}))
    made.append(clip)
    return made


def _run(store: LocalStore, paths: Paths, *, cancel=None, remote=None) -> int:
    """Tick until nothing is left to purge (the loop's own behaviour)."""
    for _ in range(400):
        if not purge.tick(store, paths, cancel=cancel, remote=remote):
            break
    return 0


def test_the_202_mark_deletes_nothing(tmp_path):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    files = _mkjob(store, paths, "a" * 32)

    marked = purge.mark(store, "a" * 32, keep_source=False)

    assert marked is not None
    assert marked["stage"] == "deleting"
    assert marked["status"] == "deleting"
    assert all(p.exists() for p in files), "the mark must not unlink a single file"
    assert purge.pending(store) == ["a" * 32]


def test_a_purge_never_touches_another_batch(tmp_path):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    mine = _mkjob(store, paths, "a" * 32)
    theirs = _mkjob(store, paths, "b" * 32)

    purge.mark(store, "a" * 32)
    _run(store, paths)

    assert not any(p.exists() for p in mine), "the deleted batch is gone"
    assert all(p.exists() for p in theirs), "the other batch keeps every file"
    assert store.get_job("a" * 32) == {}
    assert store.get_job("b" * 32), "the other batch's record survives"


def test_a_job_id_that_escapes_the_media_folder_is_refused(tmp_path):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive")

    with pytest.raises(ValueError):
        purge.mark(store, "../../outside")

    roots = purge.media_roots(paths)
    assert purge.safe_target(paths.data_dir / "work" / ".." / ".." / "outside", roots, "../outside") is False
    assert purge.targets(paths, "../../evil", keep_source=False) == []

    # A document that somehow carries a traversal id (a hand-edited file, an
    # old build) must not remove anything outside the media folder either.
    store.write_job(
        {"_id": "../../outside", "stage": "deleting", "deleting": {"phase": "files"}}
    )
    purge.tick(store, paths)
    assert outside.read_text() == "must survive"
    assert store.get_job("../../outside"), "the unsafe record is left untouched for a human"


def test_a_half_finished_purge_resumes_and_is_idempotent(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    _mkjob(store, paths, "c" * 32, files=4)
    purge.mark(store, "c" * 32)

    purge.tick(store, paths)  # cancel -> files, no unlink yet
    monkeypatch.setattr(purge, "FILES_PER_TICK", 2)
    first = purge.tick(store, paths)
    assert first == 1
    job = store.get_job("c" * 32)
    assert job["stage"] == "deleting"
    assert job["deleting"]["removed"] == 2, "progress is checkpointed on the batch"
    assert store.get_job("c" * 32)["deleting"]["phase"] == "files"

    _run(store, paths)  # macro: resumes from the checkpoint
    assert store.get_job("c" * 32) == {}

    # Running it again after the batch is gone is a no-op, not a crash.
    assert purge.tick(store, paths) == 0


def test_a_failure_is_visible_on_the_batch_and_retries(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    made = _mkjob(store, paths, "d" * 32)
    purge.mark(store, "d" * 32)

    real_unlink = purge.os.unlink
    # The purge walks in sorted order, so this is the first file it tries
    # (layers/ sorts before models/, work/, downloads/ and outputs/).
    assert made
    victim = sorted(purge._iter_files(paths.data_dir / "layers" / ("d" * 32)))[0]

    def explode(path, *a, **kw):
        if str(path) == str(victim):
            raise OSError(13, "Permission denied", str(path))
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(purge.os, "unlink", explode)
    purge.tick(store, paths)  # cancel -> files
    purge.tick(store, paths)  # first unlink attempt fails

    job = store.get_job("d" * 32)
    assert job is not None, "a failed purge must never leave an invisible batch"
    assert job["stage"] == "deleting"
    assert "Permission denied" in (job["deleting"]["error"] or "")
    assert job["deleting"]["attempts"] == 1
    assert job["deleting"]["nextAttemptAt"] > 0, "the retry is scheduled, not spun"

    # The next tick is inside the backoff window and must not hammer the disk.
    assert purge.tick(store, paths) == 0

    monkeypatch.setattr(purge.os, "unlink", real_unlink)
    store.update_job("d" * 32, lambda j: j["deleting"].update(nextAttemptAt=0))
    _run(store, paths)
    assert store.get_job("d" * 32) == {}


def test_objects_and_records_are_last_and_only_once(tmp_path):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    _mkjob(store, paths, "e" * 32)
    purge.mark(store, "e" * 32)

    calls: list[tuple[str, bool]] = []

    def remote(job_id):
        # By the time the bucket is touched, no file may be left on disk.
        leftovers = [p for p in paths.outputs_dir.joinpath(job_id).rglob("*") if p.is_file()]
        calls.append((job_id, not leftovers))

    _run(store, paths, remote=remote)

    assert calls == [("e" * 32, True)]
    assert store.get_job("e" * 32) == {}


def test_a_dead_worker_cannot_block_a_delete(tmp_path):
    """Deleting a stuck batch must not need a live worker.

    `worker.cancel` is a unix-socket RPC. When the daemon is down it answers
    "absent"; when it is wedged it can raise or block. Either way the purge has
    to keep going, because the batch's record and files are all this process
    needs — that is the whole point of moving the work off the request.
    """
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    _mkjob(store, paths, "h" * 32, files=2)
    store.update_job(
        "h" * 32,
        lambda j: j.update(
            stage="failed", error="Worker process exited unexpectedly (code 0)"
        ),
    )
    purge.mark(store, "h" * 32)

    def dead_worker(job_id):
        raise OSError("worker daemon unreachable")

    _run(store, paths, cancel=dead_worker)
    assert store.get_job("h" * 32) == {}, "a dead worker must not keep the batch alive"


def test_a_failing_bucket_purge_keeps_the_batch_until_it_works(tmp_path):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    _mkjob(store, paths, "f" * 32)
    purge.mark(store, "f" * 32)

    def broken(job_id):
        raise RuntimeError("bucket unreachable")

    purge.tick(store, paths, remote=broken)  # cancel
    purge.tick(store, paths, remote=broken)  # files
    purge.tick(store, paths, remote=broken)  # first file batch
    for _ in range(50):
        job = store.get_job("f" * 32)
        if "bucket unreachable" in str(job.get("deleting", {}).get("error") or ""):
            break
        store.update_job("f" * 32, lambda j: j["deleting"].update(nextAttemptAt=0))
        purge.tick(store, paths, remote=broken)
    job = store.get_job("f" * 32)
    assert job is not None and job["stage"] == "deleting"
    assert "bucket unreachable" in (job["deleting"]["error"] or "")

    store.update_job("f" * 32, lambda j: j["deleting"].update(nextAttemptAt=0))
    _run(store, paths, remote=lambda job_id: None)
    assert store.get_job("f" * 32) == {}


def test_keep_source_leaves_the_video_behind(tmp_path):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    _mkjob(store, paths, "g" * 32)
    source = paths.downloads_dir / ("g" * 32) / "source.mp4"
    source.write_bytes(b"video")

    purge.mark(store, "g" * 32, keep_source=True)
    _run(store, paths)

    assert source.exists(), "keepSource is a promise about the source file"
    assert not (paths.outputs_dir / ("g" * 32)).exists()
    assert store.get_job("g" * 32) == {}


# --- the mass path must not delete work in flight -------------------------
#
# "Clean up dead batches" asks the server for junk, then sends every id it did
# not classify as junk. That list included a batch that was still grabbing: it
# was marked `deleting`, its worker was cancelled, and 35 files went with it
# while the batch was mid-capture. A bulk delete now refuses anything that has
# not settled; a single, explicitly confirmed delete keeps its old behaviour.


def test_bulk_mark_refuses_a_batch_that_is_still_running(tmp_path):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    job_id = "still-grabbing"
    store.write_job({"_id": job_id, "status": "queued", "stage": "grabbing", "source": "s"})
    audio = paths.data_dir / "work" / job_id / "transcribe" / "audio.m4a"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"being decoded")

    with pytest.raises(purge.JobNotDeletable) as caught:
        purge.mark(store, job_id, require_finished=True)

    assert "still grabbing" in str(caught.value)
    job = store.get_job(job_id)
    assert job["status"] == "queued" and job.get("deleting") is None
    assert audio.exists(), "the refused delete still touched the batch's files"


@pytest.mark.parametrize("status", ["running", "transcribing", "paused", "queued", "grabbing"])
def test_every_unsettled_status_is_refused(tmp_path, status):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    store.write_job({"_id": "j", "status": status, "stage": status})
    with pytest.raises(purge.JobNotDeletable):
        purge.mark(store, "j", require_finished=True)


def test_bulk_mark_still_deletes_finished_batches(tmp_path):
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    for job_id, status in (("done", "complete"), ("dead", "failed")):
        store.write_job({"_id": job_id, "status": status, "stage": status, "source": "s"})
        assert purge.mark(store, job_id, require_finished=True) is not None
        assert store.get_job(job_id)["deleting"]["phase"] == "cancel"


def test_the_single_delete_still_takes_a_running_batch(tmp_path):
    """The per-card delete is explicit: the user pressed it on that batch."""
    paths = _paths(tmp_path)
    store = LocalStore(paths)
    store.write_job({"_id": "j", "status": "running", "stage": "transcribing", "source": "s"})
    assert purge.mark(store, "j") is not None
    assert store.get_job("j")["status"] == "deleting"
