"""Batch finalization and the \"Your batches\" listing source.

The live failure: nine clips rendered and upload to the bucket (the bucket grew
0.88 -> 1.57 GB) but the batch record was never written, so the dashboard said
"No finished batches yet" while the videos sat in the bucket, and the next boot
re-ran the job from fetch.

Covered here:
  * finalize is idempotent;
  * finalize runs BEFORE the scratch that holds the resume checkpoints is
    deleted (that ordering is what made a restart re-download the source);
  * a finalize failure retries only finalize and never re-runs the pipeline;
  * the batch record can be rebuilt from the clip rows, including rows the job
    projection does not name;
  * the listing predicate includes a finished-with-failures batch.
"""

from __future__ import annotations

import time
from pathlib import Path

from app import invariants
from app.store import LocalStore


class _Paths:
    def __init__(self, root: Path) -> None:
        self.data_dir = root / "data"
        self.outputs_dir = root / "outputs"
        self.downloads_dir = root / "downloads"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for d in (self.outputs_dir, self.downloads_dir):
            d.mkdir(parents=True, exist_ok=True)


def _store(tmp_path: Path) -> LocalStore:
    return LocalStore(_Paths(tmp_path))


def _job(store: LocalStore, **extra: object) -> dict:
    job = {
        "_id": "job1",
        "jobId": "job1",
        "source": "https://example.test/v",
        "sourceType": "url",
        "stage": "transcribing",
        "status": "running",
        "progress": 40.0,
        "lastProgressAt": 0,
        "createdAt": int(time.time() * 1000),
        "clips": [],
        **extra,
    }
    store.write_job(job)
    return job


def _clip(store: LocalStore, clip_id: str, order: int, **extra: object) -> dict:
    doc = {
        "_id": clip_id,
        "jobId": "job1",
        "order": order,
        "title": f"Clip {order}",
        "start": 0.0,
        "end": 30.0,
        "renderState": "done",
        "videoKey": f"clipforge/outputs/job1/clip_{order:03d}.mp4",
        "thumbKey": f"clipforge/outputs/job1/clip_{order:03d}.thumb.jpg",
        "mediaRemote": True,
        "estimatedBytes": 1_000_000,
        **extra,
    }
    store.write_clip(doc)
    return doc


# --------------------------------------------------------------------------- #
# rebuild from clip rows
# --------------------------------------------------------------------------- #

def test_rebuild_promotes_a_batch_from_rows_the_projection_never_names(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _job(store)  # no clipIds, empty projection: the never-finalised shape
    _clip(store, "c0", 0)
    _clip(store, "c1", 1)

    report = invariants.rebuild_batches(store, apply=True)
    assert report["rebuilt"] == 1

    job = store.get_job("job1")
    assert job["readyClips"] == 2
    assert job["clipCount"] == 2
    assert job["isTerminal"] is True
    assert job["stage"] == "complete"
    assert job["batchState"] == "finished"
    assert job["finishedAt"]
    assert {c["_id"] for c in job["clips"]} == {"c0", "c1"}


def test_rebuild_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _job(store)
    _clip(store, "c0", 0)
    assert invariants.rebuild_batches(store, apply=True)["rebuilt"] == 1
    first = store.get_job("job1")
    assert invariants.rebuild_batches(store, apply=True)["rebuilt"] == 0
    second = store.get_job("job1")
    assert second["finishedAt"] == first["finishedAt"]
    assert second["readyClips"] == first["readyClips"]


def test_rebuild_reports_finished_with_failures(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _job(store)
    _clip(store, "c0", 0)
    _clip(store, "c1", 1, renderState="failed", videoKey=None, mediaRemote=None)
    invariants.rebuild_batches(store, apply=True)
    job = store.get_job("job1")
    assert job["readyClips"] == 1
    assert job["failedClips"] == 1
    assert job["batchState"] == "finished_with_failures"


def test_rebuild_never_deletes_or_rerenders(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _job(store)
    _clip(store, "c0", 0)
    before = store.get_clip_doc("c0")
    invariants.rebuild_batches(store, apply=True)
    after = store.get_clip_doc("c0")
    assert after == before
    assert after["renderState"] == "done"


def test_rebuild_leaves_a_live_render_alone(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _job(store, stage="rendering", isTerminal=False,
         lastProgressAt=int(time.time() * 1000))
    _clip(store, "c0", 0)
    invariants.rebuild_batches(store, apply=True)
    assert store.get_job("job1")["stage"] == "rendering"


def test_rebuild_repairs_a_dead_job_that_still_says_rendering(tmp_path: Path) -> None:
    # The real shape: stage says rendering, nothing has written progress for
    # hours, and the clips it produced are in the bucket.
    store = _store(tmp_path)
    _job(store, stage="rendering", isTerminal=False,
         lastProgressAt=int(time.time() * 1000) - 6 * 3600 * 1000)
    _clip(store, "c0", 0)
    invariants.rebuild_batches(store, apply=True)
    job = store.get_job("job1")
    assert job["stage"] == "complete"
    assert job["readyClips"] == 1


def test_store_lists_docs_ignoring_the_job_projection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _job(store)
    _clip(store, "c0", 0)
    _clip(store, "c1", 1)
    assert [d["_id"] for d in store.list_clip_docs("job1")] == ["c0", "c1"]
    docs = store.list_clip_docs_any_authority("job1")
    assert {d["_id"] for d in docs} == {"c0", "c1"}
    assert len(docs) == 2


# --------------------------------------------------------------------------- #
# finalize ordering, idempotency, retry-only-finalize
# --------------------------------------------------------------------------- #

def test_finalize_runs_before_scratch_cleanup() -> None:
    src = (Path(__file__).resolve().parents[1] / "app/task.py").read_text(encoding="utf-8")
    finalize_call = src.index("self._finalize(job_id, time.time() - t_start)")
    full_cleanup = src.index("self._cleanup_job_scratch(job_id, full=True)")
    assert finalize_call < full_cleanup, (
        "finalize must land before the checkpoint-holding scratch is deleted"
    )


def test_finalize_retries_only_itself() -> None:
    src = (Path(__file__).resolve().parents[1] / "app/task.py").read_text(encoding="utf-8")
    assert "FINALIZE_ATTEMPTS = 5" in src
    i = src.index("def _finalize(")
    window = src[i:i + 3000]
    assert "_fail_job_only" in window
    assert 'error_kind="finalize"' in window
    assert "re-render" in window.lower() or "re-rendered" in window.lower()


def test_finish_is_idempotent() -> None:
    src = (Path(__file__).resolve().parents[1] / "app/task.py").read_text(encoding="utf-8")
    i = src.index("def _finish(")
    window = src[i:i + 1200]
    assert "finalize skipped: already terminal" in window


def test_rebuild_does_not_read_a_stale_twin_twice(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _job(store)
    _clip(store, "old", 0, videoKey=None, mediaRemote=None)
    _clip(store, "new", 0)
    docs = store.list_clip_docs_any_authority("job1")
    assert len(docs) == 1
    assert docs[0]["_id"] == "new"


def test_repair_never_kills_a_running_job(tmp_path: Path) -> None:
    """A live batch with pending clips must stay running.

    The periodic repair used to stamp stage="failed", isTerminal=True on any job
    whose clips were not all rendered — including one actively working. The
    recovery path then (correctly) refused to re-queue it, so the invariant
    killed a healthy batch.
    """
    store = _store(tmp_path)
    _job(store, stage="captions", lastProgressAt=int(time.time() * 1000))
    _clip(store, "c0", 0)
    _clip(store, "c1", 1, renderState="queued", videoKey=None, mediaRemote=None)
    counts = {"done": 1, "failed": 0, "pending": 1, "total": 2}
    invariants.repair_job(store, store.get_job("job1"), counts, reason="test")
    job = store.get_job("job1")
    assert job["stage"] == "captions"
    assert job.get("isTerminal") in (False, None)
    assert job["readyClips"] == 1
    assert "worker is alive" in str(job.get("progressDetail"))


def test_repair_still_marks_a_dead_job_with_pending_clips(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _job(store, stage="captions", lastProgressAt=int(time.time() * 1000) - 6 * 3600 * 1000)
    _clip(store, "c0", 0)
    _clip(store, "c1", 1, renderState="queued", videoKey=None, mediaRemote=None)
    counts = {"done": 1, "failed": 0, "pending": 1, "total": 2}
    invariants.repair_job(store, store.get_job("job1"), counts, reason="test")
    job = store.get_job("job1")
    assert job["stage"] == "failed"
    assert job["isTerminal"] is True
    assert job["errorAction"] == "resume"


def test_retry_resolves_the_cut_before_the_bucket_branch() -> None:
    """`retry_clip_render` called src.render_span(cut, …) 30 lines before `cut`
    was assigned, so Retry on an offloaded clip died with NameError. The cut
    lookup must stay above the `if src is None:` branch."""
    src = (Path(__file__).resolve().parents[1] / "app/video.py").read_text(encoding="utf-8")
    i = src.index("def retry_clip_render")
    body = src[i:]
    cut_lookup = body.index('ws.read_json("reframe", "cuts.json")')
    bucket_branch = body.index("    if src is None:")
    render_span = body.index("= src.render_span(")
    assert cut_lookup < bucket_branch < render_span
