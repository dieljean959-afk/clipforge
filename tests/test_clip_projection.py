"""The job's embedded clip projection must follow its clip documents."""

from __future__ import annotations

from app import invariants
from app.config import Paths
from app.store import LocalStore


def _store(tmp_path) -> LocalStore:
    paths = Paths(
        downloads_dir=tmp_path / "downloads",
        outputs_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        static_dir=tmp_path / "static",
    )
    store = LocalStore(paths)
    store.write_job(
        {
            "_id": "job1",
            "status": "running",
            "stage": "detecting",
            "requestedClipTarget": 10,
            "clipTarget": 10,
        }
    )
    return store


def _candidates(*starts: int) -> list[dict]:
    return [
        {"start": float(s), "end": float(s + 10), "order": i, "score": 80.0}
        for i, s in enumerate(starts)
    ]


def test_a_second_detection_replaces_the_projection(tmp_path):
    store = _store(tmp_path)
    first = store.create_clips("job1", _candidates(0, 20))
    first_ids = set(first["clipIds"])
    assert {c["_id"] for c in store.get_job("job1")["clips"]} == first_ids

    second = store.create_clips("job1", _candidates(100, 120))
    second_ids = set(second["clipIds"])
    job = store.get_job("job1")

    assert {c["_id"] for c in job["clips"]} == second_ids
    assert job["clipTarget"] == 10  # the user's requested count, not 2 findings
    assert job["candidateTarget"] == 2
    # Deleting a current document is recoverable, but the PREVIOUS pass must
    # not come back from the stale projection.
    store.clip_path(next(iter(second_ids))).unlink()
    restored = invariants.restore_clip_docs(store, job)
    assert restored == 1
    assert {c["_id"] for c in store.list_clip_docs("job1")} <= second_ids


def test_sync_drops_unused_docs_from_the_projection(tmp_path):
    store = _store(tmp_path)
    created = store.create_clips("job1", _candidates(0, 20, 40))
    ids = created["clipIds"]
    keep, drop = ids[0], ids[1]

    store.sync_clip_docs(
        "job1",
        [{"clipId": keep, "start": 0.0, "end": 10.0, "order": 0}],
    )
    job = store.get_job("job1")

    assert [c["_id"] for c in job["clips"]] == [keep]
    assert job["clipTarget"] == 10
    assert store.clip_path(drop).exists() is False
    # The repair pass must not resurrect the document sync just deleted.
    assert invariants.restore_clip_docs(store, job) == 0
    assert store.clip_path(drop).exists() is False
