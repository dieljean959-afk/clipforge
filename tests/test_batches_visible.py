"""What "Your batches" reads must match what actually rendered.

The dashboard builds the finished list from `status == "complete"` and plays a
clip through `/api/clips/<id>/video`. These tests pin that contract against the
real store and the real routes (no fake list): a completed batch survives a
fresh read of the data directory, its clip is served as playable MP4, a failed
job is never counted as finished, and re-running detection cannot duplicate a
batch's clips.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import load_config
from app.store import LocalStore

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 512


@pytest.fixture()
def env(tmp_path: Path):
    base = load_config(quiet=True)
    paths = replace(
        base.app.paths,
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs",
        downloads_dir=tmp_path / "downloads",
    )
    for p in (paths.data_dir, paths.outputs_dir, paths.downloads_dir):
        p.mkdir(parents=True, exist_ok=True)
    settings = replace(base, app=replace(base.app, paths=paths))
    return settings, LocalStore(paths)


@pytest.fixture()
def client(env, monkeypatch):
    import app.main as main_mod

    settings, store = env
    monkeypatch.setattr(main_mod, "settings", settings)
    monkeypatch.setattr(main_mod, "store", store)
    return TestClient(main_mod.app), store, settings


def _finished_batch(store: LocalStore, settings) -> tuple[str, str]:
    """A completed job with one validated clip, exactly as the pipeline writes it."""
    job = store.create_job(source_type="url", source="https://youtu.be/real-ish")
    clip_id = "c11f" + "0" * 28
    mp4 = settings.app.paths.outputs_dir / job["_id"] / "clip_000.mp4"
    mp4.parent.mkdir(parents=True, exist_ok=True)
    mp4.write_bytes(MP4)
    store.write_clip(
        clip_id,
        {
            "jobId": job["_id"],
            "order": 0,
            "start": 10.0,
            "end": 40.0,
            "title": "A real finished clip",
            "renderState": "done",
            "videoPath": str(mp4),
            "thumbPath": None,
            "mediaBytes": len(MP4),
            "mediaWidth": 1080,
            "mediaHeight": 1920,
            "mediaDurationSec": 30.0,
            "contractOk": True,
            "mediaRemote": False,
        },
        job_id=job["_id"],
    )
    job.update({
        "stage": "complete",
        "status": "complete",
        "isTerminal": True,
        "clipCount": 1,
        "progress": 100,
        "progressDetail": "All clips ready",
        "finishedAt": 1_790_000_000_000,
        "completedAt": 1_790_000_000_000,
        "clips": [store.get_clip_doc(clip_id)],
    })
    store.write_job(job)
    return job["_id"], clip_id


def test_a_completed_batch_is_listed_after_a_reload(client):
    client, store, settings = client
    job_id, _clip_id = _finished_batch(store, settings)

    first = client.get("/api/jobs")
    assert first.status_code == 200
    jobs = first.json()["jobs"]
    assert [j["_id"] for j in jobs] == [job_id]
    assert jobs[0]["status"] == "complete"
    assert jobs[0]["clipCount"] == 1

    # A reload is a new store instance reading the same directory. It must
    # still show the batch and its clip count, not "No finished batches yet".
    again = client.get("/api/jobs").json()["jobs"]
    assert again[0]["_id"] == job_id
    assert again[0]["status"] == "complete"
    assert again[0]["clipCount"] == 1


def test_the_finished_clip_is_playable_through_the_normal_route(client):
    client, store, settings = client
    job_id, clip_id = _finished_batch(store, settings)

    clips = client.get("/api/clips", params={"jobId": job_id})
    assert clips.status_code == 200
    listed = clips.json()
    assert [c["_id"] for c in listed] == [clip_id]
    assert listed[0]["renderState"] == "done"
    assert listed[0]["videoPath"]

    video = client.get(f"/api/clips/{clip_id}/video")
    assert video.status_code == 200, video.text
    assert video.headers["content-type"].startswith("video/mp4")
    assert video.content.startswith(MP4)


def test_a_failed_job_is_never_counted_as_a_finished_batch(client):
    client, store, settings = client
    failed = store.create_job(source_type="url", source="https://youtu.be/broken")
    failed.update({
        "stage": "failed",
        "status": "failed",
        "isTerminal": True,
        "error": "Worker could not start this job — only 335 MB of disk free",
        "failedStage": "captions",
        "finishedAt": 1_790_000_000_000,
        "completedAt": 1_790_000_000_000,
    })
    store.write_job(failed)

    jobs = client.get("/api/jobs").json()["jobs"]
    assert jobs[0]["status"] == "failed"
    assert [j for j in jobs if j["status"] == "complete"] == []
    assert "335 MB" in jobs[0]["error"]


def test_re_detection_does_not_duplicate_the_batch_clips(env):
    _settings, store = env
    job = store.create_job(source_type="url", source="https://youtu.be/repeat")
    clip_id = "abc1" + "0" * 28
    candidates = [{
        "_id": clip_id,
        "order": 0,
        "start": 5.0,
        "end": 35.0,
        "score": 80.0,
        "hook_text": "hook",
        "transcript": "words",
        "hashtags": ["#fyp"],
    }]

    store.create_clips(job["_id"], [dict(candidates[0])], detected_source="test")
    before = {c["_id"] for c in store.list_clip_docs(job["_id"])}
    store.create_clips(job["_id"], [dict(candidates[0])], detected_source="test")
    after = {c["_id"] for c in store.list_clip_docs(job["_id"])}

    assert before == after == {clip_id}
    assert len(store.list_clip_docs(job["_id"])) == 1
