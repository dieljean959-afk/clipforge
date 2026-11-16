"""The fetch stage, without touching the network or yt-dlp.

Both bugs this covers were crashes that hid the real reason a batch failed:

* ``Pipeline`` has no ``_on_stage_log`` attribute — assigning to it killed
  every URL job right after the download succeeded;
* the failure path called ``on_log(...)``, a name that only exists inside
  ``download.download_url``, so a FAILED download died with
  ``NameError: name 'on_log' is not defined`` and the yt-dlp reason (YouTube's
  bot check) never reached the dashboard.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app import download, video
from app.config import load_config
from app.stages import StageWorkspace
from app.store import LocalStore
from app.task import Pipeline


@pytest.fixture()
def env(tmp_path, monkeypatch):
    # These two tests cover the LOCAL fallback, which only runs on an install
    # with no bucket at all. This machine has one configured, so without this
    # the fetch stage correctly takes the streaming path instead.
    from types import SimpleNamespace

    from app import objectstore

    monkeypatch.setattr(objectstore, "configured", lambda *a, **k: False)
    # `_push_source` reads the bucket config directly, so `configured` alone
    # is not enough: without this the fake 1.8 KB "video" in the first test
    # was uploaded to the real production bucket as job_fetch_test/source.mp4.
    monkeypatch.setattr(
        objectstore, "config_from_env", lambda *a, **k: SimpleNamespace(enabled=False)
    )
    base = load_config(quiet=True)
    paths = replace(
        base.app.paths,
        data_dir=tmp_path,
        downloads_dir=tmp_path / "downloads",
    )
    settings = replace(base, app=replace(base.app, paths=paths))
    store = LocalStore(settings.app.paths)
    job = {
        "_id": "job_fetch_test",
        "sourceType": "url",
        "source": "https://youtu.be/aaaaaaaaaaa",
        "state": "running",
        "stage": "grabbing",
        "createdAt": 0,
    }
    store.write_job(job)
    return settings, store, job, StageWorkspace(paths.data_dir, job["_id"])


def test_url_job_finishes_fetch_without_the_missing_log_dict(env, monkeypatch):
    settings, store, job, ws = env

    def fake_download(url, dest_dir, settings, on_progress=None, on_log=None):
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / "source.mp4"
        out.write_bytes(b"not really a video" * 100)
        if on_log:
            on_log("[download] fake finished")
        if on_progress:
            on_progress(100, 0.0)
        return out

    monkeypatch.setattr(download, "download_url", fake_download)
    monkeypatch.setattr(video, "verify_source", lambda p: None)
    monkeypatch.setattr(video, "probe", lambda p: {"duration": 12.0, "width": 320, "height": 180})

    src = Pipeline(settings, store)._stage_fetch(job, ws)

    assert src.name == "source.mp4" and src.stat().st_size > 0
    assert ws.is_done("fetch")
    assert (ws.done_meta("fetch") or {}).get("duration") == 12.0


def test_download_failure_keeps_its_own_error(env, monkeypatch):
    settings, store, job, ws = env
    reason = (
        "This site asked this server to sign in before it would serve the video "
        "(YouTube: \"Sign in to confirm you're not a bot\")."
    )

    def failing(url, dest_dir, settings, on_progress=None, on_log=None):
        if on_log:
            on_log("[youtube] extracting")
        raise download.DownloadError(reason)

    monkeypatch.setattr(download, "download_url", failing)

    with pytest.raises(download.DownloadError) as caught:
        Pipeline(settings, store)._stage_fetch(job, ws)

    assert str(caught.value) == reason
    # The old handler raised NameError here, replacing the real cause.
    assert "on_log" not in str(caught.value)
    assert not ws.is_done("fetch")


# --- resume must not re-download a source that is still in the bucket ------
#
# `done_meta["path"]` for a bucket capture is an OBJECT KEY, so the old
# `Path(path).exists()` check was always false and every host restart
# re-streamed the whole video: nine sections and ~13 minutes for a 20:37
# source whose objects were still sitting in the bucket.

_REMOTE_META = {
    "remote": True,
    "videoKey": "clipforge/uploads/job_fetch_test/sections/sec_000.mp4",
    "audioKey": "clipforge/uploads/job_fetch_test/audio.m4a",
    "method": "sections",
    "format": "(sections)",
    "audioFormat": "ba",
    "bytes": 123456,
    "sections": [
        {"key": "clipforge/uploads/job_fetch_test/sections/sec_000.mp4", "start": 0.0, "end": 150.0},
        {"key": "clipforge/uploads/job_fetch_test/sections/sec_008.mp4", "start": 1200.0, "end": 1237.0},
    ],
    "duration": 1237.0,
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
}


def _done_remote_fetch(ws) -> None:
    ws.write_json("fetch", "meta.json", _REMOTE_META)
    ws.mark_done(
        "fetch", artifacts=["fetch/meta.json"], path=_REMOTE_META["videoKey"],
        duration=1237.0, width=1920, height=1080, fps=30.0,
    )


def test_resume_reuses_the_bucket_source_without_downloading_it_again(env, monkeypatch):
    settings, store, job, ws = env
    from app import objectstore, source_stream

    _done_remote_fetch(ws)
    monkeypatch.setattr(objectstore, "configured", lambda *a, **k: True)
    monkeypatch.setattr(objectstore, "head", lambda key, *a, **k: {"size": 999})

    def boom(*a, **k):
        raise AssertionError("the source was re-streamed on resume")

    monkeypatch.setattr(source_stream, "capture", boom)

    src = Pipeline(settings, store)._stage_fetch(job, ws)

    assert isinstance(src, source_stream.RemoteSource)
    assert len(src.sections) == 2, "the section spans were lost on resume"
    assert src.probe["duration"] == 1237.0, "the full length was lost on resume"
    assert src.probe["width"] == 1920


def test_a_missing_bucket_object_forces_a_re_capture(env, monkeypatch):
    settings, store, job, ws = env
    from app import objectstore, source_stream

    _done_remote_fetch(ws)
    monkeypatch.setattr(objectstore, "configured", lambda *a, **k: True)
    monkeypatch.setattr(
        objectstore, "head",
        lambda key, *a, **k: None if key.endswith("audio.m4a") else {"size": 1},
    )
    captured = {}

    def fake_capture(url, job_id, settings, **kw):
        captured["yes"] = True
        return source_stream.RemoteSource(
            job_id, video_key="v", audio_key="a",
            probe={"duration": 1237.0, "width": 1920, "height": 1080, "fps": 30.0},
        )

    monkeypatch.setattr(source_stream, "capture", fake_capture)

    src = Pipeline(settings, store)._stage_fetch(job, ws)

    assert captured.get("yes") is True, "a missing object must not be reused"
    assert src.video_key == "v"


def test_a_local_download_is_still_reused_from_disk(env, monkeypatch):
    """The bucket check must not break the old local-source resume."""
    settings, store, job, ws = env
    local = ws.stage_dir("fetch") / "source.mp4"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"x" * 64)
    ws.write_json("fetch", "meta.json", {"path": str(local), "duration": 12.0})
    ws.mark_done("fetch", artifacts=["fetch/meta.json"], path=str(local), duration=12.0)

    src = Pipeline(settings, store)._stage_fetch(job, ws)

    assert src == local


def test_a_swept_workspace_adopts_the_capture_recorded_on_the_job(env, monkeypatch):
    """A cleanup must not turn a retry into a second 2.6-hour re-capture.

    The stage marker lives under `data/work/<job>`, which a disk sweep removes.
    The capture itself is recorded on the job (`sourceStream`), and every
    object is still in the bucket — so the fetch stage must adopt it instead of
    asking yt-dlp for the same 22 sections again.
    """
    settings, store, job, ws = env
    from app import objectstore, source_stream

    job["sourceStream"] = {
        "videoKey": "clipforge/uploads/job_fetch_test/sections/sec_000.mp4",
        "audioKey": "clipforge/uploads/job_fetch_test/audio.m4a",
        "method": "sections",
        "format": "(sections)",
        "audioFormat": "ba",
        "bytes": 470_000_000,
        "sections": [
            {"key": "clipforge/uploads/job_fetch_test/sections/sec_000.mp4",
             "start": 0.0, "end": 75.0},
            {"key": "clipforge/uploads/job_fetch_test/sections/sec_008.mp4",
             "start": 600.0, "end": 675.0},
        ],
        "probe": {"duration": 675.0, "width": 1280, "height": 720, "fps": 60.0},
        "audioStartTime": 0.0,
    }
    store.write_job(job)
    assert not ws.is_done("fetch"), "the marker must genuinely be gone"

    monkeypatch.setattr(objectstore, "configured", lambda *a, **k: True)
    monkeypatch.setattr(objectstore, "head", lambda key, *a, **k: {"size": 999})

    def boom(*a, **k):
        raise AssertionError("the source was captured again after its marker was swept")

    monkeypatch.setattr(source_stream, "capture", boom)

    src = Pipeline(settings, store)._stage_fetch(job, ws)

    assert isinstance(src, source_stream.RemoteSource)
    assert len(src.sections) == 2
    assert src.probe["duration"] == 675.0
    assert ws.is_done("fetch"), "the adopted capture must be recorded for the next retry"
    meta = ws.read_json("fetch", "meta.json") or {}
    assert meta.get("videoKey") == job["sourceStream"]["videoKey"]


def test_a_capture_with_a_missing_middle_section_is_not_adopted(env, monkeypatch):
    """Adopting a source with one evicted piece fails a clip later.

    The check used to HEAD only the first and last section. A middle object
    can be gone after a bucket cleanup, so verify every recorded key before
    promising the resume will not download again.
    """
    settings, store, job, _ws = env
    from app import objectstore, source_stream

    sections = [
        {"key": f"clipforge/uploads/{job['_id']}/sections/sec_{i:03d}.mp4",
         "start": i * 75.0, "end": (i + 1) * 75.0}
        for i in range(3)
    ]

    def head(key, *a, **k):
        return {"size": 0 if key.endswith("sec_001.mp4") else 100}

    monkeypatch.setattr(objectstore, "head", head)
    src = source_stream.RemoteSource(
        job["_id"], video_key=sections[0]["key"], audio_key="audio.m4a",
        method="sections", sections=sections,
    )

    assert Pipeline(settings, store)._remote_source_intact(src, job["_id"]) is False
