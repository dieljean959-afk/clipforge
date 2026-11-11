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
    from app import objectstore

    monkeypatch.setattr(objectstore, "configured", lambda *a, **k: False)
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
