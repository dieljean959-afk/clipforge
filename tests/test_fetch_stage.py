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
def env(tmp_path):
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
