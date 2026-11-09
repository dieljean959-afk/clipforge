"""Disk exhaustion must be a message, not a loop; job logs must not grow forever.

Seen on this host: the filesystem sat at 176 MB free. The worker's preflight
only demanded 100 MB, so the job started, downloaded a 166 MB source, and died
mid-render on ENOSPC. Every retry re-ran the same attempt into the same full
disk, which read as "it keeps redoing the job until there is no space left".
The other half was silent weight: the job log array is appended to on every
progress line and rewritten to disk each time, so one 21-minute batch reached
3 500 lines / 600 KB that the dashboard still shipped on every poll.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import load_config
from app.store import LocalStore
from app.worker import InProcessWorker


@pytest.fixture()
def env(tmp_path):
    base = load_config(quiet=True)
    paths = replace(
        base.app.paths,
        data_dir=tmp_path,
        downloads_dir=tmp_path / "downloads",
    )
    settings = replace(base, app=replace(base.app, paths=paths))
    return settings, LocalStore(paths)


def test_job_logs_are_capped_and_keep_the_newest_lines(env):
    _settings, store = env
    job_id = store.create_job(source_type="url", source="https://youtu.be/x")["_id"]

    for i in range(LocalStore.MAX_JOB_LOGS + 250):
        store.append_job_log(job_id, f"line {i}")

    logs = store.get_job(job_id)["logs"]
    assert len(logs) == LocalStore.MAX_JOB_LOGS
    assert logs[-1] == f"line {LocalStore.MAX_JOB_LOGS + 249}"
    assert logs[0] == "line 250"


def test_preflight_refuses_to_start_a_job_that_cannot_fit(env, monkeypatch):
    settings, store = env
    job_id = store.create_job(source_type="url", source="https://youtu.be/x")["_id"]
    worker = InProcessWorker(settings, store)
    worker._stop.set()

    from app import hosting

    monkeypatch.setattr(
        hosting, "disk_stats", lambda path: {"freeBytes": 180 * 1024 * 1024}
    )
    # A host this full is a terminal message, not a start-and-die.
    assert worker._preflight_environment(job_id) != []
    job = store.get_job(job_id)
    assert job["stage"] == "failed"
    assert "MB of disk free" in job["error"]


def test_preflight_allows_a_job_with_real_headroom(env, monkeypatch):
    settings, store = env
    job_id = store.create_job(source_type="url", source="https://youtu.be/x")["_id"]
    worker = InProcessWorker(settings, store)
    worker._stop.set()

    from app import hosting

    monkeypatch.setattr(
        hosting, "disk_stats", lambda path: {"freeBytes": 1200 * 1024 * 1024}
    )
    assert worker._preflight_environment(job_id) == []
    assert store.get_job(job_id)["stage"] != "failed"
