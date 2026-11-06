"""Reconciling a dead job must not erase the reason it died.

Seen on job ``0d9f0a3a`` (a YouTube link): the runner wrote
"Could not fetch video: This site asked this server to sign in…" at 00:49:18,
its stage was still ``grabbing`` when the dashboard's reconciliation pass ran
four seconds later, and the pass replaced that actionable message with
"The worker stopped while this job was in progress and never reported back".
The user was told the server was broken when the fix was to attach the file or
set cookies — the real line was still in the job log right above it.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.config import load_config
from app.store import LocalStore
from app.worker import InProcessWorker

GENERIC = "The worker stopped while this job was in progress"


@pytest.fixture()
def env(tmp_path):
    base = load_config(quiet=True)
    paths = replace(
        base.app.paths,
        data_dir=tmp_path,
        downloads_dir=tmp_path / "downloads",
    )
    settings = replace(base, app=replace(base.app, paths=paths))
    store = LocalStore(paths)
    return settings, store


def _worker(settings, store):
    worker = InProcessWorker(settings, store)
    worker._stop.set()  # never start the supervisor threads in a test
    return worker


def test_reconcile_keeps_the_reason_the_runner_reported(env):
    settings, store = env
    store.write_job({
        "_id": "job_blocked_download",
        "sourceType": "url",
        "source": "https://youtu.be/aaaaaaaaaaa",
        "stage": "grabbing",
        "progress": 0,
        "createdAt": 1,
        "error": (
            "Could not fetch video: This site asked this server to sign in "
            "before it would serve the video"
        ),
        "completedAt": 1793926000000,
    })

    fixed = _worker(settings, store).reconcile_stale_running(force=True)

    job = store.get_job("job_blocked_download")
    assert fixed == ["job_blocked_download"]
    assert job["stage"] == "failed"
    assert job["failedStage"] == "grabbing"
    assert "asked this server to sign in" in job["error"]
    assert GENERIC not in job["error"]
    assert job["completedAt"] == 1793926000000


def test_reconcile_requeues_a_silent_job_instead_of_failing_it(env):
    """LOCKED_SPEC §4: stale heartbeat -> interrupted -> re-queued, no Retry."""
    settings, store = env
    store.write_job({
        "_id": "job_silent",
        "sourceType": "url",
        "source": "https://example.com/x.mp4",
        "stage": "transcribing",
        "progress": 40,
        "createdAt": 1,
    })
    worker = _worker(settings, store)

    fixed = worker.reconcile_stale_running(force=True)

    job = store.get_job("job_silent")
    assert fixed == ["job_silent"]
    assert job["stage"] == "transcribing"       # resumes from the cached stage
    assert job.get("interrupted") is True
    assert job.get("interruptCount") == 1
    assert job.get("error") in (None, "")
    assert worker.is_queued("job_silent") is True


def test_reconcile_gives_up_after_three_attempts(env):
    settings, store = env
    store.write_job({
        "_id": "job_loop",
        "sourceType": "url",
        "source": "https://example.com/x.mp4",
        "stage": "transcribing",
        "progress": 40,
        "createdAt": 1,
        "interruptCount": 3,
    })
    worker = _worker(settings, store)

    worker.reconcile_stale_running(force=True)

    job = store.get_job("job_loop")
    assert job["stage"] == "failed"
    assert job["failedStage"] == "transcribing"
    assert GENERIC in job["error"]
    assert job["progress"] == 0
    assert worker.is_queued("job_loop") is False


def test_replayed_progress_never_revives_a_failed_job(env):
    """The stdout pump runs behind the child: its buffered events must not
    move a job the child already failed back to a running stage."""
    settings, store = env
    store.write_job({
        "_id": "job_race",
        "sourceType": "url",
        "source": "https://example.com/x.mp4",
        "stage": "failed",
        "failedStage": "grabbing",
        "error": "Could not fetch video: blocked",
        "completedAt": 123,
        "logs": [],
    })
    worker = _worker(settings, store)

    worker._replay_runner_event(
        "job_race",
        json.dumps({"stage": "stage_started", "stage_name": "grabbing", "progress": 0}),
    )

    job = store.get_job("job_race")
    assert job["stage"] == "failed"
    assert job["error"] == "Could not fetch video: blocked"
    assert job["completedAt"] == 123


def test_replayed_progress_still_moves_a_running_job(env):
    settings, store = env
    store.write_job({
        "_id": "job_live",
        "sourceType": "url",
        "source": "https://example.com/x.mp4",
        "stage": "fetch",
        "logs": [],
    })
    worker = _worker(settings, store)

    worker._replay_runner_event(
        "job_live",
        json.dumps({"stage": "stage_started", "stage_name": "grabbing", "progress": 12}),
    )

    job = store.get_job("job_live")
    assert job["stage"] == "grabbing"
    assert job["progress"] == 12
