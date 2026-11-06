"""LOCKED_SPEC §4: the worker runs in its own process, the API only proxies.

These cover the two things that can silently break the split: the control
protocol (a call must reach the worker and its failure must come back as an
error, never as a fake success) and the supervisor's behaviour when no daemon is
listening (report offline; never invent a status or spawn one in a test).
"""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import replace

import pytest

from app.config import load_config
from app.store import LocalStore
from app.workerd import (
    OPS,
    WorkerDaemon,
    WorkerSupervisor,
    WorkerUnavailable,
    acquire_single_instance_lock,
    offline_status,
    socket_path,
)


class _FakeWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def enqueue(self, job_id: str) -> int:
        self.calls.append(("enqueue", {"job_id": job_id}))
        return 4

    def status(self) -> dict:
        return {"online": True, "activeCount": 1, "queueDepth": 2}

    def boom(self) -> None:
        raise RuntimeError("pipeline exploded")


def test_dispatch_calls_the_worker_with_the_arguments_it_was_given():
    daemon = WorkerDaemon.__new__(WorkerDaemon)  # no store, no config
    daemon.worker = _FakeWorker()

    assert daemon.dispatch("enqueue", {"job_id": "job_1"}) == 4
    assert daemon.worker.calls == [("enqueue", {"job_id": "job_1"})]
    assert daemon.dispatch("ping", {})["pid"] > 0


def test_dispatch_refuses_anything_outside_the_whitelist():
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    daemon.worker = _FakeWorker()

    assert "boom" not in OPS
    with pytest.raises(ValueError):
        daemon.dispatch("boom", {})
    with pytest.raises(ValueError):
        daemon.dispatch("_kill_all", {})


@pytest.fixture()
def settings(tmp_path):
    base = load_config(quiet=True)
    paths = replace(base.app.paths, data_dir=tmp_path, downloads_dir=tmp_path / "downloads")
    return replace(base, app=replace(base.app, paths=paths))


def _fake_daemon(sock_path, reply: dict, seen: list) -> threading.Thread:
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(4)

    def run() -> None:
        with server:
            conn, _ = server.accept()
            with conn:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                seen.append(json.loads(buf.decode()))
                conn.sendall(json.dumps(reply).encode() + b"\n")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_supervisor_proxies_calls_over_the_socket(settings, tmp_path):
    store = LocalStore(settings.app.paths)
    supervisor = WorkerSupervisor(settings, store, autostart=False)
    seen: list = []
    thread = _fake_daemon(socket_path(settings), {"ok": True, "result": 3}, seen)

    assert supervisor.enqueue("job_7") == 3
    thread.join(timeout=5)
    assert seen == [{"op": "enqueue", "args": {"job_id": "job_7"}}]


def test_supervisor_reports_offline_instead_of_inventing_a_status(settings):
    store = LocalStore(settings.app.paths)
    supervisor = WorkerSupervisor(settings, store, autostart=False)

    status = supervisor.status()

    assert status["online"] is False
    assert status["activeJobs"] == []
    assert status["process"] == "worker-daemon"
    assert set(offline_status()) <= set(status)
    assert supervisor.queue_snapshot() == {}
    assert supervisor.reconcile_stale_running(force=True) == []


def test_enqueue_without_a_daemon_does_not_raise(settings):
    """The job stays `queued` and the daemon's reconciliation queues it later."""
    store = LocalStore(settings.app.paths)
    supervisor = WorkerSupervisor(settings, store, autostart=False)

    assert supervisor.enqueue("job_8") == 0


def test_user_actions_still_fail_loudly_without_a_daemon(settings):
    """Pause/resume/retry must not pretend to have happened."""
    store = LocalStore(settings.app.paths)
    supervisor = WorkerSupervisor(settings, store, autostart=False)

    with pytest.raises(WorkerUnavailable):
        supervisor.resume("job_9")
    with pytest.raises(WorkerUnavailable):
        supervisor.pause("job_9")


def test_a_second_daemon_cannot_own_the_same_data_dir(tmp_path):
    """A socket file is not a lock: the second daemon must refuse to serve."""
    lock = tmp_path / "work" / "worker.lock"

    first = acquire_single_instance_lock(lock)
    assert first is not None
    try:
        assert acquire_single_instance_lock(lock) is None
    finally:
        os.close(first)


def test_api_shutdown_does_not_flag_running_jobs(settings):
    """The daemon outlives the API, so its job is not interrupted."""
    store = LocalStore(settings.app.paths)
    supervisor = WorkerSupervisor(settings, store, autostart=False)

    assert supervisor.mark_in_flight_interrupted() == []
