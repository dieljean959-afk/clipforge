"""The SIGTERM reporter names the sender, not just the signal."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

from app import runner


def test_own_pid_is_described_with_its_command_line():
    text = runner.sender_description(os.getpid())
    assert f"pid {os.getpid()}" in text
    assert "python" in text.lower() or "pytest" in text.lower()


def test_pid_one_is_named_as_the_container_init():
    assert runner.sender_description(1).startswith("pid 1 (the container init")


def test_a_missing_sender_is_still_explained():
    assert "unknown" in runner.sender_description(0)
    assert "unknown" in runner.sender_description(None)


def test_the_reporter_logs_the_real_sender_pid():
    """Send SIGTERM from this test process and read what the child recorded."""
    script = r"""
import os, signal, sys, time
sys.path.insert(0, os.getcwd())
from app import runner

class Store:
    lines = []
    def append_job_log(self, job_id, line):
        Store.lines.append(line)

runner._install_process_handlers("job-test", Store(), None)
Store.lines.append("READY")
print("READY", flush=True)
time.sleep(5)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.getcwd(),
    )
    try:
        assert proc.stdout is not None
        assert "READY" in proc.stdout.readline()
        os.kill(proc.pid, signal.SIGTERM)
        out, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert f"pid {os.getpid()}" in out, out
    assert "SIGTERM" in out, out
