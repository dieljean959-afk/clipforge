"""In-process supervisor: a daemon thread that drains the local job queue.

FastAPI and the supervisor live in the SAME process — "worker offline" is
structurally impossible. Jobs are picked FIFO; there is no broker and no
second process to forget to start. "Worker" here means the QUEUE; the actual
pipeline work for a job runs in a dedicated child process (app.runner).

Why a child process per job: a hung whisper/ffmpeg call inside the API
process cannot be interrupted — Python cannot kill a thread. A child process
can be SIGKILLed (taking its ffmpeg grandchildren with it, via its own
process group), so a stalled job is genuinely terminated, its memory/locks
released, and Retry starts a genuinely fresh process.

The watchdog fails a job after STALL_TIMEOUT_SEC of progress silence. The
threshold is a SILENCE window, not a total-runtime cap: transcription emits
progress per decoded segment (see subtitle.py), so silence genuinely means a
dead worker, and the same threshold is correct for a 2-minute or a 2-hour
source. Override with CLIPFORGE_STALL_TIMEOUT_SEC (used by the test suite to
keep stall tests fast).
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time

from loguru import logger

from app.config import ROOT, Settings
from app.store import LocalStore

# Seconds of progress silence before a running job is declared stalled.
STALL_TIMEOUT_SEC = int(os.getenv("CLIPFORGE_STALL_TIMEOUT_SEC", "1200"))


class InProcessWorker:
    def __init__(self, settings: Settings, store: LocalStore) -> None:
        self.settings = settings
        self.store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._current_job: str | None = None
        self._current_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stalled_job: str | None = None  # watchdog already killed it

    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="clipforge-worker", daemon=True)
            self._thread.start()
            logger.info("In-process worker online, consuming the hosted job queue")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._kill_current()  # don't leave a child running past shutdown
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def enqueue(self, job_id: str) -> int:
        self._queue.put(job_id)
        depth = self._queue.qsize()
        logger.info(f"[worker] job {job_id} queued (depth {depth})")
        return depth

    def retry(self, job_id: str) -> int:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)

        # If a user retries while the same job is still being supervised, stop
        # the old process before queueing the new attempt. Otherwise two
        # processes could write the same job/stage artifacts concurrently.
        with self._lock:
            current_proc = self._current_proc
            current_job = self._current_job
        if current_job == job_id and current_proc is not None and current_proc.poll() is None:
            self._stalled_job = job_id
            self._kill_current()
            logger.info(f"[worker] retry requested for active job {job_id}; old process group killed")

        job.update({
            "stage": "queued",
            "progress": 0,
            "progressDetail": "Re-queued — a fresh server worker will pick it up.",
            "error": None,
            "errorDetail": None,
            "completedAt": None,
            "lastProgressAt": int(time.time() * 1000),
        })
        self.store.write_job(job)
        # A retry is a genuinely fresh process; stage artifacts on disk let it
        # resume from the first incomplete stage (see app/stages.py). Keep the
        # old attempt marked as superseded until the new queue item starts;
        # this prevents the old supervisor from labeling the fresh attempt
        # failed while it is being killed/reaped.
        return self.enqueue(job_id)

    # ------------------------------------------------------------------
    # queue loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        logger.info("[worker] consuming the hosted queue (one child process per job)")
        watchdog = threading.Thread(target=self._watchdog, name="clipforge-watchdog", daemon=True)
        watchdog.start()
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._current_job = job_id
            self._stalled_job = None
            try:
                self._run_job_process(job_id)
            except Exception as exc:  # defensive: never lose the queue
                logger.exception(f"[worker] job {job_id} supervisor error: {exc}")
            finally:
                self._current_job = None
                self._current_proc = None
                self._queue.task_done()

    def _run_job_process(self, job_id: str) -> None:
        """Run one job in a child process; supervise exit; surface crashes."""
        log_dir = self.settings.app.paths.data_dir / "work" / job_id
        log_dir.mkdir(parents=True, exist_ok=True)
        runner_log = log_dir / "runner.log"

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # start_new_session: the child gets its own process group so the
        # watchdog can kill the ffmpeg descendants along with the runner
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.runner", job_id],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with self._lock:
            self._current_proc = proc
        logger.info(f"[worker] job {job_id} running in pid {proc.pid}")

        # tail the child's output into the job log (structured surfacing —
        # 'View logs' shows the real worker output, not just headlines)
        def _pump() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.store.append_job_log(job_id, line[:500])
            except Exception:
                pass

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()

        while proc.poll() is None:
            if self._stop.is_set():
                self._kill_current()
                break
            time.sleep(0.4)

        pump.join(timeout=5)
        rc = proc.returncode
        # A retry may have started a new process for this job before the old
        # supervisor finished reaping. Never let the old process report the new
        # attempt's terminal state.
        with self._lock:
            is_current_process = self._current_proc is proc
        if not is_current_process:
            return
        job = self.store.get_job(job_id)
        stage = (job or {}).get("stage")
        if self._stalled_job == job_id:
            # watchdog already failed it with a specific reason
            return
        if stage in {"complete", "failed"}:
            return
        # the child died without recording a terminal state (crash, OOM-kill,
        # SIGSEGV in native code...) — surface it instead of hanging
        tail = ""
        try:
            if runner_log.exists():
                tail = "\n".join(runner_log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
        except Exception:
            pass
        msg = f"Worker process exited unexpectedly (code {rc}) before the job finished"
        detail = msg + ("\n" + tail if tail else "\n(no runner output captured)")
        logger.error(f"[worker] {msg} for job {job_id}")
        if job is not None:
            job.update({
                "stage": "failed",
                "error": msg,
                "errorDetail": detail[:4000],
                "completedAt": int(time.time() * 1000),
            })
            self.store.write_job(job)
        self.store.append_job_log(job_id, f"FAILED: {msg}")

    # ------------------------------------------------------------------
    # watchdog
    # ------------------------------------------------------------------

    def _kill_current(self) -> None:
        """SIGKILL the current job's process group (and its ffmpeg children)."""
        proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                continue
        logger.error(f"[worker] pid {proc.pid} survived SIGKILL")

    def _check_watchdog_once(self) -> bool:
        """Check the current process once; return True when it was killed.

        Kept separate from the daemon loop so focused tests can verify the
        actual process-group kill without waiting for the polling interval.
        """
        job_id = self._current_job
        if job_id is None or self._stalled_job == job_id:
            return False
        job = self.store.get_job(job_id)
        if job is None or job.get("stage") in {"complete", "failed"}:
            return False
        proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return False
        last = job.get("lastProgressAt") or job.get("stageStartedAt") or job.get("createdAt") or 0
        silent_for = time.time() * 1000 - last
        if silent_for <= STALL_TIMEOUT_SEC * 1000:
            return False

        logger.error(
            f"[worker] job {job_id} stalled: no progress for {silent_for / 1000:.0f}s "
            f"at stage {job.get('stage')} — killing pid {proc.pid}"
        )
        self._stalled_job = job_id
        self._kill_current()
        job.update({
            "stage": "failed",
            "error": (
                f"Pipeline stalled — no progress for {int(silent_for / 1000)}s "
                f"at stage {job.get('stage')}. The stuck worker process was killed. "
                "Retry resumes from the last completed stage."
            ),
            "errorDetail": f"stall watchdog killed pid {proc.pid} after {int(silent_for / 1000)}s of silence",
            "completedAt": int(time.time() * 1000),
        })
        self.store.write_job(job)
        # Write after the terminal state so this log cannot be overwritten by
        # the stale job snapshot above.
        self.store.append_job_log(
            job_id,
            f"stall watchdog: no progress for {int(silent_for / 1000)}s at stage "
            f"{job.get('stage')} — killed the stuck worker process",
        )
        return True

    def _watchdog(self) -> None:
        while not self._stop.wait(5.0):
            self._check_watchdog_once()

    # ------------------------------------------------------------------
    # status for the dashboard card
    # ------------------------------------------------------------------

    def status(self) -> dict:
        proc = self._current_proc
        return {
            "online": self._thread is not None and self._thread.is_alive(),
            "currentJob": self._current_job,
            "currentPid": proc.pid if proc is not None and proc.poll() is None else None,
            "queueDepth": self._queue.qsize(),
            "stallTimeoutSec": STALL_TIMEOUT_SEC,
        }
