"""In-process supervisor: daemon threads that drain the local job queue.

FastAPI and the supervisor live in the SAME process — "worker offline" is
structurally impossible. Jobs are picked FIFO; there is no broker and no
second process to forget to start. "Worker" here means the QUEUE; the actual
pipeline work for a job runs in a dedicated child process (app.runner).

Pool size: ``[hosted] worker_concurrency`` (default 2). A serial pool was the
cause of "stuck at 0% / Waiting for worker..." reports — one slow or hung job
blocked every job behind it. Each slot runs its OWN child process, so slots are
genuinely independent, a crashed slot is reclaimed on the next loop, and a
stalled job is killed without disturbing the others.

Why a child process per job: a hung whisper/ffmpeg call inside the API process
cannot be interrupted — Python cannot kill a thread. A child process can be
SIGKILLed (taking its ffmpeg grandchildren with it, via its own process group),
so a stalled job is genuinely terminated, its memory/locks released, and Retry
starts a genuinely fresh process.

The watchdog fails a job after STALL_TIMEOUT_SEC of progress silence. The
threshold is a SILENCE window, not a total-runtime cap: transcription emits
progress per decoded segment (see app/asr.py), so silence genuinely means a
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

from app import hostlimits
from app.config import ROOT, Settings
from app.store import LocalStore

# Seconds of progress silence before a running job is declared stalled.
STALL_TIMEOUT_SEC = int(os.getenv("CLIPFORGE_STALL_TIMEOUT_SEC", "1200"))


class InProcessWorker:
    def __init__(self, settings: Settings, store: LocalStore) -> None:
        self.settings = settings
        self.store = store
        # The configured pool is an upper bound; the HOST decides the real one.
        # Each slot loads its own model (~1.7 GB for `medium`), so on a 2 GB
        # container a pool of 2 guarantees an OOM kill mid-job — which is what
        # "stuck at 0% / worker exited unexpectedly" looked like from outside.
        configured = max(1, min(8, int(getattr(settings.hosted, "worker_concurrency", 2) or 1)))
        # the model that will actually load — the configured one may not fit
        self.model, _note = hostlimits.resolved_model(settings.whisper.model)
        self.concurrency = hostlimits.recommended_concurrency(self.model, configured)
        self.configured_concurrency = configured
        if self.concurrency != configured:
            logger.warning(
                f"[worker] pool narrowed {configured} → {self.concurrency}: {hostlimits.describe()}. "
                f"Each parallel job loads its own '{self.model}' model "
                f"(~{hostlimits.MODEL_RSS_BYTES.get(self.model, 0) // 1024**2} MB)."
            )
        self._queue: queue.Queue[str] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._watchdog_thread: threading.Thread | None = None
        self._stop = threading.Event()
        # job_id -> running child process (one per occupied pool slot)
        self._active: dict[str, subprocess.Popen] = {}
        # FIFO mirror of the queue so the UI can show a real queue position
        self._pending: list[str] = []
        # legacy single-job view: the most recently started job. Kept because
        # status()/the focused supervision test address the worker this way.
        self._current_job: str | None = None
        self._current_proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._stalled_job: str | None = None  # last job the watchdog killed
        self._stalled_jobs: set[str] = set()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
                return
            self._stop.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog, name="clipforge-watchdog", daemon=True
            )
            self._watchdog_thread.start()
            for i in range(self.concurrency):
                thread = threading.Thread(
                    target=self._run, name=f"clipforge-worker-{i}", daemon=True
                )
                thread.start()
                self._threads.append(thread)
            logger.info(
                f"In-process worker online — {self.concurrency} parallel slot(s) "
                f"(configured {self.configured_concurrency}, model '{self.model}', "
                f"{hostlimits.describe()}), consuming the hosted job queue"
            )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._kill_all()  # don't leave children running past shutdown
        with self._lock:
            threads = list(self._threads)
            self._threads = []
        for thread in threads:
            thread.join(timeout=timeout)

    def enqueue(self, job_id: str) -> int:
        self._queue.put(job_id)
        with self._lock:
            self._pending.append(job_id)
            depth = len(self._pending)
        self._write_queue_position(job_id, depth - 1)
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
            active_proc = self._active.get(job_id)
            current_proc = self._current_proc
            current_job = self._current_job
        target = active_proc
        if target is None and current_job == job_id:
            target = current_proc
        if target is not None and target.poll() is None:
            self._stalled_job = job_id
            self._stalled_jobs.add(job_id)
            self._kill_process(target)
            logger.info(f"[worker] retry requested for active job {job_id}; old process group killed")

        job.update({
            "stage": "queued",
            "progress": 0,
            "progressDetail": "Re-queued — a fresh server worker will pick it up.",
            "error": None,
            "errorDetail": None,
            "completedAt": None,
            "stageEtaSec": None,
            "totalEtaSec": None,
            "stageStartedAt": int(time.time() * 1000),
            "lastProgressAt": int(time.time() * 1000),
        })
        self.store.write_job(job)
        # A retry is a genuinely fresh process; stage artifacts on disk let it
        # resume from the first incomplete stage (see app/stages.py).
        return self.enqueue(job_id)

    # ------------------------------------------------------------------
    # queue introspection (the "waiting behind N other jobs" indicator)
    # ------------------------------------------------------------------

    def queue_position(self, job_id: str) -> int | None:
        """0-based position in the waiting queue, or None when it is running."""
        with self._lock:
            if job_id in self._active:
                return None
            try:
                return self._pending.index(job_id)
            except ValueError:
                return None

    def queue_snapshot(self) -> dict[str, int]:
        with self._lock:
            return {job_id: pos for pos, job_id in enumerate(self._pending)}

    def _write_queue_position(self, job_id: str, position: int) -> None:
        """Persist the position so a client that never sees it move still knows."""
        job = self.store.get_job(job_id)
        if job is None or job.get("queuePosition") == position:
            return
        job["queuePosition"] = position
        job["progressDetail"] = (
            "Queued — waiting behind 1 other job."
            if position == 1
            else (
                f"Queued — waiting behind {position} other jobs."
                if position > 1
                else "Queued — starting now on the server worker."
            )
        )
        self.store.write_job(job)

    def _promote_positions(self) -> None:
        with self._lock:
            pending = list(self._pending)
        for position, job_id in enumerate(pending):
            self._write_queue_position(job_id, position)

    # ------------------------------------------------------------------
    # queue loops (one per pool slot)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                if job_id in self._pending:
                    self._pending.remove(job_id)
                self._current_job = job_id
            self._clear_queue_position(job_id)
            self._promote_positions()
            try:
                self._run_job_process(job_id)
            except Exception as exc:  # defensive: never lose the queue
                logger.exception(f"[worker] job {job_id} supervisor error: {exc}")
            finally:
                with self._lock:
                    if self._current_job == job_id:
                        self._current_job = None
                        self._current_proc = None
                self._queue.task_done()

    def _clear_queue_position(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None or job.get("queuePosition") is None:
            return
        job["queuePosition"] = None
        self.store.write_job(job)

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
            self._active[job_id] = proc
            self._current_proc = proc
            self._stalled_jobs.discard(job_id)
        logger.info(
            f"[worker] job {job_id} running in pid {proc.pid} "
            f"({len(self._active)}/{self.concurrency} slots busy)"
        )

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
                self._kill_process(proc)
                break
            time.sleep(0.4)

        pump.join(timeout=5)
        rc = proc.returncode
        # A retry may have started a new process for this job before the old
        # supervisor finished reaping. Never let the old process report the new
        # attempt's terminal state.
        with self._lock:
            is_current_process = self._active.get(job_id) is proc
            if is_current_process:
                self._active.pop(job_id, None)
            if self._current_proc is proc:
                self._current_proc = None
        if not is_current_process:
            return
        job = self.store.get_job(job_id)
        stage = (job or {}).get("stage")
        if self._stalled_job == job_id or job_id in self._stalled_jobs:
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
        # A SIGKILL'd child reports no traceback at all — name the OOM kill and
        # the fix instead of leaving the user with "exited unexpectedly".
        hint = hostlimits.oom_hint(rc, stage)
        if hint:
            msg = f"Worker process was killed (code {rc})" + (
                f" during {stage}" if stage else ""
            )
        detail = msg + ("\n" + hint if hint else "") + ("\n" + tail if tail else "\n(no runner output captured)")
        logger.error(f"[worker] {msg} for job {job_id}" + (f" — {hint}" if hint else ""))
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

    def _kill_process(self, proc: subprocess.Popen) -> None:
        """SIGKILL one job's process group (and its ffmpeg children)."""
        if proc.poll() is not None:
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

    def _kill_current(self) -> None:
        """Kill the most recently started child (legacy single-job helper)."""
        proc = self._current_proc
        if proc is not None:
            self._kill_process(proc)

    def _kill_all(self) -> None:
        with self._lock:
            procs = list(self._active.values())
            legacy = self._current_proc
        for proc in procs:
            self._kill_process(proc)
        if legacy is not None and legacy not in procs:
            self._kill_process(legacy)

    def _targets(self) -> list[tuple[str, subprocess.Popen]]:
        """Every job that currently has a live child process."""
        with self._lock:
            targets = list(self._active.items())
            legacy_job = self._current_job
            legacy_proc = self._current_proc
        if legacy_job and legacy_proc is not None and all(job_id != legacy_job for job_id, _ in targets):
            targets.append((legacy_job, legacy_proc))
        return targets

    def _check_watchdog_once(self) -> bool:
        """Check running processes once; return True when any was killed.

        Kept separate from the daemon loop so focused tests can verify the
        actual process-group kill without waiting for the polling interval.
        """
        killed_any = False
        for job_id, proc in self._targets():
            if job_id in self._stalled_jobs or self._stalled_job == job_id:
                continue
            job = self.store.get_job(job_id)
            if job is None or job.get("stage") in {"complete", "failed"}:
                continue
            if proc.poll() is not None:
                continue
            last = job.get("lastProgressAt") or job.get("stageStartedAt") or job.get("createdAt") or 0
            silent_for = time.time() * 1000 - last
            if silent_for <= STALL_TIMEOUT_SEC * 1000:
                continue

            logger.error(
                f"[worker] job {job_id} stalled: no progress for {silent_for / 1000:.0f}s "
                f"at stage {job.get('stage')} — killing pid {proc.pid}"
            )
            self._stalled_job = job_id
            self._stalled_jobs.add(job_id)
            self._kill_process(proc)
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
            # Write after the terminal state so this log cannot be overwritten
            # by the stale job snapshot above.
            self.store.append_job_log(
                job_id,
                f"stall watchdog: no progress for {int(silent_for / 1000)}s at stage "
                f"{job.get('stage')} — killed the stuck worker process",
            )
            killed_any = True
        return killed_any

    def _watchdog(self) -> None:
        while not self._stop.wait(5.0):
            self._check_watchdog_once()

    # ------------------------------------------------------------------
    # status for the dashboard card
    # ------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            active = {job_id: proc for job_id, proc in self._active.items()}
            legacy_proc = self._current_proc
            legacy_job = self._current_job
            queue_depth = len(self._pending)
            threads = list(self._threads)
        if not active and legacy_job and legacy_proc is not None:
            active = {legacy_job: legacy_proc}
        running = {job_id: proc for job_id, proc in active.items() if proc.poll() is None}
        first = next(iter(running.items()), None)
        return {
            "online": bool(threads) and all(t.is_alive() for t in threads),
            "concurrency": self.concurrency,
            "configuredConcurrency": self.configured_concurrency,
            "hostLimits": hostlimits.limits(),
            "activeJobs": sorted(running.keys()),
            "activeCount": len(running),
            "currentJob": first[0] if first else None,
            "currentPid": first[1].pid if first else None,
            "queueDepth": queue_depth,
            "queuedJobs": sorted(self.queue_snapshot().keys()),
            "stallTimeoutSec": STALL_TIMEOUT_SEC,
        }
