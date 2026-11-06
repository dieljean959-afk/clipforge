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

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from app import hostlimits
from app.config import ROOT, Settings
from app.store import LocalStore

# Seconds of progress silence before a running job is declared stalled.
# Was 1200 (20 min), which is what the dashboard was reporting verbatim: a job
# sat at "Rendering clip 10 of 10" for twenty minutes with nothing happening.
# The per-clip encode deadline in app.encode now bounds a hung ffmpeg at
# 6-30 min depending on clip length; this is the coarser backstop for stages
# that produce no progress lines at all. Kept above the longest legitimate
# silent stretch (a long transcription window on 1 CPU) but well under the old
# window, and a stall now earns one automatic retry before the user is told.
STALL_TIMEOUT_SEC = int(os.getenv("CLIPFORGE_STALL_TIMEOUT_SEC", "600"))

# How many times a job killed by an EXTERNAL signal (SIGTERM/SIGKILL — host
# restart, redeploy, scale-down) is auto-resumed before it is finally failed.
# Bounded on purpose: a job that cannot survive a restart loop must stop
# looping eventually, but losing hours of transcription to a 30-second host
# bounce is the failure this prevents.
# LOCKED_SPEC §4: a job resumes automatically at most THREE times.
INTERRUPT_MAX_RESUMES = int(os.getenv("CLIPFORGE_INTERRUPT_MAX_RESUMES", "3"))

# Matches the signalled-death banners this module writes, e.g.
# "exited unexpectedly (code -15)". Kept deliberately narrow so it cannot match
# a stray mention of "code" inside an ordinary traceback and cause a bad job to
# be retried forever.
_SIGNAL_BANNER = re.compile(r"\(code -\d+\)")

SOURCE_LOST_MESSAGE = (
    "Source file was lost, please re-upload. "
    "The uploaded video is no longer on the server, so this job cannot be "
    "resumed — attach the file again to start a new one."
)

# Stages a job never leaves on its own. Anything that writes progress must
# refuse to move a job OUT of one of these: only a retry/resume may.
_TERMINAL_STAGES = frozenset({"complete", "failed", "paused"})

# The runner's own failure line, e.g.
#   "FAILED at grabbing: Could not fetch video: This site asked this server …"
# or the one the supervisor lands from events.jsonl: "FAILED: <reason>".
_FAILED_LINE = re.compile(r"\bFAILED(?: at [A-Za-z]+)?: (.+)$")


def _reason_from_logs(job: dict[str, Any]) -> str:
    """Recover the runner's last explicit failure from the job log.

    A job can lose the `error` field to a stale write and still have the real
    reason sitting in its log. Reading it back beats telling the user the
    generic "worker stopped" when the log says exactly what happened.
    """
    logs = job.get("logs")
    if not isinstance(logs, list):
        return ""
    for line in reversed(logs[-300:]):
        match = _FAILED_LINE.search(str(line))
        if match:
            return match.group(1).strip()[:1000]
    return ""


def unrecoverable_reason(store: Any, job: dict[str, Any]) -> str | None:
    """Can this job EVER be resumed? Return the reason when it cannot.

    Resuming a job whose upload is gone cannot work at any stage: `fetch` is
    the first thing every later stage reads its input from, and the fetch stage
    only knows how to recreate a URL source, never an uploaded one. Re-queueing
    such a job just burns a full pipeline launch per server restart and then
    fails again.

    URL jobs are always recoverable: the source is re-downloaded from its URL.
    Upload jobs are recoverable only while their file is still on disk.
    """
    if job.get("sourceType") != "upload":
        return None
    if store.find_source_file(job["_id"]) is not None:
        return None
    return SOURCE_LOST_MESSAGE


MEMORY_MAX_JOB_ATTEMPTS = 3

#: Names that can be left behind by a job that died. `app.asr_child` holds the
#: model; ffmpeg and yt-dlp hold the source. An orphan of any of them keeps RAM
#: or CPU busy that the next attempt is budgeted on.
_ORPHAN_MARKERS = ("app.asr_child", "app.runner", "ffmpeg", "ffprobe", "yt-dlp")


def cleanup_orphans(data_dir: Path, *, protect_pids: set[int] | None = None) -> list[str]:
    """Kill orphaned job processes and delete stale temp audio.

    An orphan is a process whose parent is gone (ppid 1) or is not this worker
    and not alive. Live children of the current worker are never touched.
    """
    protected = set(protect_pids or set())
    protected.add(os.getpid())
    killed: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in protected:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", "replace"
            )
        except OSError:
            continue
        if not any(marker in cmdline for marker in _ORPHAN_MARKERS):
            continue
        if "app.worker" in cmdline or "app.main" in cmdline or "app.workerd" in cmdline:
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == os.getpid():
            # A live child of this worker: its own job is still using it.
            continue
        if ppid != 1 and Path(f"/proc/{ppid}").exists():
            # Owned by a live daemon/supervisor: not ours to kill.
            continue
        try:
            os.kill(pid, 9)
            killed.append(f"{pid} ({cmdline.strip()[:60]})")
            logger.warning(f"[worker] killed orphaned job process {pid}: {cmdline.strip()[:80]}")
        except OSError:
            continue
    # Stale temp audio: chunk WAVs are per-attempt scratch, never inputs to a
    # resume (the JSON checkpoints are). Anything older than a day is debris.
    cutoff = time.time() - 24 * 3600
    for pattern in ("asr-tmp/*.wav", "*/chunks/*.wav", "*/chunks/span_*.wav"):
        for path in Path(data_dir).glob(pattern):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    killed.append(f"temp audio {path.name}")
            except OSError:
                continue
    return killed


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
        # monotonic time of the last reconciliation pass (see
        # reconcile_stale_running's min_interval_sec throttle)
        self._reconciled_at = 0.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        # LOCKED_SPEC §6: never inherit the previous run's debris. Orphans from
        # a killed job hold the RAM the next attempt is budgeted against.
        try:
            cleaned = cleanup_orphans(self.settings.app.paths.data_dir)
            if cleaned:
                logger.info(f"[worker] startup cleanup removed {len(cleaned)} item(s)")
        except Exception as exc:  # noqa: BLE001 - cleanup must never block boot
            logger.warning(f"[worker] startup cleanup failed: {exc}")
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
        # Never queue the same id twice. Retry, the boot reconciler and the
        # auto-retry can all fire for one job, and a second entry means two
        # child processes writing the same stage artifacts concurrently — the
        # exact corruption the per-job process was introduced to prevent.
        #
        # Only `_pending` is consulted, deliberately. A job that is RUNNING is
        # not in the queue, and `retry()` needs to be able to queue it: it kills
        # the old process group first, but `_current_job` stays set until the
        # supervisor reaps it, so treating "current" as "already queued" would
        # silently swallow the retry and leave the job failed forever.
        with self._lock:
            if job_id in self._pending:
                logger.info(f"[worker] job {job_id} is already queued — not enqueuing again")
                return len(self._pending)
        self._queue.put(job_id)
        with self._lock:
            self._pending.append(job_id)
            depth = len(self._pending)
        self._write_queue_position(job_id, depth - 1)
        logger.info(f"[worker] job {job_id} queued (depth {depth})")
        return depth

    def is_queued(self, job_id: str) -> bool:
        """Is this job sitting in the queue right now?"""
        with self._lock:
            return job_id in self._pending

    def cancel(self, job_id: str) -> str:
        """Stop a job for good. Returns what was actually cancelled.

        Deleting a job must also stop the work, otherwise the user deletes a
        slow batch and the server keeps downloading, transcribing and spending
        LLM calls on a job whose record no longer exists.

        - "queued"   — dropped from the queue before it ever started
        - "running"  — its process group was SIGKILLed (ffmpeg children too)
        - "absent"   — nothing was in flight for this id
        """
        killed = False
        dropped = False
        with self._lock:
            if job_id in self._pending:
                self._pending.remove(job_id)
                dropped = True
            proc = self._active.get(job_id)
            if proc is None and self._current_job == job_id:
                proc = self._current_proc
        if proc is not None and proc.poll() is None:
            # Same signal path as retry: a negative rc marks the job
            # interrupted, and the tombstone stops it being written back.
            self._stalled_job = job_id
            self._stalled_jobs.add(job_id)
            self._kill_process(proc)
            killed = True
        if killed:
            logger.info(f"[worker] cancel: killed running job {job_id}")
            return "running"
        if dropped:
            # Everyone behind it moves up a place; the lock is not held here
            # because _promote_positions takes it itself.
            self._promote_positions()
            logger.info(f"[worker] cancel: dropped queued job {job_id}")
            return "queued"
        return "absent"

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
            # Unit counters belong to the attempt that died; carrying them over
            # showed "10 of 10 clips" beside a freshly reset 0% bar.
            "unitLabel": None,
            "unitsDone": None,
            "unitsTotal": None,
        })
        self.store.write_job(job)
        # A retry is a genuinely fresh process; stage artifacts on disk let it
        # resume from the first incomplete stage (see app/stages.py).
        return self.enqueue(job_id)

    def retry_from_cache(self, job_id: str) -> int:
        """Retry a job, resuming from the last checkpointed stage.

        Named separately from `retry` because that is the promise the API makes
        to the button: finished clips are on disk, so a retry re-renders only
        the unfinished ones instead of starting the pipeline over. The resume
        itself is done by the stage workspace, not by a different queue call —
        so this is the same enqueue with the contract spelled out at the call
        site.
        """
        return self.retry(job_id)

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

    def _queue_wait_estimate_sec(self, job_id: str, position: int) -> int | None:
        """How long until this queued job plausibly starts.

        The dashboard used to render "~any moment" for every queued job,
        because a queued job has no ETA of its own and `fmtEta` turns a missing
        number into "any moment". With one worker and a real queue in front of
        you, that is the single most misleading thing the card can say.

        So estimate it: whatever the running job still needs, plus the work of
        the jobs queued ahead of this one. Returns None when there is genuinely
        nothing to wait for (first in line, worker idle), which the UI can then
        honestly call "any moment".
        """
        with self._lock:
            pending = list(self._pending)
        ahead = pending[:position]
        running = [j for j, _ in self._targets() if j != job_id]
        if not ahead and not running:
            return None

        total = 0.0
        known = False
        for jid in running + ahead:
            other = self.store.get_job(jid)
            if not other:
                continue
            remaining = self._remaining_estimate_sec(other)
            if remaining is not None:
                total += remaining
                known = True
        return int(total) if known else None

    @staticmethod
    def _remaining_estimate_sec(job: dict[str, Any]) -> float | None:
        """Best available estimate of seconds left for one job."""
        stage_left = job.get("stageEtaSec")
        total_left = job.get("totalEtaSec")
        if job.get("stage") == "queued":
            return float(total_left) if isinstance(total_left, (int, float)) else None
        if isinstance(stage_left, (int, float)) and stage_left > 0:
            tail = job.get("totalEtaSec")
            # stageEtaSec is the current stage; totalEtaSec already includes
            # the stages after it, so prefer the larger, whole-job number.
            if isinstance(tail, (int, float)) and tail > 0:
                return float(max(stage_left, tail))
            return float(stage_left)
        if isinstance(total_left, (int, float)) and total_left > 0:
            return float(total_left)
        return None

    def _write_queue_position(self, job_id: str, position: int) -> None:
        """Persist the position so a client that never sees it move still knows."""
        job = self.store.get_job(job_id)
        if job is None:
            return
        wait = self._queue_wait_estimate_sec(job_id, position)
        if job.get("queuePosition") == position and job.get("queueEtaSec") == wait:
            return
        job["queuePosition"] = position
        # The wait estimate is part of the job's truth: the card shows it
        # instead of inventing "any moment" for a job with a queue in front.
        job["queueEtaSec"] = wait
        if position == 0 and wait is None:
            job["progressDetail"] = "Queued — starting now on the server worker."
        else:
            behind = "1 other job" if position == 1 else f"{position} other jobs"
            job["progressDetail"] = (
                f"Queued — waiting behind {behind}."
                if position > 0
                else "Queued — waiting for the current job to finish."
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
            # The job may have been deleted while it sat in the queue. Spawning
            # a process for it wastes a worker slot and a process launch on
            # work nobody is waiting for.
            if self.store.get_job(job_id) is None:
                logger.info(f"[worker] dropping queued job {job_id} — no longer exists")
                with self._lock:
                    if self._current_job == job_id:
                        self._current_job = None
                self._queue.task_done()
                continue
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
                # LOCKED_SPEC §6: after EVERY job, not just on failure. A
                # finished job can still leave a stuck ffmpeg or an ASR child
                # holding the model, and the next job is budgeted on the RAM
                # those processes are sitting in.
                try:
                    cleanup_orphans(self.settings.app.paths.data_dir)
                except Exception as exc:  # noqa: BLE001 - never break the queue
                    logger.warning(f"[worker] post-job cleanup failed: {exc}")
                self._queue.task_done()

    def _clear_queue_position(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None or job.get("queuePosition") is None:
            return
        job["queuePosition"] = None
        self.store.write_job(job)

    #: A job created by the upload route exists before its bytes do. It must
    #: never be handed to a runner in that state — the runner would look for a
    #: file called "pending-upload" and fail a job whose upload is still
    #: streaming. It waits, visibly, and expires if the upload never lands.
    WAITING_UPLOAD_STAGE = "waiting-upload"
    UPLOAD_WAIT_MAX_SEC = 30 * 60

    def _hold_for_upload(self, job_id: str, job: dict[str, Any]) -> bool:
        """True when this job must NOT start because its upload is missing."""
        if str(job.get("source") or "") != "pending-upload":
            return False
        created_ms = float(job.get("createdAt") or 0.0)
        age_sec = (time.time() * 1000 - created_ms) / 1000.0 if created_ms else 0.0
        if age_sec > self.UPLOAD_WAIT_MAX_SEC:
            msg = (
                "Waiting for upload — the file never arrived, so this job "
                "expired after 30 minutes. Attach the video again to start it."
            )
            fresh = self.store.get_job(job_id) or job
            fresh.update({
                "stage": "failed",
                "failedStage": None,
                "error": msg,
                "errorDetail": None,
                "interrupted": False,
                "progress": 0,
                "completedAt": int(time.time() * 1000),
            })
            from app import jobstatus

            jobstatus.describe(fresh)
            self.store.write_job(fresh)
            self.store.append_job_log(job_id, f"FAILED: {msg}")
            return True
        if job.get("stage") != self.WAITING_UPLOAD_STAGE:
            job.update({
                "stage": self.WAITING_UPLOAD_STAGE,
                "progress": 0,
                "progressDetail": (
                    "Waiting for upload — the job starts by itself when the "
                    "file is attached (expires in 30 minutes)."
                ),
            })
            self.store.write_job(job)
            self.store.append_job_log(
                job_id, "Waiting for upload — no file yet, so no stage was started"
            )
        return True

    def _run_job_process(self, job_id: str) -> None:
        """Run one job in a child process; supervise exit; surface crashes."""
        job_now = self.store.get_job(job_id)
        if job_now is not None and self._hold_for_upload(job_id, job_now):
            return
        # Preflight BEFORE the child process exists: a missing binary or a
        # starved host is a message with a fix, not a crashed job.
        if self._preflight_environment(job_id, job_now):
            return
        log_dir = self.settings.app.paths.data_dir / "work" / job_id
        log_dir.mkdir(parents=True, exist_ok=True)
        runner_log = log_dir / "runner.log"

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # The supervisor watches app/ for edits. Bytecode written into
        # app/__pycache__ is noise on that tree, so keep the runner's bytecode
        # in memory — a job must never be able to look like a source change.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # start_new_session: the child gets its own process group so the
        # watchdog can kill the ffmpeg descendants along with the runner.
        #
        # `nice -n 15`: transcription and x264 are pure CPU, and on a 1-vCPU
        # host an unpinned runner competes with the API for the only core. At
        # nice 15 the kernel always prefers the API, so a job can never make
        # /api/health time out while the process is alive (the platform proxy
        # reports that as an empty 500).
        prefix: list[str] = []
        if sys.platform != "win32" and shutil.which("nice"):
            prefix = ["nice", "-n", "15"]
        proc = subprocess.Popen(
            [*prefix, sys.executable, "-m", "app.runner", job_id],
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
        # The runner writes terminal-shaped events to STDOUT; we replay them
        # into the job doc so the supervisor reads explicit events instead of
        # inferring success or failure from the exit code.
        def _pump() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self.store.append_job_log(job_id, line[:500])
                        self._replay_runner_event(job_id, line)
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

        # Section 4: explicit events — the runner writes `stage:`, `done` and
        # `failed` to STDOUT; the supervisor reads them. An exit (any code)
        # before `done` or `failed` is a failure, never a 0-success inference.
        self._replay_events(job_id)

        # Section 3: ring buffer of the last 500 log lines, flushed every
        # second and on exit, so "View logs" and "Show details" never go empty.
        self._log_ringbuffer(job_id)

        try:
            if runner_log.exists():
                lines = runner_log.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-60:])
        except Exception:
            tail = ""

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
        if stage in {"complete", "failed", "paused"}:
            # A pause is a user decision, not a failure: never auto-retry it.
            if stage == "paused":
                return
            # A MEMORY failure is environmental, and the in-process ladder has
            # already spent its cheaper rungs. Clean up whatever the dead
            # attempt left behind and re-queue the job (max 3), which resumes
            # from the chunk checkpoints on disk instead of from zero.
            if (job or {}).get("errorKind") == "memory" and self._memory_requeue(job_id, job or {}):
                return
            # A retryable failure (render deadline, stall watchdog) gets ONE
            # automatic second attempt before the user is told anything. Both
            # are environmental — a hung ffmpeg or a starved host — and the
            # pipeline resumes from the last COMPLETED clip, so a retry is
            # cheap: it does not repeat the download, transcription or the paid
            # detection call, only the render that actually failed.
            if self._auto_retry(job_id, job or {}, "stall/render deadline"):
                return
            return
        # the child died without recording a terminal state (crash, OOM-kill,
        # SIGSEGV in native code...) — surface it instead of hanging
        tail = ""
        try:
            if runner_log.exists():
                tail = "\n".join(runner_log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
        except Exception:
            pass
        # failed. That is an external event (host restart/redeploy/scale-down),
        # so the job did not fail on its own merit — treat it as interrupted and
        # resumable. Without this, a host bounce permanently fails work whose
        # expensive stages are already cached on disk.
        # A negative return code means the child was SIGNALLED, not that it
        if rc is not None and rc < 0 and self._note_interrupted(job, job_id, rc, stage, tail):
            return

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
                # keep the stage the child died in, for the dashboard banner
                "failedStage": stage,
                # a later hard failure supersedes any earlier interrupt flag
                "interrupted": False,
                "error": msg,
                "errorDetail": detail[:4000],
                "completedAt": int(time.time() * 1000),
            })
            self.store.write_job(job)
        self.store.append_job_log(job_id, f"FAILED: {msg}")

    # ------------------------------------------------------------------
    # watchdog
    # ------------------------------------------------------------------

    @staticmethod
    def has_signal_failure(job: dict[str, Any]) -> bool:
        """True when a job's recorded FAILURE was really an external kill.

        Jobs killed by SIGTERM/SIGKILL before the interrupted/resumed
        distinction existed were stamped ``stage: failed`` with a message like
        "exited unexpectedly (code -15)". Their stage cache is intact on disk,
        so they are recoverable — this is what lets boot reconciliation rescue
        them instead of leaving them dead forever.

        Deliberately narrow: it matches the signalled-failure messages this
        module writes, not the word "code" appearing anywhere in a traceback.
        """
        if job.get("stage") != "failed":
            return False
        text = f"{job.get('error') or ''}\n{job.get('errorDetail') or ''}"
        return bool(_SIGNAL_BANNER.search(text))

    @staticmethod
    def has_delivered_output(job: dict[str, Any], rendered: int, promised: int) -> bool:
        """True when a job marked ``failed`` actually delivered its whole batch.

        Live case: job ``c57d7782c548`` rendered all 10 of its clips, then a
        LATER re-detect attempt hit a Groq 429 and ``_fail()`` overwrote
        ``stage=complete`` with ``stage=failed``. The clips were all on disk and
        downloadable the whole time, so the dashboard advertised a failure the
        user could do nothing about and did not need to act on.

        A failure that cost no output is not a failed batch. Requiring the FULL
        set matters: a run that lost clips really did fail, and saying
        "complete" there would hide a genuine hole.
        """
        return (
            job.get("stage") == "failed"
            and promised > 0
            and rendered >= promised
        )

    #: Failures worth exactly one silent automatic retry. A hung ffmpeg and a
    #: starved host are both environmental and often transient; making the user
    #: click Retry for a hang that a second attempt would have cleared is a
    #: poor trade. Everything else (a bad URL, a missing key, a genuinely broken
    #: input) fails immediately — a retry would just spend quota to reach the
    #: same error.
    AUTO_RETRY_ERRORS = (
        "render deadline",
        "was killed",
        "stalled",
        "no progress",
        "exceeded its",
        # The failure this host actually produces most often. ffmpeg reports
        # `exited -9 (SIGKILL) — this was an EXTERNAL kill` when the container
        # is recycled or redeployed mid-render, so without these tokens the one
        # failure most worth retrying silently did NOT retry.
        "sigkill",
        "external kill",
        "terminated by a signal",
        "interrupted",
    )
    AUTO_RETRY_BUDGET = 1

    @classmethod
    def _is_retryable(cls, job: dict[str, Any]) -> bool:
        blob = " ".join(
            str(job.get(f) or "") for f in ("error", "errorDetail", "progressDetail")
        ).lower()
        return any(token in blob for token in cls.AUTO_RETRY_ERRORS)

    def pause(self, job_id: str) -> dict[str, Any]:
        """Ask a job to stop at its next safe boundary.

        Cooperative, and deliberately so. A child process cannot be interrupted
        part-way through writing an ffmpeg output or a stage artifact without
        risking exactly the corruption the user is trying to avoid, so the flag
        is polled by the pipeline between units of work (transcript chunks,
        detection chunks, rendered clips) and the job stops itself.

        The response says which it is: `paused` once the child has actually
        stopped, `pausing` while it is still finishing the current unit. That
        distinction matters — a single ffmpeg encode of a long clip can take
        minutes, and claiming "paused" instantly would be a lie.
        """
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.get("stage") in {"complete", "failed"}:
            raise ValueError(f"job is {job.get('stage')} — nothing to pause")
        job.update({
            "pauseRequested": True,
            "progressDetail": "Pause requested — stopping at the next safe checkpoint…",
        })
        self.store.write_job(job)
        self.store.append_job_log(
            job_id,
            "PAUSE requested — the job will stop between units of work "
            "(no file is left half-written).",
        )
        # A job that is queued but has no live process can be stopped outright:
        # nothing is in flight, so there is no write to interrupt. This matters
        # because a host restart can leave a job marked "queued" before the boot
        # reconciler enqueues it — reporting "pausing" there would leave the
        # user waiting forever for a pause that never lands.
        with self._lock:
            in_pending = job_id in self._pending
            if in_pending:
                self._pending.remove(job_id)
            running = self._current_job == job_id and self._current_proc is not None \
                and self._current_proc.poll() is None
        if in_pending or (job.get("stage") == "queued" and not running):
            if in_pending:
                self._promote_positions()
            job = self.store.get_job(job_id) or job
            job.update({
                "stage": "paused",
                "pausedStage": "queued",
                "pauseRequested": False,
                "pausedAt": int(time.time() * 1000),
                "progressDetail": "Paused by user before it started — resume picks it up here.",
            })
            self.store.write_job(job)
            self.store.append_job_log(job_id, "PAUSED while queued — nothing had started")
            return {"state": "paused", "stage": "queued"}
        return {"state": "pausing", "stage": job.get("stage")}

    def resume(self, job_id: str) -> int:
        """Resume a paused job. Reuses the ordinary stage-marker resume.

        Nothing about the pipeline changes: the download, transcript chunks,
        per-chunk detection results and finished clips are all still on disk, so
        the runner picks up at the first incomplete stage — and inside a stage,
        at the first unfinished chunk or clip.
        """
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        job.update({
            "stage": "queued",
            "pauseRequested": False,
            "pausedStage": None,
            "progress": 0,
            "error": None,
            "errorDetail": None,
            "completedAt": None,
            "interrupted": False,
            "interruptReason": None,
            "failedStage": None,
            "unitLabel": None,
            "unitsDone": None,
            "unitsTotal": None,
            "progressDetail": "Resuming from the last completed checkpoint…",
        })
        self.store.write_job(job)
        self.store.append_job_log(
            job_id,
            "RESUME — continuing from cached stages/chunks; completed work is not redone.",
        )
        return self.enqueue(job_id)

    def _auto_retry(self, job_id: str, job: dict[str, Any], reason: str) -> bool:
        """Requeue a stalled/timed-out job once. True when a retry was started.

        The retry is a fresh child process, and the pipeline resumes from the
        last COMPLETED clip — the download, the transcript and the (paid)
        detection call are all cached on disk, so the second attempt only
        redoes the render that actually failed.
        """
        used = int(job.get("autoRetryCount") or 0)
        if used >= self.AUTO_RETRY_BUDGET or not self._is_retryable(job):
            return False
        job.update({
            "autoRetryCount": used + 1,
            "stage": "queued",
            "progress": 0,
            "error": None,
            "errorDetail": None,
            "completedAt": None,
            "interrupted": False,
            "interruptReason": None,
            "failedStage": None,
            "unitLabel": None,
            "unitsDone": None,
            "unitsTotal": None,
            "progressDetail": "Auto-retrying once after a render stall/timeout…",
        })
        self.store.write_job(job)
        self.store.append_job_log(
            job_id,
            f"AUTO-RETRY 1/1: {reason} — requeueing automatically; the pipeline "
            f"resumes from the last completed clip.",
        )
        logger.warning(f"[worker] auto-retrying job {job_id} after {reason}")
        self.enqueue(job_id)
        return True

    def _note_interrupted(self, job: dict[str, Any] | None, job_id: str, rc: int,
                          stage: str | None, tail: str) -> bool:
        """Flag a signalled job as interrupted and re-queue it.

        Returns False once the resume budget is spent, so the caller falls
        through to the ordinary hard-failure path (we must not loop forever).

        ``stage`` is deliberately left UNCHANGED: it stays where the child died,
        so the stage-workspace resume picks up from the last completed stage
        instead of starting over. That is the whole point — transcription on a
        15-minute source is the expensive part and it is already on disk.
        """
        if job is None:
            return False
        count = int(job.get("interruptCount") or 0) + 1
        if count > INTERRUPT_MAX_RESUMES:
            return False
        try:
            how = signal.Signals(-rc).name
        except ValueError:
            how = f"signal {-rc}"
        reason = (
            f"Host terminated the worker with {how} during '{stage}' — this is an "
            f"external kill (restart/redeploy), not a pipeline error."
        )
        job.update({
            "interruptCount": count,
            "interrupted": True,
            "interruptReason": reason,
            "error": None,
            "progressDetail": (
                f"Interrupted by a host restart — resuming "
                f"(attempt {count} of {INTERRUPT_MAX_RESUMES})"
            ),
        })
        self.store.write_job(job)
        self.store.append_job_log(
            job_id,
            f"INTERRUPTED: worker killed by {how} during '{stage}' — auto-resuming "
            f"(attempt {count} of {INTERRUPT_MAX_RESUMES})",
        )
        if tail:
            logger.debug(f"[worker] {job_id} last runner output before interrupt: {tail[-400:]}")
        logger.warning(
            f"[worker] job {job_id} interrupted by {how} during '{stage}' — "
            f"re-queuing (attempt {count}/{INTERRUPT_MAX_RESUMES})"
        )
        self.enqueue(job_id)
        return True

    def _memory_requeue(self, job_id: str, job: dict[str, Any]) -> bool:
        """Re-queue a job that died of memory, after cleaning up its debris."""
        count = int(job.get("memoryAttempts") or 0)
        if count >= MEMORY_MAX_JOB_ATTEMPTS:
            return False
        try:
            cleanup_orphans(self.settings.app.paths.data_dir)
        except Exception as exc:  # noqa: BLE001 - requeue regardless
            logger.warning(f"[worker] cleanup before memory retry failed: {exc}")
        job.update({
            "memoryAttempts": count + 1,
            "stage": "queued",
            "failedStage": None,
            "error": None,
            "errorDetail": None,
            "errorKind": None,
            "completedAt": None,
            "progress": 0,
            "progressDetail": (
                f"Ran out of memory — retrying with a lighter setting "
                f"(attempt {count + 1} of {MEMORY_MAX_JOB_ATTEMPTS}); finished "
                "chunks are saved"
            ),
            "interrupted": True,
            "interruptReason": "Ran out of memory while transcribing — resuming from the last saved chunk.",
        })
        for field in ("stagePlan", "unitsDone", "unitsTotal", "unitLabel",
                      "stageEtaSec", "totalEtaSec"):
            job.pop(field, None)
        self.store.write_job(job)
        self.store.append_job_log(
            job_id,
            f"MEMORY: re-queued automatically (attempt {count + 1} of "
            f"{MEMORY_MAX_JOB_ATTEMPTS}) — chunk checkpoints are reused",
        )
        logger.warning(
            f"[worker] job {job_id} hit a memory failure — re-queued "
            f"(attempt {count + 1}/{MEMORY_MAX_JOB_ATTEMPTS})"
        )
        self.enqueue(job_id)
        return True

    def reconcile_stale_running(
        self, *, min_interval_sec: float = 0.0, force: bool = False
    ) -> list[str]:
        """Clear jobs that claim to be running with no worker behind them.

        A job's `stage` is written by a CHILD process. If that process is
        SIGKILLed, OOM-reaped or lost to a host restart, nothing overwrites
        `stage` — the document is left saying "finding, 61%" forever, with no
        process and no queue entry. That is the state behind "Stage 3 of 4,
        running" on a job that had already failed.

        The worker knows exactly which jobs it is running and which are queued,
        so it is the authority: anything on an active stage that is in neither
        set is stale by definition. Reconciled on every read rather than by a
        timer, so the dashboard is right the moment it is looked at.

        Terminal jobs are never touched — a genuinely running job is not
        affected, and a paused/failed/complete one is already settled.

        ``min_interval_sec`` skips the pass when the previous one is newer than
        that, because the dashboard polls the list every couple of seconds and
        the answer cannot change faster than the queue does. ``force=True``
        always runs — the "Check now" path must be able to tell the truth on
        demand.
        """
        if not force and min_interval_sec > 0:
            now = time.monotonic()
            with self._lock:
                if now - self._reconciled_at < min_interval_sec:
                    return []
                self._reconciled_at = now
        else:
            with self._lock:
                self._reconciled_at = time.monotonic()
        with self._lock:
            live = set(self._active) | set(self._pending)
        fixed: list[str] = []
        for job in self.store.list_jobs():
            if job.get("_id") in live:
                continue
            from app import jobstatus  # local: avoids an import cycle at module load

            if jobstatus.is_terminal(job):
                continue
            job_id = job["_id"]
            # Re-read before acting: the listing above is a snapshot, and the
            # runner may have written its own terminal state in between. A stale
            # snapshot written back here is how a precise reason ("the site asked
            # this server to sign in") was replaced by this generic guess, with
            # the child's failure line still sitting right above it in the log.
            fresh = self.store.get_job(job_id)
            if fresh is None:
                continue
            job = fresh
            if jobstatus.is_terminal(job):
                continue
            stage = job.get("stage")
            # A job that already carries a reason (the runner reported back, the
            # stage guard refused to start, an upload went missing) keeps it.
            # Only a job with no explanation at all gets the generic one: its
            # message is what tells the user whether to retry, re-upload or set
            # cookies, and a guess would throw that away.
            reported_error = str(job.get("error") or "").strip() or _reason_from_logs(job)
            # LOCKED_SPEC §4: a job that was mid-flight with no worker behind it
            # and NO reason of its own was INTERRUPTED (host restart, redeploy,
            # OOM kill), not broken. Bump its attempt counter, mark it
            # interrupted, put it back on the queue, and let the cached stages
            # resume it — the user never presses Retry. A job the runner itself
            # explained keeps that explanation and fails as before, and the
            # 3-attempt budget stops a crash loop from requeueing forever.
            if not reported_error and str(stage or "") != "paused":
                if self._note_interrupted(job, job_id, -signal.SIGKILL, str(stage or "queued"), ""):
                    logger.warning(
                        f"[worker] stale job {job_id} was '{stage}' with no worker "
                        f"— marked interrupted and re-queued (attempt "
                        f"{job.get('interruptCount')}/{INTERRUPT_MAX_RESUMES})"
                    )
                    fixed.append(job_id)
                    continue
            silent = int((job.get("lastProgressAt") or job.get("stageStartedAt")
                          or job.get("createdAt") or 0))
            job.update({
                "stage": "failed",
                "failedStage": job.get("failedStage") or stage,
                "error": reported_error or (
                    "The worker stopped while this job was in progress and never "
                    "reported back. Completed stages are cached, so Retry resumes "
                    "from the last finished one."
                ),
                "interrupted": False,
                "interruptReason": None,
                "completedAt": job.get("completedAt") or silent or None,
            })
            # Clear the running-shaped fields through the same derivation the
            # API uses, so the stored document matches what the API serves.
            jobstatus.describe(job)
            for field in ("stagePlan", "unitsDone", "unitsTotal", "unitLabel",
                          "stageEtaSec", "totalEtaSec", "queueEtaSec",
                          "queuePosition", "progressDetail"):
                job.pop(field, None)
            job["progress"] = 0
            self.store.write_job(job)
            self.store.append_job_log(
                job_id,
                f"RECONCILED: job was marked '{stage}' but no worker held it — "
                f"cleared to failed (stale running state)"
                + ("; reason kept from the runner" if reported_error else ""),
            )
            logger.warning(
                f"[worker] reconciled stale job {job_id}: stage '{stage}' with no "
                f"active worker and no queue entry"
                + ("" if not reported_error else " (kept the reason the runner reported)")
            )
            fixed.append(job_id)
        return fixed

    def mark_in_flight_interrupted(self) -> list[str]:
        """Flag every job with a live child as interrupted, for the next boot.

        Called from the app's shutdown hook. Best-effort by nature — the host may
        SIGKILL us before this runs — but when we do get a graceful shutdown it
        turns the next boot's re-queue into an informed resume instead of a
        guess about what was happening when the lights went out.
        """
        flagged: list[str] = []
        for job_id, proc in self._targets():
            if proc.poll() is not None:
                continue
            job = self.store.get_job(job_id)
            if job is None or job.get("stage") in {"complete", "failed"}:
                continue
            job.update({
                "interrupted": True,
                "interruptReason": "Host shut down while this job was running — it will resume.",
                "progressDetail": "Host restarting — this job will resume automatically.",
            })
            self.store.write_job(job)
            self.store.append_job_log(
                job_id, "INTERRUPTED: host shutting down mid-job — will resume on next boot"
            )
            flagged.append(job_id)
        if flagged:
            logger.warning(f"[worker] flagged {len(flagged)} in-flight job(s) as interrupted before shutdown")
        return flagged

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
                "failedStage": job.get("stage"),
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
            # Same rule as a render deadline: one silent automatic retry before
            # the user is involved. The pipeline resumes from the last
            # COMPLETED clip, so this costs only the render that hung.
            self._auto_retry(job_id, job, "stall watchdog")
            killed_any = True
        return killed_any

    def _watchdog(self) -> None:
        while not self._stop.wait(5.0):
            self._check_watchdog_once()



    # Instance method, not a staticmethod: it reads and writes the job store
    # (stage transitions + verbatim stdout). As a staticmethod every call
    # raised NameError on `self` and the whole branch was silently swallowed by
    # the `except Exception: pass` below, so runner events never landed.
    def _replay_runner_event(self, job_id: str, line: str) -> None:
        """Plain STDOUT line from the runner: land stage events and everything
        else verbatim in the job log so "View logs" shows the real worker output."""
        try:
            import json as _json
            try:
                ev = _json.loads(line)
            except Exception:
                ev = None
            if isinstance(ev, dict):
                stage = ev.get("stage")
                if stage:
                    def apply(job: dict, stage=stage, ev=ev) -> None:
                        # A replayed line describes what the child was doing
                        # SECONDS ago, and the child may have failed since. This
                        # is how job 0d9f0a3a ended up 'grabbing' again with its
                        # error gone, one second after the child reported it.
                        if str(job.get("stage")) in _TERMINAL_STAGES:
                            return
                        if stage in _TERMINAL_STAGES:
                            job["stage"] = stage
                            if stage == "failed":
                                job["completedAt"] = int(time.time() * 1000)
                        elif stage == "stage_started":
                            job["stage"] = ev.get("stage_name", job.get("stage", ""))
                            job["stageStartedAt"] = int(time.time() * 1000)
                            job["progress"] = ev.get("progress", job.get("progress"))
                            job["progressDetail"] = ev.get(
                                "progressDetail",
                                job.get("progressDetail", ""),
                            )
                        elif stage == "stage_done":
                            job["progress"] = ev.get("progress", job.get("progress"))
                            job["progressDetail"] = ev.get(
                                "progressDetail",
                                job.get("progressDetail", ""),
                            )

                    self.store.update_job(job_id, apply)
            # Plain (non-JSON) runner output lands verbatim in the job log so
            # "View logs" always shows the real worker output.
            self.store.append_job_log(job_id, line[:500])
        except Exception:
            pass

    def _replay_events(self, job_id: str) -> None:
        """Read `stage:`, `done` and `failed` lines the runner wrote."""
        path = self.settings.app.paths.data_dir / "work" / job_id / "events.jsonl"
        if not path.exists():
            return
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                stage = ev.get("stage")
                if stage == "failed":
                    msg = str(ev.get("error", "unknown"))
                    detail = str(ev.get("errorDetail", ""))[:4000]
                    if not detail:
                        detail = msg
                    job = self.store.get_job(job_id)
                    if job is not None:
                        job.update({
                            "stage": "failed",
                            "failedStage": ev.get("failedStage"),
                            "error": msg,
                            "errorDetail": detail,
                            "interrupted": False,
                            "completedAt": int(time.time() * 1000),
                        })
                        self.store.write_job(job)
                    self.store.append_job_log(job_id, f"FAILED: {msg}")
                elif stage == "complete":
                    job = self.store.get_job(job_id)
                    if job is not None:
                        job["stage"] = "complete"
                        job["completedAt"] = int(time.time() * 1000)
                        self.store.write_job(job)
                elif stage == "failed" or stage == "done":
                    continue
        except Exception:
            pass

    def _log_ringbuffer(self, job_id: str, max_lines: int = 500) -> None:
        """Keep the last 500 log lines in the job doc, flushed every second."""
        try:
            def apply(job: dict) -> None:
                lines = job.get("logs")
                if not isinstance(lines, list):
                    lines = [str(entry) for entry in lines] if lines else []
                # Only the log field is touched, on the document read under the
                # lock — writing a whole snapshot back here would be another
                # chance to undo the child's terminal state.
                job["logs"] = lines[-max_lines:]

            self.store.update_job(job_id, apply)
        except Exception:
            pass

    def _preflight_environment(self, job_id: str, job: dict[str, Any] | None = None) -> list[str]:
        """Everything that must be true BEFORE a job starts (LOCKED_SPEC §4).

        Returns the problems found (empty when the job may start). Each problem
        names the exact cause and the fix. Called before the runner is spawned:
        the old version ran AFTER the child had already exited, and swallowed
        every exception it raised, so a missing binary surfaced as a generic
        "worker stopped" instead of a fixable message.
        """
        import shutil as _shutil

        from app import hosting as _hosting
        from app import memguard as _memguard

        problems: list[str] = []
        missing = [name for name in ("yt-dlp", "ffmpeg") if _shutil.which(name) is None]
        if missing:
            problems.append(
                "missing tool(s) in PATH: " + ", ".join(missing)
                + " — install them (apt-get install -y ffmpeg yt-dlp) and retry"
            )
        # Presence only: never read, compare or log a key's value.
        if not any(os.environ.get(name) for name in (
            "GROQ_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
            "GOOGLE_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY",
        )):
            problems.append(
                "no detection provider key is set — add one in the Keys tab "
                "(GROQ_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY or NVIDIA_API_KEY)"
            )
        try:
            free = _hosting.disk_stats(self.settings.app.paths.data_dir).get("freeBytes")
            if free is not None and int(free) < 100 * 1024 * 1024:
                problems.append(
                    f"only {int(free) // (1024 * 1024)} MB of disk free — free space "
                    "(delete finished jobs) and retry; the render needs at least 100 MB"
                )
        except Exception:
            pass
        admission = _memguard.check(getattr(self.settings.whisper, "model", None))
        if not admission.ok and admission.message:
            problems.append(admission.message)
        if not problems:
            return []
        doc = self.store.get_job(job_id) or job
        if doc is not None:
            msg = "Worker could not start this job — " + " | ".join(problems)
            doc.update({
                "stage": "failed",
                "failedStage": "fetch",
                "error": msg,
                "errorDetail": msg,
                "interrupted": False,
                "completedAt": int(time.time() * 1000),
            })
            try:
                from app import jobstatus

                jobstatus.describe(doc)
            except Exception:
                pass
            self.store.write_job(doc)
            self.store.append_job_log(job_id, f"FAILED: {msg}")
        return problems

    # ------------------------------------------------------------------
    # status for the dashboard card
    # ------------------------------------------------------------------

    def active_pids(self) -> dict[str, int]:
        """job_id -> pid of the child running it (telemetry, one RPC away)."""
        with self._lock:
            return {jid: proc.pid for jid, proc in self._active.items()
                    if proc.poll() is None}

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
