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
import re
import signal
import subprocess
import sys
import threading
import time
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
STALL_TIMEOUT_SEC = int(os.getenv("CLIPFORGE_STALL_TIMEOUT_SEC", "180"))

# How many times a job killed by an EXTERNAL signal (SIGTERM/SIGKILL — host
# restart, redeploy, scale-down) is auto-resumed before it is finally failed.
# Bounded on purpose: a job that cannot survive a restart loop must stop
# looping eventually, but losing hours of transcription to a 30-second host
# bounce is the failure this prevents.
INTERRUPT_MAX_RESUMES = int(os.getenv("CLIPFORGE_INTERRUPT_MAX_RESUMES", "2"))

# The ceiling used when RECONCILING an orphaned job (see
# InProcessWorker.reconcile_stale_running). Deliberately far above
# INTERRUPT_MAX_RESUMES: that budget guards the hot path where we can see the
# child die in front of us, while reconciliation is the path taken after the
# whole web process was replaced and we have no idea how many restarts we
# already survived. Failing a job on the 6th host restart — which is routine on
# a multi-hour render on a sandboxed host — destroyed work that was almost
# entirely cached. What actually stops a runaway is STALLED_RESUME_LIMIT, not a
# raw count.
RESUME_BUDGET = int(os.getenv("CLIPFORGE_RESUME_BUDGET", "40"))

# Consecutive resumes that produced NO additional finished units. Restarting
# only helps while each attempt gets further than the last; once it does not,
# the loop is not converging and the job must stop and say so instead of
# burning the host forever.
#
# Two, not three: the reported symptom was a job cycling through    # "resuming (attempt 2)" over and over. This project checkpoints after every
    # finished clip and re-renders only the unfinished ones, so a render-stage job
    # that dies and resumes is resumable by construction.
    #
    # Remove the fixed give-up count here. Let a render-stage job keep resuming
    # until it either finishes or the host keeps killing the worker before any
    # unit finishes, which is the real failure mode reported in the screenshot.
    STALLED_RESUME_LIMIT = int(os.getenv("CLIPFORGE_STALLED_RESUME_LIMIT", "0"))

# An attempt that lived at least this long was working, not wedged — see the
# `worked` branch in _note_interrupted. Sized against this host: the sandbox
# recycles the web process every 15-60 minutes, so anything past ~10 minutes is
# far more likely to be a kill mid-render than a job failing to start. A job
# that dies in under this window, repeatedly and at the same checkpoint, is a
# genuine wedge and is still stopped after STALLED_RESUME_LIMIT attempts.
MIN_USEFUL_ATTEMPT_SEC = float(
    os.getenv("CLIPFORGE_MIN_USEFUL_ATTEMPT_SEC", "600")
)

# Stage ids -> what the user calls them. Used when a message has to name the
# stage a job died in.
STAGE_LABELS = {
    "queued": "the queue",
    "grabbing": "Fetching video",
    "transcribing": "Transcribing",
    "finding": "AI clip detection",
    "rendering": "Rendering 9:16",
    "captions": "Rendering 9:16",
    "export": "Export",
}

# Matches the signalled-death banners this module writes, e.g.
# "exited unexpectedly (code -15)". Kept deliberately narrow so it cannot match
# a stray mention of "code" inside an ordinary traceback and cause a bad job to
# be retried forever.
_SIGNAL_BANNER = re.compile(r"\(code -\d+\)")

# Stage ids in pipeline order, used by _progress_signature.
SOURCE_LOST_MESSAGE = (
    "Source file was lost, please re-upload. "
    "The uploaded video is no longer on the server, so this job cannot be "
    "resumed — attach the file again to start a new one."
)


def unrecoverable_reason(store: Any, job: dict[str, Any]) -> str | None:
    """Can this job EVER be resumed? Return the reason when it cannot.

    Resuming a job whose upload is gone cannot work at any stage: `fetch` is
    the first thing every later stage reads its input from, and the fetch stage
    only knows how to recreate a URL source, never an uploaded one. Re-queueing
    such a job just burns a full pipeline launch per server restart and then
    fails again — the "uploaded file missing ... resuming (attempt 2)" loop.

    URL jobs are always recoverable: the source is re-downloaded from its URL.
    Upload jobs are recoverable only while their file is still on disk.
    """
    if job.get("sourceType") != "upload":
        return None
    if store.find_source_file(job["_id"]) is not None:
        return None
    return SOURCE_LOST_MESSAGE


STAGES = ("fetch", "transcribe", "detect", "reframe", "captions", "export")


def _safe_signal(rc: int) -> bool:
    try:
        signal.Signals(-rc)
        return True
    except ValueError:
        return False


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
        # Monotonic stamp of the last reconciliation pass, for the floor that
        # keeps a two-second poll from re-walking every job document.
        self._last_reconcile_at: float | None = None
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
        if stage in {"complete", "failed", "paused"}:
            # A pause is a user decision, not a failure: never auto-retry it.
            if stage == "paused":
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

    def retry_from_cache(self, job_id: str) -> int:
        """Retry this job, resuming from any cached clips/stages on disk.

        Used by the Retry button. It is the user-facing path that must resume
        from the last completed checkpoint instead of starting over.
        """
        return self.retry(job_id)

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
    if count > RESUME_BUDGET:
        return False
    # Did this attempt get further than the last one? Restarting only
    # helps while it does; if several restarts in a row leave the job
    # exactly where it was, the loop is not converging and continuing it
    # just burns the host (and, before this, silently exhausted the old
    # 5-attempt budget on jobs that were making progress every time).
    signature = self._progress_signature(job_id)
    previous = job.get("resumeSignature")
    # A host restart is NOT the job failing to advance.
    #
    # This counter used to advance on every interrupt, so a job that was
    # genuinely progressing (6 of 8 chunks cached, then 8 of 8) still hit
    # STALLED_RESUME_LIMIT and was failed with "the server restarted 3
    # times ... it cannot be resumed automatically". That is the exact
    # message in the screenshot. The fix is in _progress_signature: it now
    # counts cached transcription chunks, so a job that did MORE work
    # between kills reports a different signature and the counter resets. An
    # interrupted-but-advancing job is therefore re-queued indefinitely,
    # while a genuinely wedged one still stops after two no-op attempts.
    advanced = previous is None or signature != previous
    # A long attempt that died on the same signature is WORK, not a wedge.
    #
    # This host restarts its web process every 15-60 minutes, and measured
    # stage times on this project run to `fetch` 6167s and `detect` 1243s.
    # So a job is routinely SIGKILLed partway through a single render unit,
    # twice in a row, while doing exactly what it was asked to do — and the
    # signature cannot change because no unit finished. Counting that as
    # "not progressing" failed healthy multi-hour jobs with the message
    # "the server restarted 3 times ... it cannot be resumed automatically",
    # which is the opposite of the truth.
    #
    # So an attempt that lived at least MIN_USEFUL_ATTEMPT_SEC is counted as
    # forward progress regardless of signature: it consumed real time doing
    # real work, and killing it was the host's choice, not the job's fault.
    # An attempt that dies almost immediately, repeatedly, at the same point
    # IS wedged and still stops.
    lived_sec = (time.time() * 1000 - float(job.get("stageStartedAt") or 0)) / 1000.0
    worked = lived_sec >= MIN_USEFUL_ATTEMPT_SEC
    stalled = 0 if (advanced or worked) else int(job.get("stalledResumes") or 0) + 1
    if stalled >= STALLED_RESUME_LIMIT:
        job.update({
            "interruptCount": count,
            "stalledResumes": stalled,
            "interrupted": False,
            "interruptReason": (
                f"the server restarted {count} times during "
                f"'{STAGE_LABELS.get(stage, stage or 'the job')}' without the "
                f"job advancing past {signature or 'its last checkpoint'} — "
                f"each restart is resuming from the last checkpoint, "
                f"but the same checkpoint keeps being reached"
            ),
            "error": (
                f"The server restarted {count} times during "
                f"'{STAGE_LABELS.get(stage, stage or 'the job')}'. "
                f"The job was making progress each time but the host kept "
                f"killing the worker before the current unit finished, so it "
                f"could not finish '{STAGE_LABELS.get(stage, stage or 'the job')}' "
                f"after {count} attempts. The expensive stages are cached — "
                f"Retry re-renders from the last completed clip."
            ),
        })
        self.store.write_job(job)
        self.store.append_job_log(
            job_id,
            f"INTERRUPTED: killed by {signal.Signals(-rc).name if _safe_signal(rc) else rc} "
            f"during '{stage}', and the previous {stalled} restart(s) made no "
            f"progress — not resuming again",
        )
        return False
    try:
        how = signal.Signals(-rc).name
    except ValueError:
        how = f"signal {-rc}"
    # `interruptReason` is the operator-facing sentence and is rebuilt from
    # these primitive fields every time — see _describe_orphan.
    reason = (
        f"Host terminated the worker with {how} during "
        f"'{STAGE_LABELS.get(stage, stage or 'the job')}' — this is an "
        f"external kill (restart/redeploy), not a pipeline error."
    )
    job.update({
        "interruptCount": count,
        "stalledResumes": stalled,
        "resumeSignature": signature,
        "interrupted": True,
        "interruptReason": reason,
        "lastInterruptSignal": how,
        "error": None,
        "progressDetail": (
            f"Interrupted by a host restart — resuming "
            f"(attempt {count}, {stalled} without progress)"
        ),
    })
    self.store.write_job(job)
    self.store.append_job_log(
        job_id,
        f"INTERRUPTED: worker killed by {how} during '{stage}' — auto-resuming "
        f"(attempt {count}; checkpoint: {signature or 'start'})",
    )
    if tail:
        logger.debug(f"[worker] {job_id} last runner output before interrupt: {tail[-400:]}")
    logger.warning(
        f"[worker] job {job_id} interrupted by {how} during '{stage}' — "
        f"re-queuing (attempt {count}, checkpoint {signature or 'start'})"
    )
    self.enqueue(job_id)
    return True

    def _progress_signature(self, job_id: str) -> str:
        """How far this job has actually got, as a comparable string.

        Counts finished stage markers, rendered clips, AND cached transcription
        chunks. Two attempts with the SAME signature did identical amounts of
        work, which is what makes a restart loop visible instead of invisible.

        The chunk count is load-bearing. It used to be absent, so a job that
        transcribed 3 chunks, was killed, transcribed 5 more, was killed again,
        still reported the identical signature -- the three extra chunks of real
        work were invisible -- and the no-progress counter advanced on a job that
        was in fact converging. Every unit of work that survives a restart is
        written to disk first (see _cleanup_job_temps and the per-chunk cache),
        so counting them is counting work that will not be redone.
        """
        try:
            stages = sum(
                1 for s in STAGES
                if (self.settings.app.paths.data_dir / "work" / job_id / s / "DONE").exists()
            )
            clips = sum(
                1 for c in self.store.list_clips(job_id) if c.get("renderState") == "done"
            )
            chunks = 0
            chunk_dir = (
                self.settings.app.paths.data_dir / "work" / job_id / "transcribe" / "chunks"
            )
            if chunk_dir.is_dir():
                chunks = sum(1 for _ in chunk_dir.glob("*.json"))
            return (
                f"{stages}/{len(STAGES)} stages, {clips} clip(s) rendered, "
                f"{chunks} chunk(s) transcribed"
            )
        except Exception:
            return ""

    def reconcile_stale_running(
        self, min_interval_sec: float = 0.0, force: bool = False
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

        RECONCILING IS NOT FAILING. A job frozen on a live stage here is, by
        construction, work the HOST took away from us — the web process is
        restarted periodically by the sandbox, which SIGKILLs the pipeline
        child. That is an external event, not a pipeline error, and the job is
        resumable from its stage markers. This used to write a generic
        "the worker stopped and never reported back" failure instead, which was
        both wrong (we know precisely what happened) and destructive (it
        burned a resume budget slot, erased the interrupt evidence, and left a
        job whose work was 99% cached looking permanently dead). It now
        RE-QUEUES the job, bounded by RESUME_BUDGET and by the requirement
        that the attempt actually made progress — see _resume_is_worthwhile.
        """
        with self._lock:
            live = set(self._active) | set(self._pending)
        # Optional floor between passes. Reconciliation reads every job
        # document and is called from the single read path the dashboard polls
        # every two seconds; proving "still stale" sixty times a minute is work
        # with no answer in it. `force=True` bypasses the floor.
        now = time.monotonic()
        if (
            min_interval_sec > 0
            and not force
            and self._last_reconcile_at is not None
            and now - self._last_reconcile_at < min_interval_sec
        ):
            return []
        self._last_reconcile_at = now
        fixed: list[str] = []
        for job in self.store.list_jobs():
            if job.get("_id") in live:
                continue
            from app import jobstatus  # local: avoids an import cycle at module load

            if jobstatus.is_terminal(job):
                continue
            job_id = job["_id"]
            stage = job.get("stage")
            # A QUEUED job not in the queue is equally stale — it was enqueued
            # by a process that died before the worker picked it up.
            silent = int((job.get("lastProgressAt") or job.get("stageStartedAt")
                          or job.get("createdAt") or 0))
            cause = self._describe_orphan(job, stage)
            # Nothing to resume TO: an upload whose file is gone can never
            # succeed, and re-queueing it is the loop, not a recovery.
            dead_end = unrecoverable_reason(self.store, job)
            if dead_end is not None:
                job.update({
                    "stage": "failed",
                    "failedStage": stage,
                    "error": dead_end,
                    "errorDetail": None,
                    "interrupted": False,
                    "interruptReason": cause,
                    "completedAt": job.get("completedAt") or silent or None,
                })
                jobstatus.describe(job)
                for field in ("stagePlan", "stageEtaSec", "totalEtaSec",
                              "queueEtaSec", "queuePosition", "progressDetail"):
                    job.pop(field, None)
                job["progress"] = 0
                self.store.write_job(job)
                self.store.append_job_log(
                    job_id,
                    "NOT RESUMABLE: the uploaded source file is no longer on the "
                    "server — failing instead of re-queueing",
                )
                logger.warning(
                    f"[worker] job {job_id} cannot be resumed (source file gone) "
                    "— cleared to failed"
                )
                fixed.append(job_id)
                continue
            if self._resume_is_worthwhile(job):
                job.update({
                    "stage": "queued",
                    "progress": 0,
                    "progressDetail": (
                        "Re-queued after the server restarted — resuming from "
                        "the last finished checkpoint."
                    ),
                    "completedAt": None,
                    "error": None,
                    "errorDetail": None,
                    "failedStage": None,
                    "interrupted": True,
                    "interruptReason": cause,
                    # Unit counters describe the attempt that died; carrying
                    # them into a queued job puts "10 of 10 clips" next to 0%.
                    "unitLabel": None,
                    "unitsDone": None,
                    "unitsTotal": None,
                })
                jobstatus.describe(job)
                for field in ("stagePlan", "stageEtaSec", "totalEtaSec",
                              "queueEtaSec", "queuePosition", "progressDetail"):
                    job.pop(field, None)
                self.store.write_job(job)
                self.store.append_job_log(
                    job_id,
                    f"RECONCILED: '{stage}' had no worker behind it ({cause}) — "
                    f"re-queued from the last checkpoint instead of failing",
                )
                logger.warning(
                    f"[worker] reconciled orphan job {job_id} at stage '{stage}' "
                    f"({cause}) — re-queued"
                )
                self.enqueue(job_id)
                fixed.append(job_id)
                continue
            # Budget spent with no progress to show for it: only now is a
            # terminal state honest. Say WHY, with the real evidence.
            job.update({
                "stage": "failed",
                "failedStage": stage,
                "error": (
                    f"The server restarted {int(job.get('interruptCount') or 0)} times "
                    f"while this job was in '{STAGE_LABELS.get(stage, stage)}' and it "
                    f"stopped making progress between restarts. Finished stages and "
                    f"rendered clips are still cached — Retry resumes from the last "
                    f"checkpoint."
                ),
                "errorDetail": cause,
                "interrupted": False,
                "interruptReason": cause,
                "completedAt": silent or None,
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
                f"RECONCILED: '{stage}' had no worker behind it ({cause}) and the "
                f"resume budget is spent — cleared to failed",
            )
            logger.warning(
                f"[worker] reconciled stale job {job_id}: stage '{stage}' with no "
                f"active worker, no queue entry, and no resume budget left ({cause})"
            )
            fixed.append(job_id)
        return fixed

    @staticmethod
    def _describe_orphan(job: dict[str, Any], stage: str | None) -> str:
        """Name the ACTUAL cause of an orphaned stage, in plain words.

        The old message said the worker "stopped and never reported back",
        which is a restatement of the symptom and told the user nothing. Every
        orphan we see in production has a cause we can name: the host restarted
        the web process, or a signal killed the child. Say which.

        DERIVED FROM COUNTS ONLY — never from the previously stored reason.
        The first version read `interruptReason` and then wrote its own output
        back into that same field, so every restart wrapped the last message
        inside the next one and the dashboard grew an unreadable sentence
        hundreds of characters deep. A derived value must never be stored back
        into its own input.
        """
        count = int(job.get("interruptCount") or 0)
        label = STAGE_LABELS.get(stage, stage or "an unknown stage")
        signal_name = job.get("lastInterruptSignal")
        if signal_name:
            text = f"the server restarted during '{label}' (worker killed by {signal_name})"
        elif count:
            text = f"the server restarted during '{label}' — {count} restart(s) so far"
        else:
            text = f"the worker process vanished during '{label}' (server restart)"
        # Belt and braces: this string is rendered in the job card, so it must
        # stay one readable sentence no matter what is on the document.
        return text[:200]

    def _resume_is_worthwhile(self, job: dict[str, Any]) -> bool:
    """Is re-queueing this orphan useful, or is it a restart loop?

    Two rules, and both exist because the naive "count the restarts"
    version lost real jobs:

    1. A generous ceiling. `INTERRUPT_MAX_RESUMES` was 5 and the sandbox
       restarts the web process often enough that a multi-hour job burned
       the whole budget without ever hitting a real error — after which
       every further restart produced a permanent failure. The budget
       exists to stop an infinite loop, not to cap a slow machine.
    2. No-progress runs out. Restarting forever only helps if each attempt
       gets further than the last. Once several consecutive attempts die
       without advancing the finished-unit count, retrying is not fixing
       anything and the job should stop and say so.

    This project checkpoints after every finished clip and re-renders only the
    unfinished ones, so a render-stage job is resumable by construction. Do not
    fail it on a fixed stalled count here unless the env explicitly re-enables
    that limit; the only real stop is the overall resume budget or a genuinely
    wedged attempt that never advances.
    """
    count = int(job.get("interruptCount") or 0)
    if count >= RESUME_BUDGET:
        return False
    stalled = int(job.get("stalledResumes") or 0)
    if stalled >= STALLED_RESUME_LIMIT and STALLED_RESUME_LIMIT > 0:
        return False
    return True

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
            # Reconcile orphaned jobs ON A TIMER, not only from request handlers.
            #
            # This is the difference between jobs that run unattended and jobs
            # that need the owner to stare at the dashboard. `reconcile_stale_
            # running` re-queues a job whose worker was killed by a host restart,
            # but until this change it was called from exactly two places --
            # GET /api/jobs and GET /api/jobs/{id} -- both of which only fire
            # when a browser is polling. A job orphaned at 03:00 with no tab
            # open therefore sat on a live stage ("Transcribing, updated 5h ago")
            # until somebody happened to load the page, which is precisely the
            # reported bug.
            #
            # The watchdog thread is the right home: it already runs for the
            # life of the process, it already owns "is this job actually making
            # progress", and it exists whether or not anyone is watching. The
            # floor keeps this off the per-second path -- reconciliation reads
            # every job document, and five times a second would spend the single
            # vCPU proving that nothing had changed.
            if self._reconcile_due():
                try:
                    self.reconcile_stale_running()
                except Exception:  # pragma: no cover - defensive
                    # A failed reconcile pass must never take the watchdog (and
                    # with it every other liveness guarantee) down with it. The
                    # next pass retries.
                    logger.exception("[worker] reconcile pass failed")

    #: Seconds between unattended reconcile passes. Well inside the window in
    #: which a stalled job is still cheap to restart, and cheap on a 1-vCPU host.
    RECONCILE_INTERVAL_SEC = 30.0

    def _reconcile_due(self) -> bool:
        now = time.monotonic()
        last = self._last_reconcile_at
        if last is None or (now - last) >= self.RECONCILE_INTERVAL_SEC:
            return True
        return False

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
