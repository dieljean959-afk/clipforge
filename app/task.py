"""Task service: resumable pipeline orchestration for the in-process worker.

Stages (each writes artifacts under data/work/{job_id}/{stage}/ and a DONE
marker before advancing):

    fetch → transcribe → detect → reframe → captions → export

On retry the pipeline resumes from the FIRST INCOMPLETE stage — a long job that
stalls at 85% transcription does not re-download or re-transcribe from zero.
Transcription additionally caches per-window results so an intra-stage resume
loses at most one decode window.

All progress writes go through the store so the dashboard shows real per-stage
progress. Every exception is surfaced with a full traceback in the job log and
the job's errorDetail — never swallowed.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app import (
    asr,
    asr_accurate,
    asr_iso,
    clipcount,
    diskwatch,
    download,
    encode,
    reframe,
    eta,
    llm,
    modelstore,
    source_stream,
    subtitle,
    translate,
    video,
)

# The reframe stage now raises PipelinePaused at a checkpoint so the outer
# handler can mark the job paused instead of failed. Import it here so the
# exception handler in run() can catch it by name.
from app.config import Settings
from app.hosting import DispatchError, dispatch_job
from app.llm import ChunkCache
from app.stages import STAGES, StageWorkspace
from app.store import LocalStore


class _PauseRequested(Exception):
    """A cooperative pause request reached a safe checkpoint.

    Raised by ``Pipeline._checkpoint`` and caught in ``run`` to mark the job
    paused. It MUST be defined at module level: this exception is raised from
    several sibling stages, and a class that only exists inside one of them is
    precisely the NameError that made the pipeline's own failure handler crash
    while trying to record the stage's real error.
    """




#: How often a long, silent step re-asserts that the pipeline is alive.
#:
#: Five seconds is well inside the 600s stall timeout and comfortably longer
#: than any real UI poll, so this is liveness only -- never progress.
HEARTBEAT_SEC = 5.0

#: How long a heartbeat can go unwritten before a RUNNING job is considered
#: stale by the startup reaper / watchdog. This is the authoritative "is this
#: child actually alive" threshold for a job whose lastProgressAt is silent.
HEARTBEAT_STALE_SEC = 15.0


STAGE_STATUS_FIELDS = frozenset({
    "status", "startedAt", "doneAt", "error", "errorDetail",
})


def _normalize_stage_status(raw: Any) -> dict[str, Any]:
    """One canonical blob per stage, used as the per-stage status default."""
    if not isinstance(raw, dict):
        return {f: None for f in STAGE_STATUS_FIELDS}
    return {
        "status": raw.get("status"),
        "startedAt": raw.get("startedAt"),
        "doneAt": raw.get("doneAt"),
        "error": raw.get("error"),
        "errorDetail": raw.get("errorDetail"),
    }


def _validate_and_normalize_stage_statuses(raw: Any) -> dict[str, dict[str, Any]]:
    """Strip anything beyond canonical fields. Keeps older/unknown keys harmless."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for stage, blob in raw.items():
        if isinstance(stage, str) and isinstance(blob, dict):
            out[stage] = _normalize_stage_status(blob)
    return out


def _chunk_cached(cache_dir: Path | None, index: int) -> bool:
    """True when a chunk's transcript has already been cached."""
    if cache_dir is None or not cache_dir.is_dir():
        return False
    return (cache_dir / f"chunk_{index}.json").exists()

#: How often a long, silent step re-asserts that the pipeline is alive.
#:
#: Five seconds is well inside the 600s stall timeout and comfortably longer
#: than any real UI poll, so this is liveness only -- never progress.
HEARTBEAT_SEC = 5.0

#: How long a heartbeat can go unwritten before a RUNNING job is considered
#: stale by the startup reaper / watchdog. This is the authoritative "is this
#: child actually alive" threshold for a job whose lastProgressAt is silent.
HEARTBEAT_STALE_SEC = 15.0


STAGE_STATUS_FIELDS = frozenset({
    "status", "startedAt", "doneAt", "error", "errorDetail",
})




class Pipeline:
    def __init__(self, settings: Settings, store: LocalStore) -> None:
        self.settings = settings
        self.store = store
        # Remaining-time prior for the CURRENT stage (e.g. duration x RTF while
        # transcribing). Reset by _progress whenever the stage changes.
        self._prior_eta_sec: float | None = None
        # Heartbeat writes that failed. Surfaced by the stall check so a job
        # whose liveness signal could not be recorded is diagnosed as
        # "the worker is alive but cannot report", not "the worker died".
        self._heartbeat_faults = 0
        # Working-stage cache for the current run. Written to the job doc on
        # every transition so the dashboard can show per-stage state without
        # reading the workspace markers.
        self._stage_status: dict[str, dict[str, Any]] = {}
        # Liveness for THIS run. Written by the heartbeat on every beat and by
        # the startup reaper to decide whether a running-looking job still has a
        # live child. Persisted by _set_stage_status / _fail so a recovered job
        # carries its heartbeat history.
        self._heartbeat_at: int | None = None
        # Disk ledger for the CURRENT run. Set in _stages, read by stages that
        # want a stage-scoped measurement, always cleared in the finally.
        self._ledger: diskwatch.JobLedger | None = None

    # ------------------------------------------------------------------
    # disk bookkeeping
    # ------------------------------------------------------------------

    def _scratch_roots(self, job_id: str) -> list[Path]:
        """The directories that belong to ONE job (never .venv/node_modules)."""
        paths = self.settings.app.paths
        roots = [
            paths.downloads_dir / job_id,
            paths.data_dir / "work" / job_id,
            paths.outputs_dir / job_id,
        ]
        roots.extend(paths.job_source_dirs(job_id))
        seen: list[Path] = []
        for r in roots:
            if r not in seen:
                seen.append(r)
        return seen

    def _cleanup_job_scratch(self, job_id: str, *, full: bool) -> None:
        """Delete this job's local scratch. `full` also drops resume state.

        Partial (failed/paused): the source copy and every decoded audio file
        go; the small JSON markers stay so a retry resumes instead of paying
        for the download and the transcript again.
        """
        ws = StageWorkspace(self.settings.app.paths.data_dir, job_id)
        try:
            self._cleanup_job_temps(job_id, ws)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[job {job_id}] temp cleanup failed: {exc}")
        # The source copy, the per-clip renders and any section temp: with the
        # bucket as the library these are reproducible, not precious.
        for root in self._scratch_roots(job_id):
            try:
                if not root.exists():
                    continue
                for p in sorted(root.rglob("*")):
                    if not p.is_file():
                        continue
                    # Media is reproducible from the bucket: the source is an
                    # object, and a clip only exists until its HEAD check. The
                    # JSON markers are what resume actually reads.
                    if p.suffix.lower() in {".mp4", ".m4a", ".webm", ".part", ".wav", ".png"}:
                        p.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[job {job_id}] could not clean {root}: {exc}")
        if full:
            for root in self._scratch_roots(job_id):
                shutil.rmtree(root, ignore_errors=True)
        freed = 0
        try:
            freed = sum(diskwatch.tree_bytes(r) for r in self._scratch_roots(job_id))
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            f"[job {job_id}] scratch cleanup ({'full' if full else 'partial'}): "
            f"{diskwatch.mb(freed)} MB still held"
        )

    def _persist_ledger(
        self, job_id: str, ledger: diskwatch.JobLedger,
    ) -> None:
        """Put the per-stage table and the peak on the job doc (job details)."""
        try:
            job = self.store.get_job(job_id)
            if job is None:
                return
            job["disk"] = {
                **ledger.report(),
                "scratchCapMb": diskwatch.mb(ledger.cap_bytes),
            }
            self.store.write_job(job)
        except Exception as exc:  # noqa: BLE001 - stats never fail a job
            logger.warning(f"[job {job_id}] could not persist disk ledger: {exc}")

    # ------------------------------------------------------------------
    # resume plumbing: pick up a job from the persisted stageStatus, including
    # the very first pipeline heartbeat so a recovered run is visible to the
    # reaper as soon as it starts working.
    # ------------------------------------------------------------------

    def _resume_from_store(self, job_id: str, job: dict[str, Any]) -> None:
        """Rehydrate the run's working copies from the persisted job doc.

        ``run()`` already loaded the job, but it never repopulated
        ``_stage_status`` or ``_heartbeat_at`` from the disk copy. That is fine
        for a fresh job, and fatal for a resumed one: ``_stages`` uses
        ``_stage_status_for`` / ``_first_incomplete_stage``, and the exit-path
        heartbeat flush reads ``self._heartbeat_at`` -- both of which kept the
        module defaults ({}) / None and then crashed the moment they were used
        as if they were real data.
        """
        if isinstance(job.get("stageStatus"), dict):
            self._stage_status = _validate_and_normalize_stage_statuses(job["stageStatus"])
        if isinstance(job.get("heartbeatAt"), (int, float)):
            self._heartbeat_at = int(job["heartbeatAt"])

    # ------------------------------------------------------------------
    # long-input guard queries (descriptive only; no caps)
    # ------------------------------------------------------------------

    def _long_source_note(self, duration_sec: float) -> None:
        """Log once, at pipeline start, if the source is unusually long.

        No cap, no shortcut, no change to the plan or the job: purely a
        descriptive note so a reviewer can distinguish "took a while" from
        "something hung". The actual pacing comes from the stage progress
        callbacks and the ETA machinery, not from this log.
        """
        try:
            if duration_sec > self.settings.hosted.max_source_min * 60 * 2:
                logger.info(
                    f"[long-source] {duration_sec / 60:.0f} min "
                    f"(> {self.settings.hosted.max_source_min * 2:.0f} min) — "
                    f"transcription will chunk and render is per-clip; memory is the "
                    f"likely limit, not time"
                )
        except Exception:
            pass

    def _transcript_size_note(self, n_words: int) -> None:
        try:
            if n_words > 5000:
                logger.info(
                    f"[long-transcript] {n_words} words — "
                    f"the transcript dict is now the largest single Python object; "
                    f"do not keep extra copies of it in memory"
                )
        except Exception:
            pass

    def _render_workload_note(self, n_clips: int, duration_sec: float) -> None:
        try:
            if n_clips >= 10 and duration_sec > 60 * 15:
                logger.info(
                    f"[long-render] {n_clips} clips from a "
                    f"{duration_sec / 60:.0f} min source — render is the "
                    f"memory-critical stage; each clip is its own ffmpeg process"
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # disk-space guard (the method whose absence crashed two jobs)
    # ------------------------------------------------------------------

    def _check_disk_space(self, job_id: str, source_duration_sec: float) -> None:
        """Fail fast when the host is out of room before we write more artifacts.

        Called once at the top of _stages and once inside _stage_render (after
        probe) so a host that fills up mid-render fails at the cheapest point.
        This is a guard, not a quota system: it does not try to predict whether
        the WHOLE batch will fit, only whether the host is already in trouble.
        """
        from app import hostlimits
        from app.hosting import mb_from_bytes, wait_for_space

        path = self.settings.app.paths.data_dir
        lim = hostlimits.describe()
        try:
            # hostlimits has no dir_bytes/dir_free helpers; shutil answers the
            # same question directly (total/free on the filesystem holding the
            # data dir — the conservative free figure, not an invented number).
            usage = shutil.disk_usage(path)
            free = usage.free
        except Exception as exc:
            logger.warning(f"[job {job_id}] could not measure disk: {exc}")
            return
        if free is None:
            return
        # Waiting instead of failing: a render that needs ~400 MB on a 4 GB box
        # is a scheduling problem, not a reason to throw away a finished
        # transcript. The wait runs the reclaim on every tick, so space that a
        # finished batch still holds is released first, and only then does the
        # job pause with "Waiting for disk space".
        from app import reclaim as _reclaim

        def _on_wait(reason: str, free_b: int, scratch_b: int) -> None:
            self._progress(
                job_id, "waiting", 0,
                f"Waiting for disk space — {reason}",
            )
            self._log(job_id, f"waiting for disk space: {reason}")
            try:
                _reclaim.reclaim(self.store, self.settings.app.paths)
            except Exception:  # noqa: BLE001 - reclaim is best effort
                pass

        need = 400 * 1024 * 1024 if source_duration_sec > 0 else 200 * 1024 * 1024
        ok, reason = wait_for_space(
            self.settings.app.paths,
            need,
            cap_bytes=int(getattr(self.settings.hosted, "scratch_cap_mb", 1536)) * 1024 * 1024,
            timeout_sec=1800.0,
            on_wait=_on_wait,
        )
        if not ok:
            raise video.RenderError(
                f"not enough free disk ({lim}) — {reason} on {path}. The "
                f"download and transcript are still cached; Retry will resume "
                f"from the last completed stage once space is cleared."
            )
        if source_duration_sec > 0:
            est = source_duration_sec * 0.05
            if est and free < est + 500 * 1024 * 1024:
                logger.warning(
                    f"[job {job_id}] low disk headroom: {mb_from_bytes(free)} MB "
                    f"free, ~{est:.0f} MB estimated render footprint for a "
                    f"{source_duration_sec / 60:.0f} min source"
                )

    # ------------------------------------------------------------------
    # structured failure path (the method whose absence masked the real crash)
    # ------------------------------------------------------------------

    def _fail(self, job_id: str, message: str, detail: str) -> None:
        """Record a terminal failure and stop the job.

        This is the ONE place a structured exception path writes the terminal
        state, so the exception handler in run() does not itself crash while
        trying to report an earlier crash (the bug behind both failed jobs:
        the handler called self._fail which did not exist).
        """
        job = self.store.get_job(job_id)
        if job is None:
            logger.error(f"[job {job_id}] cannot fail — job gone from store")
            return
        stage = job.get("stage") or "unknown"
        self._stage_status.setdefault(stage, _normalize_stage_status({}))
        self._stage_status[stage]["status"] = "failed"
        self._stage_status[stage]["error"] = message
        self._stage_status[stage]["errorDetail"] = detail
        now_ms = int(time.time() * 1000)
        job.update({
            "stage": "failed",
            "status": "failed",
            "failedStage": stage,
            "isTerminal": True,
            "finishedAt": now_ms,
            "error": message,
            "errorDetail": detail,
            # The worker re-queues only this kind, after cleaning up orphans.
            "errorKind": "memory" if asr_iso.is_memory_error(message) else None,
            "completedAt": now_ms,
            "stageStatus": self._stage_status,
            "interrupted": False,
            "progress": 0.0,
            "unitLabel": None,
            "unitsDone": None,
            "unitsTotal": None,
            "stageEtaSec": None,
            "totalEtaSec": None,
            "stagePlan": None,
            "lastProgressAt": now_ms,
        })
        self.store.write_job(job)
        self.store.append_job_log(job_id, f"FAILED at {stage}: {message}")
        logger.error(f"[job {job_id}] FAILED at {stage}: {message}")

    # ------------------------------------------------------------------
    # temp cleanup (the method whose absence was never reached, but must exist)
    # ------------------------------------------------------------------

    def _cleanup_job_temps(
        self, job_id: str, ws: StageWorkspace,
    ) -> None:
        """Drop scratch files that are no longer needed once the batch finished.

        Called after _finish so a completed job does not keep its extracted
        audio WAV and per-chunk transcript scratch around for the lifetime of
        the job. Finished stage artifacts (words.json, candidates.json, cuts,
        per-clip windows) are NOT touched — those are what resume reuses.
        """
        try:
            audio = ws.stage_dir("reframe") / "audio.wav"
            if audio.exists():
                audio.unlink(missing_ok=True)
                logger.info(f"[job {job_id}] removed scratch audio {audio}")
        except Exception as exc:
            logger.warning(f"[job {job_id}] could not remove scratch audio: {exc}")
        try:
            chunk_dir = ws.stage_dir("transcribe") / "chunks"
            if chunk_dir.is_dir():
                for p in chunk_dir.glob("*.tmp"):
                    p.unlink(missing_ok=True)
                # The 16 kHz mono copy of the whole source is the biggest file
                # the ASR stage leaves behind — 41 MB for a 22-minute video,
                # left on disk for the life of the batch. `words.json` is the
                # transcript that resume reads, so the audio itself is scratch.
                for p in chunk_dir.glob("*.wav"):
                    freed = p.stat().st_size if p.exists() else 0
                    p.unlink(missing_ok=True)
                    logger.info(
                        f"[job {job_id}] removed ASR audio {p.name} "
                        f"({freed / 1048576:.0f} MB)"
                    )
        except Exception as exc:
            logger.warning(f"[job {job_id}] could not clean transcript tmp files: {exc}")

    # ------------------------------------------------------------------
    # per-stage status helpers
    # ------------------------------------------------------------------

    def _stage_status_for(self, job_id: str, stage: str) -> dict[str, Any]:
        """Current per-stage status blob, loaded from the job doc if needed.

        Provided for call sites that still want a one-shot lookup without
        relying on the run-level ``_stage_status`` working copy. After the
        first call in a run the working copy is already populated by
        ``_first_incomplete_stage`` / ``run``, so this is cheap.
        """
        if self._stage_status:
            return self._stage_status.get(stage, _normalize_stage_status({}))
        return _normalize_stage_status({})

    def _set_stage_status(
        self, job_id: str, stage: str, *, status: str,
        startedAt: int | None = None, doneAt: int | None = None,
        error: str | None = None, errorDetail: str | None = None,
    ) -> None:
        """Persist a per-stage transition atomically and update the working copy."""
        now = int(time.time() * 1000)
        blob = self._stage_status_for(job_id, stage)
        blob["status"] = status
        if startedAt is not None:
            blob["startedAt"] = startedAt
        if doneAt is not None:
            blob["doneAt"] = doneAt
        if error is not None:
            blob["error"] = error
        if errorDetail is not None:
            blob["errorDetail"] = errorDetail
        def apply(job: dict[str, Any]) -> None:
            job["stageStatus"] = dict(self._stage_status)
            job["lastProgressAt"] = now
            # A stage transition is real progress for that stage: reset its own
            # clock so the per-stage silence budget starts from here.
            job["stageProgressAt"] = now

        # Same locked read-modify-write as _progress: a whole-document write
        # here could undo a newer stage/progress write that landed meanwhile.
        self.store.update_job(job_id, apply)

    def _mark_stage_running(self, job_id: str, stage: str) -> None:
        prev = self._stage_status_for(job_id, stage)
        if prev.get("status") in {"done", "failed"}:
            return
        self._set_stage_status(
            job_id, stage, status="running",
            startedAt=prev.get("startedAt") or int(time.time() * 1000),
        )

    def _mark_stage_done(self, job_id: str, stage: str) -> None:
        self._set_stage_status(
            job_id, stage, status="done",
            doneAt=int(time.time() * 1000),
            error=None, errorDetail=None,
        )

    def _mark_stage_failed(self, job_id: str, stage: str, error: str, detail: str) -> None:
        self._set_stage_status(
            job_id, stage, status="failed",
            doneAt=int(time.time() * 1000),
            error=error, errorDetail=detail,
        )

    def _first_incomplete_stage(self, job_id: str) -> str | None:
        """First stage that is not 'done' according to the persisted stageStatus.

        This is the authoritative resume point. It defaults to the workspace
        markers when stageStatus has not been written yet (older jobs), so
        existing cached jobs still resume correctly.
        """
        job = self.store.get_job(job_id)
        if job and isinstance(job.get("stageStatus"), dict):
            self._stage_status = _validate_and_normalize_stage_statuses(job["stageStatus"])
        elif not self._stage_status:
            self._stage_status = _validate_and_normalize_stage_statuses(None)
        for s in STAGES:
            if self._stage_status.get(s, {}).get("status") != "done":
                return s
        return None

    # ------------------------------------------------------------------
    # progress / logging helpers
    # ------------------------------------------------------------------

    def _progress(
        self,
        job_id: str,
        stage: str,
        pct: float,
        detail: str,
        *,
        unit_label: str | None = None,
        units_done: float | None = None,
        units_total: float | None = None,
        prior_eta_sec: float | None = None,
    ) -> None:
        """Write progress AND a live remaining-time estimate.

        The ETA comes from measured progress once the stage has visibly moved
        (elapsed / done * remaining) and from an RTF prior before that, so a
        long transcription shows a number from the first percent instead of an
        indefinite spinner. Every estimate is labelled with how it was derived
        (``etaSource``) so the UI never presents a guess as a measurement.
        """
        now_ms = int(time.time() * 1000)
        fraction = max(0.0, min(1.0, pct))

        # Read-modify-write INSIDE the job lock. This used to read the doc,
        # think for a while and then overwrite it, so a heartbeat thread (or
        # the worker's log pump) that had read an older copy wrote that copy
        # back afterwards — observed as a 25-minute job sitting on
        # "Transcribing chunk 27 of 27, 96%" for the five minutes detection was
        # actually running: the transcribe heartbeat's snapshot landed after
        # detection's first "finding" write, and detection then reported
        # nothing until its last chunk. `store.update_job` closes that window.
        def apply(job: dict[str, Any]) -> None:
            self._apply_progress(
                job_id, job, stage, pct, detail, now_ms, fraction,
                unit_label=unit_label, units_done=units_done, units_total=units_total,
                prior_eta_sec=prior_eta_sec,
            )

        self.store.update_job(job_id, apply)

    def _apply_progress(
        self,
        job_id: str,
        job: dict[str, Any],
        stage: str,
        pct: float,
        detail: str,
        now_ms: int,
        fraction: float,
        *,
        unit_label: str | None = None,
        units_done: float | None = None,
        units_total: float | None = None,
        prior_eta_sec: float | None = None,
    ) -> None:
        """Mutate `job` with this tick's progress/ETA (caller holds the lock)."""
        # `status` was written once at creation (`queued`) and never updated, so
        # a job that was transcribing still said "status: queued" in its own
        # document. Every view that trusted it (or derived a stage from it) then
        # disagreed with the stage chips. Progress IS the proof of life: if the
        # pipeline is writing ticks, the job is running.
        job["status"] = "running"
        prev_stage = job.get("stage")
        same_stage = prev_stage == stage
        if prev_stage != stage:
            self._prior_eta_sec = None
            timings = list(job.get("stageTimings") or [])
            prev_start = job.get("stageStartedAt")
            # Only record a hand-off when it crosses a PLAN stage. The raw names
            # differ for steps the user thinks of as one ("fetch" -> "grabbing"
            # is both "Downloading"), and splitting on those wrote the same
            # stage twice with a zero-length row in between.
            if prev_stage and prev_start and eta.plan_stage_for(prev_stage) != eta.plan_stage_for(stage):
                timings.append({
                    "stage": prev_stage,
                    "sec": round(max(0.0, (now_ms - prev_start) / 1000), 1),
                    "at": prev_start,
                })
            job["stageTimings"] = [
                {"stage": eta.plan_stage_for(t.get("stage")), "sec": t.get("sec"), "at": t.get("at")}
                for t in timings[-12:]
            ]
            # A stage being RE-ENTERED (a resume, a host restart, a paused job)
            # must not restart its clock. Resetting `stageStartedAt` to now made
            # a resumed render report "0.2 min left" with nine clips still to
            # encode, because the elapsed it was dividing was only the seconds
            # since the restart. Wind the clock back over what the stage has
            # already spent so "measured" stays measured across the seam.
            spent = sum(
                float(t.get("sec") or 0.0)
                for t in job["stageTimings"]
                if t.get("stage") == eta.plan_stage_for(stage)
            )
            job["stageStartedAt"] = now_ms - int(spent * 1000) if spent > 0 else now_ms
            # Moving into a new stage means the job is alive again, so any
            # error left by the attempt that died stops being true. Without
            # this, a job that failed detection with a 429 and later resumed
            # spent its whole render advertising "AI detection failed" beside a
            # live progress bar.
            if job.get("error") is not None:
                logger.info(f"[job {job_id}] clearing stale error on stage change -> {stage}")
            job["error"] = None
            job["errorDetail"] = None
            # This job is RUNNING now, so it is not waiting behind anything. A
            # re-queue that raced a live run left `queuePosition` set on a job
            # that was busy, and the card renders that as a queue message beside
            # a moving progress bar.
            job["queuePosition"] = None
        if prior_eta_sec is not None:
            self._prior_eta_sec = float(prior_eta_sec)

        elapsed = max(0.0, (now_ms - (job.get("stageStartedAt") or now_ms)) / 1000)
        stage_eta, eta_source = eta.estimate(
            elapsed_sec=elapsed,
            fraction=fraction,
            prior_remaining_sec=self._prior_eta_sec,
        )
        # A countable stage (clips rendered, chunks done) extrapolates far
        # better on cost-per-unit than on a percentage: the first clip pays for
        # probing and filter-graph setup, and clips are not all the same length,
        # so a straight line through "clip 3 of 23" is wrong at both ends.
        if stage_eta is None and units_done and units_total:
            per_unit = eta.eta_from_units(elapsed, units_done, units_total)
            if per_unit is not None:
                stage_eta, eta_source = per_unit, "measured"

        # Per-stage plan: what is left, stage by stage, with each number
        # labelled by where it came from. The current stage's estimate is the
        # live one, completed stages use their real duration, and the rest use
        # priors scaled by the source length.
        plan = eta.build_plan(
            current_stage=stage,
            duration_sec=float(job.get("durationSec") or 0.0),
            clip_count=int(job.get("clipTarget") or (job.get("options") or {}).get("maxClips")
                           or self.settings.gemini.max_clips),
            clip_seconds=float(job.get("expectedClipSec") or self.settings.gemini.max_clip_sec),
            rtf=self._current_rtf(),
            stage_eta_sec=stage_eta,
            measured={t.get("stage"): t.get("sec") for t in (job.get("stageTimings") or [])},
        )
        total_eta = eta.plan_total_sec(plan) if plan else None

        # Monotonic within a stage: a resume heartbeat can report a lower
        # percentage than the work already done (the clock restarted before the
        # completed clips were counted), and the bar visibly ran backwards.
        floor = float(job.get("progressMax") or 0.0) if same_stage else 0.0
        shown = max(floor, max(0.0, min(100.0, pct * 100)))
        job.update({
            "stage": stage,
            "progress": shown,
            "progressMax": shown,
            "progressDetail": detail,
            "unitLabel": unit_label,
            "unitsDone": units_done,
            "unitsTotal": units_total,
            "stageEtaSec": None if stage_eta is None else round(stage_eta, 1),
            "totalEtaSec": total_eta,
            "etaSource": eta_source,
            "stagePlan": plan,
            "lastProgressAt": now_ms,
            # Progress-only clock. `lastProgressAt` is also refreshed by the
            # liveness heartbeat, so it cannot answer "is THIS stage moving?";
            # the per-stage watchdog budget needs a timestamp nothing but real
            # progress touches.
            "stageProgressAt": now_ms,
        })

    @contextmanager
    def _heartbeat(self, job_id: str, what: str):
        """Prove liveness across a long step that reports nothing.

        A single ffmpeg call can run for minutes, and ffmpeg's own progress is
        parsed only for error text -- nothing calls `_progress` while it works.
        So `lastProgressAt` goes stale during exactly the step most likely to be
        legitimately slow, and two things then misfire: the UI shows "no recent
        activity", and the watchdog (which kills a job after STALL_TIMEOUT_SEC
        of silence) terminates a render that was working fine.

        This bumps `lastProgressAt` every HEARTBEAT_SEC and nothing else. It
        deliberately does NOT touch `progress` or the ETA: inventing movement
        that did not happen is worse than admitting the step is opaque. A
        killed-but-healthy process cannot survive it either -- it stops the
        moment the block exits, including on the exception path.
        """
        stop = threading.Event()

        def beat() -> None:
            while not stop.wait(HEARTBEAT_SEC):
                try:
                    job = self.store.get_job(job_id)
                    if job is None or job.get("stage") in {"complete", "failed"}:
                        return
                    self._patch_job(job_id, lastProgressAt=int(time.time() * 1000))
                    self._heartbeat_at = int(time.time() * 1000)
                except Exception as exc:  # noqa: BLE001 - never fail a job
                    # A heartbeat that cannot write must not raise into the
                    # render; the watchdog's own timeout remains the backstop.
                    # WARNING, not debug: this is the exact failure behind the
                    # dashboard's "updated 4m ago" while the worker was alive
                    # and busy. A silently-dropping heartbeat makes a healthy
                    # job look stalled to both the user and the watchdog, so it
                    # must be visible in the log without stopping the render.
                    logger.warning(f"[heartbeat] {what}: could not write: {exc}")
                    self._heartbeat_faults += 1

        thread = threading.Thread(
            target=beat, name=f"heartbeat-{job_id[:8]}", daemon=True,
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()

    def _current_rtf(self) -> float:
        """This host's transcription speed, measured where possible."""
        try:
            engine = asr.status(self.settings)
            store = eta.store(self.settings.app.paths.data_dir)
            value, _ = store.effective_rtf(
                engine.engine, engine.device, engine.model, engine.compute_type
            )
            return float(value)
        except Exception as exc:  # a missing prior must not break progress
            logger.debug(f"[eta] falling back to the default RTF: {exc}")
            return 0.3

    def _log(self, job_id: str, line: str) -> None:
        logger.info(f"[job {job_id}] {line}")
        self.store.append_job_log(job_id, line)

    def _caption_settings(self, job: dict[str, Any]) -> tuple[Settings, dict[str, Any]]:
        """(Settings for the renderer, THE resolved caption object).

        One call decides every caption number this batch will burn: the stored
        preset + the dashboard's slider overrides + this job's own preset and
        emoji choice. Everything downstream reads the returned Settings, so a
        value cannot be applied in one place and ignored in another.

        The per-job fields are folded in as overrides rather than replacing
        the stored ones, so a batch created from the dashboard keeps the
        sliders and picks its own look on top of them.
        """
        from app import caption_style

        overrides = dict(caption_style.read(self.settings.app.paths))
        preset = str(job.get("captionPreset") or "").strip()
        if preset:
            overrides["preset_key"] = preset
        if job.get("emojiEnabled") is not None:
            overrides["emoji_enabled"] = bool(job.get("emojiEnabled"))
        if job.get("emojiDensity"):
            overrides["emoji_density"] = job.get("emojiDensity")

        resolved = caption_style.resolve_caption_settings(self.settings, overrides)
        sub = self.settings.subtitle
        applied = replace(
            sub,
            font_size=int(resolved["fontSize"]),
            words_per_cue=int(resolved["wordsPerCue"]),
            words_per_cue_max=int(resolved["wordsPerCueMax"]),
            pause_sec=float(resolved["pauseSec"]),
            size_scale=float(resolved["sizeScale"]),
            highlight_color=str(resolved["highlightColor"] or ""),
            position_y=float(resolved["positionY"]),
            emoji_enabled=bool(resolved["emojiEnabled"]),
            emoji_density=str(resolved["emojiDensity"]),
            preset=str(resolved["presetKey"]),
        )
        return replace(self.settings, subtitle=applied), resolved

    def _patch_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        """Re-read the job, apply `fields`, write it back.

        Never write the dict a stage was HANDED. `run()` reads the job once and
        passes that same object down through every stage, while `_progress`
        writes a freshly-read copy to the store on every tick. Writing the
        handed-down dict back therefore resurrects whatever `stage`, `progress`
        and ETA the job had when the pipeline started — which is how a job
        detecting 20 chunks in a row reported itself "queued" at 0%: this
        method's whole reason for existing.
        """
        return self.store.update_job(job_id, lambda job: job.update(fields)) or {}

    # ------------------------------------------------------------------ #
    # cooperative pause
    # ------------------------------------------------------------------ #

    def pause_requested(self, job_id: str) -> bool:
        """Has the user asked for this job to stop?

        Read from the job document rather than a signal so it works across
        processes: the API process sets the flag, the per-job child process
        (which owns all the real work) sees it at its next safe boundary.
        """
        job = self.store.get_job(job_id)
        return bool(job and job.get("pauseRequested"))

    def _checkpoint(self, job_id: str, where: str) -> None:
        """Stop here if a pause is pending. Called ONLY between units of work.

        The point of a cooperative pause is that it never interrupts a write.
        Every call site is a boundary: between transcript chunks, between
        detection chunks, between rendered clips, between stage transitions.
        """
        if self.pause_requested(job_id):
            raise _PauseRequested(where)

    def mark_paused(self, job_id: str, where: str) -> None:
        """Record a clean pause for the dashboard."""
        job = self.store.get_job(job_id)
        if job is None:
            return
        job.update({
            "stage": "paused",
            "pausedStage": where,
            "pauseRequested": False,
            "pausedAt": int(time.time() * 1000),
            "interrupted": False,
            "interruptReason": None,
            "progressDetail": f"Paused by user at {where} — resume continues from here.",
            "completedAt": None,
        })
        self.store.write_job(job)
        self._log(job_id, f"PAUSED by user at {where} — nothing in flight; resume from here")

    def _write_caption_tracks(
        self, job_id: str, clip_index: int, cues: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Write per-language SRT/VTT sidecars for one clip.

        Deliberately never raises: the render has already succeeded by the time
        this runs, so a translation failure must degrade to a missing track and
        a visible log line, not a failed job. Each language is independent, so
        one bad batch cannot take the other eleven down with it.

        The output file doubles as a resume cache — a re-render (Resume/Retry)
        reuses an existing non-empty track instead of paying for the same LLM
        calls twice.
        """
        return translate.write_tracks(
            self.settings.app.paths.outputs_dir / job_id,
            clip_index, cues, self.settings,
            log=lambda line: self._log(job_id, line),
        )

    @staticmethod
    def _srt_time(sec: float) -> str:
        """Proper SRT timecode: 00:00:01,250 (cues are clip-relative)."""
        ms = max(0, round(sec * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def run(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            logger.error(f"[job {job_id}] missing at pipeline start")
            return
        # Rehydrate the run's working copies from the persisted job doc before
        # we open the pipeline heartbeat or touch any stage machinery. A fresh
        # job leaves both at the safe module defaults; a resumed job must carry
        # its real stageStatus and heartbeatAt or the run will crash on the
        # first resume query (empty dict used as a dict) or on the first
        # heartbeat-free flush at the end of the run.
        self._resume_from_store(job_id, job)
        t_start = time.time()
        # A killed attempt (host restart, SIGKILL) skips its `finally`, so the
        # model weights it restored can still be sitting in data/models when the
        # next attempt starts — that is the ~600 MB the disk lost between two
        # readings of the same job. Clear them before this run touches the disk:
        # the bucket copy is the source of truth, so the next stage restores
        # what it needs and nothing is downloaded again.
        try:
            stale = modelstore.release_all()
            if stale.get("freed"):
                self._log(
                    job_id,
                    f"[disk] cleared {stale['freed'] / 1048576:.0f} MB of model weights "
                    "left behind by an interrupted attempt",
                )
        except Exception as exc:  # noqa: BLE001 - never block a run over this
            logger.warning(f"[job {job_id}] could not clear stale models: {exc}")
        # The heartbeat spans the WHOLE pipeline, not just the encode and the
        # decode. The stall deadline is now 3 minutes, and every stage has a
        # step that legitimately exceeds it while reporting nothing: a Whisper
        # chunk, an LLM call parked in the provider's rate-limit queue, a
        # reframe analysis. Without a process-wide heartbeat those look
        # identical to a hang and the watchdog kills work that is working. It
        # still proves liveness and nothing more -- `progress` is never
        # invented -- and it stops the instant the block exits.
        try:
            with self._heartbeat(job_id, f"pipeline ({job.get('stage') or 'starting'})"):
                self._stages(job_id, job, t_start)
        except subtitle.TranscriptionPaused as pause:
            self.mark_paused(job_id, "transcribing")
            self._log(job_id, f"transcription paused: {pause}")
        except _PauseRequested as pause:
            # A user-requested stop, not an error. Whatever completed is on
            # disk; the job reports itself paused and resume picks it up from
            # the same stage via the ordinary stage-marker resume.
            self.mark_paused(job_id, str(pause) or "the current stage")

        except llm.DetectionPaused as pause:
            self.mark_paused(job_id, "finding")
            self._log(job_id, f"detection paused: {pause}")
        except DispatchError as exc:
            # remote worker refused/unreachable — honest error, job stays failed
            self._fail(job_id, f"Could not hand the job to your worker: {exc}", self._detail(exc))
        except llm.LlmNotConfigured as exc:
            self._fail(job_id, f"AI detection offline — {exc}", self._detail(exc))
        except llm.LlmError as exc:
            self._fail(job_id, f"AI detection failed: {exc}", self._detail(exc))
        except download.DownloadError as exc:
            self._fail(job_id, f"Could not fetch video: {exc}", self._detail(exc))
        except asr.AsrError as exc:
            self._fail(job_id, f"Transcription failed: {exc}", self._detail(exc))
        except video.RenderError as exc:
            self._fail(job_id, f"Render failed: {exc}", self._detail(exc))
        except Exception as exc:  # noqa: BLE001 - the UI must see the real error
            # Log the ORIGINAL traceback FIRST, before any cleanup. If the cleanup
            # below also raises (store I/O, disk full, lock contention), the second
            # exception must never replace the real one the UI shows.
            logger.exception(f"[job {job_id}] unexpected pipeline error")
            try:
                self._fail(job_id, f"Pipeline error: {exc}", self._detail(exc))
            except Exception as fail_exc:
                # _fail itself must not turn a render crash into a store crash in
                # the UI. Persist what we can and still record the original message.
                logger.error(
                    f"[job {job_id}] _fail itself raised while reporting: {fail_exc}"
                )
        finally:
            # Make the liveness heartbeat visible to the reaper/web. A child that
            # reaches here — including on a hard crash — has had its last beat at
            # the moment the heartbeat context manager stopped, so persist that
            # timestamp once more with the (possibly terminal) stageStatus we have
            # now. The reaper uses heartbeatAt + lastProgressAt together to decide
            # whether a running-looking job still has a live child.
            #
            # This cleanup must NEVER raise into the job's real error path.
            try:
                job = self.store.get_job(job_id)
                if job is not None:
                    job["heartbeatAt"] = self._heartbeat_at
                    self.store.write_job(job)
            except Exception:
                logger.debug(f"[job {job_id}] could not persist heartbeat on exit")


    def _stages(self, job_id: str, job: dict[str, Any], t_start: float) -> None:
        """The stage sequence, isolated from run()'s failure handling.

        Split out only so the heartbeat context manager can span every stage at
        once — it must stay open from the first fetch to the last export.
        """
        # ---- remote dispatch mode: hand off, then the agent reports via
        # /api/internal/* — this thread's work is done after handoff.

        if self.settings.hosted.worker_url:
            # Remote dispatch still needs the source fetched locally first;
            # the source pointer is reused by the worker-agent handoff.
            ws = StageWorkspace(self.settings.app.paths.data_dir, job_id)
            src = self._stage_fetch(job, ws)
            ack = dispatch_job(self.settings, job, src)
            job = self.store.get_job(job_id) or job
            job.update({
                "stage": "rendering",
                "progress": 10,
                "progressDetail": "Your worker is rendering — this page updates live.",
            })
            self.store.write_job(job)
            self._log(
                job_id,
                f"dispatched to worker {self.settings.hosted.worker_url} "
                f"(agent {ack.get('version', '?')}) — heavy rendering runs on your machine",
            )
            return

        ws = StageWorkspace(self.settings.app.paths.data_dir, job_id)
        self._check_disk_space(job_id, 0.0)

        # The ledger answers "what does this batch cost on disk?" with numbers,
        # samples free space every 5 s, and is the thing that stops a runaway at
        # the 300 MB scratch cap instead of letting `df` reach zero again.
        ledger = diskwatch.JobLedger(
            job_id,
            self._scratch_roots(job_id),
            data_dir=self.settings.app.paths.data_dir,
            on_log=lambda line: self._log(job_id, line),
            cap_bytes=int(getattr(self.settings.hosted, "scratch_cap_bytes", 0) or 0)
            or diskwatch.SCRATCH_CAP_BYTES,
        )
        self._ledger = ledger
        ledger.start()

        # Where does this run actually start? Prefer the persisted
        # stageStatus (the dashboard's authoritative per-stage picture) so a
        # resumed job does not redo finished work; fall back to workspace
        # markers for jobs that predate stageStatus.
        resume_from = self._first_incomplete_stage(job_id) or ws.first_incomplete()
        done_before = [s for s in STAGES if self._stage_status_for(job_id, s).get("status") == "done"]
        if not done_before:
            done_before = [s for s in STAGES if ws.is_done(s)]
        if done_before:
            self._log(
                job_id,
                f"resume: stages already complete = {done_before or 'none'}; "
                f"starting at '{resume_from or 'export'}'",
            )

        # If a stage is done in the workspace but not yet reflected in
        # stageStatus (older jobs), sync the two views so we never re-run it.
        for s in STAGES:
            blob = self._stage_status_for(job_id, s)
            if blob.get("status") != "done" and ws.is_done(s):
                self._mark_stage_done(job_id, s)

        try:
            src = self._stage_fetch(job, ws)
            self._mark_stage_done(job_id, "fetch")
        except BaseException:
            self._mark_stage_failed(job_id, "fetch", *self._fail_args_for(sys.exc_info()))
            raise

        # Every remaining stage gets the same treatment: its OWN name and
        # error land in the per-stage status (and the run's job log) the
        # moment it raises, before run()'s generic handler fires. Previously
        # only fetch was wrapped, so detect/render failures surfaced as
        # generic messages with no stage attached while the chips stayed grey.
        # BaseException on purpose: a KeyboardInterrupt or SystemExit during a
        # stage must still mark the stage failed rather than leave it "running"
        # forever, and `raise` rethrows for run() to classify.

        try:
            self._checkpoint(job_id, "fetch")
        except BaseException:
            self._mark_stage_failed(job_id, "fetch", *self._fail_args_for(sys.exc_info()))
            raise

        def _run_stage(stage: str, fn: Callable[[], Any]) -> Any:
            """Run one stage, marking it running/done/failed on the job."""
            # Refresh the in-memory job from the store before every stage.
            # `_stage_fetch` re-reads the job to write durationSec, but that was
            # a LOCAL rebind — the dict the later stages received still had
            # durationSec=None, so a source with no speech failed detection
            # with "the video's length is unknown too" even though the length
            # had just been probed. Stages that persist progress (clipTarget,
            # stageStatus) have the same requirement.
            fresh = self.store.get_job(job_id)
            if fresh:
                job.clear()
                job.update(fresh)
            self._mark_stage_running(job_id, stage)
            try:
                with ledger.stage(stage):
                    result = fn()
            except BaseException:
                self._mark_stage_failed(job_id, stage, *self._fail_args_for(sys.exc_info()))
                raise
            self._mark_stage_done(job_id, stage)
            return result

        transcript = _run_stage("transcribe", lambda: self._stage_transcribe(job, src, ws))
        self._transcript_size_note(len(transcript.words))
        # The weights are the single largest thing a batch puts on the disk and
        # nothing after transcription reads them: give them back now. The
        # bucket keeps the copy, so the next batch restores instead of
        # downloading.
        try:
            freed = modelstore.release_all(lambda m: self._log(job_id, m))
            if freed.get("freed"):
                self._log(
                    job_id,
                    f"[disk] model weights back to the bucket: "
                    f"{freed['freed'] / 1048576:.0f} MB freed after transcription",
                )
        except Exception as exc:  # noqa: BLE001 - never fail a good batch over this
            self._log(job_id, f"[disk] could not release model weights: {exc}")

        self._checkpoint(job_id, "transcribing")
        candidates = _run_stage("detect", lambda: self._stage_detect(job, transcript, ws))

        self._checkpoint(job_id, "finding")
        _run_stage("reframe", lambda: self._stage_render(job, src, transcript, candidates, ws))
        self._render_workload_note(len(candidates), float(job.get("durationSec") or 0.0))

        _run_stage("captions", lambda: self._stage_captions(job, transcript, ws))
        try:
            _run_stage("export", lambda: self._stage_export(job, src, ws))

            # A finished batch must give back ITS OWN scratch. That is the only
            # hard check: the global free-space delta moves for reasons the
            # batch does not own (model cache, logs, WAL, other jobs) and using
            # it as a verdict failed a batch whose verified clips were already
            # in the bucket. Clean first, then verify the folder is gone.
            self._cleanup_job_scratch(job_id, full=True)
            ok, why = ledger.scratch_check()
            self._log(job_id, f"[disk] scratch check: {why}")
            if not ok:
                # Safe cleanup, then warn. Never a failure: the clips rendered
                # and verified, and deleting our own folder cannot lose them.
                try:
                    self._cleanup_job_scratch(job_id, full=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[job {job_id}] second cleanup pass failed: {exc}")
                retry_ok, retry_why = ledger.scratch_check()
                if retry_ok:
                    self._log(
                        job_id,
                        "[disk] cleanup warning: scratch was left behind, "
                        "second pass removed it — batch unaffected",
                    )
                else:
                    self._log(
                        job_id,
                        f"[disk] cleanup warning: scratch left behind ({retry_why}) "
                        "— the batch is complete and its clips are in the bucket; "
                        "a boot sweep will remove it",
                    )
                    try:
                        self.store.update_job(
                            job_id,
                            lambda j: j.update({
                                "diskWarning": f"cleanup left scratch behind: {retry_why}",
                            }),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"[job {job_id}] could not record disk warning: {exc}")
            # Information only, always logged, never a gate.
            _within, delta = ledger.global_delta()
            self._log(job_id, f"[disk] free-space delta (info): {delta}")
            self._finish(job_id, time.time() - t_start)
            self._mark_stage_done(job_id, "export")
        finally:
            ledger.stop()
            self._persist_ledger(job_id, ledger)
            self._ledger = None
            # `finally` cleanup: a crashed or paused run must not leave its
            # source copy behind either. Resume keeps the cheap artifacts
            # (transcript chunks, cuts) and re-fetches the video from the
            # bucket when it needs it again.
            self._cleanup_job_scratch(job_id, full=False)
            # A model is never kept warm on the disk between runs, failed or
            # not: the bucket copy is the source of truth for the next one.
            try:
                modelstore.release_all(lambda m: self._log(job_id, m))
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _detail(exc: BaseException) -> str:
        """Full traceback for the job log / errorDetail (structured errors)."""
        return "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-4000:]

    def _fail_args_for(self, exc_info) -> tuple[str, str]:
        """Build (message, detail) for _mark_stage_failed from an exception.

        A memory failure gets a plain sentence (the traceback stays in
        `errorDetail`, behind "Show details"), because "mkl_malloc: failed to
        allocate memory" tells the user nothing they can act on.
        """
        exc_type, exc_value, exc_tb = exc_info
        msg = f"{exc_type.__name__}: {exc_value}"
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))[-4000:]
        if asr_iso.is_memory_error(f"{exc_value}"):
            msg = asr_iso.plain_memory_reason()
            detail = "memory failure\n" + detail
        return msg, detail

    # ------------------------------------------------------------------
    # stage 1: fetch
    # ------------------------------------------------------------------

    def _stage_fetch(self, job: dict[str, Any], ws: StageWorkspace) -> Any:
        job_id = job["_id"]
        if ws.is_done("fetch"):
            # A bucket source has no local path to stat: `done_meta["path"]` is
            # an OBJECT KEY, so the old `Path(key).exists()` check was always
            # false and every resume re-streamed the whole video. On a 20:37
            # source that is nine sections and ~13 minutes re-downloaded after
            # each host restart, for objects that were still in the bucket.
            meta = ws.read_json("fetch", "meta.json") or {}
            if meta.get("remote") and meta.get("videoKey"):
                remote = self._remote_from_meta(job_id, meta)
                if remote is not None and self._remote_source_intact(remote, job_id):
                    held = len(remote.sections) or 1
                    self._log(
                        job_id,
                        f"resume: reusing the source in the bucket ({held} object(s), "
                        f"{remote.method}) — nothing is downloaded again",
                    )
                    self._progress(job_id, "grabbing", 100, "Source reused from the bucket")
                    return remote
            src = Path((ws.done_meta("fetch") or {}).get("path", ""))
            if src.exists():
                self._log(job_id, f"resume: reusing downloaded source {src.name}")
                self._progress(job_id, "grabbing", 100, "Reusing the downloaded video")
                return src

        dest_dir = self.settings.app.paths.downloads_dir / job_id
        self._progress(job_id, "grabbing", 0, "Clippy is grabbing the video...")
        # Legacy/partial job documents can carry source=None (seen in a real
        # data/jobs/*.json): logging it with [:80] raised TypeError and turned
        # a data problem into a downstream pipeline crash.
        src_ref = job.get("source") or "<missing source>"
        self._log(job_id, f"Stage: fetch — {src_ref[:80] if isinstance(src_ref, str) else src_ref}")

        if job.get("sourceType") == "upload":
            # Search BOTH storage locations. The expected name is tried first
            # so a job whose recorded filename is authoritative still resolves
            # the same way it always did; the general sweep is the fallback for
            # a job written under the other root.
            source_file = job.get("source") or ""
            expected = "source" + Path(source_file).suffix.lower() or ".mp4"
            src: Path | None = None
            for candidate_dir in self.settings.app.paths.job_source_dirs(job_id):
                named = candidate_dir / expected
                if named.exists() and named.stat().st_size > 0:
                    src = named
                    break
            if src is None:
                src = self.store.find_source_file(job_id)
            if src is None:
                searched = ", ".join(
                    str(d) for d in self.settings.app.paths.job_source_dirs(job_id)
                )
                raise download.DownloadError(
                    f"The uploaded file for this batch is no longer on the server. "
                    f"It was looked for as '{expected}' in: {searched}. Nothing else "
                    f"was deleted — every finished clip from this batch is still "
                    f"downloadable. Re-submit the file with 'Attach file' to start a "
                    f"new batch."
                )
            self._progress(job_id, "grabbing", 100, "Upload verified on disk")
            self._log(job_id, f"upload present: {src.name} ({src.stat().st_size} bytes)")
        else:
            url = job.get("source")
            if not url:
                raise download.DownloadError(
                    "This batch has no video URL recorded — create one from the dashboard "
                    "with 'Paste YouTube/TikTok URL' or 'Attach file'. The job is now "
                    "stopped; your finished clips are on the bucket and downloadable."
                )
            if not isinstance(url, str) or not url.strip():
                raise download.DownloadError(
                    "The job's `source` is not a usable URL. It was stored as "
                    f"{type(url).__name__} and cannot be fetched. Re-create the batch "
                    "from the dashboard with a real YouTube/TikTok URL, or use 'Attach "
                    "file' for a local upload."
                )
            url = url.strip()
            # The source object is deleted from the bucket when a batch finishes
            # (it is the largest thing in a 5 GB plan). A re-render of a
            # finished batch therefore has to pull it again — say so BEFORE
            # doing it, because that is a fresh download of a video the user
            # thinks is already handled.
            if job.get("completedAt") or ws.is_done("fetch"):
                host = url.split("//")[-1].split("/")[0] if "//" in url else url[:40]
                self._log(
                    job_id,
                    f"the source is no longer stored (it is deleted when a batch "
                    f"finishes) — fetching it again from {host}; the transcript "
                    f"and finished clips are kept",
                )
                self._progress(
                    job_id, "grabbing", 0,
                    f"Source expired from the bucket — fetching it again from {host}",
                )
                self._patch_job(job_id, sourceRefetched=True)

            # With a bucket configured, the source NEVER lands on this disk:
            # video-only and audio-only are piped straight into two objects.
            # There is deliberately no local fallback — a 168 MB source is the
            # difference between finishing the batch and "only 234 MB free".
            from app import objectstore

            if objectstore.configured():
                return self._capture_remote(job, ws, url)

            def on_progress(frac: float, _eta: float | None) -> None:
                self._progress(job_id, "grabbing", frac, "Clippy is grabbing the video...")

            fetch_log_out: list[str] = []
            def _on_dl_log(line: str) -> None:
                fetch_log_out.append(line)
                self._log(job_id, line)

            try:
                src = download.download_url(
                    url, dest_dir, self.settings,
                    on_progress=on_progress,
                    on_log=_on_dl_log,
                )
            except download.DownloadError as exc:
                # Make the fetch failure diagnosable instead of a generic
                # "Worker process exited unexpectedly" / empty error message.
                #
                # This MUST live inside the except: it used to sit after the
                # try/except, where `ydl_debug` is only bound on the failure
                # path — so a SUCCESSFUL download fell through to the logging
                # lines and died with UnboundLocalError. Every URL job was
                # therefore reported as "Pipeline error: cannot access local
                # variable 'ydl_debug'", while manual uploads still worked.
                ydl_debug = ""
                try:
                    import yt_dlp as _ydl

                    version = getattr(getattr(_ydl, "version", None), "__version__", None)
                    if not isinstance(version, str):
                        version = getattr(_ydl, "__version__", "unknown")
                    ydl_debug = (
                        "yt-dlp "
                        + str(version)
                        + "\n"
                        + "DOWNLOAD ATTEMPT LOG:\n"
                        + "\n".join(fetch_log_out[-40:])
                        + "\n"
                        + "".join(
                            traceback.format_exception_only(type(exc), exc)
                        )[-1200:]
                    )
                except Exception:
                    ydl_debug = "yt-dlp not importable during failure diagnostics"
                cfg_snapshot = ""
                try:
                    s = self.settings.download
                    cfg_snapshot = (
                        "download config: max_height="
                        + repr(getattr(s, "max_height", None))
                        + ", auto_update="
                        + repr(getattr(s, "auto_update", None))
                        + ", cookies_from_browser="
                        + repr(getattr(s, "cookies_from_browser", None))
                        + ", cookies_file="
                        + repr(bool(getattr(s, "cookies_file", "")))
                    )
                except Exception:
                    cfg_snapshot = "download config unavailable"
                self._log(
                    job_id,
                    "DOWNLOAD FAILED — yt-dlp debug:\n" + ydl_debug + "\n" + cfg_snapshot,
                )
                # No second `on_log(...)` call here: `on_log` is a parameter of
                # download.download_url, not of this method, so referring to it
                # raised NameError and REPLACED the real failure with
                # "name 'on_log' is not defined" — the download error (YouTube's
                # bot check) never reached the dashboard. The self._log above
                # already records the same debug text.
                raise
        video.verify_source(src)
        meta = video.probe(src)
        max_seconds = self.settings.hosted.max_source_min * 60
        if meta["duration"] > max_seconds:
            raise video.RenderError(
                f"source is {meta['duration'] / 60:.0f} min; this plan accepts "
                f"up to {self.settings.hosted.max_source_min} min"
            )
        job = self.store.get_job(job_id) or job
        job["durationSec"] = meta["duration"]
        job["downloadSource"] = "upload" if job.get("sourceType") == "upload" else "yt-dlp"
        self.store.write_job(job)
        ws.write_json("fetch", "meta.json", {"path": str(src), **meta})
        ws.mark_done("fetch", artifacts=["fetch/meta.json"], path=str(src), **meta)
        # Uploads were the only sources that ever reached the bucket; a URL
        # fetch wrote straight to the disk and stayed there. So a host restart
        # mid-run had to pull the whole video down again (445 MB for a 22-min
        # source on this box), and the batch had no durable copy until the
        # completion reclaim pushed one. The local file stays — the renderer
        # cuts from it — but the bucket copy now exists from the first minute.
        self._push_source(job_id, src)
        return src

    @staticmethod
    def _remote_from_meta(
        job_id: str, meta: dict[str, Any],
    ) -> "source_stream.RemoteSource | None":
        """Rebuild the RemoteSource a previous attempt captured.

        `fetch/meta.json` is the durable record of the capture: the object
        keys, the section list with its time spans, and the probe the renderer
        needs. It is all that is required to keep cutting clips without
        touching yt-dlp again.
        """
        try:
            return source_stream.RemoteSource(
                job_id,
                video_key=str(meta.get("videoKey") or ""),
                audio_key=str(meta.get("audioKey") or ""),
                probe={
                    k: meta[k]
                    for k in ("duration", "width", "height", "fps")
                    if meta.get(k) is not None
                },
                method=str(meta.get("method") or "sections"),
                sections=list(meta.get("sections") or []),
                bytes_total=int(meta.get("bytes") or 0),
                format_id=str(meta.get("format") or ""),
                audio_format=str(meta.get("audioFormat") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - a broken meta means re-capture
            logger.warning(f"[job {job_id}] could not rebuild the bucket source: {exc}")
            return None

    def _remote_source_intact(
        self, src: "source_stream.RemoteSource", job_id: str,
    ) -> bool:
        """Are the captured objects still in the bucket?

        Checked with HEAD on the video, the first and last section, and the
        audio track — the objects the later stages actually read. A missing
        one means the capture has to happen again, which is the honest answer
        rather than a render that fails halfway through with a 403.
        """
        from app import objectstore

        keys = [src.video_key, src.audio_key]
        if src.sections:
            keys += [str(s.get("key") or "") for s in (src.sections[0], src.sections[-1])]
        for key in keys:
            if not key:
                return False
            try:
                head = objectstore.head(key) or {}
            except Exception as exc:  # noqa: BLE001 - unknown means re-capture
                self._log(job_id, f"resume: could not verify the bucket source ({exc})")
                return False
            if int(head.get("size") or 0) <= 0:
                self._log(
                    job_id,
                    f"resume: {key.split('/')[-1]} is no longer in the bucket — "
                    "capturing the source again",
                )
                return False
        return True

    def _capture_remote(
        self, job: dict[str, Any], ws: StageWorkspace, url: str,
    ) -> "source_stream.RemoteSource":
        """Stream the URL into the bucket and return the keys, not a file.

        A bucket that is down is waited out, not worked around: retrying with
        "bucket unavailable" keeps the job and its progress, while falling back
        to a local copy is what made this box run out of disk in the first
        place.
        """
        job_id = job["_id"]
        attempts = 0
        while True:
            attempts += 1
            try:
                with self._disk_stage("fetch-stream"):
                    src = source_stream.capture(
                        url, job_id, self.settings,
                        on_log=lambda line: self._log(job_id, line),
                        on_progress=lambda frac: self._progress(
                            job_id, "grabbing", max(0.0, min(0.95, frac * 0.95)),
                            "Streaming the video into the bucket...",
                        ),
                        should_pause=lambda: self.pause_requested(job_id),
                        expected_seconds=float(job.get("durationSec") or 0.0),
                    )
                break
            except source_stream.StreamError as exc:
                if not exc.bucket_down or attempts >= 20:
                    raise download.DownloadError(
                        f"could not stream the source into the bucket: {exc}"
                    ) from exc
                wait = min(60, 5 * attempts)
                self._progress(
                    job_id, "waiting", 0,
                    f"Bucket unavailable, retrying in {wait}s ({attempts}/20)",
                )
                self._log(
                    job_id,
                    f"[stream] bucket unavailable ({exc}) — retrying in {wait}s, "
                    "keeping this job's progress; nothing is being stored locally",
                )
                time.sleep(wait)

        probe = src.probe or {}
        if int(probe.get("width") or 0) <= 0 or not probe.get("height"):
            raise video.RenderError(
                "the source streamed into the bucket but its geometry could not "
                "be read back through the presigned URL — the object may be "
                f"truncated ({src.describe()})"
            )
        duration = float(probe.get("duration") or 0.0)
        max_seconds = self.settings.hosted.max_source_min * 60
        if duration > max_seconds:
            raise video.RenderError(
                f"source is {duration / 60:.0f} min; this plan accepts up to "
                f"{self.settings.hosted.max_source_min} min"
            )
        patch = self._patch_job(
            job_id,
            durationSec=duration,
            downloadSource="bucket-stream",
            sourceRemote=True,
            sourceStream=src.describe(),
        )
        ws.write_json("fetch", "meta.json", {
            "remote": True, **src.describe(), **probe,
        })
        ws.mark_done(
            "fetch", artifacts=["fetch/meta.json"], path=src.video_key, **probe,
        )
        self._progress(job_id, "grabbing", 100, "Source is in the bucket — no local copy")
        self._log(
            job_id,
            f"source streamed to the bucket: video {src.format_id or '(sections)'} "
            f"+ audio {src.audio_format or '?'} via {src.method}; "
            f"local disk holds 0 MB of it"
            + (f" (job {patch.get('_id')})" if patch else ""),
        )
        return src

    def _disk_stage(self, name: str):
        """Stage-scoped measurement when a ledger exists, else a no-op."""
        if self._ledger is not None:
            return self._ledger.stage(name)
        from contextlib import nullcontext

        return nullcontext({})

    def _push_source(self, job_id: str, src: Path) -> None:
        """Copy a fetched source into durable storage; keep the local file.

        Best effort by design: a storage outage must not fail a job whose video
        is on disk and perfectly usable.
        """
        from app import objectstore

        try:
            cfg = objectstore.config_from_env()
            if not cfg.enabled or not src.exists():
                return
            key = objectstore.upload_key(job_id, src.name)
            if not objectstore.upload_file(src, key, cfg):
                self._log(job_id, "durable copy of the source failed — it stays local")
                return
            job = self.store.get_job(job_id) or {}
            job["sourceRemote"] = True
            self.store.write_job(job)
            self._log(
                job_id,
                f"source copied to durable storage ({src.stat().st_size / 1048576:.0f} MB)",
            )
        except Exception as exc:  # noqa: BLE001 - never fail a fetch over a copy
            logger.warning(f"[job {job_id}] source push to durable storage failed: {exc}")

    # ------------------------------------------------------------------
    # stage 2: transcribe (incremental, resumable)
    # ------------------------------------------------------------------

    def _stage_transcribe(
        self, job: dict[str, Any], src: Path, ws: StageWorkspace
    ) -> subtitle.Transcript:
        job_id = job["_id"]
        if ws.is_done("transcribe"):
            data = ws.read_json("transcribe", "words.json")
            if data:
                words = [subtitle.Word(**w) for w in data.get("words", [])]
                self._log(job_id, f"resume: reusing transcript ({len(words)} words)")
                self._progress(job_id, "transcribing", 100, "Transcript reused from earlier run")
                transcript = subtitle.Transcript(
                    words=words,
                    duration=float(data.get("duration", 0)),
                    model_size=str(data.get("model", self.settings.whisper.model)),
                    engine=str(data.get("engine", "")),
                    # Recorded detected language survives the resume; older
                    # transcripts without the key read as "" (unknown → guess).
                    language=str(data.get("language", "") or ""),
                )
                self._set_word_count(job_id, len(words))
                return transcript

        # ---- engine + ETA prior -------------------------------------------
        engine = subtitle.engine_status(self.settings)
        rtf_store = eta.store(self.settings.app.paths.data_dir)
        rtf, rtf_source = rtf_store.effective_rtf(
            engine.engine, engine.device, engine.model, engine.compute_type
        )
        job_now = self.store.get_job(job_id) or job
        duration = float(job_now.get("durationSec") or 0) or 0.0
        prior_eta = eta.eta_from_prior(duration, rtf) if duration > 0 else None
        self._log(
            job_id,
            f"Stage: transcribe — engine {engine.engine} "
            f"({engine.device}/{engine.compute_type}, batch {engine.batch_size}), "
            f"model '{engine.model}' | ETA basis: RTF {rtf:.3f} ({rtf_source})"
            + (f" → ~{prior_eta / 60:.1f} min for {duration / 60:.1f} min of audio" if prior_eta else ""),
        )
        detail = f"{engine.engine} is transcribing word timings..."
        self._progress(job_id, "transcribing", 0, detail, prior_eta_sec=prior_eta)

        # Chunk counting. This is the fix for "transcription sits at 0%": the
        # fraction alone cannot distinguish "0% because it just started" from
        # "0% because nothing is happening", because a Whisper batch yields no
        # segment until the whole batch decodes. Naming the chunk answers the
        # question the percentage could not — and `done/total` advances on every
        # finished chunk, so the number moves at least eleven times on a
        # 54-minute source instead of never.
        chunk_state = {"done": 0, "total": 1, "chunk": 0}
        # Hoisted so `on_chunk` can tell a cache reuse from a real decode in
        # the log line, and so both callbacks and `transcribe` agree on the one
        # cache location.
        chunk_cache_dir = ws.stage_dir("transcribe") / "chunks"

        def on_chunk(index: int, total: int, *, phase: str, resuming: bool = False) -> None:
            chunk_state["total"] = max(1, total)
            if phase == "start":
                chunk_state["chunk"] = index + 1
                # Deliberately leaves `progress` alone on a chunk START. Faking
                # movement that has not happened is worse than a number that
                # is merely still — the text is what tells the user work is in
                # flight, and the heartbeat keeps the liveness timestamp fresh.
                self._progress(
                    job_id, "transcribing",
                    chunk_state["done"] / chunk_state["total"],
                    f"Transcribing chunk {index + 1} of {total}",
                    unit_label="chunks",
                    units_done=chunk_state["done"], units_total=chunk_state["total"],
                    prior_eta_sec=prior_eta,
                )
                self._log(
                    job_id,
                    f"chunk {index + 1}/{total} "
                    + ("resuming from cache" if _chunk_cached(chunk_cache_dir, index) else "decoding"),
                )
            else:
                chunk_state["done"] = max(chunk_state["done"], index + 1)
                self._progress(
                    job_id, "transcribing",
                    chunk_state["done"] / chunk_state["total"],
                    (
                        f"Transcribing chunk {index + 1} of {total}"
                        + (" (from cache)" if resuming else " — done")
                    ),
                    unit_label="chunks",
                    units_done=chunk_state["done"], units_total=chunk_state["total"],
                    prior_eta_sec=prior_eta,
                )

        def on_detail(text: str) -> None:
            """Ladder detail for the Transcribing chip: "chunk 3 of 9 (small)"."""
            self._progress(
                job_id, "transcribing",
                chunk_state["done"] / chunk_state["total"],
                f"Transcribing {text}",
                unit_label="chunks",
                units_done=chunk_state["done"], units_total=chunk_state["total"],
                prior_eta_sec=prior_eta,
            )

        def on_progress(frac: float) -> None:
            # Sub-chunk smoothing only. It must never move the progress BACKWARDS
            # past what the chunk counter already proved, which is what makes
            # the first chunk usable instead of a dead 0%.
            base = chunk_state["done"] / chunk_state["total"]
            frac = max(frac, base)
            self._progress(
                job_id, "transcribing", frac,
                (
                    f"Transcribing chunk {chunk_state['chunk']} "
                    f"of {chunk_state['total']}"
                    if chunk_state["total"] > 1 else detail
                ),
                unit_label="chunks",
                units_done=chunk_state["done"], units_total=chunk_state["total"],
                prior_eta_sec=prior_eta,
            )

        # A single Whisper chunk can run for minutes on a small host. The
        # heartbeat keeps the job visibly alive across it; the chunk callback
        # still owns the real progress numbers.
        # Per-job caption language (ENG/ESP/Auto) steers transcription ONLY
        # when explicit; "auto" keeps the engine's own detection.
        job_language = str(job.get("language") or "auto").strip().lower()
        transcribe_language = None if job_language == "auto" else job_language
        # A bucket source has no local video, and it does not need one: the
        # audio-only object is 7-19 MB for 20 minutes. It lives on disk only
        # while the decoder is running, then it is deleted.
        asr_input: Any = src
        audio_local: Path | None = None
        if isinstance(src, source_stream.RemoteSource):
            with self._disk_stage("transcribe-audio"):
                audio_local = source_stream.download_audio_local(
                    src,
                    ws.stage_dir("transcribe") / "audio.m4a",
                    cap_bytes=40 * 1024 * 1024,
                )
            asr_input = audio_local
            self._log(
                job_id,
                f"transcribing from the audio-only stream "
                f"({diskwatch.mb(audio_local.stat().st_size)} MB on disk, no video)",
            )

        with self._heartbeat(job_id, "transcribe"):
            try:
                transcript = subtitle.transcribe(
                    asr_input, self.settings, on_progress=on_progress,
                    cache_dir=chunk_cache_dir,
                    should_pause=lambda: self.pause_requested(job_id),
                    on_chunk=on_chunk,
                    language=transcribe_language,
                    on_detail=on_detail,
                )
            finally:
                # Deleted even when transcription fails: the audio is one
                # presigned GET away from being fetched again.
                if audio_local is not None:
                    try:
                        freed = audio_local.stat().st_size if audio_local.exists() else 0
                        audio_local.unlink(missing_ok=True)
                        self._log(
                            job_id,
                            f"transcription done — removed the audio copy "
                            f"({diskwatch.mb(freed)} MB)",
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._log(job_id, f"could not remove the audio copy: {exc}")
        # Truthful metadata: which rung of the ladder produced these words,
        # and whether the UI must label the transcript as a downshift.
        if transcript.attempts or transcript.label:
            fresh = self.store.get_job(job_id)
            if fresh is not None:
                fresh.update({
                    "asrAttempts": transcript.attempts,
                    "transcriptLabel": transcript.label,
                    "asrPeakRssMb": transcript.peak_rss_mb,
                })
                self.store.write_job(fresh)
            if transcript.label:
                self._log(job_id, f"transcript: {transcript.label}")
        # Record what this host actually achieved so the next job's ETA starts
        # from a measurement instead of the documented prior.
        if transcript.rtf is not None:
            rtf_store.observe(
                eta.rtf_key(transcript.engine or engine.engine, engine.device, engine.model),
                transcript.rtf,
                audio_seconds=transcript.duration,
                engine=transcript.engine or engine.engine,
                device=engine.device,
                model=engine.model,
            )
        ws.write_json("transcribe", "words.json", {
            "words": [w.as_dict() for w in transcript.words],
            "duration": transcript.duration,
            "model": transcript.model_size,
            "engine": transcript.engine,
            "rtf": transcript.rtf,
            # Record the DETECTED language so the caption stage never guesses
            # when the engine already knows. Older transcripts without the key
            # get "" and fall back to the heuristic guess.
            "language": getattr(transcript, "language", "") or "",
        })
        ws.mark_done("transcribe", artifacts=["transcribe/words.json"],
                     words=len(transcript.words))
        self._set_word_count(job_id, len(transcript.words))
        self._log(
            job_id,
            f"transcribed {len(transcript.words)} words in {transcript.duration:.1f}s of audio "
            f"via {transcript.engine or engine.engine}"
            + (f" (RTF {transcript.rtf:.3f})" if transcript.rtf is not None else ""),
        )
        return transcript

    def _set_word_count(self, job_id: str, count: int) -> None:
        job = self.store.get_job(job_id)
        if job is not None:
            job["transcriptWordCount"] = count
            self.store.write_job(job)

    # ------------------------------------------------------------------
    # stage 3: detect (Gemini word indices → local timestamps)
    # ------------------------------------------------------------------

    def _release_memory_before_detect(self, job_id: str) -> None:
        """Drop decoded-audio/word buffers and return freed arenas to the OS."""
        import gc

        try:
            import ctypes
            import resource

            before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            self._log(
                job_id,
                f"memory: released ASR buffers before detection (peak {before} MB)",
            )
        except Exception as exc:  # noqa: BLE001 - never fail detection on a trim
            logger.debug(f"[job {job_id}] memory release before detection skipped: {exc}")

    def _stage_detect(
        self, job: dict[str, Any], transcript: subtitle.Transcript, ws: StageWorkspace
    ) -> list[dict[str, Any]]:
        job_id = job["_id"]
        if ws.is_done("detect"):
            cached = ws.read_json("detect", "candidates.json")
            if cached:
                # Re-attach the clip ids minted when these candidates were
                # persisted. candidates.json is written BEFORE create_clips, so
                # it has no `_id`; without this every resume handed reframe
                # id-less cuts, export fabricated "clip_000", and the run died
                # with FileNotFoundError instead of rendering.
                ids = self.store.summary_clip_ids(job_id)
                for i, c in enumerate(cached):
                    if not c.get("_id") and i < len(ids):
                        c["_id"] = ids[i]
                self._log(job_id, f"resume: reusing {len(cached)} detected candidates")
                self._progress(job_id, "finding", 100, "Reusing AI clip selection")
                # Re-derive the count explanation from the candidates we are
                # actually returning. Without this a resumed job showed an
                # empty "N found, M scored above the floor" line, which is the
                # one number that tells the user whether to move the cap.
                # `detected` is not recoverable here — the pre-gate total is
                # not stored — so it is left alone rather than invented.
                floor = float(self.settings.gemini.min_score)
                scores = [float(c.get("score") or 0.0) for c in cached]
                target = int(job.get("requestedClipTarget") or job.get("clipTarget") or len(cached))
                patch: dict[str, Any] = {
                    "clipTarget": target,
                    "requestedClipTarget": target,
                    "candidateTarget": len(cached),
                    "clipsAboveScore": sum(1 for s in scores if s >= floor),
                    "clipsBelowScore": sum(1 for s in scores if s < floor),
                    "clipScoreFloor": floor,
                }
                if job.get("candidatesDetected") is None:
                    patch["candidatesDetected"] = len(cached)
                self._patch_job(job_id, **patch)
                return cached

        # Whisper on a 20-minute source leaves large native arenas behind. The
        # ASR itself runs in child processes that already trim on exit, but the
        # parent still holds decoded-word buffers and numpy arenas; returning
        # them BEFORE the detection call is what keeps a 2 GiB container from
        # picking the API process as the OOM victim.
        self._release_memory_before_detect(job_id)

        opts = job.get("options", {}) or {}
        # Name the backend that is ACTUALLY running. These two strings used to
        # be hardcoded to "Gemini", but [llm] provider is `deepseek` here, so a
        # real job announced "Gemini is scoring viral moments..." and then
        # logged "Detecting highlights ... via deepseek (deepseek-flash)" one
        # line later. That contradiction is what made a funded-key/billing
        # problem look like "the Gemini server worker is down" — the UI named a
        # provider that was never called.
        #
        # `llm.provider_label` is already used a few lines below for the chunk
        # cache, so the real name is available; deriving it here keeps the two
        # in step by construction.
        backend = llm.provider_label(self.settings)
        pretty = backend.split(" (")[0] or backend
        self._progress(job_id, "finding", 0,
                       f"{pretty} is scoring viral moments...")
        self._log(job_id, f"Stage: detect — {backend} clip detection (word indices)")
        words = [w.as_dict() for w in transcript.words]

        # How many clips this source is worth, decided ONCE and written to the
        # job so the UI, the logs and the render loop all use the same number.
        # An explicit maxClips on the job is the user overriding this, and
        # always wins.
        duration = float(job.get("durationSec") or 0.0)
        override = opts.get("maxClips")
        if override:
            target, reason = int(override), f"you asked for {override}"
        elif self.settings.gemini.auto_clip_count:
            target, reason = clipcount.suggest(duration, self.settings.gemini.max_clips)
        else:
            target, reason = self.settings.gemini.max_clips, "flat count (auto scaling off)"
        # Explicit N is a quota. Ask detection for N + 40% (at least N + 3)
        # candidates so a failed render has a real next moment to replace it
        # with, instead of silently returning fewer than requested.
        detect_target = clipcount.candidate_target(target, exact=bool(override))
        extra = detect_target - target
        if extra:
            reason = f"{reason}; scoring {detect_target} candidates for {target} slots"
        self._log(job_id, f"Clip target: {target} — {reason}")
        self._patch_job(
            job_id,
            clipTarget=target,
            requestedClipTarget=target,
            candidateTarget=detect_target,
            clipTargetReason=reason,
        )

        # stage 2: transcribe gets its own log callback.
        def _on_transcribe_log(line: str) -> None:
            self._log(job_id, line)

        # Detection's own callbacks. `detect_clips` takes `on_log(line)` and
        # `on_progress(fraction, detail)` with a 0..1 fraction, and the call
        # below passed names that were never defined in this scope — every job
        # with a transcript died at "finding" with NameError while the API kept
        # serving. Keep them here, next to the call that uses them.
        def on_log(line: str) -> None:
            self._log(job_id, line)

        # Detection reports twice: once per chunk while its calls are still in
        # flight (that pass may cap below 1.0) and again in its sequential pass.
        # Never let the second, lower number drag the bar backwards.
        best_detect_frac = 0.0

        def on_progress(frac: float, detail: str | None = None) -> None:
            nonlocal best_detect_frac
            best_detect_frac = max(best_detect_frac, max(0.0, min(1.0, float(frac))))
            self._progress(
                job_id,
                "finding",
                best_detect_frac * 100,
                detail or "Scoring viral moments...",
            )

        # Source length drives the no-speech fallback (evenly spaced clips).
        # Prefer the job document, fall back to the fetch stage's own record,
        # so an older job whose durationSec was never patched still works.
        duration_sec = float(job.get("durationSec") or 0.0)
        if duration_sec <= 0:
            duration_sec = float((ws.read_json("fetch", "meta.json") or {}).get("duration") or 0.0)
            if duration_sec > 0:
                self._patch_job(job_id, durationSec=duration_sec)

        report: dict[str, Any] = {}
        candidates = llm.detect_clips(
            words,
            self.settings,
            min_clip_sec=opts.get("minClipSec"),
            max_clip_sec=opts.get("maxClipSec"),
            max_clips=detect_target,
            # A number the user picked is a count they expect to receive. The
            # auto-scaled target is a ceiling and keeps the old "only real
            # highlights" behaviour; an explicit 5/8/10/15/20 is a quota.
            exact_count=bool(override),
            # The source length is the last thing detection needs when there is
            # no transcript to work from: it is what lets the pipeline return
            # evenly spaced clips instead of failing a job whose source simply
            # has no speech in it.
            duration_sec=duration_sec,
            on_log=on_log,
            on_progress=on_progress,
            report=report,
            # Per-chunk persistence: a pause or host restart during chunk 5 of
            # 7 resumes at chunk 6 instead of re-running — and re-billing —
            # all seven calls.
            chunk_cache=_StageChunkCache(ws, model=llm.provider_label(self.settings)),
            should_pause=lambda: self.pause_requested(job_id),
        )
        # Say WHICH detector produced these clips. A local fallback is still a
        # finished batch, but labelling it "AI highlights" would be a lie, and
        # it is the one fact that tells the user whether to top up the account
        # and hit Re-detect.
        if report.get("detectionFallback"):
            self._log(job_id, str(report.get("detectionFallbackLabel") or "detection: local fallback"))
            self._patch_job(
                job_id,
                detectionFallback=str(report.get("detectionFallback")),
                detectionFallbackLabel=str(
                    report.get("detectionFallbackLabel") or "detection: local fallback"
                ),
                detectionFallbackReason=str(report.get("detectionFallbackReason") or "")[:400],
            )
        ws.write_json("detect", "candidates.json", candidates)
        # Persist one stage artifact per candidate so a retry can tell which clips
        # were already rendered and which still need the export stage.
        ws.write_clips("detect", candidates)
        # Keep the batch-size explanation truthful on a FRESH run, not only on
        # a resume: when the count was filled with weaker moments, the job card
        # must be able to say so instead of pretending every clip cleared the
        # highlight floor.
        floor = float(self.settings.gemini.min_score)
        scores = [float(c.get("score") or 0.0) for c in candidates]
        self._patch_job(
            job_id,
            candidatesDetected=len(candidates),
            clipsAboveScore=sum(1 for s in scores if s >= floor),
            clipsBelowScore=sum(1 for s in scores if s < floor),
            clipScoreFloor=floor,
        )
        # Record HOW these clips were chosen so the dashboard can say "AI highlights
        # via deepseek" instead of a blank "detectedSource". The provider label is
        # already used a few lines above for the chunk cache and the log line, so this
        # keeps all three named the same by construction rather than by coincidence.
        self.store.create_clips(
            job_id, candidates,
            detected_source=llm.provider_label(self.settings),
            words=words,
        )
        # Mark the workspace stage done. WITHOUT this marker every resume re-ran
        # detection: one more LLM call, and create_clips minted a BRAND NEW set
        # of clip documents — which is exactly how one batch ended up showing
        # each clip twice (rendered old card + "Not rendered yet" new twin).
        ws.mark_done("detect", artifacts=["detect/candidates.json"], count=len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # stage 4: reframe (VAD + reframe model)
    # ------------------------------------------------------------------

    def _stage_render(
        self,
        job: dict[str, Any],
        src: Path,
        transcript: subtitle.Transcript,
        candidates: list[dict[str, Any]],
        ws: StageWorkspace,
    ) -> None:
        job_id = job["_id"]
        if ws.is_done("reframe"):
            self._log(job_id, "resume: reframing already done")
            self._progress(job_id, "reframing", 100, "Clip boundaries already computed")
            return

        self._check_disk_space(job_id, float(job.get("durationSec") or 0.0))

        requested_target = int(
            job.get("requestedClipTarget") or job.get("clipTarget")
            or self.settings.gemini.max_clips
        )
        clips = clipcount.reasonable_clips(
            candidates,
            max_clips=job.get("candidateTarget") or len(candidates),
            fallback=self.settings.gemini.max_clips,
        )
        job = self._patch_job(
            job_id,
            clipTarget=requested_target,
            requestedClipTarget=requested_target,
            candidateTarget=len(clips),
        )

        # Clamp every cut into the source duration before reframe sees it, so a
        # candidate whose end drifted past the source (or whose start is below 0)
        # never asks ffmpeg to seek past EOF.
        src_dur = float(job.get("durationSec") or 0.0)
        if src_dur > 0:
            for c in clips:
                s = float(c.get("start", 0.0) or 0.0)
                e = float(c.get("end", 0.0) or 0.0)
                if s < 0:
                    s = 0.0
                if e > src_dur:
                    e = src_dur
                if e <= s:
                    e = s + float(self.settings.gemini.min_clip_sec or 5.0)
                c["start"] = round(s, 3)
                c["end"] = round(e, 3)

        rtf = self._current_rtf()
        self._progress(
            job_id, "reframing", 0,
            f"Deciding which {len(clips)} clips to render...",
        )
        with self._heartbeat(job_id, "reframe"):
            cuts = reframe.find_cuts(
                src, transcript, clips, self.settings,
                on_log=lambda line: self._log(job_id, line),
                on_progress=lambda pct, label: self._progress(
                    job_id, "reframing", pct, label,
                    unit_label="clips",
                    units_done=len(clips), units_total=len(clips),
                    prior_eta_sec=eta.eta_from_prior(
                        float(job.get("durationSec") or 0.0), rtf
                    ),
                ),
                should_pause=lambda: self.pause_requested(job_id),
            )
        self._write_cuts(job_id, ws, cuts)

    def _write_cuts(
        self, job_id: str, ws: StageWorkspace, cuts: list[dict[str, Any]],
    ) -> None:
        # A cut without a clipId makes export fabricate "clip_000", which has no
        # clip document and used to abort the whole render. Mint the id HERE, at
        # the boundary where candidates become deliverable clips.
        for cut in cuts:
            if not cut.get("clipId"):
                cut["clipId"] = self.store.new_job_id()
        # Keep the dashboard's clip set equal to what was actually cut, so a
        # rejected candidate never leaves a permanent "Not rendered yet" card.
        self.store.sync_clip_docs(job_id, cuts)
        ws.write_json("reframe", "cuts.json", cuts)
        ws.mark_done("reframe", artifacts=["reframe/cuts.json"], count=len(cuts))
        logger.info(f"[job {job_id}] wrote {len(cuts)} clip cuts")

    # ------------------------------------------------------------------
    # stage 5: captions (translate + subtitle track)
    # ------------------------------------------------------------------

    def _stage_captions(
        self,
        job: dict[str, Any],
        transcript: subtitle.Transcript,
        ws: StageWorkspace,
    ) -> None:
        job_id = job["_id"]
        if ws.is_done("captions"):
            self._log(job_id, "resume: captions already done")
            self._progress(job_id, "captions", 100, "Caption tracks already written")
            return

        cuts = ws.read_json("reframe", "cuts.json") or []
        if not cuts:
            self._log(job_id, "no cuts to caption — skipping")
            return

        # Caption language (TARGET C): use the DETECTED language recorded at
        # transcription when the job left it on "auto"; a forced job language
        # ("es"]/"en") still wins. Never machine-translate — the words are
        # burned exactly as spoken.
        job_language = str(job.get("language") or "auto").strip().lower()
        if job_language and job_language != "auto":
            caption_language: str | None = job_language
        else:
            recorded = str(getattr(transcript, "language", "") or "").strip().lower()
            caption_language = (
                recorded or translate.guess_language(transcript) or "en"
            )
            self._log(job_id, f"caption language: {caption_language} (auto)")

        # THE resolved caption object for this batch: preset defaults + the
        # dashboard's sliders + the job's own preset/emoji choice. Reading
        # self.settings here was the bug — the stored overrides were never
        # applied to the render, so every slider was inert.
        cap_settings, resolved = self._caption_settings(job)
        style_key = str(resolved["presetKey"])
        # Emoji: ON/OFF and density are per batch (job), falling back to config.
        emoji_enabled = bool(resolved["emojiEnabled"])
        emoji_density = str(resolved["emojiDensity"] or "medium").strip().lower()
        self._log(
            job_id,
            f"caption style: {style_key} ({resolved['label']}) "
            f"size={resolved['fontSize']}px outline={resolved['outline']} "
            f"lines={resolved['lines']} pitch={resolved['pitch']}px "
            f"words={resolved['wordsPerCue']}/{resolved['wordsPerCueMax']} "
            f"pause={resolved['pauseSec']}s pos={resolved['positionMode']}@{resolved['positionY']} "
            f"emoji={emoji_density if emoji_enabled else 'off'} styleId={resolved['hash']}"
        )
        # ---- COVERAGE GUARD (LOCKED_SPEC §4) ------------------------------
        # Silero's view of where the speech is, computed ONCE for the source.
        # A 30 s window that decoded to zero words is what made a 52 s clip
        # stop captioning at 24.5 s, so no cue is built before the clip is
        # checked against this.
        speech_wav = ws.stage_dir("transcribe") / "chunks" / "speech16k.wav"
        regions: list[Any] = []
        if speech_wav.exists():
            try:
                regions = asr_accurate.speech_regions(
                    speech_wav, float(transcript.duration or 0.0)
                )
                self._log(
                    job_id,
                    f"coverage guard: {len(regions)} speech region(s), "
                    f"{asr_accurate.speech_vad.speech_seconds(regions):.1f}s voiced",
                )
            except Exception as exc:  # noqa: BLE001 - a check must not kill a render
                self._log(job_id, f"coverage guard unavailable: {exc}")
        else:
            self._log(job_id, "coverage guard unavailable: no speech16k.wav for this transcript")

        for i, cut in enumerate(cuts):
            clip_id = str(cut.get("clipId") or "")
            clip_start = float(cut.get("start") or 0.0)
            clip_end = float(cut.get("end") or 0.0)
            clip_span = (clip_start, clip_end)
            # Speech for THIS clip's judgement. It shrinks when the re-decode
            # ladder proves a voiced blip carries no words at all (a laugh, a
            # clap, music), so a clip is never flagged for “missing captions”
            # over audio no model can transcribe.
            clip_regions = regions
            if regions and clip_end > clip_start:
                # Speech with no words at all is re-transcribed on that region
                # alone, before the cue builder ever sees the transcript.
                gap = asr_accurate.coverage_report(
                    [(w.start, w.end) for w in transcript.words], regions, span=clip_span,
                )
                if gap.get("uncovered"):
                    fixed, repair = asr_accurate.cover_speech(
                        [w.as_dict() for w in transcript.words], regions,
                        wav=speech_wav, span=clip_span, language=caption_language,
                    )
                    if repair.get("repairedWords"):
                        # enforce_order only: the dedupe rule (same text within
                        # 2 s) would delete the real repeated shouts this
                        # recovered region is made of, and that re-opens the
                        # very holes it was called to close (measured: 97.3 %
                        # -> 83.7 % coverage on the real clip).
                        words_fixed = [
                            subtitle.Word(**w) for w in asr_accurate.enforce_order(fixed)
                        ]
                        transcript.words = words_fixed
                        # Persist the repaired transcript: a re-render must not
                        # pay for the same re-decodes again, and the chips in
                        # the dashboard must show the words that actually exist.
                        ws.write_json("transcribe", "words.json", {
                            "words": [w.as_dict() for w in transcript.words],
                            "duration": transcript.duration,
                            "model": transcript.model_size,
                            "engine": transcript.engine,
                            "rtf": transcript.rtf,
                            "language": getattr(transcript, "language", "") or "",
                        })
                        self._set_word_count(job_id, len(transcript.words))
                        self._log(
                            job_id,
                            f"coverage guard clip {i + 1}: re-transcribed "
                            f"{repair.get('repairedWords')} word(s) into "
                            f"{len(gap.get('uncovered') or [])} uncovered speech span(s) "
                            f"({', '.join(f'{a:.1f}-{b:.1f}s' for a, b in (gap.get('uncovered') or [])[:4])})",
                        )
                    non_speech = repair.get("nonSpeech") or []
                    if non_speech:
                        clip_regions = asr_accurate.without_spans(regions, non_speech)
                        self._log(
                            job_id,
                            f"coverage guard clip {i + 1}: "
                            + ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in non_speech[:4])
                            + " is voiced audio with no words in any model — "
                            "not counted as missing captions",
                        )
            cues = translate.build_cues(
                transcript, cut, cap_settings,
                language=caption_language,
                style=style_key,
                # An explicit ENG/ESP pick wins for the whole job. On "auto",
                # each clip may overrule the job-level guess with its own
                # words — that is the case where one clip of an English video
                # is a Spanish speaker.
                allow_clip_override=(job_language == "auto"),
            )
            # Persist the cue list per clip so the export stage can burn the
            # same cues into the frames without holding the transcript in
            # memory across the stage boundary (and so re-renders reuse them).
            ws.write_json(
                "captions", f"cues_{i:03d}.json",
                {"cues": cues, "clipId": cut.get("clipId")},
            )
            # Final judgement, on the CAPTION spans the viewer will actually
            # see: >=95 % of this clip's speech and the last cue within 1 s of
            # the last speech. A clip that fails is NEVER presented as ready.
            coverage: dict[str, Any] = {"passed": True, "needsReview": False}
            if clip_regions and clip_end > clip_start:
                coverage = asr_accurate.coverage_report(
                    [(float(c["start"]) + clip_start, float(c["end"]) + clip_start) for c in cues],
                    clip_regions,
                    word_spans=[(w.start, w.end) for w in transcript.words],
                    span=clip_span,
                )
                if clip_regions is not regions:
                    coverage["nonSpeech"] = [
                        [round(float(a), 2), round(float(b), 2)]
                        for a, b in (repair.get("nonSpeech") or [])
                    ]
            warning = None
            if coverage.get("needsReview"):
                warning = "needs review: " + (
                    coverage.get("reason") or "captions missing part of the speech"
                )
                self._log(job_id, f"clip {i + 1} {warning}")
            else:
                self._log(
                    job_id,
                    f"coverage clip {i + 1}: {coverage.get('coveredFraction', 1.0):.1%} of "
                    f"{coverage.get('speechSec', 0.0):.1f}s speech, last cue "
                    f"{coverage.get('lastCaptionSec')}s / last speech {coverage.get('lastSpeechSec')}s",
                )
            if clip_id:
                self.store.write_clip(
                    clip_id,
                    {
                        # The cues have to survive the workspace. They were only
                        # written to `captions/cues_NNN.json`, and that tree is
                        # swept (and its media offloaded) when the batch
                        # finishes — so the burned-in captions were fine but the
                        # clip's subtitle data and SRT export came back EMPTY.
                        # The clip document is durable, so it is the right home
                        # for the exact cues that were burned in.
                        "captionCues": cues,
                        "needsReview": bool(coverage.get("needsReview")),
                        "captionWarning": warning,
                        "captionCoverage": coverage,
                    },
                    job_id=job_id,
                )
            # Context-aware emoji, tagged from the whole clip's chunks in ONE
            # batched LLM call and then filtered by the selection rules. Never
            # fatal: a failure is recorded as "fallback: keywords" on the track
            # and the clip still renders.
            try:
                from app import emoji as _emoji

                track = _emoji.build_track(
                    cues, self.settings,
                    enabled=emoji_enabled,
                    density=emoji_density,
                    # `_stage_captions` never received the source path, so
                    # this read raised NameError on every clip that reached the
                    # emoji track — the whole render stage died there. Look it
                    # up from the store instead of threading a new argument
                    # through every caller.
                    src=self.store.find_source_file(job_id),
                    cut_start=float(cut.get("start") or 0.0),
                    duration=float(cut.get("end") or 0.0) - float(cut.get("start") or 0.0),
                    use_loudness=self.settings.subtitle.emoji_audio_cue,
                )
                ws.write_json("captions", f"emoji_{i:03d}.json", track)
                picked = len(track.get("selected") or [])
                self._log(
                    job_id,
                    f"emoji clip {i + 1}: {picked} of {len(cues)} chunks "
                    f"({track.get('source')}, density={emoji_density})",
                )
                if track.get("clipId") is None:
                    track["clipId"] = cut.get("clipId")
            except Exception as exc:  # noqa: BLE001 - decoration must not fail a job
                self._log(job_id, f"emoji tagging failed clip {i + 1}: {str(exc)[:200]}")
                ws.write_json(
                    "captions", f"emoji_{i:03d}.json",
                    {"source": "fallback: keywords", "tags": [], "selected": [],
                     "error": f"{type(exc).__name__}: {exc}"},
                )
            self._write_caption_tracks(job_id, i, cues)
            self._progress(
                job_id, "captions", (i + 1) / len(cuts),
                f"Captioning clip {i + 1} of {len(cuts)}",
                unit_label="clips",
                units_done=i + 1, units_total=len(cuts),
            )

    def _clip_title(
        self, job: dict[str, Any], cut: dict[str, Any], cues: list[dict[str, Any]]
    ) -> tuple[str, str]:
        """The white title bar's text for ONE clip — LOCKED_SPEC §2.

        Order: the detection pass's own LLM title for this clip (already
        written from its transcript in the video's language) → one LLM call
        with a single retry → the transcript's first sentence, labelled
        "fallback: first sentence" in the clip metadata. Placeholder text is
        never an answer.
        """
        existing = " ".join(str(cut.get("title") or cut.get("hookText") or "").split())
        texts = [str(c.get("text") or "") for c in cues]
        if existing:
            return existing[:90], "llm"
        try:
            from app import llm as llm_mod

            spoken = llm_mod.clip_hook_title(
                texts, self.settings, language=str(job.get("language") or "auto")
            )
        except Exception as exc:  # noqa: BLE001 - a title must never fail a clip
            self._log(job["_id"], f"hook title: LLM unavailable ({exc}) — using first sentence")
            spoken = None
        if spoken:
            return spoken, "llm"
        from app import titlebar as titlebar_mod

        fallback = titlebar_mod.first_sentence(texts)
        self._log(
            job["_id"],
            f"hook title: fallback: first sentence — {fallback!r}",
        )
        return fallback, "fallback: first sentence"

    # ------------------------------------------------------------------
    # stage 6: export (encode + package)
    # ------------------------------------------------------------------

    def _stage_export(
        self,
        job: dict[str, Any],
        src: Path,
        ws: StageWorkspace,
    ) -> None:
        job_id = job["_id"]
        if ws.is_done("export"):
            self._log(job_id, "resume: export already done")
            self._progress(job_id, "exporting", 100, "Clips already rendered")
            return

        cuts = ws.read_json("reframe", "cuts.json") or []
        if not cuts:
            self._log(job_id, "no cuts to export — skipping")
            return

        # Reconcile before rendering. On a resumed run cuts.json may predate
        # the clip documents (or carry ids from a previous detection pass), and
        # rendering against that stale set is how a "complete" batch ended up
        # with zero playable videos.
        missing_id = False
        for cut in cuts:
            if not cut.get("clipId"):
                cut["clipId"] = self.store.new_job_id()
                missing_id = True
        if missing_id:
            ws.write_json("reframe", "cuts.json", cuts)
        self.store.sync_clip_docs(job_id, cuts)

        dest = self.settings.app.paths.outputs_dir / job_id
        dest.mkdir(parents=True, exist_ok=True)

        total_sec = 0.0
        rendered = 0
        failed: list[tuple[int, str, str]] = []
        requested_count = int(job.get("requestedClipTarget") or job.get("clipTarget") or len(cuts))
        ready_count = 0
        used_ids: set[str] = set()
        gate_previous: list[dict[str, Any]] = []
        for prior in self.store.list_clip_docs(job_id):
            if prior.get("renderState") == "done":
                ready_count += 1
                used_ids.add(str(prior.get("_id")))
                sync = prior.get("audioSync") or {}
                if sync.get("fingerprint"):
                    gate_previous.append(
                        {"clipId": prior.get("_id"), "fingerprint": sync["fingerprint"]}
                    )
        # Resolved ONCE for the whole export loop, and fingerprinted: a clip
        # whose stored styleId still matches may be reused, anything else must
        # re-encode. Without the id, "Re-render captions" skipped every clip
        # whose file already existed and served the stale render.
        cap_settings, resolved = self._caption_settings(job)
        style_id = str(resolved["hash"])

        for i, cut in enumerate(cuts):
            if ready_count >= requested_count:
                self._log(
                    job_id,
                    f"requested {requested_count} clips are ready — "
                    f"not consuming the remaining replacement candidates",
                )
                break
            self._progress(
                job_id, "exporting",
                (i + 1) / len(cuts),
                f"Rendering clip {i + 1} of {len(cuts)}",
                unit_label="clips",
                units_done=i + 1, units_total=len(cuts),
            )
            clip_id = cut.get("clipId") or f"clip_{i:03d}"
            used_ids.add(str(clip_id))
            # Per-clip resume: a clip that is already rendered AND recorded as
            # done must not pay for a second ffmpeg pass on every retry — the
            # bounded-runner path re-enters this loop from clip 0 each time.
            existing_doc = self.store.get_clip_doc(clip_id)
            existing_file = dest / f"clip_{i:03d}.mp4"
            # A clip that lives in the bucket counts as rendered: the local file
            # is deleted after a verified push, so requiring it here would
            # re-render (and re-upload) every clip on every resume.
            file_ok = existing_file.exists() and existing_file.stat().st_size > 0
            remote_ok = bool(existing_doc and existing_doc.get("mediaRemote"))
            if (
                existing_doc is not None
                and existing_doc.get("renderState") == "done"
                and existing_doc.get("styleId") == style_id
                and (file_ok or remote_ok)
            ):
                self._log(job_id, f"resume: clip {i + 1} already rendered at this style, skipping")
                continue
            if (
                existing_doc is not None
                and existing_doc.get("renderState") == "done"
                and existing_doc.get("styleId") != style_id
            ):
                self._log(
                    job_id,
                    f"style changed ({existing_doc.get('styleId')} -> {style_id}) "
                    f"— re-rendering clip {i + 1}",
                )
            self.store.write_clip(
                clip_id,
                {"renderState": "running", "jobId": job_id},
                job_id=job_id,
            )
            gate_result: dict[str, Any] | None = None
            meta: dict[str, Any] | None = None
            timings: dict[str, float] = {}
            with self._heartbeat(job_id, f"render clip {i + 1}"):
                try:
                    # The context-aware emoji chosen in the captions stage (an
                    # LLM judgement, filtered by the selection rules). Missing
                    # track = no emoji layer, never a guess at render time.
                    emoji_track = ws.read_json("captions", f"emoji_{i:03d}.json") or {}
                    clip_cues = (
                        ws.read_json("captions", f"cues_{i:03d}.json") or {}
                    ).get("cues") or []
                    # The workspace is swept when a batch finishes, so a resume,
                    # re-render or retry used to find no cue/emoji
                    # sidecar and render the clip WITHOUT its captions — same
                    # clip, different look. The durable clip document now holds
                    # exactly what was burned in; fall back to it instead of
                    # silently dropping the caption layer.
                    if not clip_cues or not emoji_track:
                        durable = existing_doc or (self.store.get_clip_doc(clip_id) or {})
                        if not clip_cues:
                            clip_cues = durable.get("captionCues") or []
                        if not emoji_track:
                            selected = durable.get("emojiTrack") or []
                            if selected:
                                emoji_track = {
                                    "selected": selected,
                                    "source": durable.get("emojiSource"),
                                    "tags": durable.get("emojiTags") or [],
                                }
                    # LOCKED_SPEC §2: the white bar's text, per clip, from the
                    # clip's own content, in the video's language.
                    clip_title, title_source = self._clip_title(job, cut, clip_cues)
                    # Presigned URLs are minted PER CLIP and expire in 15
                    # minutes: one long-lived URL would start returning 403 in
                    # the middle of a batch. The clip is seeked by byte range,
                    # so ffmpeg fetches the part it renders, not the source.
                    render_src: Any = src
                    render_audio: str | None = None
                    audio_plan: dict[str, Any] | None = None
                    seek_offset: float | None = None
                    probe_meta: dict[str, Any] | None = None
                    if isinstance(src, source_stream.RemoteSource):
                        render_src, seek_offset = src.url_for_span(
                            float(cut.get("start") or 0.0),
                            float(cut.get("end") or 0.0),
                        )
                        render_audio = src.audio_url()
                        audio_plan = src.audio_plan(
                            float(cut.get("start") or 0.0),
                            float(cut.get("end") or 0.0),
                            order=i,
                        )
                        probe_meta = src.probe
                        self._log(
                            job_id,
                            f"clip {i + 1}: cutting from the bucket at "
                            f"{float(cut.get('start') or 0.0):.1f}s "
                            f"(seek {seek_offset:.1f}s into "
                            f"{'section' if src.sections else 'the source'})",
                        )
                    if self._ledger is not None:
                        self._ledger.check(f"export clip {i + 1}")
                    _render_t0 = time.perf_counter()
                    meta = encode.render_clip(
                        render_src, cut, dest / f"clip_{i:03d}.mp4", cap_settings,
                        cues=clip_cues,
                        preset_key=str(resolved["presetKey"]),
                        emoji_selection=emoji_track.get("selected") or [],
                        title=clip_title,
                        # The FIT/FILL choice per shot is carried from the
                        # reframe stage on each cut and logged here.
                        on_log=lambda line: self._log(job_id, line),
                        audio_src=render_audio,
                        probe_meta=probe_meta,
                        start_offset=seek_offset,
                        audio_seek=float(cut.get("start") or 0.0),
                        audio_start_time=float(
                            getattr(src, "audio_start_time", 0.0) or 0.0
                        ),
                        audio_plan=audio_plan,
                        audio_source_key=getattr(src, "audio_key", None),
                    )
                    timings["renderMs"] = round((time.perf_counter() - _render_t0) * 1000.0, 1)
                    # The independent gate, on the FINAL file and the REAL
                    # source. It never trusts the renderer's own plan.
                    _gate_t0 = time.perf_counter()
                    from app import audiosync as _audiosync

                    gate_cache = dest / ".audiosync"
                    gate_cache.mkdir(parents=True, exist_ok=True)
                    gate_source = render_audio or render_src
                    gate_result = _audiosync.gate_clip(
                        dest / f"clip_{i:03d}.mp4",
                        gate_source,
                        plan_start=float(cut.get("start") or 0.0),
                        plan_end=float(cut.get("end") or 0.0),
                        cues=clip_cues,
                        settings=cap_settings,
                        cache_dir=gate_cache / f"clip_{i:03d}",
                        previous=gate_previous,
                        source_start_time=float(getattr(src, "audio_start_time", 0.0) or 0.0),
                    )
                    timings["gateMs"] = round((time.perf_counter() - _gate_t0) * 1000.0, 1)
                    if not gate_result.get("passed") and not (gate_result.get("placement") or {}).get("ok"):
                        # The brief's one automatic repair: a placement failure
                        # is re-rendered once with the safer render ladder.
                        self._log(
                            job_id,
                            f"clip {i + 1}: audio sync gate failed ({gate_result.get('reason')}) "
                            "— re-rendering once with safe settings",
                        )
                        _safe_t0 = time.perf_counter()
                        safe_meta = encode.render_clip(
                            render_src, cut, dest / f"clip_{i:03d}.mp4", cap_settings,
                            cues=clip_cues,
                            preset_key=str(resolved["presetKey"]),
                            emoji_selection=emoji_track.get("selected") or [],
                            title=clip_title,
                            on_log=lambda line: self._log(job_id, line),
                            audio_src=render_audio,
                            probe_meta=probe_meta,
                            start_offset=seek_offset,
                            audio_seek=float(cut.get("start") or 0.0),
                            audio_start_time=float(
                                getattr(src, "audio_start_time", 0.0) or 0.0
                            ),
                            audio_plan=audio_plan,
                            audio_source_key=getattr(src, "audio_key", None),
                            safe_only=True,
                        )
                        safe_meta["attempts"] = (safe_meta.get("attempts") or [])
                        safe_meta["attempts"].insert(0, {
                            "attempt": "sync-safe",
                            "ok": True,
                            "reason": str(gate_result.get("reason") or ""),
                        })
                        timings["safeRenderMs"] = round((time.perf_counter() - _safe_t0) * 1000.0, 1)
                        meta = safe_meta
                        _gate_t0 = time.perf_counter()
                        gate_result = _audiosync.gate_clip(
                            dest / f"clip_{i:03d}.mp4",
                            gate_source,
                            plan_start=float(cut.get("start") or 0.0),
                            plan_end=float(cut.get("end") or 0.0),
                            cues=clip_cues,
                            settings=cap_settings,
                            cache_dir=gate_cache / f"clip_{i:03d}",
                            previous=gate_previous,
                            source_start_time=float(
                                getattr(src, "audio_start_time", 0.0) or 0.0
                            ),
                        )
                    if "safeRenderMs" in timings:
                        timings["safeGateMs"] = round(
                            (time.perf_counter() - _gate_t0) * 1000.0, 1
                        )
                    if not gate_result.get("passed"):
                        sync_error = RuntimeError(
                            "audio_mismatch after one safe re-render: "
                            + str(gate_result.get("reason") or "sync gate failed")
                        )
                        setattr(sync_error, "attempts", meta.get("attempts"))
                        raise sync_error
                    meta["audioSync"] = gate_result
                except Exception as exc:
                    detail = str(exc)[:500]
                    ladder = getattr(exc, "attempts", None) or (
                        meta.get("attempts") if isinstance(meta, dict) else None
                    )
                    if ladder:
                        bits = []
                        for step in ladder:
                            if not isinstance(step, dict):
                                continue
                            if step.get("ok"):
                                bits.append(f"{step.get('attempt')} ok")
                            else:
                                bits.append(
                                    f"{step.get('attempt')}: "
                                    f"{str(step.get('reason') or 'failed')[:120]}"
                                )
                        if bits:
                            detail = (detail + " | retry ladder: " + "; ".join(bits))[:700]
                    self._log(job_id, f"render failed clip {i + 1}: {detail}")
                    # Keep WHERE it failed: "failed to map segment from shared
                    # object" names a library, not the code that asked for it.
                    self._log(
                        job_id,
                        "render traceback: " + traceback.format_exc().strip()[-900:],
                    )
                    failed.append((i, clip_id, detail))
                    try:
                        # Keep the encoder's own stderr on the CLIP, not just in
                        # the job log. Without it a per-clip failure is
                        # undiagnosable after the log rotates, and the retry
                        # button has nothing to tell the user.
                        stderr = ""
                        try:
                            got = getattr(exc, "stderr", None) or getattr(exc, "output", None)
                            if isinstance(got, bytes):
                                got = got.decode("utf-8", "replace")
                            stderr = str(got or "")[-1500:]
                        except Exception:
                            stderr = ""
                        render_attempts = ladder
                        self.store.update_clip(
                            clip_id,
                            lambda c: c.update({
                                "renderState": "failed",
                                "mediaError": detail,
                                "ffmpegStderr": stderr or c.get("ffmpegStderr"),
                                "renderAttempts": render_attempts,
                                "audioSync": gate_result,
                                "failedAt": int(time.time() * 1000),
                                "jobId": job_id,
                            }),
                        )
                    except Exception as exc:  # noqa: BLE001 - the batch continues
                        logger.warning(f"[job {job_id}] could not record clip failure: {exc}")
                    continue

            thumb_path = dest / f"clip_{i:03d}.thumb.jpg"
            try:
                # Source is the RENDERED CLIP FILE, not the output directory
                # (passing dest made ffmpeg read a directory and exit 1).
                video.render_first_frame_thumb(dest / f"clip_{i:03d}.mp4", thumb_path)
                self.store.write_clip(
                    clip_id,
                    {
                        "videoPath": meta["videoPath"],
                        "thumbPath": str(thumb_path),
                        "mediaBytes": meta["bytes"],
                        "mediaWidth": meta["width"],
                        "mediaHeight": meta["height"],
                        "mediaDurationSec": meta["durationSec"],
                        "fps": meta.get("fps"),
                        "mediaCodec": meta.get("codec"),
                        "mediaPixFmt": meta.get("pixFmt"),
                        "mediaAudioCodec": meta.get("audioCodec"),
                        "mediaFaststart": meta.get("faststart"),
                        "contractOk": meta.get("contractOk"),
                        "contractChecks": meta.get("contractChecks"),
                        "renderState": "done",
                        # What the clip actually got, so the dashboard can show
                        # each emoji with its reason, and say whether a model
                        # chose it or the keyword fallback did.
                        "emojiTrack": emoji_track.get("selected") or [],
                        "emojiSource": emoji_track.get("source"),
                        "emojiTags": emoji_track.get("tags") or [],
                        "title": clip_title,
                        "titleSource": title_source,
                        "renderMs": meta.get("renderMs"),
                        "encoder": meta.get("encoder"),
                        "encoderLabel": meta.get("encoderLabel"),
                        "hardwareEncoded": meta.get("hardwareEncoded"),
                        "preset": meta.get("preset"),
                        # Which resolved style this file wears, so a changed
                        # style is never mistaken for an up-to-date render.
                        "styleId": style_id,
                        "captionSettings": resolved,
                        "ffmpegCommand": meta.get("ffmpegCommand"),
                        "ffmpegStderr": meta.get("ffmpegStderr"),
                        "renderAttempts": meta.get("attempts"),
                        "audioSync": meta.get("audioSync"),
                        "audioPlan": meta.get("audioPlan"),
                        "timings": timings,
                        "jobId": job_id,
                    },
                    job_id=job_id,
                )
            except Exception as thumb_exc:
                self._log(job_id, f"thumb failed for clip {i + 1}: {thumb_exc}")
                self.store.write_clip(
                    clip_id,
                    {
                        "videoPath": meta["videoPath"],
                        "mediaBytes": meta["bytes"],
                        "mediaWidth": meta["width"],
                        "mediaHeight": meta["height"],
                        "mediaDurationSec": meta["durationSec"],
                        "fps": meta.get("fps"),
                        "mediaCodec": meta.get("codec"),
                        "mediaPixFmt": meta.get("pixFmt"),
                        "mediaAudioCodec": meta.get("audioCodec"),
                        "mediaFaststart": meta.get("faststart"),
                        "contractOk": meta.get("contractOk"),
                        "contractChecks": meta.get("contractChecks"),
                        # A ready clip must have a poster. The old fallback wrote
                        # `done` here, so the grid showed a broken image on a
                        # clip the contract said was deliverable.
                        "renderState": "failed",
                        "mediaError": f"thumbnail check failed: {thumb_exc}"[:500],
                        "title": clip_title,
                        "titleSource": title_source,
                        "renderMs": meta.get("renderMs"),
                        "encoder": meta.get("encoder"),
                        "encoderLabel": meta.get("encoderLabel"),
                        "hardwareEncoded": meta.get("hardwareEncoded"),
                        "preset": meta.get("preset"),
                        "styleId": style_id,
                        "captionSettings": resolved,
                        "ffmpegCommand": meta.get("ffmpegCommand"),
                        "ffmpegStderr": meta.get("ffmpegStderr"),
                        "renderAttempts": meta.get("attempts"),
                        "audioSync": meta.get("audioSync"),
                        "audioPlan": meta.get("audioPlan"),
                        "timings": timings,
                        "jobId": job_id,
                    },
                    job_id=job_id,
                )
            # ---- the bucket is the store; the disk is only a workbench ------
            # Push THIS clip (and its thumb), HEAD-verify both, and release the
            # local bytes immediately. Nothing is deleted before the bucket has
            # confirmed the size, and a failure leaves the file alone and marks
            # the clip "upload pending" so the completion sweep retries it.
            _upload_t0 = time.perf_counter()
            self._store_clip_media(
                job_id, clip_id, dest / f"clip_{i:03d}.mp4", thumb_path, meta,
            )
            timings["uploadMs"] = round((time.perf_counter() - _upload_t0) * 1000.0, 1)
            self.store.update_clip(
                clip_id,
                lambda c: c.update({"timings": timings}),
                job_id=job_id,
            )
            # The title plate PNG and the encoder's temp file belong to THIS
            # clip only; leaving them would accumulate one small file per clip.
            for leftover in (
                dest / f"clip_{i:03d}.title.png",
                dest / f"clip_{i:03d}.part.mp4",
            ):
                leftover.unlink(missing_ok=True)
            total_sec += meta["durationSec"]
            rendered += 1
            ready_count += 1
            sync = meta.get("audioSync") or {}
            if sync.get("fingerprint"):
                gate_previous.append({"clipId": clip_id, "fingerprint": sync["fingerprint"]})

        # Drop replacement candidates that were never consumed. A finished
        # batch must show exactly what it delivered (plus any attempted
        # failures), not 14 cards for a 10-clip request.
        final_cuts = [c for c in cuts if str(c.get("clipId")) in used_ids]
        self.store.sync_clip_docs(job_id, final_cuts)
        ws.mark_done("export", artifacts=[], count=len(final_cuts))
        unmet = ""
        if failed:
            reasons = "; ".join(f"clip {i+1}: {r}" for i, _, r in failed)
            self._log(job_id, f"{len(failed)} clip(s) failed to render: {reasons}")
        if ready_count < requested_count:
            unmet = (
                f"only {len(cuts)} candidate moment(s) were available for "
                f"{requested_count} requested; {ready_count} became ready and "
                f"{len(failed)} failed: {reasons if failed else 'none'}"
            )[:700]
            self._log(job_id, unmet)
        self._patch_job(
            job_id,
            clipTarget=requested_count,
            requestedClipTarget=requested_count,
            clipCount=ready_count,
            requestedClips=requested_count,
            readyClips=ready_count,
            failedClips=len(failed),
            unmetReason=unmet or None,
            totalClipsSec=round(total_sec, 3),
            renderFailures=len(failed),
        )

    def _store_clip_media(
        self,
        job_id: str,
        clip_id: str,
        clip_path: Path,
        thumb_path: Path | None,
        meta: dict[str, Any],
    ) -> None:
        """Push one finished clip to the bucket, verify it, then free the disk.

        Retried with backoff (2s, 4s, 8s) because the failure mode that matters
        is a flaky link, not a wrong bucket — and a clip that cannot be stored
        must stay on disk rather than disappear. `mediaRemote` is only ever set
        after a HEAD reports the exact byte count.
        """
        from app import objectstore, reclaim

        cfg = objectstore.config_from_env()
        if not cfg.enabled:
            return
        duration = float(meta.get("durationSec") or 0.0)
        # 1080x1920 is capped at ~3.7 Mbps by the encoder's own ceiling, so this
        # is what the file SHOULD weigh; the dashboard shows it next to the real
        # `mediaBytes` and the difference is the honesty check.
        estimated = int(duration * 3_732_000 / 8) if duration else None
        delay = 2.0
        last_error = ""
        for attempt in range(3):
            try:
                res = reclaim.push_clip_and_thumb(job_id, clip_path, thumb_path)
            except Exception as exc:  # noqa: BLE001 - a push must not fail the clip
                res = {"ok": False, "video": {"error": str(exc)[:200]}}
            if res.get("ok"):
                bytes_sent = int((res.get("video") or {}).get("bytes") or 0)
                self.store.write_clip(
                    clip_id,
                    {
                        "mediaRemote": True,
                        "uploadPending": False,
                        "uploadError": None,
                        "healthState": "ok",
                        "healthIssue": None,
                        "videoKey": (res.get("video") or {}).get("key"),
                        "thumbKey": (res.get("thumb") or {}).get("key"),
                        "localDropped": bool(res.get("dropped")),
                        "estimatedBytes": estimated,
                    },
                    job_id=job_id,
                )
                self._log(
                    job_id,
                    f"clip {int(meta.get('index') or 0) if meta.get('index') else clip_id[:6]}: "
                    f"{bytes_sent / 1048576:.1f} MB stored and HEAD-verified "
                    f"(est {(estimated or 0) / 1048576:.1f} MB), local copy released",
                )
                return
            last_error = str(
                res.get("error") or (res.get("video") or {}).get("error") or "upload failed"
            )[:200]
            if attempt < 2:
                self._log(job_id, f"upload attempt {attempt + 1} failed ({last_error}) — retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
        self.store.write_clip(
            clip_id,
            {
                "mediaRemote": False,
                "uploadPending": True,
                "uploadError": last_error,
                "localDropped": False,
                "estimatedBytes": estimated,
                # The local file is playable, but the deliverable is not on the
                # bucket yet. It must not wear a green Ready badge: the Retry
                # affordance and the health text come from this flag.
                "healthState": "failed",
                "healthIssue": f"upload pending: {last_error}",
            },
            job_id=job_id,
        )
        self._log(
            job_id,
            f"clip kept on local disk — upload pending ({last_error}); "
            "the batch completion sweep retries it, nothing is lost",
        )

    # ------------------------------------------------------------------
    # finish
    # ------------------------------------------------------------------

    def _finish(self, job_id: str, elapsed_sec: float) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        now_ms = int(time.time() * 1000)
        # A terminal verdict must be written ONCE, here: the document used to be
        # saved as stage=complete with isTerminal=False and finishedAt=None, so
        # anything reading the raw job (not the API's normaliser) saw a batch
        # that never stopped running.
        counts = {"done": 0, "failed": 0, "pending": 0, "total": 0}
        clips: list[dict[str, Any]] = list(job.get("clips") or [])
        try:
            from app import invariants

            clips = sorted(
                self.store.list_clip_docs(job_id),
                key=lambda c: (int(c.get("order") or 0), str(c.get("_id"))),
            )
            for c in clips:
                state = str(c.get("renderState") or "pending")
                key = "done" if state == "done" else ("failed" if state == "failed" else "pending")
                counts[key] += 1
            counts["total"] = len(clips)
            detail = (
                invariants.partial_success_label(counts)
                if counts["failed"]
                else "All clips ready"
            )
        except Exception as exc:  # noqa: BLE001 - completion must still complete
            logger.warning(f"[job {job_id}] clip tally skipped: {exc}")
            detail = "All clips ready"
        # Every fresh deliverable gets a public page token at completion, so the
        # card can share immediately instead of waiting for a Copy tap. The
        # public route still enforces expiry/revocation on every request.
        try:
            from app import sharelink as _sharelink

            for clip in clips:
                if (
                    str(clip.get("renderState") or "") == "done"
                    and clip.get("videoPath")
                    and _sharelink.token_state(clip)["state"] != "ready"
                ):
                    _sharelink.ensure_id(clip)
                    _sharelink.ensure_token(clip)
                    self.store.write_clip(clip)
        except Exception as exc:  # noqa: BLE001 - sharing must not block completion
            logger.warning(f"[job {job_id}] share token mint skipped: {exc}")
        job.update({
            "stage": "complete",
            "status": "complete",
            "progress": 100.0,
            "progressDetail": detail,
            "completedAt": now_ms,
            "finishedAt": now_ms,
            "isTerminal": True,
            "clipCount": counts["done"],
            "requestedClips": int(job.get("requestedClips") or job.get("requestedClipTarget") or counts["done"]),
            "readyClips": counts["done"],
            "failedClips": counts["failed"],
            "renderFailures": counts["failed"],
            # The projection lives in the same document as the counts, so a
            # reader never sees "1 clip ready" with an empty clips list.
            "clips": clips,
            "lastProgressAt": now_ms,
            "stageStatus": self._stage_status,
            # A completion must not inherit the flag left by an interrupted
            # attempt: boot treats `interrupted` as "resume me" and would put a
            # finished batch back in the queue on the next restart.
            "interrupted": False,
            "interruptReason": None,
            "error": None,
            "errorDetail": None,
        })
        self.store.write_job(job)
        self.store.append_job_log(job_id, f"COMPLETE in {elapsed_sec:.1f}s")
        logger.info(f"[job {job_id}] COMPLETE in {elapsed_sec:.1f}s")
        # The batch is done, so its media is dead weight: push it to the bucket
        # and drop the local copies now. Deferring this to the watchdog is what
        # left a finished 445 MB source sitting on a 4 GB disk until the next
        # batch ran out of room, and it is the reason "the bucket" looked
        # connected while every video still lived on the disk.
        try:
            from app import reclaim

            reclaim.offload_job(self.store, self.settings.app.paths, job_id)
        except Exception as exc:  # noqa: BLE001 - completion must still succeed
            logger.warning(f"[job {job_id}] post-completion offload skipped: {exc}")


# ------------------------------------------------------------------
# chunk cache helpers
# ------------------------------------------------------------------



class _StageChunkCache(ChunkCache):
    """Per-chunk persistence for LLM / ASR stages that expose a cache dir.

    Implements llm.ChunkCache's duck type (`get`/`put`), which detect_clips
    calls — previously only the stage-local key()/save()/load() helpers
    existed, so detection crashed with AttributeError on its very first
    uncached chunk.
    """

    def __init__(self, ws: StageWorkspace, *, model: str) -> None:
        self._ws = ws
        self._model = model

    def dir(self, stage: str) -> Path:
        return self._ws.stage_dir(stage) / "chunks"

    def key(self, stage: str, index: int) -> str:
        return f"{stage}_chunk_{index}.json"

    def get(self, index: int) -> list[dict[str, Any]] | None:
        """Cached candidates for chunk `index`, or None when absent.

        A malformed cache row degrades to a miss rather than failing the
        stage: re-billing one chunk is cheaper than failing a run whose data
        was on disk.
        """
        p = self.dir("detect") / self.key("detect", index)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def put(self, index: int, candidates: list[dict[str, Any]]) -> None:
        d = self.dir("detect")
        d.mkdir(parents=True, exist_ok=True)
        (d / self.key("detect", index)).write_text(json.dumps(candidates))

    def loaded(self, stage: str, index: int) -> bool:
        return (self.dir(stage) / self.key(stage, index)).exists()

    def save(self, stage: str, index: int, payload: Any) -> None:
        d = self.dir(stage)
        d.mkdir(parents=True, exist_ok=True)
        (d / self.key(stage, index)).write_text(json.dumps(payload))

    def load(self, stage: str, index: int) -> Any:
        p = self.dir(stage) / self.key(stage, index)
        if not p.exists():
            raise FileNotFoundError(p)
        return json.loads(p.read_text())



