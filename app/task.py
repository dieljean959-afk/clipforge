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
    clipcount,
    download,
    encode,
    reframe,
    eta,
    llm,
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
        if free < 200 * 1024 * 1024:
            raise video.RenderError(
                f"not enough free disk ({lim}) — less than 200 MB free on "
                f"{path}. The download and transcript are still cached; Retry "
                f"will resume from the last completed stage once space is cleared."
            )
        if source_duration_sec > 0:
            est = source_duration_sec * 0.05
            if est and free < est + 500 * 1024 * 1024:
                logger.warning(
                    f"[job {job_id}] low disk headroom: {free // (1024*1024)} MB free, "
                    f"~{est:.0f} MB estimated render footprint for a "
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
            "failedStage": stage,
            "error": message,
            "errorDetail": detail,
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
        job = self.store.get_job(job_id)
        if job is None:
            return
        job["stageStatus"] = dict(self._stage_status)
        job["lastProgressAt"] = now
        self.store.write_job(job)

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
        job = self.store.get_job(job_id)
        if job is None:
            return
        now_ms = int(time.time() * 1000)
        fraction = max(0.0, min(1.0, pct))
        prev_stage = job.get("stage")
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

        job.update({
            "stage": stage,
            "progress": max(0.0, min(100.0, pct * 100)),
            "progressDetail": detail,
            "unitLabel": unit_label,
            "unitsDone": units_done,
            "unitsTotal": units_total,
            "stageEtaSec": None if stage_eta is None else round(stage_eta, 1),
            "totalEtaSec": total_eta,
            "etaSource": eta_source,
            "stagePlan": plan,
            "lastProgressAt": now_ms,
        })
        self.store.write_job(job)

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
        job = self.store.get_job(job_id)
        if job is None:
            return {}
        job.update(fields)
        self.store.write_job(job)
        return job

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
                result = fn()
            except BaseException:
                self._mark_stage_failed(job_id, stage, *self._fail_args_for(sys.exc_info()))
                raise
            self._mark_stage_done(job_id, stage)
            return result

        transcript = _run_stage("transcribe", lambda: self._stage_transcribe(job, src, ws))
        self._transcript_size_note(len(transcript.words))

        self._checkpoint(job_id, "transcribing")
        candidates = _run_stage("detect", lambda: self._stage_detect(job, transcript, ws))

        self._checkpoint(job_id, "finding")
        _run_stage("reframe", lambda: self._stage_render(job, src, transcript, candidates, ws))
        self._render_workload_note(len(candidates), float(job.get("durationSec") or 0.0))

        _run_stage("captions", lambda: self._stage_captions(job, transcript, ws))
        _run_stage("export", lambda: self._stage_export(job, src, ws))

        self._finish(job_id, time.time() - t_start)
        self._mark_stage_done(job_id, "export")
        # Scratch audio is dead weight once the clips exist.
        self._cleanup_job_temps(job_id, StageWorkspace(self.settings.app.paths.data_dir, job_id))

    @staticmethod
    def _detail(exc: BaseException) -> str:
        """Full traceback for the job log / errorDetail (structured errors)."""
        return "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-4000:]

    def _fail_args_for(self, exc_info) -> tuple[str, str]:
        """Build (message, detail) for _mark_stage_failed from an exception."""
        exc_type, exc_value, exc_tb = exc_info
        msg = f"{exc_type.__name__}: {exc_value}"
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))[-4000:]
        return msg, detail

    # ------------------------------------------------------------------
    # stage 1: fetch
    # ------------------------------------------------------------------

    def _stage_fetch(self, job: dict[str, Any], ws: StageWorkspace) -> Path:
        job_id = job["_id"]
        if ws.is_done("fetch"):
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
                    "This batch has no video URL recorded. The job document is missing "
                    "the `source` field the pipeline needs to fetch the video. Re-create "
                    "the batch from the dashboard — 'Paste YouTube/TikTok URL' or 'Attach "
                    "file' — rather than re-running an incomplete job."
                )
            if not isinstance(url, str) or not url.strip():
                raise download.DownloadError(
                    "The job's `source` is not a usable URL. It was stored as "
                    f"{type(url).__name__} and cannot be fetched. Re-create the batch "
                    "from the dashboard with a real YouTube/TikTok URL, or use 'Attach "
                    "file' for a local upload."
                )
            url = url.strip()
            def on_progress(frac: float, _eta: float | None) -> None:
                self._progress(job_id, "grabbing", frac, "Clippy is grabbing the video...")

            fetch_log_out: list[str] = []
            def _on_dl_log(line: str) -> None:
                fetch_log_out.append(line)
                self._log(job_id, line)

            self._on_stage_log[job_id] = _on_dl_log

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
                if on_log:
                    on_log(
                        "DOWNLOAD FAILED — yt-dlp debug:\n"
                        + ydl_debug
                        + "\n"
                        + cfg_snapshot
                    )
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
        self._on_stage_log.pop(job_id, None)
        return src

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

        def on_progress(frac: float) -> None:
            # Sub-chunk smoothing only. It must never move the bar BACKWARDS
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
        with self._heartbeat(job_id, "transcribe"):
            transcript = subtitle.transcribe(
                src, self.settings, on_progress=on_progress,
                cache_dir=chunk_cache_dir,
                should_pause=lambda: self.pause_requested(job_id),
                on_chunk=on_chunk,
                language=transcribe_language,
            )
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
                patch: dict[str, Any] = {
                    "clipTarget": len(cached),
                    "clipsAboveScore": sum(1 for s in scores if s >= floor),
                    "clipsBelowScore": sum(1 for s in scores if s < floor),
                    "clipScoreFloor": floor,
                }
                if job.get("candidatesDetected") is None:
                    patch["candidatesDetected"] = len(cached)
                self._patch_job(job_id, **patch)
                return cached

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
        self._log(job_id, f"Clip target: {target} — {reason}")
        self._patch_job(job_id, clipTarget=target, clipTargetReason=reason)

        # stage 2: transcribe stashes its own callback into the shared
        # _on_stage_log dict so on_log in the download stage resolves safely.
        def _on_transcribe_log(line: str) -> None:
            self._log(job_id, line)

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
            max_clips=target,
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
        ws.write_json("detect", "candidates.json", candidates)
        # Persist one stage artifact per candidate so a retry can tell which clips
        # were already rendered and which still need the export stage.
        ws.write_clips("detect", candidates)
        # Record HOW these clips were chosen so the dashboard can say "AI highlights
        # via deepseek" instead of a blank "detectedSource". The provider label is
        # already used a few lines above for the chunk cache and the log line, so this
        # keeps all three named the same by construction rather than by coincidence.
        summary = self.store.create_clips(
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

        clips = clipcount.reasonable_clips(
            candidates,
            max_clips=job.get("clipTarget") or self.settings.gemini.max_clips,
            fallback=self.settings.gemini.max_clips,
        )
        job = self._patch_job(job_id, clipTarget=len(clips))

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
        for i, cut in enumerate(cuts):
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
                    src=src,
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

        src_dur = float(job.get("durationSec") or 0.0)
        total_sec = 0.0
        rendered = 0
        failed: list[tuple[int, str, str]] = []
        # Resolved ONCE for the whole export loop, and fingerprinted: a clip
        # whose stored styleId still matches may be reused, anything else must
        # re-encode. Without the id, "Re-render captions" skipped every clip
        # whose file already existed and served the stale render.
        cap_settings, resolved = self._caption_settings(job)
        style_id = str(resolved["hash"])

        for i, cut in enumerate(cuts):
            self._progress(
                job_id, "exporting",
                (i + 1) / len(cuts),
                f"Rendering clip {i + 1} of {len(cuts)}",
                unit_label="clips",
                units_done=i + 1, units_total=len(cuts),
            )
            clip_id = cut.get("clipId") or f"clip_{i:03d}"
            # Per-clip resume: a clip that is already rendered AND recorded as
            # done must not pay for a second ffmpeg pass on every retry — the
            # bounded-runner path re-enters this loop from clip 0 each time.
            existing_doc = self.store.get_clip_doc(clip_id)
            existing_file = dest / f"clip_{i:03d}.mp4"
            if (
                existing_doc is not None
                and existing_doc.get("renderState") == "done"
                and existing_doc.get("styleId") == style_id
                and existing_file.exists()
                and existing_file.stat().st_size > 0
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
            with self._heartbeat(job_id, f"render clip {i + 1}"):
                try:
                    # The context-aware emoji chosen in the captions stage (an
                    # LLM judgement, filtered by the selection rules). Missing
                    # track = no emoji layer, never a guess at render time.
                    emoji_track = ws.read_json("captions", f"emoji_{i:03d}.json") or {}
                    meta = encode.render_clip(
                        src, cut, dest / f"clip_{i:03d}.mp4", cap_settings,
                        cues=(ws.read_json("captions", f"cues_{i:03d}.json") or {}).get("cues") or [],
                        preset_key=str(resolved["presetKey"]),
                        emoji_selection=emoji_track.get("selected") or [],
                        # The FIT/FILL choice per shot is carried from the
                        # reframe stage on each cut and logged here.
                        on_log=lambda line: self._log(job_id, line),
                    )
                except Exception as exc:
                    detail = str(exc)[:500]
                    self._log(job_id, f"render failed clip {i + 1}: {detail}")
                    failed.append((i, clip_id, detail))
                    try:
                        self.store.write_clip(
                            clip_id,
                            {
                                "renderState": "failed",
                                "mediaError": detail,
                                "jobId": job_id,
                            },
                            job_id=job_id,
                        )
                    except Exception:
                        pass
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
                        "renderState": "done",
                        # What the clip actually got, so the dashboard can show
                        # each emoji with its reason, and say whether a model
                        # chose it or the keyword fallback did.
                        "emojiTrack": emoji_track.get("selected") or [],
                        "emojiSource": emoji_track.get("source"),
                        "emojiTags": emoji_track.get("tags") or [],
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
                        "renderState": "done",
                        "renderMs": meta.get("renderMs"),
                        "encoder": meta.get("encoder"),
                        "encoderLabel": meta.get("encoderLabel"),
                        "hardwareEncoded": meta.get("hardwareEncoded"),
                        "preset": meta.get("preset"),
                        "styleId": style_id,
                        "captionSettings": resolved,
                        "ffmpegCommand": meta.get("ffmpegCommand"),
                        "ffmpegStderr": meta.get("ffmpegStderr"),
                        "jobId": job_id,
                    },
                    job_id=job_id,
                )
            total_sec += meta["durationSec"]
            rendered += 1

        ws.mark_done("export", artifacts=[], count=len(cuts))
        if failed:
            reasons = "; ".join(f"clip {i+1}: {r}" for i, _, r in failed)
            self._log(job_id, f"{len(failed)} clip(s) failed to render: {reasons}")
        self._patch_job(
            job_id,
            clipCount=rendered,
            totalClipsSec=round(total_sec, 3),
            renderFailures=len(failed),
        )

    # ------------------------------------------------------------------
    # finish
    # ------------------------------------------------------------------

    def _finish(self, job_id: str, elapsed_sec: float) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        now_ms = int(time.time() * 1000)
        job.update({
            "stage": "complete",
            "status": "complete",
            "progress": 100.0,
            "progressDetail": "All clips ready",
            "completedAt": now_ms,
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



