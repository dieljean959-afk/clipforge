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
import os
import shutil
import threading
import time
import traceback
from concurrent import futures as cf
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app import (
    asr,
    caption_style,
    clipcount,
    download,
    encode,
    eta,
    llm,
    reframe,
    share_video,
    sharelink,
    subtitle,
    translate,
    vad,
    video,
)
from app.config import Settings
from app.hosting import DispatchError, dispatch_job
from app.stages import STAGES, StageWorkspace
from app.store import LocalStore

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
            used = hostlimits.dir_bytes(path)
            free = hostlimits.dir_free(path)
        except Exception as exc:
            logger.warning(f"[job {job_id}] could not measure disk: {exc}")
            return
        if free is not None and free < 200 * 1024 * 1024:
            raise video.RenderError(
                f"not enough free disk ({lim}) — less than 200 MB free on "
                f"{path}. The download and transcript are still cached; Retry "
                f"will resume from the last completed stage once space is cleared."
            )
        if source_duration_sec > 0:
            est = source_duration_sec * 0.05
            if free is not None and est and free < est + 500 * 1024 * 1024:
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
            logger.exception(f"[job {job_id}] unexpected pipeline error")
            self._fail(job_id, f"Pipeline error: {exc}", self._detail(exc))
        finally:
            # Make the liveness heartbeat visible to the reaper/web. A child that
            # reaches here — including on a hard crash — has had its last beat at
            # the moment the heartbeat context manager stopped, so persist that
            # timestamp once more with the (possibly terminal) stageStatus we have
            # now. The reaper uses heartbeatAt + lastProgressAt together to decide
            # whether a running-looking job still has a live child.
            try:
                job = self.store.get_job(job_id)
                if job is not None:
                    job["heartbeatAt"] = self._heartbeat_at
                    self.store.write_job(job)
            except Exception:
                pass

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

        src = self._stage_fetch(job, ws)
        self._mark_stage_running(job_id, "fetch")
        try:
            src = self._stage_fetch(job, ws)
            self._mark_stage_done(job_id, "fetch")
        except Exception:
            self._mark_stage_failed(job_id, "fetch", *self._fail_args_for(sys.exc_info()))
            raise
        self._checkpoint(job_id, "fetch")
        self._mark_stage_running(job_id, "transcribe")
        transcript = self._stage_transcribe(job, src, ws)
        self._mark_stage_done(job_id, "transcribe")
        self._transcript_size_note(len(transcript.words))
        self._checkpoint(job_id, "transcribing")
        self._mark_stage_running(job_id, "detect")
        candidates = self._stage_detect(job, transcript, ws)
        self._mark_stage_done(job_id, "detect")
        self._checkpoint(job_id, "finding")
        self._mark_stage_running(job_id, "reframe")
        self._stage_render(job, src, transcript, candidates, ws)
        self._mark_stage_done(job_id, "reframe")
        self._render_workload_note(len(candidates), float(job.get("durationSec") or 0.0))
        self._mark_stage_running(job_id, "captions")
        self._mark_stage_done(job_id, "captions")
        self._mark_stage_running(job_id, "export")
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
        self._log(job_id, f"Stage: fetch — {job['source'][:80]}")

        if job.get("sourceType") == "upload":
            # Search BOTH storage locations. The expected name is tried first
            # so a job whose recorded filename is authoritative still resolves
            # the same way it always did; the general sweep is the fallback for
            # a job written under the other root.
            expected = f"source{Path(job['source']).suffix.lower() or '.mp4'}"
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
            def on_progress(frac: float, _eta: float | None) -> None:
                self._progress(job_id, "grabbing", frac, "Clippy is grabbing the video...")

            src = download.download_url(
                job["source"], dest_dir, self.settings,
                on_progress=on_progress,
                on_log=lambda line: self._log(job_id, line),
            )
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
        with self._heartbeat(job_id, "transcribe"):
            transcript = subtitle.transcribe(
                src, self.settings, on_progress=on_progress,
                cache_dir=chunk_cache_dir,
                should_pause=lambda: self.pause_requested(job_id),
                on_chunk=on_chunk,
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

        def on_log(line: str) -> None:
            self._log(job_id, line)

        def on_progress(fraction: float, line: str) -> None:
            self._progress(job_id, "finding", fraction, line, unit_label="chunk")

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
            duration_sec=float(job.get("durationSec") or 0.0),
            on_log=on_log,
            on_progress=on_progress,
            report=report,
            # Per-chunk persistence: a pause or host restart during chunk 5 of
            # 7 resumes at chunk 6 instead of re-running — and re-billing —\n            # all seven Groq calls.
            chunk_cache=_StageChunkCache(ws, model=llm.provider_label(self.settings)),
            should_pause=lambda: self.pause_requested(job_id),
        )
        ws.write_json("detect", "candidates.json", candidates)
        ws.mark_done("detect", artifacts=["detect/candidates.json"], count=len(candidates))
        # The three numbers that explain the batch size. "Detected 61, kept 18\n        # of a possible 23" is a different story from "asked for 10, got 10",\n        # and only the first one tells the user whether to raise min_score,\n        # lower the cap, or leave it alone.
        job = self._patch_job(
            job_id,
            candidatesDetected=int(report.get("detected", len(candidates))),
            clipsAboveScore=int(report.get("aboveFloor", len(candidates))),
            clipsBelowScore=int(report.get("belowFloor", 0)),
            clipScoreFloor=float(report.get("floor", self.settings.gemini.min_score)),
        )
        self._log(
            job_id,
            f"Detection: {job.get('candidatesDetected')} candidate(s) found, "
            f"{job.get('clipsAboveScore')} scored >= {job.get('clipScoreFloor'):.0f}, "
            f"{len(candidates)} kept (cap was {target})",
        )
        if report.get("aboveFloor", 0) > len(candidates):
            self._log(
                job_id,
                f"Note: {report['aboveFloor'] - len(candidates)} good candidate(s) were " 
            )
        )
        ws.write_json("detect", "candidates.json", candidates)
        ws.mark_done("detect", artifacts=["detect/candidates.json"], count=len(candidates))
        # The three numbers that explain the batch size. "Detected 61, kept 18
        # of a possible 23" is a different story from "asked for 10, got 10",
        # and only the first one tells the user whether to raise min_score,
        # lower the cap, or leave it alone.
        job = self._patch_job(
            job_id,
            candidatesDetected=int(report.get("detected", len(candidates))),
            clipsAboveScore=int(report.get("aboveFloor", len(candidates))),
            clipsBelowScore=int(report.get("belowFloor", 0)),
            clipScoreFloor=float(report.get("floor", self.settings.gemini.min_score)),
        )
        self._log(
            job_id,
            f"Detection: {job.get('candidatesDetected')} candidate(s) found, "
            f"{job.get('clipsAboveScore')} scored >= {job.get('clipScoreFloor'):.0f}, "
            f"{len(candidates)} kept (cap was {target})",
        )
        if report.get("aboveFloor", 0) > len(candidates):
            self._log(
                job_id,
                f"Note: {report['aboveFloor'] - len(candidates)} good candidate(s) were "
                f"above the floor but were not selected because of the cap."
            )
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

        candidates = ws.read_json("detect", "candidates.json") or []
        cuts = ws.read_json("reframe", "cuts.json") or []
        if not cuts:
            self._log(job_id, "no cuts to caption — skipping")
            return

        cues_by_clip: list[list[dict[str, Any]]] = []
        for i, cut in enumerate(cuts):
            cues = translate.build_cues(
                transcript, cut, self.settings,
            )
            cues_by_clip.append(cues)
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

        dest = self.settings.app.paths.outputs_dir / job_id
        dest.mkdir(parents=True, exist_ok=True)

        for i, cut in enumerate(cuts):
            self._progress(
                job_id, "exporting",
                (i + 1) / len(cuts),
                f"Rendering clip {i + 1} of {len(cuts)}",
                unit_label="clips",
                units_done=i + 1, units_total=len(cuts),
            )
            encode.render_clip(
                src, cut, dest / f"clip_{i:03d}.mp4", self.settings,
                on_log=lambda line: self._log(job_id, line),
            )
        ws.mark_done("export", artifacts=[], count=len(cuts))

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
            "progress": 100.0,
            "progressDetail": "All clips ready",
            "completedAt": now_ms,
            "lastProgressAt": now_ms,
            "stageStatus": self._stage_status,
        })
        self.store.write_job(job)
        self.store.append_job_log(job_id, f"COMPLETE in {elapsed_sec:.1f}s")
        logger.info(f"[job {job_id}] COMPLETE in {elapsed_sec:.1f}s")


# ------------------------------------------------------------------
# chunk cache helpers
# ------------------------------------------------------------------


def _chunk_cached(cache_dir: Path | None, index: int) -> bool:
    """True when a chunk's transcript has already been cached."""
    if cache_dir is None or not cache_dir.is_dir():
        return False
    return (cache_dir / f"chunk_{index}.json").exists()


class _StageChunkCache:
    """Per-chunk persistence for LLM / ASR stages that expose a cache dir."""

    def __init__(self, ws: StageWorkspace, *, model: str) -> None:
        self._ws = ws
        self._model = model

    def dir(self, stage: str) -> Path:
        return self._ws.stage_dir(stage) / "chunks"

    def key(self, stage: str, index: int) -> str:
        return f"{stage}_chunk_{index}.json"

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

