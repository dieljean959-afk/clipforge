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

import shutil
import threading
import time
import traceback
from concurrent import futures as cf
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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


class JobPaused(Exception):
    """The user paused this job; the pipeline stopped at a safe boundary.

    Deliberately NOT a failure. Everything completed so far is on disk —
    stage markers, cached transcript chunks, per-chunk detection results and
    finished clips — so resuming continues from exactly here.
    """


class _PauseRequested(Exception):
    """Internal signal raised at a unit boundary when a pause is pending."""


class _StageChunkCache(llm.ChunkCache):
    """Per-chunk detection results on disk, inside the job's own workspace.

    Lives under `detect/chunks/` so it is namespaced by job and removed with
    the workspace on delete. The `detect` DONE marker is only written once the
    whole stage completes, so a partially-detected job keeps these chunks and
    reuses them on resume.
    """

    def __init__(self, ws: StageWorkspace, model: str = "") -> None:
        self._ws = ws
        self._model = model

    def get(self, index: int) -> list[dict[str, Any]] | None:
        data = self._ws.read_json("detect", f"chunks/chunk_{index:03d}.json")
        if isinstance(data, list):
            return data
        return None

    def put(self, index: int, candidates: list[dict[str, Any]]) -> None:
        self._ws.write_json("detect", f"chunks/chunk_{index:03d}.json", candidates)
        if self._model:
            self._note_model(index)

    # -- scoring provenance -------------------------------------------------- #
    # Detection can span more than one model over a job's life: a quota runs
    # out mid-source, the model is switched, the remaining chunks score under a
    # different model than the cached ones. That is invisible in the result —
    # every candidate looks identical — so which model scored which part of the
    # video is recorded here, as a SIDECAR.
    #
    # Deliberately not folded into the chunk files: the chunk payload is what
    # the cache hits on, and reshaping it to carry a model name would
    # invalidate every chunk already paid for.
    def _note_model(self, index: int) -> None:
        prov = self.provenance()
        prov[str(index)] = self._model
        self._ws.write_json("detect", "provenance.json", prov)

    def provenance(self) -> dict[str, str]:
        data = self._ws.read_json("detect", "provenance.json")
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
        return {}


class Pipeline:
    def __init__(self, settings: Settings, store: LocalStore) -> None:
        self.settings = settings
        self.store = store
        # Remaining-time prior for the CURRENT stage (e.g. duration x RTF while
        # transcribing). Reset by _progress whenever the stage changes.
        self._prior_eta_sec: float | None = None

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
                except Exception as exc:  # noqa: BLE001 - never fail a job
                    # A heartbeat that cannot write must not raise into the
                    # render; the watchdog's own timeout remains the backstop.
                    logger.debug(f"[heartbeat] {what}: {exc}")

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
            # The exception already names the right backend and variable
            # ("GROQ_API_KEY missing…", "ollama server unreachable…"). A
            # hardcoded "add GEMINI_API_KEY" overrode it and sent users hunting
            # for a key that was never the one being used.
            self._fail(job_id, f"AI detection offline — {exc}", self._detail(exc))
        except llm.LlmError as exc:
            self._fail(job_id, f"AI detection failed: {exc}", self._detail(exc))
        except download.DownloadError as exc:
            self._fail(job_id, f"Could not fetch video: {exc}", self._detail(exc))
        except asr.AsrError as exc:
            # covers subtitle.TranscribeError too — one class of readable
            # transcription failure reaches the UI either way
            self._fail(job_id, f"Transcription failed: {exc}", self._detail(exc))
        except video.RenderError as exc:
            self._fail(job_id, f"Render failed: {exc}", self._detail(exc))
        except Exception as exc:  # noqa: BLE001 - the UI must see the real error
            logger.exception(f"[job {job_id}] unexpected pipeline error")
            self._fail(job_id, f"Pipeline error: {exc}", self._detail(exc))

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
        resume_from = ws.first_incomplete()
        done_before = [s for s in STAGES if ws.is_done(s)]
        if done_before:
            self._log(
                job_id,
                f"resume: stages already complete = {done_before or 'none'}; "
                f"starting at '{resume_from or 'export'}'",
            )

        src = self._stage_fetch(job, ws)
        self._checkpoint(job_id, "fetch")
        transcript = self._stage_transcribe(job, src, ws)
        self._checkpoint(job_id, "transcribing")
        candidates = self._stage_detect(job, transcript, ws)
        self._checkpoint(job_id, "finding")
        self._stage_render(job, src, transcript, candidates, ws)
        self._finish(job_id, time.time() - t_start)
        # Scratch audio is dead weight once the clips exist.
        self._cleanup_job_temps(job_id, StageWorkspace(self.settings.app.paths.data_dir, job_id))

    @staticmethod
    def _detail(exc: BaseException) -> str:
        """Full traceback for the job log / errorDetail (structured errors)."""
        return "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-4000:]

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
            src = dest_dir / f"source{Path(job['source']).suffix.lower() or '.mp4'}"
            if not src.exists():
                raise download.DownloadError(
                    f"uploaded file missing at {src} — re-submit with 'Attach file'"
                )
            self._progress(job_id, "grabbing", 100, "Upload verified on disk")
            self._log(job_id, f"upload present: {src.name}")
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

        def on_progress(frac: float) -> None:
            now_job = self.store.get_job(job_id)
            total = (now_job.get("durationSec") or 0) if now_job else 0
            self._progress(
                job_id, "transcribing", frac, detail,
                unit_label="min",
                units_done=round(frac * (total / 60), 1),
                units_total=round(total / 60, 1),
                prior_eta_sec=prior_eta,
            )

        # A single Whisper chunk can run for minutes on a small host. The
        # heartbeat keeps the job visibly alive across it; the chunk callback
        # still owns the real progress numbers.
        with self._heartbeat(job_id, "transcribe"):
            transcript = subtitle.transcribe(
                src, self.settings, on_progress=on_progress,
                cache_dir=ws.stage_dir("transcribe") / "chunks",
                should_pause=lambda: self.pause_requested(job_id),
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
        self._progress(job_id, "finding", 0, "Gemini is scoring viral moments...")
        self._log(job_id, "Stage: detect — Gemini clip detection (word indices)")
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
            # 7 resumes at chunk 6 instead of re-running — and re-billing —
            # all seven Groq calls.
            chunk_cache=_StageChunkCache(ws, model=llm.provider_label(self.settings)),
            should_pause=lambda: self.pause_requested(job_id),
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
                f"dropped by the {target}-clip cap — raise [gemini] max_clips for more",
            )
        return candidates

    # ------------------------------------------------------------------
    # stage 4-6: reframe + captions + export (per-clip artifacts)
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
        opts = job.get("options", {}) or {}
        # Check the disk BEFORE encoding anything. Running out of space at
        # clip 9 of 10 wastes the whole batch; refusing up front costs nothing.
        # `src` is the probed source path; its duration is the real driver of
        # how much rendered output this job will produce.
        self._check_disk_space(job_id, float(video.probe(src).get("duration") or 0.0))
        # The caption knobs the dashboard can change are re-read ONCE, here.
        # Resolving per job (not per clip, not per request) is what keeps a
        # batch internally consistent: someone saving a new burst length while
        # ten clips are encoding changes the NEXT job, not clip 7 of this one.
        # The shared settings object is never mutated, so clearing an override
        # really does go back to config.toml.
        settings = caption_style.resolve(self.settings)
        preset = opts.get("captionPreset") or settings.subtitle.preset
        # One style for the WHOLE batch, decided once and fingerprinted so a
        # later style edit re-encodes every clip instead of leaving the batch
        # half old, half new.
        style_id = subtitle.style_fingerprint(settings, preset)
        src_meta = video.probe(src)

        # ---- stage: reframe (shot detection + crop windows per clip) ------
        if not ws.is_done("reframe"):
            self._progress(job_id, "rendering", 0, "Detecting shots and framing 9:16 crops...")
            cuts: list[float] = []
            # Analysis and encoding share one CPU on a hardware-less host, so
            # sample frames more coarsely there (see app.encode).
            enc_choice = encode.choose(self.settings)
            sample_hz = encode.analysis_sample_hz(
                src_meta.get("fps", 30.0), hardware=enc_choice.hardware
            )
            if self.settings.video.reframe:
                self._log(
                    job_id,
                    f"reframe analysis at {sample_hz:.2f} samples/s "
                    f"({enc_choice.label})",
                )
                cuts = video.detect_scene_cuts(
                    src,
                    sample_hz=sample_hz,
                    source_fps=src_meta.get("fps", 30.0),
                )
            ws.write_json("reframe", "cuts.json", cuts)
            for i, c in enumerate(candidates):
                windows, used_mp = reframe.face_crop_windows(
                    str(src), c["start"], c["end"],
                    src_meta["width"], src_meta["height"],
                    sample_hz=sample_hz,
                    scene_cuts=cuts,
                )
                ws.write_json("reframe", f"clip_{i + 1:02d}.json", {
                    "windows": windows, "usedMediapipe": used_mp,
                })
            ws.mark_done("reframe", artifacts=["reframe/cuts.json"], cuts=len(cuts))
        cuts = ws.read_json("reframe", "cuts.json") or []
        self._log(job_id, f"reframe: {len(cuts)} shot cut(s) in source")

        # VAD speech windows (once for the whole file; cached for resume)
        speech: list[dict] = ws.read_json("reframe", "vad.json") or []
        if self.settings.video.silence_trim and not speech:
            try:
                wav = ws.stage_dir("reframe") / "audio.wav"
                video.extract_wav(src, wav)
                speech = vad.detect_speech_windows(wav, self.settings)
                ws.write_json("reframe", "vad.json", speech)
                self._log(job_id, f"Silero VAD: {len(speech)} speech windows")
            except Exception as exc:
                logger.warning(f"[job {job_id}] VAD failed, skipping silence trim: {exc}")
                speech = []

        # Record which backend actually produced these candidates. This used to
        # be the literal "gemini" regardless of provider, so a Groq-detected
        # batch was filed as Gemini work — and the dashboard, which infers
        # success from that string, was one provider rename away from labelling
        # every successful job "Detection failed".
        clips = self.store.create_clips(
            job_id, candidates, detected_source=llm.provider(self.settings)
        )
        total = len(clips)
        captions_txt_parts: list[str] = []

        for idx, clip_doc in enumerate(clips):
            c0, c1 = clip_doc["start"], clip_doc["end"]
            self._progress(
                job_id, "rendering", idx / max(1, total),
                f"Rendering clip {idx + 1} of {total}",
                unit_label="clips", units_done=idx, units_total=total,
            )
            # Pause between clips, never during an encode. Everything already
            # rendered is on disk with renderState="done", so a resume skips
            # straight past it to this clip.
            self._checkpoint(job_id, f"rendering (after clip {idx} of {total})")

            # ---- resume: a clip already rendered and present is skipped ---
            # Only when the burned-in result still matches this job's request:
            # same cut (enforced upstream in create_clips) and same caption
            # style, which is rendered into the pixels, not stored beside it.
            out_path = self.settings.app.paths.outputs_dir / job_id / f"clip_{idx + 1:02d}.mp4"
            if (
                clip_doc.get("renderState") == "done"
                and out_path.exists()
                and clip_doc.get("captionPreset", preset) == preset
                and clip_doc.get("captionStyle", style_id) == style_id
            ):
                self._log(job_id, f"resume: clip {idx + 1} already rendered — skipping")
                self._progress(
                    job_id, "rendering", (idx + 1) / max(1, total),
                    f"Clip {idx + 1} already rendered",
                    unit_label="clips", units_done=idx + 1, units_total=total,
                )
                continue

            clip_doc["renderState"] = "running"
            self.store.write_clip(clip_doc)

            # caption cues from word timings inside this window
            cue_words = [w.as_dict() for w in transcript.words if w.end > c0 and w.start < c1]
            # ~1 word in 10 arrives with start == end; a zero span means a
            # flickering highlight and a cue that can flash for one frame.
            cue_words = subtitle.normalize_word_timings(cue_words)
            # Fillers/stutters are dropped from the DISPLAYED text only; the
            # stored transcript keeps every word, so the .txt, the search index
            # and a later re-render are unaffected.
            if settings.subtitle.strip_fillers or settings.subtitle.strip_stutters:
                cue_words, dropped = subtitle.strip_fillers(
                    cue_words,
                    drop_stutters=settings.subtitle.strip_stutters,
                )
                if dropped:
                    self._log(
                        job_id,
                        f"clip {idx + 1}: dropped {dropped} filler/stutter word(s) from "
                        "the captions (audio untouched)",
                    )
            if not cue_words:
                # A clip window that is nothing but "um, uh, like" has no words
                # worth captioning. Emitting an empty Dialogue line would put a
                # blank caption on screen for the whole clip.
                cues = []
            else:
                cues = subtitle.words_to_cues(
                    cue_words,
                    # Short bursts, cut on the speaker's own pauses. The target
                    # length and the hard cap are both config; a pause always
                    # wins, which is what makes the rhythm follow the speech.
                    words_per_cue=settings.subtitle.words_per_cue,
                    max_words=settings.subtitle.words_per_cue_max,
                    pause_sec=settings.subtitle.pause_sec,
                    # Configured, not hardcoded: [subtitle] low_confidence /
                    # inaudible_below in config.toml. The module defaults only
                    # apply when those keys are absent.
                    confidence_threshold=settings.subtitle.low_confidence,
                    inaudible_threshold=settings.subtitle.inaudible_below,
                )
            for cue in cues:
                cue["start"] -= c0
                cue["end"] -= c0

            # reframe windows (per-shot, cut-aware)
            crop_windows: list[dict] | None = None
            if self.settings.video.reframe:
                rf = ws.read_json("reframe", f"clip_{idx + 1:02d}.json") or {}
                crop_windows = rf.get("windows")
                if crop_windows and not rf.get("usedMediapipe"):
                    self._log(job_id, f"clip {idx + 1}: MediaPipe missing — center crop")

            # silence cuts for this clip
            cuts_for_clip = vad.silence_cuts_for_clip(speech, c0, c1, self.settings) if speech else []

            # ---- stage: captions (ASS per clip, preset-aware) --------------
            ass_path = ws.stage_dir("captions") / f"clip_{idx + 1:02d}.ass"
            subtitle.write_ass(
                cues, settings, ass_path, 1080, 1920, preset_key=preset,
            )
            if idx == 0:
                self._log(job_id, f"caption style: {preset}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Each clip is its own isolated ffmpeg, with its own verify,
                # its own retry and its own cleanup. One clip that cannot be
                # encoded must cost ONE clip, not the batch — a batch is
                # often hours of work by the time it gets here.
                meta = self._render_one_clip(
                    job_id, idx + 1, total,
                    src=src, out_path=out_path, start=c0, end=c1,
                    ass_path=ass_path if cues else None,
                    cues=cues if cues else None,
                    crop_windows=crop_windows,
                    silence_cuts=cuts_for_clip,
                    src_fps=float(src_meta.get("fps") or 30.0),
                )
                thumb = self.settings.app.paths.outputs_dir / job_id / f"clip_{idx + 1:02d}.jpg"
                video.render_first_frame_thumb(out_path, thumb)
                clip_doc.update({
                    "videoPath": str(out_path),
                    "thumbPath": str(thumb),
                    "mediaBytes": meta["bytes"],
                    "mediaWidth": meta["width"],
                    "mediaHeight": meta["height"],
                    "mediaDurationSec": meta["duration"],
                    "renderState": "done",
                    "renderMs": meta["renderMs"],
                    "cropBox": crop_windows[0] if crop_windows else None,
                    "silenceCuts": cuts_for_clip,
                    "captionCues": cues,
                    # Surfaced in the UI so a clip with [muttering] placeholders
                    # says so, instead of the count silently changing what the
                    # captions say with no visible reason.
                    "uncertainCues": sum(1 for c in cues if c.get("uncertain")),
                    "captionPreset": preset,
                    "captionStyle": style_id,
                })
                self.store.write_clip(clip_doc)

                # ---- sidecar multi-language tracks (SRT/VTT) ---------------
                # AFTER the encode, never before it. These are LLM calls (one
                # batch per language, twelve languages) and the MP4 is the
                # thing the user is waiting for. Translating first meant a slow
                # or unfunded backend stalled the encode of every single clip,
                # so the batch sat at "Rendering clip 1 of 13, 0%" for hours
                # while the video it was holding hostage was already available.
                # The render is now durable before the first caption is asked
                # for, and the tracks patch onto the finished clip doc after.
                tracks = self._write_caption_tracks(job_id, idx + 1, cues)
                if tracks:
                    clip_doc["translationTracks"] = tracks
                    self.store.write_clip(clip_doc)
                captions_txt_parts.append(
                    f"=== Clip {idx + 1}: {clip_doc['title']} (score {clip_doc['score']:.0f}) ==="
                )
                for i, cue in enumerate(cues, 1):
                    captions_txt_parts.append(
                        f"{i}\n{self._srt_time(cue['start'])} --> "
                        f"{self._srt_time(cue['end'])}\n{cue['text']}\n"
                    )
            except video.RenderError as exc:
                # SKIP, do not fail the batch. The remaining clips are still
                # worth rendering and the finished ones are already durable.
                # This used to raise, which turned one bad encode into a red
                # job and threw away the other nine.
                clip_doc.update({"renderState": "failed", "mediaError": str(exc)})
                self.store.write_clip(clip_doc)
                self._log(
                    job_id,
                    f"clip {idx + 1} of {total} could not be rendered and was "
                    f"skipped: {exc}",
                )
                self._progress(
                    job_id, "rendering", (idx + 1) / max(1, total),
                    f"Clip {idx + 1} skipped after 2 attempts — continuing",
                    unit_label="clips", units_done=idx + 1, units_total=total,
                )

            self._progress(
                job_id, "rendering", (idx + 1) / max(1, total),
                f"Rendering clip {idx + 1} of {total}",
                unit_label="clips", units_done=idx + 1, units_total=total,
            )

        # Skipped clips are reported, not fatal: the batch is still a batch.
        # Retry re-attempts ONLY the clips marked failed, so a retry that
        # happens to get a healthier host finishes the stragglers.
        failed_renders = [
            c for c in self.store.list_clips(job_id) if c.get("renderState") == "failed"
        ]
        if failed_renders:
            names = ", ".join(str(c.get("order", 0) + 1) for c in failed_renders)
            self._log(
                job_id,
                f"{len(failed_renders)} of {total} clip(s) could not be rendered "
                f"and were skipped (clip {names}). Every other clip is ready. "
                f"Hit Retry to attempt just those again.",
            )
            self._patch_job(
                job_id,
                progressDetail=(
                    f"{total - len(failed_renders)} of {total} clips ready — "
                    f"clip {names} skipped after 2 attempts"
                ),
            )

        # ---- stage: export (final state on disk) --------------------------
        ws.mark_done("captions", artifacts=[])
        ws.mark_done("export", artifacts=[], clips=total)

        # captions.txt content stored on the job
        job = self.store.get_job(job_id) or job
        job["captionsTxt"] = "\n".join(captions_txt_parts)
        job["clipCount"] = total
        job["totalClipsSec"] = sum((c["end"] - c["start"]) for c in candidates)
        self.store.write_job(job)

        # Share copies are built HERE, in the background, rather than on the
        # tap. `/share` is not a static file server: on a miss it transcodes,
        # which on a small host costs ~15s of the one CPU per clip. Thirteen
        # clips on demand is ~33MB and three minutes of queueing, so a user
        # opening the grid could never get more than the first card to Ready.
        # Warming them while the job finishes turns every later request into a
        # plain static download, and the button goes green immediately.
        #
        # Off the critical path on purpose: the batch is already durable and
        # the job must reach `complete` now, not three minutes later. A clip
        # tapped before its copy exists still works - `/share` transcodes on
        # demand - so prewarming can only ever make it faster.
        threading.Thread(
            target=self._prewarm_share_copies,
            args=(job_id,),
            name=f"share-prewarm-{job_id[:8]}",
            daemon=True,
        ).start()

    def _render_one_clip(
        self,
        job_id: str,
        number: int,
        total: int,
        *,
        src: Path,
        out_path: Path,
        start: float,
        end: float,
        ass_path: Path | None,
        cues: list[dict[str, Any]] | None,
        crop_windows: list[dict[str, Any]] | None,
        silence_cuts: list[dict[str, Any]] | None,
        src_fps: float,
    ) -> dict[str, Any]:
        """Encode one clip, verify it, retry once safely, then give up on it.

        Returns the encoder metadata on success. Raises RenderError only after
        BOTH attempts fail — the caller records the clip as skipped and moves
        on to the next one.

        The verification is not decoration. ffmpeg can exit 0 and leave a
        zero-byte or truncated MP4 behind (out of space, a killed frame
        thread), and a clip that is marked `renderState="done"` with no
        playable file is exactly the failure that surfaces much later as a
        player that spins forever.
        """
        last: Exception | None = None
        for attempt in (1, 2):
            try:
                # The heartbeat spans ONLY the encode. Nothing in it reports
                # progress -- ffmpeg's own output is read for errors, not
                # percentages -- so without this a long clip looks dead, and
                # the watchdog can kill a render that is working.
                with self._heartbeat(job_id, f"render clip {number}"):
                    meta = video.render_clip(
                        src, out_path,
                        start=start, end=end,
                        settings=self.settings,
                        ass_path=ass_path,
                        caption_cues=cues,
                        crop_windows=crop_windows,
                        silence_cuts=silence_cuts,
                        src_fps=src_fps,
                        safe=attempt == 2,
                    )
                video.verify_output(out_path, expected_sec=end - start)
                if attempt == 2:
                    self._log(
                        job_id,
                        f"clip {number} of {total}: first attempt failed "
                        f"({last}); the safer retry produced a usable file",
                    )
                return meta
            except Exception as exc:  # noqa: BLE001 - retried below, then skipped
                last = exc
                # Never let a half-written MP4 be mistaken for a finished one
                # by the resume path or by /share.
                try:
                    out_path.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt == 1:
                    self._log(
                        job_id,
                        f"clip {number} of {total} failed on attempt 1/2 "
                        f"({type(exc).__name__}: {exc}) — retrying 2/3 with "
                        f"safer settings",
                    )
                    self._progress(
                        job_id, "rendering",
                        (number - 1) / max(1, total),
                        f"Clip {number} failed, retrying 2/3 with safer settings",
                        unit_label="clips", units_done=number - 1,
                        units_total=total,
                    )
        raise video.RenderError(str(last) if last else "render failed")

    def _cleanup_job_temps(self, job_id: str, ws: StageWorkspace) -> None:
        """Delete the scratch files a job leaves behind once it is finished.

        The 16 kHz WAV used for transcription is ~19 MB per 5-minute chunk of
        extraction and is by far the largest temporary artefact; the reframe
        analysis WAV is another full copy of the source audio. Neither is
        needed after the job completes, and on a 4 GB sandbox disk a handful of
        finished batches silently fills it.
        """
        removed = 0
        freed = 0
        for stage, name in (("transcribe", "source.wav"), ("reframe", "audio.wav")):
            path = ws.stage_dir(stage) / name
            try:
                if path.exists():
                    freed += path.stat().st_size
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        if removed:
            logger.info(
                f"[job {job_id}] cleanup: removed {removed} temp file(s), "
                f"freed {freed / 1024 / 1024:.0f} MB"
            )

    def _free_disk_mb(self) -> int:
        """Free space on the volume holding outputs/, in MB."""
        try:
            usage = shutil.disk_usage(self.settings.app.paths.outputs_dir)
            return usage.free // (1024 * 1024)
        except Exception:
            return 1 << 30

    def _check_disk_space(self, job_id: str, duration_sec: float) -> None:
        """Fail early and clearly when the disk cannot hold the render.

        Discovering this at the 9th of 10 clips is the worst place to find out:
        the earlier clips were rendered, the download is still on disk, and the
        user is told only that something went wrong. Estimated need is the
        rendered MP4 size we have actually been producing plus the downloaded
        source, with a safety floor.
        """
        needed = int(max(500.0, duration_sec * 3.0))
        free = self._free_disk_mb()
        if free < needed:
            raise download.DownloadError(
                f"Not enough free disk space to finish this job: about "
                f"{needed} MB is needed but only {free} MB is left. Delete an "
                f"older batch (or its clips) and Retry — everything already "
                f"rendered is cached and will not be redone."
            )
        self._log(
            job_id,
            f"disk check: {free} MB free, ~{needed} MB needed for this render",
        )

    def _prewarm_share_copies(self, job_id: str) -> None:
        """Build every clip's share copy ahead of the first tap.

        Strictly best-effort: a share copy is an optimisation for the file
        share path, so a failure here is logged and dropped rather than
        failing a batch whose MP4s are all already rendered and served.
        """
        masters: list[Path] = []
        for clip in self.store.list_clips(job_id):
            if clip.get("renderState") != "done":
                continue
            raw = str(clip.get("videoPath") or "")
            if not raw:
                continue
            path = Path(raw)
            if path.is_file():
                masters.append(path)

        if not masters:
            return

        def warm(master: Path) -> str:
            try:
                share = share_video.build_share_version(master)
                # The share link is minted HERE, at the end of rendering, so
                # every finished clip already has a public URL and the badge
                # can go green without the user tapping anything.
                self._mark_share_ready(master, share)
                return ""
            except Exception as exc:  # noqa: BLE001 - never fail the batch
                return f"{master.name}: {exc}"

        # Two at a time: enough to overlap the transcode's I/O with its encode,
        # not so many that they fight the worker pool for a single core.
        workers = min(2, max(1, len(masters)))
        problems: list[str] = []
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            problems = [p for p in pool.map(warm, masters) if p]

        if problems:
            logger.warning(
                f"[share] prewarm incomplete for {job_id}: {len(problems)} of "
                f"{len(masters)} clip(s) failed ({problems[0]})"
            )
        else:
            logger.info(f"[share] prewarmed {len(masters)} share copies for {job_id}")

    # ------------------------------------------------------------------

    def _mark_share_ready(self, master: Path, share: Path) -> None:
        """Record the clip's real share state after a successful prewarm.

        Written to the clip document rather than computed per request, so the
        batch list can answer "Ready" from the payload it already sends. The
        probe is still the source of truth — this only records what the probe
        just concluded.
        """
        state, detail = sharelink.probe(share)
        job_dir = master.parent
        for clip in self.store.list_all_clips(limit=1000):
            if str(clip.get("videoPath") or "") != str(master):
                continue
            clip["shareId"] = sharelink.ensure_id(clip)
            clip["shareState"] = state
            clip["shareDetail"] = detail
            try:
                self.store.write_clip(clip)
            except Exception as exc:  # noqa: BLE001 - telemetry, never fatal
                logger.warning(f"[share] could not record state for {job_dir.name}: {exc}")

    def _finish(self, job_id: str, elapsed: float) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        ready = self.store.clips_ready_count(job_id)
        resumes = int(job.get("interruptCount") or 0)
        job.update({
            "stage": "complete",
            "progress": 100,
            "progressDetail": f"{ready} clips ready.",
            "completedAt": int(time.time() * 1000),
            # A completed job is no longer interrupted, whatever it survived.
            "interrupted": False,
            "interruptReason": None,
            # ...and it is certainly no longer carrying the error that killed
            # an earlier attempt. stage="complete" with an error string is a
            # contradiction, and it outlives the job in every API response.
            "error": None,
            "errorDetail": None,
        })
        if resumes:
            # Worth stating plainly: the result is real, but it took extra
            # attempts, which changes what a render-time estimate means.
            self._log(
                job_id,
                f"completed after {resumes} host-restart interruption(s) — "
                "resumed from cached stages each time",
            )
        self.store.write_job(job)
        self._log(job_id, f"pipeline complete in {elapsed:.0f}s — {ready} clips rendered")

    def _fail(self, job_id: str, message: str, detail: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        # Keep the stage that actually failed — `stage` is about to be
        # overwritten with "failed", and without this the dashboard can only
        # say "Failed" with no idea which step broke.
        failed_stage = job.get("stage")
        job.update({
            "stage": "failed",
            "failedStage": failed_stage,
            "error": message,
            "errorDetail": detail[:4000],
            # This attempt ran and finished on its own terms, so the job is no
            # longer "waiting to be resumed". Leaving `interrupted` set made
            # boot recovery re-queue the job on EVERY subsequent restart — a
            # job that can never succeed (its upload is gone) then burned a
            # full pipeline launch per restart, indefinitely, which is exactly
            # the "resuming (attempt 2)..." loop that was reported.
            "interrupted": False,
        })
        self.store.write_job(job)
        self._log(job_id, f"FAILED: {message}")
        # the traceback goes into the job log so 'View logs' shows the real
        # failure chain, not just the headline
        for line in detail.strip().splitlines()[-12:]:
            self._log(job_id, f"  {line}")
