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

import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger

from app import asr, download, encode, eta, llm, reframe, subtitle, translate, vad, video
from app.config import Settings
from app.hosting import DispatchError, dispatch_job
from app.stages import STAGES, StageWorkspace
from app.store import LocalStore

# Stages whose ETA still has to account for detection + rendering afterwards.
_PRE_TAIL_STAGES = {"queued", "grabbing", "transcribing"}


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

    def __init__(self, ws: StageWorkspace) -> None:
        self._ws = ws

    def get(self, index: int) -> list[dict[str, Any]] | None:
        data = self._ws.read_json("detect", f"chunks/chunk_{index:03d}.json")
        if isinstance(data, list):
            return data
        return None

    def put(self, index: int, candidates: list[dict[str, Any]]) -> None:
        self._ws.write_json("detect", f"chunks/chunk_{index:03d}.json", candidates)


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
            if prev_stage and prev_start:
                timings.append({
                    "stage": prev_stage,
                    "sec": round(max(0.0, (now_ms - prev_start) / 1000), 1),
                })
            job["stageTimings"] = timings[-12:]
            job["stageStartedAt"] = now_ms
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
        tail = 0.0
        if stage in _PRE_TAIL_STAGES:
            opts = job.get("options") or {}
            tail = eta.tail_estimate_sec(opts.get("maxClips") or self.settings.gemini.max_clips)

        job.update({
            "stage": stage,
            "progress": max(0.0, min(100.0, pct * 100)),
            "progressDetail": detail,
            "unitLabel": unit_label,
            "unitsDone": units_done,
            "unitsTotal": units_total,
            "stageEtaSec": None if stage_eta is None else round(stage_eta, 1),
            "totalEtaSec": None if stage_eta is None else round(stage_eta + tail, 1),
            "etaSource": eta_source,
            "lastProgressAt": now_ms,
        })
        self.store.write_job(job)

    def _log(self, job_id: str, line: str) -> None:
        logger.info(f"[job {job_id}] {line}")
        self.store.append_job_log(job_id, line)

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
        try:
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
                return cached

        opts = job.get("options", {}) or {}
        self._progress(job_id, "finding", 0, "Gemini is scoring viral moments...")
        self._log(job_id, "Stage: detect — Gemini clip detection (word indices)")
        words = [w.as_dict() for w in transcript.words]

        def on_log(line: str) -> None:
            self._log(job_id, line)
            self._progress(job_id, "finding", 50, line)

        candidates = llm.detect_clips(
            words,
            self.settings,
            min_clip_sec=opts.get("minClipSec"),
            max_clip_sec=opts.get("maxClipSec"),
            max_clips=opts.get("maxClips"),
            on_log=on_log,
            # Per-chunk persistence: a pause or host restart during chunk 5 of
            # 7 resumes at chunk 6 instead of re-running — and re-billing —
            # all seven Groq calls.
            chunk_cache=_StageChunkCache(ws),
            should_pause=lambda: self.pause_requested(job_id),
        )
        ws.write_json("detect", "candidates.json", candidates)
        ws.mark_done("detect", artifacts=["detect/candidates.json"], count=len(candidates))
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
        preset = opts.get("captionPreset") or self.settings.subtitle.preset
        # One style for the WHOLE batch, decided once and fingerprinted so a
        # later style edit re-encodes every clip instead of leaving the batch
        # half old, half new.
        style_id = subtitle.style_fingerprint(self.settings, preset)
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
            if self.settings.subtitle.strip_fillers or self.settings.subtitle.strip_stutters:
                cue_words, dropped = subtitle.strip_fillers(
                    cue_words,
                    drop_stutters=self.settings.subtitle.strip_stutters,
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
                    words_per_cue=self.settings.subtitle.words_per_cue,
                    max_words=self.settings.subtitle.words_per_cue_max,
                    pause_sec=self.settings.subtitle.pause_sec,
                    # Configured, not hardcoded: [subtitle] low_confidence /
                    # inaudible_below in config.toml. The module defaults only
                    # apply when those keys are absent.
                    confidence_threshold=self.settings.subtitle.low_confidence,
                    inaudible_threshold=self.settings.subtitle.inaudible_below,
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

            # ---- sidecar multi-language tracks (SRT/VTT) -------------------
            tracks = self._write_caption_tracks(job_id, idx + 1, cues)

            # ---- stage: captions (ASS per clip, preset-aware) --------------
            ass_path = ws.stage_dir("captions") / f"clip_{idx + 1:02d}.ass"
            subtitle.write_ass(
                cues, self.settings, ass_path, 1080, 1920, preset_key=preset,
            )
            if idx == 0:
                self._log(job_id, f"caption style: {preset}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                meta = video.render_clip(
                    src, out_path,
                    start=c0, end=c1,
                    settings=self.settings,
                    ass_path=ass_path if cues else None,
                    caption_cues=cues if cues else None,
                    crop_windows=crop_windows,
                    silence_cuts=cuts_for_clip,
                    # The adaptive bitrate ceiling is derived from the output
                    # geometry INCLUDING frame rate, so the source fps has to
                    # reach the encoder. Without it a 30fps source would be
                    # sized as if it were 60 and waste half its budget.
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
                    "translationTracks": tracks,
                })
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
                clip_doc.update({"renderState": "failed", "mediaError": str(exc)})
                self.store.write_clip(clip_doc)
                self._log(job_id, f"clip {idx + 1} render FAILED: {exc}")

            self._progress(
                job_id, "rendering", (idx + 1) / max(1, total),
                f"Rendering clip {idx + 1} of {total}",
                unit_label="clips", units_done=idx + 1, units_total=total,
            )

        # A failed clip is a failed pipeline, not a successful batch with
        # missing media. Leave the stage markers incomplete so Retry resumes
        # this render stage and keeps already-finished clips.
        failed_renders = [
            c for c in self.store.list_clips(job_id) if c.get("renderState") == "failed"
        ]
        if failed_renders:
            detail = "; ".join(
                f"clip {c.get('order', 0) + 1}: {c.get('mediaError') or 'render failed'}"
                for c in failed_renders
            )
            raise video.RenderError(detail)

        # ---- stage: export (final state on disk) --------------------------
        ws.mark_done("captions", artifacts=[])
        ws.mark_done("export", artifacts=[], clips=total)

        # captions.txt content stored on the job
        job = self.store.get_job(job_id) or job
        job["captionsTxt"] = "\n".join(captions_txt_parts)
        job["clipCount"] = total
        job["totalClipsSec"] = sum((c["end"] - c["start"]) for c in candidates)
        self.store.write_job(job)

    # ------------------------------------------------------------------

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
        })
        self.store.write_job(job)
        self._log(job_id, f"FAILED: {message}")
        # the traceback goes into the job log so 'View logs' shows the real
        # failure chain, not just the headline
        for line in detail.strip().splitlines()[-12:]:
            self._log(job_id, f"  {line}")
