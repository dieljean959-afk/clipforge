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
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger

from app import download, llm, reframe, subtitle, vad, video
from app.config import Settings
from app.hosting import DispatchError, dispatch_job
from app.stages import STAGES, StageWorkspace
from app.store import LocalStore


class Pipeline:
    def __init__(self, settings: Settings, store: LocalStore) -> None:
        self.settings = settings
        self.store = store

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
    ) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        job.update({
            "stage": stage,
            "progress": max(0.0, min(100.0, pct * 100)),
            "progressDetail": detail,
            "unitLabel": unit_label,
            "unitsDone": units_done,
            "unitsTotal": units_total,
            "lastProgressAt": int(time.time() * 1000),
        })
        self.store.write_job(job)

    def _log(self, job_id: str, line: str) -> None:
        logger.info(f"[job {job_id}] {line}")
        self.store.append_job_log(job_id, line)

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
            transcript = self._stage_transcribe(job, src, ws)
            candidates = self._stage_detect(job, transcript, ws)
            self._stage_render(job, src, transcript, candidates, ws)
            self._finish(job_id, time.time() - t_start)
        except DispatchError as exc:
            # remote worker refused/unreachable — honest error, job stays failed
            self._fail(job_id, f"Could not hand the job to your worker: {exc}", self._detail(exc))
        except llm.LlmNotConfigured as exc:
            self._fail(job_id, "AI detection offline — add GEMINI_API_KEY to .env", self._detail(exc))
        except llm.LlmError as exc:
            self._fail(job_id, f"AI detection failed: {exc}", self._detail(exc))
        except download.DownloadError as exc:
            self._fail(job_id, f"Could not fetch video: {exc}", self._detail(exc))
        except subtitle.TranscribeError as exc:
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
                )
                self._set_word_count(job_id, len(words))
                return transcript

        self._progress(job_id, "transcribing", 0, "Whisper is transcribing word timings...")
        self._log(job_id, f"Stage: transcribe — model '{self.settings.whisper.model}'")

        def on_progress(frac: float) -> None:
            job_now = self.store.get_job(job_id)
            total = job_now.get("durationSec") or 0 if job_now else 0
            self._progress(
                job_id, "transcribing", frac, "Whisper is transcribing word timings...",
                unit_label="min",
                units_done=round(frac * (total / 60), 1),
                units_total=round(total / 60, 1),
            )

        transcript = subtitle.transcribe(
            src, self.settings, on_progress=on_progress,
            cache_dir=ws.stage_dir("transcribe") / "windows",
        )
        ws.write_json("transcribe", "words.json", {
            "words": [w.as_dict() for w in transcript.words],
            "duration": transcript.duration,
            "model": transcript.model_size,
        })
        ws.mark_done("transcribe", artifacts=["transcribe/words.json"],
                     words=len(transcript.words))
        self._set_word_count(job_id, len(transcript.words))
        self._log(job_id, f"transcribed {len(transcript.words)} words in {transcript.duration:.1f}s of audio")
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
        src_meta = video.probe(src)

        # ---- stage: reframe (shot detection + crop windows per clip) ------
        if not ws.is_done("reframe"):
            self._progress(job_id, "rendering", 0, "Detecting shots and framing 9:16 crops...")
            cuts: list[float] = []
            if self.settings.video.reframe:
                cuts = video.detect_scene_cuts(src)
            ws.write_json("reframe", "cuts.json", cuts)
            for i, c in enumerate(candidates):
                windows, used_mp = reframe.face_crop_windows(
                    str(src), c["start"], c["end"],
                    src_meta["width"], src_meta["height"],
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

        clips = self.store.create_clips(job_id, candidates, detected_source="gemini")
        total = len(clips)
        captions_txt_parts: list[str] = []

        for idx, clip_doc in enumerate(clips):
            c0, c1 = clip_doc["start"], clip_doc["end"]
            self._progress(
                job_id, "rendering", idx / max(1, total),
                f"Rendering clip {idx + 1} of {total}",
                unit_label="clips", units_done=idx, units_total=total,
            )

            # ---- resume: a clip already rendered and present is skipped ---
            out_path = self.settings.app.paths.outputs_dir / job_id / f"clip_{idx + 1:02d}.mp4"
            if clip_doc.get("renderState") == "done" and out_path.exists():
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
            cues = subtitle.words_to_cues(
                cue_words, words_per_cue=self.settings.subtitle.words_per_cue,
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
                    "captionPreset": preset,
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
        job.update({
            "stage": "complete",
            "progress": 100,
            "progressDetail": f"{ready} clips ready.",
            "completedAt": int(time.time() * 1000),
        })
        self.store.write_job(job)
        self._log(job_id, f"pipeline complete in {elapsed:.0f}s — {ready} clips rendered")

    def _fail(self, job_id: str, message: str, detail: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        job.update({
            "stage": "failed",
            "error": message,
            "errorDetail": detail[:4000],
        })
        self.store.write_job(job)
        self._log(job_id, f"FAILED: {message}")
        # the traceback goes into the job log so 'View logs' shows the real
        # failure chain, not just the headline
        for line in detail.strip().splitlines()[-12:]:
            self._log(job_id, f"  {line}")
