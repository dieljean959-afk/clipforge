"""Task service: pipeline orchestration for the in-process worker.

Stage order (MPT heritage): grab → transcribe → find → render. All progress
writes go through the store so the dashboard shows real per-stage progress.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from loguru import logger

from app import download, llm, reframe, subtitle, vad, video
from app.config import Settings
from app.hosting import DispatchError, dispatch_job
from app.store import LocalStore


class Pipeline:
    def __init__(self, settings: Settings, store: LocalStore) -> None:
        self.settings = settings
        self.store = store

    # ------------------------------------------------------------------
    # progress helpers
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
    # stages
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
                src = self._stage_grab(job)
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

            src = self._stage_grab(job)
            transcript = self._stage_transcribe(job, src)
            candidates = self._stage_find(job, transcript)
            self._stage_render(job, src, transcript, candidates)
            self._finish(job_id, time.time() - t_start)
        except DispatchError as exc:
            # remote worker refused/unreachable — honest error, job stays failed
            self._fail(job_id, f"Could not hand the job to your worker: {exc}", str(exc))
        except llm.LlmNotConfigured as exc:
            self._fail(job_id, "AI detection offline — add GEMINI_API_KEY to .env", str(exc))
        except llm.LlmError as exc:
            self._fail(job_id, f"AI detection failed: {exc}", str(exc))
        except download.DownloadError as exc:
            self._fail(job_id, f"Could not fetch video: {exc}", str(exc))
        except video.RenderError as exc:
            self._fail(job_id, f"Render failed: {exc}", str(exc))
        except Exception as exc:  # noqa: BLE001 - the UI must see the real error
            logger.exception(f"[job {job_id}] unexpected pipeline error")
            self._fail(job_id, f"Pipeline error: {exc}", repr(exc))

    # ------------------------------------------------------------------

    def _stage_grab(self, job: dict[str, Any]) -> Path:
        job_id = job["_id"]
        dest_dir = self.settings.app.paths.downloads_dir / job_id
        self._progress(job_id, "grabbing", 0, "Clippy is grabbing the video...")
        self._log(job_id, f"Stage: grab — {job['source'][:80]}")

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
                self._progress(
                    job_id, "grabbing", frac,
                    "Clippy is grabbing the video...",
                )

            src = download.download_url(
                job["source"], dest_dir, self.settings,
                on_progress=on_progress,
                on_log=lambda line: self._log(job_id, line),
            )
        video.verify_source(src)
        meta = video.probe(src)
        job = self.store.get_job(job_id) or job
        job["durationSec"] = meta["duration"]
        job["downloadSource"] = "upload" if job.get("sourceType") == "upload" else "yt-dlp"
        self.store.write_job(job)
        return src

    def _stage_transcribe(self, job: dict[str, Any], src: Path) -> subtitle.Transcript:
        job_id = job["_id"]
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

        transcript = subtitle.transcribe(src, self.settings, on_progress=on_progress)
        job = self.store.get_job(job_id) or job
        job["transcriptWordCount"] = len(transcript.words)
        self.store.write_job(job)
        self._log(job_id, f"transcribed {len(transcript.words)} words in {transcript.duration:.1f}s of audio")
        return transcript

    def _stage_find(self, job: dict[str, Any], transcript: subtitle.Transcript) -> list[dict[str, Any]]:
        job_id = job["_id"]
        opts = job.get("options", {}) or {}
        self._progress(job_id, "finding", 0, "Gemini is scoring viral moments...")
        self._log(job_id, "Stage: find — Gemini clip detection")
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
        return candidates

    def _stage_render(
        self,
        job: dict[str, Any],
        src: Path,
        transcript: subtitle.Transcript,
        candidates: list[dict[str, Any]],
    ) -> None:
        job_id = job["_id"]
        self._progress(job_id, "rendering", 0, "Rendering 9:16 crops with captions...")
        self._log(job_id, f"Stage: render — {len(candidates)} clips")

        # per-job work dirs
        work = self.settings.app.paths.data_dir / "work" / job_id
        work.mkdir(parents=True, exist_ok=True)

        # word dict list for cue building
        words = [w.as_dict() for w in transcript.words]

        # VAD speech windows (once for the whole file)
        speech: list[dict] = []
        if self.settings.video.silence_trim:
            try:
                wav = work / "audio.wav"
                video.extract_wav(src, wav)
                speech = vad.detect_speech_windows(wav, self.settings)
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
            clip_doc["renderState"] = "running"
            self.store.write_clip(clip_doc)

            # caption cues from word timings inside this window
            cue_words = [w for w in words if w["end"] > c0 and w["start"] < c1]
            cues = subtitle.words_to_cues(
                cue_words,
                words_per_cue=self.settings.subtitle.words_per_cue,
            )
            for cue in cues:
                cue["start"] -= c0
                cue["end"] -= c0

            # reframe windows
            crop_windows: list[dict] | None = None
            if self.settings.video.reframe:
                try:
                    windows, used_mp = reframe.face_crop_windows(
                        str(src), c0, c1,
                        video.probe(src)["width"], video.probe(src)["height"],
                    )
                    crop_windows = windows
                    if not used_mp:
                        self._log(job_id, f"clip {idx + 1}: MediaPipe missing — center crop")
                except Exception as exc:
                    logger.warning(f"[job {job_id}] reframe failed for clip {idx + 1}: {exc}")

            # silence cuts for this clip
            cuts = vad.silence_cuts_for_clip(speech, c0, c1, self.settings) if speech else []

            # ASS file for this clip (word-pop, green outline)
            ass_path = work / f"clip_{idx + 1:02d}.ass"
            subtitle.write_ass(cues, self.settings, ass_path, 1080, 1920)

            out_path = self.settings.app.paths.outputs_dir / job_id / f"clip_{idx + 1:02d}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                meta = video.render_clip(
                    src, out_path,
                    start=c0, end=c1,
                    settings=self.settings,
                    ass_path=ass_path if cues else None,
                    crop_windows=crop_windows,
                    silence_cuts=cuts,
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
                    "silenceCuts": cuts,
                    "captionCues": cues,
                })
                self.store.write_clip(clip_doc)
                captions_txt_parts.append(f"=== Clip {idx + 1}: {clip_doc['title']} (score {clip_doc['score']:.0f}) ===")
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

        # captions.txt content stored on the job
        job = self.store.get_job(job_id) or job
        job["captionsTxt"] = "\n".join(captions_txt_parts)
        job["clipCount"] = total
        job["totalClipsSec"] = sum(
            (c["end"] - c["start"]) for c in candidates
        )
        self.store.write_job(job)

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
            "errorDetail": detail[:1500],
        })
        self.store.write_job(job)
        self._log(job_id, f"FAILED: {message}")
