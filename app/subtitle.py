"""Subtitle service: WhisperX/batched-whisper word-level transcription + ASS captions.

Transcription is delegated to ``app.asr`` (WhisperX when installed, otherwise
faster-whisper's batched CTranslate2 pipeline — see that module for why). This
module owns the parts that do not depend on the engine: resumable chunk
planning, overlap de-duplication, RTF measurement, progress reporting and the
ASS/Pillow caption rendering.

ASS style matches the reference look: bold white text with a thick neon-green
outline, word-by-word pop-in, 2-3 words visible, lower-third. If ImageMagick
is missing, rendering falls back to Pillow-composited PNG overlays via ffmpeg —
captions must export either way.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from app import asr
from app.config import Settings, ffmpeg_bin, ffprobe_bin


class TranscribeError(asr.AsrError):
    """Transcription failed. Subclasses AsrError so callers can catch either."""


class TranscriptionPaused(Exception):
    """User paused during transcription; stopped at a chunk boundary.

    NOT a failure. Decoded chunks are cached on disk, so resuming reuses them
    instead of running whisper over the whole file again.
    """


# --------------------------------------------------------------------------- #
# Anti-hallucination tuning
#
# Whisper is known to enter repetition loops on silent or music-only segments
# ("Thank you for watching..." x40). The settings themselves live in app.asr
# (single source of truth, shared by both engines); re-exported here because
# the test-suite and older callers import them from this module:
#   1. condition_on_previous_text=False — each chunk decodes without seeing
#      previous text, so a hallucination cannot anchor the next chunk.
#   2. compression_ratio_threshold — segments whose compressed text is far
#      smaller than the raw text are repetitive; they are dropped (not kept).
#   3. no_speech_threshold — segments with high silence probability are
#      skipped, so near-silent audio yields no words instead of fantasy text.
#   4. vad_filter — Silero VAD strips non-speech before the encoder sees it.
# On top of that, long files are decoded in overlapping chunks so a failure or
# resume only costs one chunk, and so nothing drifts across the whole file.
# --------------------------------------------------------------------------- #

WHISPER_KWARGS = dict(asr.ANTI_REPETITION, word_timestamps=True)

# Resume granularity. Long files are decoded in chunks of this length with this
# much overlap, so a word spanning a boundary is caught by at least one chunk
# (then deduplicated in _dedupe_words). Chunks are a RESUME unit, not a model
# constraint: the engine batches its own 30s windows inside each chunk, so a
# large chunk costs no extra compute — it just means a resume may lose up to
# CHUNK_SEC of transcription instead of 30s.
CHUNK_SEC = 300.0
OVERLAP_SEC = 2.0

# Back-compat aliases for callers/tests that used the old 30s window names.
WINDOW_SEC = CHUNK_SEC
MAX_DECODE_SEC = CHUNK_SEC


# --------------------------------------------------------------------------- #
# Caption style presets
#
# Each preset is a full override of the ASS style (colors are in ASS BGR hex
# via _hex_to_ass). The user picks one per job on the dashboard; the default
# stays the signature ClipForge look (bold pop).
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CaptionPreset:
    key: str
    label: str
    font_size: int
    primary: str          # main text color (#RRGGBB)
    secondary: str        # accent (karaoke fill / pop color)
    outline_color: str
    outline_width: int
    shadow: int
    box: bool = False     # opaque box behind text (ASS BorderStyle=3)
    karaoke: bool = False # word-by-word fill (ASS \k) instead of cue-level pop
    pop: bool = False     # scale-pop animation per cue
    position_y: float = 0.78


CAPTION_PRESETS: dict[str, CaptionPreset] = {
    "bold_pop": CaptionPreset(
        key="bold_pop", label="Bold pop (default)",
        font_size=68, primary="#FFFFFF", secondary="#39FF14",
        outline_color="#39FF14", outline_width=5, shadow=1,
        pop=True, position_y=0.78,
    ),
    "karaoke": CaptionPreset(
        key="karaoke", label="Karaoke fill",
        font_size=64, primary="#FFFFFF", secondary="#39FF14",
        outline_color="#101010", outline_width=3, shadow=1,
        karaoke=True, position_y=0.78,
    ),
    "lower_third": CaptionPreset(
        key="lower_third", label="Clean lower-third",
        font_size=48, primary="#FFFFFF", secondary="#FFFFFF",
        outline_color="#000000", outline_width=2, shadow=2,
        pop=False, position_y=0.86,
    ),
    "boxed": CaptionPreset(
        key="boxed", label="Boxed",
        font_size=60, primary="#FFFFFF", secondary="#39FF14",
        outline_color="#000000", outline_width=0, shadow=0,
        box=True, pop=False, position_y=0.78,
    ),
}


def get_preset(key: str | None) -> CaptionPreset:
    """Resolve a preset by key, falling back to the default (never raises)."""
    if not key:
        return CAPTION_PRESETS["bold_pop"]
    return CAPTION_PRESETS.get(key, CAPTION_PRESETS["bold_pop"])


@dataclass
class Word:
    start: float
    end: float
    word: str

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "word": self.word}


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    duration: float = 0.0
    model_size: str = ""
    model_path: str = ""
    # Which engine produced this transcript, and how fast it ran
    # (RTF = wall-seconds / audio-seconds; lower is faster).
    engine: str = ""
    rtf: float | None = None


_MODEL_LOCK = threading.Lock()


def engine_status(settings: Settings) -> asr.EngineStatus:
    """The transcription engine that would run the next job (for health/doctor)."""
    return asr.status(settings)


def predownload_model(settings: Settings, on_progress=None, on_log=None) -> Transcript:
    """Load (and therefore download) the configured model at boot.

    Raises TranscribeError with a readable message on failure.
    """
    status = asr.status(settings)
    t0 = time.time()
    if on_log:
        on_log(f"Loading {status.engine} model '{settings.whisper.model}' ({status.compute_type})...")
    try:
        loaded = asr.load(settings, on_log=on_log)
    except asr.AsrError as exc:
        raise TranscribeError(str(exc)) from exc
    except Exception as exc:  # defensive: boot must never crash on a model error
        raise TranscribeError(f"Whisper model load failed: {exc}") from exc
    took = time.time() - t0
    cache_dir = asr.model_cache_dir()
    line = (
        f"{loaded.engine} model loaded: {loaded.model_name} on {loaded.device}/{loaded.compute_type} "
        f"from {cache_dir} in {took:.1f}s"
    )
    logger.info(f"[subtitle] {line}")
    if on_log:
        on_log(line)
    return Transcript(
        model_size=loaded.model_name,
        model_path=str(cache_dir),
        engine=loaded.engine,
    )


def _load_model(settings: Settings) -> asr.LoadedEngine:
    return asr.load(settings)


def _extract_audio(path: Path, dest: Path) -> Path:
    """Decode the WHOLE source to 16kHz mono WAV once (no per-chunk ffmpeg).

    Chunk slicing then happens in memory, so a 2-hour source costs one ffmpeg
    pass instead of one process spawn every 30 seconds.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(".wav.part")
    try:
        subprocess.run(
            [
                ffmpeg_bin(),
                "-y", "-loglevel", "error",
                "-i", str(path),
                "-vn", "-ac", "1", "-ar", "16000",
                # `-f wav` is REQUIRED, not cosmetic: the temp name ends in
                # `.part`, and ffmpeg infers the muxer from the file extension
                # when -f is absent -> "Unable to find a suitable output format
                # for 'source.wav.part': Invalid argument". Never rely on
                # extension inference when the output is a temp path.
                "-c:a", "pcm_s16le", "-f", "wav", str(tmp),
            ],
            check=True, capture_output=True, timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        raise TranscribeError(
            "audio extraction timed out after 60 minutes — the source may be corrupt"
        ) from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        tmp.unlink(missing_ok=True)
        raise TranscribeError(f"could not extract audio from {path.name}: {tail[0]}") from exc
    tmp.replace(dest)
    return dest


class _WavReader:
    """Random-access reader for a 16kHz mono PCM WAV.

    Only the requested span is materialised. Holding a whole 2-hour source as
    float32 would be ~250 MB, which does not fit a small container alongside a
    whisper model — the decoder gets OOM-killed and the job "hangs" — so the
    audio is paged in one resume chunk at a time (~19 MB for 5 minutes).
    """

    def __init__(self, path: Path) -> None:
        import wave  # noqa: PLC0415

        self._wave = wave.open(str(path), "rb")
        if self._wave.getnchannels() != 1 or self._wave.getframerate() != 16000:
            rate, channels = self._wave.getframerate(), self._wave.getnchannels()
            self._wave.close()
            raise TranscribeError(
                f"expected 16kHz mono audio, got {rate}Hz/{channels}ch"
            )
        self.rate = 16000
        self.frames = self._wave.getnframes()

    def samples(self, start_sec: float, end_sec: float):
        """float32 samples in [-1, 1] for [start_sec, end_sec)."""
        import numpy as np  # noqa: PLC0415

        first = max(0, int(round(start_sec * self.rate)))
        last = min(self.frames, int(round(end_sec * self.rate)))
        if last <= first:
            return np.zeros(0, dtype=np.float32)
        self._wave.setpos(first)
        raw = self._wave.readframes(last - first)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def close(self) -> None:
        try:
            self._wave.close()
        except Exception:
            pass

    def __enter__(self) -> "_WavReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _dedupe_words(words: list[Word], overlap: float = OVERLAP_SEC) -> list[Word]:
    """Drop duplicated words from overlapping windows.

    A word from a later window is dropped when it closely matches one already
    kept: same normalized text with start times within `overlap` seconds, OR
    exact-timestamp twins (VAD padding can re-cut a word at the same spot).
    Genuinely repeated speech at different times is preserved.
    """
    def _norm(word: str) -> str:
        return word.lower().strip(".,!?;:\"'")

    kept: list[Word] = []
    for w in words:
        dup = False
        for k in reversed(kept):
            if w.start - k.start > overlap + 1.0:
                break  # ordered by time; nothing further can match
            same_text = _norm(w.word) == _norm(k.word)
            near = abs(w.start - k.start) <= overlap
            if same_text and near:
                dup = True
                break
        if not dup:
            kept.append(w)
    return kept


def _probe_duration(path: Path) -> float:
    """Probe media duration in seconds via ffprobe.

    Used to plan decode windows before transcription starts. Raises
    TranscribeError with a readable message instead of crashing the job when
    the file is missing or unreadable.
    """
    if not path.exists():
        raise TranscribeError(f"cannot probe duration — source file not found: {path}")
    try:
        proc = subprocess.run(
            [
                ffprobe_bin(), "-v", "error",
                "-print_format", "json",
                "-show_format",
                str(path),
            ],
            capture_output=True, text=True, timeout=60, check=True,
        )
    except FileNotFoundError as exc:
        raise TranscribeError(f"ffprobe not found — install ffmpeg or set FFPROBE_PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscribeError(f"ffprobe timed out reading {path.name} after 60s") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip().splitlines()
        tail = stderr[-1] if stderr else f"exit code {exc.returncode}"
        raise TranscribeError(f"ffprobe failed on {path.name}: {tail}") from exc
    except OSError as exc:
        raise TranscribeError(f"could not run ffprobe on {path.name}: {exc}") from exc
    try:
        meta = json.loads(proc.stdout)
        duration = float(meta.get("format", {}).get("duration", 0) or 0)
    except (ValueError, TypeError) as exc:
        raise TranscribeError(f"ffprobe returned no readable duration for {path.name}") from exc
    if duration <= 0:
        raise TranscribeError(
            f"source file {path.name} has no playable duration — the download is corrupt"
        )
    return duration


def _plan_windows(duration: float) -> list[tuple[float, float]]:
    """Plan the [start, end) resume chunks covering [0, duration).

    These are RESUME units, not decode units: the engine batches its own 30s
    windows inside each chunk. Short files are always one chunk.
    """
    if duration <= MAX_DECODE_SEC:
        return [(0.0, max(duration, 0.5))]
    windows: list[tuple[float, float]] = []
    step = WINDOW_SEC - OVERLAP_SEC
    t = 0.0
    while t < duration - 0.25:
        end = min(t + WINDOW_SEC, duration)
        windows.append((t, end))
        t += step
    if not windows:
        windows.append((0.0, max(duration, 0.5)))
    return windows


def transcribe(
    path: Path,
    settings: Settings,
    on_progress=None,
    cache_dir: Path | None = None,
    should_pause=None,
) -> Transcript:
    """Word-level transcription through the configured ASR engine.

    Engine: WhisperX when its torch stack is installed, otherwise
    faster-whisper's batched CTranslate2 pipeline (see app.asr). Both decode
    with condition_on_previous_text=False, compression_ratio_threshold=2.2,
    no_speech_threshold=0.6 and Silero VAD, so the repetition hardening is kept
    by the swap rather than traded away for speed.

    The source is decoded to audio ONCE, then split into overlapping resume
    chunks in memory: no ffmpeg process per chunk, and the engine batches its
    own 30s windows inside each chunk.

    `cache_dir` enables INCREMENTAL resume: each completed chunk is written to
    cache_dir/chunk_XXXXX.json as it decodes, so a retry after a crash/stall
    skips the chunks already done (at most one chunk of work is lost).
    """
    loaded = _load_model(settings)
    duration = _probe_duration(path)
    language = None if settings.whisper.language == "auto" else settings.whisper.language
    sample_rate = 16000

    windows = _plan_windows(duration)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if len(windows) == 1:
        logger.info(
            f"[subtitle] transcribing {duration:.1f}s with {loaded.engine} "
            f"({loaded.device}/{loaded.compute_type}, batch {loaded.batch_size}) — one chunk"
        )
    else:
        logger.info(
            f"[subtitle] transcribing {duration:.1f}s with {loaded.engine} in {len(windows)} "
            f"~{CHUNK_SEC / 60:.0f} min chunks (overlap {OVERLAP_SEC:.1f}s, "
            f"batch {loaded.batch_size}) — resumable per chunk"
        )

    # ---- decode the whole source to 16kHz mono WAV once --------------------
    # Cached next to the chunk caches so a resume does not pay for it again.
    if cache_dir is not None:
        wav_path = cache_dir / "source.wav"
    else:
        wav_path = Path(settings.app.paths.data_dir) / f"asr-{time.time_ns()}.wav"
    try:
        _extract_audio(path, wav_path)
        reader = _WavReader(wav_path)
    except TranscribeError:
        raise
    except Exception as exc:
        raise TranscribeError(f"Could not prepare audio for transcription: {exc}") from exc

    all_words: list[Word] = []
    resumed = 0
    decoded_seconds = 0.0
    decode_wall = 0.0
    try:
        for idx, (w_start, w_end) in enumerate(windows):
            # Pause between transcript chunks, never mid-decode. Every finished
            # chunk is already cached below, so a resume reuses them and only
            # decodes what is left.
            if should_pause is not None and should_pause():
                raise TranscriptionPaused(
                    f"paused after chunk {idx}/{len(windows)} — "
                    f"{resumed} chunk(s) already cached and will be reused"
                )
            if on_progress and duration > 0:
                on_progress(min(1.0, w_start / duration))

            # ---- incremental resume: reuse an already-decoded chunk --------
            cache_file = None
            if cache_dir is not None:
                cache_file = cache_dir / f"chunk_{idx:05d}.json"
                if cache_file.exists():
                    try:
                        data = json.loads(cache_file.read_text(encoding="utf-8"))
                        if abs(float(data.get("w_start", -1)) - w_start) < 0.01:
                            all_words.extend(Word(**w) for w in data.get("words", []))
                            resumed += 1
                            logger.info(
                                f"[subtitle] resume: chunk {idx + 1}/{len(windows)} reused from cache"
                            )
                            if on_progress and duration > 0:
                                on_progress(min(1.0, w_end / duration))
                            continue
                    except Exception as exc:
                        logger.warning(f"[subtitle] unreadable chunk cache {cache_file.name}: {exc}")

            # page in ONLY this chunk (~19 MB for 5 minutes), never the whole file
            audio = reader.samples(w_start, w_end)
            if audio.size < sample_rate // 4:  # under 0.25s — nothing to decode
                if on_progress and duration > 0:
                    on_progress(min(1.0, w_end / duration))
                continue

            def on_segment(local_end: float, _w_start=w_start) -> None:
                # intra-chunk liveness: chunk-local segment end → absolute time
                if on_progress and duration > 0:
                    on_progress(min(1.0, (_w_start + local_end) / duration))

            t0 = time.time()
            try:
                words, _detected = loaded.asr_words(audio, language, on_segment=on_segment)
            except Exception as exc:
                raise TranscribeError(
                    f"{loaded.engine} decode failed on chunk {idx + 1}/{len(windows)} "
                    f"({w_start:.0f}-{w_end:.0f}s): {exc}"
                ) from exc
            chunk_wall = time.time() - t0
            chunk_audio = audio.size / sample_rate
            decoded_seconds += chunk_audio
            decode_wall += chunk_wall
            logger.info(
                f"[subtitle] chunk {idx + 1}/{len(windows)}: {chunk_audio:.0f}s audio in "
                f"{chunk_wall:.1f}s (RTF {chunk_wall / max(0.001, chunk_audio):.3f})"
            )

            # chunk-local word timings → source-absolute time
            abs_words = [
                Word(start=w.start + w_start, end=w.end + w_start, word=w.word)
                for w in words
            ]
            all_words.extend(abs_words)

            # persist the chunk BEFORE moving on (incremental resume)
            if cache_file is not None:
                tmp = cache_file.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(
                        {"w_start": w_start, "w_end": w_end,
                         "words": [w.as_dict() for w in abs_words]},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                tmp.replace(cache_file)

            if on_progress and duration > 0:
                on_progress(min(1.0, w_end / duration))
    finally:
        reader.close()
        del reader
        # the extracted audio is only needed while this call runs; the chunk
        # caches (not the WAV) are what a resume reuses
        if cache_dir is None:
            wav_path.unlink(missing_ok=True)

    if resumed:
        logger.info(f"[subtitle] resumed: reused {resumed}/{len(windows)} cached chunk(s)")

    words_out = _dedupe_words(all_words)
    dropped = len(all_words) - len(words_out)
    if dropped > 0:
        logger.info(f"[subtitle] overlap dedup removed {dropped} duplicate words")

    rtf = (decode_wall / decoded_seconds) if decoded_seconds > 0 else None
    # Record the model that ACTUALLY decoded this audio, not the configured
    # one. hostlimits downgrades a model that cannot fit the container, so
    # labelling a `small` run as `medium` makes the stored transcript lie
    # about which model produced it (and about RTF, which is per-model).
    ran_model = loaded.model_name or settings.whisper.model
    if ran_model != settings.whisper.model:
        logger.info(
            f"[subtitle] configured model '{settings.whisper.model}' was downgraded to "
            f"'{ran_model}' for this host"
        )
    transcript = Transcript(
        words=words_out,
        duration=duration,
        model_size=ran_model,
        engine=loaded.engine,
        rtf=round(rtf, 4) if rtf is not None else None,
    )
    logger.info(
        f"[subtitle] transcribed {len(words_out)} words across {len(windows)} chunk(s), "
        f"{transcript.duration:.1f}s of audio, model '{ran_model}'"
        + (f", RTF {rtf:.3f} ({loaded.engine})" if rtf is not None else "")
    )
    return transcript


def _ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _hex_to_ass(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H00{b}{g}{r}".upper()


def words_to_cues(
    words: list[dict],
    *,
    words_per_cue: int = 2,
    max_chars: int = 24,
    max_duration: float = 2.2,
) -> list[dict]:
    """Group word timings into caption cues of ~2-3 words."""
    cues: list[dict] = []
    current: list[dict] = []
    for w in words:
        current.append(w)
        text = " ".join(x["word"] for x in current)
        dur = w["end"] - current[0]["start"]
        if len(current) >= words_per_cue or len(text) >= max_chars or dur >= max_duration:
            cues.append({
                "start": current[0]["start"],
                "end": w["end"],
                "text": text,
                "words": list(current),
            })
            current = []
    if current:
        cues.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": " ".join(x["word"] for x in current),
            "words": list(current),
        })
    return cues


def build_ass_header(
    settings: Settings, width: int, height: int, preset_key: str | None = None
) -> str:
    p = get_preset(preset_key)
    font = settings.subtitle.font
    primary = _hex_to_ass(p.primary)
    secondary = _hex_to_ass(p.secondary)
    outline = _hex_to_ass(p.outline_color)
    # BorderStyle 3 = opaque box (ASS renders BackColour behind the text)
    border_style = 3 if p.box else 1
    back = "&HA0000000" if p.box else "&H80000000"
    pos = p.position_y if preset_key else settings.subtitle.position_y
    return f"""[Script Info]
Title: ClipForge Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ClipForge,{font},{p.font_size},{primary},{secondary},{outline},{back},-1,0,0,0,100,100,0,0,{border_style},{p.outline_width},{p.shadow},2,60,60,{int(height * (1 - pos))},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _karaoke_tags(cue: dict) -> str:
    """ASS \\k timing tags (centiseconds per word) for karaoke-fill style."""
    words = cue.get("words") or []
    if not words:
        return ""
    total = max(0.01, cue["end"] - cue["start"])
    parts: list[str] = []
    for w in words:
        dur = max(0.0, w["end"] - w["start"])
        cs = int(round(dur / total * 100 * len(words)))
        parts.append(r"{\k%d}%s" % (max(1, cs), w["text"]))
    return "".join(parts)


def build_ass_events(
    cues: list[dict], settings: Settings, start_offset: float = 0.0,
    preset_key: str | None = None,
) -> str:
    p = get_preset(preset_key)
    lines: list[str] = []
    for c in cues:
        start = _ass_time(c["start"] - start_offset)
        end = _ass_time(c["end"] - start_offset)
        if p.karaoke and c.get("words"):
            text = _karaoke_tags(c)
        else:
            text = c["text"].replace("{", "(").replace("}", ")")
            if p.pop:
                text = (
                    r"{\fad(60,40)\fscx84\fscy84\t(0,90,\fscx105\fscy105)"
                    r"\t(90,140,\fscx100\fscy100)}" + text
                )
        lines.append(f"Dialogue: 0,{start},{end},ClipForge,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def write_ass(
    cues: list[dict],
    settings: Settings,
    path: Path,
    width: int,
    height: int,
    start_offset: float = 0.0,
    preset_key: str | None = None,
) -> Path:
    """Write an .ass subtitle file; returns the path."""
    content = build_ass_header(settings, width, height, preset_key) + build_ass_events(
        cues, settings, start_offset, preset_key
    )
    path.write_text(content, encoding="utf-8")
    return path


def imagemagick_available() -> bool:
    from shutil import which

    return which("magick") is not None or which("convert") is not None


def render_word_pngs_pillow(
    cues: list[dict],
    settings: Settings,
    out_dir: Path,
    width: int,
    height: int,
    fps: int,
) -> dict[str, str]:
    """Pillow fallback: render each cue as a transparent PNG overlay.

    Returns {cue_key: png_path}. Used when ImageMagick is unavailable so
    libass can still burn captions via ffmpeg's overlay filter instead of
    subtitles filter (which would need libass with ImageMagick fontconfig).
    """
    from PIL import Image, ImageDraw, ImageFont

    out_dir.mkdir(parents=True, exist_ok=True)
    font_size = settings.subtitle.font_size
    outline = settings.subtitle.outline_color
    paths: dict[str, str] = {}
    font = None
    for name in [settings.subtitle.font, *settings.subtitle.font_fallbacks]:
        try:
            font = ImageFont.truetype(f"{name}.ttf", font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    for i, cue in enumerate(cues):
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        text = cue["text"].upper()
        try:
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=settings.subtitle.outline_width)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textsize(text, font=font)
        x = (width - tw) // 2
        y = int(height * settings.subtitle.position_y) - th
        draw.text(
            (x, y), text, font=font, fill="white",
            stroke_width=settings.subtitle.outline_width,
            stroke_fill=outline,
        )
        p = out_dir / f"cue_{i:05d}.png"
        img.save(p)
        paths[f"{cue['start']:.3f}-{cue['end']:.3f}"] = str(p)
    return paths
