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

import hashlib
import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
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
    # Progressive word-by-word highlight. Same ASS mechanism as `karaoke` (the
    # sung part takes SecondaryColour) but the default style also keeps its
    # scale-pop entrance, which is the look Opus-style captions ship with:
    # short bursts, the word being spoken lit up as it is spoken.
    highlight: bool = False
    position_y: float = 0.78
    # Per-chunk 1-2 keyword emphasis with rotating accent colours (see
    # EMPHASIS_COLORS below). Not every cue gets a highlight; the choice
    # and the colours are decided by the emphasis planner, not by the ASS
    # writer.
    word_emphasis: bool = False
    # Display text in UPPERCASE. Spanish accents and inverted punctuation
    # survive uppercasing because it is done with str.upper() — Unicode
    # aware ("é".upper() -> "É", "¿" stays "¿", "ñ" -> "Ñ").
    uppercase: bool = False
    # Per-preset font override: some looks are font-specific (the
    # Ssemble-style preset is a Montserrat ExtraBold look; the others keep
    # the configured family). ``None`` inherits [subtitle] font.
    font_override: str | None = None
    fonts_dir_override: str | None = None
    max_chars_per_line: int = 0  # 0 = no per-line cap (words wrap naturally)


EMPHASIS_COLORS: tuple[str, ...] = ("#1FE01F", "#FFF000", "#FF1A1A")


CAPTION_PRESETS: dict[str, CaptionPreset] = {
    "viral": CaptionPreset(
        key="viral", label="Ssemble-style viral",
        font_size=70, primary="#FFFFFF", secondary="#1FE01F",
        outline_color="#0A0A0A", outline_width=7, shadow=1,
        font_override="Montserrat ExtraBold",
        fonts_dir_override="fonts",
        pop=True, word_emphasis=True, position_y=0.71,
        uppercase=True, max_chars_per_line=18,
    ),
    "bold_pop": CaptionPreset(
        key="bold_pop", label="Bold pop (default)",
        font_size=68, primary="#FFFFFF", secondary="#39FF14",
        outline_color="#39FF14", outline_width=5, shadow=1,
        pop=True, highlight=True, position_y=0.50,
    ),
    "viral": CaptionPreset(
        key="viral", label="Ssemble-style viral",
        # Requested: ALL CAPS, extra-bold sans-serif (Montserrat Black / Impact
        # style), 90-100px, white text with a thick black outline (8-10px) and
        # a soft shadow, NO black box or background behind the text.
        #   font_size=96, primary="#FFFFFF", secondary="#FFE600",
        #   outline_color="#000000", outline_width=8, shadow=2,
        #   box=False, karaoke=False, pop=True, highlight=True,
        #   uppercase=True, font_override="Montserrat Black",
        #   fonts_dir_override="fonts",
        #   max_chars_per_line=0,
        font_size=96, primary="#FFFFFF", secondary="#FFE600",
        outline_color="#000000", outline_width=8, shadow=2,
        # Lower-center, about 65-70% down the screen. ASS MarginV is a
        # fraction of the FRAME height measured DOWN from the bottom, so a
        # caption that must sit at 65-70% from the top (30-35% from the
        # bottom) uses MarginV ~ 0.31-0.35 — never in the bottom 20%.
        position_y=0.67,
        # Ssemble-style viral: short bursts 2-4 words, max 2 lines, centred,
        # never one word at a time. Break at natural pauses (the subtitle
        # pipeline's pause_sec cut) not mid-phrase.
        box=False, karaoke=False, pop=True, highlight=True,
        uppercase=True, font_override="Montserrat Black",
        fonts_dir_override="fonts",
        max_chars_per_line=0,
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


def get_preset(key: str | None, highlight_color: str | None = None) -> CaptionPreset:
    """Resolve a preset by key, falling back to the default (never raises).

    `highlight_color` replaces the preset's own accent — the colour the word
    being spoken is painted in. Empty or absent keeps the preset's accent, so
    picking "Karaoke fill" still looks like karaoke fill and a preset switch
    does not quietly inherit a colour chosen for a different one.
    """
    preset = CAPTION_PRESETS.get(key or "", CAPTION_PRESETS["vertical_916"])
    color = (highlight_color or "").strip()
    if not color or color == preset.secondary:
        return preset
    return replace(preset, secondary=color)


@dataclass
class Word:
    start: float
    end: float
    word: str
    # The model's own confidence in this token (0..1). Absent on transcripts
    # written before this field existed, which is why every consumer treats a
    # missing value as 1.0 — "never measured" must not silently become
    # "measured and certain".
    conf: float = 1.0

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "word": self.word,
            "conf": round(float(self.conf), 3),
        }


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
    # ISO code the engine detected or the job forced ("en", "es"...). Empty on
    # transcripts written before this field existed — every reader treats ""
    # as "unknown, guess from the words" (translate.guess_language).
    language: str = ""


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


# Per-chunk retry policy. Three attempts with exponential backoff, then one
# attempt on a smaller model. Constants (not literals) so the test suite can
# assert the policy and a deployment can reason about the worst-case cost of a
# chunk: 3 + 1 attempts and 1+2+4 = 7s of backoff.
CHUNK_ATTEMPTS = 3
CHUNK_BACKOFF_SEC = 1.0
# Ordered smallest first. `tiny` fits anywhere; `base` is the smallest model
# that is still worth shipping, so it is tried before `tiny`.
_FALLBACK_MODELS = ("base.en", "base", "tiny.en", "tiny")


def _load_smaller_model(settings: Settings, current: asr.LoadedEngine) -> asr.LoadedEngine:
    """Load a smaller Whisper model for one stubborn chunk.

    Returns the CURRENT engine when nothing smaller is available, so the
    caller can tell "we retried with a genuinely smaller model" from "we had
    nothing to fall back to".
    """
    from dataclasses import replace as _replace

    for name in _FALLBACK_MODELS:
        if name == current.model_name:
            continue
        try:
            smaller = _replace(settings.whisper, model=name)
            engine = asr.load(_replace(settings, whisper=smaller))
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            logger.warning(f"[subtitle] fallback model '{name}' unavailable: {exc}")
            continue
        logger.warning(
            f"[subtitle] falling back to the smaller model '{name}' for one chunk "
            f"(was '{current.model_name}')"
        )
        return engine
    logger.warning(
        f"[subtitle] no smaller model available than '{current.model_name}' — "
        f"retrying with the same model"
    )
    return current


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
    on_chunk=None,
    language: str | None = None,
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

    `on_chunk(done, total, *, phase, resuming)` is the caller's progress hook
    for CHUNK-granular reporting. It fires when a chunk STARTS and when one
    FINISHES, which is what makes the dashboard show "Transcribing chunk 4 of
    11" instead of a percentage that sits at 0% until the first chunk lands.
    `on_progress` (fractions of audio) is still called, but it is a secondary,
    smoother signal — the two together give a number that never stalls at zero.
    """
    loaded = _load_model(settings)
    duration = _probe_duration(path)
    # Explicit call-site language (per-job ENG/ESP selection) wins over the
    # global config; None falls back to the config's own auto/explicit choice.
    if language:
        language = str(language).strip().lower() or None
    else:
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
    # Language pinning (TARGET C): None means "auto" — the first chunk lets the
    # engine detect, then every later chunk receives the DETECTED language
    # explicitly instead of re-detecting. `language` given by the caller means
    # the job forced a language and detection never runs.
    detect_pinned: str | None = None if language is None else "en-forced"
    detected_language: str | None = None
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
            # Announce the chunk BEFORE decoding it. The old code only moved
            # the number once a chunk finished, so a 54-minute source showed
            # "Transcribing 0%" for the whole first chunk (~90s here, longer on
            # a busier host) and read as a hang.
            if on_chunk is not None:
                on_chunk(idx, len(windows), phase="start")

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
                            if on_chunk is not None:
                                on_chunk(idx, len(windows), phase="done", resuming=True)
                            continue
                    except Exception as exc:
                        logger.warning(f"[subtitle] unreadable chunk cache {cache_file.name}: {exc}")

            # page in ONLY this chunk (~19 MB for 5 minutes), never the whole file
            audio = reader.samples(w_start, w_end)
            if audio.size < sample_rate // 4:  # under 0.25s — nothing to decode
                if on_progress and duration > 0:
                    on_progress(min(1.0, w_end / duration))
                if on_chunk is not None:
                    on_chunk(idx, len(windows), phase="done")
                continue

            def on_segment(local_end: float, _w_start=w_start) -> None:
                # intra-chunk liveness: chunk-local segment end → absolute time
                if on_progress and duration > 0:
                    on_progress(min(1.0, (_w_start + local_end) / duration))

            t0 = time.time()
            words = None
            # A chunk is retried on its own before the whole job is failed.
            # Whisper on a shared, memory-constrained box fails transiently
            # far more often than it fails permanently: a cgroup reclaim, a
            # ctranslate2 thread that could not allocate, an interrupted read.
            # After CHUNK_ATTEMPTS the chunk falls back to a SMALLER model,
            # which needs a fraction of the memory and still produces correct
            # words with correct timings -- a slightly worse transcript beats
            # a failed batch, and the offsets are computed from w_start either
            # way so a fallback cannot desynchronise the captions.
            for attempt in range(1, CHUNK_ATTEMPTS + 2):
                engine = loaded
                if attempt > CHUNK_ATTEMPTS:
                    engine = _load_smaller_model(settings, loaded)
                try:
                    # 'auto' pins NOTHING on the first chunk: the engine is
                    # allowed to detect, and whatever it detected is then
                    # PASSED EXPLICITLY to every later chunk — Whisper's own
                    # per-chunk redetection is what breaks a mixed stream.
                    # An explicitly configured/forced language always wins.
                    # (detect_pinned is the sentinel "en-forced" when the
                    # caller forced a language, so the truthy test below keeps
                    # chunks honest in both modes.)
                    chunk_lang = language or (detect_pinned if detect_pinned != "en-forced" else None)
                    words, _detected = engine.asr_words(
                        audio, chunk_lang, on_segment=on_segment,
                    )
                    if _detected and not language and detect_pinned is None:
                        # first auto chunk: pin what the engine heard
                        detect_pinned = _detected
                        detected_language = _detected
                        logger.info(
                            f"[subtitle] engine detected language=\"{_detected}\" "
                            "on the first chunk — pinning it for the rest"
                        )
                    if attempt > 1:
                        logger.warning(
                            f"[subtitle] chunk {idx + 1}/{len(windows)} decoded on "
                            f"attempt {attempt} using '{engine.model_name}'"
                        )
                    break
                except Exception as exc:  # noqa: BLE001 - retried, then escalated
                    if attempt > CHUNK_ATTEMPTS and engine is loaded:
                        # even the small model failed on the last attempt
                        raise TranscribeError(
                            f"{loaded.engine} decode failed on chunk "
                            f"{idx + 1}/{len(windows)} ({w_start:.0f}-{w_end:.0f}s) "
                            f"{CHUNK_ATTEMPTS} times and the smaller fallback model "
                            f"also failed: {exc}"
                        ) from exc
                    if attempt <= CHUNK_ATTEMPTS:
                        back = CHUNK_BACKOFF_SEC * (2 ** (attempt - 1))
                        logger.warning(
                            f"[subtitle] chunk {idx + 1}/{len(windows)} attempt "
                            f"{attempt}/{CHUNK_ATTEMPTS} failed ({exc}) — retrying "
                            f"in {back:.0f}s"
                        )
                        if on_progress:
                            on_progress(min(1.0, w_start / duration))
                        time.sleep(back)
            if words is None:
                raise TranscribeError(
                    f"{loaded.engine} decode failed on chunk {idx + 1}/{len(windows)} "
                    f"({w_start:.0f}-{w_end:.0f}s)"
                )
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
            if on_chunk is not None:
                on_chunk(idx, len(windows), phase="done")
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
        language=language or detected_language or "",
    )
    logger.info(
        f"[subtitle] transcribed {len(words_out)} words across {len(windows)} chunk(s), "
        f"{transcript.duration:.1f}s of audio, model '{ran_model}'"
        + (f", RTF {rtf:.3f} ({loaded.engine})" if rtf is not None else "")
        + (f", language '{transcript.language}'" if transcript.language else ""),
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


#: Mean per-word confidence below which a cue is treated as a GUESS rather
#: than a transcription, and the text is replaced instead of shown.
#:
#: Calibrated on this app's own batches (tools/uncertainty_demo.py), and on the
#: CUE rather than the segment, because the cue is the unit the rule decides
#: on. Measured cue-mean word probability on two real windows:
#:
#:   clear single talker    median 0.87   (only 2 of 24 cues below 0.45, and
#:                                      both were genuinely truncated: "it
#:                                      like", "ain't...")
#:   music + rap + overlap  median 0.66   (cues at 0.16-0.44 were the
#:                                      "What what what what Charlotte
#:                                      Wayne's do I'm like wait wait what
#:                                      code" stretch; correct text in the
#:                                      same clip sat at 0.47-0.60)
#:
#: 0.45 sits in the gap between those two groups. It was deliberately NOT set
#: at 0.60: measured on per-SEGMENT means that looked safe, but at cue level
#: 0.60 replaced 44% of a clip's captions, most of them correct — the rule has
#: to be as conservative as its worst false positive, not as bold as its best
#: true one.
LOW_CONFIDENCE = 0.45

#: Below this the output is not "unclear speech", it is noise the model
#: invented; a different placeholder says so honestly.
INAUDIBLE_BELOW = 0.30

PLACEHOLDER_MUTTERING = "[muttering]"
PLACEHOLDER_INAUDIBLE = "[inaudible]"


def cue_confidence(words: list[dict]) -> float | None:
    """Mean confidence of a cue's words, or None if it was never measured.

    The MEAN, not the minimum: a single 0.45 token inside an otherwise clean
    sentence is a common artifact (a name, a number) and blanking the whole cue
    for it would be worse than the doubt. Averaging asks the question the
    viewer is actually asking — is this line intelligible at all?
    """
    vals = [float(w.get("conf")) for w in words if w.get("conf") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _placeholder_for(conf: float, inaudible_threshold: float = INAUDIBLE_BELOW) -> str:
    return PLACEHOLDER_INAUDIBLE if conf < inaudible_threshold else PLACEHOLDER_MUTTERING


# --------------------------------------------------------------------------- #
# Filler words and stutters
#
# These are removed from the DISPLAYED TEXT only. The audio is never touched:
# a clip is a straight cut of the source, so dropping a spoken "um" from the
# soundtrack would need a time-stretch that pulls every later word out of sync
# with the lips. A caption that says "so I was like yeah" while the speaker
# says "so I was um like yeah" is what every tool in this space does, and it
# costs nothing in lip-sync.
#
# Deliberately conservative: "like", "so", "just" and "well" are real words
# far more often than fillers, and blanking them turns real speech into
# nonsense. Only unambiguous vocal fillers are in the list.
# --------------------------------------------------------------------------- #

FILLER_WORDS = frozenset({
    "um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "eh", "hmm", "hm",
    "mhm", "mm", "mmh", "uhhuh", "huh", "huhhuh", "ummm", "hmmmm",
    # Spanish vocal fillers heard on Caribbean/US LatAm speech (same rule as
    # the English ones: unambiguous noise only, real words never blanked —
    # notably NOT "pues", which carries meaning).
    "ehh", "ahh", "ehm", "mmm", "mmmh",
})

#: Two identical words count as a stutter only if this little time separates
#: them. "the the" is one; "very ... very good" with a beat between is emphasis
#: and must survive.
STUTTER_GAP_SEC = 0.15


def _norm_token(word: str) -> str:
    """Lowercase, strip punctuation and the possessive — for filler matching only."""
    return word.strip().strip(".,!?;:'\"()[]…").lower()


def is_filler(word: str) -> bool:
    return _norm_token(word) in FILLER_WORDS


# --------------------------------------------------------------------------- #
# Bilingual caption fixes (English + Spanish)
#
# Whisper's known failure modes on this content are the same in both languages:
#   * numbers split into many words ("tres mil millones", "one point five")
#   * contractions broken into two words by the tokenizer (English "isn't" ->
#     "is n't", "gon na", "wan na")
#
# Both fixes are TIMING-preserving: the words are merged into one visual word
# and the merged span is the union of the spans, so caption sync never drifts.
# Spanish keeps every accent and inverted mark (¿ ¡) intact throughout — none
# of these rules ever normalise or strip characters.
# --------------------------------------------------------------------------- #

#: Number words that merge with their neighbour when they form one number.
#: "tres mil millones" -> one number phrase, not three captions.
#: Purely additive, never subtractive: all three words stay on screen, they are
#: simply shown as one longer burst rather than three separate ones.
_ES_NUMBER_WORDS = frozenset({
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho",
    "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "veinte",
    "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta",
    "noventa", "cien", "ciento", "doscientos", "trescientos", "quinientos",
    "mil", "millón", "millon", "millones", "mil millones", "medio", "media",
})

_EN_NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "trillion", "point", "half", "double", "triple",
})

#: English tokenizer splits that Whisper commonly emits, merged back.
#: Keys are the split pair in sequence; the first word ends the lookbehind.
#:
#: This is DEBATEABLE on tod-day speech only when the pair arrives as separate
#: Whisper tokens, which is the split case, not genuine speech — genuine speech
#: arrives as one token ("isn't"), so the merge never touches it.
EN_CONTRACTIONS = {
    ("is", "n't"): "isn't",    ("are", "n't"): "aren't",
    ("was", "n't"): "wasn't",
    ("were", "n't"): "weren't",
    ("does", "n't"): "doesn't",
    ("did", "n't"): "didn't",
    ("do", "n't"): "don't",
    ("wo", "n't"): "won't",
    ("ca", "n't"): "can't",
    ("could", "n't"): "couldn't",
    ("would", "n't"): "wouldn't",
    ("should", "n't"): "shouldn't",
    ("has", "n't"): "hasn't",
    ("have", "n't"): "haven't",
    ("had", "n't"): "hadn't",
    ("gon", "na"): "gonna",
    ("wan", "na"): "wanna",
    ("got", "ta"): "gotta",
    ("kind", "of"): "kinda",
    ("lot", "of"): "lotta",
    ("out", "of"): "outta",
}


def merge_broken_contractions(
    words: list[dict],
) -> tuple[list[dict], int]:
    """Merge Whisper-split English contractions back into one word.

    "is n't" -> "isn't". The merged word keeps the UNION of the two time
    spans, so the karaoke highlight for the word covers the whole spoken
    contraction rather than only its first half.

    Never merges a pair that is genuinely separated in time or across a
    detected pause (a gap larger than a third of a second means the "is" and
    the "n't" came from different speech), and never merges across removed
    filler words.
    """
    out: list[dict] = []
    merged = 0
    i = 0
    while i < len(words):
        w = words[i]
        w0 = _norm_token(str(w.get("word", "")))
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is not None:
            w1 = _norm_token(str(nxt.get("word", "")))
            gap = float(nxt.get("start") or 0.0) - float(w.get("end") or 0.0)
            joined = EN_CONTRACTIONS.get((w0, w1))
            if joined and gap <= 0.33:
                out.append({
                    **w,
                    "word": joined,
                    "end": max(float(w.get("end") or 0.0), float(nxt.get("end") or 0.0)),
                    "conf": min(float(w.get("conf") or 1.0), float(nxt.get("conf") or 1.0)),
                })
                merged += 1
                i += 2
                continue
        out.append(w)
        i += 1
    return out, merged


def merge_number_runs(
    words: list[dict],
    language: str | None = None,
) -> tuple[list[dict], int]:
    """Glue spoken-out numbers into runs of words grouped per caption.

    "tres mil millones" spoken across three Whisper tokens is ONE number; the
    three tokens usually land in two different captions, which is how a
    caption showing a bare "mil" reads like a mistake. Merging the run into a
    single visual word (union span, single highlight) puts it together.

    The merge NEVER spans a pause (gap > 0.4s): a number followed by a breath
    and new speech is not one phrase.

    Runs are bounded: at most four consecutive number-words become one visual
    word, so a pathological run of digits/number-words cannot build a caption
    so wide it overflows the frame.
    """
    es = _ES_NUMBER_WORDS if (language or "").startswith("es") else _EN_NUMBER_WORDS
    out: list[dict] = []
    merged = 0
    i = 0
    while i < len(words):
        w = words[i]
        w0 = _norm_token(str(w.get("word", "")))
        if w0 not in es:
            out.append(w)
            i += 1
            continue
        run: list[dict] = [w]
        j = i + 1
        while (
            j < len(words)
            and len(run) < 4
            and _norm_token(str(words[j].get("word", ""))) in es
        ):
            gap = float(words[j].get("start") or 0.0) - float(run[-1].get("end") or 0.0)
            if gap > 0.4:
                break
            run.append(words[j])
            j += 1
        if len(run) > 1:
            joined = " ".join(str(x.get("word", "")) for x in run)
            out.append({
                **run[0],
                "word": joined,
                "end": max(float(x.get("end") or 0.0) for x in run),
                "conf": min(float(x.get("conf") or 1.0) for x in run),
            })
            merged += len(run) - 1
            i = j
        else:
            out.append(w)
            i += 1
    return out, merged


def strip_fillers(
    words: list[dict], *, drop_stutters: bool = True
) -> tuple[list[dict], int]:
    """Return (kept words, how many were removed) for caption display.

    Two passes:
      1. unambiguous fillers ("um", "uh", "hmm") — they carry no meaning and
         Opus-style captions drop them;
      2. an immediately repeated identical word ("the the", "I I mean") — a
         stutter, dropped once. Only ADJACENT repeats count, so a genuine
         "very very good" spread across a pause survives.
    """
    kept: list[dict] = []
    removed = 0
    for w in words:
        if is_filler(str(w.get("word", ""))):
            removed += 1
            continue
        if drop_stutters and kept:
            prev = _norm_token(str(kept[-1].get("word", "")))
            gap = float(w.get("start") or 0.0) - float(kept[-1].get("end") or 0.0)
            # A stutter is a RAPID repetition. Requiring the two to be adjacent
            # in TIME as well as in the list is what stops "very ... very good"
            # (deliberate emphasis, spoken with a beat between) from losing a
            # word — collapsing on adjacency alone silently deletes real speech.
            if prev and prev == _norm_token(str(w.get("word", ""))) and gap < STUTTER_GAP_SEC:
                # Keep the LONGER timing of the pair so the highlight does not
                # jump backwards if the stutter was held longer.
                if float(w.get("end") or 0) > float(kept[-1].get("end") or 0):
                    kept[-1]["end"] = w["end"]
                removed += 1
                continue
        kept.append(w)
    return kept, removed


#: Shortest caption a viewer can read. Also the floor for a cue whose words
#: all report a zero-length span.
MIN_CUE_SEC = 0.12


def normalize_word_timings(words: list[dict], min_dur: float = 0.08) -> list[dict]:
    """Give zero/negative-length words a real span, without overlapping the next.

    faster-whisper emits start == end for roughly one word in ten on this
    content (a chunk-boundary artifact — 116 of 1194 words in the zg50i-UmRwA
    batch). A zero-length word produces a zero-length ``\\k`` tag (clamped to
    1cs, so the highlight flickers) and can leave a whole cue with start == end,
    which renders as a caption that flashes for one frame or not at all.

    The span is taken from the gap up to the next word, so nothing is invented
    outside the silence that was actually recorded.
    """
    out: list[dict] = []
    for i, w in enumerate(words):
        start = float(w.get("start") or 0.0)
        end = float(w.get("end") or 0.0)
        if end - start < min_dur:
            nxt = words[i + 1] if i + 1 < len(words) else None
            limit = float(nxt.get("start") or 0.0) if nxt else start + min_dur
            if limit - start >= min_dur:
                end = limit
            elif limit > start:
                # Several words share one timestamp; split what there is.
                end = start + max(0.04, (limit - start) / 2)
            else:
                end = start + min_dur
        out.append({**w, "start": start, "end": end})
    return out


def words_to_cues(
    words: list[dict],
    *,
    words_per_cue: int = 3,
    max_words: int = 4,
    max_chars: int = 24,
    max_duration: float = 2.2,
    pause_sec: float = 0.30,
    confidence_threshold: float = LOW_CONFIDENCE,
    inaudible_threshold: float = INAUDIBLE_BELOW,
) -> list[dict]:
    """Group word timings into SHORT bursts, cut on natural pauses.

    A burst ends at whichever comes first:
      * a pause of ``pause_sec`` or more before the next word (the speaker
        actually breathed — carrying the caption across that gap is what makes
        captions feel detached from the speech),
      * ``max_words`` words,
      * ``words_per_cue`` words (the target burst length),
      * ``max_chars`` of text, or ``max_duration`` seconds of speech.

    Pauses come from the word timings, so the boundaries land where the speech
    lands rather than every N words. With `words_per_cue=3`/`max_words=4` a
    fast run of speech gives 3-4 word bursts and a hesitant one gives 1-2,
    which is the short-burst rhythm this style is for.

    A cue whose words the model was unsure about is rendered as a placeholder
    (``[muttering]`` / ``[inaudible]``) instead of the guess. Showing a wrong
    caption is worse than showing none: the viewer reads it as fact, and burned
    into the video it cannot be corrected at all. The original words stay on the
    cue, so nothing is lost for the transcript, search or re-render.

    A cue built from words that carry no ``conf`` (any transcript cached before
    this existed) is passed through untouched — there is no evidence of doubt,
    and inventing doubt would relabel good captions on every old batch.
    """
    cues: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if not current:
            return
        cues.append(_make_cue(current[0]["start"], current[-1]["end"],
                              " ".join(str(x["word"]) for x in current),
                              current, confidence_threshold, inaudible_threshold))
        current.clear()

    for w in words:
        start = float(w.get("start") or 0.0)
        if current:
            gap = start - float(current[-1].get("end") or 0.0)
            text = " ".join(str(x["word"]) for x in current + [w])
            dur = float(w.get("end") or 0.0) - current[0]["start"]
            if (
                # `pause_sec = 0` means "never cut on a pause", NOT "cut on
                # every one": a gap is never negative, so `gap >= 0` is always
                # true and the burst collapsed to one word per caption — the
                # exact clunky output the burst settings exist to avoid. The
                # zero case has to be excluded explicitly.
                (pause_sec > 0 and gap >= pause_sec)
                or len(current) + 1 > max_words
                or len(current) >= words_per_cue
                or len(text) > max_chars
                or dur > max_duration
            ):
                flush()
        current.append(w)
    flush()
    return cues


def _selected_highlight_words(words: list[dict], highlight_color: str) -> list[str]:
    """Pick 1-2 key words in a cue to light up in bright color.

    The viral style shows 2-4 words per burst and highlights the word (or
    two) being spoken. We always take the FIRST word of the burst — that is
    the one the speaker lands on as the burst opens — and then one more
    content word (a number, a strong verb, a noun) if the cue has a few
    words to spare. Short cues (1-2 words) get exactly one highlight, no
    matter what, so a 2-word caption never comes out "word-word" with two
    colored boxes.

    The colours rotate per cue so two adjacent chunks never share the same
    accent (the syllables are themselves broken up by the render so the
    highlight colour changes every few words anyway).
    """
    if not words:
        return []
    # The first word of the burst is the one the speaker starts speaking
    # first — the primary highlight. (The render paints the full burst in
    # the break text, then re-colours nothing: a per-word highlight colour
    # is NOT part of the Ssemble viral style, so this function is purely
    # advisory for the dashboard and the tests.)
    return [str(words[0].get("word", "")) or ""]


def _highlight_break(cue: dict, highlight_color: str) -> str:
    """Turn a cue into its ASS text with the highlight word in bright color.

    The Ssemble viral style is ALL CAPS, one line (max 2 lines), and the
    spoken word is painted _over_ the white text in bright color. We do that
    by wrapping just the highlight word in inline colour tags and keeping the
    rest of the break white.

    The ASS ``\1c`` colour tag paints ALL text that follows it, so a naive
    ``{\1c&H...&}{\1c&HFFFFFF&}`` chain would colour the ENTIRE break the
    bright color. Instead we build three pieces — the words before the
    highlight (white), the highlight word (bright), the words after (white)
    — and concatenate white-tagged + highlight-tagged + white-tagged. The
    ASS spec is ambiguous about inline colours, and libass's implementation
    has historically painted everything after a ``\1c`` in that colour, so
    producing a single ``{\1c...}`` block that spans the whole line is the
    only way to get the highlight to sit over the white text without
    re-colouring the rest of the same cue.
    """
    plain = str(cue.get("text", "")).replace("{", "(").replace("}", ")")
    words = [str(w).strip() for w in plain.split()]
    if not words:
        return plain
    selected = _selected_highlight_words(cue.get("words") or [], highlight_color)
    if not selected:
        return plain
    hw = selected[0].upper()
    # Build the white run that ends immediately before the highlight word.
    idx = None
    for i, w in enumerate(words):
        if w.upper() == hw:
            idx = i
            break
    if idx is None:
        return plain
    before = " ".join(w for w in words[:idx] if w)
    after = " ".join(w for w in words[idx + 1:] if w)
    # Wrap each pre-highlight word in its own white tag so a multi-word
    # "before" run stays white even though the previous word's tag is still
    # "open" (the dark text is the white part, not the highlight).
    parts: list[str] = []
    if before:
        parts.extend(
            r"{\1c" + _hex_to_ass("#FFFFFF") + "&}" + w
            for w in before.split()
        )
    parts.append(
        r"{\1c" + _hex_to_ass(highlight_color) + "&}" + hw
        + r"{\1c" + _hex_to_ass("#FFFFFF") + "&}" + r"{\1c" + _hex_to_ass("#FFFFFF") + "&}"
    )
    if after:
        parts.append(
            r"{\1c" + _hex_to_ass("#FFFFFF") + "&}" + after.replace(" ", r" ")
        )
    return "".join(parts)


def _hex_to_ass(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H00{b}{g}{r}".upper()


# --------------------------------------------------------------------------- #
# Keyword emphasis (TARGET B) — 1-2 highlighted words per chunk in rotating
# #1FE01F / #FFF000 / #FF1A1A, chosen with language-aware stopword lists so
# Spanish and English get the same treatment by the same code.
#
# Not every chunk gets a highlight: a chunk whose words are all stopwords, or
# one that already had one in the previous chunk, stays white. The rotation is
# a module-level counter shared across one build_emphasis() call (per clip),
# which is what makes the colours alternate rather than repeat.
# --------------------------------------------------------------------------- #

#: Features that make a word worth emphasising. Numbers first (they carry the
#: claim), then strong verbs/nouns, then anything long and uncommon. Built from
#: the target's own rule: "numbers, strong verbs, emotional or punchline words".
_STRONG_WORD_SUFFIXES_ES = ("ísimo", "ísima", "azo", "aza", "ote", "ota", "mente")
_STRONG_WORD_SUFFIXES_EN = ("ing", "ed", "est", "er")

_STOPWORDS_ES = frozenset("""
    a al algo alguien algunas algunos ante antes como con contra cual cuando
    de del desde donde dos el ella ellas ellos en entre era erais eran eras
    eres es esa esas ese eso esos esta estaba estaban estamos estar esto estos
    estoy fue fueron fui ha habia han hasta hay he hemos la las le les lo los
    mas me mi mia mias mientras mio mios mis misma mismo mucho muy nada ni no
    nos nosotros nuestra nuestras nuestro nuestros o os otra otras otro otros
    para pero poco por porque quien que se sea segun ser si sido siempre sin
    sobre solo son su sus tambien tan tanto te tiene tienen toda todas todo
    todos tu tus un una uno unos ustede ustedes va vale y ya yo ese esa
    este esta esos esas esto entonces entonces pero bueno pues mira oye
    claro si vale ok perfecto Gracias gracias hola adios
""".split())

_STOPWORDS_EN = frozenset("""
    a about above after again all also am an and any are as at be because
    been before being between both but by can cant come could did do does
    doing dont down during each even every few for from get gets getting give
    go goes going gonna got great had has have having he her here hers him
    his how i id if im in into is it its ive just know like me mean more most
    much must my never new no not now of off on once one only or other our
    out over own really right said same say see she should so some still such
    sure take than that the their them then there these they thing think
    this those though through to too under until up us use used using very
    was way we well were what when where which while who why will with would
    yeah yes you your youre okay ok um uh like so just well actually literally
""".split())

_NUMBER_TOKEN = re.compile(r"[0-9]+([.,][0-9]+)?")


def _word_feature(word: str, language: str | None) -> int:
    """How worth emphasising a single word is. Higher is better."""
    w = _norm_token(word)
    if not w:
        return 0
    stops = _STOPWORDS_ES if (language or "").startswith("es") else _STOPWORDS_EN
    suffixes = (
        _STRONG_WORD_SUFFIXES_ES if (language or "").startswith("es")
        else _STRONG_WORD_SUFFIXES_EN
    )
    if _NUMBER_TOKEN.search(word):
        return 10  # numbers carry the claim — always emphasise them first
    if w in stops:
        return 0
    score = min(len(w), 6)
    if any(w.endswith(s) for s in suffixes):
        score += 2
    # ALL-CAPS source words (model emphasis) and quoted span go first
    if word.isupper():
        score += 3
    return score


def build_emphasis(
    cues: list[dict],
    *,
    language: str | None = None,
    enabled: bool = True,
) -> None:
    """Pick 1-2 key words per cue and tag them with rotating accent colours.

    Mutates each cue dict in place, adding ``emphasis`` = list of
    ``{"word": str, "color": "#RRGGBB"}`` pairs. Consumed by the ASS writer
    (``_karaoke_tags``/``build_ass_events``) so the highlighted word is
    painted in its colour for the span the word is SPOKEN, and white
    otherwise — the target's rule, not "every word karaoke-filled".

    Rule: at most 1-2 words per chunk, highest feature score first, the two
    colours rotate through EMPHASIS_COLORS across the clip as a whole, and a
    chunk with no strong word at all (all stopwords) gets nothing — not every
    chunk is emphasised.
    """
    if not enabled:
        return
    rotation = 0
    last_cue_had = False
    for cue in cues:
        if cue.get("uncertain"):
            continue
        words = cue.get("words") or []
        scored = sorted(
            (( _word_feature(str(w.get("word", "")), language), i, w)
             for i, w in enumerate(words)),
            key=lambda t: (-t[0], t[1]),
        )
        picks: list[dict] = []
        for score, _i, w in scored:
            if score < 2:
                continue
            wtxt = str(w.get("word", "")).strip()
            if any(p["word"].lower() == wtxt.lower() for p in picks):
                continue
            picks.append({"word": wtxt, "color": EMPHASIS_COLORS[rotation % len(EMPHASIS_COLORS)]})
            rotation += 1
            if len(picks) >= (2 if score >= 6 else 1):
                break
        # Not every chunk: skip when nothing scored and the previous chunk was
        # ALSO left plain, so at least every other cue carries colour.
        if not picks:
            if not last_cue_had and scored and scored[0][0] >= 2:
                pass  # nothing strong enough — stay white
            last_cue_had = False
            cue["emphasis"] = []
            continue
        cue["emphasis"] = picks
        last_cue_had = True


def display_text(text: str, *, uppercase: bool, max_line: int = 0) -> str:
    """Caption display text: Unicode uppercase, optionally wrapped per line.

    Uppercasing is str.upper() — Unicode-aware, so Spanish accents and the
    inverted ¿ ¡ survive ("á”.upper() -> "Á", "¿" unchanged). The optional
    per-line cap is a target (about 18 chars), not a hard cut: it wraps on a
    word boundary so no word is ever split.
    """
    t = text.upper() if uppercase else text
    if max_line and len(t) > max_line:
        words = t.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_line:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        if cur:
            lines.append(cur)
        t = "\\N".join(lines)
    return t


def write_hook_ass(
    title: str,
    settings: Settings,
    path: Path,
    width: int,
    height: int,
) -> Path:
    """The optional hook title: bold black text on a white rounded box.

    Rendered as a SECOND tiny .ass script layered by a second ``subtitles``
    filter (libass handles the rounded-box look via BorderStyle=3 +
    BackColour=white, and rounds the corners itself with
    ScaledBorderAndShadow). Centred horizontally at 5-13% height — the band
    above the captions where the target places it. Duration is the whole clip
    (the hook is the clip's pitch, it leads), which the caller reads as
    "always on"; the target's own examples keep it for the clip's first
    seconds, which a caller can trim with the file's own Dialogue line.
    """
    p = get_preset("viral")
    black = _hex_to_ass("#0A0A0A")
    white = _hex_to_ass("#FFFFFF")
    font = p.font_override or settings.subtitle.font
    # 5-13% window: centre at 9% via Alignment 8 (top-middle) + MarginV 9%.
    margin_v = int(height * 0.09)
    content = f"""[Script Info]
Title: Clipmake Hook
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,{font},{int(p.font_size * 0.66)},{black},{white},{white},{white},-1,0,0,0,100,100,0,0,3,{int(p.font_size * 0.18)},0,8,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,Hook,,0,0,0,,{title.replace("{", "(").replace("}", ")")}
"""
    path.write_text(content, encoding="utf-8")
    return path


def hook_title_text(
    cue_texts: list[str],
    *,
    language: str | None = None,
    existing_title: str | None = None,
) -> str:
    """A 3-8 word hook title for the whole clip, in the audio's language.

    Prefers the detection stage's own title (it is already the AI's chosen
    hook), truncated/filled to the 3-8 word window; falls back to the clip's
    first strong sentence fragment. Never translates — whatever language the
    transcript is in is what comes out.
    """
    words: list[str]
    if existing_title and existing_title.strip():
        words = existing_title.strip().split()
    else:
        joined = " ".join(cue_texts)
        # take the longest strong-worded opening — enough for a hook
        strong = [w for w in joined.split() if _word_feature(w, language) >= 4]
        words = (strong or joined.split())[:8] or [""]
    if len(words) < 3:
        # pad from the transcript's own first words so the hook is 3+ words
        extra = " ".join(cue_texts).split()
        for w in extra:
            if len(words) >= 3:
                break
            if w not in words:
                words.append(w)
    return " ".join(words[:8]).strip()


def _make_cue(
    start: float, end: float, text: str, words: list[dict],
    threshold: float, inaudible_threshold: float,
) -> dict:
    # Backstop for a cue whose last word reports a zero span: a caption that
    # flashes for one frame is worse than one that lingers a beat.
    end = max(float(end), float(start) + MIN_CUE_SEC)
    conf = cue_confidence(words)
    cue = {
        "start": start,
        "end": end,
        "text": text,
        "words": list(words),
    }
    if conf is not None and conf < threshold:
        cue["text"] = _placeholder_for(conf, inaudible_threshold)
        cue["uncertain"] = True
        cue["confidence"] = round(conf, 3)
    return cue


def ass_font_dir(settings: Settings, preset_key: str | None = None) -> str | None:
    """The fonts dir libass should load for this preset, or None.

    The Viral preset ships Montserrat inside ``app/fonts``; libass must be
    told to look there (fontsdir) because the host has no system Montserrat.
    Returns a POSIX path, ready to interpolate into a filtergraph argument.
    """
    p = get_preset(preset_key)
    dir_name = p.fonts_dir_override
    if not dir_name:
        return None
    # app/fonts, relative to this module
    d = Path(__file__).resolve().parent / dir_name
    return str(d).replace("\\", "/") if d.exists() else None


def style_fingerprint(settings: Settings, preset_key: str | None = None) -> str:
    """Short hash of everything that decides what a caption LOOKS like.

    The preset key alone is not enough: font, size, colours, outline and
    vertical position also come from ``[subtitle]``, so changing any of them
    leaves every already-rendered clip wearing the old style while a newly
    rendered one wears the new — the exact per-clip drift this guards against.
    The render stage stores this next to each clip and re-encodes when it
    changes, so a style edit lands on the whole batch at once.

    The RHYTHM knobs belong here too, not just the looks: burst length, the
    pause that ends a burst and the filler filter all change the words and
    their times, which are burned into the pixels just as firmly as the colour.
    Leaving them out meant a burst length saved from the dashboard produced a
    new style that fingerprinted IDENTICALLY to the old one, so "Re-render
    captions" skipped every clip and the change appeared to do nothing.
    """
    p = get_preset(preset_key, settings.subtitle.highlight_color)
    sub = settings.subtitle
    payload = "|".join(str(v) for v in (
        preset_key or "bold_pop", p.label, p.font_size, p.primary, p.secondary,
        p.outline_color, p.outline_width, p.shadow, p.box, p.karaoke, p.highlight,
        p.pop, p.position_y, sub.font, sub.font_fallbacks,
        sub.words_per_cue, sub.words_per_cue_max, round(sub.pause_sec, 3),
        sub.strip_fillers, sub.strip_stutters,
    ))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_ass_header(
    settings: Settings, width: int, height: int, preset_key: str | None = None
) -> str:
    p = get_preset(preset_key, settings.subtitle.highlight_color)
    # Per-preset font override: the Viral look is a Montserrat ExtraBold look;
    # shipping it in the configured family would erase the style. Falls back
    # to the [subtitle] font when the preset does not name one.
    font = p.font_override or settings.subtitle.font
    primary = _hex_to_ass(p.primary)
    secondary = _hex_to_ass(p.secondary)
    outline = _hex_to_ass(p.outline_color)
    # BorderStyle 3 = opaque box (ASS renders BackColour behind the text)
    border_style = 3 if p.box else 1
    back = "&HA0000000" if p.box else "&H80000000"
    pos = p.position_y if preset_key else settings.subtitle.position_y
    # ASS MarginV is a fraction of the FRAME height measured DOWN from the
    # bottom. Viral: lower-center, 65-70% down the screen = 30-35% from the
    # bottom, so MarginV ~ 0.31-0.35. Always out of the bottom 20% no matter
    # what the preset says.
    margin_v = int(height * min(0.35, max(0.30, 1 - pos)))
    # Vertical strokes (the word being spoken) run RIGHT, so they must NOT be
    # flushed left: ASS rounds the left edge to the box's exact pixel, and with
    # word{black},word{whitesecondary}-style inline colour tags the colour
    # never bleeds outside the box, so a left-stropped caption turns into a
    # stack of un-scrollable words at the box's left edge. Alignment 8 keeps
    # each stroke flush to the box; the 60/60 L/R margins keep the box inside
    # the safe zone on both sides.
    return f"""[Script Info]
Title: Clipmake Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Clipmake,{font},{p.font_size},{primary},{secondary},{outline},{back},-1,0,0,0,100,100,0,0,{border_style},{p.outline_width},{p.shadow},8,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def make_highlight_color(preset_key: str) -> str:
    """One bright accent per preset, rotating so adjacent cues never match.

    The viral style asks for 1-2 key words highlighted in bright colour,
    rotating between yellow #FFE600, green #00FF3C and red #FF1E1E so a
    long clip never lands on two identical colours back-to-back.
    """
    palette = ("#FFE600", "#00FF3C", "#FF1E1E")
    idx = len(CAPTION_PRESETS) % 3
    return palette[idx]


def _karaoke_tags(cue: dict) -> str:
    """ASS \\k timing tags (centiseconds per word) for karaoke-fill style.

    Word dicts come from ``Word.as_dict()``, which keys the text as ``word``.
    This function indexed ``w["text"]`` — the key a *cue* uses — so the karaoke
    preset raised ``KeyError: 'text'`` on the first word and killed the whole
    render at the subtitle stage (job f3b81a1dd7e6, "Pipeline error: 'text'").
    Both spellings are accepted: the payload on disk may predate the dataclass.

    Each word is separated by a SPACE. The tag itself renders as nothing, so
    concatenating bare words put them on screen fused — "bro." + "Did" burned
    in as "bro.Did", and "ain't" + "nigga," as "ain'tnigga,". Whisper keeps
    punctuation attached to the preceding word, so the missing space was
    visible on almost every cue with a comma or full stop in it.
    """
    words = [
        w
        for w in (cue.get("words") or [])
        if (w.get("word") or w.get("text") or "").strip()
    ]
    if not words:
        return ""
    total = max(0.01, cue["end"] - cue["start"])
    parts: list[str] = []
    for w in words:
        dur = max(0.0, float(w.get("end") or 0.0) - float(w.get("start") or 0.0))
        cs = int(round(dur / total * 100 * len(words)))
        # A literal brace in a word would close the tag early and corrupt the
        # rest of the line, so it is folded to a paren like the non-karaoke path.
        text = (w.get("word") or w.get("text") or "").strip().replace("{", "(").replace("}", ")")    # The karaoke preset is a single flat cue: one word per ASS \k tag,
    # each word on its own line, so the sung word is the ONLY text on
    # screen. Without the line break every word lands at the same cue time
    # and ASS merges the whole line into one un-scrollable block (the
    # visible "sang several words at once" bug).
    parts.append(r"{\k%d}%s" % (max(1, cs), text))
    return "\n".join(parts)


def build_ass_events(
    cues: list[dict], settings: Settings, start_offset: float = 0.0,
    preset_key: str | None = None,
) -> str:
    p = get_preset(preset_key, settings.subtitle.highlight_color)
    # The highlight colour lives on the caption track: the ASS `subtitles`
    # filter does NOT read [Events] colour tags, and the viral style paints
    # the spoken word bright (yellow #FFE600, green #00FF3C, red #FF1E1E)
    # over the white break. So we carry the preset's own accent as the
    # highlight colour for this clip's cues — that's the colour the word
    # being spoken gets painted.
    highlight_color = p.secondary if p.secondary else settings.subtitle.highlight_color
    lines: list[str] = []
    for c in cues:
        start = _ass_time(c["start"] - start_offset)
        end = _ass_time(c["end"] - start_offset)
        # The Ssemble-style viral path is a SINGLE flat cue: one burst of
        # 2-4 words, ALL CAPS, white with thick black outline + soft shadow,
        # the spoken word painted bright, pop-in entrance.
        # ``c["text"]`` is the burst text; ``c["words"]`` is what we highlight
        # on. The pop entrance is layered on top of the word tags rather than
        # replacing them, so the default style keeps its scale-in AND the
        # word being spoken lights up.
        text = c["text"].replace("{", "(").replace("}", ")")
        if p.pop:
            text = (
                r"{\fad(60,40)\fscx84\fscy84\t(0,90,\fscx105\fscy105)"
                r"\t(90,140,\fscx100\fscy100)}" + text
            )
        if c.get("words") and not c.get("uncertain"):
            tagged = _highlight_break(c, highlight_color)
            if tagged:
                text = tagged
        lines.append(
            f"Dialogue: 0,{start},{end},{p.key},,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def _emphasis_tags(cue: dict, uppercase: bool) -> str:
    """Per-word inline colouring for the keyword-emphasis styles.

    Every word is wrapped in ``{\\1c&H..&}`` — white for normal words, the
    planned accent for emphasized ones. Unlike karaoke (\\k), the colour is
    static per word: the target's look paints the KEY WORDS, not the word
    being spoken. The cue's own text is used (already display-cased) so the
    burn matches word-for-word what the plain path renders.

    ``_hex_to_ass`` maps #RRGGBB into ASS BGR (&H00BBGGRR), so quoting the
    converted string directly is correct and no channel reshuffling is done
    here — doing it twice is the classic inline-colour bug.
    """
    emph: dict[str, str] = {}
    colored_words: dict[str, str] = {}
    for e in (cue.get("emphasis") or []):
        key = str(e.get("word", "")).strip()
        if not key:
            continue
        color = _hex_to_ass(str(e.get("color", "#FFFFFF")))
        emph[key.upper()] = color
        # A merged phrase ("tres mil millones") is one emphasis entry but
        # arrives as several display tokens — register each of its words so
        # every token of the phrase is painted, not just the first.
        for word in key.split():
            colored_words.setdefault(_norm_display_word(word), color)
    if not emph:
        return ""
    white = _hex_to_ass("#FFFFFF")
    parts: list[str] = []
    for w in [str(w or "") for w in str(cue.get("text", "")).split()]:
        # A display token can carry the ASS \\N line-break inside it
        # (max_chars_per_line wrapping), and the two halves may belong to
        # DIFFERENT states — "CASI\\NIMPOSIBLE" has CASI emphasized and
        # IMPOSIBLE plain. Colour per sub-word, keep the break where it was.
        subs = w.split("\\N")
        colored: list[str] = []
        for sub in subs:
            text = sub.replace("{", "(").replace("}", ")")
            color = colored_words.get(_norm_display_word(sub))
            colored.append(
                r"{\1c" + color + "&}" + text + r"{\1c" + white + "&}"
                if color else text
            )
        parts.append("\\N".join(colored))
    return " ".join(parts)


def _norm_display_word(w: str) -> str:
    """Display token -> comparable key: strip the ASS line-break, case-fold,
    and strip edge punctuation."""
    return (
        w.replace("\\N", " ").strip().strip(".,!?;:\"()[]…¿¡").strip()
        .upper()
    )


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
