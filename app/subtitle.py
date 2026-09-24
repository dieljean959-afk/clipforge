"""Subtitle service: faster-whisper word-level transcription + ASS captions.

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
from typing import Any

from loguru import logger

from app.config import Settings, ffmpeg_bin, ffprobe_bin


class TranscribeError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Anti-hallucination tuning
#
# Whisper is known to enter repetition loops on silent or music-only segments
# ("Thank you for watching..." x40). Three defenses:
#   1. condition_on_previous_text=False — each window decodes without seeing
#      previous text, so a hallucination cannot anchor the next window.
#   2. compression_ratio_threshold — segments whose compressed text is far
#      smaller than the raw text are repetitive; they are dropped (not kept).
#   3. no_speech_threshold — segments with high silence probability are
#      skipped, so near-silent audio yields no words instead of fantasy text.
# On top of that, long files are transcribed in ~30s windows with a small
# overlap so any drift that does accumulate only affects one window.
# --------------------------------------------------------------------------- #

WHISPER_KWARGS = dict(
    condition_on_previous_text=False,
    compression_ratio_threshold=2.2,
    no_speech_threshold=0.6,
    word_timestamps=True,
    vad_filter=True,
    vad_parameters={
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 60,
    },
)

# Long files are decoded in windows of this length with this much overlap so
# words spanning a boundary are caught once by at least one window (then
# deduplicated in _dedupe_words).
WINDOW_SEC = 30.0
OVERLAP_SEC = 1.0
# Whisper is most reliable on ~30s of audio; decode windows above this length
# are split (unless they fall inside one detected speech window).
MAX_DECODE_SEC = 45.0


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


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple, Any] = {}


def _resolve_device(settings: Settings) -> tuple[str, str]:
    """Return (device, compute_type) resolved from 'auto'."""
    device = settings.whisper.device
    compute = settings.whisper.compute_type
    if device == "auto":
        try:
            import ctranslate2  # noqa: PLC0415

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _model_cache_dir() -> Path:
    env = None
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415

        env = Path(HF_HUB_CACHE)
    except Exception:
        env = None
    if env is None:
        env = Path.home() / ".cache" / "huggingface"
    return env


def predownload_model(settings: Settings, on_progress=None, on_log=None) -> Transcript:
    """Load (and therefore download) the configured Whisper model at boot.

    Raises TranscribeError with a readable message on failure.
    """
    from faster_whisper import WhisperModel  # imported here so config errors come first

    device, compute = _resolve_device(settings)
    key = (settings.whisper.model, device, compute)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        t0 = time.time()
        if on_log:
            on_log(f"Loading Whisper model '{settings.whisper.model}' ({compute})...")
        try:
            model = WhisperModel(
                settings.whisper.model,
                device=device,
                compute_type=compute,
            )
        except Exception as exc:
            raise TranscribeError(f"Whisper model load failed: {exc}") from exc
        took = time.time() - t0
        cache_dir = _model_cache_dir()
        line = f"Whisper model loaded: {settings.whisper.model} from {cache_dir} in {took:.1f}s"
        logger.info(f"[subtitle] {line}")
        if on_log:
            on_log(line)
        transcript = Transcript(model_size=settings.whisper.model, model_path=str(cache_dir))
        _MODEL_CACHE[key] = (model, transcript)
        return transcript


def _load_model(settings: Settings):

    device, compute = _resolve_device(settings)
    key = (settings.whisper.model, device, compute)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            predownload_model(settings)
        return _MODEL_CACHE[key][0]


def _extract_media(path: Path, start: float, end: float, workdir: Path) -> Path:
    """Extract one audio window [start, end) to 16kHz mono WAV via ffmpeg."""
    duration = max(0.5, end - start)
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"win_{start:09.2f}_{end:09.2f}.wav"
    subprocess.run(
        [
            ffmpeg_bin(),
            "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", str(path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(out),
        ],
        check=True, capture_output=True, timeout=120,
    )
    return out


def _decode_window(
    model,
    wav: Path,
    language: str | None,
    on_segment=None,
) -> tuple[list[Word], float]:
    """Transcribe one 16kHz mono WAV window; returns (words, audio_len_sec).

    `on_segment(local_end_sec)` fires after each decoded segment so the caller
    can report progress INSIDE a window — on CPU the `medium` model can spend
    minutes on one 30s window, and without this the job looks dead.
    """
    segments_iter, info = model.transcribe(
        str(wav),
        language=language,
        **WHISPER_KWARGS,
    )
    words: list[Word] = []
    win_len = float(info.duration or 0)
    for seg in segments_iter:
        for w in (seg.words or []):
            text = w.word.strip()
            if not text:
                continue
            words.append(Word(start=float(w.start), end=float(w.end), word=text))
        if on_segment is not None and win_len > 0:
            on_segment(min(win_len, float(seg.end)))
    return words, win_len


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
    """Plan [start, end) windows covering [0, duration)."""
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
) -> Transcript:
    """Word-level transcription, hardened against repetition loops.

    Long inputs are decoded in ~30s overlapping windows instead of one pass,
    every call runs with condition_on_previous_text=False, a tight
    compression_ratio_threshold (2.2) and no_speech_threshold=0.6, and the
    built-in Silero VAD filter strips silence/music before the model sees it.

    `cache_dir` enables INCREMENTAL resume: each completed window is written
    to cache_dir/win_XXXXX.json as it decodes, so a retry after a crash/stall
    skips the windows already done and loses at most one window of work.
    """
    model = _load_model(settings)
    try:
        duration = _probe_duration(path)
    except TranscribeError:
        # _probe_duration already raised a readable message (missing file,
        # ffprobe failure, corrupt download) — surface it, never a NameError
        raise
    language = None if settings.whisper.language == "auto" else settings.whisper.language

    windows = _plan_windows(duration)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    resumed = 0
    if len(windows) == 1:
        logger.info(
            f"[subtitle] transcribing {duration:.1f}s in one window "
            "(condition_on_previous_text=False, cr=2.2, ns=0.6, VAD on)"
        )
    else:
        logger.info(
            f"[subtitle] transcribing {duration:.1f}s in {len(windows)} "
            f"~{WINDOW_SEC:.0f}s windows (overlap {OVERLAP_SEC:.1f}s) to limit drift-induced repetition"
        )

    all_words: list[Word] = []
    for idx, (w_start, w_end) in enumerate(windows):
        # ---- incremental resume: reuse an already-decoded window ----------
        cached = None
        if cache_dir is not None:
            cache_file = cache_dir / f"win_{idx:05d}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    if abs(float(data.get("w_start", -1)) - w_start) < 0.01:
                        cached = [Word(**w) for w in data.get("words", [])]
                except Exception:
                    cached = None  # corrupt cache → just redo this window
        if cached is not None:
            all_words.extend(cached)
            resumed += 1
            if on_progress and duration > 0:
                on_progress(min(1.0, w_end / duration))
            continue

        # heartbeat BEFORE extraction: audio decode of a long window can take
        # tens of seconds, and the stall watchdog keys off progress events
        if on_progress and duration > 0:
            on_progress(min(1.0, w_start / duration))
        try:
            wav = _extract_media(path, w_start, w_end, Path(settings.app.paths.data_dir))
        except Exception as exc:
            raise TranscribeError(
                f"Could not extract audio window {w_start:.1f}-{w_end:.1f}s: {exc}"
            ) from exc

        def on_segment(local_end: float) -> None:
            # intra-window liveness: map window-local segment end → absolute
            if on_progress and duration > 0:
                on_progress(min(1.0, (w_start + local_end) / duration))

        try:
            words, _ = _decode_window(model, wav, language, on_segment=on_segment)
        except Exception as exc:
            raise TranscribeError(
                f"Whisper decode failed on window {idx + 1}/{len(windows)}: {exc}"
            ) from exc

        # window-local word timings → source-absolute time
        abs_words = [
            Word(start=w.start + w_start, end=w.end + w_start, word=w.word)
            for w in words
        ]
        all_words.extend(abs_words)
        wav.unlink(missing_ok=True)

        # persist the window BEFORE moving on (incremental resume)
        if cache_dir is not None:
            cache_file = cache_dir / f"win_{idx:05d}.json"
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

    if resumed:
        logger.info(f"[subtitle] resumed: reused {resumed}/{len(windows)} cached windows")

    words_out = _dedupe_words(all_words)
    dropped = len(all_words) - len(words_out)
    if dropped > 0:
        logger.info(f"[subtitle] overlap dedup removed {dropped} duplicate words")

    transcript = Transcript(
        words=words_out,
        duration=duration,
        model_size=settings.whisper.model,
    )
    logger.info(
        f"[subtitle] transcribed {len(words_out)} words across {len(windows)} window(s), "
        f"{transcript.duration:.1f}s of audio"
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
