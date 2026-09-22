"""Subtitle service: faster-whisper word-level transcription + ASS captions.

ASS style matches the reference look: bold white text with a thick neon-green
outline, word-by-word pop-in, 2-3 words visible, lower-third. If ImageMagick
is missing, rendering falls back to Pillow-composited PNG overlays via ffmpeg —
captions must export either way.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import Settings, ffprobe_bin


class TranscribeError(Exception):
    pass


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


def _probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            [ffprobe_bin(), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip() or 0)
    except Exception as exc:
        logger.warning(f"[subtitle] ffprobe duration failed for {path}: {exc}")
        return 0.0


def transcribe(path: Path, settings: Settings, on_progress=None) -> Transcript:
    """Word-level transcription with real progress reporting."""
    model = _load_model(settings)
    duration = _probe_duration(path)
    segments_iter, info = model.transcribe(
        str(path),
        word_timestamps=True,
        language=None if settings.whisper.language == "auto" else settings.whisper.language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    words: list[Word] = []
    seg_count = 0
    for seg in segments_iter:
        seg_count += 1
        for w in (seg.words or []):
            words.append(Word(start=float(w.start), end=float(w.end), word=w.word.strip()))
        if on_progress and info.duration:
            on_progress(min(1.0, float(seg.end) / info.duration))
    transcript = Transcript(words=words, duration=duration or float(info.duration or 0),
                            model_size=settings.whisper.model)
    logger.info(f"[subtitle] transcribed {len(words)} words, {seg_count} segments, {transcript.duration:.1f}s")
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
            })
            current = []
    if current:
        cues.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": " ".join(x["word"] for x in current),
        })
    return cues


def build_ass_header(settings: Settings, width: int, height: int) -> str:
    font = settings.subtitle.font
    outline = _hex_to_ass(settings.subtitle.outline_color)
    return f"""[Script Info]
Title: ClipForge Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ClipForge,{font},{settings.subtitle.font_size},&H00FFFFFF,&H00FFFFFF,{outline},&H80000000,-1,0,0,0,100,100,0,0,1,{settings.subtitle.outline_width},0,2,60,60,{int(height * (1 - settings.subtitle.position_y))},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass_events(cues: list[dict], settings: Settings, start_offset: float = 0.0) -> str:
    lines: list[str] = []
    for c in cues:
        start = _ass_time(c["start"] - start_offset)
        end = _ass_time(c["end"] - start_offset)
        text = c["text"].replace("{", "(").replace("}", ")")
        if settings.subtitle.pop_animation:
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
) -> Path:
    """Write an .ass subtitle file; returns the path."""
    content = build_ass_header(settings, width, height) + build_ass_events(cues, settings, start_offset)
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
