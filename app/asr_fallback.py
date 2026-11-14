"""Hosted Whisper fallback for transcription.

The local ladder is deliberately first: it is private, and on a working host
it is the quality the caption style was built around.  When it cannot finish
(load failure, OOM after the model ladder, a hard stall, or a crash), this
module sends mono 16 kHz FLAC chunks to Groq and returns the SAME
``subtitle.Transcript`` shape the local engine returns.  Caption formatting,
word timing fields and the transcript view therefore do not change.

This module only runs when ``[llm] allow_groq_fallback`` is true and a Groq
key is present.  It never prints the key and it never swallows the reason the
local engine failed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app import groq_asr, subtitle


class GroqUnavailable(groq_asr.GroqAsrError):
    """No usable key/model, so the fallback cannot even be attempted."""


def reason_line(exc: BaseException) -> str:
    """A short, user-visible reason without a traceback or a key."""
    text = " ".join(str(exc).split())
    return (text or type(exc).__name__)[:240]


def enabled(settings: Any) -> bool:
    if not bool(getattr(getattr(settings, "llm", None), "allow_groq_fallback", True)):
        return False
    return bool(getattr(settings, "groq_api_key", None))


def transcribe(
    src: Path,
    settings: Any,
    work_dir: Path,
    *,
    on_progress: Callable[[float], None] | None = None,
    should_pause: Callable[[], bool] | None = None,
    language: str | None = None,
) -> subtitle.Transcript:
    """Transcribe through Groq and return the normal Transcript object."""
    if not enabled(settings):
        raise GroqUnavailable("Groq fallback is off or GROQ_API_KEY is not set")
    cfg = groq_asr.config_from_settings(settings)
    if not cfg.model:
        raise GroqUnavailable(
            "Groq returned no hosted Whisper model for this key; "
            "the local engine is still unavailable"
        )
    key = getattr(settings, "groq_api_key", None)
    if not key:
        raise GroqUnavailable("GROQ_API_KEY is not set")

    started = time.time()

    def progress(frac: float, detail: str) -> None:
        if on_progress is not None:
            on_progress(max(0.0, min(1.0, float(frac))))

    segments, duration = groq_asr.transcribe_source(
        Path(src),
        Path(work_dir),
        cfg,
        key,
        language=language or "auto",
        on_progress=progress,
        should_pause=should_pause,
    )
    words = [
        subtitle.Word(
            start=float(w["start"]),
            end=float(w["end"]),
            word=str(w["word"]),
            conf=float(w.get("conf") or 0.0),
        )
        for w in groq_asr.segments_to_words(segments)
    ]
    wall = max(0.001, time.time() - started)
    logger.info(
        f"[asr_fallback] Groq transcribed {duration:.1f}s in {wall:.1f}s "
        f"({len(words)} words, model {cfg.model})"
    )
    return subtitle.Transcript(
        words=words,
        duration=duration,
        model_size=cfg.model,
        engine="groq whisper (fallback)",
        rtf=round(wall / duration, 4) if duration > 0 else None,
        language=(language or "") if language and language != "auto" else "",
        label=f"transcribed with Groq (fallback: {reason_line(last_error)})"
        if (last_error := _LAST_ERROR.get("value"))
        else "transcribed with Groq (fallback)",
        attempts=[],
        peak_rss_mb=0.0,
    )


# The wrapper in `task` sets this immediately before calling `transcribe`, so
# the label can name the local failure without changing this module's return
# contract or passing exception objects through several layers.
_LAST_ERROR: dict[str, BaseException | None] = {"value": None}


def set_fallback_reason(exc: BaseException | None) -> None:
    _LAST_ERROR["value"] = exc


__all__ = [
    "GroqUnavailable",
    "enabled",
    "reason_line",
    "set_fallback_reason",
    "transcribe",
]
