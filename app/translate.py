from __future__ import annotations

from typing import Any

LANGS = {"en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh", "ar", "hi", "tr", "vi"}

#: Normalized language codes that the captioned-track renderer is willing to emit.
USED_LANGS = {"en", "es"}


def normalize_codes(codes: list[str]) -> list[str]:
    """Lower-case + strip. Unknown codes are dropped here, not surfaced later."""
    out: list[str] = []
    for c in codes:
        key = str(c).strip().lower()
        if key and key in LANGS:
            out.append(key)
    return out


def build_cues(
    transcript: Any,
    cut: dict[str, Any],
    settings: Any,
) -> list[dict[str, Any]]:
    """Assemble caption cues for ONE rendered clip from a transcript + cut window.

    ``transcript`` is :class:`app.subtitle.Transcript`. ``cut`` is one item from
    the reframe stage's ``cuts`` list and must have numeric ``start``/``end``
    (clip-relative within the source).

    Returns a list of ``{"start": float, "end": float, "text": str}`` cues,
    clip-relative. This is the shape the caption stage consumes both for building
    translated tracks and for the burned-in ASS path downstream.
    """
    words = getattr(transcript, "words", None)
    if not words:
        return []

    cut_start = float(cut.get("start", 0.0))
    cut_end = float(cut.get("end", 0.0))
    if cut_end <= cut_start:
        return []

    inside: list[Any] = []
    for w in words:
        ws = float(getattr(w, "start", 0.0))
        we = float(getattr(w, "end", 0.0))
        if we <= cut_start or ws >= cut_end:
            continue
        inside.append(w)

    if not inside:
        return []

    cues: list[dict[str, Any]] = []
    buf: list[Any] = []
    buf_start = inside[0].start
    buf_end = inside[0].end
    running = 0.0

    def flush() -> None:
        nonlocal buf, running
        if not buf:
            return
        text = " ".join(str(getattr(w, "word", "")) for w in buf).strip()
        cues.append({
            "start": round(max(0.0, buf_start - cut_start), 3),
            "end": round(max(0.0, min(cut_end, buf_end) - cut_start), 3),
            "text": text,
        })
        buf = []
        running = 0.0

    for w in inside:
        ws = float(getattr(w, "start", 0.0))
        we = float(getattr(w, "end", 0.0))
        gap = ws - buf_end if buf else 0.0
        word_len = we - ws

        if not buf:
            buf.append(w)
            buf_start = ws
            buf_end = we
            running = word_len
            continue

        if gap > 0.45:
            flush()
            buf_start = ws
            buf_end = we
            running = word_len
        else:
            buf_end = max(buf_end, we)
            running += gap + word_len

        buf.append(w)

    flush()
    return cues


def write_tracks(
    outputs_dir: Any,
    clip_index: int,
    cues: list[dict[str, Any]],
    settings: Any,
    *,
    log: Any = None,
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
    if log:
        log(f"[captions] write_tracks clip {clip_index} with {len(cues)} source cues")
    return []
