from typing import Any


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
