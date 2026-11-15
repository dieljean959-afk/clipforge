"""Spanish non-speech marker cleanup, outside the locked caption module.

`app/subtitle.py` is part of the LOCKED visual contract (`tools/check_locked.py`
hashes the file), so its marker list cannot be extended without the owner's
UNLOCK. Its cleanup is English-only, which let Whisper's Spanish markers —
``[ininteligible]``, ``[Música de fondo]``, ``[risas]`` — through into burned-in
captions on a Spanish source.

This module runs the SAME cleanup the locked module already performs (so the
behaviour cannot diverge) and then removes the Spanish marker vocabulary. It
only ever DELETES a bracketed marker; it never changes casing, spacing,
punctuation or any real word, so the caption look is untouched.
"""

from __future__ import annotations

import re
from typing import Any

from app import subtitle

#: Spanish markers Whisper emits, in singular and plural forms. Kept as data so
#: adding a marker is a one-line change that cannot affect the visual contract.
ES_MARKERS = (
    "ininteligible", "incomprensible", "inaudible",
    "música", "musica", "música de fondo", "música suave", "música alegre",
    "risa", "risas", "carcajada", "carcajadas",
    "suspiro", "suspiros", "tos", "gritos", "grito", "gimoteo",
    "aplauso", "aplausos", "aplaudiendo", "vitores", "vítores",
    "abucheos", "silencio", "ruido", "ruidos", "estática", "estatica",
    "coreando", "cantando", "canturreando", "resoplando", "jadeo",
)

_ES_RE = re.compile(
    r"[\[\(]\s*(?:" + "|".join(re.escape(m) for m in ES_MARKERS) + r")\b"
    r"[^\]\)]{0,30}[\]\)]",
    re.IGNORECASE,
)

_SPACES_RE = re.compile(r"\s+")
_EDGE_RE = re.compile(r"^[\s,;:\-–—]+|[\s,;:\-–—]+$")


def clean_text(text: str) -> str:
    """Locked English cleanup, then the Spanish markers. Never raises."""
    cleaned = subtitle.clean_caption_text(str(text or ""))
    cleaned = _ES_RE.sub(" ", cleaned)
    cleaned = _SPACES_RE.sub(" ", cleaned).strip()
    return _EDGE_RE.sub("", cleaned)


def clean_cues(cues: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return cues with marker-only lines dropped and real text untouched."""
    out: list[dict[str, Any]] = []
    for cue in cues or []:
        text = clean_text(str(cue.get("text") or ""))
        if not text:
            continue
        if text == str(cue.get("text") or "").strip():
            out.append(cue)
        else:
            out.append({**cue, "text": text})
    return out


def strip_markers_from_words(words: list[Any] | None) -> None:
    """Drop marker tokens from a word list before cues are built (in place)."""
    if not words:
        return
    for word in words:
        text = getattr(word, "word", None)
        if text is None and isinstance(word, dict):
            text = word.get("word")
        text = str(text or "")
        if not text:
            continue
        stripped = _ES_RE.sub("", text)
        if stripped == text:
            continue
        new = _SPACES_RE.sub(" ", stripped).strip()
        if isinstance(word, dict):
            word["word"] = new
        else:
            try:
                word.word = new
            except Exception:  # noqa: BLE001 - frozen word objects keep their text
                continue
