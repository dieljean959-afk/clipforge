"""Multi-language caption tracks: translate cues and export sidecar SRT/VTT.

Design notes, because two of these are easy to get wrong:

1. Translation goes through :func:`app.llm.call_llm`, so it reuses whatever
   detection backend is configured (Gemini by default, Ollama for a fully
   local/private run). That means no new dependency and no multi-gigabyte
   model download — and it is why there is no ``predownload`` here.

2. **Whisper cannot do this.** Its ``task="translate"`` targets English only,
   so it covers 1 of the 12 languages below; everything else needs a real
   translation step. Do not "optimise" this into the ASR path.

3. Timings are never touched. Translation is per-cue and timing-preserving by
   construction: only ``cue["text"]`` is replaced, so a bad translation can
   shift wording but can never desynchronise a caption.

4. Per-cue translation loses the context a human translator would use. Each
   batch is therefore sent with the surrounding cues and adjacent lines as
   hints, and cues are batched (not sent one-per-request) so the model sees a
   coherent stretch of dialogue.

5. Line budgets are script-dependent, not universal. The familiar 42-character
   guideline is for Latin scripts; full-width CJK glyphs advance roughly twice
   as far, so a 42-character Japanese line overflows badly. See ``LINE_BUDGET``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from loguru import logger


class TranslateError(RuntimeError):
    """Translation failed in a way the caller must surface, not swallow."""


# --------------------------------------------------------------------------
# languages
# --------------------------------------------------------------------------

#: Ordered so the source language is emitted first and the order is stable in
#: the UI. Codes are ISO-639-1; ``zh`` is Mandarin (Simplified).
LANGS: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "de": "German",
    "ru": "Russian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Mandarin Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
}

#: Scripts written without spaces between words. Character wrapping is the only
#: correct option for these — word-boundary wrapping would never split them.
UNSPACED_LANGS = frozenset({"ja", "ko", "zh"})

#: Max characters per line, per language. Latin/Cyrillic ~42 is the broadcast
#: norm; Arabic and Devanagari run slightly wide; CJK is full-width, so ~18 is
#: the equivalent reading load rather than 42. These are *display* budgets and
#: deliberately separate from the burned-in ASS path (words_per_cue/max_chars),
#: which is a different presentation.
LINE_BUDGET: dict[str, int] = {
    "en": 42, "fr": 42, "es": 42, "pt": 42, "de": 42,
    "ru": 42, "it": 42,
    "ar": 40, "hi": 40,
    "ja": 18, "ko": 18, "zh": 18,
}

#: Cues per translation request. Large enough to carry dialogue context, small
#: enough that a malformed reply only costs one batch rather than the whole job.
BATCH_SIZE = 30

DEFAULT_MAX_LINES = 2


def lang_name(code: str) -> str:
    return LANGS.get(code, code)


def is_known(code: str) -> bool:
    return code in LANGS


def normalize_codes(codes: Iterable[str]) -> list[str]:
    """Filter to known languages, de-duplicate, preserve order."""
    out: list[str] = []
    for raw in codes:
        code = str(raw).strip().lower()
        if code and code in LANGS and code not in out:
            out.append(code)
    return out


# --------------------------------------------------------------------------
# line wrapping
# --------------------------------------------------------------------------


@dataclass
class Wrapped:
    lines: list[str]
    #: True when the text did not fit ``max_lines`` at ``budget``. The caller
    #: surfaces this rather than truncating: dropping subtitle text silently is
    #: worse than an over-long caption, and an over-long caption is fixable.
    overflow: bool = False


def wrap_text(text: str, budget: int, max_lines: int = DEFAULT_MAX_LINES,
              lang: str = "en") -> Wrapped:
    """Wrap into readable lines without ever discarding characters."""
    text = " ".join(str(text).split())
    if not text:
        return Wrapped([""])
    if budget <= 0:
        return Wrapped([text])

    if lang in UNSPACED_LANGS:
        chunks = [text[i:i + budget] for i in range(0, len(text), budget)]
    else:
        chunks: list[str] = []
        current = ""
        for word in text.split(" "):
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > budget:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)

    if not chunks:
        return Wrapped([text])
    return Wrapped(chunks, overflow=len(chunks) > max_lines)


def _rebalance(lines: list[str], budget: int, max_lines: int) -> list[str]:
    """Move a word down to even out two very lopsided lines.

    A 40/1 split is technically compliant and reads badly; a 22/19 split is the
    same text and much better on screen. Only ever moves whole words, and only
    when the result still fits.
    """
    if len(lines) != 2 or max_lines < 2:
        return lines
    first, second = lines
    if len(first) <= budget * 0.7 or " " not in first:
        return lines
    moved = first.rsplit(" ", 1)
    if len(moved) != 2:
        return lines
    head, tail = moved
    candidate = f"{tail} {second}".strip()
    if head and len(candidate) <= budget:
        return [head, candidate]
    return lines


# --------------------------------------------------------------------------
# glossary
# --------------------------------------------------------------------------


def glossary_terms(settings: Any) -> list[str]:
    """Terms that must survive translation verbatim (brand/product names)."""
    raw = getattr(getattr(settings, "translate", None), "glossary", None) or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    return [str(t).strip() for t in raw if str(t).strip()]


def glossary_misses(src_text: str, out_text: str, terms: Sequence[str]) -> list[str]:
    """Terms present in the source but absent from the translation.

    Checked case-insensitively and only for terms that actually occurred, so a
    glossary entry the speaker never said cannot raise a false alarm.
    """
    src_low, out_low = src_text.lower(), out_text.lower()
    return [t for t in terms if t.lower() in src_low and t.lower() not in out_low]


# --------------------------------------------------------------------------
# prompt / parsing
# --------------------------------------------------------------------------


def build_prompt(texts: Sequence[str], lang: str, glossary: Sequence[str],
                 context: str = "") -> str:
    """One batched translation request.

    Numbered JSON in, JSON array out: a positional contract is the only thing
    that can be validated cheaply, and a length mismatch is a hard error rather
    than a silently misaligned subtitle track.
    """
    lines = [f"Translate these {len(texts)} subtitle lines into {lang_name(lang)} ({lang}).", ""]
    lines.append("Rules:")
    lines.append("- Return a JSON array of exactly the same length, same order.")
    lines.append("- Natural, fluent phrasing as a native speaker would subtitle it.")
    lines.append("- NOT word-for-word. Preserve meaning, tone and register.")
    lines.append("- Keep each line short: it is read on screen for ~2 seconds.")
    lines.append("- Do not merge, split, reorder or omit entries.")
    lines.append("- Do not add speaker labels, quotes or notes not present in the source.")
    if glossary:
        lines.append("- Keep these terms verbatim, untranslated and untransliterated: "
                     + ", ".join(glossary))
    if context:
        lines.append("")
        lines.append("For context only (do NOT translate or return this):")
        lines.append(context)
    lines.append("")
    lines.append("Subtitle lines to translate:")
    lines.append(json.dumps(list(texts), ensure_ascii=False, indent=1))
    lines.append("")
    lines.append("Reply with only the JSON array.")
    return "\n".join(lines)


def parse_translations(raw: str, expected: int) -> list[str]:
    """Extract the JSON array, tolerating code fences and stray prose."""
    if not raw or not raw.strip():
        raise TranslateError("translation backend returned an empty reply")

    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost bracketed span.
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            try:
                payload = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                payload = None
    if payload is None:
        raise TranslateError(f"could not parse a JSON array from the reply: {raw[:200]!r}")

    if isinstance(payload, dict):
        for key in ("translations", "lines", "result", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TranslateError(f"expected a JSON array, got {type(payload).__name__}")

    out = [("" if item is None else str(item).strip()) for item in payload]
    if len(out) != expected:
        # A positional mismatch would silently shift every subsequent caption.
        raise TranslateError(
            f"translation count mismatch: asked for {expected}, got {len(out)}"
        )
    return out


# --------------------------------------------------------------------------
# the translation step
# --------------------------------------------------------------------------


def translate_texts(texts: Sequence[str], lang: str, settings: Any,
                    glossary: Sequence[str] | None = None,
                    call: Callable[[str, Any], str] | None = None,
                    context: str = "") -> list[str]:
    """Translate a batch of cue strings, preserving order and count."""
    if not texts:
        return []
    terms = list(glossary if glossary is not None else glossary_terms(settings))
    if call is None:
        from app import llm as llm_mod  # noqa: PLC0415 - import cycle at module load

        call = llm_mod.call_llm
    prompt = build_prompt(texts, lang, terms, context=context)
    raw = call(prompt, settings)
    result = parse_translations(raw, len(texts))

    # Terminology drift is reported, never silently rewritten: a wrong-term
    # brand name is a content bug the operator should see.
    for src, out in zip(texts, result):
        missed = glossary_misses(src, out, terms)
        if missed:
            logger.warning(f"[translate] {lang}: glossary term(s) missing in translation: {missed}")
    return result


def translate_cues(cues: Sequence[dict], lang: str, settings: Any,
                   glossary: Sequence[str] | None = None,
                   call: Callable[[str, Any], str] | None = None,
                   batch_size: int = BATCH_SIZE) -> list[dict]:
    """Return cues with translated text and byte-identical timings.

    ``cues`` are ``{"start": float, "end": float, "text": str}`` (clip-relative).

    An uncertain cue carries a ``[muttering]`` / ``[inaudible]`` placeholder
    rather than a transcription. It is passed through untranslated on purpose:
    those markers mean "this could not be made out", which is true in every
    language, and handing them to an LLM invites a confident invented
    translation of a line nobody can hear.
    """
    out: list[dict] = []
    pending: list[int] = []  # indexes into `out` still waiting for a translation
    for offset in range(0, len(cues), batch_size):
        window = list(cues[offset:offset + batch_size])
        to_translate = [c for c in window if not c.get("uncertain")]
        for cue in window:
            out.append({
                "start": cue["start"],
                "end": cue["end"],
                "text": str(cue.get("text", "")) if cue.get("uncertain") else "",
            })
            if not cue.get("uncertain"):
                pending.append(len(out) - 1)
        if not to_translate:
            continue
        # Neighbouring cues as context: cheap, and markedly improves how natural
        # the phrasing is versus translating each line in isolation.
        before = [str(c.get("text", "")) for c in cues[max(0, offset - 3):offset]]
        after = [str(c.get("text", "")) for c in cues[offset + batch_size:offset + batch_size + 3]]
        context = ""
        if before:
            context += "preceding lines: " + json.dumps(before, ensure_ascii=False) + "\n"
        if after:
            context += "following lines: " + json.dumps(after, ensure_ascii=False)

        translated = translate_texts(
            [str(c.get("text", "")) for c in to_translate],
            lang, settings, glossary, call=call, context=context.strip(),
        )
        for idx, text in zip(pending[-len(translated):] if translated else [], translated):
            out[idx]["text"] = text
    return out


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------


def srt_time(sec: float) -> str:
    ms = max(0, round(float(sec) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(sec: float) -> str:
    return srt_time(sec).replace(",", ".")


def _render_cues(cues: Sequence[dict], lang: str, budget: int,
                 max_lines: int) -> list[tuple[dict, list[str], bool]]:
    rows = []
    for cue in cues:
        wrapped = wrap_text(str(cue.get("text", "")), budget, max_lines, lang=lang)
        lines = _rebalance(wrapped.lines, budget, max_lines)
        rows.append((cue, lines, wrapped.overflow or len(lines) > max_lines))
    return rows


def to_srt(cues: Sequence[dict], lang: str, budget: int | None = None,
           max_lines: int = DEFAULT_MAX_LINES) -> str:
    limit = budget if budget is not None else LINE_BUDGET.get(lang, 42)
    blocks, overflow = [], 0
    for i, (cue, lines, over) in enumerate(_render_cues(cues, lang, limit, max_lines), 1):
        overflow += int(over)
        blocks.append(
            f"{i}\n{srt_time(cue['start'])} --> {srt_time(cue['end'])}\n" + "\n".join(lines) + "\n"
        )
    if overflow:
        logger.warning(f"[translate] {lang}: {overflow} cue(s) exceeded {max_lines} lines at {limit} chars")
    return "\n".join(blocks)


def to_vtt(cues: Sequence[dict], lang: str, budget: int | None = None,
           max_lines: int = DEFAULT_MAX_LINES) -> str:
    limit = budget if budget is not None else LINE_BUDGET.get(lang, 42)
    blocks, overflow = ["WEBVTT", ""], 0
    for cue, lines, over in _render_cues(cues, lang, limit, max_lines):
        overflow += int(over)
        blocks.append(f"{vtt_time(cue['start'])} --> {vtt_time(cue['end'])}\n" + "\n".join(lines) + "\n")
    if overflow:
        logger.warning(f"[translate] {lang}: {overflow} cue(s) exceeded {max_lines} lines at {limit} chars")
    return "\n".join(blocks)


def track_filename(clip_index: int, lang: str, ext: str) -> str:
    """Stable, sortable, language-tagged name. Clip 0 means the whole job."""
    stem = "captions" if clip_index <= 0 else f"clip{clip_index:02d}"
    return f"{stem}_{lang}.{ext}"


def write_tracks(out_dir: Any, clip_index: int, cues: Sequence[dict], settings: Any,
                 call: Callable[[str, Any], str] | None = None,
                 log: Callable[[str], None] | None = None) -> list[dict]:
    """Write per-language SRT/VTT sidecars for one clip. Returns track records.

    Deliberately NEVER raises. By the time this runs the render has already
    succeeded, so a translation failure must degrade to a missing track plus a
    visible log line, not a failed job. Each language is independent, so one
    bad batch cannot take the other eleven down with it.

    An existing non-empty file doubles as a resume cache: a re-render reuses it
    rather than paying for the same translation twice.
    """
    from pathlib import Path  # noqa: PLC0415 - only needed on this path

    cfg = getattr(settings, "translate", None)
    if cfg is None or not getattr(cfg, "enabled", False) or not cues:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    terms = glossary_terms(settings)
    tracks: list[dict] = []

    for code in cfg.languages:
        files: dict[str, str] = {}
        source_track = code == cfg.source_lang

        if source_track:
            # Copied verbatim: round-tripping the original through a translator
            # would silently reword the source captions.
            localized = [dict(c) for c in cues]
        else:
            existing = [out_dir / track_filename(clip_index, code, ext) for ext in cfg.formats]
            if any(p.is_file() and p.stat().st_size > 0 for p in existing):
                tracks.append({"lang": code, "source": False, "cached": True,
                               "files": {ext: str(out_dir / track_filename(clip_index, code, ext))
                                         for ext in cfg.formats}})
                continue
            try:
                localized = translate_cues(cues, code, settings, terms,
                                           call=call, batch_size=cfg.batch_size)
            except Exception as exc:  # noqa: BLE001 - one language must not fail the job
                logger.warning(f"[translate] {code}: {type(exc).__name__}: {exc}")
                if log:
                    log(f"clip {clip_index}: {code} caption track skipped - {exc}")
                tracks.append({"lang": code, "source": False, "error": str(exc), "files": {}})
                continue

        for ext in cfg.formats:
            path = out_dir / track_filename(clip_index, code, ext)
            writer = to_srt if ext == "srt" else to_vtt
            # UTF-8 explicitly: these files carry CJK, Arabic and Devanagari.
            path.write_text(writer(localized, code, max_lines=cfg.max_lines), encoding="utf-8")
            files[ext] = str(path)

        tracks.append({"lang": code, "source": source_track, "files": files})
        if not source_track and log:
            log(f"clip {clip_index}: {code} caption track written")

    return tracks
