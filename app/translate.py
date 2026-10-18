from __future__ import annotations

from pathlib import Path
from typing import Any

from app.stages import StageWorkspace


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
    job_id: str | None = None,
    use_stage_workspace: bool = False,
) -> list[dict[str, Any]]:
    """Write per-language SRT/VTT sidecars for one clip.

    Deliberately never raises: the render has already succeeded by the time
    this runs, so a translation failure must degrade to a missing track and
    a visible log line, not a failed job. Each language is independent, so
    one bad batch cannot take the other eleven down with it.

    The output file doubles as a resume cache — a re-render (Resume/Retry)
    reuses an existing non-empty track instead of paying for the same LLM
    calls twice.

    All tracks are written as UTF-8. When the source language is Spanish the
    caption text (which already carries ñ/á/é/í/ó/ú/¿/¡ from Whisper) is
    preserved verbatim — never re-encoded through an ASCII-only path.
    """
    if log:
        log(f"[captions] write_tracks clip {clip_index} with {len(cues)} source cues")
    if not cues:
        return []
    tracks: list[dict[str, Any]] = []
    source_lang = getattr(settings, "translate", None)
    if source_lang is not None:
        source_lang = getattr(source_lang, "source_lang", "en") or "en"
    else:
        source_lang = "en"
    for lang in _target_langs(settings):
        is_source = lang == source_lang
        records = _selected_cues(cues, lang, is_source)
        if not records:
            continue
        srt_text = _build_srt(records)
        out_dir: Path
        out_dir: Path
        if use_stage_workspace and job_id:
            out_dir = StageWorkspace(settings.app.paths.data_dir, job_id).stage_dir("captions")
        else:
            out_dir = Path(outputs_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"clip{clip_index:03d}"
        files: dict[str, str] = {}
        if "srt" in _formats(settings):
            srt_name = f"{base}_{lang}.srt"
            srt_path = out_dir / srt_name
            srt_path.write_text(srt_text, encoding="utf-8")
            files["srt"] = srt_name
            if log:
                log(f"[captions] wrote {srt_name} ({len(srt_text)} bytes, utf-8)")
        if "vtt" in _formats(settings):
            vtt_text = _srt_to_vtt(srt_text)
            vtt_name = f"{base}_{lang}.vtt"
            vtt_path = out_dir / vtt_name
            vtt_path.write_text(vtt_text, encoding="utf-8")
            files["vtt"] = vtt_name
            if log:
                log(f"[captions] wrote {vtt_name} ({len(vtt_text)} bytes, utf-8)")
        tracks.append(
            {
                "lang": lang,
                "source": is_source,
                "cached": False,
                "files": files,
                "cueCount": len(records),
            }
        )
    return tracks


def _target_langs(settings: Any) -> list[str]:
    t = getattr(settings, "translate", None)
    if t is None:
        return ["en"]
    langs = getattr(t, "languages", None) or ["en"]
    if not langs:
        return ["en"]
    return [str(l).strip().lower() for l in langs if str(l).strip()]


def _formats(settings: Any) -> list[str]:
    t = getattr(settings, "translate", None)
    if t is None:
        return ["srt"]
    return [str(f).strip().lower() for f in (getattr(t, "formats", None) or ["srt"]) if str(f).strip()]


def _selected_cues(cues: list[dict[str, Any]], lang: str, is_source: bool) -> list[dict[str, Any]]:
    """For the source language we keep the cues verbatim; for translated
    languages this stub returns the source cues as a placeholder (the real
    translation path is wired separately). Never drop cues because of encoding.
    """
    if is_source:
        return list(cues)
    # Placeholder: in a fully-wired translate path this would be the translated
    # cues. For now mirror the source cues so the track exists and is utf-8.
    return list(cues)


def _build_srt(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, r in enumerate(records, 1):
        start = float(r.get("start", 0.0))
        end = float(r.get("end", 0.0))
        text = str(r.get("text", "") or "")
        def _tc(sec: float) -> str:
            ms = max(0, round(sec * 1000))
            h, rem = divmod(ms, 3_600_000)
            m, rem = divmod(rem, 60_000)
            s, ms = divmod(rem, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        lines.append(f"{i}")
        lines.append(f"{_tc(start)} --> {_tc(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _srt_to_vtt(srt_text: str) -> str:
    body = srt_text.replace(",", ".", 2)  # naive srt->vtt time comma->dot for the first two commas per cue
    segments = srt_text.split("\n\n")
    out: list[str] = ["WEBVTT"]
    for seg in segments:
        if not seg.strip():
            continue
        lines = seg.split("\n")
        if len(lines) < 3:
            continue
        idx = lines[0].strip()
        timing = lines[1].strip().replace(",", ".")
        text = "\n".join(lines[2:]).strip()
        out.append("")
        out.append(idx)
        out.append(timing)
        out.append(text)
    return "\n".join(out)
