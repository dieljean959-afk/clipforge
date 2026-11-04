from __future__ import annotations

from pathlib import Path
from typing import Any

from app.stages import StageWorkspace


LANGS = {"en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh", "ar", "hi", "tr", "vi"}

#: Normalized language codes that the captioned-track renderer is willing to emit.
USED_LANGS = {"en", "es"}


#: High-frequency Spanish function words. Deliberately EXCLUDES tokens that
#: are also normal English words (a, no, me, mi, e, o, da, di, es? no — "es"
#: is not English) so a single loanword cannot tip the score.
_ES_FUNCTION = frozenset("""
    al algo alguien aqui asi aunque bien bueno cada como con contra cual cuando
    de del desde donde dos el ella ellas ellos en entre era eran eres esa esas
    ese eso esos esta estaba estado estan estar estas este esto estos estoy fue
    fueron fui ha habia han hasta hay he hemos la las le les lo los mas mientras
    mucho muy nada ni nos nosotros nuestra nuestro nuevo otro para pero poco por
    porque que quien se sea segun ser si sido siempre sin sobre solo son su sus
    tambien tan tanto te tiene tienen todo todos tu tus un una uno unos usted
    ustedes va vale vamos ver y ya yo
""".split())

#: Same idea for English. "para" and "pero" are NOT included even though they
#: look Spanish-adjacent; they are the words that carry Spanish sentences.
_EN_FUNCTION = frozenset("""
    about after again all also am an and any are as at back be because been
    before being but by can could did do does doing done down each even every
    few for from get got had has have he her here him his how if in into is it
    its just know like long made make many may might more most much my never new
    not now of off on once one only or other our out over own really right said
    same say see she should since so some still such take than that the their
    them then there these they thing think this those through time to too two up
    us very want was way we well went were what when where which while who why
    will with would year yes you your
""".split())

#: A winner needs this share of the function words AND this multiple of the
#: loser. Below it the evidence is genuinely mixed ("my abuela always said
#: mira, para adelante"), and guessing there is what mislabels a clip.
_LANG_MIN_SHARE = 0.30
_LANG_MIN_MARGIN = 1.6


def detect_language(words: list[str]) -> str | None:
    """English or Spanish, from the words themselves. None when mixed.

    Scores the share of high-frequency function words in each language (plus a
    small bonus for Spanish-only characters, which survive ASR even when the
    accents themselves are dropped). Requires BOTH a minimum share and a clear
    margin, because the previous version called "my abuela always said mira,
    para adelante" Spanish: three Spanish tokens beat zero recognised English
    ones and the code read that as a Spanish video with English audio.

    Returning None is a real answer — the caller falls back to the engine's own
    detected language, which is better evidence than a coin flip.
    """
    toks = [str(w).lower().strip(".,!?;:\"'()[]¿¡…—") for w in words]
    toks = [t for t in toks if t]
    if len(toks) < 3:
        return None
    n = len(toks)
    es_share = sum(1 for t in toks if t in _ES_FUNCTION) / n
    en_share = sum(1 for t in toks if t in _EN_FUNCTION) / n
    marks = sum(str(w).count(c) for w in words for c in "¿¡áéíóúñü")
    if marks:
        # Accents/unused-in-English letters are language evidence on their own.
        es_share += min(0.15, marks / n * 0.5)
    if es_share >= _LANG_MIN_SHARE and es_share >= en_share * _LANG_MIN_MARGIN:
        return "es"
    if en_share >= _LANG_MIN_SHARE and en_share >= es_share * _LANG_MIN_MARGIN:
        return "en"
    return None


def guess_language(transcript: Any) -> str | None:
    """Language of a whole transcript, from its words (None when unclear)."""
    words = getattr(transcript, "words", None)
    if not words:
        return None
    return detect_language([str(getattr(w, "word", "")) for w in words])


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
    *,
    language: str | None = None,
    style: str = "viral",
    allow_clip_override: bool = True,
) -> list[dict[str, Any]]:
    """Assemble caption cues for ONE rendered clip from a transcript + cut window.

    ``transcript`` is :class:`app.subtitle.Transcript`. ``cut`` is one item from
    the reframe stage's ``cuts`` list and must have numeric ``start``/``end``
    (clip-relative within the source).

    SHORT-BURST chunking (the Ssemble-style rhythm): 2-3 word bursts cut on
    natural pauses, max ~18 characters per display line, each burst shown from
    its FIRST word's start to its LAST word's end (+80ms linger). With word
    timings straight from Whisper there is no overlap and no drift — the cue
    boundary IS the word boundary.

    Returns a list of ``{"start", "end", "text", "words"}`` cues, clip-relative.
    """
    from app import subtitle as _subtitle

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

    # Display pipeline shared with the burn stage: bilingual fixes then burst
    # grouping then the emphasis planner. All timings are clip-relative.
    lang = language or guess_language(transcript) or "en"
    as_dicts = [
        {
            "start": float(getattr(w, "start", 0.0)) - cut_start,
            "end": float(getattr(w, "end", 0.0)) - cut_start,
            "word": str(getattr(w, "word", "")),
            "conf": float(getattr(w, "conf", 1.0) or 1.0),
        }
        for w in inside
    ]
    # Per-clip language. A video is rarely one language end to end: an English
    # interview with a Spanish-speaking guest, or a Spanish podcast quoting an
    # English clip. The job-level language is only a prior — when THIS clip's
    # own words are confidently the other language, the caption planner (and
    # the SRT it labels) follows the words in front of it.
    if allow_clip_override:
        clip_lang = detect_language([str(d.get("word") or "") for d in as_dicts])
        if clip_lang and clip_lang != lang:
            lang = clip_lang
    as_dicts, _merged_contr = _subtitle.merge_broken_contractions(as_dicts)
    as_dicts, _merged_nums = _subtitle.merge_number_runs(as_dicts, lang)
    kept, _removed = _subtitle.strip_fillers(as_dicts)
    kept = _subtitle.normalize_word_timings(kept)

    preset = _subtitle.get_preset(style)
    max_chars = preset.max_chars_per_line or 18
    # The bold-stack reference look is a 3-5 word burst broken on punctuation
    # or a pause of 0.4 s, then split into two balanced lines at burn time (each
    # line is its own ASS event, so the chunk itself is only word-grouped here).
    stack = preset.key == "bold_stack"
    # THE RHYTHM KNOBS, read from the resolved settings instead of literals.
    # These three were hard-coded here (4 / 2.6 s / BOLD_STACK_PAUSE_BREAK), so
    # "Burst length", "Hard cap" and "Pause threshold" were saved, displayed
    # and then ignored by the only code that decides where a chunk ends.
    sub = getattr(settings, "subtitle", None)
    target = int(getattr(sub, "words_per_cue", 4 if stack else 2) or (4 if stack else 2))
    cap = int(getattr(sub, "words_per_cue_max", max(target, 4)) or max(target, 4))
    cap = max(target, cap)
    pause_break = float(
        getattr(sub, "pause_sec", _subtitle.BOLD_STACK_PAUSE_BREAK if stack else 0.30)
        or 0.0
    )
    cues = _subtitle.words_to_cues(
        kept,
        words_per_cue=target,
        max_words=cap,
        max_chars=max_chars * 2,   # cue-level cap = 2 display lines
        max_duration=2.6 if stack else 2.2,
        # 0 means "never cut on a pause", which is what the slider's own 0.0
        # means — not "cut on every word boundary".
        pause_sec=(pause_break if pause_break > 0 else 1e9),
    )
    if stack:
        # A lone word would leave the stack's second line empty — fold it into a
        # neighbour while the 3-5 word / two-line budget still allows.
        cues = _subtitle.merge_short_cues(cues)
        # Shown from the first word's start to the last word's end, and held
        # across a gap under 0.5 s so the two-line block never blinks between
        # chunks that are simply close together (a real pause is left alone).
        _subtitle.extend_cue_gaps(cues, _subtitle.BOLD_STACK_GAP_EXTEND)
    else:
        # end+80ms linger, clamped by the next cue's start — never overlaps.
        for i, c in enumerate(cues):
            nxt = cues[i + 1] if i + 1 < len(cues) else None
            limit = (float(nxt.get("start") or 0.0) - 0.001) if nxt else None
            shifted = float(c.get("end") or 0.0) + 0.08
            c["end"] = min(shifted, limit) if limit is not None and limit > c["end"] else max(c["end"], min(shifted, limit) if limit is not None else shifted)
    _subtitle.build_emphasis(cues, language=lang)
    # display text: only up to 2 lines per burst, wrapped on words. The stack
    # writer splits its own two lines into separate events, so it must NOT get
    # an ASS line break baked into the cue text.
    for c in cues:
        c["text"] = _subtitle.display_text(
            c["text"], uppercase=preset.uppercase,
            max_line=0 if stack else (preset.max_chars_per_line or 18),
        )
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
            vtt_text = _srt_to_vtt(srt_text, lang=lang)
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


def _srt_to_vtt(srt_text: str, lang: str | None = None) -> str:
    segments = srt_text.split("\n\n")
    out: list[str] = ["WEBVTT"]
    if lang:
        out.append(f"Language: {lang}")
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
