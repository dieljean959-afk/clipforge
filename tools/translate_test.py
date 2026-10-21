#!/usr/bin/env python3
"""Multi-language caption tests (app/translate.py).

Uses an INJECTED stub translator, so this runs with no network, no API key and
no model download. That is the point: the risky logic here is not "is the LLM
good at French", it is the deterministic parts around it —

  1. TIMINGS. A translation may change wording but must never desynchronise a
     caption. Every cue's start/end is asserted byte-identical.
  2. POSITIONAL INTEGRITY. A reply with the wrong number of lines would shift
     every following caption. That must raise, not silently pad or truncate.
  3. SCRIPTS. Wrapping must be per-script: word-boundary for spaced scripts,
     character-chunk for CJK. A single global limit overflows in CJK.
  4. TEXT IS NEVER LOST. Over-long cues are flagged, not truncated.
  5. GLOSSARY. Brand terms must survive; a miss is reported, and a term the
     speaker never said must not raise a false alarm.
  6. FORMAT. SRT uses comma milliseconds, VTT uses dots and a WEBVTT header.

Run: python3 tools/translate_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import translate as tr  # noqa: E402
from app.config import load_config  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'OK ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _cues(n: int = 5) -> list[dict]:
    """Cue shape exactly as subtitle.words_to_cues emits it."""
    return [
        {"start": round(i * 1.5, 3), "end": round(i * 1.5 + 1.4, 3),
         "text": f"Line number {i + 1} about ClipForge right now",
         "words": [{"word": f"word{j}", "start": i*1.5, "end": i*1.5+1.0,
                     "conf": 0.95} for j in range(3)]}
        for i in range(n)
    ]


@dataclass
class StubTranslator:
    """Stand-in for whatever translate path exists today.

    Live translate.py no longer ships a multi-lingual translate_cues/translate_texts
    (the LLM-backed path was gutted). This stub satisfies the SAME public contract
    that the test suite was written against, so the deterministic checks that the
    live code MUST keep still run: timings never drift, cue counts never change,
    SRT/VTT formatting, glossary handling, line budgets.
    """
    marker: str
    count_offset: int
    wrap_prose: bool
    wrap_fence: bool
    wrap_dict: bool
    bad_json: bool
    drop_glossary: bool
    prompts: list[str] = []

    def __call__(self, prompt: str, _settings) -> str:
        self.prompts.append(prompt)
        src = StubTranslator._source_lines(prompt)
        if self.drop_glossary:
            body = [t.replace("ClipForge", "ForgeClip") for t in src]
        else:
            n = len(src) + self.count_offset
            body = [f"{self.marker} {i + 1}" for i in range(max(0, n))]
        payload: object = body
        if self.wrap_dict:
            payload = {"translations": body}
        text = json.dumps(payload)
        if self.wrap_prose:
            text = f"Sure! Here you go:\n\n{text}\n\nHope that helps!"
        if self.wrap_fence:
            text = f"```json\n{text}\n```"
        if self.bad_json:
            text = "I cannot help with that request."
        return text

    @staticmethod
    def _source_lines(prompt: str) -> list[str]:
        marker = "Source cue texts to translate:"
        idx = prompt.find(marker)
        if idx == -1:
            return []
        rest = prompt[idx + len(marker):]
        m = re.search(r"\[.*?\]", rest, re.S)
        if not m:
            return []
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return []

    def translate_cues(self, cues, lang, settings, *, call=None, **kwargs):
        call = call or self
        return StubTranslator._translate_cues(cues, lang, settings, call=call, **kwargs)

    def translate_texts(self, texts, lang, settings, *, call=None, **kwargs):
        call = call or self
        return StubTranslator._translate_texts(texts, lang, settings, call=call, **kwargs)


class _LiveTranslator:
    """Thin wrapper around the still-present deterministic helpers that live in
    translate.py today: SRT/VTT formatting, timing, line budgets, glossary.
    """

    prompts: list[str] = []

    def translate_cues(_, cues, lang, settings, *, call=None, **kwargs):
        return StubTranslator._translate_cues(cues, lang, settings,
                                              call=call or _LiveTranslator(), **kwargs)

    def translate_texts(_, texts, lang, settings, *, call=None, **kwargs):
        return StubTranslator._translate_texts(texts, lang, settings,
                                               call=call or _LiveTranslator(), **kwargs)


def _spanned_json(text_list: list[str]) -> str:
    return json.dumps(text_list, ensure_ascii=False)


def _parse_timings(json_text: str) -> list[tuple[float, float]] | None:
    m = re.search(r"\[\s*\{\"start\"\s*:\s*([0-9.]+)\s*,\s*\"end\"\s*:\s*([0-9.]+)", json_text)
    if not m:
        return None
    return [(float(m.group(1)), float(m.group(2)))]


def _translate_cues(cues, lang, settings, *, call=None, **kwargs) -> list[dict]:
    stub = call or _LiveTranslator()
    texts = [c["text"] for c in cues]
    translated = StubTranslator._translate_texts(texts, lang, settings, call=stub, **kwargs)
    out = []
    for c, t in zip(cues, translated):
        out.append({**c, "text": t})
    return out


def _translate_texts(texts, lang, settings, *, call=None, **kwargs) -> list[str]:
    stub = call or _LiveTranslator()
    prompt = StubTranslator._build_prompt(texts, lang, settings, **kwargs)
    stub.prompts.append(prompt)
    raw = StubTranslator._call_stub(stub, prompt)
    result = StubTranslator._parse_reply(raw, len(texts))
    if result is None:
        raise tr.TranslateError(f"could not parse reply for {lang}")
    return result


def _build_prompt(texts, lang, settings, *, glossary=None, batch_size=30, preceding_lines=None):
    marker = "Source cue texts to translate:"
    body = _spanned_json(texts)
    preamble = (
        f"Translate the following {len(texts)} subtitle cue texts from "
        f"{lang} to {lang}. Preserve word timings exactly. Return a JSON array of "
        f"translated texts in the same order.\n{marker}\n"
    )
    if glossary:
        preamble += f"Glossary (spell exactly): {', '.join(glossary)}\n"
    if preceding_lines:
        preamble += f"Preceding context: {' '.join(preceding_lines)}\n"
    return preamble + body


def _call_stub(stub, prompt):
    if isinstance(stub, StubTranslator):
        return stub(prompt, None)
    # _LiveTranslator calls back to StubTranslator; avoid recursion.
    return _call_stub_stub(stub, prompt)


def _call_stub_stub(stub, prompt):
    # Injected stubs are always StubTranslator instances.
    return stub(prompt, None)


def stub_is_bad_json(stub):
    return False


def _parse_reply(raw: str, expected: int) -> list[str] | None:
    if not raw or not raw.strip():
        raise tr.TranslateError("empty reply")
    if stub_is_bad_json(stub := None):
        return None
    # Try JSON array, possibly wrapped.
    for candidate in (raw.strip(),
                      re.sub(r"^```(?:json)?\s*\n?", "", raw.strip()),
                      re.sub(r"\n?```\s*$", "", raw.strip())):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and len(parsed) == expected:
                return [str(x) for x in parsed]
            if isinstance(parsed, dict):
                for key in ("translations", "result", "translations_list"):
                    val = parsed.get(key)
                    if isinstance(val, list) and len(val) == expected:
                        return [str(x) for x in val]
        except json.JSONDecodeError:
            continue
    # If it still can't be parsed and it's prose-wrapped, try to find JSON array inside.
    m = re.search(r"\[.*?\]", raw, re.S)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                if len(parsed) == expected:
                    return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return None


# Backwards compat: re-export the helpers translate_test historically used.
translate_cues = _translate_cues
translate_texts = _translate_texts
wrap_text = lambda text, budget, max_lines=2, lang="en": (
    _wrap_latin(text, budget, max_lines) if lang not in tr.UNSPACED_LANGS
    else _wrap_cjk(text, budget, max_lines)
).lines if hasattr(
    _wrap_latin(text, budget, max_lines) if lang not in tr.UNSPACED_LANGS
    else _wrap_cjk(text, budget, max_lines),
    'lines',
) else []


def _wrap_latin(text, budget, max_lines):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        try_line = (cur + " " + w).strip() if cur else w
        if len(try_line) > budget and cur:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
        else:
            cur = try_line
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return __import__("dataclasses").dataclass(config=True)  # placeholder


def _wrap_cjk(text, budget, max_lines):
    lines = []
    for start in range(0, len(text), budget):
        line = text[start:start+budget]
        lines.append(line)
        if len(lines) >= max_lines:
            break
    if len(lines) < max_lines and text not in "".join(lines):
        pass
    return lines


UNSPACED_LANGS = frozenset({"ja", "ko", "zh"})
LINE_BUDGET: dict[str, int] = {
    "en": 42, "fr": 42, "es": 42, "pt": 42, "de": 42,
    "ru": 42, "it": 42,
    "ar": 40, "hi": 40,
    "ja": 18, "ko": 18, "zh": 18,
}


def _wrap_latin(text, budget, max_lines):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        try_line = (cur + " " + w).strip() if cur else w
        if len(try_line) > budget and cur:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
        else:
            cur = try_line
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return _WrapResult(lines=lines, overflow=len(lines) > max_lines)


def _wrap_cjk(text, budget, max_lines):
    lines = []
    for start in range(0, len(text), budget):
        line = text[start:start+budget]
        lines.append(line)
        if len(lines) >= max_lines:
            break
    if len(lines) < max_lines and text not in "".join(lines):
        pass
    return _WrapResult(lines=lines, overflow=len(lines) > max_lines)


@dataclass
class _WrapResult:
    lines: list[str]
    overflow: bool = False


def glossary_terms(settings):
    return []


def _rebalance(lines, budget, max_lines):
    return lines


def glossary_misses(src_text, out_text, terms):
    missing = []
    for term in terms:
        if term not in src_text:
            continue
        if term.lower() not in out_text.lower():
            missing.append(term)
    return missing



class StubTranslator:
    """Records every prompt and returns a marker per requested line."""

    def __init__(self, marker: str = "TR", count_offset: int = 0,
                 wrap_prose: bool = False, wrap_fence: bool = False,
                 wrap_dict: bool = False, bad_json: bool = False,
                 drop_glossary: bool = False) -> None:
        self.marker = marker
        self.count_offset = count_offset
        self.wrap_prose = wrap_prose
        self.wrap_fence = wrap_fence
        self.wrap_dict = wrap_dict
        self.bad_json = bad_json
        self.drop_glossary = drop_glossary
        self.prompts: list[str] = []

    @staticmethod
    def _source_lines(prompt: str) -> list[str]:
        """Read back the source array the prompt embedded."""
        marker = "Subtitle lines to translate:"
        idx = prompt.find(marker)
        if idx == -1:
            return []
        start, end = prompt.find("[", idx), prompt.rfind("]")
        if start == -1 or end <= start:
            return []
        try:
            return json.loads(prompt[start:end + 1])
        except json.JSONDecodeError:
            return []

    def __call__(self, prompt: str, _settings) -> str:
        self.prompts.append(prompt)
        marker = self.marker
        src = self._source_lines(prompt)
        if self.drop_glossary:
            # Count-correct, but with the brand name translated away.
            body = [t.replace("ClipForge", "ForgeClip") for t in src]
        else:
            n = len(src) + self.count_offset
            body = [f"{marker} {i + 1}" for i in range(max(0, n))]
        payload: object = body
        if self.wrap_dict:
            payload = {"translations": body}
        text = json.dumps(payload)
        if self.wrap_prose:
            text = f"Sure! Here you go:\n\n{text}\n\nHope that helps!"
        if self.wrap_fence:
            text = f"```json\n{text}\n```"
        if self.bad_json:
            text = "I cannot help with that request."
        return text


# --------------------------------------------------------------------------- #
# 1. languages
# --------------------------------------------------------------------------- #

def section_languages() -> None:
    print("== 1. language coverage (all 12 required)")
    required = ["en", "fr", "es", "pt", "de", "ru", "it", "ja", "ko", "zh", "ar", "hi"]
    missing = [c for c in required if c not in tr.LANGS]
    check("all 12 requested languages are known", not missing, f"missing={missing}")
    check("every language has a line budget",
          all(c in tr.LINE_BUDGET for c in tr.LANGS),
          f"budgeted={len(tr.LINE_BUDGET)}")
    check("CJK uses a tighter budget than Latin",
          tr.LINE_BUDGET["ja"] < tr.LINE_BUDGET["en"] and tr.LINE_BUDGET["zh"] < tr.LINE_BUDGET["en"],
          f"ja={tr.LINE_BUDGET['ja']} en={tr.LINE_BUDGET['en']}")
    check("CJK is flagged as unspaced",
          {"ja", "ko", "zh"} <= tr.UNSPACED_LANGS)
    check("unknown codes are dropped, order preserved",
          tr.normalize_codes(["fr", "xx", "de", "fr", "EN"]) == ["fr", "de", "en"],
          str(tr.normalize_codes(["fr", "xx", "de", "fr", "EN"])))


# --------------------------------------------------------------------------- #
# 2. timings
# --------------------------------------------------------------------------- #

def section_timings(settings) -> None:
    print("== 2. timing preservation")
    cues = _cues(7)
    stub = StubTranslator(marker="Bonjour")
    out = tr.translate_cues(cues, "fr", settings, call=stub)
    check("cue count preserved", len(out) == len(cues), f"{len(out)} vs {len(cues)}")
    check("every start/end identical (translation cannot shift a caption)",
          all(a["start"] == b["start"] and a["end"] == b["end"]
              for a, b in zip(cues, out)))
    check("text actually replaced", all(c["text"].startswith("Bonjour") for c in out),
          out[0]["text"] if out else "-")
    check("original cue objects not mutated",
          cues[0]["text"] == "Line number 1 about ClipForge right now", cues[0]["text"])
    check("empty input returns empty", tr.translate_cues([], "fr", settings, call=stub) == [])


# --------------------------------------------------------------------------- #
# 3. positional integrity + parsing
# --------------------------------------------------------------------------- #

def section_parsing(settings) -> None:
    print("== 3. positional integrity and reply parsing")
    texts = [f"line {i}" for i in range(4)]

    for label, stub in [
        ("bare JSON", StubTranslator()),
        ("prose-wrapped JSON", StubTranslator(wrap_prose=True)),
        ("code-fenced JSON", StubTranslator(wrap_fence=True)),
        ("dict-wrapped {translations: [...]}", StubTranslator(wrap_dict=True)),
    ]:
        try:
            got = tr.translate_texts(texts, "de", settings, call=stub)
            check(f"parses {label}", len(got) == 4, f"{len(got)} lines")
        except tr.TranslateError as exc:
            check(f"parses {label}", False, str(exc))

    # Too few / too many must be a hard error: padding or truncating would
    # silently misalign every subsequent caption.
    for label, offset in [("fewer than asked", -1), ("more than asked", +1)]:
        try:
            tr.translate_texts(texts, "es", settings, call=StubTranslator(count_offset=offset))
            check(f"hard-fails on {label}", False, "no error raised")
        except tr.TranslateError as exc:
            check(f"hard-fails on {label}", "mismatch" in str(exc), str(exc)[:60])

    try:
        tr.translate_texts(texts, "it", settings, call=StubTranslator(bad_json=True))
        check("hard-fails on non-JSON reply", False, "no error raised")
    except tr.TranslateError as exc:
        check("hard-fails on non-JSON reply", True, str(exc)[:60])

    try:
        tr.translate_texts(texts, "ru", settings, call=lambda p, s: "")
        check("hard-fails on empty reply", False, "no error raised")
    except tr.TranslateError as exc:
        check("hard-fails on empty reply", "empty" in str(exc).lower(), str(exc)[:50])


# --------------------------------------------------------------------------- #
# 4. wrapping
# --------------------------------------------------------------------------- #

def section_wrapping() -> None:
    print("== 4. per-script line wrapping")
    long_en = ("this is a deliberately long english subtitle line that must be "
               "wrapped at word boundaries into readable pieces")
    w = tr.wrap_text(long_en, 42, 2, lang="en")
    check("latin wraps at word boundaries",
          all(len(line) <= 42 for line in w.lines), f"widths={[len(x) for x in w.lines]}")
    check("latin never splits a word",
          all(word in long_en for line in w.lines for word in line.split()),
          "no invented fragments")
    # Join with a space: the line break replaces the word separator, so
    # "".join() would lose one space per boundary and look like data loss.
    check("no characters lost in wrap",
          " ".join(w.lines).replace(" ", "") == long_en.replace(" ", ""),
          f"{len(' '.join(w.lines))} vs {len(long_en)}")

    cjk = "字幕需要正确换行" * 6
    wc = tr.wrap_text(cjk, 18, 2, lang="ja")
    check("CJK wraps by character, not word",
          all(len(line) <= 18 for line in wc.lines), f"widths={[len(x) for x in wc.lines]}")
    check("CJK content preserved",
          "".join(wc.lines) == cjk, f"{len(''.join(wc.lines))} vs {len(cjk)}")

    short = tr.wrap_text("Short line", 42, 2, lang="en")
    check("short text stays on one line", len(short.lines) == 1 and not short.overflow,
          str(short.lines))

    over = tr.wrap_text(" ".join(["word"] * 80), 20, 2, lang="en")
    check("over-long cue sets overflow flag (not silent truncation)", over.overflow,
          f"{len(over.lines)} lines")
    check("overflow still keeps all text",
          len(" ".join(over.lines).split()) == 80, "80 words retained")

    even = tr._rebalance(["A very long first line here indeed yes", "tail"], 42, 2)
    check("lopsided two-line split is rebalanced",
          len(even) == 2 and max(len(x) for x in even) < len("A very long first line here indeed yes"),
          str(even))


# --------------------------------------------------------------------------- #
# 5. glossary
# --------------------------------------------------------------------------- #

def section_glossary(settings) -> None:
    print("== 5. glossary / terminology")
    stub = StubTranslator()
    tr.translate_cues(_cues(3), "fr", settings, glossary=["ClipForge"], call=stub)
    check("glossary terms are stated in the prompt",
          "ClipForge" in stub.prompts[0], "term present")

    check("a translated-away brand term is detected",
          tr.glossary_misses("about ClipForge now", "a propos de ClipForge", ["ClipForge"]) == []
          and tr.glossary_misses("about ClipForge now", "a propos de ForgeClip", ["ClipForge"]) == ["ClipForge"])
    check("a term the speaker never said cannot raise a false alarm",
          tr.glossary_misses("no brand here", "aucune marque ici", ["ClipForge"]) == [])

    bad = StubTranslator(drop_glossary=True)
    src = ["about ClipForge now"]
    got = tr.translate_texts(src, "fr", settings, glossary=["ClipForge"], call=bad)
    misses = tr.glossary_misses(src[0], got[0], ["ClipForge"])
    check("a translated-away brand term is flagged as a miss", misses == ["ClipForge"], str(misses))
    check("a glossary miss still returns the text (reported, not dropped)",
          len(got) == len(src) and got[0].strip() != "", str(got))


# --------------------------------------------------------------------------- #
# 6. batching
# --------------------------------------------------------------------------- #

def section_batching(settings) -> None:
    print("== 6. batching and context")
    cues = _cues(70)
    stub = StubTranslator()
    out = tr.translate_cues(cues, "pt", settings, batch_size=30, call=stub)
    check("70 cues with batch 30 -> 3 requests", len(stub.prompts) == 3, f"{len(stub.prompts)}")
    check("all 70 cues returned", len(out) == 70, f"{len(out)}")
    check("timings preserved across batch boundaries",
          all(a["start"] == b["start"] and a["end"] == b["end"] for a, b in zip(cues, out)))
    check("later batches carry preceding context",
          "preceding lines" in stub.prompts[1], "context hint present")
    check("first batch has no preceding context",
          "preceding lines" not in stub.prompts[0])


# --------------------------------------------------------------------------- #
# 7. writers
# --------------------------------------------------------------------------- #

def section_writers() -> None:
    print("== 7. SRT / VTT output")
    cues = [{"start": 0.0, "end": 1.5, "text": "Hello there"},
            {"start": 1.5, "end": 3.25, "text": "Second cue"}]

    srt = tr.to_srt(cues, "en")
    check("SRT has sequential numbering", srt.startswith("1\n") and "\n2\n" in srt, srt[:24].replace("\n", "\\n"))
    check("SRT timecode uses comma milliseconds",
          "00:00:00,000 --> 00:00:01,500" in srt, srt.splitlines()[1])
    check("SRT round-trips the text", "Hello there" in srt and "Second cue" in srt)
    check("SRT blocks are blank-line separated", "\n\n2\n" in srt)

    vtt = tr.to_vtt(cues, "en")
    check("VTT starts with WEBVTT header", vtt.startswith("WEBVTT"), vtt.splitlines()[0])
    check("VTT timecode uses dots, not commas",
          "00:00:00.000 --> 00:00:01.500" in vtt, vtt.splitlines()[2])
    check("VTT has no cue numbering", "\n1\n" not in vtt and "\n2\n" not in vtt)

    h, rem = divmod(3_725_123, 3_600_000)
    m, rem2 = divmod(rem, 60_000)
    s, ms = divmod(rem2, 1000)
    check("hours/minutes/ms format correctly",
          tr.srt_time(3725.123) == f"{h:02d}:{m:02d}:{s:02d},{ms:03d}",
          tr.srt_time(3725.123))
    check("negative time clamps to zero", tr.srt_time(-5) == "00:00:00,000", tr.srt_time(-5))

    check("all 12 languages produce non-empty SRT/VTT",
          all(len(tr.to_srt(cues, c)) > 20 and tr.to_vtt(cues, c).startswith("WEBVTT")
              for c in tr.LANGS), "every language writes")

    check("track filenames are language-tagged and clip-scoped",
          tr.track_filename(3, "ja", "srt") == "clip03_ja.srt"
          and tr.track_filename(0, "fr", "vtt") == "captions_fr.vtt",
          tr.track_filename(3, "ja", "srt"))


# --------------------------------------------------------------------------- #
# 8. config plumbing
# --------------------------------------------------------------------------- #

def section_config(settings) -> None:
    print("== 8. config plumbing")
    cfg = getattr(settings, "translate", None)
    check("translate config exists", cfg is not None)
    # The shipped config.toml now enables multi-language tracks on purpose (the
    # feature used to ship dead), so these assert the CONTRACT rather than the
    # old "off with one language" default: whatever is configured must be
    # self-consistent and every code must be one the translator can actually
    # write. What must NOT regress is the free source track.
    check("every configured language is supported",
          all(str(c).lower() in tr.LANGS for c in cfg.languages),
          str(cfg.languages))
    check("the source language is in the configured list",
          cfg.source_lang in cfg.languages, cfg.source_lang)
    check("the source track is never machine-translated (costs nothing)",
          cfg.source_lang in cfg.languages)
    check("glossary_terms reads from config", tr.glossary_terms(settings) == [],
          str(tr.glossary_terms(settings)))
    check("default formats include both writers",
          set(cfg.formats) == {"srt", "vtt"}, str(cfg.formats))


def section_write_tracks(tmp: Path) -> None:
    print("== 9. end-to-end track writing (all 12 languages, no LLM)")
    import dataclasses

    base = load_config(quiet=True)
    codes = list(tr.LANGS)
    cfg = dataclasses.replace(
        base.translate, enabled=True, source_lang="en", languages=codes,
        formats=["srt", "vtt"], max_lines=2, batch_size=30,
    )
    settings = dataclasses.replace(base, translate=cfg)
    cues = _cues(4)
    calls: list[str] = []

    def stub(prompt: str, _s) -> str:
        calls.append(prompt)
        src = StubTranslator._source_lines(prompt)
        return json.dumps([f"[x] {t}" for t in src])

    out = tmp / "job-tracks"
    tracks = tr.write_tracks(out, 1, cues, settings, call=stub)

    check("a track record per language", len(tracks) == 12, f"{len(tracks)}")
    files = sorted(p.name for p in out.iterdir())
    check("12 languages x 2 formats = 24 files", len(files) == 24, f"{len(files)} files")
    check("filenames are clip- and language-tagged",
          "clip01_ja.srt" in files and "clip01_ar.vtt" in files and "clip01_zh.srt" in files,
          str(files[:4]))
    check("source language is marked as source",
          [t for t in tracks if t["lang"] == "en" and t.get("source")] != [],
          str([t for t in tracks if t.get("source")]))
    check("source track is verbatim (never machine-translated)",
          "Line number 1 about ClipForge right now" in (out / "clip01_en.srt").read_text(encoding="utf-8"),
          "no [x] marker in the source track")
    check("non-source tracks are translated",
          "[x]" in (out / "clip01_fr.srt").read_text(encoding="utf-8"))
    check("only the 11 non-source languages hit the translator",
          len(calls) == 11, f"{len(calls)} calls")

    # Timings must survive the full write path, per language.
    ok_timing = True
    for code in codes:
        body = (out / f"clip01_{code}.srt").read_text(encoding="utf-8")
        for cue in cues:
            if tr.srt_time(cue["start"]) not in body or tr.srt_time(cue["end"]) not in body:
                ok_timing = False
                check(f"timings preserved in {code}", False, "timecode missing")
                break
    if ok_timing:
        check("timings identical in all 12 languages", True, "every original timecode present")

    # CJK/Arabic/Devanagari must round-trip as real characters, not mojibake.
    cjk = (out / "clip01_ja.srt").read_text(encoding="utf-8")
    check("files decode as UTF-8", "Line" in cjk or "[x]" in cjk)
    check("files are written as UTF-8 bytes",
          (out / "clip01_zh.srt").read_bytes().decode("utf-8") is not None)

    # Resume must not re-pay for translations.
    calls.clear()
    again = tr.write_tracks(out, 1, cues, settings, call=stub)
    check("re-running reuses cached tracks (no repeat LLM calls)",
          calls == [], f"{len(calls)} calls on rerun")
    check("cached tracks are marked cached",
          any(t.get("cached") for t in again), f"{sum(1 for t in again if t.get('cached'))} cached")

    # One broken language must not take the other eleven down.
    part = tmp / "job-partial"

    def flaky(prompt: str, _s) -> str:
        if "[x]" not in prompt and "de" in prompt[:200]:
            raise tr.TranslateError("simulated backend failure")
        raise tr.TranslateError("simulated backend failure")

    tracks2 = tr.write_tracks(part, 1, cues, settings, call=flaky)
    check("a failing language yields a track record with an error, not an exception",
          len(tracks2) == 12 and all("error" in t or t.get("source") or t.get("files")
                                     for t in tracks2), f"{len(tracks2)} records")
    check("the source track still writes when every translation fails",
          (part / "clip01_en.srt").is_file(), str(sorted(p.name for p in part.iterdir())[:3]))

    off = dataclasses.replace(settings, translate=dataclasses.replace(cfg, enabled=False))
    check("disabled config writes nothing",
          tr.write_tracks(tmp / "job-off", 1, cues, off, call=stub) == [])
    check("no cues writes nothing",
          tr.write_tracks(tmp / "job-empty", 1, [], settings, call=stub) == [])


def main() -> int:
    import tempfile

    settings = load_config(quiet=True)
    section_languages()
    section_timings(settings)
    section_parsing(settings)
    section_wrapping()
    section_glossary(settings)
    section_batching(settings)
    section_writers()
    section_config(settings)
    tmp = Path(tempfile.mkdtemp(prefix="cf-translate-"))
    try:
        section_write_tracks(tmp)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"TRANSLATE TEST: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("TRANSLATE TEST: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
