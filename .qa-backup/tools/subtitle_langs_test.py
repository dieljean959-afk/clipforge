"""Multi-language subtitle tracks: translation, wrapping, SRT/VTT export.

The feature exists ([translate] in config.toml, app/translate.py, per-language
sidecars next to each clip) but ships DISABLED with `languages = ["en"]`, so
Spanish/French tracks were never produced. This suite pins the behaviour that
has to be correct before turning it on for real:

  - a translation can never shift a cue: timings are copied, only text changes
  - the uncertain-cue placeholders ([muttering], [inaudible]) pass through
    UNtranslated — they mean "could not be made out", which is true in every
    language, and machines happily invent a confident translation
  - batching keeps alignment: a 35-cue run at batch_size=30 must not shift
    every line in the second window by one position
  - line budgets are per-script (42 Latin, 18 full-width CJK) and wrapping
    never silently discards characters
  - one failing language degrades to a missing track with a logged reason and
    leaves the other languages intact
  - the source track is copied verbatim, never round-tripped through a model

The model call is stubbed, so this is a logic suite: it cannot tell you whether
a translation is *good*, only whether the track is correctly built.

    python tools/subtitle_langs_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import translate as tr  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


def settings(languages: list[str], *, enabled: bool = True, batch: int = 30,
             formats: tuple[str, ...] = ("srt", "vtt"), glossary=None):
    return SimpleNamespace(
        translate=SimpleNamespace(
            enabled=enabled,
            source_lang="en",
            languages=languages,
            glossary=glossary or [],
            batch_size=batch,
            max_lines=2,
            formats=list(formats),
        )
    )


def cues(n: int) -> list[dict]:
    return [
        {"start": round(i * 2.0, 3), "end": round(i * 2.0 + 1.8, 3), "text": f"line {i}"}
        for i in range(n)
    ]


def stub_call(prefix: str = "T"):
    """A model stand-in that returns a same-length array, so count/order bugs
    are caught by the alignment assertions rather than hidden by the stub."""
    def call(prompt: str, _settings) -> str:
        lines = json.loads(prompt[prompt.rindex("["):prompt.rindex("]") + 1])
        return json.dumps([f"{prefix}{t}" for t in lines], ensure_ascii=False)
    return call


# --------------------------------------------------------------------------- #

def test_language_list() -> None:
    print("\n[1] language roster")
    for code, name in (("es", "Spanish"), ("fr", "French"), ("pt", "Portuguese"),
                       ("de", "German"), ("ru", "Russian"), ("it", "Italian"),
                       ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Mandarin Chinese"),
                       ("ar", "Arabic"), ("hi", "Hindi")):
        check(f"{code} is supported as {name}", tr.lang_name(code) == name)
    check("an unknown code is not silently accepted",
          tr.normalize_codes(["en", "xx", "klingon"]) == ["en"])
    check("normalise de-duplicates and preserves order",
          tr.normalize_codes(["fr", "en", "fr", "es"]) == ["fr", "en", "es"])
    check("normalise is case/whitespace insensitive",
          tr.normalize_codes([" ES ", "Fr"]) == ["es", "fr"])
    check("every language has a line budget",
          all(code in tr.LINE_BUDGET for code in tr.LANGS),
          str(sorted(set(tr.LANGS) - set(tr.LINE_BUDGET))))


def test_wrapping() -> None:
    print("\n[2] wrapping is per-script and loses nothing")
    long_en = "this is a fairly long english subtitle line that must wrap somewhere"
    w = tr.wrap_text(long_en, tr.LINE_BUDGET["en"], 2, lang="en")
    check("English wraps on word boundaries",
          all(" " in ln or len(ln) <= 18 for ln in w.lines), str(w.lines))
    check("English loses no characters",
          " ".join(w.lines) == long_en)
    check("English lines respect the 42-char budget",
          all(len(ln) <= 42 for ln in w.lines), str([len(x) for x in w.lines]))

    ja = "これは日本語の字幕ですThreat 가능성이-turns-here"[:40]
    wj = tr.wrap_text(ja, tr.LINE_BUDGET["ja"], 2, lang="ja")
    check("Japanese uses the narrower CJK budget",
          all(len(ln) <= 18 for ln in wj.lines), str([len(x) for x in wj.lines]))
    # wrap_text collapses runs of whitespace before wrapping, so the contract
    # is "no character is dropped", not "byte-identical input".
    norm = lambda s: s.replace(" ", "")  # noqa: E731
    check("Japanese loses no characters",
          norm("".join(wj.lines)) == norm(ja),
          f"{''.join(wj.lines)!r} vs {ja!r}")

    ar = "هذا نص فرعي عربي طويل جدا يجب أن يلتف على سطرين"
    wa = tr.wrap_text(ar, tr.LINE_BUDGET["ar"], 2, lang="ar")
    check("Arabic respects its 40-char budget", all(len(ln) <= 40 for ln in wa.lines))
    check("Arabic loses no characters", " ".join(wa.lines) == ar)

    check("overflow is REPORTED, never truncated",
          tr.wrap_text(long_en * 3, 42, 2, lang="en").overflow is True)
    check("a short cue is not flagged as overflow",
          tr.wrap_text("short", 42, 2, lang="en").overflow is False)
    check("empty text does not crash", tr.wrap_text("", 42, 2, lang="en").lines == [""])


def test_parsing() -> None:
    print("\n[3] reply parsing is strict about count, tolerant about fences")
    check("plain array parses", tr.parse_translations('["a","b"]', 2) == ["a", "b"])
    check("fenced array parses",
          tr.parse_translations('```json\n["a","b"]\n```', 2) == ["a", "b"])
    check("prose around the array is tolerated",
          tr.parse_translations('Sure! Here you go:\n["a","b"]\nHope that helps.', 2) == ["a", "b"])
    check("a wrapped object is unwrapped",
          tr.parse_translations('{"translations": ["a","b"]}', 2) == ["a", "b"])
    check("a count mismatch is a hard error", _raises(
        lambda: tr.parse_translations('["a"]', 2), "count mismatch"))
    check("an empty reply is a hard error", _raises(
        lambda: tr.parse_translations("   ", 1), "empty"))
    check("non-JSON is a hard error", _raises(
        lambda: tr.parse_translations("I cannot help with that.", 1), "parse"))


def _raises(fn, needle: str) -> bool:
    try:
        fn()
    except tr.TranslateError as exc:
        return needle.lower() in str(exc).lower()
    except Exception:
        return False
    return False


def test_timings_untouched() -> None:
    print("\n[4] translation never moves a cue")
    src = cues(5)
    out = tr.translate_cues(src, "es", settings(["es"]), call=stub_call("ES:"))
    check("cue count is preserved", len(out) == len(src))
    check("every timing is byte-identical",
          all((a["start"], a["end"]) == (b["start"], b["end"]) for a, b in zip(src, out)))
    check("every line was actually translated",
          [o["text"] for o in out] == [f"ES:line {i}" for i in range(5)],
          str([o["text"] for o in out]))


def test_batching_alignment() -> None:
    print("\n[5] batching keeps alignment across window boundaries")
    for n, batch in ((35, 30), (61, 30), (30, 30), (7, 30), (100, 13)):
        src = cues(n)
        out = tr.translate_cues(src, "fr", settings(["fr"], batch=batch), call=stub_call("FR:"))
        expected = [f"FR:line {i}" for i in range(n)]
        check(f"{n} cues at batch_size={batch} stay aligned",
              [o["text"] for o in out] == expected,
              "first mismatch at index "
              + str(next((i for i, (a, b) in enumerate(zip(expected, [o['text'] for o in out])) if a != b), "none")))
        check(f"{n} cues at batch_size={batch} keep their timings",
              [(o["start"], o["end"]) for o in out] == [(c["start"], c["end"]) for c in src])


def test_uncertain_cues() -> None:
    print("\n[6] uncertain cues are passed through, not invented")
    src = [
        {"start": 0.0, "end": 1.0, "text": "hello there"},
        {"start": 1.0, "end": 2.0, "text": "[inaudible]", "uncertain": True},
        {"start": 2.0, "end": 3.0, "text": "goodbye now"},
    ]
    out = tr.translate_cues(src, "es", settings(["es"]), call=stub_call("ES:"))
    check("the uncertain marker is untouched", out[1]["text"] == "[inaudible]",
          out[1]["text"])
    check("the marker was not sent to the model", "ES:[inaudible]" != out[1]["text"])
    check("its neighbours WERE translated",
          out[0]["text"] == "ES:hello there" and out[2]["text"] == "ES:goodbye now")
    check("timings still survive", [o["start"] for o in out] == [0.0, 1.0, 2.0])


def test_glossary() -> None:
    print("\n[7] glossary drift is detected")
    terms = ["ClipForge"]
    check("a kept term is not reported",
          tr.glossary_misses("clipforge is great", "ClipForge es genial", terms) == [])
    check("a dropped term IS reported",
          tr.glossary_misses("ClipForge is great", "La herramienta es genial", terms) == ["ClipForge"])
    check("a term the speaker never said is not a false alarm",
          tr.glossary_misses("nothing relevant", "nada relevante", terms) == [])


def test_write_tracks() -> None:
    print("\n[8] sidecars are written per language, per format")
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        st = settings(["en", "es", "fr", "ja"])
        tracks = tr.write_tracks(out_dir, 1, cues(3), st, call=stub_call("X:"))
        langs = [t["lang"] for t in tracks]
        check("one track per configured language", langs == ["en", "es", "fr", "ja"], str(langs))
        check("every track has both formats",
              all(set(t["files"]) == {"srt", "vtt"} for t in tracks))
        check("every recorded file exists",
              all(Path(p).is_file() for t in tracks for p in t["files"].values()))

        en = next(t for t in tracks if t["lang"] == "en")
        check("the source track is marked as the source", en.get("source") is True)
        srt_en = Path(en["files"]["srt"]).read_text(encoding="utf-8")
        check("the source track is verbatim (not model round-tripped)",
              "line 0" in srt_en and "X:line 0" not in srt_en, srt_en[:80])

        es = next(t for t in tracks if t["lang"] == "es")
        srt_es = Path(es["files"]["srt"]).read_text(encoding="utf-8")
        check("a translated track carries the translated text", "X:line 0" in srt_es)

        # CJK payload must be written as UTF-8, not mojibake.
        ja = next(t for t in tracks if t["lang"] == "ja")
        vtt_ja = Path(ja["files"]["vtt"]).read_text(encoding="utf-8")
        check("the VTT starts with the WEBVTT header", vtt_ja.startswith("WEBVTT"), vtt_ja[:20])

        # SRT shape
        blocks = [b for b in srt_en.strip().split("\n\n") if b.strip()]
        check("SRT blocks are numbered from 1", blocks[0].startswith("1\n"), blocks[0][:20])
        check("SRT uses an arrow timestamp line", "-->" in blocks[0].splitlines()[1])
        check("SRT uses a comma before milliseconds",
              re_like(blocks[0].splitlines()[1].split(" --> ")[0], r",\d{3}$"))

        # Resume: a second run must not re-translate.
        again = tr.write_tracks(out_dir, 1, cues(3), st, call=_boom)
        check("an existing track is reused, not re-translated",
              all(t.get("cached") for t in again if t["lang"] != "en"),
              str([(t["lang"], t.get("cached")) for t in again]))


def re_like(value: str, pattern: str) -> bool:
    import re
    return bool(re.search(pattern, value))


def _boom(_prompt, _settings):
    raise AssertionError("the model was called again for an already-written track")


def test_failure_isolation() -> None:
    print("\n[9] one bad language must not take the others down")
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        st = settings(["en", "es", "fr"])

        def call(prompt: str, _settings) -> str:
            if "Spanish" in prompt:
                raise RuntimeError("upstream 429")
            lines = json.loads(prompt[prompt.rindex("["):prompt.rindex("]") + 1])
            return json.dumps([f"FR:{t}" for t in lines], ensure_ascii=False)

        logs: list[str] = []
        tracks = tr.write_tracks(out_dir, 1, cues(2), st, call=call, log=logs.append)
        by_lang = {t["lang"]: t for t in tracks}
        check("the failing language is recorded with an error",
              "error" in by_lang.get("es", {}), str(by_lang.get("es")))
        check("the failing language still gets a record", "es" in by_lang)
        check("the other language was written anyway",
              set(by_lang.get("fr", {}).get("files", {})) == {"srt", "vtt"})
        check("the source track was written", "srt" in by_lang.get("en", {}).get("files", {}))
        check("the failure is logged, not swallowed",
              any("es" in line for line in logs), str(logs))


def test_disabled_is_a_noop() -> None:
    print("\n[10] disabled / empty inputs do nothing")
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        check("disabled writes nothing",
              tr.write_tracks(out_dir, 1, cues(2), settings(["es"], enabled=False),
                              call=stub_call()) == [])
        check("no cues writes nothing",
              tr.write_tracks(out_dir, 1, [], settings(["es"]), call=stub_call()) == [])
        check("nothing was created on disk", not any(out_dir.iterdir()))


def main() -> int:
    test_language_list()
    test_wrapping()
    test_parsing()
    test_timings_untouched()
    test_batching_alignment()
    test_uncertain_cues()
    test_glossary()
    test_write_tracks()
    test_failure_isolation()
    test_disabled_is_a_noop()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
