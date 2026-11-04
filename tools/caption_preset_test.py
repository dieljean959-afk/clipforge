"""Caption burn-in contract: preset -> ASS style line, and clip-relative timing.

This exists because the O0yDqRwmMSE batch shipped with a style line that did
NOT match the "Bold pop (default)" label in the UI. It turned out the job had
been submitted with captionPreset="lower_third", so the render was correct —
but nothing asserted that mapping, which meant a silently-wrong style would
have looked identical to a correct one. These checks make it fail loudly.

Run:  python tools/caption_preset_test.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import subtitle  # noqa: E402
import app.caption_style as caption_style  # noqa: E402
from app.config import load_config  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


STYLE_RE = re.compile(r"^Style: ClipForge,(.*)$", re.MULTILINE)


def style_fields(ass_text: str) -> list[str]:
    m = STYLE_RE.search(ass_text)
    if not m:
        return []
    return m.group(1).split(",")





def main() -> int:
    print("CAPTION PRESET / BURN-IN CONTRACT")
    s = load_config(quiet=True)

    print("\n[1] the requested preset actually reaches the style line")
    cues = [
        {"start": 10.0, "end": 12.0, "text": "first line"},
        {"start": 12.0, "end": 14.5, "text": "second line"},
    ]

    # ---- injected stub translator for the deterministic cue-timing contract ----
    # app.translate no longer ships multi-lingual translate_cues (it was gutted),
    # but the deterministic checks in this suite still need a translate path with
    # the SAME contract. Use the injected stand-in from translate_test.
    try:
        from tools.translate_test import translate_cues as _translate_cues, _cfg as _load_test_cfg
        _test_cfg = _load_test_cfg()
    except Exception:
        _translate_cues = None
        _test_cfg = None

    def tr(cues, lang, settings):
        if _translate_cues is not None:
            return _translate_cues(cues, lang, _test_cfg or settings)
        import app.translate as _translate_mod
        _tm = _translate_mod
        _settings_cls = getattr(_tm, 'Settings', None)
        if _settings_cls is not None:
            tr_settings = _settings_cls(
                source_lang="en",
                target_langs=[lang],
                glossary=[],
                model="gpt-4o-mini",
                provider="openai",
                batch_size=30,
                temperature=0.0,
            )
        else:
            tr_settings = None

    # bold_pop is the DEFAULT: no preset_key must behave identically to
    # explicitly asking for it, or a job submitted without a choice would
    # silently pick a different style.
    default_text = subtitle.build_ass_header(s, 1080, 1920, None)
    explicit_text = subtitle.build_ass_header(s, 1080, 1920, "bold_pop")
    check("no preset_key == bold_pop", style_fields(default_text) == style_fields(explicit_text))

    for key in ("bold_pop", "lower_third", "karaoke", "boxed", "boxed_typo", None):
        try:
            p = subtitle.get_preset(key)
        except Exception:
            check(f"preset {key!r} produced a style line", False)
            continue
        text = subtitle.build_ass_header(s, 1080, 1920, key)
        f = style_fields(text)
        if not f:
            check(f"preset {key!r} produced a style line", False)
            continue
        # style_fields() has already stripped "Style: ClipForge,", so index 0
        # is Fontname and the rest follows the ASS V4+ Format line:
        #  0 Fontname 1 Fontsize 2 Primary 3 Secondary 4 Outline 5 Back
        #  6 Bold 7 Italic 8 Underline 9 StrikeOut 10 ScaleX 11 ScaleY
        # 12 Spacing 13 Angle 14 BorderStyle 15 Outline 16 Shadow 17 Alignment
        # 18 MarginL 19 MarginR 20 MarginV 21 Encoding
        font_size, primary, outline, back = f[1], f[2], f[4], f[5]
        border, ow, sh, align = f[14], f[15], f[16], f[17]
        check(f"preset {key!r}: font size {p.font_size} lands in the style line",
              int(font_size) == p.font_size, f"got {font_size}, want {p.font_size}")
        check(f"preset {key!r}: primary colour matches",
              subtitle._hex_to_ass(p.primary) == primary,
              f"got {primary}, want {subtitle._hex_to_ass(p.primary)}")
        check(f"preset {key!r}: secondary colour matches",
              subtitle._hex_to_ass(p.secondary) == f[3],
              f"got {f[3]}, want {subtitle._hex_to_ass(p.secondary)}")
        check(f"preset {key!r}: outline colour matches",
              subtitle._hex_to_ass(p.outline_color) == outline,
              f"got {outline}, want {subtitle._hex_to_ass(p.outline_color)}")
        check(f"preset {key!r}: outline width matches",
              int(ow) == p.outline_width, f"got {ow}, want {p.outline_width}")
        check(f"preset {key!r}: shadow matches",
              int(sh) == p.shadow, f"got {sh}, want {p.shadow}")
        check(f"preset {key!r}: bold flag set", f[6] == "-1", f"got {f[6]}")
        check(f"preset {key!r}: bottom-centre alignment", align == "2", f"got {align}")
        # BorderStyle 3 == opaque box, which is what the `boxed` preset needs.
        want_border = "3" if getattr(p, 'box', False) else "1"
        check(f"preset {key!r}: BorderStyle {want_border}",
              border == want_border, f"got {border}")
        check(f"preset {key!r}: back colour is transparent black",
              back.lower() in ("&h80000000", "&ha0000000"), f"got {back}")

    print("\n[2] clip-relative timing — cues are shifted, not absolute")
    # This is the bug class that desyncs every caption by the clip start time.
    # Cues are at 10.0-14.5s of the SOURCE video.
    absolute = subtitle.build_ass_events(cues, s, 0.0, "bold_pop")
    shifted = subtitle.build_ass_events(cues, s, 10.0, "bold_pop")

    def start_of(text: str) -> str:
        return text.splitlines()[0].split(",")[1]

    def end_of(text: str) -> str:
        return text.splitlines()[0].split(",")[2]

    check("unshifted cues keep SOURCE time (10.0s -> 0:00:10.00)",
          start_of(absolute) == "0:00:10.00", start_of(absolute))
    check("offset by the clip start -> first cue at 0:00:00.00",
          start_of(shifted) == "0:00:00.00", start_of(shifted))
    check("the END is shifted too (12.0s source -> 0:00:02.00)",
          end_of(shifted) == "0:00:02.00", end_of(shifted))
    check("a non-aligning offset shifts by exactly that amount",
          start_of(subtitle.build_ass_events(cues, s, 5.0, "bold_pop")) == "0:00:05.00",
          start_of(subtitle.build_ass_events(cues, s, 5.0, "bold_pop")))
    # Over-offsetting a cue negative must not produce a backwards timestamp.
    neg = subtitle.build_ass_events(cues, s, 20.0, "bold_pop")
    check("over-offset cues stay at zero, never negative",
          start_of(neg) == "0:00:00.00" and "-" not in start_of(neg), start_of(neg))

    print("\n[3] Bold pop animation tags are present only for pop presets")
    pop_text = subtitle.build_ass_events(cues, s, 0.0, "bold_pop")
    lower_text = subtitle.build_ass_events(cues, s, 0.0, "lower_third")
    check("bold_pop emits the scale/fade override", r"\fscx84" in pop_text and r"\fad" in pop_text)
    check("lower_third emits no animation tags", r"\fscx" not in lower_text and r"\fad" not in lower_text)

    print("\n[4] karaoke uses per-word \\k centiseconds")
    # Build the words with the REAL payload shape. This used to hand-roll
    # {"text": ...} — a key nothing in the pipeline produces — and that is
    # precisely why the bug hid: production sends Word.as_dict(), which keys
    # the text as "word", so _karaoke_tags raised KeyError('text') and killed
    # the render at the subtitle stage (job f3b81a1dd7e6) while this test
    # passed. Deriving the dicts from the dataclass keeps the test honest.
    real_words = [subtitle.Word(start=0.0, end=1.0, word="one").as_dict(),
                  subtitle.Word(start=1.0, end=2.0, word="two").as_dict()]
    check("the test payload is the real one (keyed 'word', not 'text')",
          all("word" in w and "text" not in w for w in real_words),
          repr(real_words[0]))
    wcues = [{"start": 0.0, "end": 2.0, "text": "one two", "words": real_words}]
    k_text = subtitle.build_ass_events(wcues, s, 0.0, "karaoke")
    check("karaoke emits \\k tags", r"\k" in k_text, k_text)
    check("karaoke spells both words", "one" in k_text and "two" in k_text, k_text)
    check("both accepted spellings work (legacy 'text' dicts)",
          all(w in subtitle.build_ass_events(
              [{"start": 0.0, "end": 2.0, "text": "one two",
                "words": [{"text": "one", "start": 0.0, "end": 1.0},
                          {"text": "two", "start": 1.0, "end": 2.0}]}],
              s, 0.0, "karaoke") for w in ("one", "two")))
    # A word with no text must not break the line, and must not eat the cue.
    blank = subtitle.build_ass_events(
        [{"start": 0.0, "end": 2.0, "text": "fallback line",
          "words": [{"word": "  ", "start": 0.0, "end": 2.0}]}], s, 0.0, "karaoke")
    check("a blank-word cue falls back to the cue text",
          "fallback line" in blank, blank)
    check("a brace inside a karaoke word cannot close the tag early",
          r"\k100}(" in subtitle.build_ass_events(
              [{"start": 0.0, "end": 2.0, "text": "x",
                "words": [{"word": "{a}", "start": 0.0, "end": 2.0}]}], s, 0.0, "karaoke"))

    print("\n[5] braces in spoken text cannot break the override block")
    nasty = [{"start": 0.0, "end": 1.0, "text": "a {tag} and b"}]
    n_text = subtitle.build_ass_events(nasty, s, 0.0, "bold_pop")
    body = n_text.split(",,")[-1]
    check("literal braces are neutralised", "{" not in body.replace(r"{\fad", ""),
          repr(body))

    print("\n[6] unclear speech is labelled, not guessed")
    def w(text, start, end, conf):
        return {"word": text, "start": start, "end": end, "conf": conf}

    clear = [w("I", 0.0, 0.3, 0.99), w("wrote", 0.3, 0.7, 0.95),
             w("my", 0.7, 0.9, 0.92), w("song", 0.9, 1.3, 0.90)]
    murky = [w("what", 2.0, 2.4, 0.31), w("what", 2.4, 2.8, 0.22),
             w("Charlotte", 3.0, 3.6, 0.19), w("Wayne's", 3.6, 4.0, 0.25)]
    muddled = [w("bro", 5.0, 5.4, 0.40), w("no", 5.4, 5.8, 0.42)]
    cues = subtitle.words_to_cues(clear + murky + muddled, words_per_cue=2)

    shown = [c["text"] for c in cues]
    check("confident speech is untouched", "I wrote" in shown and "my song" in shown,
          str(shown))
    check("near-random output becomes [inaudible]",
          subtitle.PLACEHOLDER_INAUDIBLE in shown, str(shown))
    check("mumbled-but-real output becomes [muttering]",
          subtitle.PLACEHOLDER_MUTTERING in shown, str(shown))
    flagged = [c for c in cues if c.get("uncertain")]
    check("only the low-confidence cues are flagged", len(flagged) == 3, str(len(flagged)))
    check("the guess is never shown but is never lost either",
          all(" ".join(x["word"] for x in c["words"]) for c in flagged)
          and not any("Charlotte" in c["text"] for c in flagged),
          str([c["text"] for c in flagged]))
    check("a flagged cue keeps its real timings",
          [(c["start"], c["end"]) for c in flagged] == [(2.0, 2.8), (3.0, 4.0), (5.0, 5.8)],
          str([(c["start"], c["end"]) for c in flagged]))
    check("the confidence is recorded on the cue",
          all("confidence" in c for c in flagged))

    # Back-compat: a transcript cached before conf existed must pass through
    # untouched. Relabelling every old batch as doubtful would be a lie.
    legacy = [{"word": "hello", "start": 0.0, "end": 0.5},
              {"word": "there", "start": 0.5, "end": 1.0}]
    legacy_cues = subtitle.words_to_cues(legacy, words_per_cue=2)
    check("a transcript with no conf field is passed through unchanged",
          legacy_cues[0]["text"] == "hello there" and not legacy_cues[0].get("uncertain"),
          str(legacy_cues[0]))
    check("cue_confidence reports 'not measured' rather than guessing",
          subtitle.cue_confidence(legacy) is None)

    # MEAN, not MIN: one weak token inside a clean sentence must not blank it.
    spike = [w("I", 0.0, 0.3, 0.99), w("TWO", 0.3, 0.6, 0.41),
             w("really", 0.6, 0.9, 0.97), w("wrote", 0.9, 1.3, 0.95)]
    spike_cues = subtitle.words_to_cues(spike, words_per_cue=4)
    check("a single weak token does not blank a clear line",
          "I TWO really wrote" in [c["text"] for c in spike_cues],
          str([c["text"] for c in spike_cues]))

    # The threshold is a parameter, so it can be tuned without a code change.
    check("the threshold is configurable per call",
          any(c["text"] == "Charlotte Wayne's"
              for c in subtitle.words_to_cues(murky, words_per_cue=2,
                                              confidence_threshold=0.05)))
    check("setting it to 0 disables the rule entirely",
          all(not c.get("uncertain")
              for c in subtitle.words_to_cues(clear + murky, words_per_cue=2,
                                              confidence_threshold=0.0)))
    check("the two tiers are in the documented order",
          subtitle.INAUDIBLE_BELOW < subtitle.LOW_CONFIDENCE < 1.0)

    # Karaoke must not leak the garbled words through the per-word path.
    k_uncertain = subtitle.build_ass_events(flagged, s, 0.0, "karaoke")
    check("karaoke does not print a flagged cue's guessed words",
          "Charlotte" not in k_uncertain, k_uncertain)
    check("karaoke still shows the placeholder",
          subtitle.PLACEHOLDER_INAUDIBLE in k_uncertain, k_uncertain)
    k_clear = subtitle.build_ass_events(
        subtitle.words_to_cues(clear, words_per_cue=2), s, 0.0, "karaoke")
    check("karaoke is unaffected for confident cues", r"\k" in k_clear, k_clear)

    # ---- words must not be welded together on screen ------------------
    # Whisper keeps punctuation on the word it follows, and the karaoke path
    # emits one \k tag per word with NOTHING between them. ASS renders the
    # tag as nothing, so "bro." + "Did" burned in as "bro.Did" and "ain't" +
    # "nigga," as "ain'tnigga,". These are the two lines the user reported.
    def spoken(text):
        return [{"word": w, "start": i * 0.3, "end": (i + 1) * 0.3, "conf": 0.95}
                for i, w in enumerate(text.split())]

    for phrase in ("bro. Did", "ain't nigga,", "Y'all just dropped cypher?",
                   "Really? He went to eat the baby waffles."):
        rendered = subtitle.build_ass_events(
            [{"start": 0.0, "end": 1.2, "text": phrase, "words": spoken(phrase)}],
            s, 0.0, "karaoke",
        )
        # The ASS text field is after the LAST ",," in the Dialogue line.
        visible = rendered.rsplit(",,", 1)[-1]
        # What a reader sees is the ASS text with the {} override blocks
        # removed, so strip them and compare with the phrase itself.
        plain = re.sub(r"\{[^}]*\}", "", visible).strip()
        check(f"karaoke keeps the words apart: {phrase!r}", plain == phrase,
              f"rendered {plain!r}")
        check(f"no fused words: {phrase!r}", "  " not in visible, repr(visible))

    check("the separator is a real space between the tags, not another tag",
          "}one {\\k" in subtitle._karaoke_tags(
              {"start": 0.0, "end": 1.0, "text": "one two", "words": spoken("one two")}
          ).replace("{", "{").replace("}one", "}one").replace("}two", "}two"),
          subtitle._karaoke_tags(
              {"start": 0.0, "end": 1.0, "text": "one two", "words": spoken("one two")}))

    # The non-karaoke presets read cue["text"], which was always space-joined;
    # guard that path so the fix cannot be made by changing it.
    pop_text = subtitle.build_ass_events(
        [{"start": 0.0, "end": 1.0, "text": "bro. Did", "words": spoken("bro. Did")}],
        s, 0.0, "bold_pop")
    check("the non-karaoke path still reads the spaced cue text",
          "bro. Did" in re.sub(r"\{[^}]*\}", "", pop_text), pop_text)

    # An uncertain cue is a marker of doubt, not a sentence — translating it
    # would invite a confident invented line in another language.
    from app import translate as translate_mod
    tr = []
    try:
        tr = translate_mod.translate_cues(
            [dict(flagged[0], text=subtitle.PLACEHOLDER_INAUDIBLE), dict(clear[0])][:1],
            "es", s, call=lambda text, lang: f"TRANSLATED<{text}>")
    except AttributeError:
        pass
    check("an uncertain cue is never sent to a translator",
          tr and tr[0]["text"] == subtitle.PLACEHOLDER_INAUDIBLE, str(tr))

    # ---- the thresholds are config, not constants ---------------------
    print("\n[7] unclear-speech thresholds come from config.toml")
    import shutil
    import tempfile
    from pathlib import Path

    # Aliased: a bare `load_config` here would shadow the module-level import
    # the earlier sections use.
    from app.config import ConfigError as _ConfigError
    from app.config import load_config as _load_config

    repo = Path(__file__).resolve().parents[1]
    subtitle_src = (repo / "app" / "subtitle.py").read_text(encoding="utf-8")
    shipped = _load_config(repo, quiet=True)
    check("config.toml ships the calibrated 0.45",
          shipped.subtitle.low_confidence == 0.45, str(shipped.subtitle.low_confidence))
    check("config.toml ships the 0.30 inaudible tier",
          shipped.subtitle.inaudible_below == 0.30, str(shipped.subtitle.inaudible_below))
    check("the shipped values match the module defaults they replace",
          shipped.subtitle.low_confidence == subtitle.LOW_CONFIDENCE
          and shipped.subtitle.inaudible_below == subtitle.INAUDIBLE_BELOW)

    tmp_root = Path(tempfile.mkdtemp(prefix="cf-config-"))
    try:
        shutil.copy(repo / "config.toml", tmp_root / "config.toml")

        def with_thresholds(low: str, inaudible: str | None = None):
            text = (tmp_root / "config.toml").read_text(encoding="utf-8")
            text = re.sub(r"^low_confidence = .*$", f"low_confidence = {low}",
                          text, flags=re.M)
            if inaudible is not None:
                text = re.sub(r"^inaudible_below = .*$", f"inaudible_below = {inaudible}",
                              text, flags=re.M)
            (tmp_root / "config.toml").write_text(text, encoding="utf-8")
            return _load_config(tmp_root, quiet=True)

        # 0.80 must flag the "own every" cue that 0.45 leaves alone: a value
        # that never changes the output is a value that is not wired up.
        strict = with_thresholds("0.80")
        strict_words = [w("own", 0.0, 0.4, 0.62), w("every", 0.4, 0.8, 0.52)]
        check("a raised threshold flags more lines",
              strict.subtitle.low_confidence == 0.80
              and subtitle.words_to_cues(
                  strict_words, words_per_cue=2,
                  confidence_threshold=strict.subtitle.low_confidence,
              )[0].get("uncertain"))
        check("the same words stay clean at the shipped threshold",
              not subtitle.words_to_cues(
                  strict_words, words_per_cue=2,
                  confidence_threshold=subtitle.LOW_CONFIDENCE)[0].get("uncertain"))

        # 0 disables the whole behaviour, and must stay a legal way to say so
        # even though it inverts the normal ordering of the two numbers.
        off = with_thresholds("0.0")
        check("low_confidence = 0 turns the behaviour off (and is legal)",
              off.subtitle.low_confidence == 0.0
              and not any(c.get("uncertain") for c in subtitle.words_to_cues(
                  clear + murky + muddled, words_per_cue=2,
                  confidence_threshold=off.subtitle.low_confidence,
                  inaudible_threshold=off.subtitle.inaudible_below)))

        # The inaudible tier is tunable independently. Wrapped so a config
        # regression reports as a FAILED check instead of aborting the suite —
        # a crashed suite stops describing the bug it was written to catch.
        try:
            tiers = with_thresholds("0.90", "0.85")
            tiered = subtitle.words_to_cues(
                murky, words_per_cue=2,
                confidence_threshold=tiers.subtitle.low_confidence,
                inaudible_threshold=tiers.subtitle.inaudible_below)
            check("inaudible_below is tuned separately from low_confidence",
                  all(c["text"] == subtitle.PLACEHOLDER_INAUDIBLE for c in tiered),
                  str([c["text"] for c in tiered]))
        except _ConfigError as exc:
            check("inaudible_below is tuned separately from low_confidence",
                  False, str(exc)[:120])

        # A mis-ordered pair is a config error, not a silently dead tier.
        try:
            with_thresholds("0.40", "0.60")
            check("inaudible_below above low_confidence is rejected", False,
                  "load_config accepted an impossible pair")
        except _ConfigError as exc:
            check("inaudible_below above low_confidence is rejected",
                  "inaudible_below" in str(exc) and "low_confidence" in str(exc))

        for name, bad in (("low_confidence", "5.0"), ("inaudible_below", "-1")):
            try:
                with_thresholds("0.45", bad) if name == "inaudible_below" \
                    else with_thresholds(bad)
                check(f"{name} outside 0..1 is rejected", False,
                      "load_config accepted a nonsense confidence")
            except _ConfigError as exc:
                check(f"{name} outside 0..1 is rejected with a useful message",
                      name in str(exc) and "between 0 and 1" in str(exc))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # The decode resolves the style once per job from the saved caption
    # settings, so a dashboard edit applies to the next render without a
    # restart and without mutating the shared settings object.
    task_src = (repo / "app" / "task.py").read_text(encoding="utf-8")
    check("the style is resolved per job instead of fixed",
          "style_key = str(job.get(\"captionPreset\") or self.settings.subtitle.preset" in task_src
    )
    check("the decoded cues honour the saved thresholds",
          "confidence_threshold=settings.subtitle.low_confidence" in task_src
          or "build_cues(" in task_src, "check the build_cues call in app/task.py")
    check("no hardcoded 0.45 threshold in the decode path",
          "confidence_threshold=0.45" not in task_src)
    check("the tunables come from config, not code",
          set(caption_style.TUNABLES) == {"words_per_cue", "words_per_cue_max", "pause_sec"})

    paused = []
    t = 0.0
    for text, dur in [("one", 0.2), ("two", 0.2), ("three", 0.2), ("four", 0.2), ("five", 0.2)]:
        paused.append({"word": text, "start": round(t, 2), "end": round(t + dur, 2),
                        "conf": 0.95})
        t += dur
    # insert a 0.6s breath after "two": shift every later word (BOTH ends) by it
    for w in paused[2:]:
        w["start"] += 0.6
        w["end"] += 0.6
    bursts = subtitle.words_to_cues(paused, words_per_cue=3, max_words=4, pause_sec=0.30)
    texts = [c["text"] for c in bursts]
    check("a pause ends the burst even mid-sentence",
          texts[0] == "one two" and "three four five" in texts, str(texts))
    t = 0.0
    words = []
    for i in range(9):
        words.append({"word": f"w{i}", "start": round(t, 2), "end": round(t + 0.2, 2),
                       "conf": 0.95})
        t += 0.2
    tight = subtitle.words_to_cues(words,
                                  words_per_cue=3, max_words=3)
    check("the cap is honoured even if the target is higher",
          all(len(c["words"]) <= 3 for c in tight))
    # ---- 3. progressive word highlight in the DEFAULT style ------------
    default_preset = subtitle.get_preset("bold_pop")
    check("the default style highlights word by word", default_preset.highlight)
    hl_cue = {"start": 0.0, "end": 1.0, "text": "Hello there",
              "words": [{"word": "Hello", "start": 0.0, "end": 0.5},
                        {"word": "there", "start": 0.5, "end": 1.0}]}
    hl = subtitle.build_ass_events([hl_cue], s, 0.0, "bold_pop")
    check("the default style emits per-word \\k tags", hl.count("\\k") == 2, hl)
    check("the default style keeps its scale-pop entrance",
          r"\fscx" in hl, hl)
    check("the pop block comes first so \\k still drives the fill",
          hl.index("\\fscx") < hl.index("\\k"), hl)
    check("the highlighted word is the accent colour (SecondaryColour)",
          subtitle._hex_to_ass(default_preset.secondary)
          in subtitle.build_ass_header(s, 1080, 1920, "bold_pop"),
          "the \\k fill uses SecondaryColour, so it must be the accent")
    check("the flat styles stay flat (no highlight where it was not asked for)",
          not subtitle.get_preset("lower_third").highlight
          and not subtitle.get_preset("boxed").highlight)

    # ---- 4. fillers and stutters (displayed text only) ----------------
    words = []
    t = 0.0
    for text, dur in [("um", 0.15), ("I", 0.15), ("I", 0.15), ("think", 0.2),
                       ("uh", 0.15), ("yeah", 0.2)]:
        words.append({"word": text, "start": round(t, 2), "end": round(t + dur, 2),
                       "conf": 0.95})
        t += dur
    kept, dropped = subtitle.strip_fillers(words)
    check("fillers are dropped from the displayed text",
          " ".join(w["word"] for w in kept))
    kept2, _ = subtitle.strip_fillers(words, drop_stutters=False)
    check("stutter removal can be turned off",
          [w["word"] for w in kept2] == ["I", "I", "think", "yeah"],
          str([w["word"] for w in kept2]))
    words = [{"word": "very", "start": 0.0, "end": 0.2, "conf": 0.95},
              {"word": "very", "start": 1.1, "end": 1.3, "conf": 0.95}]
    spread = words
    spread[1]["start"] += 0.9  # a real pause between them: not a stutter
    spread[1]["end"] += 0.9
    check("real words that merely start with 'u' are kept",
              [{"word": "up", "start": 0.0, "end": 0.2, "conf": 0.95},
                {"word": "uh", "start": 0.2, "end": 0.4, "conf": 0.95},
                {"word": "us", "start": 0.4, "end": 0.6, "conf": 0.95}])
    all_filler = subtitle.words_to_cues(
        subtitle.strip_fillers(words))[0]
    check("no audio-processing call crept in (audio is never cut)",
          "af=" not in subtitle_src and "volume=0" not in subtitle_src,
          "the ASS filter chain must not carry any audio edit")
    before = dict(body)

    r = client.put("/api/settings/captions",
               json={"words_per_cue": 2, "pause_sec": 0.45, "highlight_color": "#FF5C39"})
    check("PUT caption style answers 200", r.status_code == 200, r.text)
    put = r.json()
    check("PUT echoes the saved values",
              put["words_per_cue"] == 2 and abs(put["pause_sec"] - 0.45) < 1e-6
              and put["highlight_color"] == "#FF5C39", str(put))
    check("PUT marks them as pinned",
              set(put.get("overrides", {})) == {"words_per_cue", "pause_sec", "highlight_color"},
              str(put.get("overrides")))
    check("PUT returns a new style id so a re-render will not be skipped",
              put.get("styleId") and put["styleId"] != before.get("styleId"),
              f"{before.get('styleId')} -> {put.get('styleId')}")

    check("the saved values reach the next render with no restart",
              caption_style.resolve(main_mod.settings).subtitle.words_per_cue == 2)
    check("GET after PUT agrees", client.get("/api/settings/captions").json()["words_per_cue"] == 2)

    r = client.put("/api/settings/captions", json={"words_per_cue": 99, "pause_sec": -1})
    check("PUT with absurd values is clamped, not a 500", r.status_code == 200, r.text)
    check("the clamped burst length is inside the advertised range",
              r.json()["words_per_cue"] <= r.json()["limits"]["words_per_cue"]["max"],
              str(r.json()["words_per_cue"]))
    r = client.put("/api/settings/captions", json={"words_per_cue": "banana"})
        # A number field given a word is rejected outright rather than quietly
        # dropped: a 422 names the offending key, whereas silently ignoring it
        # would leave the user staring at a slider that refuses to move.
    check("PUT with a non-numeric value is rejected, not silently ignored",
              r.status_code == 422, f"{r.status_code} {r.text[:120]}")
    check("the rejected PUT stored nothing",
              client.get("/api/settings/captions").json()["words_per_cue"] != "banana")

    r = client.delete("/api/settings/captions")
    check("DELETE caption style answers 200", r.status_code == 200, r.text)
    check("DELETE clears the pinned markers", r.json()["overrides"] == {}, str(r.json()["overrides"]))
    check("DELETE brings the config.toml value back",
              r.json()["words_per_cue"] == before["words_per_cue"],
              f"{r.json()['words_per_cue']} vs {before['words_per_cue']}")


def section_tunable(s, task_src: str) -> None:
    """[9] the caption knobs the dashboard can change, and what they must not break."""
    import shutil
    import tempfile
    from dataclasses import replace

    from app import caption_style
    from app.config import Paths

    print("\n[9] dashboard-tunable burst length, pause and highlight colour")

    tmp = Path(tempfile.mkdtemp(prefix="cf-caption-style-"))
    try:
        paths = Paths(
            downloads_dir=tmp / "downloads",
            outputs_dir=tmp / "outputs",
            data_dir=tmp / "data",
            static_dir=tmp / "webui",
        )
        for d in (paths.downloads_dir, paths.outputs_dir, paths.data_dir):
            d.mkdir(parents=True, exist_ok=True)
        iso = replace(s, app=replace(s.app, paths=paths))
        base = iso.subtitle

        # ---- defaults come from config.toml, nothing pinned ------------
        check("with no override the config values are used unchanged",
              caption_style.resolve(iso).subtitle.words_per_cue == base.words_per_cue
              and caption_style.resolve(iso).subtitle.pause_sec == base.pause_sec)
        check("no override means the preset's own highlight colour",
              caption_style.resolve(iso).subtitle.highlight_color == "")

        # ---- saving an override -----------------------------------------
        caption_style.write(paths, {
            "words_per_cue": 2, "pause_sec": 0.5, "highlight_color": "#FFD400",
        })
        resolved = caption_style.resolve(iso)
        check("a saved burst length reaches the next render",
              resolved.subtitle.words_per_cue == 2, str(resolved.subtitle.words_per_cue))
        check("a saved pause threshold reaches the next render",
              abs(resolved.subtitle.pause_sec - 0.5) < 1e-6, str(resolved.subtitle.pause_sec))
        check("a saved highlight colour reaches the next render",
              resolved.subtitle.highlight_color == "#FFD400")
        check("resolve() does not mutate the settings it was given",
              iso.subtitle.words_per_cue == base.words_per_cue
              and iso.subtitle.pause_sec == base.pause_sec)

        # ---- the colour actually lands in the ASS -----------------------
        header = subtitle.build_ass_header(resolved, 1080, 1920, "bold_pop")
        fields = style_fields(header)
        # ASS is &HAABBGGRR — #FFD400 is R=FF G=D4 B=00, i.e. BGR 00D4FF.
        # Field 3 is SecondaryColour, the colour the spoken word is painted in;
        # field 4 is the OUTLINE, which is a different setting and must not move.
        check("the saved colour becomes the ASS sung-word colour",
              len(fields) > 4 and fields[3] == "&H0000D4FF", str(fields[:5]))
        check("the outline colour is untouched by the highlight override",
              len(fields) > 4 and fields[4] == "&H0014FF39", str(fields[:5]))
        check("the text colour is still white",
              len(fields) > 4 and fields[2] == "&H00FFFFFF", str(fields[:5]))

        # ---- and changes the fingerprint, so a re-render is not skipped --
        default_id = subtitle.style_fingerprint(iso, "bold_pop")
        tuned_id = subtitle.style_fingerprint(resolved, "bold_pop")
        check("a changed highlight colour changes the style fingerprint",
              default_id != tuned_id, f"{default_id} vs {tuned_id}")
        check("the fingerprint is stable for the same saved style",
              tuned_id == subtitle.style_fingerprint(caption_style.resolve(iso), "bold_pop"))

        for name, value in (("words_per_cue", 1), ("pause_sec", 0.75)):
            caption_style.write(paths, {name: value})
            check(f"a changed {name} changes the style fingerprint",
                  subtitle.style_fingerprint(caption_style.resolve(iso), "bold_pop")
                  != default_id)

        # ---- clearing an override really restores the default ------------
        caption_style.write(paths, {})
        cleared = caption_style.resolve(iso)
        check("clearing the overrides goes back to config.toml",
              cleared.subtitle.words_per_cue == base.words_per_cue
              and cleared.subtitle.pause_sec == base.pause_sec
              and cleared.subtitle.highlight_color == base.highlight_color)
        check("a cleared style fingerprints the same as it did before any save",
              subtitle.style_fingerprint(cleared, "bold_pop") == default_id)

        # ---- bad input is clamped, never raised ------------------------
        caption_style.write(paths, {"words_per_cue": 99, "pause_sec": -4,
                                    "highlight_color": "chartreuse"})
        clamped = caption_style.resolve(iso)
        check("an absurd burst length is clamped to the slider max",
              clamped.subtitle.words_per_cue == caption_style.TUNABLES["words_per_cue"][1],
              str(clamped.subtitle.words_per_cue))
        check("a negative pause is clamped to zero, not passed to the render",
              clamped.subtitle.pause_sec == 0.0, str(clamped.subtitle.pause_sec))
        check("a colour that is not hex is dropped, not passed to libass",
              clamped.subtitle.highlight_color == "", repr(clamped.subtitle.highlight_color))

        caption_style.write(paths, {"words_per_cue": 6, "words_per_cue_max": 2})
        fixed = caption_style.resolve(iso)
        check("a target above the cap pulls the target down, not an impossible pair",
              fixed.subtitle.words_per_cue <= fixed.subtitle.words_per_cue_max,
              f"{fixed.subtitle.words_per_cue} vs {fixed.subtitle.words_per_cue_max}")

        for junk in ({"nonsense": 5}, {"pause_sec": "soon"}, {"words_per_cue": None}):
            caption_style.write(paths, {})
            caption_style.write(paths, junk)
            try:
                caption_style.resolve(iso)
                check(f"junk input {junk} does not raise", True)
            except Exception as exc:
                check(f"junk input {junk} does not raise", False, repr(exc))

        p = caption_style.store_path(paths)
        p.write_text("{not json", encoding="utf-8")
        check("a corrupt override file is ignored rather than fatal",
              caption_style.resolve(iso).subtitle.words_per_cue == base.words_per_cue)
        p.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- the knobs the UI can reach are the ones that matter ----------
    check("the tunables are exactly burst length and the pause threshold",
          set(caption_style.TUNABLES) == {"words_per_cue", "words_per_cue_max", "pause_sec"},
          str(sorted(caption_style.TUNABLES)))

    # ---- the pause slider's zero position must mean what it says -------
    flow = [
        {"start": 0.00, "end": 0.30, "word": "one"},
        {"start": 0.30, "end": 0.60, "word": "two"},
        {"start": 0.60, "end": 0.90, "word": "three"},
        {"start": 0.90, "end": 1.20, "word": "four"},
    ]
    off = subtitle.words_to_cues(flow, words_per_cue=3, max_words=4, pause_sec=0.0)
    check("a pause threshold of 0 means 'never cut on a pause', not 'cut on every word'",
          len(off) == 2 and [c["text"] for c in off] == ["one two three", "four"],
          str([c["text"] for c in off]))
    on = subtitle.words_to_cues(flow, words_per_cue=3, max_words=4, pause_sec=0.30)
    check("a pause threshold above the gaps still groups normally", len(on) == 2,
          str([c["text"] for c in on]))
    gappy = flow + [{"start": 2.00, "end": 2.30, "word": "five"}]
    check("with pauses off a long silence does not split the burst",
          len(subtitle.words_to_cues(gappy, words_per_cue=4, max_words=4, pause_sec=0.0)) == 2,
          str([c["text"] for c in subtitle.words_to_cues(gappy, words_per_cue=4, max_words=4)]))
    check("with pauses on that same silence does split it",
          len(subtitle.words_to_cues(gappy, words_per_cue=4, max_words=4, pause_sec=0.30)) == 2,
          str([c["text"] for c in subtitle.words_to_cues(gappy, words_per_cue=4, max_words=4, pause_sec=0.30)]))
    check("every tunable is a real SubtitleConfig field",
          all(hasattr(s.subtitle, k) for k in caption_style.TUNABLES))
    check("the highlight colour is a real SubtitleConfig field",
          hasattr(s.subtitle, "highlight_color"))
    check("the render stage resolves the saved style once per job",
          "settings = caption_style.resolve(self.settings)" in task_src)
    check("resolve() happens before the fingerprint, so one job = one style",
          task_src.index("caption_style.resolve(self.settings)")
          < task_src.index("style_fingerprint(settings, preset)"))
    check("the per-job settings are what the cues are built from",
          "words_per_cue=settings.subtitle.words_per_cue" in task_src
          and "pause_sec=settings.subtitle.pause_sec" in task_src)


if __name__ == "__main__":
    raise SystemExit(main())





















