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

    # bold_pop is the DEFAULT: no preset_key must behave identically to
    # explicitly asking for it, or a job submitted without a choice would
    # silently pick a different style.
    default_text = subtitle.build_ass_header(s, 1080, 1920, None)
    explicit_text = subtitle.build_ass_header(s, 1080, 1920, "bold_pop")
    check("no preset_key == bold_pop", style_fields(default_text) == style_fields(explicit_text))

    for key in ("bold_pop", "lower_third", "karaoke", "boxed", "boxed_typo", None):
        p = subtitle.get_preset(key)
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
        want_border = "3" if p.box else "1"
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
    wcues = [{"start": 0.0, "end": 2.0, "text": "one two",
              "words": [{"text": "one", "start": 0.0, "end": 1.0},
                        {"text": "two", "start": 1.0, "end": 2.0}]}]
    k_text = subtitle.build_ass_events(wcues, s, 0.0, "karaoke")
    check("karaoke emits \\k tags", r"\k" in k_text, k_text)

    print("\n[5] braces in spoken text cannot break the override block")
    nasty = [{"start": 0.0, "end": 1.0, "text": "a {tag} and b"}]
    n_text = subtitle.build_ass_events(nasty, s, 0.0, "bold_pop")
    body = n_text.split(",,")[-1]
    check("literal braces are neutralised", "{" not in body.replace(r"{\fad", ""),
          repr(body))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
