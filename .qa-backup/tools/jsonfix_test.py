"""Unit tests for the JSON repair + candidate validation path.

Fixtures are REAL model outputs, captured from the live Groq API during the
`json_validate_failed` investigation, plus the shapes that broke the previous
regex-based parser. Every one of these either killed the whole job or silently
returned zero clips before.

    python tools/detect_json_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import jsonfix  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# fixtures: real malformed replies
# --------------------------------------------------------------------------- #

# A) fenced block with prose either side. Returned zero clips before, because
#    the old parser's \[.*\] regex matched the fence-adjacent array but the
#    leading sentence made the first pass ambiguous.
FENCED = """Here are the best moments from this section:

```json
[
  {"start_word": 120, "end_word": 310, "score": 92, "title": "The one thing nobody tells you"},
  {"start_word": 480, "end_word": 640, "score": 78, "title": "Why it actually works"}
]
```

Let me know if you want more!"""

# B) truncated mid-array because max_tokens ran out. This is the common case the
#    retry ladder exists for, and a reason to raise max_tokens on retry.
TRUNCATED = """[{"start_word": 12, "end_word": 140, "score": 88, "title": "Cold open"},
  {"start_word": 300, "end_word": 455, "score": 74, "title": "The core argument"},
  {"start_word": 900, "end_word": 1020, "score": 81, "title": "Payoff"""

# C) trailing comma before the closing bracket.
TRAILING_COMMA = """[{"start_word": 5, "end_word": 90, "score": 70, "title": "One"},
  {"start_word": 200, "end_word": 310, "score": 65, "title": "Two"},]"""

# D) json_object mode coerced the requested ARRAY into a single clip OBJECT.
#    The old parser found no '[' and raised "no JSON array in model reply",
#    failing the entire job over one recoverable clip.
SINGLE_OBJECT = '{"start_word": 41, "end_word": 190, "score": 83, "title": "Lost the thread"}'

# E) the object envelope strict json_schema produces.
ENVELOPE = '{"clips": [{"start_word": 3, "end_word": 90, "score": 61, "title": "Setup"}]}'

# F) one broken element among good ones — a stray unescaped quote in a title.
MIXED = """[{"start_word": 1, "end_word": 50, "score": 70, "title": "Good clip"},
  {"start_word": 60, "end_word": 150, "score": 90, "title": "The "best" moment ever"},
  {"start_word": 200, "end_word": 280, "score": 55, "title": "Also fine"}]"""

# G) prose, no JSON at all.
PROSE = "I could not identify any particularly strong moments in this section."

# H) Python-flavoured literals leaking through.
PY_LITERALS = '[{"start_word": 7, "end_word": 95, "score": None, "title": "Fine", "x": True}]'

# I) reversed range — end before start.
REVERSED = '[{"start_word": 400, "end_word": 120, "score": 70, "title": "Backwards"}]'

# J) alternative key spellings for the index range.
ALT_KEYS = '[{"start": 10, "end": 120, "score": 70, "title": "Alt keys"}]'

# K) a bracket inside a string — the exact thing a \[.*\] regex truncates on.
BRACKET_IN_STRING = '[{"start_word": 1, "end_word": 50, "score": 70, "title": "The [good] part"}]'


# --------------------------------------------------------------------------- #

def main() -> int:
    # --- the fixtures that used to lose a whole job -------------------------
    r = jsonfix.repair_candidates(FENCED)
    check("fenced block salvages 2 clips", len(r) == 2, f"got {len(r)}")
    check("fenced block keeps the first title",
          bool(r) and r[0]["title"] == "The one thing nobody tells you",
          str(r[:1]))

    r = jsonfix.repair_candidates(TRUNCATED)
    check("truncated array keeps the 2 complete clips", len(r) == 2, f"got {len(r)}")
    check("truncated array drops the half-written clip",
          all(c["title"] != "Payoff" for c in r), str(r))

    r = jsonfix.repair_candidates(TRAILING_COMMA)
    check("trailing comma repaired", len(r) == 2, f"got {len(r)}")

    r = jsonfix.repair_candidates(SINGLE_OBJECT)
    check("json_object single-object coercion recovered", len(r) == 1, f"got {len(r)}")
    check("recovered single object keeps its index range",
          bool(r) and (r[0]["start_word"], r[0]["end_word"]) == (41, 190), str(r[:1]))

    r = jsonfix.repair_candidates(ENVELOPE)
    check("object envelope unwrapped", len(r) == 1, f"got {len(r)}")
    check("envelope clip keeps title",
          bool(r) and r[0]["title"] == "Setup", str(r[:1]))

    # --- salvage and normalisation ------------------------------------------
    r = jsonfix.repair_candidates(MIXED)
    check("broken element skipped, neighbours kept", len(r) >= 2, f"got {len(r)}")

    r = jsonfix.repair_candidates(PROSE)
    check("prose with no JSON yields no clips (not an exception)", r == [], str(r))

    r = jsonfix.repair_candidates(PY_LITERALS)
    check("None/True tolerated", len(r) == 1, f"got {len(r)}")
    check("None score becomes 0.0", bool(r) and r[0]["score"] == 0.0, str(r[:1]))

    r = jsonfix.repair_candidates(REVERSED)
    check("reversed range is normalised, not dropped", len(r) == 1, f"got {len(r)}")
    check("reversed range sorted",
          bool(r) and r[0]["start_word"] < r[0]["end_word"], str(r[:1]))

    r = jsonfix.repair_candidates(ALT_KEYS)
    check("alternative index keys accepted", len(r) == 1, f"got {len(r)}")

    r = jsonfix.repair_candidates(BRACKET_IN_STRING)
    check("bracket inside a string does not truncate the array",
          len(r) == 1 and r[0]["title"] == "The [good] part", str(r))

    # --- validation drops what cannot be timed ------------------------------
    for name, payload in [
        ("no indices", '[{"title": "nameless", "score": 90}]'),
        ("non-numeric indices", '[{"start_word": "a", "end_word": "b", "title": "x"}]'),
        ("zero-length range", '[{"start_word": 10, "end_word": 10, "title": "x"}]'),
        ("not an object", '["just a string", 42]'),
        ("empty array", "[]"),
        ("empty reply", ""),
        ("whitespace only", "   \n  "),
    ]:
        r = jsonfix.repair_candidates(payload)
        check(f"invalid clip dropped: {name}", r == [], f"got {r}")

    # score is clamped, not trusted
    r = jsonfix.repair_candidates('[{"start_word":1,"end_word":2,"score":9000,"title":"t"}]')
    check("score clamped to 100", bool(r) and r[0]["score"] == 100.0, str(r[:1]))
    r = jsonfix.repair_candidates('[{"start_word":1,"end_word":2,"score":-50,"title":"t"}]')
    check("score clamped to 0", bool(r) and r[0]["score"] == 0.0, str(r[:1]))

    # a missing title defaults rather than dropping a real moment
    r = jsonfix.repair_candidates('[{"start_word":1,"end_word":9,"score":70}]')
    check("missing title defaulted, clip kept",
          len(r) == 1 and r[0]["title"] == "Untitled clip", str(r))

    # long title truncated
    r = jsonfix.repair_candidates('[{"start_word":1,"end_word":9,"title":"%s"}]' % ("x" * 400))
    check("title truncated to 120", bool(r) and len(r[0]["title"]) == 120, str(len(r[0]["title"]) if r else 0))

    # never raises on hostile input
    for name, payload in [
        ("null", "null"), ("number", "123"), ("deep fences", "```\n```\n"),
        ("only fences", "```json\n```"), ("nul bytes", "[{\x00}]"),
        ("unclosed array", "[{"), ("unclosed string", '[{"a": "b'),
    ]:
        try:
            r = jsonfix.repair_candidates(payload)
            check(f"hostile input survives: {name}", isinstance(r, list), str(type(r)))
        except Exception as exc:  # noqa: BLE001
            check(f"hostile input survives: {name}", False, f"raised {type(exc).__name__}: {exc}")

    # a valid reply still parses exactly as before (no regression)
    good = '[{"start_word": 1, "end_word": 9, "score": 70, "title": "fine", "hook_text": "x", "hashtags": ["#fyp"]}]'
    r = jsonfix.repair_candidates(good)
    check("valid reply unchanged", len(r) == 1 and r[0]["hook_text"] == "x", str(r[:1]))
    check("optional fields preserved", bool(r) and r[0].get("hashtags") == ["#fyp"], str(r[:1]))

    print(f"\njsonfix: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
