#!/usr/bin/env python3
"""The dialect vocabulary builder.

Two things make this worth pinning:

1. Groq REJECTS (HTTP 400) any Whisper `prompt` longer than 224 tokens. It
   does not truncate server-side. So a user who pastes a generous glossary
   would otherwise have every chunk of every job fail — the builder has to
   fit, and fit by DROPPING TERMS rather than by being rejected.

2. `auto` and `caribbean_mix` must both produce usable vocabulary, because
   those are the defaults. An empty or near-empty vocabulary means the dialect
   selector does nothing at all, silently.

    python tools/dialect_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import dialect  # noqa: E402

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


def test_prompt_fits_groq_limit() -> None:
    print("\n[1] the Whisper prompt always fits Groq's 224-token ceiling")
    limit = dialect.GROQ_PROMPT_TOKEN_LIMIT
    check("the documented ceiling is 224 tokens", limit == 224, str(limit))

    # A pathological glossary: long, duplicated, and in several separators.
    huge = ", ".join(["nombrelarguisimodeempresa" * 2] * 40) + "\n" + ("x" * 4000)
    for label, dia, gloss in (
        ("no glossary", "caribbean_mix", ""),
        ("huge glossary", "caribbean_mix", huge),
        ("dominican + glossary", "dominican", "ClipForge, Don Tato, Mela"),
        ("colombian", "colombian", ""),
        ("puerto rican", "puerto_rican", ""),
        ("auto", "auto", ""),
        ("unknown id", "klingon", ""),
        ("whitespace only", "dominican", "   \n\t  "),
    ):
        p = dialect.whisper_prompt(dia, gloss)
        toks = dialect.estimate_tokens(p)
        check(f"{label}: {toks} tokens <= {limit}",
              toks <= limit, f"{toks} tokens, {len(p)} chars")
        check(f"{label}: prompt is never empty", bool(p.strip()))


def test_priority_when_truncating() -> None:
    print("\n[2] when the budget runs out, built-ins win over custom terms")
    # Enough custom terms to blow the budget on their own.
    flood = ", ".join(f"termino{i}deprueba" for i in range(200))
    p = dialect.whisper_prompt("dominican", flood)
    check("it still fits", dialect.estimate_tokens(p) <= dialect.GROQ_PROMPT_TOKEN_LIMIT)
    # The built-ins that make the dialect switch do anything must survive.
    for must in ("klk", "guagua", "vaina"):
        check(f"built-in {must!r} survived the truncation", must in p.lower())
    # A custom term near the END of a very long list is expected to be dropped;
    # a custom term in a SHORT list is expected to survive.
    short = dialect.whisper_prompt("dominican", "Mela, ClipForge")
    check("custom terms survive when the glossary is short",
          "mela" in short.lower() and "clipforge" in short.lower(), short)


def test_dialects_differ() -> None:
    print("\n[3] each dialect actually changes the vocabulary")
    dom = " ".join(dialect.vocabulary("dominican")).lower()
    col = " ".join(dialect.vocabulary("colombian")).lower()
    mix = " ".join(dialect.vocabulary("caribbean_mix")).lower()
    check("dominican carries its own words", "motoconcho" in dom)
    check("colombian carries its own words", "parche" in col and "bacano" in col)
    check("puerto rican carries its own words",
          "wepa" in " ".join(dialect.vocabulary("puerto_rican")).lower())
    check("caribbean_mix is the union of the caribbean dialects",
          "motoconcho" in mix and "wepa" in mix)
    check("auto is not empty (it is a selectable default)",
          len(dialect.vocabulary("auto")) > 5)
    check("an unknown dialect falls back rather than returning nothing",
          len(dialect.vocabulary("klingon")) > 5)


def test_glossary_parsing() -> None:
    print("\n[4] a glossary pasted on a phone parses sensibly")
    check("comma separated", dialect.split_terms("a, b, c") == ["a", "b", "c"])
    check("newline separated", dialect.split_terms("a\nb\nc") == ["a", "b", "c"])
    check("mixed separators", dialect.split_terms("a, b\nc; d") == ["a", "b", "c", "d"])
    # Multi-word entries are KEPT WHOLE: the field exists for names and brands,
    # and "Don Tato" must survive as one term for the cleanup model, not two
    # unrelated words. An earlier version split on spaces too, which was wrong.
    check("a multi-word entry stays whole",
          dialect.split_terms("la cosa, x") == ["la cosa", "x"],
          str(dialect.split_terms("la cosa, x")))
    check("brand names keep their spacing",
          dialect.split_terms("ClipForge, Don Tato") == ["ClipForge", "Don Tato"],
          str(dialect.split_terms("ClipForge, Don Tato")))
    check("a bare list of three or fewer words is ONE term",
          dialect.split_terms("Casa Bonita") == ["Casa Bonita"],
          str(dialect.split_terms("Casa Bonita")))
    check("four or more bare words is a list, not one term",
          dialect.split_terms("tigre pana parce guacha") ==
          ["tigre", "pana", "parce", "guacha"],
          str(dialect.split_terms("tigre pana parce guacha")))
    check("extra internal whitespace is normalised",
          dialect.split_terms("Casa   Bonita") == ["Casa Bonita"],
          str(dialect.split_terms("Casa   Bonita")))
    check("duplicates are dropped, order kept",
          dialect.split_terms("b, a, b") == ["b", "a"])
    check("quotes and stray dots are stripped",
          dialect.split_terms('"ClipForge.", Mela') == ["ClipForge", "Mela"])
    check("empty input is empty", dialect.split_terms("") == [])


def test_cleanup_hint() -> None:
    print("\n[5] the cleanup hint carries the full vocabulary, unpoliced")
    hint = dialect.cleanup_hint("colombian", "ClipForge, Don Tato")
    for must in ("chimba", "bacano", "parce", "ClipForge", "Don Tato",
                 "Colombian Spanish"):
        check(f"cleanup prompt mentions {must!r}", must in hint)
    # The two prompts must not be the same thing: one is a spelling hint, the
    # other is an instruction to an editor.
    w = dialect.whisper_prompt("colombian", "ClipForge")
    c = dialect.cleanup_hint("colombian", "ClipForge")
    check("the two prompts are genuinely different", w != c)
    check("the cleanup hint is the longer one", len(c) > len(w))
    check("the Whisper prompt is a bare spelling hint, not prose",
          "CORRECT SPELLINGS" not in w, w[:120])


def main() -> int:
    test_prompt_fits_groq_limit()
    test_priority_when_truncating()
    test_dialects_differ()
    test_glossary_parsing()
    test_cleanup_hint()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
