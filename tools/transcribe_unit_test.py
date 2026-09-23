#!/usr/bin/env python3
"""Unit checks for the repetition-hardened transcription helpers.

Pure-logic tests (no model load): window planning, overlap dedup, and the
anti-repetition kwargs. Run: python3 tools/transcribe_unit_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.subtitle import (  # noqa: E402
    OVERLAP_SEC,
    WINDOW_SEC,
    WHISPER_KWARGS,
    Word,
    _dedupe_words,
    _plan_windows,
)


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"   {'OK ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # --- anti-repetition kwargs ------------------------------------------
    check("condition_on_previous_text=False", WHISPER_KWARGS["condition_on_previous_text"] is False)
    check("compression_ratio_threshold=2.2", WHISPER_KWARGS["compression_ratio_threshold"] == 2.2)
    check("no_speech_threshold=0.6", WHISPER_KWARGS["no_speech_threshold"] == 0.6)
    check("vad_filter=True", WHISPER_KWARGS["vad_filter"] is True)
    check("word_timestamps=True", WHISPER_KWARGS["word_timestamps"] is True)

    # --- window planning ---------------------------------------------------
    check("short file → single window", _plan_windows(10.0) == [(0.0, 10.0)],
          str(_plan_windows(10.0)))
    long = _plan_windows(95.0)
    check("95s file → multiple ~30s windows", len(long) == 4 and long[0] == (0.0, 30.0),
          f"{len(long)} windows, first={long[0]}, last={long[-1]}")
    check("windows overlap (no gaps)",
          all(s2 < e1 for (s1, e1), (s2, e2) in zip(long, long[1:])))
    check("windows cover to the end", abs(long[-1][1] - 95.0) < 1e-9, f"last end={long[-1][1]}")
    check("windows never exceed WINDOW_SEC",
          all(e - s <= WINDOW_SEC + 1e-9 for s, e in long))

    # --- overlap dedup -----------------------------------------------------
    # word "world" re-decoded inside the 1s overlap between window 1 and 2
    win1 = [Word(0.0, 0.3, "hello"), Word(0.4, 0.8, "world")]
    win2 = [Word(29.4, 29.7, "world"), Word(30.0, 30.4, "again")]  # "world" dup, shifted +29
    out = _dedupe_words(win1 + win2)
    check("overlap duplicate removed", [w.word for w in out] == ["hello", "world", "again"],
          str([w.word for w in out]))

    # genuinely repeated speech at different times must survive
    speech = [Word(1.0, 1.2, "yeah"), Word(10.0, 10.2, "yeah")]
    check("real repeats at different times kept", len(_dedupe_words(speech)) == 2)

    # different words near the boundary must survive
    diff = [Word(5.0, 5.2, "cat"), Word(5.9, 6.1, "dog")]
    check("distinct words near boundary kept", len(_dedupe_words(diff)) == 2)

    # punctuation/case normalization in the dup comparison
    punct = [Word(2.0, 2.2, "Yeah!"), Word(2.3, 2.6, "yeah")]
    check("case/punctuation normalized for dup detection",
          len(_dedupe_words(punct)) == 1, str([w.word for w in _dedupe_words(punct)]))

    print()
    if failures:
        print(f"TRANSCRIBE UNIT: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("TRANSCRIBE UNIT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
