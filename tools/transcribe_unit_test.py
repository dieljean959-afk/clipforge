#!/usr/bin/env python3
"""Unit checks for the repetition-hardened, engine-backed transcription helpers.

Pure-logic tests (no model load): chunk planning, overlap dedup, the
anti-repetition kwargs shared by both engines, and the engine resolver.
Run: python3 tools/transcribe_unit_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import asr  # noqa: E402
from app.subtitle import (  # noqa: E402
    CHUNK_SEC,
    OVERLAP_SEC,
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

    # --- anti-repetition kwargs (single source of truth: app.asr) ---------
    check("condition_on_previous_text=False", WHISPER_KWARGS["condition_on_previous_text"] is False)
    check("compression_ratio_threshold=2.2", WHISPER_KWARGS["compression_ratio_threshold"] == 2.2)
    check("no_speech_threshold=0.6", WHISPER_KWARGS["no_speech_threshold"] == 0.6)
    check("vad_filter=True", WHISPER_KWARGS["vad_filter"] is True)
    check("word_timestamps=True", WHISPER_KWARGS["word_timestamps"] is True)
    check(
        "kwargs are shared with the engine layer (no drift)",
        asr.ANTI_REPETITION["condition_on_previous_text"] is False
        and WHISPER_KWARGS["compression_ratio_threshold"]
        == asr.ANTI_REPETITION["compression_ratio_threshold"],
    )

    # --- chunk planning (resume units, not decode units) -------------------
    check("short file → single chunk", _plan_windows(10.0) == [(0.0, 10.0)],
          str(_plan_windows(10.0)))
    check("95s file → single chunk (under CHUNK_SEC)", _plan_windows(95.0) == [(0.0, 95.0)],
          str(_plan_windows(95.0)))

    # 2-hour source: the resume granularity must stay coarse (dozens of chunks,
    # not hundreds) while still covering the whole file.
    long = _plan_windows(7200.0)
    check("2h file → a handful of chunks (not hundreds)", 20 <= len(long) <= 30,
          f"{len(long)} chunks")
    check("chunks never exceed CHUNK_SEC",
          all(e - s <= CHUNK_SEC + 1e-9 for s, e in long))
    check("chunks overlap (no gaps)",
          all(s2 < e1 for (s1, e1), (s2, e2) in zip(long, long[1:])))
    check("chunks cover to the end", abs(long[-1][1] - 7200.0) < 1e-9, f"last end={long[-1][1]}")
    check("overlap matches OVERLAP_SEC",
          all(abs((e1 - s2) - OVERLAP_SEC) < 1e-6 for (s1, e1), (s2, e2) in zip(long, long[1:])),
          f"overlap={OVERLAP_SEC}")

    # --- overlap dedup -----------------------------------------------------
    # chunk 1 = [0, 300): "world" is spoken at 299.4s, near the end of it.
    # chunk 2 = [298, 600): the same "world" is re-decoded inside the overlap
    # and lands at the SAME absolute time. Dedupe must drop the twin, while
    # genuine repeated speech at different times is preserved.
    win1 = [Word(0.0, 0.3, "hello"), Word(299.4, 299.8, "world")]
    win2 = [Word(299.4, 299.8, "world"), Word(300.0, 300.4, "again")]
    out = _dedupe_words(win1 + win2)
    check("overlap duplicate removed", [w.word for w in out] == ["hello", "world", "again"],
          str([w.word for w in out]))

    far = [Word(0.4, 0.8, "world"), Word(299.4, 299.8, "world")]
    check("same word far apart in time kept", len(_dedupe_words(far)) == 2)

    # VAD padding can re-cut a word as an exact-timestamp twin — dropped
    twins = [Word(12.0, 12.4, "uh"), Word(12.0, 12.4, "uh")]
    check("exact-timestamp twin dropped", len(_dedupe_words(twins)) == 1)

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

    # a word just outside the overlap window is NOT treated as a duplicate
    edge = [Word(1.0, 1.2, "word"), Word(1.0 + OVERLAP_SEC + 0.5, 1.2 + OVERLAP_SEC + 0.5, "word")]
    check("word outside the overlap window kept", len(_dedupe_words(edge)) == 2)

    # --- engine resolution -------------------------------------------------
    check("engine names are the documented pair",
          asr.ENGINES == ("whisperx", "faster-whisper"), str(asr.ENGINES))
    ok, reason = asr.whisperx_available()
    check("whisperx availability is reported honestly", isinstance(ok, bool) and bool(reason),
          f"{ok} — {reason}")
    check("batch size is clamped to a sane range", 1 <= asr.batch_size(_Settings()) <= 64)
    check("whisperx_available() never raises", reason != "")

    print()
    if failures:
        print(f"TRANSCRIBE UNIT: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("TRANSCRIBE UNIT: ALL CHECKS PASSED")
    return 0


class _Settings:
    """Minimal stand-in for Settings.whisper (engine resolution is pure)."""

    class whisper:  # noqa: N801
        model = "medium"
        device = "cpu"
        compute_type = "int8"
        language = "auto"
        engine = "auto"
        batch_size = 8


if __name__ == "__main__":
    raise SystemExit(main())
