#!/usr/bin/env python3
"""Groq transcription: chunk planning and the merge.

The network parts need a key and are exercised live separately; what is
pinned here is the logic that is easy to get quietly wrong and impossible to
eyeball:

  * a merge that DROPS the last seconds of a source (the ending hook of a
    clip is exactly what a creator is selling);
  * a merge that KEEPS the overlap twice, which shows up as a repeated
    sentence mid-transcript and is very visible in burned-in captions;
  * chunk planning that produces a chunk over Groq's 25 MB ceiling, which
    comes back as an opaque 400 and kills the job;
  * a confidence value invented for a transcript that never measured one.

    python tools/groq_asr_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import groq_asr as g  # noqa: E402

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


def test_plan_covers_everything() -> None:
    print("\n[1] chunk planning covers the whole source, including the tail")
    for dur in (0, 1, 30, 599, 600, 601, 1200, 1800, 3600, 7260):
        chunks = g.plan_chunks(dur)
        if not chunks:
            check(f"{dur}s: at least one chunk", False)
            continue
        first = chunks[0][0]
        last_end = chunks[-1][1]
        ok = first == 0.0 and abs(last_end - max(0.0, dur)) < 1e-6
        check(f"{dur}s: starts at 0 and ends exactly at the source end "
              f"({chunks[0]} .. {chunks[-1]})", ok)

    # No gaps, and every chunk respects the overlap.
    chunks = g.plan_chunks(3600)
    gaps = [chunks[i][0] - chunks[i - 1][1] for i in range(1, len(chunks))]
    check("no gap between consecutive chunks", all(gp <= 0 for gp in gaps), str(gaps))
    check("every interior chunk overlaps its predecessor",
          all(chunks[i][0] < chunks[i - 1][1] for i in range(1, len(chunks) - 1)))


def test_budget_fits_upload_limit() -> None:
    print("\n[2] planned chunks stay inside Groq's upload ceiling")
    # 2 hours of audio at a chunk size that would blow the limit without
    # this guard.
    dur = 7200
    chunk, overlap = g.chunk_seconds_for_budget(dur, g.TARGET_UPLOAD_BYTES, g.CHUNK_SEC, g.OVERLAP_SEC)
    est_bytes = chunk * g.FLAC_BITS_PER_SEC / 8
    check(f"estimated chunk upload {est_bytes / 1024 / 1024:.1f} MB <= "
          f"{g.TARGET_UPLOAD_BYTES / 1024 / 1024:.0f} MB",
          est_bytes <= g.TARGET_UPLOAD_BYTES, f"{est_bytes}")
    check("overlap never exceeds a third of the chunk", overlap <= chunk / 3.0 + 1e-6)

    # A deliberately tiny ceiling must still produce something usable.
    chunk, overlap = g.chunk_seconds_for_budget(dur, 1024 * 1024, g.CHUNK_SEC, g.OVERLAP_SEC)
    check("a 1 MB ceiling shrinks the chunk", chunk < g.CHUNK_SEC, str(chunk))
    check("the shrunken chunk still fits", chunk * g.FLAC_BITS_PER_SEC / 8 <= 1024 * 1024)
    check("a short source is not split into pointless chunks",
          g.chunk_seconds_for_budget(45)[0] <= 45)


def test_merge_offsets_and_dedupes() -> None:
    print("\n[3] merge offsets timestamps and drops overlap duplicates")

    def seg(s, e, t, lp=-0.25):
        return {"start": s, "end": e, "text": t, "avg_logprob": lp}

    # Two 100s chunks overlapping by 3s. The second re-decodes the last 3
    # seconds; those segments must appear once.
    results = [
        g.ChunkResult(0.0, 100.0, [
            seg(0.0, 4.0, "KLK qué lo que"),
            seg(4.0, 8.0, "tú tá' loco"),
            seg(96.0, 100.0, "motoconcho"),
        ]),
        g.ChunkResult(97.0, 200.0, [
            seg(0.0, 3.0, "motoconcho"),      # overlap re-decode -> dropped
            seg(3.0, 8.0, "dime a ver"),
            seg(80.0, 100.0, "la guagua"),
        ]),
    ]
    merged = g.merge_chunks(results)
    texts = [m["text"] for m in merged]
    check("the overlap segment appears once, not twice",
          texts.count("motoconcho") == 1, str(texts))
    check("both chunks contribute text",
          "KLK qué lo que" in texts and "dime a ver" in texts, str(texts))

    # Timestamps must be ABSOLUTE, i.e. offset by the chunk start.
    second = [m for m in merged if m["text"] == "la guagua"]
    check("timestamps are offset by the chunk start",
          bool(second) and abs(second[0]["start"] - 177.0) < 0.01,
          str(second))
    check("timestamps are ordered and non-overlapping",
          all(merged[i]["start"] <= merged[i + 1]["start"] + 1e-6
              for i in range(len(merged) - 1)), str([m["start"] for m in merged]))
    check("no zero-length segments", all(m["end"] > m["start"] for m in merged))


def test_merge_edges() -> None:
    print("\n[4] merge survives the awkward inputs")

    def seg(s, e, t):
        return {"start": s, "end": e, "text": t}

    check("an empty result list merges to nothing", g.merge_chunks([]) == [])
    check("a chunk with no segments merges to nothing",
          g.merge_chunks([g.ChunkResult(0, 10, [])]) == [])
    check("blank segment text is dropped",
          g.merge_chunks([g.ChunkResult(0, 10, [seg(0, 2, "   ")])]) == [])
    check("a backwards segment is dropped, not emitted as negative time",
          g.merge_chunks([g.ChunkResult(0, 10, [seg(5, 2, "x")])]) == [])
    check("non-numeric timestamps do not crash the merge",
          g.merge_chunks([g.ChunkResult(0, 10, [{"start": "x", "end": "y", "text": "z"}])]) == [])


def test_confidence_is_not_invented() -> None:
    print("\n[5] confidence is measured or absent, never assumed")
    check("a real avg_logprob converts",
          abs(g.segment_confidence({"avg_logprob": -0.1}) - 0.905) < 0.01,
          str(g.segment_confidence({"avg_logprob": -0.1})))
    check("a missing avg_logprob is 0, not 1.0",
          g.segment_confidence({}) == 0.0)
    check("a nonsense avg_logprob is 0, not a crash",
          g.segment_confidence({"avg_logprob": "banana"}) == 0.0)
    check("confidence is clamped into 0..1",
          g.segment_confidence({"avg_logprob": -99.0}) >= 0.0)


def test_words_from_segments() -> None:
    print("\n[6] merged segments become a usable word timeline")
    segs = [{"start": 10.0, "end": 12.0, "text": "klk que lo", "conf": 0.9}]
    words = g.segments_to_words(segs)
    check("one word per token", len(words) == 3, str(words))
    check("words are inside the segment",
          all(10.0 <= w["start"] and w["end"] <= 12.0 for w in words), str(words))
    check("words are ordered and contiguous",
          all(words[i]["end"] <= words[i + 1]["start"] + 1e-6 for i in range(len(words) - 1)))
    check("an empty segment yields no words",
          g.segments_to_words([{"start": 0, "end": 1, "text": "   "}]) == [])


def test_retry_classification() -> None:
    print("\n[7] a retryable fault is retried; a hopeless one is not")
    # Build the error object without needing a live socket.
    import urllib.error

    def http_error(code):
        return urllib.error.HTTPError("u", code, "m", {}, None)

    class _Raiser:
        def __init__(self, exc):
            self.exc = exc

    import unittest.mock as mock

    audio = ROOT / "tools" / "_groq_test_chunk.flac"
    audio.write_bytes(b"RIFF" + b"\0" * 512)
    try:
        for code, want_hard in ((429, False), (500, False), (503, False),
                                (400, True), (401, True), (403, True), (413, True)):
            cfg = g.GroqConfig(retries=2, timeout_sec=5)
            with mock.patch("urllib.request.urlopen", side_effect=http_error(code)), \
                 mock.patch("time.sleep"):
                try:
                    g._post_chunk(audio, cfg, "k", "p", "es")
                    got = "no error"
                except g.GroqHardError:
                    got = "hard"
                except g.GroqAsrError:
                    got = "soft"
            expected = "hard" if want_hard else "soft"
            check(f"HTTP {code} -> {expected}", got == expected, f"got {got}")
    finally:
        audio.unlink(missing_ok=True)


def test_oversize_refused_before_upload() -> None:
    print("\n[8] an oversize chunk is refused locally, not by a 400 from Groq")
    import unittest.mock as mock

    audio = ROOT / "tools" / "_groq_big.flac"
    audio.write_bytes(b"RIFF" + b"\0" * 1024)
    try:
        cfg = g.GroqConfig(max_upload_bytes=512, retries=1)
        with mock.patch("urllib.request.urlopen") as urlopen:
            try:
                g._post_chunk(audio, cfg, "k", "p", "es")
                check("an oversize chunk raises", False, "no error raised")
            except g.GroqHardError as exc:
                check("an oversize chunk raises before any request",
                      True, str(exc)[:70])
            check("and never touched the network", not urlopen.called)
    finally:
        audio.unlink(missing_ok=True)


def test_silence_is_dropped() -> None:
    print("\n[9] Groq's own no_speech_prob gate stops hallucinations over silence")
    # Observed live: 700s of pure tone with NO speech came back as
    # "Subtitulado por Jnkoil" repeated in every chunk. That is the Whisper
    # prompt bias plus no silence gate, and it would land in burned-in
    # captions over every quiet gap in real footage.
    check("a high no_speech_prob segment is dropped",
          g.is_silence({"no_speech_prob": 0.95}))
    check("the threshold matches the local engine's",
          g.NO_SPEECH_THRESHOLD == 0.6, str(g.NO_SPEECH_THRESHOLD))
    check("a genuinely spoken segment is kept",
          not g.is_silence({"no_speech_prob": 0.05}))
    check("a segment with no probability is KEPT, not dropped",
          not g.is_silence({}),
          "dropping everything would empty a working transcript")
    check("a non-numeric probability does not crash the gate",
          not g.is_silence({"no_speech_prob": "banana"}))

    results = [g.ChunkResult(0.0, 100.0, [
        {"start": 0, "end": 4, "text": "KLK qué lo que", "no_speech_prob": 0.02},
        {"start": 4, "end": 20, "text": "Subtitulado por Jnkoil", "no_speech_prob": 0.99},
        {"start": 20, "end": 24, "text": "dime a ver", "no_speech_prob": 0.10},
    ])]
    merged = g.merge_chunks(results)
    texts = [m["text"] for m in merged]
    check("the hallucinated silence segment is gone from the merge",
          "Subtitulado por Jnkoil" not in texts, str(texts))
    check("the real speech either side survives",
          len(merged) == 2, str(texts))


def test_repetition_loop_collapsed() -> None:
    print("\n[10] a Whisper repetition loop is collapsed, not transcribed")
    # The no_speech gate does NOT catch this: measured live, Groq returned the
    # invented segment over a 700s pure tone with no_speech_prob=0.21 and
    # avg_logprob=-0.70 — confident it had heard speech. Repetition is the only
    # reliable signal, and 21 copies of one invented line is exactly the
    # failure that ends up burned into a caption.
    def seg(s, e, t):
        return {"start": s, "end": e, "text": t, "no_speech_prob": 0.21}

    loop = g.merge_chunks([g.ChunkResult(0.0, 300.0, [
        seg(0, 10, "Subtitulado por Jnkoil"),
        seg(33, 43, "Subtitulado por Jnkoil"),
        seg(66, 76, "Subtitulado por Jnkoil"),
        seg(99, 109, "Subtitulado por Jnkoil"),
    ])])
    check("four identical hallucinated segments collapse to one",
          len(loop) == 1, f"kept {len(loop)}: {[m['text'] for m in loop]}")
    check("the surviving copy keeps its real timestamp",
          bool(loop) and loop[0]["start"] == 0.0, str(loop))

    # Real speech must not be eaten: alternation is not repetition.
    real = g.merge_chunks([g.ChunkResult(0.0, 60.0, [
        seg(0, 3, "KLK qué lo que"),
        seg(3, 6, "tú tá' loco"),
        seg(6, 9, "KLK qué lo que"),
        seg(9, 12, "dime a ver"),
    ])])
    check("genuinely alternating speech is left alone",
          len(real) == 4, str([m["text"] for m in real]))


def main() -> int:
    test_plan_covers_everything()
    test_budget_fits_upload_limit()
    test_merge_offsets_and_dedupes()
    test_merge_edges()
    test_confidence_is_not_invented()
    test_words_from_segments()
    test_retry_classification()
    test_oversize_refused_before_upload()
    test_silence_is_dropped()
    test_repetition_loop_collapsed()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

