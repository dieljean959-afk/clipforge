"""Captions must cover the WHOLE clip (LOCKED_SPEC §4).

The bug these tests pin: a 52.5 s clip lost its captions at 24.5 s because one
30 s ASR window (924-954 s) decoded to ZERO words, the empty window was stored
as a success, the second pass skipped it explicitly, and nothing compared the
words with where Silero says people are talking. The clip rendered with 16.3 s
of speech uncaptioned.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest

from app import asr_accurate, speech_vad, translate
from app.config import load_config
from app.subtitle import TranscribeError, Transcript, Word, _WavReader


@pytest.fixture(scope="module")
def settings():
    return load_config(quiet=True)


def _wav(path: Path, seconds: float = 30.0) -> Path:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


# --------------------------------------------------------------------------- #
# 1. a 60 s word list that crosses the 30 s window boundary, with bad words
# --------------------------------------------------------------------------- #

def test_captions_survive_the_window_boundary_and_bad_words(settings):
    """60 s of words across the 30 s boundary: the last cue still reaches 60 s.

    One word per 0.4 s in two windows that overlap by 2 s around 30 s, plus the
    three kinds of bad word the builder used to choke on: NaN timing, end before
    start, and empty text.
    """
    alphabet = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    words: list[Word] = []
    for i in range(150):  # 0.0 .. 60.0 s
        if i == 40:
            words.append(Word(start=float("nan"), end=16.4, word="nanword"))  # NaN start
            continue
        if i == 70:
            words.append(Word(start=28.4, end=28.0, word="backwards"))  # end < start
            continue
        if i == 90:
            words.append(Word(start=36.0, end=36.3, word="   "))  # empty text
            continue
        start = round(i * 0.4, 2)
        words.append(Word(start=start, end=round(start + 0.3, 2), word=alphabet[i % 6],
                          conf=0.95))

    transcript = Transcript(words=words, duration=60.0, language="en")
    cues = translate.build_cues(transcript, {"start": 0.0, "end": 60.0}, settings, language="en")

    assert cues, "no cues at all"
    assert cues[0]["start"] <= 0.5
    # THE regression: captions used to stop wherever the bad/truncated data was.
    assert cues[-1]["end"] >= 59.0, f"captions stop at {cues[-1]['end']:.2f}s of a 60s clip"
    # Nothing after the 30 s window boundary was dropped.
    assert max(c["end"] for c in cues if c["start"] < 31.0) > 29.0
    assert any(c["start"] > 31.0 for c in cues)


def test_a_bad_word_does_not_end_the_builder(settings):
    """The bad word is skipped, the words after it are still captioned."""
    words = [
        Word(start=1.0, end=1.4, word="before"),
        Word(start=float("inf"), end=2.0, word="broken"),
        Word(start=2.2, end=2.6, word="after"),
        Word(start=3.0, end=3.4, word="again"),
    ]
    cues = translate.build_cues(
        Transcript(words=words, duration=4.0), {"start": 0.0, "end": 4.0}, settings,
    )
    text = " ".join(c["text"] for c in cues).lower()
    assert "after" in text and "again" in text
    assert "broken" not in text


# --------------------------------------------------------------------------- #
# 2. a failed / empty ASR window is retried, never silently dropped
# --------------------------------------------------------------------------- #

class _PromptSensitive:
    """A decode child that returns NOTHING whenever it is given a prompt."""

    calls: list[dict] = []

    def __init__(self, model: str, **_kwargs):
        self.model = model
        _PromptSensitive.calls.append({"model": model})

    def decode(self, _wav, **_settings):
        _PromptSensitive.calls.append({"prompt": _settings.get("initial_prompt")})
        if _settings.get("initial_prompt"):
            return {"words": [], "segments": [], "language": "en", "peak_rss_mb": 900.0}
        return {
            "words": [{"word": " hello", "start": 0.1, "end": 0.4, "prob": 0.9}],
            "segments": [{"start": 0.1, "end": 0.4, "text": " hello", "avg_logprob": -0.2,
                          "no_speech_prob": 0.01, "compression_ratio": 1.0, "temperature": 0.0}],
            "language": "en",
            "peak_rss_mb": 900.0,
        }

    def close(self):
        pass

    def kill(self):
        pass


class _AlwaysEmpty(_PromptSensitive):
    def decode(self, _wav, **_settings):
        return {"words": [], "segments": [], "language": "en", "peak_rss_mb": 900.0}


def _decode_one_window(monkeypatch, tmp_path, fake):
    _PromptSensitive.calls = []
    monkeypatch.setattr(asr_accurate.asr_quality, "DecodeProcess", fake)
    wav = _wav(tmp_path / "speech16k.wav")
    with _WavReader(wav) as reader:
        return asr_accurate._decode_windows(
            reader, [(0.0, 30.0)], languages=["en"], prompt_base=("Title", "desc"),
            cache_dir=None, tmp_dir=tmp_path, should_pause=None, on_progress=None,
            on_chunk=None, on_detail=None, duration=30.0, regions=None,
        )


def test_an_empty_window_is_re_decoded_without_the_prompt(monkeypatch, tmp_path):
    words, cycles, attempts, _model, _lang, _peak = _decode_one_window(
        monkeypatch, tmp_path, _PromptSensitive
    )

    assert words, "the window was silently dropped"
    assert [a.ok for a in attempts] == [True, True]
    assert attempts[0].reason == "returned 0 words"
    assert cycles[0]["words"]
    assert [c.get("prompt") for c in _PromptSensitive.calls if "prompt" in c][0]
    assert [c.get("prompt") for c in _PromptSensitive.calls if "prompt" in c][1] is None


def test_a_window_no_model_can_decode_fails_loudly(monkeypatch, tmp_path):
    with pytest.raises(TranscribeError) as err:
        _decode_one_window(monkeypatch, tmp_path, _AlwaysEmpty)
    assert "refusing a transcript with a hole" in str(err.value)


def test_an_empty_checkpoint_is_not_resumed(monkeypatch, tmp_path):
    """A stored window with zero words is re-decoded, not trusted."""
    regions = [speech_vad.Region(5.0, 20.0)]  # Silero says this window has speech
    assert asr_accurate._checkpoint_usable({"words": []}, 0.0, 30.0, regions) is False
    assert asr_accurate._checkpoint_usable({"words": [{"word": "hi"}]}, 0.0, 30.0, regions) is True
    silent = [speech_vad.Region(100.0, 130.0)]
    assert asr_accurate._checkpoint_usable({"words": []}, 0.0, 30.0, silent) is True


# --------------------------------------------------------------------------- #
# 3. the coverage guard: uncovered speech is re-transcribed, then flagged
# --------------------------------------------------------------------------- #

class _HoleFiller:
    """Fills any hole with one word per second of it."""

    def __init__(self, work: bool = True):
        self.work = work
        self.calls: list[tuple[float, float]] = []
        self.closed = False

    def decode(self, start: float, end: float):
        self.calls.append((start, end))
        if not self.work:
            return []
        out, t = [], start
        while t + 0.4 < end:
            out.append({"start": round(t, 2), "end": round(t + 0.35, 2), "word": "w", "conf": 0.9})
            t += 0.45
        return out

    def close(self):
        self.closed = True


def test_uncovered_speech_is_re_transcribed():
    regions = [speech_vad.Region(10.0, 20.0)]
    words = [{"start": 9.0, "end": 10.2, "word": "hi", "conf": 0.9}]
    decoder = _HoleFiller()

    fixed, report = asr_accurate.cover_speech(
        words, regions, wav=Path("unused.wav"), span=(0.0, 30.0),
        language="en", decoder=decoder,
    )

    assert decoder.calls and decoder.calls[0][0] >= 10.0
    assert report["repairedWords"] > 0
    assert report["coveredFraction"] >= asr_accurate.COVERAGE_MIN_FRACTION
    assert report["needsReview"] is False
    assert len(fixed) > len(words)


def test_still_uncovered_marks_needs_review_with_the_seconds():
    regions = [speech_vad.Region(10.0, 20.0)]
    words = [{"start": 9.0, "end": 10.2, "word": "hi", "conf": 0.9}]

    _fixed, report = asr_accurate.cover_speech(
        words, regions, wav=Path("unused.wav"), span=(0.0, 30.0),
        language="en", decoder=_HoleFiller(work=False),
    )

    assert report["needsReview"] is True
    assert "captions missing at" in report["reason"]
    assert report["uncovered"][0][0] >= 10.0


def test_coverage_report_fails_a_clip_whose_captions_stop_early():
    """The real shape of the bug: words stop at 24.5 s of a 52.5 s clip."""
    regions = [speech_vad.Region(5.0, 24.5), speech_vad.Region(27.0, 43.5),
               speech_vad.Region(49.0, 52.5)]
    words = [(5.0, 24.5), (49.0, 52.5)]

    report = asr_accurate.coverage_report(words, regions, span=(0.0, 52.5))

    assert report["passed"] is False
    assert report["needsReview"] is True
    assert report["coveredFraction"] < 0.95
    assert [round(a) for a, _b in report["uncovered"]] == [27]
    assert "captions missing at 27.0-43.5 s" in report["reason"]
    assert report["lastSpeechSec"] == 52.5


def test_a_sub_second_residual_hole_still_passes_the_95_percent_rule():
    """A laugh Silero counts but no model has words for is inside the budget."""
    regions = [speech_vad.Region(0.0, 20.0)]
    words = [(0.0, 9.0), (9.7, 20.0)]  # 0.7 s with no words

    report = asr_accurate.coverage_report(words, regions, span=(0.0, 20.0))

    assert report["passed"] is True
    assert report["coveredFraction"] >= 0.95
    assert report["uncovered"] == [[9.0, 9.7]]  # still reported, for honesty


def test_a_wordless_voiced_blip_is_not_missing_captions():
    """The real batch: a 1.0 s VAD blip every model refuses is not speech.

    Both flagged clips in production had exactly this shape (0.85 s and 1.04 s
    of voiced audio that distil-medium.en, small and base all decoded to zero
    words). Flagging those told the user a finished clip needed review over a
    sound nobody can caption.
    """
    regions = [speech_vad.Region(0.0, 10.0), speech_vad.Region(11.0, 12.0)]
    words = [{"start": 0.0, "end": 10.0, "word": "hi", "conf": 0.9}]

    _fixed, report = asr_accurate.cover_speech(
        words, regions, wav=Path("unused.wav"), span=(0.0, 12.0),
        language="en", decoder=_HoleFiller(work=False),
    )

    assert report["nonSpeech"] == [[11.0, 12.0]]
    assert report["needsReview"] is False
    assert report["coveredFraction"] == 1.0


def test_a_long_wordless_stretch_is_still_a_hole():
    """The 924-954 s signature stays flagged: 20 s of speech, zero words."""
    regions = [speech_vad.Region(0.0, 30.0)]
    words = [{"start": 0.0, "end": 10.0, "word": "hi", "conf": 0.9}]

    _fixed, report = asr_accurate.cover_speech(
        words, regions, wav=Path("unused.wav"), span=(0.0, 30.0),
        language="en", decoder=_HoleFiller(work=False),
    )

    assert report["nonSpeech"] == []
    assert report["needsReview"] is True
    assert "captions missing at" in report["reason"]


def test_coverage_report_passes_a_fully_captioned_clip():
    regions = [speech_vad.Region(1.0, 10.0), speech_vad.Region(12.0, 20.0)]
    words = [(1.0, 10.0), (12.0, 20.0)]

    report = asr_accurate.coverage_report(words, regions, span=(0.0, 20.0))

    assert report["passed"] is True
    assert report["coveredFraction"] == 1.0
    assert math.isclose(report["lastSpeechSec"] - report["lastCaptionSec"], 0.0)
