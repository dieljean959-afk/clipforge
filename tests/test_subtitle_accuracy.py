"""The accuracy pipeline's decisions, pinned without a model or a network.

Everything here is a pure function of data: window planning, the overlap
merge, the silence/repetition gates, the sync arithmetic, WER, subtitle
parsing, and the second-pass flagging. The model paths that need weights are
exercised by tools/verify_subtitles.py, which prints measurements.

    python -m pytest tests/test_subtitle_accuracy.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import asr_accurate as A  # noqa: E402
from app import speech_vad, subs_source  # noqa: E402
from app.subtitle import Transcript, Word  # noqa: E402


def R(start: float, end: float) -> speech_vad.Region:
    return speech_vad.Region(start, end)


def W(word: str, start: float, end: float, conf: float = 0.9, check: bool = False) -> dict:
    return {"word": word, "start": start, "end": end, "conf": conf, "check": check}


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #

def test_windows_cover_the_source_with_two_second_overlap():
    for duration in (0.4, 1, 29, 30, 31, 90, 1200):
        windows = A.plan_windows(duration)
        assert windows[0][0] == 0.0
        assert abs(windows[-1][1] - max(0.5, duration)) < 1e-9
        for a, b in zip(windows, windows[1:]):
            assert b[0] < a[1], "consecutive windows must overlap"
            assert abs(a[1] - b[0] - A.OVERLAP_SEC) < 1e-9
        for start, end in windows:
            assert 0 < end - start <= A.WINDOW_SEC + 1e-9


# --------------------------------------------------------------------------- #
# merge / order
# --------------------------------------------------------------------------- #

def test_enforce_order_sorts_and_removes_overlap():
    words = [W("b", 2.0, 3.0), W("a", 1.0, 2.5), W("c", 2.4, 2.6)]
    out = A.enforce_order(words)
    assert [w["word"] for w in out] == ["a", "b", "c"]
    assert out[1]["start"] >= out[0]["end"]
    assert all(w["end"] > w["start"] for w in out)


def test_dedupe_removes_the_seam_duplicate_only():
    from app.subtitle import _dedupe_words

    words = [
        Word(start=1.0, end=1.4, word="motoconcho"),
        Word(start=2.0, end=2.4, word="dime"),
        Word(start=2.1, end=2.5, word="motoconcho"),  # overlap re-decode, same text
        Word(start=9.0, end=9.4, word="motoconcho"),  # genuinely said again later
    ]
    out = _dedupe_words(words, 2.0)
    assert [w.word for w in out] == ["motoconcho", "dime", "motoconcho"]


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def test_silence_gate_drops_only_words_in_real_gaps():
    regions = [R(1.0, 2.0), R(6.0, 8.0)]
    words = [W("hello", 1.0, 1.4), W("music", 3.5, 4.0), W("edge", 6.0, 6.2)]
    kept, dropped = A.drop_silence_words(words, regions)
    assert [w["word"] for w in kept] == ["hello", "edge"]
    assert dropped == 1


def test_repetition_loop_collapses_but_real_repeats_survive():
    loop = [W("kai,", 60 + i * 0.4, 60 + i * 0.4 + 0.3, 0.3) for i in range(6)]
    kept, removed = A.collapse_repeats(loop)
    assert len(kept) == 3 and removed == 3

    deliberate = [W("no,", 10.0, 10.3), W("no,", 10.4, 10.7), W("no,", 10.8, 11.1)]
    kept, removed = A.collapse_repeats(deliberate)
    assert len(kept) == 3 and removed == 0


# --------------------------------------------------------------------------- #
# sync report
# --------------------------------------------------------------------------- #

def test_sync_report_scores_speech_without_caption_and_late_starts():
    regions = [R(1.0, 2.0), R(5.0, 6.0)]
    # perfect: captions start on the onset and cover the speech
    good = [W("a", 1.0, 2.0), W("b", 5.05, 6.0)]
    report = A.sync_report(good, regions, 10.0)
    assert report["passed"] is True
    assert report["speechWithoutCaption"] < 0.3
    assert report["avgStartErrorMs"] <= 150

    # a caption that starts 400 ms late and never covers the second region
    bad = [W("a", 1.4, 1.8)]
    report = A.sync_report(bad, regions, 10.0)
    assert report["passed"] is False
    assert report["avgStartErrorMs"] >= 400
    assert report["speechWithoutCaption"] > 0.3


def test_sync_report_cannot_pass_without_speech():
    report = A.sync_report([W("x", 0.0, 1.0)], [], 10.0)
    assert report["passed"] is False


def test_sync_report_clamps_words_past_the_end():
    report = A.sync_report([W("x", 50.0, 51.0)], [R(1.0, 2.0)], 10.0)
    assert report["captionOnSilence"] == 0.0
    assert isinstance(report["speechWithoutCaption"], float)


# --------------------------------------------------------------------------- #
# WER
# --------------------------------------------------------------------------- #

def test_wer_is_case_and_punctuation_insensitive():
    assert A.wer("HELLO there, world", "hello there world") == 0.0
    assert A.wer("a b c d", "a b x d") == 0.25
    assert A.wer("", "anything") is None
    assert A.wer("MISTER QUILTER", "Mr Quilter") == 0.0


# --------------------------------------------------------------------------- #
# subtitles
# --------------------------------------------------------------------------- #

JSON3 = json.dumps({"events": [
    {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "hello "}, {"utf8": "there"}]},
    {"tStartMs": 2000, "dDurationMs": 1500, "segs": [{"utf8": "general"}]},
]})

VTT = """WEBVTT

00:00:01.000 --> 00:00:02.500
hello there

00:00:02.500 --> 00:00:04.000
general
"""


def test_parse_json3_and_vtt():
    cues = subs_source.parse_json3(JSON3)
    assert [(c.start, c.end, c.text) for c in cues] == [
        (0.0, 2.0, "hello there"), (2.0, 3.5, "general")
    ]
    vtt = subs_source.parse_vtt(VTT)
    assert vtt[0].text == "hello there" and vtt[0].start == 1.0


def test_auto_caption_rolling_duplicates_collapse():
    rolling = """WEBVTT

00:00:01.000 --> 00:00:02.000
we are

00:00:02.000 --> 00:00:03.000
we are going

00:00:03.000 --> 00:00:04.000
going home
"""
    cues = subs_source.parse_vtt(rolling)
    assert [c.text for c in cues] == ["we are going", "going home"]


def test_source_hint_reads_the_job_document(tmp_path):
    settings = SimpleNamespace(app=SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)))
    job_id = "abc123"
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / f"{job_id}.json").write_text(json.dumps({
        "source": "https://youtu.be/vjR8dq43BBo", "title": "Kai", "description": "unboxing",
    }))
    media = tmp_path / "work" / job_id / "downloads" / "source.mp4"
    cache = tmp_path / "work" / job_id / "transcribe" / "chunks"
    hint = A.source_hint(media, cache, settings)
    assert hint.url == "https://youtu.be/vjR8dq43BBo"
    assert hint.title == "Kai"


def test_source_hint_is_optional(tmp_path):
    settings = SimpleNamespace(app=SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)))
    hint = A.source_hint(tmp_path / "x.mp4", None, settings)
    assert hint.url is None and hint.title == ""


# --------------------------------------------------------------------------- #
# transcript data contract
# --------------------------------------------------------------------------- #

def test_word_flags_serialise_only_when_set():
    assert Word(start=0, end=1, word="a").as_dict() == {
        "start": 0, "end": 1, "word": "a", "conf": 1.0,
    }
    assert Word(start=0, end=1, word="a", check=True).as_dict()["check"] is True


def test_transcript_carries_the_self_check_without_new_consumers():
    t = Transcript(words=[Word(start=0, end=1, word="a")], sync={"passed": True},
                   check_words=0, needs_review=False, aligned=True)
    assert t.sync["passed"] and t.aligned and not t.needs_review


# --------------------------------------------------------------------------- #
# alignment refusal + second pass
# --------------------------------------------------------------------------- #

class _StubReader:
    rate = 16000
    frames = 16000 * 60

    def samples(self, start: float, end: float):
        import numpy as np

        return np.zeros(int((end - start) * 16000), dtype="float32")


class _StubProcess:
    """Agrees on the first word, disagrees on the second, drops the third."""

    instances: list["_StubProcess"] = []

    def __init__(self, model: str, **kwargs) -> None:
        self.model = model
        _StubProcess.instances.append(self)

    def decode(self, wav, **kwargs):
        return {"ok": True, "peak_rss_mb": 500.0, "words": [
            {"word": "hello", "start": 0.1, "end": 0.4, "prob": 0.9},
            {"word": "word", "start": 0.5, "end": 0.9, "prob": 0.9},
        ]}

    def close(self) -> None:
        pass

    def kill(self) -> None:
        pass


def test_alignment_is_skipped_when_the_model_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(A.ctc_align, "model_ready", lambda root: False)
    words = [W("hello", 1.0, 1.4)]
    out, did = A.align_words(words, _StubReader(), [R(1.0, 2.0)], tmp_path)
    assert out == words and did is False


def test_second_pass_flags_disagreeing_words_and_never_replaces_them(monkeypatch, tmp_path):
    monkeypatch.setattr(A.asr_quality, "DecodeProcess", _StubProcess)
    words = [W("hello", 0.1, 0.4), W("wird", 0.5, 0.9), W("gone", 1.0, 1.3)]
    cycle = {"start": 0.0, "end": 2.0, "model": "small", "words": words,
             "segments": [{"start": 0.0, "end": 2.0, "avg_logprob": -1.4,
                           "no_speech_prob": 0.0, "temperature": 0.0}]}
    attempts: list = []
    flagged = A._second_pass([cycle], _StubReader(), tmp_dir=tmp_path,
                            language="en", title="t", attempts=attempts)
    assert flagged == 2
    assert [w["word"] for w in words] == ["hello", "wird", "gone"]  # text untouched
    assert [w["check"] for w in words] == [False, True, True]
    assert attempts and attempts[0].pass_no == 2


def test_second_pass_skips_confident_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(A.asr_quality, "DecodeProcess", _StubProcess)
    cycle = {"start": 0.0, "end": 2.0, "model": "small", "words": [W("a", 0.0, 0.5)],
             "segments": [{"start": 0.0, "end": 2.0, "avg_logprob": -0.2,
                           "no_speech_prob": 0.0, "temperature": 0.0}]}
    attempts: list = []
    assert A._second_pass([cycle], _StubReader(), tmp_dir=tmp_path, language="en",
                          title="t", attempts=attempts) == 0
    assert attempts == []


# --------------------------------------------------------------------------- #
# model ladder
# --------------------------------------------------------------------------- #

def test_model_ladder_is_largest_fitting_first():
    ladder = A.model_ladder("en")
    assert ladder[0] in ("distil-medium.en", "small.en", "small")
    assert all(A.asr_quality.fits(m) for m in ladder)
    spanish = A.model_ladder("es")
    assert "distil-medium.en" not in spanish


def test_manual_cue_timing_falls_back_when_alignment_refuses(monkeypatch):
    monkeypatch.setattr(A.ctc_align, "model_ready", lambda root: False)
    cues = [subs_source.Cue(10.0, 12.0, "hello there world")]
    words = A._align_cues(cues, _StubReader(), [R(10.0, 12.0)], Path("/nonexistent"),
                          None, 60.0)
    assert [w["word"] for w in words] == ["hello", "there", "world"]
    assert words[0]["start"] == 10.0
    assert words[-1]["end"] <= 12.0 + 1e-6
    assert all(b["start"] >= a["start"] for a, b in zip(words, words[1:]))


def test_prompt_uses_title_and_tail_without_exceeding_the_cap():
    prompt = A._prompt("Kai unboxing", "a very long description " * 40, "and then he said")
    assert prompt.startswith("Kai unboxing")
    assert prompt.endswith("and then he said")
    assert len(prompt) <= A.PROMPT_MAX_CHARS


class _ExplodingProcess:
    def __init__(self, *a, **k) -> None:
        raise AssertionError("a checkpointed window must never be decoded again")


class _MemoryProcess:
    def __init__(self, model: str, **kwargs) -> None:
        raise A.asr_quality.DecodeError("mkl_malloc: failed to allocate memory", memory=True)


def test_checkpointed_window_is_reused(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "win_00000.json").write_text(json.dumps({
        "start": 0.0, "end": 30.0, "model": "small",
        "params": {"window": [0.0, 30.0], "beam": 5, "temperature": [0.0, 0.2, 0.4]},
        "words": [W("cached", 1.0, 1.4), W("words", 1.5, 1.9)],
        "segments": [], "peak_rss_mb": 500.0,
    }), encoding="utf-8")
    monkeypatch.setattr(A.asr_quality, "DecodeProcess", _ExplodingProcess)
    words, cycles, attempts, model, detected, peak = A._decode_windows(
        _StubReader(), [(0.0, 30.0)], languages=["en"], prompt_base=("t", None),
        cache_dir=cache, tmp_dir=tmp_path, should_pause=None, on_progress=None,
        on_chunk=None, on_detail=None, duration=30.0,
    )
    assert [w["word"] for w in words] == ["cached", "words"]
    assert model == "small" and peak == 500.0
    assert attempts == []


def test_memory_failure_on_every_model_is_a_transcribe_error(monkeypatch, tmp_path):
    monkeypatch.setattr(A.asr_quality, "DecodeProcess", _MemoryProcess)
    with pytest_raises():
        A._decode_windows(
            _StubReader(), [(0.0, 30.0)], languages=["en"], prompt_base=("t", None),
            cache_dir=None, tmp_dir=tmp_path, should_pause=None, on_progress=None,
            on_chunk=None, on_detail=None, duration=30.0,
        )


def pytest_raises():
    """Local re-export so this module needs no pytest import."""
    import contextlib

    from app.subtitle import TranscribeError

    @contextlib.contextmanager
    def _cm():
        try:
            yield
        except TranscribeError:
            return
        raise AssertionError("a decode that cannot fit must raise TranscribeError")

    return _cm()


def test_transcribe_falls_back_and_says_so(monkeypatch, tmp_path):
    """If the accuracy pipeline cannot run, the legacy decoder still answers.

    The fallback must be LABELLED: a job that silently ran the old decoder
    while the transcript claims the new one would be the same class of lie the
    ladder was built to avoid.
    """
    from app import subtitle

    def boom(*args, **kwargs):
        raise RuntimeError("no model fits this host")

    monkeypatch.setattr(A, "transcribe_source", boom)
    monkeypatch.setattr(
        subtitle, "_transcribe_legacy",
        lambda *a, **k: Transcript(words=[Word(start=0, end=1, word="legacy")], label=""),
    )
    out = subtitle.transcribe(tmp_path / "x.mp4", SimpleNamespace())
    assert out.words[0].word == "legacy"
    assert "accuracy pipeline unavailable" in out.label
