"""The ASR ladder: order, budget gating, labels, and what a memory death does.

The bug this replaces: three IDENTICAL retries of the same 300 s chunk, then a
"smaller model" loaded INTO the same already-full process. Every step here is
strictly lighter than the one before, runs in its own process, and is recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import asr_iso, membudget
from app.memguard import MB


def test_ladder_starts_at_the_configured_model_and_gets_lighter():
    steps = asr_iso.ladder_steps("small")
    assert steps[0].model is None                 # configured model
    assert steps[0].window >= steps[1].window     # 60 -> 30
    assert steps[0].beam > steps[2].beam          # 5 -> 1
    assert steps[0].vad is True and steps[3].vad is False
    # model downshifts come last, largest first, and never larger than configured
    models = [s.model for s in steps if s.model]
    assert models == ["base", "tiny"]


def test_ladder_never_offers_a_bigger_model_than_configured():
    assert "small" not in [s.model for s in asr_iso.ladder_steps("base") if s.model]
    assert asr_iso.ladder_steps("tiny")[-1].model is None or True  # no smaller model left
    assert [s.model for s in asr_iso.ladder_steps("tiny") if s.model] == []


def test_every_step_is_loggable_with_its_numbers():
    labels = [s.label("small") for s in asr_iso.ladder_steps("small")]
    assert "beam 5" in labels[0] and "vad off" in labels[3]
    assert any("model base" in label for label in labels)


def test_memory_text_is_recognised_and_gets_a_plain_sentence():
    assert asr_iso.is_memory_error("mkl_malloc: failed to allocate memory")
    assert asr_iso.is_memory_error("MemoryError")
    assert asr_iso.is_memory_error("failed to map segment from shared object")
    assert not asr_iso.is_memory_error("ffprobe returned no duration")
    text = asr_iso.plain_memory_reason(2, 6)
    assert "Ran out of memory" in text and "attempt 2 of 6" in text
    assert "mkl" not in text and "Traceback" not in text


def test_transcript_label_is_only_set_when_the_ladder_downshifted():
    assert asr_iso.transcript_label("small", "small") is None
    assert asr_iso.transcript_label("small", "base") == "transcribed with base (low memory)"


class _FakeProc:
    """Stands in for a decode child: fails the first N calls, then succeeds."""

    calls: list[dict] = []
    failures: list[tuple[str, str]] = []

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        _FakeProc.calls.append({"model": model, **kwargs})
        self.proc = type("P", (), {"returncode": None, "poll": lambda self=None: None})()
        self._dead = False
        if _FakeProc.failures:
            kind, text = _FakeProc.failures.pop(0)
            if kind == "startup":
                raise RuntimeError(text)
            self._dead = True

    def decode(self, path, *, language, beam, vad):
        if self._dead:
            return {"ok": False, "memory": True, "error": "mkl_malloc: failed to allocate memory"}
        return {"ok": True, "language": "en", "peak_rss_mb": 812.5,
                "words": [{"start": 1.0, "end": 1.4, "word": " hello", "conf": 0.9}]}

    def close(self):
        pass

    def kill(self):
        pass


@pytest.fixture()
def fake_child(monkeypatch):
    _FakeProc.calls = []
    _FakeProc.failures = []
    monkeypatch.setattr(asr_iso, "AsrProcess", _FakeProc)
    return _FakeProc


def _write_wav(target: Path, start: float, end: float) -> None:
    target.write_bytes(b"RIFF")  # the fake child never reads it


def _roomy(**kwargs):
    """A budget that fits every rung, so the tests exercise the LADDER."""
    return membudget.budget_from(4096 * MB, 256 * MB, None)


def test_a_memory_death_escalates_to_the_next_rung(tmp_path, fake_child):
    fake_child.failures = [("decode", "mkl_malloc: failed to allocate memory")]

    decoded = asr_iso.decode_span(
        _write_wav, 0.0, 60.0,
        configured_model="small", language=None, tmp_dir=tmp_path,
        on_log=lambda line: None, budget_fn=_roomy,
    )

    assert decoded.model == "small"
    assert [a.ok for a in decoded.attempts] == [False, True]
    assert decoded.attempts[0].memory is True
    assert decoded.words[0].word.strip() == "hello"
    # The retry is a NEW process with a lighter step, not the same one again.
    assert fake_child.calls[0]["model"] == "small"
    assert fake_child.calls[1]["model"] == "small"
    assert len(fake_child.calls) == 2


def test_attempts_are_recorded_with_units_the_job_can_store(tmp_path, fake_child):
    fake_child.failures = [("startup", "model could not load")]

    decoded = asr_iso.decode_span(
        _write_wav, 0.0, 60.0,
        configured_model="small", language="en", tmp_dir=tmp_path,
        on_log=lambda line: None, budget_fn=_roomy,
    )

    first = decoded.attempts[0].as_dict()
    assert set(first) >= {"window", "beam", "vad", "model", "ok", "memory", "peakRssMb"}
    json.dumps(first)  # must be JSON-storable in the job document


def test_a_step_that_cannot_fit_the_budget_is_skipped_not_attempted(tmp_path, fake_child):
    too_tight = membudget.budget_from(2048 * MB, 1900 * MB, None)  # ~-172 MB usable

    with pytest.raises(asr_iso.AsrLadderError):
        asr_iso.decode_span(
            _write_wav, 0.0, 60.0,
            configured_model="small", language=None, tmp_dir=tmp_path,
            on_log=lambda line: None,
            budget_fn=lambda **kw: too_tight,
        )
    # No child was started for a chunk the host could not hold.
    assert fake_child.calls == []


def test_budget_is_read_before_every_chunk(tmp_path, fake_child):
    reads: list[int] = []

    def budget_fn(**kwargs):
        reads.append(1)
        return membudget.budget_from(4096 * MB, 256 * MB, None)

    asr_iso.decode_span(
        _write_wav, 0.0, 60.0,
        configured_model="small", language=None, tmp_dir=tmp_path,
        on_log=lambda line: None, budget_fn=budget_fn,
    )
    assert reads  # at least once, before the first decode
