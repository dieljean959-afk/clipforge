"""Checkpoint resume and stage truth.

Two guarantees from the memory work that are easy to lose in a refactor:

* a job that already has transcribed chunks NEVER decodes them again — a
  restart resumes at the next unfinished chunk;
* a transcription failure is labelled "Transcribing", not "AI clip detection",
  and a memory failure shows a plain sentence with the traceback kept aside.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app import asr_iso, jobstatus, subtitle
from app.config import load_config

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tools" / "fixtures" / "speech-jfk-11s.wav"


@pytest.fixture()
def settings(tmp_path):
    base = load_config(quiet=True)
    paths = replace(base.app.paths, data_dir=tmp_path, downloads_dir=tmp_path / "downloads")
    return replace(base, app=replace(base.app, paths=paths))


def test_a_checkpointed_chunk_is_never_decoded_again(settings, tmp_path, monkeypatch):
    cache = tmp_path / "chunks"
    cache.mkdir()
    (cache / "chunk_00000.json").write_text(json.dumps({
        "w_start": 0.0,
        "w_end": 11.0,
        "words": [
            {"start": 0.2, "end": 0.6, "word": "cached"},
            {"start": 0.7, "end": 1.1, "word": "words"},
        ],
    }), encoding="utf-8")

    def explode(*args, **kwargs):
        raise AssertionError("decode must not run for a checkpointed chunk")

    monkeypatch.setattr(asr_iso, "decode_span", explode)

    transcript = subtitle.transcribe(FIXTURE, settings, cache_dir=cache)

    assert [w.word for w in transcript.words] == ["cached", "words"]
    assert transcript.duration > 0


def test_a_memory_failure_is_labelled_transcribing(settings, tmp_path, monkeypatch):
    def out_of_memory(*args, **kwargs):
        raise asr_iso.AsrLadderError(
            "could not transcribe 0-11s; last error: mkl_malloc: failed to allocate memory",
            attempts=[],
        )

    monkeypatch.setattr(asr_iso, "decode_span", out_of_memory)

    with pytest.raises(subtitle.TranscribeError) as exc:
        subtitle.transcribe(FIXTURE, settings, cache_dir=tmp_path / "c2")

    assert "ran out of memory" in str(exc.value).lower()
    assert "mkl_malloc" not in str(exc.value).split("; tried")[0]


def test_stage_names_never_mix_detection_with_transcription():
    assert jobstatus.plan_stage("transcribe") == "transcribing"
    failed = jobstatus.describe({
        "_id": "j", "stage": "failed", "failedStage": "transcribe",
        "error": "Ran out of memory while transcribing.", "progress": 42,
    })
    assert failed["stageLabel"] == "Transcribing"
    assert failed["progress"] == 0


def test_plain_reason_keeps_the_traceback_out_of_the_headline():
    plain = asr_iso.plain_memory_reason(2, 6)
    assert "Ran out of memory" in plain
    assert "attempt 2 of 6" in plain
    assert "Traceback" not in plain and "mkl_malloc" not in plain


def test_a_memory_failure_on_a_job_is_plain_and_keeps_the_traceback(settings, tmp_path):
    from app.store import LocalStore
    from app.task import Pipeline

    store = LocalStore(settings.app.paths)
    store.write_job({"_id": "job_mem", "stage": "transcribing", "createdAt": 0})
    pipeline = Pipeline(settings, store)

    try:
        raise RuntimeError("model small could not load: mkl_malloc: failed to allocate memory")
    except RuntimeError:
        info = __import__("sys").exc_info()
    message, detail = pipeline._fail_args_for(info)

    assert message == asr_iso.plain_memory_reason()
    assert "mkl_malloc" not in message
    assert "RuntimeError" in detail and "mkl_malloc" in detail
