"""Every exit row of the classifier, in both code conventions."""

from __future__ import annotations

import signal

import pytest

from app import exits


@pytest.mark.parametrize("code", [-15, 143])
def test_sigterm_is_one_external_interruption(code):
    v = exits.classify(code, stage="transcribing")
    assert v.kind == exits.EXTERNAL_INTERRUPTION
    assert v.action == exits.AUTO_RESUME
    assert "SIGTERM" in v.message and "transcribing" in v.message
    assert "not a crash" in v.message


@pytest.mark.parametrize("code", [-9, 137])
def test_sigkill_checks_the_oom_record(code):
    oom = exits.classify(code, stage="transcribing", oom=True)
    assert oom.kind == exits.RESOURCE and oom.action == exits.RETRY_LIGHTER
    assert "out-of-memory" in oom.message

    plain = exits.classify(code, stage="transcribing", oom=False)
    assert plain.kind == exits.EXTERNAL_KILL and plain.action == exits.AUTO_RESUME
    assert "SIGKILL" in plain.message


@pytest.mark.parametrize(
    "code,signame",
    [(-11, "SIGSEGV"), (139, "SIGSEGV"), (-6, "SIGABRT"), (134, "SIGABRT"),
     (-7, "SIGBUS"), (135, "SIGBUS")],
)
def test_native_crashes_retry_lighter(code, signame):
    v = exits.classify(code, stage="transcribing")
    assert v.kind == exits.NATIVE_CRASH
    assert v.action == exits.RETRY_LIGHTER
    assert signame in v.message


@pytest.mark.parametrize("code", [-2, 130])
def test_sigint_is_an_interruption(code):
    v = exits.classify(code)
    assert v.kind == exits.INTERRUPTED and v.action == exits.AUTO_RESUME


def test_zero_before_done_is_incomplete_not_success():
    v = exits.classify(0, stage="exporting")
    assert v.kind == exits.INCOMPLETE_EXIT and v.resumable


def test_code_one_with_a_traceback_is_an_app_error():
    v = exits.classify(1, stage="finding", traceback_seen=True)
    assert v.kind == exits.APP_ERROR and v.action == exits.RETRY_STAGE


def test_timeouts_retry_the_stage_after_dumping_the_stack():
    # 124 is GNU timeout's exit code; it is not a signal number, so this is the
    # one row that must NOT be normalised through the 128+N rule.
    code = 124
    v = exits.classify(code, stage="finding")
    assert v.kind == exits.STAGE_TIMEOUT
    assert v.action == exits.RETRY_STAGE
    assert "stack" in v.hint.lower()


@pytest.mark.parametrize("code", [126, 127])
def test_missing_binary_fails_fast(code):
    v = exits.classify(code, stage="exporting")
    assert v.kind == exits.MISSING_BINARY and v.action == exits.FAIL_FAST
    assert not v.resumable


def test_a_named_missing_tool_beats_the_code():
    v = exits.classify(1, stage="exporting", missing="ffmpeg")
    assert v.kind == exits.MISSING_BINARY
    assert "ffmpeg" in v.message and "ffmpeg" in v.hint


def test_an_unheard_of_code_is_unknown_and_still_resumed():
    v = exits.classify(42, stage="fetch")
    assert v.kind == exits.UNKNOWN
    assert v.action == exits.CAPTURE_AND_RESUME and v.resumable


def test_an_unheard_of_signal_is_unknown_and_still_resumed():
    v = exits.classify(-40)
    assert v.kind == exits.UNKNOWN and v.action == exits.CAPTURE_AND_RESUME


def test_the_label_names_the_signal():
    assert exits.code_label(-15).endswith("(SIGTERM)")
    assert exits.code_label(143).endswith("(SIGTERM)")
    assert exits.code_label(0) == "0"


def test_both_conventions_normalise_to_the_same_signal():
    for sig in (signal.SIGTERM, signal.SIGKILL, signal.SIGSEGV, signal.SIGINT):
        assert exits.signum(-int(sig)) == int(sig) == exits.signum(128 + int(sig))
