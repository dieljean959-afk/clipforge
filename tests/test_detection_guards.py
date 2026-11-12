"""Guards added for the "stuck at 99%, never detects, never renders" loop.

Three separate behaviours, each one a way the job could burn minutes (or the
whole batch) without telling anyone why:

1. Detection windows: wide enough that a 2-hour source is a handful of calls.
2. A provider that answers "Insufficient Balance" is asked ONCE, not once per
   chunk — a 402 is not going to change between chunks 2 and 3.
3. The per-stage watchdog measures PROGRESS, not the liveness heartbeat, so a
   stage that is hung inside an HTTP call still trips its own budget.

    python -m pytest tests/test_detection_guards.py -q
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import llm as llm_mod  # noqa: E402
from app import worker as worker_mod  # noqa: E402
from app.config import GeminiConfig  # noqa: E402
from tools._failover_helpers import _settings, _Stub  # noqa: E402


def _transcript(seconds: float, step: float = 0.35) -> list[dict]:
    words: list[dict] = []
    t = 0.0
    while t < seconds:
        words.append({"start": round(t, 2), "end": round(t + step, 2), "word": f"w{len(words)}"})
        t += step
    return words


# --- 1. windowing ---------------------------------------------------------


def test_a_twenty_minute_source_is_three_windows_not_ten():
    words = _transcript(1200)
    chunks = llm_mod.chunk_words(words)
    assert len(chunks) == 3, f"got {len(chunks)} calls for 20 min"


# --- 1b. an explicit clip count is a quota ------------------------------

@pytest.mark.parametrize(
    "seconds,target",
    [
        (300.0, 5), (300.0, 8), (300.0, 10), (300.0, 15),
        (1237.0, 5), (1237.0, 8), (1237.0, 10), (1237.0, 15), (1237.0, 20),
    ],
)
def test_an_explicit_clip_count_is_always_reached(target, seconds):
    """Asking for N clips must produce N clips, not "the N best".

    Before this, a 20-minute source whose provider returned two candidates
    above the score floor produced a two-clip batch — the reader's number was
    silently reinterpreted as a ceiling. The fill pass must reach the target
    from the model's below-floor picks and transcript density, never overlap,
    and never exceed the request.
    """
    words = _transcript(seconds)
    cfg = GeminiConfig(
        model="test", min_clip_sec=15, max_clip_sec=60,
        max_clips=target, max_retries=1,
    )
    selected = [{"start": 20.0, "end": 50.0, "score": 88.0}]
    candidates = [
        selected[0],
        {"start": 120.0, "end": 150.0, "score": 55.0},
    ]
    filled, stats = llm_mod._fill_to_target(selected, candidates, words, seconds, cfg)
    chosen = [*selected, *filled]

    assert len(chosen) == target, f"{seconds}s source, asked {target}, got {len(chosen)}"
    assert len(chosen) <= target, "fill must never exceed the request"

    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            assert not (
                a["start"] < b["end"] - llm_mod.OVERLAP_TOLERANCE_SEC
                and b["start"] < a["end"] - llm_mod.OVERLAP_TOLERANCE_SEC
            ), f"two chosen clips overlap: {a} / {b}"

    assert stats["below"] + stats["heuristic"] == len(filled)


def test_a_source_too_short_for_the_count_returns_what_fits_without_overlap():
    """20 x 15 s cannot fit in 300 s alongside a selected 30 s clip."""
    words = _transcript(300)
    cfg = GeminiConfig(
        model="test", min_clip_sec=15, max_clip_sec=60,
        max_clips=20, max_retries=1,
    )
    selected = [{"start": 20.0, "end": 50.0, "score": 88.0}]
    filled, _ = llm_mod._fill_to_target(selected, list(selected), words, 300.0, cfg)
    chosen = [*selected, *filled]
    assert len(chosen) < 20, "this source cannot hold 20 separate moments"
    assert len(chosen) >= 15, "but it must fill as many as genuinely fit"
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            assert not (
                a["start"] < b["end"] - llm_mod.OVERLAP_TOLERANCE_SEC
                and b["start"] < a["end"] - llm_mod.OVERLAP_TOLERANCE_SEC
            )


def test_filled_clips_are_marked_lower_confidence():
    words = _transcript(1200)
    cfg = GeminiConfig(
        model="test", min_clip_sec=15, max_clip_sec=60,
        max_clips=8, max_retries=1,
    )
    selected = [{"start": 20.0, "end": 50.0, "score": 88.0}]
    filled, _ = llm_mod._fill_to_target(selected, list(selected), words, 1200.0, cfg)
    assert filled, "expected the fill pass to add clips"
    assert all(c.get("belowFloor") is True for c in filled)
    assert all("transcript_heuristic" in str(c.get("detection_source", "")) or c.get("belowFloor") for c in filled)


def test_fill_does_nothing_when_the_target_is_already_met():
    words = _transcript(1200)
    cfg = GeminiConfig(
        model="test", min_clip_sec=15, max_clip_sec=60,
        max_clips=3, max_retries=1,
    )
    selected = [
        {"start": 10.0, "end": 40.0, "score": 90.0},
        {"start": 100.0, "end": 130.0, "score": 85.0},
        {"start": 200.0, "end": 230.0, "score": 80.0},
    ]
    filled, stats = llm_mod._fill_to_target(selected, list(selected), words, 1200.0, cfg)
    assert filled == []
    assert stats == {"below": 0, "heuristic": 0}


def test_windows_overlap_and_cover_every_word():
    words = _transcript(1800)
    chunks = llm_mod.chunk_words(words)
    covered = 0
    prev_end = -1.0
    for start_index, chunk in chunks:
        assert chunk, "empty window"
        # The next window begins inside the previous one (30s of shared tail),
        # so a highlight straddling the seam is still seen whole.
        assert start_index <= covered, "a word would be skipped between windows"
        assert float(chunk[0]["start"]) >= prev_end - 30.5
        covered = max(covered, start_index + len(chunk))
        prev_end = float(chunk[-1]["end"])
    assert covered == len(words), "the tail of the transcript was dropped"


def test_neighbouring_windows_share_about_thirty_seconds():
    words = _transcript(1800)
    chunks = llm_mod.chunk_words(words)
    first, second = chunks[0][1], chunks[1][1]
    shared = float(first[-1]["end"]) - float(second[0]["start"])
    assert 25.0 <= shared <= 35.0, shared


def test_window_length_stays_near_the_six_minute_target():
    words = _transcript(1200)
    for _start, chunk in llm_mod.chunk_words(words):
        span = float(chunk[-1]["end"]) - float(chunk[0]["start"])
        assert span <= 460.0, f"window of {span:.0f}s is over the 420s+30s overlap target"


def test_a_long_source_never_exceeds_the_chunk_cap():
    chunks = llm_mod.chunk_words(_transcript(2 * 60 * 60))
    assert 0 < len(chunks) <= llm_mod.MAX_DETECTION_CHUNKS


# --- 2. an out-of-credit provider is asked once ---------------------------


def _keys(monkeypatch, mapping: dict[str, str | None]) -> None:
    monkeypatch.setattr(llm_mod, "openai_api_key", lambda which, _s: mapping.get(which))


def test_insufficient_balance_is_asked_once_for_the_whole_run(monkeypatch):
    """Three chunks, one HTTP request: the dead-provider cache does its job."""
    monkeypatch.setattr(llm_mod, "_DEAD_PROVIDERS", {})
    _keys(monkeypatch, {"deepseek": "sk-test-0123456789", "groq": None})
    s = _settings()
    s.llm.fallback_provider = ""
    body = {"error": {"message": "Insufficient Balance", "type": "insufficient_balance_error"}}
    with _Stub(lambda _b: (402, body)) as deepseek:
        s.deepseek.base_url = deepseek.base_url
        errors = []
        for _ in range(3):
            try:
                llm_mod.call_llm("transcript text", s)
            except Exception as exc:  # noqa: BLE001 - the error text is the assertion
                errors.append(str(exc))
        calls = len(deepseek.requests)
    assert calls == 1, f"the 402 endpoint was asked {calls} times"
    assert len(errors) == 3, "a chunk was reported as succeeding"
    assert "credit" in errors[-1].lower() or "billing" in errors[-1].lower(), errors[-1][:160]


def test_a_dead_provider_is_skipped_before_the_http_call(monkeypatch):
    """The cache alone must answer, with no request at all."""
    monkeypatch.setattr(llm_mod, "_DEAD_PROVIDERS", {})
    _keys(monkeypatch, {"deepseek": "sk-test-0123456789", "groq": None})
    s = _settings()
    s.llm.fallback_provider = ""
    body = {"error": {"message": "Insufficient Balance", "type": "insufficient_balance_error"}}
    with _Stub(lambda _b: (402, body)) as deepseek:
        s.deepseek.base_url = deepseek.base_url
        try:
            llm_mod.call_llm("transcript text", s)
        except Exception:  # noqa: BLE001 - expected
            pass
        first = len(deepseek.requests)
        for _ in range(2):
            try:
                llm_mod.call_llm("transcript text", s)
            except Exception:  # noqa: BLE001 - expected
                pass
        assert len(deepseek.requests) == first == 1


def test_the_reason_names_the_provider_and_is_not_cached_forever():
    llm_mod._DEAD_PROVIDERS.clear()
    llm_mod._mark_provider_dead("deepseek", "Insufficient Balance")
    assert llm_mod.provider_dead("deepseek"), "a 402 must mark the backend dead"
    assert not llm_mod.provider_dead("groq"), "an unrelated backend was blamed"
    llm_mod._DEAD_PROVIDERS.clear()
    assert not llm_mod.provider_dead("deepseek"), "the mark must expire / be clearable"


# --- 3. the watchdog measures progress, not the heartbeat -----------------


def test_a_budgeted_stage_ignores_a_healthy_heartbeat():
    job = {
        "stage": "finding",
        "lastProgressAt": 9_000,       # heartbeat: fresh
        "stageProgressAt": 1_000,      # this stage: stale
        "stageStartedAt": 500,
    }
    assert worker_mod.stall_reference_ms(job) == 1_000


def test_a_budgeted_stage_falls_back_to_its_start_time():
    job = {"stage": "captions", "lastProgressAt": 9_000, "stageStartedAt": 500}
    assert worker_mod.stall_reference_ms(job) == 500


def test_a_long_quiet_stage_keeps_the_heartbeat_window():
    """transcribing is legitimately silent for minutes — heartbeat must count."""
    job = {"stage": "transcribing", "lastProgressAt": 9_000, "stageProgressAt": 1_000}
    assert worker_mod.stall_reference_ms(job) == 9_000


def test_every_budgeted_stage_has_its_own_budget():
    for stage, seconds in worker_mod.STAGE_BUDGET_SEC.items():
        assert 60 <= seconds <= 600, f"{stage} budget {seconds}s is outside the brief"
    assert worker_mod.STAGE_BUDGET_SEC["finding"] <= 120


def test_an_oversized_prompt_is_not_re_asked_every_chunk(monkeypatch):
    """Groq's 8k TPM cannot take a 6-minute window: fail over ONCE, not per clip.

    Each doomed call paid a 30-second pacing wait first, so a 15-clip batch spent
    seven minutes waiting to be rejected.
    """
    monkeypatch.setattr(llm_mod, "_DEAD_PROVIDERS", {})
    _keys(monkeypatch, {"deepseek": "sk-test-0123456789", "groq": "gsk_test_0123456789"})
    s = _settings()
    s.llm.fallback_provider = "groq"
    too_big = {
        "error": {
            "message": "Request too large for model openai/gpt-oss-120b on tokens per "
                       "minute (TPM): Limit 8000, Requested 10410",
            "type": "invalid_request_error",
        }
    }
    with _Stub(lambda _b: (400, too_big)) as groq:
        s.groq.base_url = groq.base_url
        for _ in range(3):
            try:
                llm_mod.call_llm("transcript text", s)
            except Exception:  # noqa: BLE001 - expected while over budget
                pass
        calls = len(groq.requests)
    assert calls == 1, f"the oversized prompt was sent {calls} times"
    assert "prompts this large" in llm_mod.provider_dead("groq")


def test_billing_and_oversize_are_reported_differently():
    assert llm_mod._unusable_reason("Insufficient Balance") .startswith("out of credit")
    assert llm_mod._unusable_reason(
        "Request too large ... tokens per minute (TPM): Limit 8000"
    ).startswith("cannot accept prompts this large")
    assert llm_mod._unusable_reason("502 Bad Gateway") == ""
