"""The memory guard: explicit units, and the exact case from the phone screen.

The card said: "Host is low on memory (1992 MB usable, needs ~1000000000 MB to
start)". 1992 MB free must ADMIT a small int8 Whisper job, so that value is a
test, not an anecdote.
"""

from __future__ import annotations

from app import memguard


def test_small_whisper_needs_about_900_mb():
    assert memguard.requirement_mb("small") == 900.0


def test_1992_mb_usable_admits_small_whisper():
    admission = memguard.check("small", avail_bytes=memguard.mb_to_bytes(1992))
    assert admission.ok is True, admission.message
    assert admission.message is None
    assert admission.available_mb == 1992.0
    assert admission.required_mb == 900.0


def test_never_refuses_when_available_equals_required():
    exact = memguard.mb_to_bytes(memguard.requirement_mb("small"))
    assert memguard.check("small", avail_bytes=exact).ok is True


def test_refuses_only_below_the_requirement():
    admission = memguard.check("small", avail_bytes=memguard.mb_to_bytes(200))
    assert admission.ok is False
    # The reason must be in MB and must be the real requirement, never bytes
    # printed as MB (the 1000000000 MB bug).
    assert admission.message is not None
    assert "200 MB usable" in admission.message
    assert "900 MB" in admission.message
    assert "1000000000" not in admission.message


def test_bytes_and_mb_round_trip():
    assert memguard.bytes_to_mb(1024 * 1024 * 512) == 512.0
    assert memguard.mb_to_bytes(900) == 943718400
    assert memguard.bytes_to_mb(memguard.mb_to_bytes(1992)) == 1992.0


def test_unknown_model_uses_small_not_medium():
    assert memguard.requirement_mb("not-a-model") == memguard.requirement_mb("small")
    assert memguard.requirement_mb(None) == 900.0


def test_render_stage_needs_no_model():
    assert memguard.requirement_mb("medium", stage="rendering") == memguard.RENDER_OVERHEAD_MB


def test_unreadable_meminfo_is_not_a_refusal(monkeypatch):
    monkeypatch.setattr(memguard, "available_bytes", lambda: None)
    assert memguard.check("small").ok is True
