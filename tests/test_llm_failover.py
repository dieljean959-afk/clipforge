"""The DeepSeek failover accelerator in `_call_openai_chat`.

A DeepSeek key that is missing or unusable is a key-manager mistake, not a
model outage. Rather than spend the whole DeepSeek ladder finding that out, the
engine routes the chunk straight to the configured backup — and raises one
clear `LlmNotConfigured` when there is no usable backup either.

The key resolver reads the workspace's env files, so these tests patch
`openai_api_key` rather than trying to unset a real saved key in `os.environ`.

    python -m pytest tests/test_llm_failover.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import llm as llm_mod  # noqa: E402
from tools._failover_helpers import (  # noqa: E402
    _candidate,
    _ok,
    _reply,
    _settings,
    _Stub,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f" — {detail}"))
    assert ok, name


def _keys_by_provider(monkeypatch, mapping: dict[str, str | None]) -> None:
    """Answer `openai_api_key(which, settings)` from a fixed table."""
    monkeypatch.setattr(llm_mod, "openai_api_key", lambda which, _s: mapping.get(which))


def test_unusable_primary_is_served_by_the_ready_backup(monkeypatch) -> None:
    """No DeepSeek key, a ready Groq backup: the backup answers the chunk."""
    _keys_by_provider(monkeypatch, {"deepseek": None, "groq": "gsk_test_0123456789"})
    with _Stub(_ok(_reply([_candidate()]))) as backup:
        s = _settings()
        s.groq.base_url = backup.base_url
        s.llm.fallback_provider = "groq"
        text = llm_mod.call_llm("transcript text", s)
    check("the backup produced the clips", "clips" in text, text[:80])
    check("the backup was actually called once", len(backup.requests) == 1,
          str(len(backup.requests)))
    check("...with the Groq key, not a DeepSeek one",
          "gsk_test" in str(backup.requests[0].get("__auth")),
          str(backup.requests[0].get("__auth"))[:40])


def test_no_usable_provider_names_the_key_and_the_tab(monkeypatch) -> None:
    """No DeepSeek key and no ready backup: one loud, actionable error."""
    _keys_by_provider(monkeypatch, {"deepseek": None, "groq": None})
    s = _settings()
    s.llm.fallback_provider = ""
    err = None
    try:
        llm_mod.call_llm("transcript text", s)
    except llm_mod.LlmNotConfigured as exc:
        err = str(exc)
    check("it raises LlmNotConfigured", err is not None, "no exception")
    check("the message names DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY" in (err or ""), str(err))
    check("the message points at the Keys tab", "Keys" in (err or ""), str(err))


def test_a_healthy_primary_takes_the_whole_chunk(monkeypatch) -> None:
    """A working DeepSeek key must not be routed around."""
    _keys_by_provider(monkeypatch, {"deepseek": "sk-test-0123456789", "groq": None})
    with _Stub(_ok(_reply([_candidate()]))) as primary:
        s = _settings()
        s.deepseek.base_url = primary.base_url
        s.llm.fallback_provider = "groq"
        text = llm_mod.call_llm("transcript text", s)
    check("the primary produced the clips", "clips" in text, text[:80])
    check("exactly one request — no fallback was touched", len(primary.requests) == 1,
          str(len(primary.requests)))
    check("the request carried the DeepSeek key",
          "sk-test" in str(primary.requests[0].get("__auth")),
          str(primary.requests[0].get("__auth"))[:40])
