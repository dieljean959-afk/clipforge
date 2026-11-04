"""Unit tests for the DeepSeek key-missing failover accelerator.

A missing key is a typo in the key manager, not a quota or model outage, so the
first attempt that fails with a 401/403 is retried on a configured, ready
backup BEFORE the ladder is spent. A missing primary key is STILL a hard,
loud misconfiguration and is never silently redirected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Allow the suite to run from the project root or a checked-out tools dir.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(APP_ROOT))

from app import llm as llm_mod  # noqa: E402
from app.config import load_config  # noqa: E402
from tools._failover_helpers import (
    _candidate,
    _keys,
    _reply,
    _settings,
    _Stub,
)  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))
        raise AssertionError(name)


def test_failover_accelerates_missing_key_to_ready_backup() -> None:
    """A 401/403 from the absent key is retried on the configured backup."""

    def _deepseek_handler(method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object], dict[str, str]]:
        if path.endswith("chat/completions"):
            return (401, {"error": {"message": "Invalid API key"}}, {"Content-Type": "application/json"})
        raise AssertionError(f"unexpected path {path!r}")

    def _groq_handler(method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object], dict[str, str]]:
        if path.endswith("chat/completions"):
            return (200, _reply([_candidate()]), {"Content-Type": "application/json"})
        raise AssertionError(f"unexpected path {path!r}")

    s = _settings()
    with _Stub(_deepseek_handler) as ds_stub, _Stub(_groq_handler) as groq_stub:
        s.deepseek.base_url = ds_stub.base_url
        s.groq.base_url = groq_stub.base_url
        with _keys(None, "gsk_test_0123456789"):
            text = llm_mod.call_llm("transcript text", s)
    check("the backup actually finished the chunk", "clips" in text, text[:80])
    check("the primary was called once for its 401", len(ds_stub.requests) == 1,
          str(len(ds_stub.requests)))
    check("the backup made the call", len(groq_stub.requests) == 1,
          str(len(groq_stub.requests)))


def test_missing_primary_key_is_a_hard_error_and_never_fails_over() -> None:
    """LlmNotConfigured stands; a ready backup is never silently used."""

    def _handler(method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object], dict[str, str]]:
        raise AssertionError("backup should never be called")

    s = _settings()
    with _Stub(_handler) as stub, _Stub(_groq_handler) as groq_stub:
        s.deepseek.base_url = stub.base_url
        s.groq.base_url = groq_stub.base_url
        with _keys(None, "gsk_test_0123456789"):
            err = None
            try:
                llm_mod.call_llm("transcript text", s)
            except llm_mod.LlmNotConfigured as exc:
                err = str(exc)
    check("the error names DEEPSEEK_API_KEY", err is not None and "DEEPSEEK_API_KEY" in err, str(err))
    check("...and points at the Keys tab", err is not None and "Keys" in err, str(err))
    check("...and the backup was never called", len(stub.requests) == 0, str(len(stub.requests)))


def test_nothing_fails_over_when_the_primary_key_is_present() -> None:
    """A valid primary key takes the normal ladder; the accelerator is skipped."""

    seen: list[dict[str, object]] = []

    def _handler(method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object], dict[str, str]]:
        seen.append({"method": method, "path": path, "body": body.decode("utf-8")})
        return (200, _reply([_candidate()]), {"Content-Type": "application/json"})

    s = _settings()
    with _Stub(_handler) as stub:
        s.deepseek.base_url = stub.base_url
        s.groq.base_url = stub.base_url
        with _keys("sk-test-0123456789", "gsk_test_0123456789"):
            text = llm_mod.call_llm("transcript text", s)
    check("the primary did the whole chunk", "clips" in text, text[:80])
    check("only one request was made (accelerator skipped)", len(seen) == 1, str(len(seen)))
    check("the request went to the deepseek path", "deepseek" in str(seen[0]),
          str(seen[0]))
