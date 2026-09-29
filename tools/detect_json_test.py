"""Regression tests for the `json_validate_failed` job death (BUG 1).

The bug: `llm.detect_clips` -> `call_groq` raised on a 400 `json_validate_failed`,
killing the whole job at chunk 3/7 and discarding every chunk already paid for
and cached. The fix is four things, each tested here:

  1. the wire contract asks for something the provider can actually produce
     (strict json_schema with a top-level object), not a bare array;
  2. a JSON-validation 400 escalates a ladder (lower temperature, higher
     max_tokens, then no response_format with local repair) instead of failing;
  3. a 400 is only PERMANENT when it is not a JSON-validation error;
  4. one unfixable chunk is skipped and logged — the job continues on the rest,
     and the cache written for the good chunks survives.

Runs entirely against a local stub. No network, no quota.

    python tools/detect_json_test.py
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm as llm_mod  # noqa: E402
from app.config import load_config  # noqa: E402

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


# --------------------------------------------------------------------------- #
# stub server
# --------------------------------------------------------------------------- #

class _Stub:
    """Scripted OpenAI-compatible endpoint. `handler` sees each request body."""

    def __init__(self, handler: Callable[[dict], tuple[int, dict]]) -> None:
        self.requests: list[dict] = []
        self._handler = handler
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(body)
                status, payload = outer._handler(body)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a: Any) -> None:
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}/openai/v1"

    def __enter__(self) -> "_Stub":
        return self

    def __exit__(self, *a: Any) -> None:
        self.srv.shutdown()
        self.srv.server_close()

    def bodies(self) -> list[dict]:
        return self.requests

    def modes(self) -> list[str | None]:
        out: list[str | None] = []
        for b in self.requests:
            rf = b.get("response_format")
            out.append(rf.get("type") if rf else None)
        return out


def _candidates(n: int = 2, start: int = 10) -> str:
    return json.dumps([
        {
            "start_word": start + i * 60,
            "end_word": start + i * 60 + 40,
            "score": 80 - i,
            "title": f"Clip {i}",
            "hook_text": "a hook",
            "hashtags": ["#fyp", "#foryou", "#viral", "#topic"],
        }
        for i in range(n)
    ])


def _candidates_for_prompt(body: dict, n: int = 2) -> str:
    """Candidates whose indices fall INSIDE this chunk's word range.

    Word indices are global, so fixed indices from a canned reply are out of
    range for every chunk but the first — `resolve_candidates` clamps them to a
    zero-length span and drops the clip. Reading the real range out of the
    prompt makes the stub behave like a model that read its own transcript.
    """
    prompt = (body.get("messages") or [{}])[-1].get("content", "")
    idx = [int(m) for m in re.findall(r"^\[(\d+)\]", prompt, re.M)]
    if not idx:
        return _candidates(n, 10)
    lo, hi = idx[0], idx[-1]
    span = max(2, (hi - lo) // (n + 1))
    return json.dumps([
        {
            "start_word": lo + i * span,
            "end_word": lo + i * span + max(1, span - 5),
            "score": 80 - i,
            "title": f"Clip {i}",
            "hook_text": "a hook",
            "hashtags": ["#fyp", "#foryou", "#viral", "#topic"],
        }
        for i in range(n)
    ])


def _reply(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 100},
    }


def _json400() -> tuple[int, dict]:
    return 400, {"error": {"message": "Failed to validate JSON. Please adjust your prompt.",
                           "type": "invalid_request_error",
                           "code": "json_validate_failed"}}


def _settings(base: str, **kw: Any):
    s = load_config(quiet=True)
    s.groq = replace(
        s.groq, base_url=base, min_interval_sec=0.0, tokens_per_min=0, **kw
    )
    return s


#: Key injected into `Settings.groq_api_key` for the whole run. The property
#: re-resolves from .env on every access, so a stub value has to be installed at
#: the CLASS level — the same trick tools/llm_provider_test.py uses. Set once
#: at startup and restored on exit, so no test can leak a patch into another.
_KEY: str | None = "gsk_test_key_0123456789"
_ORIG_KEY_PROP = None

#: Distinguishes "argument omitted" (use the stub key) from "no key at all"
#: (the missing-key case), which a bare None cannot express.
NO_KEY = "\x00no-key"


def _install_key() -> None:
    global _ORIG_KEY_PROP
    _ORIG_KEY_PROP = type(load_config(quiet=True)).groq_api_key
    type(load_config(quiet=True)).groq_api_key = property(  # type: ignore[assignment]
        lambda self: _KEY
    )


def _uninstall_key() -> None:
    if _ORIG_KEY_PROP is not None:
        type(load_config(quiet=True)).groq_api_key = _ORIG_KEY_PROP  # type: ignore[assignment]


def _with_key(s: Any, key: str = "gsk_test_key_0123456789") -> Any:
    """Set the process-wide injected key and return the settings unchanged.

    Call sites read `_with_key(settings)` then use `s`; the key applies to every
    Settings instance until the next `_with_key` call, which is the granularity
    these tests need (one stub per test). Pass `NO_KEY` to exercise the
    missing-key path.
    """
    global _KEY
    _KEY = None if key == NO_KEY else key
    return s


def _words(n: int = 400) -> list[dict]:
    return [{"word": f"w{i}", "start": i * 0.25, "end": i * 0.25 + 0.2} for i in range(n)]


# --------------------------------------------------------------------------- #
# 1. the wire contract
# --------------------------------------------------------------------------- #

def test_schema_contract() -> None:
    print("\n[1] wire contract — the shape the provider can actually produce")
    with _Stub(lambda b: (200, _reply(_candidates()))) as stub:
        s = _with_key(_settings(stub.base), "gsk_test_key_0123456789")
        llm_mod.call_groq("transcript text", s)
        body = stub.bodies()[0]
        rf = body.get("response_format") or {}
        check("default mode is json_schema", rf.get("type") == "json_schema", str(rf))
        check("strict decoding is requested",
              (rf.get("json_schema") or {}).get("strict") is True, str(rf))
        schema = (rf.get("json_schema") or {}).get("schema") or {}
        check("top level is an object, not the array the old prompt asked for",
              schema.get("type") == "object", str(schema.get("type")))
        check("clips is the object property", schema.get("required") == ["clips"],
              str(schema.get("required")))
        item = (schema.get("properties", {}).get("clips", {}).get("items") or {})
        check("every clip field is required under strict mode",
              set(item.get("required") or []) >=
              {"start_word", "end_word", "score", "title"}, str(item.get("required")))
        check("clips forbid extra properties (strict requirement)",
              item.get("additionalProperties") is False, str(item.get("additionalProperties")))

    # The prompt must not contradict the envelope the schema enforces.
    s = load_config(quiet=True)
    aligned = llm_mod._groq_envelope_prompt("Return ONLY a JSON array (no prose, no markdown fences):\n[]", "json_schema")
    check("prompt no longer demands a bare array under json_schema",
          "JSON array" not in aligned and "clips" in aligned, aligned[:120])
    plain = llm_mod._groq_envelope_prompt("Return ONLY a JSON array:", "plain")
    check("plain mode keeps the legacy wording", "JSON array" in plain, plain[:80])


# --------------------------------------------------------------------------- #
# 2. the retry ladder
# --------------------------------------------------------------------------- #

def test_json_400_escalates() -> None:
    print("\n[2] JSON-validation 400 escalates the ladder instead of failing")
    calls = {"n": 0}

    def handler(body: dict) -> tuple[int, dict]:
        calls["n"] += 1
        if calls["n"] <= 2:
            return _json400()
        return 200, _reply(_candidates())

    with _Stub(handler) as stub:
        s = _with_key(_settings(stub.base), "gsk_test_key_0123456789")
        text = llm_mod.call_groq("transcript", s)
    check("recovers after two JSON failures", text.startswith("["), text[:40])
    check("made 3 calls", len(stub.bodies()) == 3, str(len(stub.bodies())))

    bodies = stub.bodies()
    mt = [b.get("max_tokens") for b in bodies]
    temps = [b.get("temperature") for b in bodies]
    check("max_tokens grows on each retry", mt[0] < mt[1] < mt[2], str(mt))
    check("max_tokens is capped by max_tokens_ceiling",
          max(mt) <= s.groq.max_tokens_ceiling, str(mt))
    check("temperature never increases on retry",
          all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1)), str(temps))


def test_ladder_drops_response_format() -> None:
    print("\n[2b] the last ladder rung drops response_format and repairs locally")
    seen: list[str | None] = []

    def handler(body: dict) -> tuple[int, dict]:
        rf = body.get("response_format")
        seen.append(rf.get("type") if rf else None)
        # Everything with a format is rejected; the plain one is fenced prose.
        if rf is not None:
            return _json400()
        return 200, _reply("Here you go:\n\n```json\n" + _candidates() + "\n```")

    with _Stub(handler) as stub:
        s = _with_key(_settings(stub.base), "gsk_test_key_0123456789")
        text = llm_mod.call_groq("transcript", s)
    check("eventually calls with no response_format", None in seen, str(seen))
    check("the plain rung is last", seen[-1] is None, str(seen))
    check("fenced reply still parses afterwards", len(llm_mod.parse_candidates(text)) == 2,
          str(llm_mod.parse_candidates(text)))


def test_model_fallback_chain() -> None:
    print("\n[2c] the model chain is tried in order when a model cannot do JSON")
    models: list[str] = []

    def handler(body: dict) -> tuple[int, dict]:
        m = body.get("model", "")
        models.append(m)
        if m == "openai/gpt-oss-120b":
            return _json400()
        return 200, _reply(_candidates())

    with _Stub(handler) as stub:
        s = _with_key(
            _settings(stub.base, json_retries=1),
            "gsk_test_key_0123456789",
        )
        text = llm_mod.call_groq("transcript", s)
    check("recovered on a fallback model", text.startswith("["))
    check("primary tried first", models and models[0] == "openai/gpt-oss-120b", str(models[:1]))
    check("a different model was eventually used",
          models[-1] != "openai/gpt-oss-120b", str(models[-1:]))
    check("did not re-ask the primary after moving on",
          models[-1] != "openai/gpt-oss-120b" or models.count("openai/gpt-oss-120b") <= 2,
          str(models))


def test_whole_chain_exhausted_raises() -> None:
    print("\n[2d] when every model fails, the error is specific")
    with _Stub(lambda b: _json400()) as stub:
        s = _with_key(_settings(stub.base, json_retries=1), "gsk_test_key_0123456789")
        err = None
        try:
            llm_mod.call_groq("transcript", s)
        except llm_mod.LlmError as exc:
            err = str(exc)
    check("raises LlmError", err is not None, str(err))
    check("names the chain", "detection chain" in (err or ""), str(err))
    check("says it tried JSON recovery", "valid JSON" in (err or ""), str(err))
    check("is bounded (does not hang)", len(stub.bodies()) <= 30, str(len(stub.bodies())))


# --------------------------------------------------------------------------- #
# 3. which 400s are permanent
# --------------------------------------------------------------------------- #

def test_permanent_400_fails_fast() -> None:
    print("\n[3] a genuinely permanent 400 still fails fast (no retry burn)")
    with _Stub(lambda b: (400, {"error": {"message": "messages: field required"}})) as stub:
        s = _with_key(_settings(stub.base), "gsk_test_key_0123456789")
        err = None
        try:
            llm_mod.call_groq("transcript", s)
        except llm_mod.LlmError as exc:
            err = str(exc)
    check("raises on a permanent 400", err is not None, str(err))
    check("says 'permanently'", "permanent" in (err or "").lower(), str(err))
    check("does NOT walk the whole ladder on a bad request",
          len(stub.bodies()) == 1, str(len(stub.bodies())))


def test_json_classifier() -> None:
    print("\n[3b] the JSON-validation classifier")
    for body, want in [
        ('{"error":{"code":"json_validate_failed","message":"Failed to validate JSON."}}', True),
        ("Failed to generate JSON. Please adjust your prompt.", True),
        ("Generated JSON does not match the expected schema", True),
        ('{"error":{"message":"messages: field required"}}', False),
        ('{"error":{"message":"response_format is not supported for this model"}}', False),
        ('{"error":{"message":"model has been decommissioned"}}', False),
        ("", False),
    ]:
        check(f"classifies {body[:44]!r}", llm_mod._is_json_validation_error(body) is want)

    for body, want in [
        ("response_format is not supported for this model", True),
        ("This model does not support json_schema", True),
        ("Failed to validate JSON", False),
        ("messages: field required", False),
    ]:
        check(f"format-unsupported {body[:40]!r}",
              llm_mod._format_unsupported(body) is want)


def test_format_unsupported_degrades() -> None:
    print("\n[3c] a model without Structured Outputs degrades instead of failing")
    seen: list[str | None] = []

    def handler(body: dict) -> tuple[int, dict]:
        rf = body.get("response_format")
        seen.append(rf.get("type") if rf else None)
        if rf is not None:
            return 400, {"error": {"message": "response_format is not supported for this model"}}
        return 200, _reply(_candidates())

    with _Stub(handler) as stub:
        s = _with_key(_settings(stub.base), "gsk_test_key_0123456789")
        text = llm_mod.call_groq("transcript", s)
    check("still returns a usable reply", text.startswith("["), text[:40])
    check("exactly one wasted call, not the whole ladder",
          len(stub.bodies()) == 2, str(len(stub.bodies())))
    check("then drops the format", seen[-1] is None, str(seen))


# --------------------------------------------------------------------------- #
# 4. one bad chunk must not fail the job
# --------------------------------------------------------------------------- #

def test_detect_skips_failed_chunk() -> None:
    print("\n[4] a failing chunk is skipped; the rest of the job continues")
    calls = {"n": 0}

    def handler(body: dict) -> tuple[int, dict]:
        calls["n"] += 1
        # Every rung of the FIRST chunk's ladder fails (2 rungs, no fallbacks).
        # Later chunks succeed immediately.
        if calls["n"] <= 2:
            return _json400()
        return 200, _reply(_candidates_for_prompt(body, 2))

    logs: list[str] = []
    progress: list[tuple[float, str]] = []
    report: dict = {}
    with _Stub(handler) as stub:
        s = _with_key(
            _settings(stub.base, json_retries=1, fallback_models=()),
            "gsk_test_key_0123456789",
        )
        clips = llm_mod.detect_clips(
            _words(1600), s,
            on_log=logs.append,
            on_progress=lambda f, d: progress.append((f, d)),
            report=report,
        )
    check("the job did NOT fail", isinstance(clips, list), str(type(clips)))
    check("clips came from the surviving chunks", len(clips) > 0, str(len(clips)))
    check("report records the skipped chunks",
          report.get("chunksSkipped", 0) >= 1, str(report.get("chunksSkipped")))
    check("report records how many chunks there were",
          report.get("chunksTotal", 0) > 1, str(report.get("chunksTotal")))
    check("the user was told a chunk was skipped",
          any("skipped" in line for line in logs), str(logs[:4]))
    check("the skip names the chunk number",
          any("chunk 1/" in line for line in logs), str(logs[:4]))
    check("progress still advanced past the failed chunk",
          progress and progress[-1][0] > 0, str(progress[-1:]))


def test_all_chunks_failing_still_fails() -> None:
    print("\n[4b] when EVERY chunk fails the stage must still fail loudly")
    with _Stub(lambda b: _json400()) as stub:
        s = _with_key(
            _settings(stub.base, json_retries=0, fallback_models=()),
            "gsk_test_key_0123456789",
        )
        err = None
        try:
            llm_mod.detect_clips(_words(600), s)
        except llm_mod.LlmError as exc:
            err = str(exc)
    check("raises rather than reporting an empty success", err is not None, str(err))
    check("names how many chunks failed", "chunk" in (err or "").lower(), str(err))


def test_unparseable_reply_is_skipped_not_fatal() -> None:
    print("\n[4c] a 200 with an unusable body is skipped, not fatal")
    n = {"i": 0}

    def handler(body: dict) -> tuple[int, dict]:
        n["i"] += 1
        # First call: HTTP 200 but prose only. Later: real JSON.
        if n["i"] == 1:
            return 200, _reply("I'm sorry, I cannot help with that request.")
        return 200, _reply(_candidates_for_prompt(body, 2))

    logs: list[str] = []
    with _Stub(handler) as stub:
        s = _with_key(
            _settings(stub.base, json_retries=0, fallback_models=()),
            "gsk_test_key_0123456789",
        )
        clips = llm_mod.detect_clips(_words(1600), s, on_log=logs.append)
    check("a 200 with no JSON does not fail the job", isinstance(clips, list) and len(clips) > 0,
          str(clips)[:80])
    check("it was reported as a skip",
          any("no usable clips" in line for line in logs), str(logs[:4]))


def test_cache_preserved_across_failure() -> None:
    print("\n[4d] chunks cached before a failure survive for the resume")
    cache = llm_mod.DictChunkCache()
    n = {"i": 0}

    def handler(body: dict) -> tuple[int, dict]:
        n["i"] += 1
        # The SECOND chunk's ladder fails; every other chunk is cached fine.
        if n["i"] in (3, 4):
            return _json400()
        return 200, _reply(_candidates_for_prompt(body, 2))

    with _Stub(handler) as stub:
        s = _with_key(
            _settings(stub.base, json_retries=0, fallback_models=()),
            "gsk_test_key_0123456789",
        )
        llm_mod.detect_clips(_words(1600), s, chunk_cache=cache)
    cached = [i for i in range(20) if cache.get(i) is not None]
    check("good chunks were cached", len(cached) >= 1, str(cached))
    check("the failed chunk was NOT cached as empty",
          all(cache.get(i) for i in cached), str(cached))
    # A resume must reuse them without re-calling.
    before = n["i"]
    with _Stub(lambda b: (200, _reply(_candidates_for_prompt(b, 2)))) as stub2:
        s2 = _with_key(
            _settings(stub2.base, json_retries=0, fallback_models=()),
            "gsk_test_key_0123456789",
        )
        llm_mod.detect_clips(_words(1600), s2, chunk_cache=cache)
    check("resume re-billed only the missing chunks",
          len(stub2.bodies()) <= before, f"{len(stub2.bodies())} vs {before}")


def test_missing_key_still_raises() -> None:
    print("\n[4e] a missing key is still a hard, named error")
    s = _with_key(_settings("http://127.0.0.1:1/openai/v1"), NO_KEY)
    err = None
    try:
        llm_mod.detect_clips(_words(400), s)
    except llm_mod.LlmNotConfigured as exc:
        err = str(exc)
    check("raises LlmNotConfigured", err is not None, str(err))
    check("names GROQ_API_KEY", "GROQ_API_KEY" in (err or ""), str(err))


def test_ladder_is_bounded() -> None:
    print("\n[4f] the ladder is finite — it cannot spin")
    t0 = time.monotonic()
    with _Stub(lambda b: _json400()) as stub:
        s = _with_key(
            _settings(stub.base, json_retries=3, max_retries=1),
            "gsk_test_key_0123456789",
        )
        try:
            llm_mod.call_groq("transcript", s)
        except llm_mod.LlmError:
            pass
    dt = time.monotonic() - t0
    # 4 rungs x (1 + 1 retry) x 3 models = 24 calls max.
    check("call count is bounded", len(stub.bodies()) <= 24, str(len(stub.bodies())))
    check("terminates quickly", dt < 20, f"{dt:.1f}s")


def main() -> int:
    _install_key()
    try:
        test_schema_contract()
        test_json_400_escalates()
        test_ladder_drops_response_format()
        test_model_fallback_chain()
        test_whole_chain_exhausted_raises()
        test_permanent_400_fails_fast()
        test_json_classifier()
        test_format_unsupported_degrades()
        test_detect_skips_failed_chunk()
        test_all_chunks_failing_still_fails()
        test_unparseable_reply_is_skipped_not_fatal()
        test_cache_preserved_across_failure()
        test_missing_key_still_raises()
        test_ladder_is_bounded()
    finally:
        _uninstall_key()
    print(f"\ndetect_json: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
