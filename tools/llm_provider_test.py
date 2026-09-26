"""Provider contract tests for the AI detection backend (groq / gemini / ollama).

The Groq path is exercised against a LOCAL OpenAI-compatible stub rather than
the real api.groq.com, so the suite is deterministic, offline, and free. What
is under test is everything that actually breaks in production:

  * the wire contract  — endpoint, Bearer auth, model, messages, JSON reply
  * rate-limit defence — Retry-After honoured, exponential backoff, pacing
  * terminal failures  — retry exhaustion and dead models fail LOUDLY and fast
  * the key precondition — a missing GROQ_API_KEY is a clear config error

Run:  python tools/llm_provider_test.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm as llm_mod  # noqa: E402
from app.config import Settings, load_config  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


# --------------------------------------------------------------------------- #
# OpenAI-compatible stub
# --------------------------------------------------------------------------- #

class _Stub:
    """A tiny chat-completions server whose behaviour the test scripts.

    `script` is a list of responses consumed one per request; the last entry
    repeats forever. Every request is recorded so the wire contract can be
    asserted, not just the outcome.
    """

    def __init__(self, script: list[tuple[int, dict, dict | None]]) -> None:
        self.script = script
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # silence stderr spam
                pass

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    body = {}
                with stub.lock:
                    idx = min(len(stub.requests), len(stub.script) - 1)
                    status, payload, headers = stub.script[idx]
                    stub.requests.append({
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "contentType": self.headers.get("Content-Type"),
                        "body": body,
                    })
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, str(v))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                data = json.dumps({"data": [{"id": "openai/gpt-oss-120b"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/openai/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Stub":
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def _reply(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _candidates(start_word: int, end_word: int, score: int = 90) -> str:
    return json.dumps({"candidates": [{
        "start_word": start_word,
        "end_word": end_word,
        "score": score,
        "title": f"Clip from {start_word}",
        "hook_text": "watch this",
    }]})


def _settings(base_url: str, key: str | None = "gsk_test_key_0123456789", **groq) -> Settings:
    s = load_config(quiet=True)
    s.llm.provider = "groq"
    s.groq.base_url = base_url
    s.groq.min_interval_sec = groq.pop("min_interval_sec", 0.0)
    s.groq.max_retries = groq.pop("max_retries", 3)
    for k, v in groq.items():
        setattr(s.groq, k, v)
    if key is None:
        # Bypass the live env/files resolver — this test owns the key value.
        s.__dict__["groq_api_key_override"] = None
    return s


def _with_key(s: Settings, key: str | None) -> Settings:
    """Force settings.groq_api_key without touching the real environment.

    The property re-resolves from .env on every access, so the stub value has
    to be injected at the class level for the duration of the call.
    """
    orig = type(s).groq_api_key
    type(s).groq_api_key = property(lambda self: key)  # type: ignore[assignment]
    return s, orig


def _restore(orig) -> None:
    Settings.groq_api_key = orig  # type: ignore[assignment]


def _words(n: int = 400) -> list[dict]:
    """Synthetic word-level timings: 0.25s apart, like faster-whisper output."""
    return [{"word": f"w{i}", "start": i * 0.25, "end": i * 0.25 + 0.2} for i in range(n)]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_wire_contract() -> None:
    print("\n[groq] wire contract — endpoint, auth, model, JSON reply")
    with _Stub([(200, _reply(_candidates(10, 120)), None)]) as stub:
        s, orig = _with_key(_settings(stub.base_url), "gsk_test_key_0123456789")
        try:
            check("key resolves from the stub", s.groq_api_key == "gsk_test_key_0123456789")
            check("provider label names the model", "openai/gpt-oss-120b" in llm_mod.provider_label(s))
            text = llm_mod.call_groq("hello transcript", s)
        finally:
            _restore(orig)
        req = stub.requests[0]
        check("posts to /chat/completions", req["path"] == "/openai/v1/chat/completions", req["path"])
        check("sends Bearer token", req["auth"] == "Bearer gsk_test_key_0123456789", str(req["auth"]))
        check("sends JSON content-type", "application/json" in (req["contentType"] or ""))
        check("sends the configured model", req["body"].get("model") == "openai/gpt-oss-120b",
              str(req["body"].get("model")))
        msgs = req["body"].get("messages") or []
        check("sends a user message carrying the prompt",
              any(m.get("role") == "user" and m.get("content") == "hello transcript" for m in msgs))
        check("asks for JSON output mode",
              req["body"].get("response_format") == {"type": "json_object"})
        check("returns the assistant message verbatim", text == _candidates(10, 120))
        parsed = llm_mod.parse_candidates(text)
        check("reply parses through the shared parser", len(parsed) == 1, str(parsed))
        check("parsed candidate keeps the contract fields",
              parsed and {"start_word", "end_word", "score", "title"} <= set(parsed[0]))


def test_429_retry_with_retry_after() -> None:
    print("\n[groq] 429 — Retry-After is honoured, then the call succeeds")
    script = [
        (429, {"error": {"message": "Rate limit reached"}}, {"Retry-After": "1"}),
        (200, _reply(_candidates(1, 40)), None),
    ]
    with _Stub(script) as stub:
        s, orig = _with_key(_settings(stub.base_url), "gsk_test_key_0123456789")
        try:
            t0 = time.monotonic()
            text = llm_mod.call_groq("prompt", s)
            elapsed = time.monotonic() - t0
        finally:
            _restore(orig)
        check("recovers after one 429", text.startswith("{"))
        check("retried exactly once", len(stub.requests) == 2, str(len(stub.requests)))
        check("waited the server's Retry-After (~1s)", elapsed >= 1.0, f"{elapsed:.2f}s")


def test_429_exhaustion() -> None:
    print("\n[groq] 429 — retry budget is finite, then it fails loudly")
    with _Stub([(429, {"error": {"message": "Rate limit reached"}}, {"Retry-After": "0"})]) as stub:
        s, orig = _with_key(_settings(stub.base_url, max_retries=2), "gsk_test_key_0123456789")
        try:
            err = None
            try:
                llm_mod.call_groq("prompt", s)
            except llm_mod.LlmError as exc:
                err = str(exc)
        finally:
            _restore(orig)
        check("raises LlmError instead of hanging", err is not None, str(err))
        check("error names the backend", "Groq" in (err or ""), str(err))
        check("stops after max_retries attempts (1 + 2)",
              len(stub.requests) == 3, str(len(stub.requests)))
        check("does NOT fall through to another provider", "gemini" not in (err or "").lower())


def test_dead_model_fails_fast() -> None:
    print("\n[groq] retired model — fails immediately, no retry burn")
    with _Stub([(400, {"error": {"message":
                 "The model `openai/gpt-oss-120b` has been decommissioned"}}, None)]) as stub:
        s, orig = _with_key(_settings(stub.base_url), "gsk_test_key_0123456789")
        try:
            err = None
            try:
                llm_mod.call_groq("prompt", s)
            except llm_mod.LlmError as exc:
                err = str(exc)
        finally:
            _restore(orig)
        check("raises LlmError", err is not None, str(err))
        check("error names the dead model", "openai/gpt-oss-120b" in (err or ""), str(err))
        check("points at the model docs", "console.groq.com/docs/models" in (err or ""), str(err))
        check("made only ONE call (no pointless retries)", len(stub.requests) == 1,
              str(len(stub.requests)))


def test_response_format_degrades() -> None:
    print("\n[groq] a model that rejects response_format still works")
    script = [
        (400, {"error": {"message": "response_format is not supported for this model"}}, None),
        (200, _reply(_candidates(5, 50)), None),
    ]
    with _Stub(script) as stub:
        s, orig = _with_key(_settings(stub.base_url), "gsk_test_key_0123456789")
        try:
            text = llm_mod.call_groq("prompt", s)
        finally:
            _restore(orig)
        check("succeeds by dropping the JSON-mode hint", text.startswith("{"))
        check("second request had no response_format",
              "response_format" not in stub.requests[1]["body"])


def test_pacing() -> None:
    print("\n[groq] client-side pacing keeps calls under the free-tier RPM")
    with _Stub([(200, _reply(_candidates(1, 30)), None)]) as stub:
        s, orig = _with_key(_settings(stub.base_url, min_interval_sec=0.5),
                            "gsk_test_key_0123456789")
        try:
            t0 = time.monotonic()
            for _ in range(3):
                llm_mod.call_groq("prompt", s)
            elapsed = time.monotonic() - t0
        finally:
            _restore(orig)
        # 3 calls at a 0.5s floor = 2 gaps = ~1.0s minimum.
        check("3 calls spaced by the configured floor (>=1.0s)", elapsed >= 1.0, f"{elapsed:.2f}s")


def test_missing_key() -> None:
    print("\n[groq] missing key — a clear config error, not an API 401")
    s, orig = _with_key(_settings("http://127.0.0.1:1/openai/v1"), None)
    try:
        err = None
        try:
            llm_mod.detect_clips(_words(400), s)
        except llm_mod.LlmNotConfigured as exc:
            err = str(exc)
    finally:
        _restore(orig)
    check("raises LlmNotConfigured", err is not None, str(err))
    check("names the missing env var", "GROQ_API_KEY" in (err or ""), str(err))
    check("tells the user where to fix it", "Keys/API keys tab" in (err or ""), str(err))


def test_detect_clips_end_to_end() -> None:
    print("\n[groq] detect_clips end-to-end — word indices resolved to real times")
    with _Stub([(200, _reply(_candidates(10, 130)), None)]) as stub:
        s, orig = _with_key(_settings(stub.base_url), "gsk_test_key_0123456789")
        try:
            clips = llm_mod.detect_clips(_words(400), s)
        finally:
            _restore(orig)
        check("returns one clip", len(clips) == 1, str(len(clips)))
        c = clips[0] if clips else {}
        check("resolves start from Whisper word timings", abs(c.get("start", -1) - 2.5) < 0.01,
              str(c.get("start")))
        # word 130 ends at 130*0.25 + 0.2 = 32.7
        check("resolves end from Whisper word timings", abs(c.get("end", -1) - 32.7) < 0.01,
              str(c.get("end")))
        check("keeps the MPT contract", {"start", "end", "score", "title", "hook_text"} <= set(c))


def test_provider_roster() -> None:
    print("\n[provider] roster and dispatch")
    check("groq is the default", load_config(quiet=True).llm.provider == "groq")
    check("all three are offered", llm_mod.PROVIDERS == ("groq", "gemini", "ollama"))
    s = load_config(quiet=True)
    s.llm.provider = "ollama"
    check("ollama still dispatches to its own client",
          "ollama" in llm_mod.provider_label(s))
    s.llm.provider = "gemini"
    check("gemini still dispatches to its own client",
          "gemini" in llm_mod.provider_label(s))
    s.llm.provider = "groq"
    check("groq still dispatches to its own client",
          "groq" in llm_mod.provider_label(s))


def main() -> int:
    print("LLM provider tests (groq contract / rate limits / key handling)")
    test_wire_contract()
    test_429_retry_with_retry_after()
    test_429_exhaustion()
    test_dead_model_fails_fast()
    test_response_format_degrades()
    test_pacing()
    test_missing_key()
    test_detect_clips_end_to_end()
    test_provider_roster()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
