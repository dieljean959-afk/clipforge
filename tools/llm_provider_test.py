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
import re
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
                        "userAgent": self.headers.get("User-Agent"),
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
    # The production client paces itself against a per-minute TOKEN budget so a
    # long source cannot walk into a 429 mid-detection. These tests make a
    # handful of small calls back to back against a local stub, so the pacer
    # would just add 50s of sleeping to a suite that should be instant. Off.
    s.groq.tokens_per_min = 0
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

def test_user_agent_is_sent() -> None:
    """Cloudflare bans Python-urllib/3.x at the edge with 403 / error 1010.

    This is a signature ban applied BEFORE the key is checked, so a perfectly
    valid key still fails. Verified against the live API: urllib default ->
    403/1010, an explicit UA -> 401. The header must therefore be present, and
    must not be the urllib default.
    """
    print("\n[groq] User-Agent — Cloudflare 1010 guard")
    with _Stub([(200, _reply(_candidates(1, 40)), None)]) as stub:
        s, orig = _with_key(_settings(stub.base_url), "gsk_test_key_0123456789")
        try:
            llm_mod.call_groq("prompt", s)
        finally:
            _restore(orig)
        ua = stub.requests[0].get("userAgent") or ""
        check("sends a User-Agent header", bool(ua), repr(ua))
        check("User-Agent is not the Cloudflare-banned urllib default",
              not ua.lower().startswith("python-urllib"), ua)
        check("User-Agent is the shared constant", ua == llm_mod.GROQ_USER_AGENT, ua)

    # And a 403/1010 must fail fast with an explanation, not "bad key".
    with _Stub([(403, {"error": {"message": "error code: 1010"}}, None)]) as stub:
        s, orig = _with_key(_settings(stub.base_url), "gsk_test_key_0123456789")
        try:
            err = None
            try:
                llm_mod.call_groq("prompt", s)
            except llm_mod.LlmError as exc:
                err = str(exc)
        finally:
            _restore(orig)
        check("403/1010 raises LlmError", err is not None, str(err))
        check("1010 error says it is NOT a bad key",
              "not a bad api key" in (err or "").lower(), str(err))
        check("1010 error names the User-Agent actually sent",
              llm_mod.GROQ_USER_AGENT in (err or ""), str(err))
        check("1010 made only ONE call (edge blocks are not transient)",
              len(stub.requests) == 1, str(len(stub.requests)))


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
        # The default is strict json_schema, NOT json_object. json_object forces
        # a top-level OBJECT while the prompt asks for a top-level ARRAY; the
        # model then had two contradictory instructions and failed validation
        # non-deterministically (reproduced live: 200 and 400 on consecutive
        # calls with one config). Strict schema constrains the decoder, so the
        # shape cannot be violated.
        rf = req["body"].get("response_format") or {}
        check("asks for strict json_schema", rf.get("type") == "json_schema", str(rf))
        check("schema is strict", (rf.get("json_schema") or {}).get("strict") is True, str(rf))
        check("schema puts clips under a top-level object",
              (rf.get("json_schema") or {}).get("schema", {}).get("required") == ["clips"], str(rf))
        check("no bare max_tokens default of 1600 left",
              req["body"].get("max_tokens") == 3200, str(req["body"].get("max_tokens")))
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
        # A spent 429 budget is now a LADDER failure: the rung exhausts its
        # in-attempt retries, and the ladder escalates through its rungs and
        # then the model chain. What must stay true is that it terminates,
        # names what it tried, and never silently switches PROVIDER.
        check("error names the detection chain", "detection chain" in (err or ""), str(err))
        check("error lists the models it tried",
              "openai/gpt-oss-120b" in (err or ""), str(err))
        check("error explains the rate limit",
              "429" in (err or "") or "Rate limit" in (err or ""), str(err))
        # Bounded: 3 in-attempt retries x 4 ladder rungs x 3 models in the
        # chain. The old behaviour stopped at 3; the ladder is what makes a
        # malformed reply survivable, and it is still strictly finite.
        check("ladder is bounded, not unbounded", len(stub.requests) <= 40, str(len(stub.requests)))
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
    check("deepseek is the default", load_config(quiet=True).llm.provider == "deepseek")
    check("groq is the configured backup",
          load_config(quiet=True).llm.fallback_provider == "groq")
    check("all five are offered",
          llm_mod.PROVIDERS == ("deepseek", "nvidia", "groq", "gemini", "ollama"),
          str(llm_mod.PROVIDERS))
    s = load_config(quiet=True)
    s.llm.provider = "deepseek"
    check("deepseek still dispatches to its own client",
          "deepseek" in llm_mod.provider_label(s))
    s.llm.provider = "nvidia"
    check("nvidia still dispatches to its own client",
          "nvidia" in llm_mod.provider_label(s))
    s.llm.provider = "ollama"
    check("ollama still dispatches to its own client",
          "ollama" in llm_mod.provider_label(s))
    s.llm.provider = "gemini"
    check("gemini still dispatches to its own client",
          "gemini" in llm_mod.provider_label(s))
    s.llm.provider = "groq"
    check("groq still dispatches to its own client",
          "groq" in llm_mod.provider_label(s))


def test_no_gemini_shaped_strings():
    """A missing-key error must name the provider actually in use.

    Two places hardcoded GEMINI_API_KEY after the Groq migration: the
    /redetect precondition, which refused outright for anyone with only a
    Groq key, and the pipeline's offline failure message, which overrode the
    exception's own provider-accurate text. Both sent users hunting for a key
    that was never the one being used.
    """
    root = Path(__file__).resolve().parents[1] / "app"
    main_src = (root / "main.py").read_text(encoding="utf-8")
    task_src = (root / "task.py").read_text(encoding="utf-8")
    llm_src = (root / "llm.py").read_text(encoding="utf-8")

    redetect = main_src.split("def redetect_job(", 1)[-1].split("\n@app.", 1)[0]
    # Strip comments before grepping: the fix's own comment explains that the
    # old string was wrong, and a naive substring check then fails on the
    # explanation of the fix.
    redetect_code = "\n".join(
        ln for ln in redetect.splitlines() if not ln.lstrip().startswith("#")
    )
    task_code = "\n".join(
        ln for ln in task_src.splitlines() if not ln.lstrip().startswith("#")
    )
    check("/redetect no longer checks a Gemini key",
          "gemini_api_key" not in redetect_code and "live_gemini_key" not in redetect_code)
    check("/redetect reuses the provider-aware readiness check",
          "detection_status()" in redetect_code)
    check("/redetect surfaces the active provider's own reason",
          re.search(r"status\.get\(\s*['\"]detail['\"]", redetect_code) is not None)
    check("the pipeline's offline error no longer hardcodes GEMINI_API_KEY",
          "GEMINI_API_KEY" not in task_code)
    check("the pipeline surfaces the exception's own provider-accurate message",
          re.search(r"AI detection offline\s*—\s*\{exc\}", task_code) is not None)

    check("llm.py names DEEPSEEK_API_KEY when the primary's key is missing",
          "DEEPSEEK_API_KEY" in llm_src)
    check("llm.py names NVIDIA_API_KEY when NVIDIA's key is missing",
          "NVIDIA_API_KEY" in llm_src)
    check("llm.py names GROQ_API_KEY when Groq's key is missing",
          "GROQ_API_KEY" in llm_src)
    check("llm.py still names GEMINI_API_KEY for the fallback",
          "GEMINI_API_KEY" in llm_src)

    # Provenance must name the provider that actually ran. The literal
    # "gemini" was written unconditionally, so a Groq-detected batch was
    # filed as Gemini work — and the dashboard inferred success from that
    # exact string, so correcting the provenance without fixing the UI would
    # have labelled every successful job "Detection failed".
    task_render = task_src.split("def _stage_render(", 1)[-1]
    check("the pipeline no longer hardcodes detectedSource=gemini",
          'detected_source="gemini"' not in task_render)
    check("detectedSource records the provider that actually ran",
          "detected_source=llm.provider(" in task_render)
    for ui in ("src/components/clipforge/JobCard.tsx", "src/pages/Dashboard.tsx"):
        ui_src = (Path(__file__).resolve().parents[1] / ui).read_text(encoding="utf-8")
        check(f"{Path(ui).name} no longer infers success from the string 'gemini'",
              'detectedSource === "gemini"' not in ui_src)
        check(f"{Path(ui).name} treats ANY provider as a successful detection",
              re.search(r"detectedSource\s*\n?\s*\?", ui_src) is not None)
        check(f"{Path(ui).name} still shows the honest failure label",
              "Detection failed" in ui_src)


def main() -> int:
    print("LLM provider tests (groq contract / rate limits / key handling)")
    test_user_agent_is_sent()
    test_wire_contract()
    test_429_retry_with_retry_after()
    test_429_exhaustion()
    test_dead_model_fails_fast()
    test_response_format_degrades()
    test_pacing()
    test_missing_key()
    test_detect_clips_end_to_end()
    test_provider_roster()
    test_no_gemini_shaped_strings()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
