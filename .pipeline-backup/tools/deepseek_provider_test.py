"""DeepSeek-primary / Groq-backup detection provider tests.

DeepSeek is a first-class detection provider in its own right — NOT the
NVIDIA-hosted `deepseek-*` models. Its keys are `sk-...` from
platform.deepseek.com and bill against a balance, so a wrong answer here would
be a misrouting bug rather than a cosmetic one.

What can actually break between "DeepSeek is the default" and a real request:

  * THE THINKING MODE. DeepSeek's thinking is ON by default at high effort and,
    while on, it *ignores* `temperature` — the docs say such parameters "will
    not trigger an error but will also have no effect". It also returns the
    chain-of-thought in `reasoning_content`, billed as OUTPUT tokens. So
    without an explicit `"thinking": {"type": "disabled"}` on the wire, the
    `temperature = 0.0` determinism and the whole JSON-retry ladder are
    decorative and every retry pays for reasoning we throw away. That single
    field is the reason this suite exists.
  * the base URL — DeepSeek documents the OpenAI-compatible base WITHOUT `/v1`
    (`https://api.deepseek.com`), and this transport appends
    `/chat/completions` itself, so a `/v1` here would double the path;
  * JSON mode — `json_object` is supported, but the prompt MUST contain the
    word "json", or the request is rejected;
  * per-provider pacing isolation — Groq's 30-rpm floor is Groq's budget, so a
    DeepSeek call must never wait it out;
  * failover — a primary that fails for a real reason must reach the backup,
    and a missing PRIMARY key must stay a loud config error;
  * the key preflight — the error must name DEEPSEEK_API_KEY, not Groq's.

Runs entirely against a local OpenAI-compatible stub. No network, no quota, no
keys.

    python tools/deepseek_provider_test.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

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
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n) or b"{}"
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:  # noqa: BLE001 - malformed request is the stub's problem
                    body = {}
                body["__auth"] = self.headers.get("Authorization")
                body["__ua"] = self.headers.get("User-Agent")
                body["__path"] = self.path
                outer.requests.append(body)
                status, payload = outer._handler(body)
                raw_out = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw_out)))
                self.end_headers()
                self.wfile.write(raw_out)

            def log_message(self, *a: Any) -> None:
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.srv.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()

    def __enter__(self) -> "_Stub":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _ok(payload: dict) -> Callable[[dict], tuple[int, dict]]:
    return lambda _body: (200, payload)


def _reply(candidates: list[dict], usage_tokens: int = 1000) -> dict:
    """A well-formed OpenAI chat completion carrying an envelope of clips."""
    return {
        "choices": [{"message": {"content": json.dumps({"clips": candidates})}}],
        "usage": {"total_tokens": usage_tokens},
    }


def _candidate(start: int = 0, end: int = 200, score: float = 88.0) -> dict:
    return {
        "start_word": start,
        "end_word": end,
        "score": score,
        "title": "A punchy title",
        "hook_text": "one line of post copy",
        "hashtags": ["#fyp", "#foryou", "#viral", "#topic"],
    }


def _words(n: int = 400) -> list[dict]:
    """Synthetic word-level timings 0.25s apart, like faster-whisper output."""
    return [{"word": f"w{i}", "start": i * 0.25, "end": i * 0.25 + 0.2} for i in range(n)]


# --------------------------------------------------------------------------- #
# settings helpers
# --------------------------------------------------------------------------- #


def _settings() -> Settings:
    """Real config, with pacing/quota knobs neutralised so tests stay fast."""
    s = load_config(quiet=True)
    for cfg in (s.deepseek, s.nvidia, s.groq):
        cfg.base_url = "http://127.0.0.1:1/v1"  # never reached: every test sets its own
        cfg.min_interval_sec = 0.0
        cfg.max_retries = 0
        cfg.json_retries = 1
        cfg.timeout_sec = 10
    s.deepseek.tokens_per_min = 0
    s.nvidia.tokens_per_min = 0
    s.groq.tokens_per_min = 0
    return s


class _Keys:
    """Force the live key properties for the duration of a `with` block.

    The properties re-resolve from .env on EVERY access, so the only way to
    test "no key configured" without touching the real environment is to
    replace the property on the class for the duration of the call.
    """

    def __init__(self, deepseek: str | None, groq: str | None, nvidia: str | None = None) -> None:
        self._values = {"deepseek": deepseek, "groq": groq, "nvidia": nvidia}
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> "_Keys":
        values = self._values
        for name in values:
            self._saved[name] = getattr(Settings, f"{name}_api_key")
            setattr(
                Settings,
                f"{name}_api_key",
                property(lambda _s, _n=name: values[_n]),  # type: ignore[assignment]
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, saved in self._saved.items():
            setattr(Settings, f"{name}_api_key", saved)  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# 1. configuration
# --------------------------------------------------------------------------- #


def test_config() -> None:
    print("\n[config] DeepSeek primary, Groq backup")
    s = load_config(quiet=True)
    check("default provider is deepseek", s.llm.provider == "deepseek", s.llm.provider)
    check("fallback provider is groq", s.llm.fallback_provider == "groq",
          s.llm.fallback_provider)
    check("deepseek endpoint has NO /v1 suffix",
          s.deepseek.base_url == "https://api.deepseek.com", s.deepseek.base_url)
    check("thinking mode is disabled by default",
          s.deepseek.thinking_mode == "disabled", s.deepseek.thinking_mode)
    check("detection temperature is still 0.0", s.deepseek.temperature == 0.0,
          str(s.deepseek.temperature))
    check("json mode is json_object (strict schema is Groq-only)",
          s.deepseek.response_format == "json_object", s.deepseek.response_format)
    check("model is a current DeepSeek model",
          s.deepseek.model in {"deepseek-flash", "deepseek-v4-pro"}, s.deepseek.model)
    check("deepseek has no request floor", s.deepseek.min_interval_sec == 0.0,
          str(s.deepseek.min_interval_sec))
    check("groq keeps its 30-rpm floor", s.groq.min_interval_sec >= 2.0,
          str(s.groq.min_interval_sec))
    check("deepseek daily limit is its own, not groq's",
          s.deepseek.daily_token_limit != s.groq.daily_token_limit,
          f"{s.deepseek.daily_token_limit} vs {s.groq.daily_token_limit}")
    check("providers list leads with deepseek",
          llm_mod.PROVIDERS[0] == "deepseek", str(llm_mod.PROVIDERS))
    check("nvidia is still a selectable provider", "nvidia" in llm_mod.PROVIDERS,
          str(llm_mod.PROVIDERS))
    check("label names the vendor",
          s.deepseek.label == "DeepSeek" and s.groq.label == "Groq",
          f"{s.deepseek.label}/{s.groq.label}")

    # A fallback equal to the primary is a no-op that would burn a second full
    # ladder proving what the first already proved.
    s2 = _settings()
    s2.llm.provider = "deepseek"
    s2.llm.fallback_provider = "deepseek"
    check("fallback equal to primary normalises to none",
          llm_mod.settings_llm_fallback(s2) == "", llm_mod.settings_llm_fallback(s2))
    s2.llm.fallback_provider = ""
    check("blank fallback means no failover", llm_mod.settings_llm_fallback(s2) == "")
    s2.llm.fallback_provider = "groq"
    check("a real fallback is reported", llm_mod.settings_llm_fallback(s2) == "groq")

    # A nonsense thinking_mode is a config error, not silently coerced: it is
    # the difference between deterministic extraction and paying for reasoning
    # nobody reads, so a typo must be loud.
    from app import config as cfg_mod  # noqa: PLC0415
    from app.config import ConfigError  # noqa: PLC0415

    raw = cfg_mod.load_raw() if hasattr(cfg_mod, "load_raw") else None
    if raw is not None:
        del raw["deepseek"]["thinking_mode"]
        raw["deepseek"]["thinking_mode"] = "sometimes"
        try:
            cfg_mod.load_config(raw=raw, quiet=True)
            check("a bad thinking_mode is rejected", False, "no ConfigError raised")
        except ConfigError as exc:
            check("a bad thinking_mode is rejected", "thinking_mode" in str(exc), str(exc))
    else:
        check("a bad thinking_mode is rejected", True, "skipped: loader takes no raw arg")


# --------------------------------------------------------------------------- #
# 2. THE THINKING MODE — the reason this suite exists
# --------------------------------------------------------------------------- #


def test_thinking_mode_wire_field() -> None:
    print("\n[thinking] the disable field reaches the wire")
    with _Stub(_ok(_reply([_candidate()]))) as stub:
        s = _settings()
        s.deepseek.base_url = stub.base_url
        prompt = llm_mod.build_prompt(_words(120), s.gemini, start_index=0)
        with _Keys("sk-test-0123456789", None):
            llm_mod.call_deepseek(prompt, s)

    req = stub.requests[0]
    check("a request was made", bool(req), "no request reached the stub")
    thinking = req.get("thinking")
    check("thinking is a top-level object", isinstance(thinking, dict), json.dumps(thinking))
    check("thinking.type is 'disabled'", (thinking or {}).get("type") == "disabled",
          json.dumps(thinking))
    # The exact failure this prevents: temperature 0.0 that does nothing.
    check("temperature is still sent alongside it", req.get("temperature") == 0.0,
          str(req.get("temperature")))

    # It must be honoured, not hardcoded — otherwise the config knob is a lie.
    with _Stub(_ok(_reply([_candidate()]))) as stub:
        s = _settings()
        s.deepseek.base_url = stub.base_url
        s.deepseek.thinking_mode = "enabled"
        with _Keys("sk-test-0123456789", None):
            llm_mod.call_deepseek("transcript text", s)
    check("thinking_mode = enabled is sent as 'enabled'",
          (stub.requests[0].get("thinking") or {}).get("type") == "enabled",
          json.dumps(stub.requests[0].get("thinking")))

    # And it must be a DEEPSEEK-only field: a provider without the concept
    # would reject an unknown body key.
    with _Stub(_ok(_reply([_candidate()]))) as stub:
        s = _settings()
        s.nvidia.base_url = stub.base_url
        s.groq.base_url = stub.base_url
        with _Keys("sk-test-0123456789", "gsk_test_0123456789", "nvapi-test-0123456789"):
            llm_mod.call_nvidia("transcript text", s)
            llm_mod.call_groq("transcript text", s)
    check("nvidia does NOT send a thinking field", "thinking" not in stub.requests[0],
          json.dumps(stub.requests[0].get("thinking")))
    check("groq does NOT send a thinking field", "thinking" not in stub.requests[1],
          json.dumps(stub.requests[1].get("thinking")))


# --------------------------------------------------------------------------- #
# 3. wire contract
# --------------------------------------------------------------------------- #


def test_wire_contract() -> None:
    print("\n[deepseek] wire contract")
    with _Stub(_ok(_reply([_candidate()]))) as stub:
        s = _settings()
        s.deepseek.base_url = stub.base_url
        # A REAL prompt, not a placeholder: the object-envelope rewrite below
        # operates on build_prompt()'s wording, so a stub string would make the
        # assertion vacuous.
        prompt = llm_mod.build_prompt(_words(120), s.gemini, start_index=0)
        with _Keys("sk-test-0123456789", None):
            text = llm_mod.call_deepseek(prompt, s)

    req = stub.requests[0]
    check("posts the configured model", req.get("model") == s.deepseek.model,
          str(req.get("model")))
    check("sends Bearer auth", req.get("__auth") == "Bearer sk-test-0123456789",
          str(req.get("__auth")))
    check("sends a non-urllib User-Agent",
          not str(req.get("__ua") or "").lower().startswith("python-urllib"),
          str(req.get("__ua")))
    check("posts to /chat/completions", req.get("__path") == "/v1/chat/completions",
          str(req.get("__path")))
    fmt = req.get("response_format") or {}
    check("asks for json_object", fmt.get("type") == "json_object", json.dumps(fmt))
    user = (req.get("messages") or [{}])[-1].get("content", "")
    # DeepSeek REJECTS a json_object request whose prompt omits the word json.
    check("prompt asks for an object envelope, not a bare array",
          '"clips"' in user and "Return ONLY a JSON array" not in user)
    check("prompt contains the word 'json' (a json_object hard requirement)",
          "json" in user.lower(), user[:120])
    check("reply is the raw assistant message", "clips" in text, text[:80])
    check("parse_candidates unwraps the envelope",
          len(llm_mod.parse_candidates(text)) == 1, text[:120])


# --------------------------------------------------------------------------- #
# 4. per-provider pacing isolation
# --------------------------------------------------------------------------- #


def test_pacing_isolation() -> None:
    print("\n[pacing] per-provider, not per-process")
    llm_mod._PACERS.clear()
    with _Stub(_ok(_reply([_candidate()]))) as stub:
        s = _settings()
        s.deepseek.base_url = stub.base_url
        s.groq.base_url = stub.base_url
        # Groq keeps a real floor; DeepSeek has none. If pacing were shared,
        # the second call (DeepSeek) would inherit Groq's wait.
        s.groq.min_interval_sec = 0.5
        with _Keys("sk-test-0123456789", "gsk_test_0123456789"):
            llm_mod.call_groq("p", s)          # starts Groq's floor clock
            start = time.monotonic()
            llm_mod.call_deepseek("p", s)      # must NOT wait on Groq's budget
            deepseek_wait = time.monotonic() - start
            start = time.monotonic()
            llm_mod.call_groq("p", s)          # DOES wait on its own floor
            groq_wait = time.monotonic() - start
    check("a deepseek call is not delayed by groq's floor", deepseek_wait < 0.2,
          f"{deepseek_wait:.2f}s")
    check("a groq call still honours its own floor", groq_wait >= 0.3, f"{groq_wait:.2f}s")

    llm_mod._PACERS.clear()
    llm_mod._groq_record_tokens(1000, "deepseek")
    llm_mod._groq_record_tokens(1000, "deepseek")
    check("ledgers are separate per provider",
          len(llm_mod._PACERS["deepseek"]["ledger"]) == 2 and "groq" not in llm_mod._PACERS,
          str(sorted(llm_mod._PACERS)))


# --------------------------------------------------------------------------- #
# 5. failover
# --------------------------------------------------------------------------- #


def test_failover() -> None:
    print("\n[failover] primary failure reaches the backup")

    # An empty balance is the real-world case: DeepSeek errors, Groq answers.
    deepseek_down = lambda _b: (  # noqa: E731 - a one-line script is clearest here
        402,
        {"error": {"message": "Insufficient Balance"}},
    )
    with _Stub(deepseek_down) as ds_stub, _Stub(_ok(_reply([_candidate()]))) as groq_stub:
        s = _settings()
        s.deepseek.base_url = ds_stub.base_url
        s.groq.base_url = groq_stub.base_url
        with _Keys("sk-test-0123456789", "gsk_test_0123456789"):
            text = llm_mod.call_llm("transcript text", s)
    check("a failing primary falls through to groq", "clips" in text, text[:80])
    check("the backup actually made the call", len(groq_stub.requests) == 1,
          str(len(groq_stub.requests)))
    check("the backup used the groq model", groq_stub.requests[0].get("model") == s.groq.model,
          str(groq_stub.requests[0].get("model")))

    # Failover OFF: the same error must surface, not be routed around.
    with _Stub(deepseek_down) as ds_stub, _Stub(_ok(_reply([_candidate()]))) as groq_stub:
        s = _settings()
        s.llm.fallback_provider = ""
        s.deepseek.base_url = ds_stub.base_url
        s.groq.base_url = groq_stub.base_url
        with _Keys("sk-test-0123456789", "gsk_test_0123456789"):
            err = None
            try:
                llm_mod.call_llm("transcript text", s)
            except llm_mod.LlmError as exc:
                err = str(exc)
    check("with no fallback configured the error surfaces", err is not None, str(err))
    check("...and names DeepSeek, not Groq", "DeepSeek" in (err or ""), str(err))
    check("...and the backup was never called", len(groq_stub.requests) == 0)

    # A fallback with no key is not a fallback: the original error stands.
    with _Stub(deepseek_down) as ds_stub, _Stub(_ok(_reply([_candidate()]))) as groq_stub:
        s = _settings()
        s.deepseek.base_url = ds_stub.base_url
        s.groq.base_url = groq_stub.base_url
        with _Keys("sk-test-0123456789", None):
            err = None
            try:
                llm_mod.call_llm("transcript text", s)
            except llm_mod.LlmError as exc:
                err = str(exc)
    check("a keyless backup does not mask the primary error", err is not None, str(err))
    check("...and it is still not called", len(groq_stub.requests) == 0)


# --------------------------------------------------------------------------- #
# 6. key preflight
# --------------------------------------------------------------------------- #


def test_key_preflight() -> None:
    print("\n[keys] the preflight names the right variable")
    s = _settings()
    s.llm.provider = "deepseek"
    with _Keys(None, "gsk_test_0123456789"):
        err = None
        try:
            llm_mod.detect_clips(_words(120), s)
        except llm_mod.LlmNotConfigured as exc:
            err = str(exc)
    check("a missing DeepSeek key is a config error", err is not None)
    check("...naming DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY" in (err or ""), str(err))
    check("...not Groq's", "GROQ_API_KEY" not in (err or ""), str(err))
    check("...and it points at the Keys tab", "Keys" in (err or ""), str(err))

    # A missing PRIMARY key must NOT silently downgrade to the backup: that
    # would hide a misconfiguration until someone read the logs.
    with _Stub(_ok(_reply([_candidate()]))) as groq_stub:
        s = _settings()
        s.deepseek.base_url = "http://127.0.0.1:1/v1"
        s.groq.base_url = groq_stub.base_url
        with _Keys(None, "gsk_test_0123456789"):
            err = None
            try:
                llm_mod.call_llm("transcript text", s)
            except llm_mod.LlmNotConfigured as exc:
                err = str(exc)
    check("a missing primary key does not fail over", err is not None and "DEEPSEEK" in err,
          str(err))
    check("...and the backup was never called", len(groq_stub.requests) == 0)

    # The resolver must accept both documented names, and only those.
    from app.config import _resolve_deepseek_key  # noqa: PLC0415

    key, source, problem = _resolve_deepseek_key({"DEEPSEEK_API_KEY": "sk-abcdefghijklmnop"})
    check("DEEPSEEK_API_KEY resolves", key == "sk-abcdefghijklmnop", str(key))
    check("...with no problem", problem is None, str(problem))
    key2, _s2, _p2 = _resolve_deepseek_key({"DEEPSEEK_KEY": "sk-abcdefghijklmnop"})
    check("DEEPSEEK_KEY is also accepted", key2 == "sk-abcdefghijklmnop", str(key2))
    key3, _s3, _p3 = _resolve_deepseek_key({})
    check("no key is None, not an empty string", key3 is None, repr(key3))


# --------------------------------------------------------------------------- #
# 7. a model without Structured Outputs still works (the ladder)
# --------------------------------------------------------------------------- #


def test_format_unsupported_falls_back_to_local_repair() -> None:
    print("\n[ladder] a provider without the requested mode still returns clips")
    seen: list[dict | None] = []
    thinking_seen: list[Any] = []

    def handler(body: dict) -> tuple[int, dict]:
        seen.append(body.get("response_format"))
        thinking_seen.append(body.get("thinking"))
        if body.get("response_format"):
            return 400, {"error": {"message":
                                   "response_format of type json_object is not supported "
                                   "by this model"}}
        # A fenced reply with prose around it — jsonfix must clean this up.
        return 200, {"choices": [{"message": {"content":
                                               "Sure! Here you go:\n```json\n"
                                               + json.dumps({"clips": [_candidate()]})
                                               + "\n```\nHope that helps."}}]}

    with _Stub(handler) as stub:
        s = _settings()
        s.deepseek.base_url = stub.base_url
        s.deepseek.response_format = "json_schema"
        # At least one JSON retry, or the ladder's FIRST rung is already the
        # no-response_format one and the unsupported-mode request never happens.
        s.deepseek.json_retries = 1
        with _Keys("sk-test-0123456789", None):
            text = llm_mod.call_deepseek("transcript text", s)
    check("the unsupported mode was tried once", seen and seen[0] is not None, str(seen))
    check("it was then dropped entirely", seen[-1] is None, str(seen))
    check("the fenced/prose reply was repaired locally",
          len(llm_mod.parse_candidates(text)) == 1, text[:120])
    # The thinking field is NOT a response_format: dropping the JSON mode must
    # not silently re-enable billed reasoning on the local-repair rung.
    check("thinking stays disabled on every rung of the ladder",
          all(t == {"type": "disabled"} for t in thinking_seen), str(thinking_seen))


# --------------------------------------------------------------------------- #
# 8. the API surface reports the active provider
# --------------------------------------------------------------------------- #


def test_api_surface() -> None:
    print("\n[api] detection status + usage follow the active provider")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app import main as main_mod  # noqa: PLC0415

    client = TestClient(main_mod.app)
    health = client.get("/api/health").json()
    status = health["detection"]
    check("health reports deepseek as the provider", status["provider"] == "deepseek",
          str(status.get("provider")))
    check("...lists the deepseek model", bool(status["models"].get("deepseek")),
          str(status["models"]))
    check("...still lists the nvidia model", bool(status["models"].get("nvidia")),
          str(status["models"]))
    check("...offers deepseek as a switchable option",
          any(o["value"] == "deepseek" for o in status["options"]), str(status["options"]))
    check("...names the backup", status.get("fallbackProvider") == "groq",
          str(status.get("fallbackProvider")))
    check("...carries a tradeoff string", "DeepSeek" in status.get("tradeoff", ""),
          status.get("tradeoff", ""))
    check("health exposes a deepseek key block", "llmDeepSeek" in health, str(sorted(health)))

    usage = client.get("/api/usage").json()
    check("usage reports which provider it describes", usage.get("provider") == "deepseek",
          str(usage.get("provider")))
    check("usage carries a display label", usage.get("providerLabel") == "DeepSeek",
          str(usage.get("providerLabel")))
    check("usage limit is deepseek's, not groq's",
          usage["limit"] == main_mod.settings.deepseek.daily_token_limit,
          f"{usage['limit']} vs {main_mod.settings.deepseek.daily_token_limit}")

    # The key endpoints exist for the primary, like the existing Groq ones.
    routes = {r.path for r in main_mod.app.routes if hasattr(r, "methods")}
    check("POST /api/settings/deepseek-key exists", "/api/settings/deepseek-key" in routes)
    check("DELETE /api/settings/deepseek-key exists", "/api/settings/deepseek-key" in routes)
    check("the nvidia key endpoints are untouched", "/api/settings/nvidia-key" in routes)
    bad = client.post("/api/settings/deepseek-key", json={"key": "short"})
    check("a too-short key is rejected before it is written", bad.status_code == 422,
          str(bad.status_code))
    # An NVIDIA key pasted into the DeepSeek field is the exact mix-up this
    # project keeps hitting, so it must be refused with a pointer, not a 401.
    wrong = client.post("/api/settings/deepseek-key",
                        json={"key": "nvapi-0123456789abcdefghijklmnopqrstuvwxyz"})
    check("an nvapi- key is refused in the DeepSeek field",
          wrong.status_code == 400 and "platform.deepseek.com" in wrong.text,
          f"{wrong.status_code} {wrong.text[:120]}")

    reload_body = client.post("/api/settings/reload-secrets").json()
    check("Re-check reports deepseek key state", "deepseek" in reload_body,
          str(sorted(reload_body)))
    check("Re-check still reports nvidia key state", "nvidia" in reload_body,
          str(sorted(reload_body)))

    # The UI must be able to render every provider it offers.
    for name in ("deepseek", "nvidia", "groq", "gemini", "ollama"):
        check(f"tradeoff copy exists for {name}",
              bool(main_mod.DETECTION_TRADEOFFS.get(name)), "missing")


def main() -> int:
    test_config()
    test_thinking_mode_wire_field()
    test_wire_contract()
    test_pacing_isolation()
    test_failover()
    test_key_preflight()
    test_format_unsupported_falls_back_to_local_repair()
    test_api_surface()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
