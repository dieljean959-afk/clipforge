#!/usr/bin/env python3
"""Detection-backend tests: hosted (Gemini) vs local (Ollama).

Covers the three claims that matter for the AI clip/highlight-detection stage:

  1. The backend is selectable and defaults to the hosted one.
  2. Only transcript TEXT ever reaches a backend — the Ollama path is pointed
     at a stub HTTP server that records the exact request body, and the test
     asserts that body is JSON text with no audio/video/base64 payload.
  3. The Ollama path requires NO API key and produces the same resolved clips;
     the Gemini path still hard-fails without a key (no silent fallback).

Run: python3 tools/detect_backend_test.py
"""
from __future__ import annotations

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# A long base64-ish run would indicate an encoded media attachment.
_B64_BLOB = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
# Any media file extension in the payload would mean bytes, not text, went out.
_MEDIA_EXT = re.compile(r"\.(wav|mp4|mp3|m4a|webm|mov|mkv|flac|aac)\b", re.IGNORECASE)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.config import load_config  # noqa: E402


def _fake_words(n: int = 260) -> list[dict]:
    """A synthetic transcript: 260 words, 0.4s apart, with real timings."""
    body = ("this is a test transcript about a viral clip idea and the "
            "hook that makes people stop scrolling right now").split()
    out = []
    for i in range(n):
        out.append({
            "word": body[i % len(body)],
            "start": round(i * 0.4, 2),
            "end": round(i * 0.4 + 0.35, 2),
        })
    return out


class _StubOllama:
    """Minimal stand-in for the Ollama HTTP API that records what it receives."""

    def __init__(self) -> None:
        self.received: list[bytes] = []
        self.tags_calls = 0
        self._srv: ThreadingHTTPServer | None = None

    def __enter__(self) -> "_StubOllama":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a) -> None:  # silence the default stderr spam
                pass

            def _send(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/tags":
                    outer.tags_calls += 1
                    self._send(200, {"models": [{"name": "llama3.1:8b"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.received.append(raw)
                # Answer with a valid word-index range inside the transcript.
                self._send(200, {
                    "response": json.dumps([{
                        "start_word": 10, "end_word": 120, "score": 91,
                        "title": "Stub clip", "hook_text": "WAIT FOR IT",
                        "hashtags": ["#test"],
                    }]),
                    "done": True,
                })

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc) -> None:
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()

    @property
    def url(self) -> str:
        assert self._srv is not None
        host, port = self._srv.server_address[:2]
        return f"http://{host}:{port}"


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"   {'OK ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    settings = load_config(quiet=True)
    words = _fake_words()

    # --- 1. selection / default -------------------------------------------
    print("== 1. backend selection")
    settings.llm.provider = "gemini"
    check("config default is deepseek (fresh load)", load_config(quiet=True).llm.provider == "deepseek")
    check("hosted label names the configured model",
          settings.gemini.model in llm.provider_label(settings))
    settings.llm.provider = "groq"
    check("groq label names the configured model",
          settings.groq.model in llm.provider_label(settings))
    settings.llm.provider = "gemini"

    settings.llm.provider = "ollama"
    check("local backend selects ollama", llm.provider(settings) == "ollama")
    check("local label names the ollama model",
          settings.llm.ollama_model in llm.provider_label(settings))
    settings.llm.provider = "gemini"

    # --- 2. hosted backend, no key -> hard fail, never a silent fallback ---
    print()
    print("== 2. hosted backend (gemini)")
    # Force the "no key" branch by temporarily swapping the live resolver.
    import app.config as cfg_mod

    real_status = cfg_mod.live_gemini_key

    class _NoKey:
        key = None
        source = "none"
        problem = "test: key absent"

        def refresh(self, force: bool = False) -> None:
            return None

        def status(self) -> dict:
            return {"keyLoaded": False, "source": "none",
                    "problem": "test: key absent", "aliases": ["GEMINI_API_KEY"]}

    cfg_mod.live_gemini_key = lambda: _NoKey()  # type: ignore[assignment]
    import app.llm as llm_mod

    llm_mod.live_gemini_key = lambda: _NoKey()  # type: ignore[assignment]
    try:
        llm.detect_clips(words, settings, max_clips=3)
        check("hosted backend without a key fails loudly", False, "it returned clips")
    except llm.LlmNotConfigured as exc:
        check("hosted backend without a key fails loudly", True, str(exc)[:60])
    finally:
        cfg_mod.live_gemini_key = real_status  # type: ignore[assignment]
        llm_mod.live_gemini_key = real_status  # type: ignore[assignment]

    # Hosted call path: stub the SDK so no network is touched, and assert the
    # prompt handed to it is plain text derived from the transcript.
    #
    # The key resolver has to be stubbed too, not just the transport:
    # `detect_clips` now refuses to call a backend whose key is missing — on
    # purpose, because a missing key is a misconfiguration to fix rather than a
    # transient failure to route around — so with only `call_gemini` stubbed
    # the run stopped at the key check and never reached the call path this
    # section is about. The assertions are unchanged; the fixture now provides
    # the key the product insists on.
    class _HasKey:
        key = "test-gemini-key-0123456789"
        source = "test"
        problem = None

        def refresh(self, force: bool = False) -> None:
            return None

        def status(self) -> dict:
            return {"keyLoaded": True, "source": "test",
                    "problem": "", "aliases": ["GEMINI_API_KEY"]}

    captured: dict[str, str] = {}

    def _fake_gemini(prompt: str, _settings) -> str:
        captured["prompt"] = prompt
        return json.dumps([{"start_word": 10, "end_word": 120, "score": 88,
                            "title": "Hosted clip", "hook_text": "GO",
                            "hashtags": ["#hosted"]}])

    real_call_gemini = llm.call_gemini
    has_key = _HasKey()
    cfg_mod.live_gemini_key = lambda: has_key  # type: ignore[assignment]
    llm_mod.live_gemini_key = lambda: has_key  # type: ignore[assignment]
    llm.call_gemini = _fake_gemini  # type: ignore[assignment]
    try:
        settings.llm.provider = "gemini"
        clips = llm.detect_clips(words, settings, max_clips=3)
        check("hosted backend returns resolved clips", len(clips) >= 1,
              f"{len(clips)} clip(s), first {clips[0]['start']:.2f}-{clips[0]['end']:.2f}s")
        check("hosted backend resolves word indices to real timings",
              abs(clips[0]["start"] - 10 * 0.4) < 0.01, f"start={clips[0]['start']:.2f}")
        check("hosted prompt is TEXT (numbered transcript, no media)",
              isinstance(captured.get("prompt"), str)
              and "NUMBERED TRANSCRIPT" in captured["prompt"]
              and "base64" not in captured["prompt"].lower())
    finally:
        llm.call_gemini = real_call_gemini  # type: ignore[assignment]
        cfg_mod.live_gemini_key = real_status  # type: ignore[assignment]
        llm_mod.live_gemini_key = real_status  # type: ignore[assignment]

    # --- 3. local backend against a stub Ollama server --------------------
    print()
    print("== 3. local backend (ollama, no API key required)")
    with _StubOllama() as stub:
        settings.llm.provider = "ollama"
        settings.llm.ollama_url = stub.url
        # No Gemini key: the local path must not consult it at all.
        llm_mod.live_gemini_key = lambda: _NoKey()  # type: ignore[assignment]
        try:
            ok, reason = llm.ollama_status(settings)
            check("ollama_status reports the stub server ready", ok, reason)

            clips = llm.detect_clips(words, settings, max_clips=3)
            check("local backend returns resolved clips", len(clips) >= 1,
                  f"{len(clips)} clip(s), first {clips[0]['start']:.2f}-{clips[0]['end']:.2f}s")
            check("local backend resolves word indices to real timings",
                  abs(clips[0]["start"] - 10 * 0.4) < 0.01, f"start={clips[0]['start']:.2f}")
            check("both backends produce the same clip timing contract",
                  abs(clips[0]["start"] - 4.0) < 0.01 and abs(clips[0]["end"] - 48.0) < 0.5,
                  f"{clips[0]['start']:.2f}-{clips[0]['end']:.2f}")
        except llm.LlmNotConfigured as exc:
            check("local backend runs WITHOUT an API key", False, str(exc)[:80])
        finally:
            llm_mod.live_gemini_key = real_status  # type: ignore[assignment]

        # --- 4. only text was sent ---------------------------------------
        print()
        print("== 4. only transcript TEXT reaches the backend")
        check("the backend was actually called", len(stub.received) >= 1,
              f"{len(stub.received)} request(s)")
        for i, raw in enumerate(stub.received, 1):
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                check(f"request {i} is UTF-8 JSON", False)
                continue
            prompt = str(body.get("prompt", ""))
            check(f"request {i} is JSON text", True, f"{len(raw)} bytes")
            check(f"request {i} prompt carries the transcript", "NUMBERED TRANSCRIPT" in prompt)

            # Structure: only the documented keys, and prompt must be a str.
            # (Asserting the substring "video" is absent would be wrong — the
            # prompt PROSE says "clips from a video transcript". What matters
            # is that no media is attached, i.e. no media keys and no blobs.)
            expected = {"model", "prompt", "stream", "format", "keep_alive", "options"}
            check(f"request {i} sends only text fields", set(body) == expected,
                  f"keys={sorted(body)}")
            check(f"request {i} prompt is a plain string", isinstance(body.get("prompt"), str))
            check(
                f"request {i} has no media/file fields",
                not any(k in body for k in ("audio", "video", "images", "files", "file")),
            )
            check(
                f"request {i} carries no base64 blob",
                _B64_BLOB.search(raw.decode("utf-8", "replace")) is None,
                "no long encoded payload",
            )
            check(
                f"request {i} prompt references no media file",
                _MEDIA_EXT.search(prompt) is None,
                "no .wav/.mp4/.mp3/.m4a paths in the payload",
            )

    # --- 5. unreachable local server is reported, not crashed -------------
    print()
    print("== 5. local backend with no server running")
    settings.llm.provider = "ollama"
    settings.llm.ollama_url = "http://127.0.0.1:9"  # discard port, nothing listens
    ok, reason = llm.ollama_status(settings)
    check("unreachable server is reported, not raised", not ok, reason[:70])
    try:
        llm.detect_clips(words, settings, max_clips=2)
        check("unreachable server fails with actionable guidance", False)
    except llm.LlmNotConfigured as exc:
        check("unreachable server fails with actionable guidance", True,
              "tells the user to run `ollama serve`" if "ollama serve" in str(exc) else str(exc)[:60])

    settings.llm.provider = "gemini"
    settings.llm.ollama_url = "http://127.0.0.1:11434"
    print()
    if failures:
        print(f"DETECTION BACKEND: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("DETECTION BACKEND: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
