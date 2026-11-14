"""Shared helpers for the LLM failover tests.

Deliberately mirrors `tools/nvidia_provider_test.py`: a real local HTTP server
speaks the OpenAI chat-completions dialect, because the engine calls a URL —
a fake transport object would only test the fake.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402


def _settings() -> Any:
    """A Settings with every provider pointed at a dead port and no pacing."""
    s = load_config(quiet=True)
    s.llm.provider = "deepseek"
    for cfg in (s.deepseek, s.nvidia, s.groq):
        cfg.base_url = "http://127.0.0.1:1/v1"
        cfg.min_interval_sec = 0.0
        cfg.max_retries = 0
        cfg.json_retries = 1
        cfg.timeout_sec = 5
        cfg.tokens_per_min = 0
    return s


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


class _Keys:
    """Context manager: set the provider keys, then put the env back."""

    def __init__(self, **values: str | None) -> None:
        self.wanted = values
        self.saved = {k: os.environ.get(k) for k in values}

    def __enter__(self) -> "_Keys":
        for name, value in self.wanted.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
