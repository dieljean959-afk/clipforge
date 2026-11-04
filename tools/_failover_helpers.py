"""Shared helpers for the LLM failover/fail-key tests."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

sys_path = str(Path(__file__).resolve().parents[1])
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from app import llm as llm_mod  # noqa: E402
from app.config import load_config  # noqa: E402


def _settings() -> object:
    s = load_config(quiet=True)
    s.llm.provider = "deepseek"
    for cfg in (s.deepseek, s.nvidia, s.groq):
        cfg.base_url = "http://127.0.0.1:1/v1"
        cfg.min_interval_sec = 0.0
        cfg.max_retries = 0
        cfg.json_retries = 1
        cfg.timeout_sec = 10
    for cfg in (s.deepseek, s.nvidia, s.groq):
        cfg.tokens_per_min = 0
    return s


class _Stub:
    """Scripted OpenAI-compatible endpoint. `handler` sees each request body."""

    class _Handler(BaseHTTPRequestHandler):
        def __init__(self, *args: Any, handler: Callable, **kwargs: Any) -> None:
            self._handler = handler
            super().__init__(*args, **kwargs)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, payload, headers = self._handler(self, body)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if payload:
                self.wfile.write(payload)

        def log_message(self, *_a: object, **_k: object) -> None:  # noqa: D102
            pass

    def __init__(self, handler: Callable) -> None:
        self.handler = handler
        self.requests = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> "_Stub":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._Handler, bind_and_activate=False)
        self._server.server_address = ("127.0.0.1", 0)
        self._server.timeout = 5
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self._server = None
        self._thread = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def request(self, method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object], dict[str, str]]:
        self.requests.append({"method": method, "path": path, "headers": dict(headers), "body": body.decode("utf-8")})
        return self.handler(method, path, headers, body)

    def get_request(self) -> dict[str, object]:
        with self._lock:
            if self.requests:
                return self.requests.pop(0)
            raise IndexError("no more requests")


class _keys:
    """Context manager: set DEEPSEEK_API_KEY (and the aliases) for the block."""

    def __init__(self, key: str | None, *aliases: str | None) -> None:
        self.env = {"DEEPSEEK_API_KEY": key}
        for a in aliases:
            if a is not None:
                self.env[a] = a
        self.saved = {k: os.environ.get(k) for k in self.env}

    def __enter__(self) -> dict[str, str | None]:
        import os

        for k, v in self.env.items():
            os.environ[k] = v  # type: ignore[assignment]
        return self.saved

    def __exit__(self, *exc: object) -> None:
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _reply(candidates: list[dict], usage_tokens: int = 1000) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps({"clips": candidates})}}],
        "usage": {"total_tokens": usage_tokens},
    }


def _ok(payload: dict) -> callable:
    return lambda _body: (200, payload)


_ok_groq = _ok


def _candidate(start: int = 0, end: int = 200, score: float = 88.0) -> dict:
    return {
        "start_word": start,
        "end_word": end,
        "score": score,
        "title": "A punchy title",
        "hook_text": "one line of post copy",
        "hashtags": ["#fyp", "#foryou", "#viral", "#topic"],
    }
