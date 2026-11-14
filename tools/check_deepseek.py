#!/usr/bin/env python3
"""Test the DeepSeek detection backend — without ever printing the key.

Run it on the host that has DEEPSEEK_API_KEY:

    .venv/bin/python tools/check_deepseek.py            # models + one real call
    .venv/bin/python tools/check_deepseek.py --models   # model list only
    .venv/bin/python tools/check_deepseek.py --timeout 1  # force a timeout

It answers the only three questions that matter when detection "doesn't run":

  1. is the model id in config.toml (`deepseek-flash`) actually served?
  2. does the key authenticate, and is there credit on the account?
  3. does one real call return valid JSON, and how long did it take?

Exit codes: 0 ok, 1 no key, 2 model missing, 3 auth, 4 balance, 5 rate limit,
6 timeout, 7 bad reply.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load() -> tuple[str, str, str]:
    """(key, base_url, model) — only ever returned, never printed."""
    try:
        from app.config import load_config  # noqa: PLC0415

        settings = load_config(quiet=True)
        model = str(getattr(settings.deepseek, "model", "") or "deepseek-flash")
        base = str(
            os.getenv("DEEPSEEK_BASE_URL")
            or getattr(settings.deepseek, "base_url", "")
            or "https://api.deepseek.com"
        ).rstrip("/")
    except Exception:  # noqa: BLE001 - the tool must work even if config is broken
        model, base = "deepseek-flash", "https://api.deepseek.com"
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    return key, base, model


def _request(url: str, key: str, *, body: dict | None, timeout: float):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8", "replace"), time.time() - started
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), time.time() - started
    except Exception as exc:  # noqa: BLE001 - returned as a labelled failure
        return 0, f"{type(exc).__name__}: {exc}", time.time() - started


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe the DeepSeek detection backend.")
    ap.add_argument("--models", action="store_true", help="only list the served model ids")
    ap.add_argument("--model", default="", help="override the model id to test")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-request timeout (s)")
    args = ap.parse_args()

    key, base, model = _load()
    model = args.model or model
    print(f"endpoint: {base}")
    print(f"model:    {model}")
    print(f"key:      {'present (' + str(len(key)) + ' chars, not printed)' if key else 'MISSING'}")
    if not key:
        print("BLOCKED: no DEEPSEEK_API_KEY in the environment")
        return 1

    status, text, dt = _request(f"{base}/models", key, body=None, timeout=args.timeout)
    if status != 200:
        print(f"GET /models -> HTTP {status} in {dt:.2f}s: {text[:200]}")
    else:
        ids = [m.get("id") for m in (json.loads(text).get("data") or [])]
        print(f"GET /models -> {status} in {dt:.2f}s | served: {ids}")
        if ids and model not in ids:
            print(f"BAD MODEL: {model!r} is not in the served list — set [deepseek] model in config.toml")
            return 2
    if args.models:
        return 0

    body = {
        "model": model,
        "messages": [{"role": "user", "content": 'Reply with exactly this json: {"ok": true}'}],
        "max_tokens": 16,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    status, text, dt = _request(f"{base}/chat/completions", key, body=body, timeout=args.timeout)
    print(f"POST /chat/completions -> HTTP {status} in {dt:.2f}s")
    if status == 0:
        print(f"TIMEOUT/NETWORK: {text[:200]}  (timeout {args.timeout:.0f}s)")
        return 6
    if status in (401, 403):
        print(f"AUTH: {text[:200]}")
        return 3
    if status == 402:
        print(f"BALANCE: {text[:200]} — top up platform.deepseek.com (the app now skips "
              f"this backend instead of retrying it per chunk)")
        return 4
    if status == 429:
        print(f"RATE LIMIT: {text[:200]}")
        return 5
    if status == 400 and "model" in text.lower():
        print(f"BAD MODEL: {text[:200]}")
        return 2
    if status != 200:
        print(f"ERROR: {text[:200]}")
        return 7
    try:
        payload = json.loads(text)
        reply = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "")
        json.loads(reply)
    except Exception as exc:  # noqa: BLE001
        print(f"BAD REPLY ({exc}): {text[:200]}")
        return 7
    print(f"OK: valid JSON in {dt:.2f}s | usage {payload.get('usage')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
