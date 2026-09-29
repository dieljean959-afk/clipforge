"""Is the live Groq API usable right now, and what does it cost per call?

The benchmark cannot be designed blind. This makes ONE tiny call per candidate
model and reports the real status, the real token cost, and whether strict
json_schema is accepted — the three facts that decide how many runs we can
afford and which response_format modes are worth benchmarking.

Reads the key through the app's own config resolver (the raw .env is not
readable from a tool by design).

    python tools/groq_live_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.llm import GROQ_USER_AGENT  # noqa: E402

CANDIDATES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

PROMPT = (
    "Select the best moment from this numbered transcript. "
    'Reply ONLY with a JSON array: [{"start_word": 0, "end_word": 3, '
    '"score": 80, "title": "t"}]\n\n[0] 0.00s hello\n[1] 0.50s world\n'
)

SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_word": {"type": "integer"},
                    "end_word": {"type": "integer"},
                    "score": {"type": "number"},
                    "title": {"type": "string"},
                },
                "required": ["start_word", "end_word", "score", "title"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def post(base: str, key: str, body: dict, timeout: int = 60) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": GROQ_USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return 200, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            return exc.code, {}
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": {"message": f"{type(exc).__name__}: {exc}"}}


def main() -> int:
    settings = load_config()
    key = os.environ.get("GROQ_API_KEY", "") or settings.groq_api_key
    if not key:
        print("NO KEY — cannot probe. Add GROQ_API_KEY in the Keys/API keys tab.")
        return 1
    base = settings.groq.base_url
    print(f"base={base}\n")

    modes = {
        "json_object": {"type": "json_object"},
        "json_schema_strict": {
            "type": "json_schema",
            "json_schema": {
                "name": "clips",
                "strict": True,
                "schema": SCHEMA,
            },
        },
        "plain": None,
    }

    for model in CANDIDATES:
        for name, fmt in modes.items():
            body: dict = {
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "temperature": 0.0,
                "max_tokens": 300,
            }
            if fmt:
                body["response_format"] = fmt
            t0 = time.monotonic()
            status, data = post(base, key, body)
            dt = time.monotonic() - t0
            usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
            if status == 200:
                text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                print(
                    f"  {model:26s} {name:20s} 200 {dt:5.1f}s "
                    f"tok={usage.get('total_tokens')} out={text[:60]!r}"
                )
            else:
                err = data.get("error") if isinstance(data, dict) else {}
                msg = (err or {}).get("message", str(data))[:200]
                code = (err or {}).get("code", "")
                print(f"  {model:26s} {name:20s} {status} code={code} {msg}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
