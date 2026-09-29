"""WHY does gpt-oss fail json_object? Is it the top-level ARRAY?

Hypothesis from the live probe: the detection prompt asks for a top-level JSON
ARRAY (`[{...}]`). Groq's json_object mode requires a top-level JSON OBJECT, so
every response-format call is rejected before the model even gets a vote —
whichever model and whichever mode. If true, wrapping the payload as
{"clips": [...]} makes the same prompt succeed, which is a far better fix than
retrying a request that can never pass.

    python tools/groq_shape_probe.py
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

TAIL = "\n\n[0] 0.00s hello\n[1] 0.50s world\n[2] 1.00s this\n[3] 1.50s changed\n"

# The three shapes, identical apart from what the prompt says the top level is.
SHAPES = {
    "array_toplevel": 'Reply ONLY with a JSON array: [{"start_word": 0, "end_word": 3, "score": 80, "title": "t"}]',
    "object_toplevel": 'Reply ONLY with a JSON object of the form {"clips": [{"start_word": 0, "end_word": 3, "score": 80, "title": "t"}]}',
    "object_no_hint": 'Reply ONLY with JSON.',
}

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
        print("NO KEY")
        return 1
    base = settings.groq.base_url

    modes = {
        "json_object": lambda: {"type": "json_object"},
        "strict_schema": lambda: {
            "type": "json_schema",
            "json_schema": {"name": "clips", "strict": True, "schema": SCHEMA},
        },
        "plain": lambda: None,
    }

    for model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"):
        print(f"== {model}")
        for shape, instruction in SHAPES.items():
            row = []
            for mname, mfn in modes.items():
                body: dict = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Select the best moment from this numbered transcript. "
                            + instruction
                            + TAIL,
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 300,
                }
                fmt = mfn()
                if fmt:
                    body["response_format"] = fmt
                t0 = time.monotonic()
                status, data = post(base, key, body)
                dt = time.monotonic() - t0
                if status == 200:
                    text = ((data.get("choices") or [{}])[0].get("message") or {}).get(
                        "content", ""
                    )
                    row.append(f"{mname}=OK({dt:.1f}s,{text[:26]!r})")
                else:
                    err = (data.get("error") or {}) if isinstance(data, dict) else {}
                    code = err.get("code", "")
                    msg = str(err.get("message", ""))[:70]
                    row.append(f"{mname}={status}/{code}({msg})")
            print(f"  {shape:16s} " + " | ".join(row))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
