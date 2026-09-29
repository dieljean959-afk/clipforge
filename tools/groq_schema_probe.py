"""Does the strict clip schema actually get ACCEPTED on a real detection chunk?

The benchmark showed json_schema with 0 rate-limit hits and 100% failures,
which is a different error from the 429s every other mode hit. This prints the
actual response body so the schema can be corrected rather than guessed at.

    python tools/groq_schema_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.llm import CLIP_SCHEMA, GROQ_USER_AGENT, build_prompt, chunk_words  # noqa: E402

TRANSCRIPT = Path("data/work/8d2dd9608be0/transcribe/words.json")


def post(base: str, key: str, body: dict, timeout: int = 90) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": GROQ_USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return 200, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            return exc.code, {}
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": {"message": f"{type(exc).__name__}: {exc}"}}


def main() -> int:
    s = load_config(quiet=True)
    key = os.environ.get("GROQ_API_KEY", "") or s.groq_api_key
    if not key:
        print("no key")
        return 1
    data = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    chunks = chunk_words(data["words"])
    start_index, chunk = chunks[2]
    base_prompt = build_prompt(chunk, s.gemini, start_index=start_index)
    prompt = base_prompt.replace(
        "Return ONLY a JSON array (no prose, no markdown fences):",
        'Return ONLY a JSON object shaped {"clips": [...]} (no prose, no fences):',
    )

    # Try the schema as shipped, then with the natural-language fields dropped
    # from `required`, which is the other strict-mode requirement.
    relaxed = json.loads(json.dumps(CLIP_SCHEMA))
    item = relaxed["properties"]["clips"]["items"]
    item["required"] = ["start_word", "end_word", "score", "title"]
    for name in ("hook_text", "hashtags"):
        item["properties"].pop(name, None)

    variants = {
        "as-shipped (6 required)": CLIP_SCHEMA,
        "relaxed (4 required)": relaxed,
    }
    for model in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b"):
        for vname, schema in variants.items():
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content":
                        "You are a video-clipping assistant. Reply with a single JSON "
                        "object and nothing else."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 3200,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "clips", "strict": True, "schema": schema},
                },
            }
            status, resp = post(s.groq.base_url, key, body)
            if status == 200:
                text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                print(f"  {model:22s} {vname:24s} 200  {text[:100]!r}")
            else:
                err = resp.get("error") or {}
                print(f"  {model:22s} {vname:24s} {status} {err.get('code','')} "
                      f"{str(err.get('message',''))[:220]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
