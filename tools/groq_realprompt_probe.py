"""Does the REAL detection prompt get coerced from an array into one object?

`json_object` mode forces a top-level JSON *object*. The current prompt asks
for a top-level *array*, so gpt-oss/qwen may satisfy the mode by returning a
SINGLE clip object instead of a list — which `parse_candidates` (which greps
for `[...]`) cannot read at all. That would be a silent correctness bug behind
the loud 400s, so it is worth proving on a real chunk before designing around it.

    python tools/groq_realprompt_probe.py
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
from app.llm import GROQ_USER_AGENT, build_prompt, chunk_words  # noqa: E402

TRANSCRIPT = Path("data/work/8d2dd9608be0/transcribe/words.json")


def post(base: str, key: str, body: dict, timeout: int = 120) -> tuple[int, dict]:
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
    data = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    words = data["words"]
    chunks = chunk_words(words)
    start_index, chunk = chunks[2]
    prompt = build_prompt(chunk, settings.gemini, start_index=start_index)
    print(f"chunk 3/{len(chunks)}: {len(chunk)} words, prompt {len(prompt)} chars\n")

    for model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
        for mode, fmt in (
            ("json_object", {"type": "json_object"}),
            ("plain", None),
        ):
            body: dict = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a video-clipping assistant. Reply with a "
                        "single JSON object and nothing else — no prose, no markdown fence.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": settings.groq.temperature,
                "max_tokens": 1600,
            }
            if fmt:
                body["response_format"] = fmt
            status, resp = post(settings.groq.base_url, key, body)
            if status != 200:
                err = resp.get("error") or {}
                print(f"  {model:22s} {mode:12s} {status} {err.get('code','')} "
                      f"{str(err.get('message',''))[:80]}")
                continue
            text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            usage = (resp.get("usage") or {})
            shape = "ARRAY" if text.lstrip().startswith("[") else (
                "OBJECT" if text.lstrip().startswith("{") else "OTHER"
            )
            n = None
            if shape == "ARRAY":
                try:
                    n = len(json.loads(text))
                except Exception:  # noqa: BLE001
                    shape = "ARRAY(unparseable)"
            print(
                f"  {model:22s} {mode:12s} 200 tok={usage.get('total_tokens')} "
                f"finish={(resp.get('choices') or [{}])[0].get('finish_reason')} "
                f"top={shape} n={n} head={text[:70]!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
