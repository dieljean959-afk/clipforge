"""End-to-end check: does a REAL detection chunk now return usable JSON?

The unit tests prove the ladder behaves; this proves the actual fix works
against the live Groq API on a real chunk of the 54-minute source — the call
that used to die with `json_validate_failed` at chunk 3/7.

Reports the full ladder outcome: which mode, how many attempts, whether the
reply parsed, and what came back.

    python tools/live_detection_check.py [chunk_index]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm as llm_mod  # noqa: E402
from app.config import load_config  # noqa: E402

TRANSCRIPT = Path("data/work/8d2dd9608be0/transcribe/words.json")


def main() -> int:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    settings = load_config(quiet=True)
    data = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    chunks = llm_mod.chunk_words(data["words"])
    start_index, chunk = chunks[idx]
    prompt = llm_mod.build_prompt(chunk, settings.gemini, start_index=start_index)
    print(f"chunk {idx + 1}/{len(chunks)}: {len(chunk)} words, prompt {len(prompt)} chars")
    print(f"model={settings.groq.model} mode={settings.groq.response_format} "
          f"temp={settings.groq.temperature} max_tokens={settings.groq.max_tokens}")
    print(f"fallbacks={list(settings.groq.fallback_models)}\n")

    # Count ladder steps so the report says how much it cost to get there.
    original = llm_mod._groq_single
    attempts: list[str] = []

    def counting(prompt, cfg, model, key, mode, temperature, max_tokens):
        attempts.append(f"{model.split('/')[-1]}/{mode}/mt{max_tokens}")
        return original(prompt, cfg, model, key, mode, temperature, max_tokens)

    llm_mod._groq_single = counting  # type: ignore[assignment]
    t0 = time.monotonic()
    try:
        text = llm_mod.call_groq(prompt, settings)
    except llm_mod.LlmError as exc:
        print(f"FAILED after {len(attempts)} attempt(s): {str(exc)[:300]}")
        print(f"ladder: {' -> '.join(attempts)}")
        return 1
    finally:
        llm_mod._groq_single = original  # type: ignore[assignment]
    dt = time.monotonic() - t0

    parsed = llm_mod.parse_candidates(text)
    print(f"OK in {dt:.1f}s after {len(attempts)} attempt(s)")
    print(f"ladder: {' -> '.join(attempts) if attempts else 'succeeded first try'}")
    print(f"reply head: {text[:110]!r}")
    print(f"parsed candidates: {len(parsed)}")
    for c in parsed[:4]:
        print(f"  [{c['score']:.0f}] words {c['start_word']}-{c['end_word']}  {c['title'][:52]}")
    if parsed:
        resolved = llm_mod.resolve_candidates(parsed, data["words"], start_index, settings.gemini)
        print(f"resolved to {len(resolved)} timed clip(s)")
        for r in resolved[:4]:
            print(f"  {r['start']:.1f}s -> {r['end']:.1f}s  ({r['end'] - r['start']:.1f}s)")
    ok = len(parsed) > 0 and len(resolved) > 0
    print(f"\n{'PASS' if ok else 'FAIL'}: chunk {idx + 1} produced usable clips")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
