#!/usr/bin/env python3
"""Check DeepSeek/Groq reachability without ever printing a key.

    .venv/bin/python tools/check_providers.py
    .venv/bin/python tools/check_providers.py --audio data/logs/smoke.wav

The detection check is one real 1-token call per provider.  The optional audio
check runs the Groq Whisper fallback on a short local file; no source media is
generated here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import asr_fallback, groq_asr, llm  # noqa: E402
from app.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=None)
    args = ap.parse_args(argv)
    settings = load_config(quiet=True)
    failed = False
    for provider in ("deepseek", "groq"):
        key = getattr(settings, f"{provider}_api_key", None)
        print(f"{provider}: key {'set' if key else 'not set'}")
        if not key:
            print(f"{provider}: BLOCKED (key missing)")
            failed = True
            continue
        result = llm.probe_provider(provider, settings)
        print(
            f"{provider}: {result.get('class')} "
            f"({result.get('latencyMs')} ms"
            + (f", model {result.get('model')}" if result.get("model") else "")
            + ")"
        )
        failed = failed or not result.get("ok")
    if args.audio:
        if not args.audio.is_file():
            print(f"audio: BLOCKED (not found: {args.audio})")
            return 2
        if not asr_fallback.enabled(settings):
            print("audio: BLOCKED (Groq fallback off or key missing)")
            failed = True
        else:
            cfg = groq_asr.config_from_settings(settings)
            print(f"audio: model {cfg.model or 'none'}")
            try:
                transcript = asr_fallback.transcribe(
                    args.audio, settings, ROOT / "data" / "logs" / "provider_smoke"
                )
                print(
                    f"audio: ok, {len(transcript.words)} words, "
                    f"{transcript.duration:.1f}s, {transcript.rtf} RTF"
                )
            except Exception as exc:  # noqa: BLE001 - a smoke result, not a crash
                print(f"audio: failed ({type(exc).__name__}: {asr_fallback.reason_line(exc)})")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
