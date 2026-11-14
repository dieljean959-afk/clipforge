#!/usr/bin/env python3
"""Identify the encoding scheme of GEMINI_API_KEY — structural facts only.

Tries common envelope schemes (dotenvx enc:, encrypted:, whole-value base64,
JSON envelopes) and prints only field names / byte lengths. Never prints the
key, its decoded bytes, or any secret substring.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def b64(s: str) -> bytes | None:
    try:
        return base64.b64decode(s + "=" * (-len(s) % 4), validate=True)
    except Exception:
        return None


def main() -> int:
    for name in (".env", ".env.local", ".env.keys"):
        p = ROOT / name
        if p.exists():
            load_dotenv(p, override=False)

    raw = os.environ.get("GEMINI_API_KEY", "")
    if not raw:
        print("GEMINI_API_KEY not present")
        return 3

    print(f"total length: {len(raw)}")
    print(f"charset: ascii-only={raw.isascii()} alnum+eq/slash/plus={all(c.isalnum() or c in '+/=' for c in raw)}")
    print(f"has colon: {':' in raw}  has '=': {'=' in raw}  starts {{: {raw.startswith('{')}")

    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                print("JSON envelope with fields:")
                for k, v in obj.items():
                    print(f"  - {k}: {len(str(v))} chars")
                return 0
        except Exception:
            print("starts with { but is not valid JSON")

    decoded = b64(raw)
    if decoded is not None:
        printable = sum(32 <= c < 127 for c in decoded) / max(1, len(decoded))
        print(f"decodes as base64: yes ({len(decoded)} bytes, printable fraction {printable:.2f})")
        if printable > 0.9:
            try:
                txt = decoded.decode("ascii")
                if txt.startswith("{"):
                    obj2 = json.loads(txt)
                    if isinstance(obj2, dict):
                        print("base64 wraps a JSON envelope with fields:")
                        for k, v in obj2.items():
                            print(f"  - {k}: {len(str(v))} chars")
                        return 0
                print(f"decoded text is printable ascii, length {len(txt)}")
            except Exception:
                print("decoded bytes are not ascii text")
        else:
            print("decoded bytes look binary (encrypted blob), not text")
    else:
        print("does not decode as base64")

    for prefix in ("enc:", "encrypted:", "vault:", "secret:"):
        if raw.startswith(prefix):
            print(f"prefix found: {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
