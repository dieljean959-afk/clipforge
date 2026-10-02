#!/usr/bin/env python3
"""Structural probe: how many bytes did the platform actually encrypt?

Prints ONLY lengths and booleans — never values. Safe to run anywhere.

blob = base64( ephemeral_pubkey[65] || nonce[16] || tag[16] || ciphertext )
plaintext length == ciphertext length, so:

    plaintext_bytes = decoded_len - 97

A result of 0 means the platform saved an EMPTY value under that variable
name — the encrypted envelope exists, but there is no key inside it.
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    for name in (".env", ".env.local", ".env.keys"):
        p = ROOT / name
        if p.exists():
            load_dotenv(p, override=False)

    raw = os.environ.get("GEMINI_API_KEY", "")
    print("GEMINI_API_KEY blob present:", bool(raw), "len:", len(raw))
    if not raw.startswith("encrypted:"):
        print("not an encrypted: blob — nothing structural to report")
        return 0

    body = raw[len("encrypted:"):]
    try:
        decoded = base64.b64decode(body + "=" * (-len(body) % 4))
    except Exception as exc:
        print(f"base64 decode failed: {type(exc).__name__}")
        return 1

    print("decoded bytes:", len(decoded))
    if len(decoded) >= 97:
        print("ciphertext bytes:", len(decoded) - 97)
        print("=> plaintext length stored by the platform:", len(decoded) - 97)
        if len(decoded) == 97:
            print("CONCLUSION: the stored value is EMPTY — the platform saved an")
            print("encrypted envelope with no key inside. Re-save the key in the")
            print("Keys/API keys tab (paste the full AIza... value), then Re-check.")
    else:
        print("blob shorter than the 97-byte header — unexpected layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
