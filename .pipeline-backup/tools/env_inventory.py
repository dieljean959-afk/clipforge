#!/usr/bin/env python3
"""Names-only env inventory — prints variable NAMES and byte counts, never values.

Used to diagnose "key not detected" states without ever exposing a secret:
which env files exist, which variables they define, duplicates, and which
Gemini key aliases are present in the current process environment.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEMINI_ALIASES = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "GEMINI_KEY",
    "VITE_GEMINI_API_KEY",
)

STALE_NAMES = (
    "OCI_FINGERPRINT",
    "OCI_INSTANCE_OCID",
    "OCI_PRIVATE_KEY",
    "OCI_REGION",
    "OCI_TENANCY_OCID",
    "OCI_USER_OCID",
    "OCI_COMPARTMENT_OCID",
)


def main() -> int:
    print("== env files ==")
    for name in (".env", ".env.local", ".env.keys"):
        p = ROOT / name
        if not p.exists():
            print(f"{name}: MISSING")
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"{name}: unreadable ({exc})")
            continue
        names = [m.group(1) for m in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", text, re.M)]
        dupes = sorted({n for n in names if names.count(n) > 1})
        print(f"{name}: {len(text)} bytes, {len(names)} var(s): {names}")
        if dupes:
            print(f"  !! duplicate definitions: {dupes}")

    print("\n== process env (names only) ==")
    gemini_present = [n for n in GEMINI_ALIASES if (os.environ.get(n) or "").strip()]
    print(f"Gemini aliases present: {gemini_present or 'NONE'}")
    stale = [n for n in STALE_NAMES if n in os.environ]
    print(f"Stale OCI vars present: {stale or 'none'}")
    dotenv_priv = [k for k in os.environ if k.startswith("DOTENV_PRIVATE_KEY_")]
    print(f"dotenvx private keys in process env: {dotenv_priv or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
