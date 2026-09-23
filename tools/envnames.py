#!/usr/bin/env python3
"""Probe which secret-related variable NAMES exist in the process environment.

Reads the RAW environment BEFORE any dotenv file loading, so we can tell what
the platform injects versus what comes from project env files. Prints names
and boolean flags only — never values.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    raw_keys = set(os.environ.keys())
    interesting = sorted(k for k in raw_keys if any(
        t in k.upper()
        for t in ("GEMINI", "DOTENV", "VLY", "SECRET", "TOKEN", "API_KEY")
    ))
    print("raw process env (before dotenv) — matching variable names:")
    for k in interesting:
        v = os.environ[k]
        print(f"  {k}: present, {len(v)} chars, encrypted-prefix={v.startswith('encrypted:')}")

    # also: which env FILES exist and which var NAMES they define (names only)
    try:
        for name in (".env", ".env.local", ".env.keys"):
            p = ROOT / name
            if not p.exists():
                print(f"{name}: absent")
                continue
            names = []
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    names.append(line.split("=", 1)[0].strip())
            print(f"{name}: defines {names}")
    except Exception as exc:
        print(f"env file name scan failed: {type(exc).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
