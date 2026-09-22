#!/usr/bin/env python3
"""Compile-check every backend module without importing heavy deps."""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

failed = 0
for p in sorted(Path("app").glob("*.py")):
    try:
        py_compile.compile(str(p), doraise=True)
        print(f"OK   {p}")
    except Exception as exc:
        failed += 1
        print(f"FAIL {p}: {exc}")
sys.exit(1 if failed else 0)
