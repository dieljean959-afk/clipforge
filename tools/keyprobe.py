#!/usr/bin/env python3
"""Read-only probe: does the live resolver find a usable key from env files?

Prints only variable NAMES and status — never a value.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (  # noqa: E402
    _decrypt_encrypted_values,
    _fresh_env_snapshot,
    _load_env_files,
    _resolve_gemini_key,
)
from app.config import ROOT  # noqa: E402


def main() -> int:
    fingerprint, file_values = _fresh_env_snapshot(ROOT)
    print("snapshot fingerprint:", fingerprint or "(empty)")
    print("file snapshot var names:", sorted(file_values))
    _load_env_files(ROOT)
    _decrypt_encrypted_values(os.environ)
    key, src, prob = _resolve_gemini_key(file_values)
    print("resolve from file snapshot:", "FOUND" if key else "not found", "| source:", src or "-", "| problem:", prob or "-")
    key2, src2, prob2 = _resolve_gemini_key(os.environ)
    print("resolve from process env:", "FOUND" if key2 else "not found", "| source:", src2 or "-", "| problem:", prob2 or "-")

    # the delete-flow bug: after DELETE, the process env still carries the old
    # key (load_dotenv never unsets), so a stale key keeps "working".
    leftovers = [n for n in os.environ if n in _resolve_gemini_key.__globals__["GEMINI_KEY_ALIASES"]]
    print("gemini aliases in process env:", leftovers or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
