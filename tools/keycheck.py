#!/usr/bin/env python3
"""Report whether the loaded config sees GEMINI_API_KEY (boolean only).

Never prints the value — safe to run anywhere. Exit 0 = key present,
exit 3 = key missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402


def main() -> int:
    settings = load_config()
    in_env = "GEMINI_API_KEY" in os.environ
    in_cfg = settings.gemini_api_key is not None
    print(f"GEMINI_API_KEY in process env : {in_env}")
    print(f"backend config sees the key   : {in_cfg}")
    if in_cfg:
        key = settings.gemini_api_key or ""
        # shape only — enough to spot a bad paste, never the secret
        print(f"key shape: {len(key)} chars, starts {key[:3]!r}, ends {key[-3:]!r}")
        return 0
    print("\nNOT WIRED. Likely causes:")
    print("  1. Key saved under the Frontend tab — backend vars live under Backend.")
    print("  2. Web process started before the key was saved — restart/redeploy the app.")
    print("  3. Key name typo — it must be exactly GEMINI_API_KEY.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
