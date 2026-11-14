#!/usr/bin/env python3
"""Report whether the loaded config sees GEMINI_API_KEY (boolean only).

Never prints the value — safe to run anywhere. exit 0 = key present, exit 3 = key missing, exit 2 = key present but
rejected (empty envelope, encrypted blob, placeholder — reason printed).

Uses the LIVE resolver, the same one the API uses, so this reflects what a
running server would see after a Re-check (not a stale boot-time snapshot).

Run tools/keylength.py alongside this to distinguish "no value stored" from
"an EMPTY envelope stored" (the platform saved it with nothing inside).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import live_gemini_key, load_config  # noqa: E402


def main() -> int:
    load_config()
    live = live_gemini_key()
    live.refresh(force=True)
    status = live.status()

    print(f"GEMINI_API_KEY in process env : {bool(os.environ.get('GEMINI_API_KEY'))}")
    print(f"backend sees a usable key     : {status['keyLoaded']}")
    print(f"key source                    : {status['source']}")
    if status["problem"]:
        print(f"problem                       : {status['problem']}")
    if status["keyLoaded"]:
        key = live.key or ""
        # shape only — enough to spot a bad paste, never the secret
        print(f"key shape: {len(key)} chars, starts {key[:3]!r}, ends {key[-3:]!r}")
        return 0
    if status["problem"]:
        return 2
    print("\nNOT WIRED. Likely causes:")
    print("  1. Key saved under the Frontend tab — backend vars live under Backend.")
    print("  2. Key name typo — it must be exactly GEMINI_API_KEY (or GOOGLE_API_KEY).")
    print("  3. Value is a placeholder or still encrypted — re-save it, then hit")
    print("     Re-check on the dashboard (no restart needed with the live resolver).")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
