#!/usr/bin/env python3
"""Nightly disk cleanup: `app.disksweep.sweep`, logged, with a dry-run mode.

    .venv/bin/python tools/disk_cleanup.py --dry-run
    .venv/bin/python tools/disk_cleanup.py

The buttons in the panel call the same function through
`POST /api/storage/reclaim`, so a cron entry can never be "the thing that
deletes more than the button does".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list what would go, delete nothing")
    ap.add_argument("--include-models", action="store_true", default=True)
    ap.add_argument("--no-models", dest="include_models", action="store_false")
    args = ap.parse_args()

    from app import disksweep
    from app.config import load_config
    from app.store import LocalStore

    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)
    out = disksweep.sweep(
        store, settings.app.paths, dry_run=args.dry_run, include_models=args.include_models
    )
    print(json.dumps(out, indent=2))
    print(
        f"{'would free' if args.dry_run else 'freed'} {out['freedMb']} MB — "
        f"disk {out['freeBeforeMb']} → {out['freeAfterMb']} MB free"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
