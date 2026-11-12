#!/usr/bin/env python3
"""Hard reset for the 4 GB sandbox. Dry-run by default.

    .venv/bin/python tools/disk_reset.py            # show exactly what would go
    .venv/bin/python tools/disk_reset.py --apply    # back up DB, then remove it

Never touches .env/keys, cookies.txt, the job documents or backups, the venv,
node_modules, app code/assets/fonts/emoji, a model in use, a running batch, or
any local media whose bucket HEAD size does not match. The same function backs
`POST /api/storage/reset`, so the UI cannot delete more than this CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="list what would be removed (this is the default)",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="back up the job DB, then actually remove the listed debris",
    )
    args = ap.parse_args()
    if args.apply and args.dry_run:
        ap.error("choose --apply or --dry-run, not both")

    from app import diskreset
    from app.config import load_config
    from app.store import LocalStore

    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)
    out = diskreset.reset(store, settings, apply=args.apply)
    print(json.dumps(out, indent=2, default=str))
    print(
        f"{'would free' if out['dryRun'] else 'freed'} {out['freedMb']} MB — "
        f"disk {out['freeBeforeMb']} → {out['freeAfterMb']} MB free"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
