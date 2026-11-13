#!/usr/bin/env python3
"""Scoped reset for the 4 GB sandbox and its bucket. Dry-run by default.

    .venv/bin/python tools/disk_reset.py --scope local
    .venv/bin/python tools/disk_reset.py --scope bucket
    .venv/bin/python tools/disk_reset.py --scope all --apply

`local` never touches .env/keys, cookies.txt, the job documents or backups,
the venv, node_modules, app code/assets/fonts/emoji, a model in use, a running
batch, or any local media whose bucket HEAD size does not match. `bucket`
backs up job/clip documents first and only removes objects a live batch does
not reference (plus stale multipart uploads). The same functions back
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
    ap.add_argument("--scope", choices=("local", "bucket", "all"), default="local")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="list what would be removed (this is the default)",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="back up the DB, then actually remove the listed items",
    )
    args = ap.parse_args()
    if args.apply and args.dry_run:
        ap.error("choose --apply or --dry-run, not both")

    from app import bucketreset, diskreset
    from app.config import load_config
    from app.store import LocalStore

    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)
    out: dict = {"scope": args.scope, "dryRun": not args.apply}

    if args.scope in ("local", "all"):
        local = diskreset.reset(store, settings, apply=args.apply)
        out["local"] = local
        if args.scope == "local":
            out = {"scope": "local", **local}
    if args.scope in ("bucket", "all"):
        planned = bucketreset.plan(store, settings)
        bucket = bucketreset.apply(store, settings, planned) if args.apply else planned
        bucket.pop("_keys", None)
        bucket.pop("_multipart", None)
        out["bucket"] = bucket
        if args.scope == "bucket":
            out = bucket

    print(json.dumps(out, indent=2, default=str))
    if args.scope == "local" or args.scope == "all":
        local = out if args.scope == "local" else (out.get("local") or {})
        print(
            f"{'would free' if not args.apply else 'freed'} "
            f"{local.get('freedMb', 0)} MB local — disk "
            f"{local.get('freeBeforeMb')} → {local.get('freeAfterMb')} MB free"
        )
    if args.scope == "bucket" or args.scope == "all":
        b = out if args.scope == "bucket" else (out.get("bucket") or {})
        print(
            f"{'would free' if not args.apply else 'freed'} "
            f"{b.get('freedMb', b.get('plannedMb', 0))} MB bucket — "
            f"{b.get('usageBeforeMb')} → {b.get('usageAfterMb', b.get('usageBeforeMb'))} MB used"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
