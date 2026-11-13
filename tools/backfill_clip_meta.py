#!/usr/bin/env python3
"""Backfill fps/duration/codec on existing clips without downloading media.

Run on the host that owns the bucket credentials:

    .venv/bin/python tools/backfill_clip_meta.py

Each offloaded clip is probed through a 5-minute presigned GET. ffprobe uses
HTTP range requests, so only the MP4 headers (and the decode probe) are read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, video  # noqa: E402
from app.store import LocalStore  # noqa: E402


def main() -> int:
    settings = config.load_config()
    store = LocalStore(settings.app.paths)
    out = video.backfill_clip_metadata(store, settings, limit=2000)
    print(json.dumps(out, indent=2))
    return 0 if out["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
