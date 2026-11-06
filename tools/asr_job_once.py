"""Transcribe one file and print the result — a stand-in job for the restart drill."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import subtitle  # noqa: E402
from app.config import load_config  # noqa: E402


def main() -> int:
    media, cache = Path(sys.argv[1]), Path(sys.argv[2])
    settings = load_config(quiet=True)
    paths = replace(settings.app.paths, data_dir=cache.parent / "data")
    settings = replace(settings, app=replace(settings.app, paths=paths))
    t = subtitle.transcribe(media, settings, cache_dir=cache)
    print(f"words={len(t.words)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
