#!/usr/bin/env python3
"""Prepare the configured Whisper model while the app is idle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import modelstore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    args = ap.parse_args(argv)
    result = modelstore.prepare(args.model, log=lambda line: print(line, file=sys.stderr))
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
