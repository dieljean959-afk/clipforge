"""Pre-scale the bundled Twemoji PNGs to the size the renderer blits.

Twemoji ships 72x72. The caption overlay draws at 140 px, so on a 1 CPU host
every clip used to pay ffmpeg's scaler for every frame of every emoji window.
Scaling once, here, means the render is a 1:1 blit.

Run once at build time (never at render time):

    python3 tools/scale_twemoji.py [--size 144] [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMOJI_DIR = ROOT / "app" / "assets" / "emoji"
WHITELIST = EMOJI_DIR / "whitelist.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=144)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from PIL import Image

    out_dir = EMOJI_DIR / str(args.size)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = json.loads(WHITELIST.read_text(encoding="utf-8"))["emojis"]
    written = skipped = missing = 0
    for item in items:
        src = EMOJI_DIR / item["file"]
        if not src.exists():
            missing += 1
            print(f"MISSING {item['emoji']} {src.name}")
            continue
        dest = out_dir / item["file"]
        if dest.exists() and not args.force:
            skipped += 1
            continue
        with Image.open(src) as im:
            im = im.convert("RGBA").resize((args.size, args.size), Image.LANCZOS)
            im.save(dest, "PNG", optimize=True)
        written += 1
    print(
        f"{len(items)} whitelisted | written {written} | reused {skipped} "
        f"| missing {missing} | dir {out_dir} @ {args.size}px"
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
