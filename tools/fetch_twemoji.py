"""Download the bundled Twemoji PNGs the emoji whitelist references.

Run ONCE, at build time — the renderer never touches the network. Reads
``app/assets/emoji/whitelist.json`` and fetches only the codepoints it lists
from the pinned Twemoji release, so the assets on disk always match the
whitelist (and a whitelist entry with no PNG still fails loudly in code).

    python3 tools/fetch_twemoji.py [--force]
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMOJI_DIR = ROOT / "app" / "assets" / "emoji"
WHITELIST = EMOJI_DIR / "whitelist.json"
BASE = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.0.3/assets/72x72"


def main(argv: list[str]) -> int:
    force = "--force" in argv
    data = json.loads(WHITELIST.read_text(encoding="utf-8"))
    missing = 0
    for item in data["emojis"]:
        dest = EMOJI_DIR / item["file"]
        if dest.exists() and not force:
            continue
        url = f"{BASE}/{item['file']}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                blob = resp.read()
        except Exception as exc:  # noqa: BLE001 - report, do not skip silently
            print(f"FAILED {item['emoji']} {url}: {exc}")
            missing += 1
            continue
        dest.write_bytes(blob)
        print(f"{item['emoji']} {item['file']} {len(blob)} bytes")
    print(f"{len(data['emojis'])} whitelisted, {missing} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
