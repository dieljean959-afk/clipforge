#!/usr/bin/env python3
"""Fail if a LOCKED visual-contract file changed.

Usage:
    python tools/check_locked.py            # verify (tests + deploy)
    python tools/check_locked.py --update --confirm-unlock UNLOCK

The update form is intentionally hard to run by accident. It only accepts the
literal confirmation after the owner has said "UNLOCK" in chat. Never call it
as part of an automated test, deploy or repair.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "locked.sha256"
FILES = (
    "LOCKED_SPEC.md",
    "app/caption_presets.py",
    "app/caption_style.py",
    "app/captions_animated.py",
    "app/subtitle.py",
    "app/titlebar.py",
    "app/emoji.py",
    "app/encode.py",
    "app/video.py",
    "app/memguard.py",
    "src/components/clipmake/CaptionStyleDialog.tsx",
    "src/components/clipmake/NewJobForm.tsx",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    out: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            # Manifest is "sha256<TAB>path" (git-style hash first).
            out[parts[-1]] = parts[0]
    return out


def verify() -> int:
    expected = read_manifest()
    problems: list[str] = []
    for name in FILES:
        path = ROOT / name
        if not path.is_file():
            problems.append(f"MISSING  {name}")
            continue
        want = expected.get(name)
        got = digest(path)
        if want is None:
            problems.append(f"UNHASHED {name} (run --update only with approval)")
        elif want != got:
            problems.append(f"CHANGED  {name}\n  expected {want}\n  actual   {got}")
    for name in sorted(set(expected) - set(FILES)):
        problems.append(f"STALE    {name} is in locked.sha256 but not in FILES")
    if problems:
        print("LOCKED CONTRACT FAILED — do not deploy:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"LOCKED CONTRACT OK — {len(FILES)} files match locked.sha256")
    return 0


def update(confirm: str) -> int:
    if confirm != "UNLOCK":
        print(
            "Refusing to update locked hashes. The owner must say UNLOCK in chat, "
            "then run with --confirm-unlock UNLOCK.",
            file=sys.stderr,
        )
        return 2
    missing = [n for n in FILES if not (ROOT / n).is_file()]
    if missing:
        print(f"cannot hash missing files: {missing}", file=sys.stderr)
        return 2
    lines = [
        "# LOCKED visual contract — update only after the owner says UNLOCK.",
        "# Format: sha256<TAB>path",
    ]
    for name in FILES:
        lines.append(f"{digest(ROOT / name)}\t{name}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated locked.sha256 ({len(FILES)} files)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--confirm-unlock", default="")
    args = parser.parse_args()
    return update(args.confirm_unlock) if args.update else verify()


if __name__ == "__main__":
    raise SystemExit(main())
