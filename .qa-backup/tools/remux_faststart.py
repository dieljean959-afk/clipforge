"""Remux already-rendered clips so their `moov` atom sits at the front.

New renders carry `-movflags +faststart`, but every clip rendered BEFORE that
fix is still a "moov at the end" file. Such a clip downloads fine and refuses
to play in a browser: the player has to pull the whole file before it learns
where any frame is, which over a phone connection times out ("This browser
could not play the rendered file") or finishes so late that the duration never
arrives ("0:00 / --:--").

Remuxing with `-c copy` moves the atom without re-encoding, so it costs a few
seconds and no quality. Nothing is re-encoded; only the container layout
changes.

    python tools/remux_faststart.py            # dry run, reports what it would do
    python tools/remux_faststart.py --apply    # rewrite the files in place
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ffmpeg_bin  # noqa: E402
from app.store import LocalStore  # noqa: E402

CHUNK = 1 << 20


def moov_is_at_front(path: Path) -> bool:
    """True when the `moov` atom precedes `mdat` (i.e. faststart is present).

    Read as a stream rather than `read_bytes()`: a 4K clip is hundreds of MB
    and the two atoms are the only thing we need.
    """
    moov = -1
    mdat = -1
    offset = 0
    tail = b""
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            buf = tail + block
            base = offset - len(tail)
            if moov < 0:
                i = buf.find(b"moov")
                if i >= 0:
                    moov = base + i
            if mdat < 0:
                j = buf.find(b"mdat")
                if j >= 0:
                    mdat = base + j
            if moov >= 0 and mdat >= 0:
                break
            offset += len(block)
            tail = buf[-8:]
    if moov < 0:
        # No moov at all: not a normal MP4 (or a fragmented one). Treat as
        # "needs attention" rather than silently claiming it is fine.
        return False
    return mdat < 0 or moov < mdat


def remux(path: Path) -> tuple[bool, str]:
    """Rewrite `path` in place with `moov` first. Returns (ok, detail)."""
    tmp = path.with_name(path.stem + ".faststart.tmp.mp4")
    try:
        proc = subprocess.run(
            [
                ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-c", "copy",
                "-movflags", "+faststart",
                str(tmp),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            return False, (proc.stderr or "").strip()[-200:] or "ffmpeg failed"
        if not tmp.exists() or tmp.stat().st_size == 0:
            return False, "ffmpeg produced no output"
        # Only replace the original once the new file is verifiably playable.
        if not moov_is_at_front(tmp):
            return False, "remux did not move moov to the front"
        shutil.move(str(tmp), str(path))
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually rewrite files")
    args = ap.parse_args()

    from app.config import load_config

    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)

    clips = store.list_all_clips(limit=10_000)
    seen: set[str] = set()
    checked = needs = fixed = failed = missing = 0

    for clip in clips:
        vp = clip.get("videoPath")
        if not vp:
            continue
        p = Path(str(vp))
        if str(p) in seen:
            continue
        seen.add(str(p))
        if not p.exists():
            missing += 1
            print(f"  MISSING  {p.name} — the file is gone but the clip still points at it")
            continue
        checked += 1
        if moov_is_at_front(p):
            continue
        needs += 1
        if not args.apply:
            print(f"  WOULD FIX  {p.name} ({p.stat().st_size // 1024} KB, moov at end)")
            continue
        ok, detail = remux(p)
        if ok:
            fixed += 1
            print(f"  FIXED      {p.name}")
        else:
            failed += 1
            print(f"  FAILED     {p.name}: {detail}")

    mode = "applied" if args.apply else "dry run"
    print(
        f"\n{mode}: {checked} clip file(s) checked, {needs} needed faststart, "
        f"{fixed} fixed, {failed} failed, {missing} missing"
    )
    if not args.apply and needs:
        print("re-run with --apply to rewrite them")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
