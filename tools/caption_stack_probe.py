"""Measure the burned-in bold-stack caption on a real frame.

Method: the caption is the difference between the rendered clip and the SAME
clip rendered without captions. Every row that differs is caption paint, which
is how the text is isolated from a busy scene where colour thresholds alone
pick up a yellow jacket.

    python3 tools/caption_stack_probe.py --plain plain.mp4 clip.mp4

Per frame it prints the two ink bands (outline included), the pitch between
them, and the widest line as a % of the frame width. The cap height is the ink
height minus the 6 px outline above and below; run --calibrate as well to print
the same measurement for a controlled render of the identical style block on a
flat background, which is where the cap height is read without the compression
noise of a real scene (charged edges on red/green glyphs).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

CHANGED = 20      # per-channel difference that counts as caption paint
GAP = 8           # rows this close belong to one line
MIN_BAND = 25     # shorter than this is noise, not a caption line
MAX_BAND = 70     # taller than this is not a caption line
PITCH_RANGE = (60, 110)
OUTLINE = 6       # style Outline (px at 1080 width)
TIMES = (1.0, 2.2, 3.5, 8.0, 16.5)


def _frame(path: Path, t: float, dest: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path),
         "-frames:v", "1", str(dest)],
        check=True,
    )
    return dest


def _diff_rows(a: Path, b: Path) -> tuple[list[int], int, int, Image.Image]:
    ia = Image.open(a).convert("RGB")
    ib = Image.open(b).convert("RGB").resize(ia.size)
    w, h = ia.size
    pa, pb = ia.load(), ib.load()
    rows = []
    for y in range(h):
        for x in range(w):
            ra, ga, ba = pa[x, y]
            rb, gb, bb = pb[x, y]
            if max(abs(ra - rb), abs(ga - gb), abs(ba - bb)) >= CHANGED:
                rows.append(y)
                break
    return rows, w, h, ia


def _bands(rows: list[int]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for y in rows:
        if out and y - out[-1][1] <= GAP:
            out[-1][1] = y
        else:
            out.append([y, y])
    return [(t, b) for t, b in out if MIN_BAND <= b - t + 1 <= MAX_BAND]


def _x_extent(image: Image.Image, plain: Image.Image, top: int, bottom: int) -> tuple[int, int]:
    w = image.size[0]
    pa, pb = image.load(), plain.load()
    left, right = w, -1
    for y in range(top, bottom + 1):
        for x in range(w):
            ra, ga, ba = pa[x, y]
            rb, gb, bb = pb[x, y]
            if max(abs(ra - rb), abs(ga - gb), abs(ba - bb)) >= CHANGED:
                left = min(left, x)
                right = max(right, x)
    return (left, right) if right >= 0 else (0, 0)


def _colour_of(image: Image.Image, plain: Image.Image, top: int, bottom: int) -> str:
    """Which caption colour this line is: yellow / green / red / white."""
    pa, pb = image.load(), plain.load()
    tally = {"yellow": 0, "green": 0, "red": 0, "white": 0}
    for y in range(top, bottom + 1):
        for x in range(0, image.size[0], 2):
            r, g, b = pa[x, y]
            pr, pg, pbb = pb[x, y]
            if max(abs(r - pr), abs(g - pg), abs(b - pbb)) < CHANGED:
                continue
            if r > 150 and g > 150 and b < 130:
                tally["yellow"] += 1
            elif g > 150 and r < 130 and b < 130:
                tally["green"] += 1
            elif r > 150 and g < 130 and b < 130:
                tally["red"] += 1
            elif r > 200 and g > 200 and b > 200:
                tally["white"] += 1
    return max(tally, key=tally.get) if any(tally.values()) else "?"


def measure(clip: Path, plain: Path, t: float, tmp: Path) -> dict | None:
    a = _frame(clip, t, tmp / "cap.png")
    b = _frame(plain, t, tmp / "plain.png")
    rows, w, h, img = _diff_rows(a, b)
    # The reference block lives at 1283-1413 of 1920 (67-74 %); anything above
    # that window is scene motion the two encodes scored differently, not text.
    bands = [band for band in _bands(rows) if h * 0.65 <= band[0] and band[1] <= h * 0.80]
    if len(bands) < 2:
        return None
    bands.sort()
    line1 = bands[0]
    line2 = next((band for band in bands[1:]
                  if PITCH_RANGE[0] <= band[0] - line1[0] <= PITCH_RANGE[1]), None)
    if line2 is None:
        return None
    plain_img = Image.open(b).convert("RGB")
    l1, r1 = _x_extent(img, plain_img, *line1)
    l2, r2 = _x_extent(img, plain_img, *line2)
    scale = 1080.0 / w
    ink1 = line1[1] - line1[0] + 1
    ink2 = line2[1] - line2[0] + 1
    return {
        "t": t, "w": w, "h": h, "scale": scale,
        "line1": line1, "line2": line2,
        "ink": (ink1, ink2),
        "cap": round((ink1 - 2 * OUTLINE) * scale, 1),
        "cap2": round((ink2 - 2 * OUTLINE) * scale, 1),
        "pitch": round((line2[0] - line1[0]) * scale, 1),
        "width1": round((r1 - l1 + 1) / w * 100, 1),
        "width2": round((r2 - l2 + 1) / w * 100, 1),
        "colour1": _colour_of(img, plain_img, *line1),
        "colour2": _colour_of(img, plain_img, *line2),
    }


def main(argv: list[str]) -> int:
    args = argv[1:]
    if "--plain" not in args:
        print(__doc__)
        return 2
    i = args.index("--plain")
    plain = Path(args[i + 1])
    clips = [Path(a) for a in args if a not in ("--plain", args[i + 1])]
    for clip in clips:
        print(f"\n{clip}  (plain = {plain.name})")
        print(f"{'t':>5}  {'line1 ink':>14} {'line2 ink':>14} {'cap1':>6} {'cap2':>6}"
              f" {'pitch':>6} {'w1%':>6} {'w2%':>6}  colours")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for t in TIMES:
                m = measure(clip, plain, t, tmp)
                if m is None:
                    print(f"{t:>5.2f}  no caption ink found")
                    continue
                print(f"{t:>5.2f}  {str(m['line1']):>14} {str(m['line2']):>14}"
                      f" {m['cap']:>6} {m['cap2']:>6} {m['pitch']:>6}"
                      f" {m['width1']:>6} {m['width2']:>6}"
                      f"  {m['colour1']}/{m['colour2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
