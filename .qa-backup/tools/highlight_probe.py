"""Prove the word highlight actually reaches the pixels.

`\\k` in the ASS file proves the tag was written; this proves libass honoured
it in the encoded video. The default style paints the spoken word in the accent
colour (#39FF14) over white text, so on the new side the column with the most
accent pixels must MOVE as the clip plays. On the old side every word is white
for the whole cue, so the accent pixels are only the static outline and stay
put.

Run:  .venv/bin/python tools/highlight_probe.py
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "outputs" / "caption_compare.mp4"
TIMES = (0.15, 0.45, 0.75, 1.05)  # all inside ONE caption of the new render
                                   # (0.00-1.20), so the movement can only be
                                   # the highlight — not a different caption
BAND = (690, 820)   # caption rows in the 540x960 half-frame
HALF = 540


def frame(t: float, side: str) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(VIDEO),
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
    ]
    png = subprocess.run(cmd, capture_output=True, check=True).stdout
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    x0 = 0 if side == "old" else HALF
    return img[BAND[0]:BAND[1], x0:x0 + HALF]


def accent_columns(band: np.ndarray) -> np.ndarray:
    r, g, b = (band[:, :, i].astype(int) for i in range(3))
    return ((g > 170) & (r < 130) & (b < 130)).sum(axis=0)


def white_pixels(band: np.ndarray) -> int:
    r, g, b = (band[:, :, i].astype(int) for i in range(3))
    return int(((r > 200) & (g > 200) & (b > 200)).sum())


def main() -> int:
    if not VIDEO.exists():
        print(f"missing {VIDEO} — run tools/caption_side_by_side.py first")
        return 1
    peaks: dict[str, list[int]] = {"old": [], "new": []}
    fill: dict[str, list[float]] = {"old": [], "new": []}
    print(f"{'t(s)':>6} {'side':>4} {'accent px':>10} {'peak column':>13} "
          f"{'white px':>10} {'accent/white':>14}")
    for t in TIMES:
        for side in ("old", "new"):
            band = frame(t, side)
            cols = accent_columns(band)
            white = white_pixels(band)
            peaks[side].append(int(cols.argmax()))
            fill[side].append(cols.sum() / max(1, white))
            print(
                f"{t:6.2f} {side:>4} {int(cols.sum()):>10} "
                f"{str(int(cols.argmax())) + '/' + str(HALF):>13} {white:>10} "
                f"{cols.sum() / max(1, white):>14.2f}"
            )

    old_span = max(peaks["old"]) - min(peaks["old"])
    new_span = max(peaks["new"]) - min(peaks["new"])
    old_fill = sum(fill["old"]) / len(fill["old"])
    new_fill = sum(fill["new"]) / len(fill["new"])
    print()
    print(f"old: accent peak moved {old_span} px; accent/white {old_fill:.2f} "
          f"(that green is only the outline)")
    print(f"new: accent peak moved {new_span} px; accent/white {new_fill:.2f} "
          f"(a filled word)")
    moved = new_span > 40
    filled = new_fill > old_fill * 2
    ok = moved and filled
    if ok:
        print("HIGHLIGHT MOVES WITH THE SPEECH, INSIDE ONE CAPTION")
    else:
        print("NO HIGHLIGHT MOVEMENT DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
