#!/usr/bin/env python3
"""Generate the Clipmake PWA icon set.

One drawing, four files, because a single PNG cannot be both a maskable icon and
an Apple touch icon:

  icon-192.png            rounded black tile, art at 78% (home-screen "any")
  icon-512.png            same at 512
  icon-512-maskable.png   FULL-BLEED black square, art at 70% so it survives the
                          circular/squircle mask Android applies (the maskable
                          safe zone is a circle of 80% diameter — art outside it
                          gets clipped, which is why a rounded tile is wrong here)
  apple-touch-icon.png    180x180 full-bleed black square; iOS applies its own
                          rounding and rejects transparency
  favicon.ico             16/32 derived from the same art

Art: neon-green (#39ff14) phone outline, a green play triangle in the screen and
a small home-indicator bar at the bottom.

Everything is drawn at 4x and downsampled with LANCZOS, so the stroke edges are
smooth at 192px instead of stair-stepped.

    .venv/bin/python scripts/make-icons.py [--out public]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

NEON = (0x39, 0xFF, 0x14, 255)
TILE = (0x06, 0x05, 0x0B, 255)  # #06050b — the app's background, reads as black
SS = 4  # supersampling factor


def _rounded_tile(size: int, radius_ratio: float) -> Image.Image:
    """Black tile with rounded corners (transparent outside them)."""
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=int(size * radius_ratio), fill=TILE
    )
    return canvas


def _full_bleed(size: int) -> Image.Image:
    """Opaque black square — for maskable and Apple icons."""
    return Image.new("RGBA", (size, size), TILE)


def _draw_art(canvas: Image.Image, art_scale: float) -> None:
    """Draw the phone + play + bar centred, sized to `art_scale` of the canvas.

    Coordinates are authored in a 512 design space and scaled from there, so the
    geometry stays identical at every output size.
    """
    size = canvas.width
    s = (size / 512.0) * art_scale
    off = (size - 512 * s) / 2.0
    draw = ImageDraw.Draw(canvas)

    def px(v: float) -> float:
        return off + v * s

    # Phone body: 232x336 in design space, 20px outline.
    left, top, right, bottom = px(140), px(88), px(372), px(424)
    stroke = max(1, int(round(20 * s)))
    draw.rounded_rectangle(
        (left, top, right, bottom), radius=int(round(44 * s)), outline=NEON, width=stroke
    )

    # Play triangle, centred in the screen area (above the home bar).
    cx, cy = px(256), px(246)
    draw.polygon(
        [(cx - 24 * s, cy - 44 * s), (cx - 24 * s, cy + 44 * s), (cx + 46 * s, cy)],
        fill=NEON,
    )

    # Small bar at the bottom of the phone.
    bar_w, bar_h = 88 * s, 12 * s
    draw.rounded_rectangle(
        (px(256) - bar_w / 2, px(368), px(256) + bar_w / 2, px(368) + bar_h),
        radius=bar_h / 2,
        fill=NEON,
    )


def _render(size: int, *, art_scale: float, rounded: bool, radius_ratio: float = 0.22) -> Image.Image:
    big = _rounded_tile(size * SS, radius_ratio) if rounded else _full_bleed(size * SS)
    _draw_art(big, art_scale)
    return big.resize((size, size), Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = {
        "icon-192.png": _render(192, art_scale=0.78, rounded=True),
        "icon-512.png": _render(512, art_scale=0.78, rounded=True),
        "icon-192-maskable.png": _render(192, art_scale=0.70, rounded=False),
        "icon-512-maskable.png": _render(512, art_scale=0.70, rounded=False),
        "apple-touch-icon.png": _render(180, art_scale=0.78, rounded=False),
    }
    for name, image in targets.items():
        path = out / name
        # 512/192 stay RGBA (transparent corners); the Apple icon must be opaque.
        image.save(path, "PNG", optimize=True)
        print(f"{path}  {image.width}x{image.height}  {path.stat().st_size} bytes")

    favicon = out / "favicon.ico"
    _render(256, art_scale=0.82, rounded=True).save(
        favicon, sizes=[(16, 16), (32, 32), (48, 48)]
    )
    print(f"{favicon}  16/32/48  {favicon.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
