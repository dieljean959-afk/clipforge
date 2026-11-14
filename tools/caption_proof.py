"""End-to-end caption proofs: render real clips and measure the frames.

Everything here goes through the SAME entry point the export stage uses
(``encode.render_clip`` -> ``subtitle.write_ass`` -> ffmpeg/libass), so a
number printed here is a number the shipped renderer produces.

    python3 tools/caption_proof.py setup          # build source videos
    python3 tools/caption_proof.py render ...     # render one configuration
    python3 tools/caption_proof.py measure ...    # cap height / line pitch
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORK = Path("/tmp/cap")

#: A fixed transcript: 5 chunks of 4 words, 1.2 s apart, with a 0.9 s pause
#: after chunk 3 so `pause_sec` has something real to fire on.
WORDS = (
    "every single morning i wake up early and check the numbers "
    "because nobody hands you the win you take it yourself "
    "stay hungry stay foolish and keep building the thing "
    "the best time to start was yesterday the next best is now "
)
SENTENCE = [
    {"text": "every single morning i"},
    {"text": "wake up early and"},
    {"text": "check the numbers because"},
    {"text": "nobody hands you the win"},
    {"text": "you take it yourself"},
    {"text": "stay hungry stay foolish and"},
    {"text": "keep building the thing"},
]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"FAILED {' '.join(cmd[:6])}...\n{proc.stderr[-2000:]}")


def setup() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    # 16:9 source -> 9:16 frame gets the blurred band, so the caption anchor
    # is the CENTRED VIDEO's lower edge (case C in the brief).
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=12",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=12",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(WORK / "src16x9.mp4"),
    ])
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30:duration=12",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=12",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(WORK / "src9x16.mp4"),
    ])
    for name in ("src16x9.mp4", "src9x16.mp4"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(WORK / name)],
            capture_output=True, text=True,
        ).stdout.strip().splitlines()
        print(f"{name}: {probe[0]}")


def _dims(src: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    w, h = out.split(",")[:2]
    return int(w), int(h)


def cues() -> list[dict]:
    """Word-timed cues with a deliberate 0.9 s gap after chunk 3."""
    out = []
    t = 0.20
    for i, chunk in enumerate(SENTENCE):
        words = chunk["text"].split()
        ws = []
        for w in words:
            ws.append({"word": w, "start": round(t, 3), "end": round(t + 0.24, 3), "conf": 0.95})
            t += 0.26
        out.append({
            "start": round(ws[0]["start"], 3),
            "end": round(ws[-1]["end"], 3),
            "text": chunk["text"],
            "words": ws,
        })
        t += 0.9 if i == 2 else 0.12
    return out


def build_settings(size_scale: float = 1.0, preset: str = "bold_stack", **overrides):
    from app.config import load_config
    from app import caption_style

    s = load_config(quiet=True)
    values = dict(overrides)
    values["preset_key"] = preset
    values["size_scale"] = size_scale
    from dataclasses import replace

    resolved = caption_style.resolve_caption_settings(s, values)
    sub = replace(
        s.subtitle,
        font_size=int(resolved["fontSize"]),
        words_per_cue=int(resolved["wordsPerCue"]),
        words_per_cue_max=int(resolved["wordsPerCueMax"]),
        pause_sec=float(resolved["pauseSec"]),
        size_scale=float(resolved["sizeScale"]),
        highlight_color=str(resolved["highlightColor"] or ""),
        position_y=float(resolved["positionY"]),
        emoji_enabled=bool(resolved["emojiEnabled"]),
        emoji_density=str(resolved["emojiDensity"]),
        preset=str(resolved["presetKey"]),
    )
    return replace(s, subtitle=sub), resolved


def render(
    out: Path,
    preset: str = "bold_stack",
    size_scale: float = 1.0,
    source: str = "src16x9.mp4",
    emoji: bool = False,
    seconds: float = 6.0,
) -> dict:
    from app import encode
    from app.emoji import overlays_for_render

    settings, resolved = build_settings(size_scale, preset, emoji_enabled=emoji)
    src = WORK / source
    src_w, src_h = _dims(src)
    # A 16:9 source in a 9:16 frame is WIDER than the target, so it is shown
    # whole over its blurred self (FIT); a 9:16 source fills the frame.
    fit = (src_w / src_h) > (1080 / 1920) + 0.05
    cut = {
        "order": 0,
        "start": 0.0,
        "end": seconds,
        "cropBox": {"x": 0.0, "y": 0.0, "w": float(src_w), "h": float(src_h)},
        # encode keys the FIT (blur-fill) layout off framingSegments, not a
        # single "fit" flag — without this the 16:9 fixture renders FILL and
        # the caption anchor is never computed.
        "framingSegments": (
            [{"start": 0.0, "end": seconds, "mode": "fit"}] if fit else []
        ),
    }
    selection = []
    if emoji:
        selection = json.loads((WORK / "emoji_selection.json").read_text())
    meta = encode.render_clip(
        src, cut, out, settings,
        cues=cues(), preset_key=resolved["presetKey"],
        emoji_selection=selection,
        on_log=lambda line: None,
    )
    return {"meta": meta, "resolved": resolved, "settings": settings}


def frame_hash(clip: Path, at: float) -> str:
    """md5 of one decoded frame's raw RGB pixels."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{at}", "-i", str(clip),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise SystemExit(f"frame extraction failed at {at}: {proc.stderr[-400:]!r}")
    return hashlib.md5(proc.stdout).hexdigest()


def frame_png(clip: Path, at: float, dest: Path) -> Path:
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{at}",
          "-i", str(clip), "-frames:v", "1", str(dest)])
    return dest


def black_source() -> Path:
    """A near-black 16:9 source: text pixels are unambiguous to measure.

    0x1a1a1a rather than pure black because the renderer's own guard rejects a
    clip whose sampled luma is black ("rendered clip is pitch black") — that
    guard is right, so the fixture goes just above it.
    """
    dest = WORK / "srcblack.mp4"
    if not dest.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=0x1a1a1a:size=1280x720:rate=30:duration=12",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=12",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(dest),
        ])
    return dest


def _txt_bands(png: Path, colour: str | None = None) -> list[tuple[int, int, int]]:
    """Row bands (top, bottom, count) that contain caption text pixels.

    White text is (255,255,255); the coloured line is the preset accent. Both
    are fully saturated after libass draws them over black, so the threshold
    is exact (>=250 on every channel for white, the accent's own channels for
    the coloured band).
    """
    from PIL import Image

    im = Image.open(png).convert("RGB")
    w, h = im.size
    px = im.load()
    bg = (26, 26, 26)  # the fixture's 0x1a1a1a background
    rows = [0] * h
    for y in range(h):
        n = 0
        for x in range(0, w, 2):
            r, g, b = px[x, y]
            if colour is None:
                # BRIGHT fill only (any hue): white text and the coloured
                # line both qualify, the black outline does not.
                if max(abs(r - bg[0]), abs(g - bg[1]), abs(b - bg[2])) > 128:
                    n += 1
            else:
                cr, cg, cb = colour
                # The accent's own hue, allowing for anti-aliasing against the
                # background (pure #FF0000 has a luminance of 76, so a
                # brightness threshold would miss it entirely).
                if abs(r - cr) <= 90 and abs(g - cg) <= 90 and abs(b - cb) <= 90 \
                        and (r + g + b) > (bg[0] + bg[1] + bg[2] + 60):
                    n += 1
        rows[y] = n
    bands: list[tuple[int, int, int]] = []
    start = None
    gap = 0
    for y, n in enumerate(rows):
        if n >= 3:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= 6:
                bands.append((start, y - gap, y - gap - start + 1))
                start = None
    if start is not None:
        bands.append((start, len(rows) - 1, len(rows) - start))
    return bands


def measure(clip: Path, at: float, accents: dict[str, str]) -> dict:
    """Cap height + line pitch, measured off a rendered frame."""
    png = frame_png(clip, at, WORK / f"measure_{clip.stem}_{at}.png")
    white = _txt_bands(png)
    out: dict = {"frame": str(png), "whiteBands": white}
    for name, hexcolour in accents.items():
        rgb = tuple(int(hexcolour[i:i + 2], 16) for i in (1, 3, 5))
        bands = _txt_bands(png, rgb)
        out[f"{name}Bands"] = bands
        out[f"{name}CapHeight"] = max((b[2] for b in bands), default=0)
    allb = sorted(set(white) | set(b for k, v in out.items() if k.endswith("Bands") and k != "whiteBands" for b in v))
    if len(allb) >= 2:
        # Pitch = distance between the two line BASE rows.
        out["linePitch"] = abs(allb[-1][1] - allb[0][1])
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "setup":
        setup()
        return 0
    print("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
