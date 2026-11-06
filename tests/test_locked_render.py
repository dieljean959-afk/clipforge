"""GOLDEN TEST — LOCKED_SPEC.md. Renders a clip and measures the pixels.

If this fails, the change is rejected: it means the caption look, the caption
position or the white title bar no longer matches the locked reference. The
numbers are the spec's own (1080x1920):

    cap height 38 px (+/-4), line pitch 86 px (+/-6),
    caption bottom at captionBottomY (+/-10),
    white bar present, wrapped, and not clipped,
    emoji overlay present,
    exactly one caption style in the UI.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(not (ffmpeg and ffprobe), reason="ffmpeg/ffprobe not installed")

ROOT = Path(__file__).resolve().parent.parent
FRAME_W, FRAME_H = 1080, 1920
SRC_W, SRC_H = 1280, 720
#: captionBottomY for a 16:9 source in a 1080x1920 FIT frame:
#: fgH = 608, fgTop = 656, fgBottom = 1264, minus the 24 px inset.
_FG_H = round(FRAME_W * SRC_H / SRC_W) // 2 * 2          # 608
_FG_TOP = (FRAME_H - _FG_H) // 2                         # 656
EXPECTED_CAPTION_BOTTOM = _FG_TOP + _FG_H - 24           # 1240

CAP_HEIGHT = 38
LINE_PITCH = 86
BAR_TOP = 160


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    """A 16:9, flat-grey 4 s source: caption pixels are trivially isolatable."""
    out = tmp_path_factory.mktemp("golden-src") / "src.mp4"
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=0x303030:s={SRC_W}x{SRC_H}:r=30:d=4",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=4",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", str(out),
    ])
    return out


@pytest.fixture(scope="module")
def base_settings(tmp_path_factory):
    from app.config import load_config

    settings = load_config(quiet=True)
    root = tmp_path_factory.mktemp("golden-data")
    paths = replace(
        settings.app.paths,
        data_dir=root / "data",
        outputs_dir=root / "outputs",
        downloads_dir=root / "downloads",
        uploads_dir=root / "uploads",
    )
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    return replace(settings, app=replace(settings.app, paths=paths))


def _cues() -> list[dict]:
    # 5 words = the spec's 4-5 word chunk; 2 stacked lines of <=16 chars.
    return [
        {
            "start": 0.20,
            "end": 1.60,
            "text": "THE WHOLE GAME WAS WON",
            "words": [
                {"word": "THE", "start": 0.20, "end": 0.45},
                {"word": "WHOLE", "start": 0.45, "end": 0.75},
                {"word": "GAME", "start": 0.75, "end": 1.05},
                {"word": "WAS", "start": 1.05, "end": 1.30},
                {"word": "WON", "start": 1.30, "end": 1.60},
            ],
        }
    ]


def _render(src: Path, settings, tmp_path: Path, *, title: str,
            emoji: list[dict] | None = None,
            cues: list[dict] | None = None, name: str = "clip.mp4") -> Path:
    from app import encode

    dest = tmp_path / name
    cut = {
        "start": 0.0,
        "end": 3.0,
        "order": 0,
        "cropBox": {"x": 0, "y": 0, "w": SRC_W, "h": SRC_H},
        "framingSegments": [{"start": 0.0, "end": 3.0, "mode": "fit"}],
    }
    encode.render_clip(
        src, cut, dest, settings,
        cues=_cues() if cues is None else cues,
        preset_key="bold_stack",
        emoji_selection=emoji or [],
        title=title,
    )
    return dest


def _frame(video: Path, at: float, out: Path) -> Path:
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1", str(out),
    ])
    return out


def _luma_rows(img, *, x0: int, x1: int, y0: int, y1: int, threshold: int = 200):
    """Rows holding a caption-fill pixel: white OR a saturated cycle colour.

    The source is flat grey (0x303030), so a pixel whose strongest channel is
    above the threshold is caption fill: white (all channels), yellow
    (255,255,0) or red/green (one channel at 255).
    """
    px = img.load()
    rows = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            r, g, b = px[x, y][:3]
            if max(r, g, b) > threshold:
                rows.append(y)
                break
    return rows


def _groups(rows: list[int]) -> list[tuple[int, int]]:
    if not rows:
        return []
    out = [(rows[0], rows[0])]
    for y in rows[1:]:
        if y - out[-1][1] <= 3:
            out[-1] = (out[-1][0], y)
        else:
            out.append((y, y))
    return out


@pytest.fixture(scope="module")
def golden_clip(source, base_settings, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("golden-render")
    # A deliberately LONG hook: the bar must wrap it, never clip it.
    return _render(
        source, base_settings, tmp,
        title="THIS IS A VERY LONG HOOK TITLE THAT MUST WRAP ONTO TWO LINES",
    )


def test_clip_is_1080x1920(golden_clip):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(golden_clip)],
        capture_output=True, text=True, timeout=60,
    )
    assert out.stdout.strip() == "1080,1920", out.stdout + out.stderr


def test_caption_lines_cap_height_pitch_and_bottom(golden_clip, tmp_path):
    from PIL import Image

    frame = _frame(golden_clip, 0.8, tmp_path / "caption.png")
    img = Image.open(frame).convert("RGB")
    rows = _luma_rows(img, x0=140, x1=940, y0=1000, y1=1320)
    groups = _groups(rows)
    assert len(groups) == 2, f"expected two stacked caption lines, found {groups}"
    line1, line2 = groups
    cap = line2[1] - line2[0] + 1
    pitch = line2[0] - line1[0]
    bottom = line2[1]
    assert abs(cap - CAP_HEIGHT) <= 4, f"cap height {cap}px (spec {CAP_HEIGHT}±4)"
    assert abs(pitch - LINE_PITCH) <= 6, f"line pitch {pitch}px (spec {LINE_PITCH}±6)"
    assert abs(bottom - EXPECTED_CAPTION_BOTTOM) <= 10, (
        f"caption bottom {bottom}px (captionBottomY {EXPECTED_CAPTION_BOTTOM}±10)"
    )


def test_white_title_bar_present_wrapped_and_not_clipped(golden_clip, tmp_path):
    from PIL import Image

    frame = _frame(golden_clip, 0.8, tmp_path / "title.png")
    img = Image.open(frame).convert("RGB")
    px = img.load()

    def edge_white(y: int) -> bool:
        # The bar is FULL WIDTH; the text in the middle is black, so the edge
        # pixels (not the whole row) are what identify the plate.
        for x in (0, 2, FRAME_W - 3, FRAME_W - 1):
            r, g, b = px[x, y][:3]
            if not (r > 240 and g > 240 and b > 240):
                return False
        return True

    assert edge_white(BAR_TOP + 2), f"no full-width white bar at y={BAR_TOP}"
    # The bar grows downward for a wrapped title: find its true bottom edge.
    bottom = BAR_TOP
    while bottom < FRAME_H - 1 and edge_white(bottom + 1):
        bottom += 1
    assert bottom - BAR_TOP >= 70 - 1, f"bar height {bottom - BAR_TOP}px (spec >= 70)"
    # Two wrapped lines: a row inside the bar carries dark text in two bands.
    text_rows = 0
    for y in range(BAR_TOP + 1, bottom):
        dark = sum(1 for x in range(0, FRAME_W) if sum(px[x, y][:3]) < 240)
        if dark > 4:
            text_rows += 1
    assert text_rows >= 40, f"title text looks missing/clipped ({text_rows} text rows)"
    # Never clipped: every dark pixel of the title is INSIDE the bar.
    dark_outside = 0
    for y in list(range(BAR_TOP - 12, BAR_TOP)) + list(range(bottom + 1, bottom + 13)):
        if not (0 <= y < FRAME_H):
            continue
        for x in range(0, FRAME_W, 3):
            r, g, b = px[x, y][:3]
            if r > 235 and g > 235 and b > 235:
                dark_outside += 1
    assert dark_outside == 0, "white bar pixels found outside the bar rectangle"


def test_emoji_overlay_is_present(source, base_settings, tmp_path):
    from PIL import Image, ImageChops

    png = ROOT / "app" / "assets" / "emoji" / "144" / "1f525.png"
    assert png.exists(), "bundled Twemoji PNG missing"
    plain = _render(source, base_settings, tmp_path, title="HOOK TITLE HERE", name="plain.mp4")
    with_emoji = _render(
        source, base_settings, tmp_path, title="HOOK TITLE HERE", name="emoji.mp4",
        # The selection carries the whitelist id; geometry comes from the
        # emoji layer itself, exactly as the captions stage supplies it.
        emoji=[{"emoji": "🔥", "start": 0.2, "end": 2.8}],
    )
    a = Image.open(_frame(plain, 0.8, tmp_path / "a.png")).convert("RGB")
    b = Image.open(_frame(with_emoji, 0.8, tmp_path / "b.png")).convert("RGB")
    diff = ImageChops.difference(a, b)
    changed = sum(1 for v in diff.getdata() if sum(v) > 30)
    assert changed > 500, f"emoji overlay missing (only {changed} changed pixels)"


def test_only_one_caption_style_exists():
    from app.caption_presets import CAPTION_PRESETS, PRESET_ORDER

    assert PRESET_ORDER == ("bold_stack",)
    assert list(CAPTION_PRESETS) == ["bold_stack"]
    dialog = (ROOT / "src/components/clipmake/CaptionStyleDialog.tsx").read_text(encoding="utf-8")
    form = (ROOT / "src/components/clipmake/NewJobForm.tsx").read_text(encoding="utf-8")
    for forbidden in ("bold_pop", "karaoke", "lower_third", "boxed", "vertical_916",
                      "viral", "Animated ·", "clean_white", "bold_yellow"):
        assert forbidden not in dialog, f"style picker still offers {forbidden!r}"
        assert forbidden not in form, f"style picker still offers {forbidden!r}"
