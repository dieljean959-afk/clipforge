"""GOLDEN real-media tests for LOCKED_SPEC.md.

The fixture is a real 14 s excerpt of the real 23:36 source in the bucket,
muxed with that source's real audio. No lavfi/testsrc/generated media is used.
The test renders the excerpt with the pinned production settings and compares
frames with the stored reference render, then checks the measured geometry and
the clip/audio contracts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent
FIXTURE = GOLDEN / "fixtures" / "look_source.mp4"
REFERENCE = GOLDEN / "fixtures" / "golden_reference.mp4"
MANIFEST = json.loads((GOLDEN / "look_manifest.json").read_text(encoding="utf-8"))
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe missing")


def _frame(video: Path, at: float, dest: Path) -> Image.Image:
    proc = subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1", str(dest),
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return Image.open(dest).convert("RGB")


def _caption_mask(img: Image.Image) -> np.ndarray:
    a = np.asarray(img)
    r, g, b = [a[:, :, i].astype(int) for i in range(3)]
    spread = a.max(2).astype(int) - a.min(2).astype(int)
    white = (r > 225) * (g > 225) * (b > 225) * (spread < 25)
    yellow = (r > 175) * (g > 175) * (b < 100) * ((r - b) > 100) * ((g - b) > 100)
    green = (g > 155) * (r < 100) * (b < 100)
    red = (r > 160) * (g < 90) * (b < 90)
    return (white + yellow + green + red) > 0


def _caption_groups(img: Image.Image) -> list[tuple[int, int]]:
    m = _caption_mask(img)[1100:1500, 80:1000]
    rows = [1100 + i for i, n in enumerate(m.sum(axis=1)) if n >= 12]
    groups: list[tuple[int, int]] = []
    for y in rows:
        if groups and y - groups[-1][1] <= 3:
            groups[-1] = (groups[-1][0], y)
        else:
            groups.append((y, y))
    return sorted(sorted(groups, key=lambda g: g[1] - g[0], reverse=True)[:2])


def _white_bar(img: Image.Image) -> tuple[int, int]:
    a = np.asarray(img)
    white = (a[:, :, 0] > 245) * (a[:, :, 1] > 245) * (a[:, :, 2] > 245)
    edge = white[:, [0, 2, -3, -1]].all(axis=1)
    ys = np.where(edge[:500])[0]
    assert len(ys), "full-width white title plate not found"
    return int(ys.min()), int(ys.max())


def _band_edges(img: Image.Image) -> tuple[int, int]:
    a = np.asarray(img.convert("L"), dtype=np.int16)
    strips = np.concatenate([a[:, 0:180], a[:, 900:1080]], axis=1)
    d = np.abs(np.diff(strips, axis=0)).mean(axis=1)
    top = max(range(500, 800), key=lambda y: float(d[y - 1]))
    bottom = max(range(1100, 1400), key=lambda y: float(d[y - 1]))
    return top, bottom


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> Path:
    from app import encode
    from app.config import load_config

    settings = load_config(quiet=True)
    dest = tmp_path_factory.mktemp("golden-render") / "clip.mp4"
    meta = encode.render_clip(
        FIXTURE,
        dict(MANIFEST["cut"]),
        dest,
        settings,
        cues=MANIFEST["cues"],
        preset_key="bold_stack",
        emoji_selection=MANIFEST["emoji"],
        title=MANIFEST["title"],
        audio_src=str(FIXTURE),
        probe_meta={"duration": 14.0, "width": 1280, "height": 720, "fps": 30.0},
        start_offset=0.0,
        audio_seek=0.0,
        audio_start_time=0.0,
        audio_plan=MANIFEST["audioPlan"],
        audio_source_key=MANIFEST["audioPlan"]["sourceKey"],
    )
    assert meta["width"] == 1080 and meta["height"] == 1920
    return dest


def test_real_fixture_and_reference_exist():
    assert FIXTURE.is_file() and FIXTURE.stat().st_size > 1_000_000
    assert REFERENCE.is_file() and REFERENCE.stat().st_size > 1_000_000
    assert MANIFEST["cues"] and MANIFEST["emoji"]


def test_frame_matches_real_golden_reference(rendered, tmp_path):
    # Three real frames: two caption states and the emoji overlay.
    for at, name in ((1.4, "look_1_4.png"), (6.0, "look_6_0.png"), (13.0, "look_13_0.png")):
        got = _frame(rendered, at, tmp_path / "got.png")
        want = Image.open(GOLDEN / name).convert("RGB")
        assert got.size == want.size == (1080, 1920)
        a = np.asarray(got, dtype=np.int16)
        b = np.asarray(want, dtype=np.int16)
        mad = float(np.abs(a - b).mean())
        changed = float((np.abs(a - b).max(axis=2) > 6).mean())
        assert mad <= 1.5, f"{name}: mean pixel diff {mad:.3f} (tight tolerance 1.5)"
        assert changed <= 0.025, f"{name}: {changed:.3%} pixels differ (limit 2.5%)"


def test_measured_contract_and_layout(rendered, tmp_path):
    from app import video

    assert video.probe(rendered)["width"] == 1080
    assert video.probe(rendered)["height"] == 1920
    contract = video.clip_contract(rendered, expected_sec=14.0)
    assert contract["ok"], contract.get("failed")

    img = _frame(rendered, 1.4, tmp_path / "layout.png")
    top, bottom = _band_edges(img)
    assert (top, bottom) == (656, 1264)
    bar = _white_bar(img)
    assert bar == (160, 230), bar
    groups = _caption_groups(img)
    assert len(groups) == 2, groups
    assert abs((groups[1][1] - groups[0][1]) - 86) <= 2, groups
    assert abs((groups[0][1] - 1340)) <= 4, groups
    assert abs((groups[1][1] - 1426)) <= 4, groups


def test_audio_sync_gate_on_real_media(rendered, tmp_path):
    from app import audiosync
    from app.config import load_config

    gate = audiosync.gate_clip(
        rendered,
        str(FIXTURE),
        plan_start=0.0,
        plan_end=13.0,
        cues=MANIFEST["cues"],
        settings=load_config(quiet=True),
        cache_dir=tmp_path / "audiosync",
        previous=[],
        source_start_time=0.0,
    )
    assert gate["passed"] is True, gate
    placement = gate["placement"]
    assert abs(float(placement["lagMs"])) <= 40, placement
    assert float(placement["correlation"]) >= 0.70, placement


def test_emoji_geometry_is_the_locked_value():
    from app import emoji
    from app.config import load_config

    geo = emoji.emoji_geometry(load_config(quiet=True))
    assert geo["size"] == 140
    assert geo["x"] == 470
    assert geo["y"] == 1132


def test_locked_files_guard_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_locked.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LOCKED CONTRACT OK" in proc.stdout
