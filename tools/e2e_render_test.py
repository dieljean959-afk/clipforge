#!/usr/bin/env python3
"""End-to-end render test with REAL ffmpeg (no network, no AI keys).

Generates a synthetic source MP4, then exercises the actual pipeline pieces:
word cues → ASS file → frame-accurate cut → face-crop filter → 1080x1920
render with burned captions → real first-frame thumbnail. Verifies the output
with ffprobe.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import subtitle, video
from app.config import ffmpeg_bin, load_config


def make_source(dest: Path, seconds: int = 12) -> Path:
    """Synthetic 1280x720 MP4: moving test pattern + 440Hz tone."""
    subprocess.run(
        [
            ffmpeg_bin(), "-y",
            "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(dest),
        ],
        check=True, capture_output=True,
    )
    return dest


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("SKIP: ffmpeg not available")
        return 0
    settings = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="cf-e2e-"))
    try:
        print("== 1. synthetic source (12s, 1280x720)")
        src = make_source(tmp / "source.mp4")
        meta = video.probe(src)
        print(f"   source: {meta['width']}x{meta['height']}, {meta['duration']:.1f}s")
        assert meta["duration"] > 10 and meta["width"] == 1280

        print("== 2. word timings → cues → ASS")
        words = [
            {"start": 1.0 + i * 0.35, "end": 1.0 + (i + 1) * 0.35, "word": w}
            for i, w in enumerate(
                "this is a real render test with green outlined captions burned in".split()
            )
        ]
        cues = subtitle.words_to_cues(words, words_per_cue=2)
        cue_start, cue_end = words[0]["start"], words[-1]["end"]
        shifted = [
            {"start": c["start"] - cue_start, "end": c["end"] - cue_start, "text": c["text"]}
            for c in cues
        ]
        print(f"   {len(cues)} cues, first: {shifted[0]['text']!r}")
        ass = subtitle.write_ass(shifted, settings, tmp / "clip_01.ass", 1080, 1920)
        assert "ClipForge" in ass.read_text() and "39FF14" in ass.read_text().upper().replace("H00", "").replace("H", "") or "&H0014FF39" in ass.read_text()
        print(f"   ass written: {ass.name} ({ass.stat().st_size} bytes)")

        print("== 3. reframe windows (MediaPipe absent → honest center-crop fallback)")
        windows, used_mp = __import__("app.reframe", fromlist=["face_crop_windows"]).face_crop_windows(
            str(src), cue_start, cue_end, meta["width"], meta["height"]
        )
        print(f"   windows: {len(windows)}, mediapipe used: {used_mp}")
        assert used_mp is False and len(windows) == 1

        print("== 4. REAL render: cut 1.0-9.0s → 9:16 1080x1920 + burned captions")
        dest = tmp / "clip_01.mp4"
        info = video.render_clip(
            src, dest,
            start=cue_start, end=cue_end,
            settings=settings,
            ass_path=ass,
            crop_windows=windows,
            silence_cuts=[],
        )
        print(f"   rendered: {info['width']}x{info['height']}, {info['duration']:.1f}s, {info['bytes'] // 1024} KB in {info['renderMs'] / 1000:.1f}s")
        assert info["width"] == 1080 and info["height"] == 1920, info
        assert abs(info["duration"] - (cue_end - cue_start)) < 0.5, info["duration"]

        print("== 5. thumbnail = real first frame")
        thumb = video.render_first_frame_thumb(dest, tmp / "clip_01.jpg")
        assert thumb.exists() and thumb.stat().st_size > 5_000
        print(f"   thumb: {thumb.stat().st_size // 1024} KB")

        print("== 6. caption frame differs from a caption-less render (burn verified)")
        plain = tmp / "plain.mp4"
        video.render_clip(src, plain, start=cue_start, end=cue_end,
                          settings=settings, crop_windows=windows)
        cap_frame = tmp / "cap.jpg"
        plain_frame = tmp / "plain.jpg"
        for vid, out in ((dest, cap_frame), (plain, plain_frame)):
            subprocess.run(
                [ffmpeg_bin(), "-y", "-ss", "1", "-i", str(vid), "-frames:v", "1", str(out)],
                check=True, capture_output=True,
            )
        same = cap_frame.read_bytes() == plain_frame.read_bytes()
        assert not same, "caption frame identical to plain frame — captions were NOT burned"
        print("   burned-caption frame differs from caption-less frame ✓")

        print("\nE2E RENDER TEST PASSED (real ffmpeg output, real captions, real thumbnail)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
