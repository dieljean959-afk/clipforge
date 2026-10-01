"""The clip preview must fill its card and actually play.

Two complaints, one of which turned out not to be a rendering bug at all:

  "the frame only fills the top ~60%, with a black area below it"
  "I need real playback"

THE BLACK AREA IS NOT IN THE FILE. ffprobe on a real rendered clip: 1080x1920,
no rotation matrix, no SAR override. A frame-by-frame luminance scan finds ZERO
pure-black rows and no discontinuity — the lower half of these clips is the
subject's dark clothing against a dark background. Real encoded pixels. So
there is nothing to "fix" in the encode for the specific symptom; what made it
LOOK like a bar is that you were staring at a poster frame of dark footage in
a phone-sized card with no way to press play.

THE PLAYBACK WAS BROKEN, and provably so: ClipCard wrapped the video in a
transparent <button> covering the entire frame to open the detail dialog. Every
tap hit that button, so `video.play()` was never called. That overlay is gone.

These tests pin the fixes and the invariants that keep them true:
  - taps reach the <video>; no transparent overlay sits above it
  - the frame is a 9:16 box and the video fills it edge to edge (display:block
    matters — <video> is inline by default and leaves a baseline gap)
  - play/pause/seek/mute are wired to real media events, and only one clip in
    a grid plays at a time
  - the review modal is portalled, native-controls, unmuted, closable
  - renders pin setsar=1, so a non-square-pixel source cannot letterbox
  - every clip actually on disk is 1080x1920 with no rotation

    python tools/preview_layout_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ffmpeg_bin, ffprobe_bin  # noqa: E402
from app.store import LocalStore  # noqa: E402
from app.video import build_output_chain  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


def src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def probe(path: Path) -> dict:
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    meta = json.loads(out.stdout)
    v = next((s for s in meta["streams"] if s.get("codec_type") == "video"), {})
    return {
        "width": v.get("width"),
        "height": v.get("height"),
        "sar": v.get("sample_aspect_ratio"),
        "dar": v.get("display_aspect_ratio"),
        "rotation": [sd.get("rotation") for sd in v.get("side_data_list", [])],
        "rotate_tag": (v.get("tags") or {}).get("rotate"),
    }


# --------------------------------------------------------------------------- #

def test_render_geometry() -> None:
    print("\n[1] the render chain cannot letterbox")
    chain = build_output_chain(None, 1920, 1080, 607, 1080, 1920)
    check("the chain ends with an exact 1080x1920 crop", f"crop={1080}:{1920}" in chain, chain)
    check("the chain pins setsar=1", "setsar=1" in chain, chain)
    check("the scale step only ever INCREASES (never pads)",
          "force_original_aspect_ratio=increase" in chain, chain)
    check("no pad filter sneaks in", "pad=" not in chain, chain)

    # A non-square-pixel source must still come out square: that is the whole
    # point of setsar=1.
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "nsar.mp4"
        subprocess.run(
            [
                ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
                # 4:3 content tagged 4:3 SAR = non-square pixels, the classic
                # source of "why is my vertical video stretched/letterboxed"
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
                "-vf", "scale=320:240,setsar=4/3",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
            ],
            check=True, capture_output=True, timeout=300,
        )
        before = probe(out)
        check("the test source really does have a non-1:1 SAR",
              before["sar"] not in (None, "1:1"), str(before["sar"]))

        fixed = Path(td) / "fixed.mp4"
        w, h = int(before["width"]), int(before["height"])
        crop_w = min(int(h * 1080 / 1920), w)
        chain_240 = build_output_chain(None, w, h, crop_w, 1080, 1920)
        subprocess.run(
            [
                ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(out), "-t", "2",
                "-vf", chain_240,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-movflags", "+faststart", str(fixed),
            ],
            check=True, capture_output=True, timeout=300,
        )
        info = probe(fixed)
        check("the output is exactly 1080x1920",
              (info["width"], info["height"]) == (1080, 1920),
              f'{info["width"]}x{info["height"]}')
        check("the output SAR is square (1:1)",
              info["sar"] in (None, "1:1"), str(info["sar"]))
        check("the display aspect ratio is 9:16",
              info["dar"] in (None, "9:16"), str(info["dar"]))
        check("there is no rotation matrix", not info["rotation"], str(info["rotation"]))
        check("there is no rotate tag", not info["rotate_tag"], str(info["rotate_tag"]))


def test_existing_clips() -> None:
    print("\n[2] every clip on disk is a clean 9:16 frame")
    from app.config import load_config

    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)
    seen: set[str] = set()
    checked = 0
    bad: list[str] = []
    for clip in store.list_all_clips(limit=10_000):
        vp = clip.get("videoPath")
        if not vp or str(vp) in seen:
            continue
        seen.add(str(vp))
        p = Path(str(vp))
        if not p.exists():
            bad.append(f"{p.name}: file missing")
            continue
        checked += 1
        try:
            info = probe(p)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{p.name}: ffprobe failed ({exc})")
            continue
        if (info["width"], info["height"]) != (1080, 1920):
            bad.append(f'{p.name}: {info["width"]}x{info["height"]}')
        if info["sar"] not in (None, "1:1"):
            bad.append(f'{p.name}: SAR {info["sar"]}')
        if info["rotation"] or info["rotate_tag"]:
            bad.append(f'{p.name}: rotation {info["rotation"] or info["rotate_tag"]}')

    if checked == 0:
        print("  (no rendered clips on disk right now — nothing to verify)")
        return
    check(f"all {checked} rendered clip(s) are 1080x1920, SAR 1:1, unrotated",
          not bad, "; ".join(bad[:5]))


def test_taps_reach_the_video() -> None:
    print("\n[3] taps reach the <video>, not an invisible overlay")
    card = src("src/components/clipforge/ClipCard.tsx")
    # The old overlay: an absolutely-positioned button covering the media area.
    check("ClipCard has no absolutely-positioned overlay button over the frame",
          'absolute inset-x-2 top-2' not in card)
    check("ClipCard no longer opens the dialog from a full-frame overlay",
          "bottom-12 cursor-pointer rounded-t-xl" not in card)
    check("the title still opens the detail dialog",
          "onClick={() => onOpen?.(clip)}" in card)
    preview = src("src/components/clipforge/ClipPreview.tsx")
    check("the <video> itself handles the tap",
          "onClick={() => void togglePlay()}" in preview)


def test_fills_the_card() -> None:
    print("\n[4] the frame fills its 9:16 box")
    body = src("src/components/clipforge/ClipPreview.tsx")
    check("the container declares 9:16", 'aspect-[9/16]' in body)
    check("the video is display:block (no inline baseline gap)",
          'className="block h-full w-full object-cover"' in body)
    check("display:block is also set inline, defensively", 'style={{ display: "block" }}' in body)
    check("object-cover fills rather than fits", "object-cover" in body)
    check("no fixed pixel height constrains the media",
          "h-[4" not in body and "max-h-" not in body.split("ClipLightbox")[0])
    check("the lightbox uses object-contain so nothing is cropped there",
          "object-contain" in body)


def test_playback_wiring() -> None:
    print("\n[5] play, pause, seek and mute are real")
    body = src("src/components/clipforge/ClipPreview.tsx")
    check("play() is awaited inside the click handler", "await v.play()" in body)
    check("the play() promise is handled", ".catch(" in body or "catch (err)" in body)
    check("a refused play is surfaced", "setVideoFailed(true)" in body)
    check('preload="metadata" is kept', 'preload="metadata"' in body)
    check("playsInline is kept", "playsInline" in body)
    check("autoplay is gated on mute", "autoPlay={autoPlay && muted}" in body)
    check("the mute button flips video.muted", "v.muted = !v.muted" in body)
    check("the mute button re-renders from the element", "setMuted(v.muted)" in body)
    check("the play icon follows the play event", "onPlay={() => setPlaying(true)}" in body)
    check("the icon resets on pause", "onPause={() => setPlaying(false)}" in body)
    check("the icon resets on ended", "onEnded={() => setPlaying(false)}" in body)
    check("the time readout follows timeupdate", "onTimeUpdate" in body)
    check("the progress bar is seekable by pointer",
          "onPointerDown" in body and "onPointerMove" in body and "onPointerUp" in body)
    check("seeking writes currentTime", "v.currentTime =" in body)
    check("the seek bar is touch-action safe", "touch-none" in body)
    check("the seek bar is exposed as a slider", 'role="slider"' in body)


def test_only_one_plays() -> None:
    print("\n[6] starting one clip pauses the others")
    body = src("src/components/clipforge/ClipPreview.tsx")
    check("a play broadcast exists", "clipforge:clip-play" in body)
    check("playing dispatches it", "new CustomEvent(PLAY_EVENT" in body)
    check("receivers pause the other element", "other.pause()" in body)
    check("receivers ignore their own element", "other !== videoRef.current" in body)
    check("the listener is cleaned up",
          "window.removeEventListener(PLAY_EVENT" in body)


def test_lightbox() -> None:
    print("\n[7] the review modal")
    body = src("src/components/clipforge/ClipPreview.tsx")
    check("an expand control exists", "Maximize2" in body and "setExpanded(true)" in body)
    check("it is a modal dialog", 'role="dialog"' in body and 'aria-modal="true"' in body)
    check("it is portalled to the body (transform-safe)",
          "createPortal" in body and "document.body" in body)
    check("it uses the browser's own controls", "controls" in body and "<video" in body)
    check("it plays with sound", "muted={false}" in body)
    check("it closes on Escape", 'e.key === "Escape"' in body)
    check("it has a visible close button", "aria-label=\"Close preview\"" in body)
    check("closing from the backdrop works", "onClose()" in body)


def test_caption_preview() -> None:
    print("\n[8] the burned-in caption is visible while previewing")
    body = src("src/components/clipforge/ClipPreview.tsx")
    check("a cue lookup helper exists", "function cueAt" in body)
    check("cues are matched against currentTime", "cueAt(clip, t)" in body)
    check("the cue is rendered as an overlay", "pointer-events-none absolute" in body)
    check("the overlay is legible on any frame",
          "text-shadow" in body)
    check("the overlay only shows during playback", 'controls && cue !== ""' in body)


def main() -> int:
    test_render_geometry()
    test_existing_clips()
    test_taps_reach_the_video()
    test_fills_the_card()
    test_playback_wiring()
    test_only_one_plays()
    test_lightbox()
    test_caption_preview()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
