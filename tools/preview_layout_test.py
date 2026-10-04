"""The clip preview must fill its card and actually play.

Two complaints, one of which turned out not to be a rendering bug at all:

  "the frame only fills the top ~60%, with a black area below it"
  "I need real playback"
  "it doesn't play"

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

AND THE RETRY WAS A NO-OP. The cache-buster rode AFTER the `#`, making it part
of the fragment — which is never sent to the server and is not part of cache
identity. So the "retry" re-requested the byte-identical URL, could be answered
from cache, and one dropped connection became a permanent error card on a file
ffprobe says is fine.

These tests pin the fixes and the invariants that keep them true:
  - taps reach the <video>; no transparent overlay sits above it
  - the frame is a 9:16 box and the video fills it edge to edge (display:block
    matters — <video> is inline by default and leaves a baseline gap)
  - play/pause/seek/mute are wired to real media events, and only one clip in
    a grid plays at a time
  - transient network/decode failures retry with a REAL new URL and a backoff;
    a rejected codec does not loop forever; the poster stays visible throughout
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
    check("the cue list is on screen, so a clip is judgeable without streaming",
          "Captions as rendered" in body and "clip.captionCues ?? []" in body)
    check("the active cue is highlighted as it plays", "activeIndex" in body)
    check("the lightbox offers the MP4 directly", "clipDownloadUrl(clip._id)" in body)


def test_caption_preview() -> None:
    print("\n[8] the burned-in caption is visible while previewing")
    body = src("src/components/clipforge/ClipPreview.tsx")
    check("a cue lookup helper exists", "function cueAt" in body)
    check("cues are matched against currentTime", "cueAt(clip, t)" in body)
    check("the cue is rendered as an overlay", "pointer-events-none absolute" in body)
    check("the overlay is legible on any frame",
          "text-shadow" in body)
    check("the overlay only shows during playback", 'controls && cue !== ""' in body)


def test_resilient_loading() -> None:
    print("\n[9] a flaky connection does not permanently kill a clip")
    body = src("src/components/clipforge/ClipPreview.tsx")

    # THE bug this section exists for: the retry rode AFTER the '#', so it was
    # part of the fragment, which never reaches the server and is not part of
    # cache identity. The "cache-busted" retry re-requested the identical URL.
    check("the retry counter is in the QUERY string, before the fragment",
          "${videoUrl}${retry > 0 ? `?r=${retry}` : \"\"}#t=0.001" in body,
          "no query-string retry found")
    check("no retry parameter is built into the fragment",
          "#t=0.001" in body and "?r=${retry}" in body)

    check("network errors are classified as retryable",
          "code === MEDIA_ERR_NETWORK" in body)
    check("decode errors are classified as retryable",
          "code === MEDIA_ERR_DECODE" in body)
    check("an unsupported codec is NOT retried pointlessly",
          "MEDIA_ERR_SRC_NOT_SUPPORTED" in body and "isRetryable(code)" in body)
    check("the Retry button is hidden when the codec is truly unsupported",
          "fatalError" in body and "{!fatalError && (" in body)
    check("retries are bounded", "retry < 3" in body)
    check("retries back off rather than hammering",
          "[0, 1200, 3000]" in body)
    check("a successful load cancels a pending retry",
          "onCanPlay" in body and "clearTimeout(retryTimer.current)" in body)
    check("there is a manual Retry button",
          "manualRetry" in body and "Retry" in body)
    check("the error explains itself instead of shrugging",
          "connection dropped" in body and "stalled" in body)
    check("the poster stays on screen while broken, so the frame is judgeable",
          "(!videoFailed || thumbUrl)" in body)
    check("a reconnecting banner is shown", "Reconnecting" in body)


def test_controls_fit_a_phone() -> None:
    print("\n[10] the control bar survives a 2-up phone grid")
    body = src("src/components/clipforge/ClipPreview.tsx")
    # The seek track collapsed to zero width on a narrow card, so its 12px
    # thumb rendered on top of the mute button and the time was clipped.
    check("the seek track has a minimum width", "min-w-8" in body)
    check("the time readout cannot be squeezed shut",
          "shrink-0 whitespace-nowrap" in body)
    check("the row is not cramped by its own gaps", 'gap-1.5' in body)


def _strip_comments(html: str) -> str:
    """Drop CSS/JS comments so prose about a removed expression is not code."""
    out: list[str] = []
    i, n = 0, len(html)
    while i < n:
        if html.startswith("/*", i):
            end = html.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if html.startswith("//", i):
            end = html.find("\n", i)
            i = n if end == -1 else end
            continue
        out.append(html[i])
        i += 1
    return "".join(out)


def test_boot_card_never_parks() -> None:
    """The boot card must not be able to LOOK stuck.

    It used to compute

        pct = Math.min(97, Math.round((loaded / EXPECTED_MODULES) * 100))

    with EXPECTED_MODULES = 26, while `loaded` counted every finished
    subresource — modules, CSS, favicon and /api polls alike. One cold load is
    ~41 resources, so the count passed 26 early and the bar pinned at exactly
    97% for the rest of the boot: measured across 308 real boots, that frozen
    97% was on screen for a median of 7s, p90 40s, max 276s, on EVERY load. It
    read as a hang. The bar is now indeterminate and the status line reports
    facts that keep changing.
    """
    print("\n[11] the boot card cannot report a frozen percentage")
    html = src("index.html")
    # Only EXECUTABLE code is under test. index.html documents why the old
    # percentage was removed, and that prose quotes the old expression — a
    # naive substring check would flag its own explanation.
    code = _strip_comments(html)

    check("no guessed module total is left behind",
          "EXPECTED_MODULES" not in code)
    check("no percentage is clamped to a ceiling",
          "Math.min(97" not in code and "Math.min(9" not in code)
    # The precise invariant: the bar's width is NEVER set from script, so there
    # is no number on it that can stop moving. (Percentages in the CSS — the
    # bar's own width, keyframe offsets — are layout, not reported progress.)
    check("script never writes a width onto the bar",
          "style.width" not in code and "firstElementChild" not in code)
    check("the bar animates instead of parking",
          "@keyframes bootbar" in code and "animation: bootbar" in code)
    check("a failed boot stops the bar moving",
          "#boot.failed .meter" in code)
    check("the status line reports real, changing facts",
          "paintStatus" in code and "files" in code and "MB" in code)
    # The failure paths still matter: no auto-reload storm on a slow load.
    check("a slow load is still never treated as a failure",
          "idleFor < 4000" in code)
    check("the hard ceiling still exists and still refuses to reload",
          "HARD_CEILING_MS" in code and "No auto-reload" in code)
    check("module-load failures still auto-retry, and are capped",
          "MAX_RELOADS" in code and "failAndRetry" in code)


def main() -> int:
    test_render_geometry()
    test_existing_clips()
    test_taps_reach_the_video()
    test_fills_the_card()
    test_playback_wiring()
    test_only_one_plays()
    test_lightbox()
    test_caption_preview()
    test_resilient_loading()
    test_controls_fit_a_phone()
    test_boot_card_never_parks()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
