"""Clip playback: the served bytes must actually be playable in a browser.

Two user-visible bugs, and neither is diagnosable from the UI code:

  BUG A  "This browser could not play the rendered file" — while the
          downloaded MP4 is perfectly fine.
  BUG B  the player shows "0:00 / --:--" instead of the clip's duration.

BUG A's cause is in the RENDER, not the transport. Without
`-movflags +faststart`, ffmpeg writes the `moov` atom (the index saying where
every frame lives) at the END of the file, so the player must download the
whole clip before it can start. On a phone that times out. It also explains
BUG B: metadata cannot be read until the entire body has arrived, so
`video.duration` never becomes a number.

So the transport assertions here (206, Content-Range, Accept-Ranges) are
insurance, not the fix — they passed before and after. The load-bearing checks
are that every render branch carries faststart + a pinned H.264 profile and
that a real file round-trips through the endpoint.

    python tools/playback_test.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    Paths,
    ffmpeg_bin,
    ffprobe_bin,
)
from app.store import LocalStore  # noqa: E402
from app.video import (  # noqa: E402
    AUDIO_ARGS,
    MP4_MOVFLAGS,
    VIDEO_PROFILE_ARGS,
    browser_safe_args,
)
from tools.remux_faststart import moov_is_at_front, remux  # noqa: E402

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


# --------------------------------------------------------------------------- #
# 1. render args
# --------------------------------------------------------------------------- #

def test_render_args() -> None:
    print("\n[1] every render branch carries faststart + a pinned profile")
    body = src("app/video.py")

    check(
        "faststart is declared",
        MP4_MOVFLAGS == ["-movflags", "+faststart"],
        str(MP4_MOVFLAGS),
    )
    check(
        "profile/level are High 4.0",
        VIDEO_PROFILE_ARGS == ["-profile:v", "high", "-level:v", "4.0"],
        str(VIDEO_PROFILE_ARGS),
    )
    check(
        "audio is AAC 128k",
        AUDIO_ARGS == ["-c:a", "aac", "-b:a", "128k", "-ac", "2"],
        str(AUDIO_ARGS),
    )

    # All three command branches (pillow overlay, libass subtitles, plain) must
    # include them, not just one. A branch that forgets faststart is exactly
    # how half a batch ends up unplayable.
    check("all three ffmpeg branches call browser_safe_args()", body.count("browser_safe_args(enc.encoder)") == 3,
          f"found {body.count('browser_safe_args(enc.encoder)')}")
    check("all three ffmpeg branches use AUDIO_ARGS", body.count("*AUDIO_ARGS") == 3,
          f"found {body.count('*AUDIO_ARGS')}")
    check("no branch is still on the old 160k audio", '"-b:a", "160k"' not in body)

    args = browser_safe_args("")
    check("browser_safe_args keeps faststart", "-movflags" in args and "+faststart" in args)
    check("browser_safe_args pins the profile", "-profile:v" in args and "-level:v" in args)
    vaapi = browser_safe_args("h264_vaapi")
    check("VAAPI is exempted from profile/level (its muxer rejects them)",
          "+faststart" in " ".join(vaapi) and "-profile:v" not in vaapi, str(vaapi))


# --------------------------------------------------------------------------- #
# 2. real file, real endpoint
# --------------------------------------------------------------------------- #

def _make_clip(path: Path, *, faststart: bool) -> None:
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=360x640:rate=25:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", "4.0",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
    ]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)


def _probe_codecs(path: Path) -> dict:
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    import json
    meta = json.loads(out.stdout)
    v = next((s for s in meta["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in meta["streams"] if s["codec_type"] == "audio"), {})
    return {
        "vcodec": v.get("codec_name"),
        "profile": (v.get("profile") or "").lower(),
        "pix_fmt": v.get("pix_fmt"),
        "acodec": a.get("codec_name"),
        "duration": float(meta.get("format", {}).get("duration", 0) or 0),
    }


def test_served_file() -> None:
    print("\n[2] a rendered clip is h264/aac/yuv420p with moov at the front")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        good = tmp / "clip.mp4"
        _make_clip(good, faststart=True)

        info = _probe_codecs(good)
        check("video is h264", info["vcodec"] == "h264", str(info["vcodec"]))
        check("profile is High", info["profile"] == "high", str(info["profile"]))
        check("pixel format is yuv420p", info["pix_fmt"] == "yuv420p", str(info["pix_fmt"]))
        check("audio is aac", info["acodec"] == "aac", str(info["acodec"]))
        check("duration is readable", info["duration"] > 2.5, str(info["duration"]))
        check("moov is at the FRONT (faststart)", moov_is_at_front(good))

        # The endpoint: real bytes, real headers, real Range handling.
        paths = Paths(
            downloads_dir=tmp / "downloads",
            outputs_dir=tmp / "outputs",
            data_dir=tmp / "data",
            static_dir=tmp / "static",
        )
        store = LocalStore(paths)
        job = store.create_job("upload", "x.mp4")
        clips = store.create_clips(job["_id"], [{"start": 0.0, "end": 3.0, "title": "c"}], "src")
        clips[0]["videoPath"] = str(good)
        clips[0]["renderState"] = "done"
        store.write_clip(clips[0])

        from fastapi.testclient import TestClient

        import app.main as main_mod

        main_mod.store = store
        client = TestClient(main_mod.app)
        url = f"/api/clips/{clips[0]['_id']}/video"

        r = client.get(url)
        check("GET returns 200", r.status_code == 200, str(r.status_code))
        check("Content-Type is video/mp4", r.headers.get("content-type") == "video/mp4",
              str(r.headers.get("content-type")))
        check("Content-Length is the real file size",
              int(r.headers.get("content-length") or 0) == good.stat().st_size,
              f'{r.headers.get("content-length")} vs {good.stat().st_size}')
        check("Accept-Ranges: bytes is advertised", r.headers.get("accept-ranges") == "bytes",
              str(r.headers.get("accept-ranges")))
        check("Cache-Control forbids a stale body",
              "no-cache" in (r.headers.get("cache-control") or ""),
              str(r.headers.get("cache-control")))
        check("the body is the whole file, not chunked",
              len(r.content) == good.stat().st_size, str(len(r.content)))

        r2 = client.get(url, headers={"Range": "bytes=0-1023"})
        check("a Range request returns 206", r2.status_code == 206, str(r2.status_code))
        check("206 reports Content-Range",
              r2.headers.get("content-range") == f"bytes 0-1023/{good.stat().st_size}",
              str(r2.headers.get("content-range")))
        check("206 body is exactly 1024 bytes", len(r2.content) == 1024, str(len(r2.content)))
        check("206 Content-Length is 1024", r2.headers.get("content-length") == "1024",
              str(r2.headers.get("content-length")))

        r3 = client.get(url, headers={"Range": "bytes=100-"})
        check("an open-ended Range is honoured", r3.status_code == 206, str(r3.status_code))
        check("the open-ended body is the tail",
              len(r3.content) == good.stat().st_size - 100, str(len(r3.content)))

        rh = client.head(url)
        check("HEAD returns 200 with a length", rh.status_code == 200
              and int(rh.headers.get("content-length") or 0) == good.stat().st_size)


def test_backfill() -> None:
    print("\n[3] a pre-fix clip can be remuxed into a playable one")
    with tempfile.TemporaryDirectory() as td:
        legacy = Path(td) / "legacy.mp4"
        _make_clip(legacy, faststart=False)
        check("the pre-fix render really does have moov at the end",
              not moov_is_at_front(legacy))
        before = _probe_codecs(legacy)
        ok, detail = remux(legacy)
        check("remux succeeds", ok, detail)
        check("moov moved to the front", moov_is_at_front(legacy))
        after = _probe_codecs(legacy)
        check("remux did not re-encode the video", before["vcodec"] == after["vcodec"])
        check("remux kept the audio codec", before["acodec"] == after["acodec"])
        check("remux did not change the duration",
              abs(before["duration"] - after["duration"]) < 0.3,
              f'{before["duration"]} -> {after["duration"]}')


# --------------------------------------------------------------------------- #
# 4. frontend
# --------------------------------------------------------------------------- #

def test_frontend() -> None:
    print("\n[4] the <video> element asks for the right things")
    body = src("src/components/clipforge/ClipPreview.tsx")

    check('preload="metadata" is set', 'preload="metadata"' in body)
    check("playsInline is set", "playsInline" in body)
    check("the source carries a #t= media fragment", "#t=0.001" in body)
    check("loadedmetadata is handled", "onLoadedMetadata" in body)
    check("durationchange is handled", "onDurationChange" in body)
    check("an error triggers a retry", "setRetryCount" in body and "onError" in body)
    check("the retry is cache-busted", "&r=" in body)
    check("the error card offers a download",
          "Download MP4" in body and "clipDownloadUrl" in body)

    # Duration fallback: the clip's own length, used when duration is NaN or
    # Infinity. This is what turns "0:00 / --:--" into "0:00 / 0:17".
    check("duration falls back to end - start",
          "clip.end - clip.start" in body)
    check("a NaN/Infinity duration is rejected",
          "Number.isFinite(mediaDuration)" in body)

    # m:ss formatting for both counters
    check("the time counter uses fmtClock", body.count("fmtClock(") >= 2)

    # No component may fall back to an <img> silently replacing a playable file
    check("the honest error message is kept", "could not play the rendered file" in body)


def main() -> int:
    test_render_args()
    test_served_file()
    test_backfill()
    test_frontend()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
