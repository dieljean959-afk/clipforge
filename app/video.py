"""Video service: ffmpeg 9:16 rendering with face reframe + burned captions.

Cuts the REAL segment from the source (re-encode, frame-accurate), applies the
face-tracked crop, burns ASS captions in, and writes thumbnails as true first
frames. libx264/AAC, 1080x1920. MoviePy is available for any future composite
needs, but the render path is straight ffmpeg for speed and reliability.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import Settings, ffmpeg_bin, ffprobe_bin


class RenderError(Exception):
    pass


def probe(path: Path) -> dict[str, Any]:
    """ffprobe a media file → {duration, width, height}."""
    try:
        out = subprocess.run(
            [
                ffprobe_bin(), "-v", "error",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path),
            ],
            capture_output=True, text=True, timeout=60, check=True,
        )
        meta = json.loads(out.stdout)
        vstream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
        return {
            "duration": float(meta.get("format", {}).get("duration", 0) or 0),
            "width": int(vstream.get("width", 0)) if vstream else 0,
            "height": int(vstream.get("height", 0)) if vstream else 0,
        }
    except Exception as exc:
        raise RenderError(f"ffprobe failed for {path.name}: {exc}") from exc


def verify_source(path: Path) -> None:
    """Verify the source MP4 plays (duration > 0) before any pipeline work."""
    meta = probe(path)
    if meta["duration"] <= 0:
        raise RenderError(f"source file {path.name} has no playable duration — download is corrupt")
    logger.info(f"[video] source verified: {meta['width']}x{meta['height']}, {meta['duration']:.1f}s")


def _run(cmd: list[str], *, timeout: int = 3600) -> None:
    logger.debug("[video] " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise RenderError("ffmpeg failed: " + " | ".join(tail))


def extract_wav(src: Path, dest: Path) -> Path:
    _run([ffmpeg_bin(), "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dest)])
    return dest


def silence_trim_filter(cuts: list[dict], total_dur: float) -> tuple[str, float]:
    """Build an ffmpeg filter that drops [start,end] windows (clip-relative).

    Returns (filtergraph, new_duration). Uses select/asetpts with an expression
    built from the cut list; empty cuts => passthrough.
    """
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for c in sorted(cuts, key=lambda x: x["start"]):
        if c["start"] > cursor:
            keep.append((cursor, min(c["start"], total_dur)))
        cursor = max(cursor, c["end"])
    if cursor < total_dur:
        keep.append((cursor, total_dur))
    if not keep or len(keep) == 1 and abs(keep[0][0]) < 1e-6 and abs(keep[0][1] - total_dur) < 1e-6:
        return "", total_dur

    conds = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
    new_dur = sum(b - a for a, b in keep)
    return (
        f"select='{conds}',setpts=N/FRAME_RATE/TB,"
        f"aselect='{conds}',asetpts=N/SR/TB",
        new_dur,
    )


def crop_pan_filter(windows: list[dict], src_w: int, src_h: int, out_w: int, out_h: int) -> str:
    """crop filter with a time-interpolated x for face tracking.

    Builds a nested ffmpeg if(lt(t,x), ...) expression: between samples the
    crop x eases linearly toward the next face position.
    """
    if not windows:
        return f"crop={out_w}:{out_h}:(in_w-out_w)/2:0"
    if len(windows) == 1:
        return f"crop={out_w}:{out_h}:{int(windows[0]['x'])}:0"
    parts: list[str] = []
    for i, wdw in enumerate(windows):
        t = wdw["t"]
        x = int(wdw["x"])
        if i == 0:
            parts.append(f"if(lt(t\\,{t:.3f})\\,{x}")
        else:
            prev = windows[i - 1]["t"]
            prev_x = int(windows[i - 1]["x"])
            frac = f"(t-{prev:.3f})/{(t - prev):.6f}" if t > prev else "0"
            parts.append(
                f"if(lt(t\\,{t:.3f})\\,{prev_x}+({x}-{prev_x})*min(1\\,max(0\\,{frac}))"
            )
    # every comma inside the expression (including between nested ifs) must be
    # escaped, or ffmpeg's filtergraph parser treats it as a filter separator
    expr = "\\,".join(parts) + f"\\,{int(windows[-1]['x'])}" + ")" * len(parts)
    return f"crop={out_w}:{out_h}:{expr}:0"


def render_clip(
    src: Path,
    dest: Path,
    *,
    start: float,
    end: float,
    settings: Settings,
    ass_path: Path | None = None,
    crop_windows: list[dict] | None = None,
    silence_cuts: list[dict] | None = None,
) -> dict[str, Any]:
    """Cut + reframe + burn captions into one 9:16 clip. Returns metadata."""
    t0 = time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    out_w, out_h = settings.video.width, settings.video.height
    clip_dur = end - start

    # 1) cut segment (re-encode, frame-accurate) into an intermediate.
    # Input-seeking with an explicit -t duration: unambiguous across ffmpeg
    # versions (with -ss before -i, -to counts from the seek point on some
    # builds, which would silently shorten the cut).
    tmp = dest.with_suffix(".tmp.mp4")
    _run([
        ffmpeg_bin(), "-y",
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{clip_dur:.3f}",
        "-c:v", "libx264", "-preset", settings.video.preset, "-crf", str(settings.video.crf),
        "-c:a", "aac", "-b:a", "160k",
        str(tmp),
    ])

    # 2) build the crop + trim + subtitle filtergraph
    src_meta = probe(tmp)
    src_w, src_h = src_meta["width"] or out_w, src_meta["height"] or out_h
    crop_w = min(int(src_h * out_w / out_h), src_w)

    chain: list[str] = []
    if crop_windows:
        chain.append(crop_pan_filter(crop_windows, src_w, src_h, crop_w, src_h))
    else:
        chain.append(f"crop={crop_w}:{src_h}:(in_w-{crop_w})/2:0")
    chain.append(f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase")
    chain.append(f"crop={out_w}:{out_h}")

    vf = ",".join(chain)
    if silence_cuts:
        trim_graph, _ = silence_trim_filter(silence_cuts, clip_dur)
        if trim_graph:
            # split select for video only; audio passthrough keeps sync close enough
            vf = vf + "," + trim_graph.split(",asetpts")[0]

    if ass_path is not None:
        ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
        cmd = [
            ffmpeg_bin(), "-y", "-i", str(tmp),
            "-vf", f"{vf},subtitles={ass_escaped}",
            "-c:v", "libx264", "-preset", settings.video.preset, "-crf", str(settings.video.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(dest),
        ]
    else:
        cmd = [
            ffmpeg_bin(), "-y", "-i", str(tmp),
            "-vf", vf,
            "-c:v", "libx264", "-preset", settings.video.preset, "-crf", str(settings.video.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(dest),
        ]
    _run(cmd)
    tmp.unlink(missing_ok=True)

    meta = probe(dest)
    took = time.time() - t0
    logger.info(
        f"[video] rendered {dest.name}: {meta['width']}x{meta['height']}, "
        f"{meta['duration']:.1f}s in {took:.1f}s"
    )
    return {
        "duration": meta["duration"],
        "width": meta["width"],
        "height": meta["height"],
        "bytes": dest.stat().st_size,
        "renderMs": int(took * 1000),
    }


def render_first_frame_thumb(clip_path: Path, thumb_path: Path) -> Path:
    """Thumbnail = REAL first frame of the rendered clip (no generated images)."""
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg_bin(), "-y",
        "-ss", "0", "-i", str(clip_path),
        "-frames:v", "1",
        "-q:v", "3",
        str(thumb_path),
    ], timeout=120)
    return thumb_path
