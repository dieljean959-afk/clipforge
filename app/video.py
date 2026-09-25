"""Video service: ffmpeg 9:16 rendering with face reframe + burned captions.

Cuts the REAL segment from the source (re-encode, frame-accurate), applies the
face-tracked crop, burns ASS captions in, and writes thumbnails as true first
frames. libx264/AAC, 1080x1920. MoviePy is available for any future composite
needs, but the render path is straight ffmpeg for speed and reliability.

Shot-aware: scene cuts are detected up front (ffmpeg scdet) so the vertical
crop is framed per shot and never interpolates across a cut.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from loguru import logger

from app import subtitle as subtitle_svc
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
    # `-f wav` is explicit for the same reason as app.subtitle._extract_audio:
    # `dest` is caller-supplied and may be a `.part`/temp path, where ffmpeg's
    # extension-based muxer inference fails with "Invalid argument". Pinning
    # the muxer also pins the codec, so the file is always readable by the
    # stdlib `wave` module that app.subtitle._WavReader relies on.
    _run([
        ffmpeg_bin(), "-y", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", "-f", "wav", str(dest),
    ])
    return dest


def _ffmpeg_has_filter(filt: str) -> bool:
    """True when this ffmpeg build supports `filt` (e.g. libass `subtitles`)."""
    try:
        out = subprocess.run(
            [ffmpeg_bin(), "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30,
        )
        return (
            re.search(rf"^\s*\S+\s+{re.escape(filt)}\s", out.stdout, re.MULTILINE)
            is not None
        )
    except Exception:
        return False


def detect_scene_cuts(src: Path, threshold: int = 10) -> list[float]:
    """Return shot-boundary timestamps (seconds) via ffmpeg's scdet filter.

    Shot-aware reframing needs the cut list: a crop path must never be
    interpolated across a cut, because the subject jumps discontinuously.
    Returns [] honestly when the build lacks scdet or detection fails.
    """
    if not _ffmpeg_has_filter("scdet"):
        logger.info("[video] ffmpeg build has no scdet filter — cut-aware crop disabled")
        return []
    try:
        proc = subprocess.run(
            [
                ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(src),
                "-vf", f"scdet=threshold={threshold}:metadata=print",
                "-an", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=3600,
        )
    except Exception as exc:
        logger.warning(f"[video] scene-cut detection failed (continuing without cuts): {exc}")
        return []
    # metadata lines look like:
    #   frame:12  pts:12345  pts_time:12.345
    #   lavfi.scd.score=...  lavfi.scd.time=12.345
    cuts: list[float] = []
    for m in re.finditer(r"lavfi\.scd\.time=([0-9.]+)", proc.stderr or ""):
        try:
            t = float(m.group(1))
        except ValueError:
            continue
        if t > 0 and (not cuts or t - cuts[-1] > 0.25):
            cuts.append(t)
    logger.info(f"[video] scene cuts detected: {len(cuts)}")
    return cuts


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

    Between two samples the crop x eases linearly toward the next face
    position. When a sample is marked ``"cut": True`` (a detected shot
    boundary) the x SNAPS to the new shot's position instead of interpolating
    across the cut — the subject jumps at a cut, so a smooth pan through it
    would be a lie.
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
            if wdw.get("cut"):
                # hold prev_x until the cut, then jump hard to this shot's x
                parts.append(f"if(lt(t\\,{t:.3f})\\,{prev_x}")
            else:
                frac = f"(t-{prev:.3f})/{(t - prev):.6f}" if t > prev else "0"
                parts.append(
                    f"if(lt(t\\,{t:.3f})\\,{prev_x}+({x}-{prev_x})*min(1\\,max(0\\,{frac}))"
                )
    # every comma inside the expression (including between nested ifs) must be
    # escaped, or ffmpeg's filtergraph parser treats it as a filter separator
    expr = "\\,".join(parts) + f"\\,{int(windows[-1]['x'])})" * len(parts)
    return f"crop={out_w}:{out_h}:{expr}:0"


def render_clip(
    src: Path,
    dest: Path,
    *,
    start: float,
    end: float,
    settings: Settings,
    ass_path: Path | None = None,
    caption_cues: list[dict] | None = None,
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
    # Temp name ends in `.mp4`, but the muxer is pinned explicitly below so no
    # ffmpeg call in this module ever depends on extension inference.
    tmp = dest.with_suffix(".tmp.mp4")
    _run([
        ffmpeg_bin(), "-y",
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{clip_dur:.3f}",
        "-c:v", "libx264", "-preset", settings.video.preset, "-crf", str(settings.video.crf),
        "-c:a", "aac", "-b:a", "160k",
        "-f", "mp4", str(tmp),
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

    # Caption burn path: libass `subtitles` filter when the build has it;
    # otherwise the pure-Python Pillow fallback (cue PNGs + overlay) so captions
    # export either way. Never silently drop captions.
    ass_escaped = (
        str(ass_path).replace("\\", "/").replace(":", "\\:")
        if ass_path is not None
        else None
    )
    use_pillow = (
        ass_path is not None
        and caption_cues
        and not _ffmpeg_has_filter("subtitles")
    )
    if use_pillow:
        png_dir = dest.parent / f"{dest.stem}_cues"
        pngs = subtitle_svc.render_word_pngs_pillow(
            caption_cues, settings, png_dir, out_w, out_h, settings.video.fps,
        )
        cmd: list[str] = [ffmpeg_bin(), "-y", "-i", str(tmp)]
        for cue in caption_cues:
            cmd += ["-i", pngs[f"{cue['start']:.3f}-{cue['end']:.3f}"]]
        chain: list[str] = []
        prev = "[base]"
        for i, cue in enumerate(caption_cues, 1):
            outl = f"[v{i}]"
            chain.append(
                f"{prev}[{i}:v]overlay=0:0:enable='between(t,{cue['start']:.3f},{cue['end']:.3f})'{outl}"
            )
            prev = outl
        cmd += [
            "-filter_complex", f"[0:v]{vf}[base];" + ";".join(chain),
            "-map", prev, "-map", "0:a?",
            "-c:v", "libx264", "-preset", settings.video.preset, "-crf", str(settings.video.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-f", "mp4", str(dest),
        ]
        logger.info(f"[video] libass unavailable — burning {len(caption_cues)} caption PNGs via Pillow overlay")
    elif ass_path is not None:
        cmd = [
            ffmpeg_bin(), "-y", "-i", str(tmp),
            "-vf", f"{vf},subtitles={ass_escaped}",
            "-c:v", "libx264", "-preset", settings.video.preset, "-crf", str(settings.video.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-f", "mp4", str(dest),
        ]
    else:
        cmd = [
            ffmpeg_bin(), "-y", "-i", str(tmp),
            "-vf", vf,
            "-c:v", "libx264", "-preset", settings.video.preset, "-crf", str(settings.video.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-f", "mp4", str(dest),
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
        # Pin encoder + muxer: `thumb_path` is caller-supplied, and inferring
        # either from its extension is the same class of bug as the `.wav.part`
        # extraction failure.
        "-c:v", "mjpeg", "-q:v", "3",
        "-f", "image2", str(thumb_path),
    ], timeout=120)
    return thumb_path
