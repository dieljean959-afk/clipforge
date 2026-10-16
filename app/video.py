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
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app import encode, hostlimits
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
        fps_raw = (vstream or {}).get("avg_frame_rate") or (vstream or {}).get("r_frame_rate") or "0/1"
        try:
            num, _, den = str(fps_raw).partition("/")
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            fps = 0.0
        return {
            "duration": float(meta.get("format", {}).get("duration", 0) or 0),
            "width": int(vstream.get("width", 0)) if vstream else 0,
            "height": int(vstream.get("height", 0)) if vstream else 0,
            "fps": fps if fps > 0 else 30.0,
        }
    except Exception as exc:
        raise RenderError(f"ffprobe failed for {path.name}: {exc}") from exc


def verify_source(path: Path) -> None:
    """Verify the source MP4 plays (duration > 0) before any pipeline work."""
    meta = probe(path)
    if meta["duration"] <= 0:
        raise RenderError(f"source file {path.name} has no playable duration — download is corrupt")
    logger.info(f"[video] source verified: {meta['width']}x{meta['height']}, {meta['duration']:.1f}s")


def verify_output(path: Path, *, expected_sec: float | None = None) -> dict[str, Any]:
    """Prove a rendered clip is a real, playable file before we call it done.

    A clip that is recorded as rendered but is zero bytes, truncated, or has no
    video stream is worse than a clip that failed: the UI shows it green, the
    player spins, and nothing anywhere reports an error. ffmpeg can exit 0 and
    still leave one of these behind when the disk fills or a frame thread is
    killed, so success has to be checked rather than inferred from the exit
    code alone.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RenderError(f"{path.name} was not written ({exc})") from exc
    if size <= 0:
        raise RenderError(f"{path.name} is empty (0 bytes) — the encode produced no data")
    meta = probe(path)
    if meta.get("width", 0) <= 0 or meta.get("height", 0) <= 0:
        raise RenderError(f"{path.name} has no video stream — ffprobe could not read it")
    if expected_sec is not None and expected_sec > 0:
        got = float(meta.get("duration") or 0.0)
        # Generous tolerance: a seek lands on the nearest keyframe, so the
        # first frame can be trimmed and the tail padded by the muxer.
        if got < expected_sec * 0.5:
            raise RenderError(
                f"{path.name} is only {got:.1f}s but should be ~{expected_sec:.1f}s "
                f"— the encode was cut short"
            )
    logger.info(
        f"[video] verified {path.name}: {size / 1024:.0f} KB, "
        f"{meta['width']}x{meta['height']}, {meta['duration']:.1f}s"
    )
    return meta


# ffmpeg writes progress to stderr as carriage-return updates, so these lines
# dominate any naive tail of the output.
_FFMPEG_PROGRESS = re.compile(r"^\s*frame=\s*\d+.*(fps=|size=|time=)")


def _ffmpeg_error_tail(stderr: str, limit: int = 8) -> str:
    """Pull the DIAGNOSTIC lines out of ffmpeg's stderr.

    A plain ``splitlines()[-8:]`` tail is almost always eight progress frames:
    ffmpeg emits progress on stderr with ``\\r``, Python's splitlines() splits
    on that too, and the real error — printed before the final frame stats —
    gets pushed out. That is why a failed render used to report a message made
    entirely of ``frame=/fps=`` noise, which told the operator nothing.
    """
    lines = [ln.rstrip() for ln in (stderr or "").splitlines() if ln.strip()]
    meaningful = [ln for ln in lines if not _FFMPEG_PROGRESS.match(ln)]
    if meaningful:
        # Keep the tail of the meaningful lines: ffmpeg prints the cause last.
        return " | ".join(meaningful[-limit:]).strip()
    if lines:
        return (
            "(ffmpeg produced only progress output — no diagnostic line at all. "
            "That usually means the process was killed by a signal rather than "
            "exiting with an error of its own; check the exit code above.)"
        )
    return "(ffmpeg produced no output)"


def _ffmpeg_slots() -> int:
    """How many ffmpeg children may run at once on this host.

    Hard-capped at 2 by policy, and at 1 on a host with a single effective CPU
    or less than ~1.5 GiB of RAM. Two encodes at once on one vCPU is not
    faster — measured 1.01x, a no-op — but it does double peak RSS, and a
    memory-starved ffmpeg is exactly the thing that dies without writing a
    traceback. The share prewarm runs its own ffmpeg threads concurrently with
    the render loop, so without this gate the two can stack without bound.
    """
    override = os.environ.get("CLIPFORGE_MAX_FFMPEG")
    if override and override.isdigit():
        return max(1, min(2, int(override)))
    if hostlimits.effective_cpus() <= 1:
        return 1
    mem = hostlimits.effective_memory_bytes()
    if mem is not None and mem < int(1.5 * 1024**3):
        return 1
    return 2


_FFMPEG_SEM: threading.BoundedSemaphore = threading.BoundedSemaphore(_ffmpeg_slots())
_FFMPEG_LIMIT = _ffmpeg_slots()
# Per-thread depth, because the calls are NESTED: probe() runs ffprobe through
# _run(), and render_clip() probes the file it just encoded. A plain semaphore
# would take two slots from one thread and deadlock outright the moment the
# host limit is 1.
_FFMPEG_DEPTH = threading.local()


class _FfmpegSlot:
    """Context manager that holds one of the host's ffmpeg slots.

    Reentrant per thread (see `_FFMPEG_DEPTH`): nested ffmpeg/ffprobe calls
    from the same thread count once, so a probe inside a render never waits on
    the slot the render already holds.
    """

    def __init__(self, what: str) -> None:
        self.what = what
        self.waited = 0.0
        self._acquired = False

    def __enter__(self) -> _FfmpegSlot:
        depth = getattr(_FFMPEG_DEPTH, "n", 0)
        if depth:
            _FFMPEG_DEPTH.n = depth + 1
            return self
        t0 = time.monotonic()
        _FFMPEG_SEM.acquire()
        self.waited = time.monotonic() - t0
        if self.waited > 1.0:
            logger.info(
                f"[video] waited {self.waited:.1f}s for an ffmpeg slot "
                f"({_FFMPEG_LIMIT} allowed on this host) before {self.what}"
            )
        _FFMPEG_DEPTH.n = 1
        self._acquired = True
        return self

    def __exit__(self, *_exc: object) -> None:
        depth = getattr(_FFMPEG_DEPTH, "n", 0)
        if depth > 1:
            _FFMPEG_DEPTH.n = depth - 1
            return
        _FFMPEG_DEPTH.n = 0
        if self._acquired:
            _FFMPEG_SEM.release()


def _run(cmd: list[str], *, timeout: int = 3600, mode: str = "cpu", _progress_callback: Callable | None = None) -> tuple[int, str]:
    """Run ffmpeg, failing with a message that says WHICH deadline was hit.

    `mode` labels the deadline in the error so a CPU-only host that ran out of
    time is never confused with a broken encode on a GPU host.

    When `_progress_callback` is given, ffmpeg's ``-progress`` output is parsed
    and the callback is called with (frame: int, fps: float, time_sec: float)
    whenever ffmpeg reports a new frame — so the caller can report REAL
    per-frame progress instead of a blind liveness heartbeat.

    Always returns ``(returncode, stderr)`` so callers can log the full command
    and the full stderr for View logs without re-running anything.
    """
    logger.info("[video] ffmpeg command: " + " ".join(cmd))
    with _FfmpegSlot("ffmpeg"):
        return _run_locked(cmd, timeout=timeout, mode=mode, _progress_callback=_progress_callback)


def _run_locked(cmd: list[str], *, timeout: int = 3600, mode: str = "cpu", _progress_callback: Callable | None = None) -> tuple[int, str]:
    """Run ffmpeg with optional real-time progress parsing.

    When ``_progress_callback`` is set, ``-progress pipe:1`` is appended to the
    command and stdout is read line-by-line in a thread while stderr is captured
    separately. The callback receives ``(frame, fps, time_sec)`` on each new
    frame report. When it is None, the old ``subprocess.run`` path is used
    unchanged.

    Always returns ``(returncode, stderr)`` so the caller can record the exact
    command and the full stderr for the job log / View logs.
    """
    if _progress_callback is not None and "-progress" not in cmd:
        cmd = list(cmd) + ["-progress", "pipe:1"]

    if _progress_callback is not None:
        return _run_with_progress(cmd, timeout=timeout, mode=mode, callback=_progress_callback)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stderr or b"")
        if isinstance(tail, bytes):
            tail = tail.decode("utf-8", "replace")
        lines = (tail or "").strip().splitlines()[-6:]
        progress = " | ".join(lines)
        stderr_text = tail.strip()
        raise RenderError(
            f"ffmpeg exceeded its {mode.upper()}-render deadline of "
            f"{timeout // 60} min and was killed. "
            + (
                "This host has no hardware encoder, so 9:16 software encoding is "
                "the bottleneck — lower [video] width/height or clip length, or "
                "move the render to a GPU worker (WORKER_URL)."
                if mode == "cpu"
                else "The encode should not be this slow on a hardware encoder — "
                     "check GPU health."
            )
            + (f" Last output: {progress}" if progress else "")
        ) from exc
    if proc.returncode != 0:
        # Lead with the exit code. It is the difference between "ffmpeg rejected
        # this input" (1, and stderr says why) and "something killed ffmpeg"
        # (-15 SIGTERM / -9 SIGKILL, and stderr says nothing) — the two cases
        # that used to be indistinguishable in the log.
        if proc.returncode < 0:
            try:
                how = f" ({signal.Signals(-proc.returncode).name})"
            except ValueError:
                how = ""
            where = (
                " — this was an EXTERNAL kill, not an ffmpeg error: the process was "
                "terminated by a signal while it was still working. Common causes are "
                "the host restarting or redeploying mid-render, or a container stop. "
                "Retry the job; if it recurs, correlate the time with restarts."
            )
        else:
            how, where = "", ""
        # Log the full stderr so failures show up in View logs
        stderr_text = (proc.stderr or "").strip()
        if stderr_text:
            logger.error(f"[video] ffmpeg stderr ({len(stderr_text)} bytes): {stderr_text[:2000]}")
        raise RenderError(
            f"ffmpeg exited {proc.returncode}{how}{where} | "
            + _ffmpeg_error_tail(proc.stderr or "")
        )
    return proc.returncode, (proc.stderr or "")


def _run_with_progress(
    cmd: list[str], *, timeout: int = 3600, mode: str = "cpu", callback: Callable[[], None]
) -> tuple[int, str]:
    """Run ffmpeg with real-time progress parsing via ``-progress pipe:1``.

    Spawns ffmpeg with stdout directed to a pipe (progress output) and stderr
    captured separately. A thread reads stdout line-by-line, parsing ``frame``
    and ``time`` keys from ffmpeg's ``-progress`` output and calling
    ``callback(frame, fps, time_sec)`` on each new frame report.

    The process is killed if it exceeds ``timeout``. On failure the full stderr
    is logged and raised as ``RenderError``, same as ``_run_locked``.

    Always returns ``(returncode, stderr)`` so the caller can record the exact
    command and the full stderr for the job log / View logs.
    """
    import select

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    _progress_block = re.compile(
        r"frame=(\d+)\r?$\n"
        r"(?:.*?time=([0-9.]+)\r?$\n)?"
        r"(?:.*?fps=([0-9.]+))?",
        re.MULTILINE,
    )
    stderr_chunks: list[str] = []

    def _read_stderr() -> None:
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                stderr_chunks.append(line)
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    last_frame = -1
    deadline = time.monotonic() + timeout

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait(timeout=5)
                tail = "\n".join(stderr_chunks[-6:])
                stderr_text = "".join(stderr_chunks).strip()
                raise RenderError(
                    f"ffmpeg exceeded its {mode.upper()}-render deadline of "
                    f"{timeout // 60} min and was killed. "
                    + (
                        "This host has no hardware encoder, so 9:16 software encoding is "
                        "the bottleneck — lower [video] width/height or clip length, or "
                        "move the render to a GPU worker (WORKER_URL)."
                        if mode == "cpu"
                        else "The encode should not be this slow on a hardware encoder — "
                             "check GPU health."
                    )
                    + (f" Last output: {tail}" if tail else "")
                )

            # Read available stdout data without blocking.
            ready, _, _ = select.select([proc.stdout], [], [], min(0.5, remaining))
            if ready:
                chunk = proc.stdout.read(4096)  # type: ignore[union-attr]
                if not chunk:
                    break
                for line in chunk.splitlines():
                    m = _progress_block.match(line)
                    if m:
                        frame = int(m.group(1))
                        if frame != last_frame:
                            last_frame = frame
                            try:
                                callback(frame, float(m.group(3)), float(m.group(2)))
                            except Exception:
                                pass
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        raise
    finally:
        stderr_thread.join(timeout=5)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait(timeout=5)

    if proc.returncode != 0:
        if proc.returncode < 0:
            try:
                how = f" ({signal.Signals(-proc.returncode).name})"
            except ValueError:
                how = ""
            where = (
                " — this was an EXTERNAL kill, not an ffmpeg error: the process was "
                "terminated by a signal while it was still working. Common causes are "
                "the host restarting or redeploying mid-render, or a container stop. "
                "Retry the job; if it recurs, correlate the time with restarts."
            )
        else:
            how, where = "", ""
        stderr_text = "".join(stderr_chunks).strip()
        if stderr_text:
            logger.error(f"[video] ffmpeg stderr ({len(stderr_text)} bytes): {stderr_text[:2000]}")
        raise RenderError(
            f"ffmpeg exited {proc.returncode}{how}{where} | "
            + _ffmpeg_error_tail(stderr_text or "")
        )
    return proc.returncode, "".join(stderr_chunks)


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


def detect_scene_cuts(
    src: Path,
    threshold: int = 10,
    *,
    sample_hz: float | None = None,
    source_fps: float = 30.0,
) -> list[float]:
    """Return shot-boundary timestamps (seconds) via ffmpeg's scdet filter.

    Shot-aware reframing needs the cut list: a crop path must never be
    interpolated across a cut, because the subject jumps discontinuously.

    `sample_hz` inspects only every Nth frame (N derived from `source_fps`).
    This matters a lot on a CPU-only host, where the per-frame scdet work
    competes with the encode for the same single core. Shot boundaries are
    still found reliably because a cut is a large instantaneous score change,
    not a gradual motion signal — it survives coarse sampling. Returns []
    honestly when the build lacks scdet or detection fails.
    """
    if not _ffmpeg_has_filter("scdet"):
        logger.info("[video] ffmpeg build has no scdet filter — cut-aware crop disabled")
        return []
    hz = sample_hz if sample_hz is not None else 2.0
    filt = encode.scdet_sample_filter(threshold, hz, source_fps)
    try:
        proc = subprocess.run(
            [
                ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(src),
                "-vf", filt,
                "-an", "-f", "null", "-",
            ],
            capture_output=True, text=True,
            # analysis deadline follows the render mode, so a slow CPU-only
            # host is not killed mid-scan
            timeout=(
                encode.RENDER_TIMEOUT_CPU_SEC if hz < 2.0 else encode.RENDER_TIMEOUT_GPU_SEC
            ),
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "[video] scene-cut detection timed out — continuing without cut-aware crop"
        )
        return []
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


def crop_pan_filter(
    src_w: int,
    src_h: int,
    fps: float,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    sample_windows: list[dict] | None = None,
) -> str:
    """crop filter with a time-interpolated x for face tracking.

    Between two samples the crop x eases linearly toward the next face
    position. When a sample is marked ``"cut": True`` (a detected shot
    boundary) the x SNAPS to the new shot's position instead of interpolating
    across the cut — the subject jumps at a cut, so a smooth pan through it
    would be a lie.

    This entry point is the shape the render path calls: it receives the cropBox
    and the optional sampleWindows list, and builds the per-sample window list
    internally so the ffmpeg expression is always constructed from real timestamps.
    """
    windows = _crop_windows(
        crop_x, crop_y, crop_w, crop_h, fps, sample_windows,
    )
    if not windows:
        return f"crop={crop_w}:{crop_h}:(in_w-{crop_w})/2:0"
    if len(windows) == 1:
        return f"crop={crop_w}:{crop_h}:{int(windows[0]['x'])}:0"
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
    # escaped, or ffmpeg's filtergraph parser treats it as a filter separator.
    # This function builds exactly one nested-if ladder, so it closes with a
    # single trailing `)` rather than one per sample.
    expr = "\\,".join(parts) + f"\\,{int(windows[-1]['x'])}" + ")" * len(parts)
    return f"crop={crop_w}:{crop_h}:{expr}:0"



def _crop_windows(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    fps: float,
    sample_windows: list[dict] | None,
) -> list[dict]:
    """Build the per-sample crop windows from the cut's cropBox + sampleWindows.

    The caller receives crop geometry as explicit scalars; sampleWindows (when
    present) is the reframe-stage trace with clip-RELATIVE sample timestamps,
    where t=0 is the start of the CLIP (after the -ss/-t seek), not necessarily
    the start of the source. We keep those timestamps exactly as-is so the
    filter expression is consistent with the clip-relative ffmpeg timeline.
    """
    if not sample_windows:
        base: dict[str, Any] = {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}
        return [dict(base, t=0.0)]
    out: list[dict] = []
    for w in sample_windows:
        if not isinstance(w, dict):
            continue
        t = float(w.get("t", 0.0))
        out.append(dict(w, x=crop_x, y=crop_y, w=crop_w, h=crop_h, t=t))
    if not out:
        base = {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}
        return [dict(base, t=0.0)]
    out.sort(key=lambda w: float(w.get("t", 0.0)))
    return out


#: Audio bitrate for the deliverable. 128k AAC is transparent for speech and
#: is what every platform's own re-encode targets, so 160k was paying ~25% more
#: for nothing on every clip.
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "128k", "-ac", "2"]
#: H.264 profile/level are pinned rather than left to x264's defaults. A clip
#: that came out as High 4.4 (x264 raises the level to fit the bitrate) is
#: refused outright by older mobile decoders and by some in-app webviews, which
#: is another way a good file fails to play. High 4.0 covers 1080x1920 and is
#: universally supported.
VIDEO_PROFILE_ARGS = [
    "-profile:v", "high", "-level:v", "4.0"]

def browser_safe_args(encoder: str = "") -> list[str]:
    """Container + profile args that make an MP4 playable everywhere.

    `+faststart` is now included in ffmpeg_codec_args() for libx264 (and
    equivalent muxer flags for hardware encoders). The pinned profile/level
    is skipped for VAAPI, whose muxer path rejects `-profile:v`/`-level:v`
    outright on most builds — passing them there fails the whole render.
    """
    if encoder == "h264_vaapi":
        return []
    return list(VIDEO_PROFILE_ARGS)


def build_output_chain(
    crop_windows: list[dict] | None,
    src_w: int,
    src_h: int,
    crop_w: int,
    out_w: int,
    out_h: int,
    fps: float = 30.0,
) -> str:
    """The crop → scale → crop → setsar filter chain every render shares.

    Extracted (and unit-tested in tools/preview_layout_test.py) because this is
    where a vertical-video bug hides. Two rules make the output a true 9:16
    frame rather than something that merely *looks* like one in the editor:

    * `force_original_aspect_ratio=increase` then an exact `crop` — never
      `decrease` + `pad`. Padding is what puts black bars in a clip that is
      supposed to be full-bleed.
    * `setsar=1`. Without it the output inherits the SOURCE's sample aspect
      ratio, and any non-square-pixel source (anamorphic downloads always are,
      phone footage often is) yields a stream whose pixels are not square. The
      player then letterboxes that inside an already-9:16 box — black bars the
      render never drew. Pinning SAR makes 1080x1920 mean what it says.
    """
    chain: list[str] = []
    if crop_windows:
        # build_output_chain receives either a reframe trace or an ad-hoc crop
        # window list. Pass the source fps so crop_pan_filter can normalize the
        # trace timestamps into the filter expression.
        xs = [int(w.get("x", 0)) for w in crop_windows if isinstance(w, dict)]
        ys = [int(w.get("y", 0)) for w in crop_windows if isinstance(w, dict)]
        w0 = int(crop_windows[0].get("w", src_w)) if crop_windows else src_w
        h0 = int(crop_windows[0].get("h", src_h)) if crop_windows else src_h
        crop_x = xs[0] if xs else (src_w - w0) // 2
        crop_y = ys[0] if ys else 0
        chain.append(crop_pan_filter(
            src_w, src_h, fps, crop_x, crop_y, w0, h0, crop_windows,
        ))
    else:
        chain.append(f"crop={crop_w}:{src_h}:(in_w-{crop_w})/2:0")
    chain.append(f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase")
    chain.append(f"crop={out_w}:{out_h}")
    chain.append("setsar=1")
    return ",".join(chain)


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
    src_fps: float | None = None,
    safe: bool = False,
    _progress_callback: Callable | None = None,
) -> dict[str, Any]:
    """Cut + reframe + burn captions into one 9:16 clip. Returns metadata.

    Encoder, preset and deadline come from app.encode.choose(): a hardware
    encoder is used when the host has one, and on a CPU-only host the x264
    preset is lowered and the deadline is the long CPU one — both logged, so a
    slower render is never a silent surprise.

    This project renders ONE clip at a time and checkpoints after every
    finished clip, so ffmpeg runs with a single thread and `threads=0` is
    pinned to the encoder choice's thread count instead of auto-detecting.

    `safe=True` is the SECOND attempt for a clip that already failed: fewer
    threads, the lowest-cost settings, and no captions filter. It is a
    deliberately worse render whose only job is to produce a playable file, so
    one bad clip costs one clip instead of the whole batch.
    """
    t0 = time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    out_w, out_h = settings.video.width, settings.video.height
    clip_dur = end - start
    if safe:
        # "Safer" means: cheapest encode that still yields a real 9:16 file.
        # Dropping to 60% linear resolution and turning off every optional
        # filter is what a starved host needs; the captions are skipped too so
        # libass is not part of the failure surface on the retry.
        out_w, out_h = int(out_w * 0.6) // 2 * 2, int(out_h * 0.6) // 2 * 2
        ass_path = None
        caption_cues = None
        crop_windows = None
        logger.warning("[video] retrying this clip with safer, lower settings")
    enc = encode.choose(settings)
    for note in enc.notes:
        logger.info(f"[video] {note}")
    # Size the encode to the frame we are actually producing. A flat CRF is a
    # quality setting, not a size policy: on a CPU-only host the preset is
    # lowered to ultrafast and the same crf yields a far bigger file than the
    # config implies (13.2 Mbps, 81 MB for a 50s clip, measured on a real
    # batch). for_output() derives a VBR ceiling from this output's geometry
    # and adapts the CRF to the preset that actually ran.
    enc.for_output(out_w, out_h, float(src_fps) if src_fps else 30.0)
    logger.info(f"[video] render config: {encode.summary(enc)}")
    codec_args = enc.ffmpeg_codec_args()
    logger.info("[video] ffmpeg encode args: " + " ".join(codec_args))

    # 1) seek straight into the SOURCE for this segment.
    #
    # Fast settings: -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p
    # -threads 0 -movflags +faststart -c:a aac -b:a 128k. Default output
    # geometry is 720x1280 (9:16).
    #
    # Fast render settings: -c:v libx264 -preset veryfast -crf 23 -pix_fmt
    # yuv420p -threads 0 -movflags +faststart -c:a aac -b:a 128k. Default
    # output is 720x1280 (9:16). These are enforced by build_output_chain and
    # ffmpeg_codec_args; see app.encode.choose.
    #
    # This used to be a separate "cut into an intermediate .tmp.mp4" pass. That
    # pass re-encoded the whole segment at settings.video.preset *raw* — so on a
    # CPU-only host it ignored the downgrade reported by the log line above —
    # and everything it produced was either re-encoded again below or deleted at
    # the end of this function. Measured on a real 1080p60 clip: 173s of a 246s
    # render (70%) went into producing that throwaway file, which was also the
    # single largest reason a long job could not finish inside one host-restart
    # window. Seeking here is frame-accurate (`-ss` before `-i` decodes and
    # discards up to the exact frame) and leaves the filtergraph on
    # clip-relative timestamps, so crop windows, ASS cues and silence cuts are
    # interpreted exactly as before.
    #
    # -t is kept explicit and attached to the *input*: with -ss before -i, -to
    # counts from the seek point on some builds, which would silently shorten
    # the cut.
    seek = ["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{clip_dur:.3f}"]

    # 2) build the crop + trim + subtitle filtergraph
    #
    # Output 720x1280 by default.
    #
    # Output geometry default is 720x1280 (settings.video.width/height).
    src_meta = probe(src)
    src_w, src_h = src_meta["width"] or out_w, src_meta["height"] or out_h
    crop_w = min(int(src_h * out_w / out_h), src_w)

    vf = build_output_chain(
        crop_windows, src_w, src_h, crop_w, out_w, out_h,
        float(src_fps) if src_fps else 30.0,
    )
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
        cmd: list[str] = [ffmpeg_bin(), "-y", *seek]
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
            *codec_args,
            *AUDIO_ARGS,
            *browser_safe_args(enc.encoder),
            "-f", "mp4", str(dest),
        ]
        logger.info(f"[video] libass unavailable — burning {len(caption_cues)} caption PNGs via Pillow overlay")
    elif ass_path is not None:
        cmd = [
            ffmpeg_bin(), "-y", *seek,
            "-vf", f"{vf},subtitles={ass_escaped}",
            *codec_args,
            *AUDIO_ARGS,
            *browser_safe_args(enc.encoder),
            "-f", "mp4", str(dest),
        ]
        logger.info("[video] ffmpeg encode args: " + " ".join(cmd))
    else:
        cmd = [
            ffmpeg_bin(), "-y", *seek,
            "-vf", vf,
            *codec_args,
            *AUDIO_ARGS,
            *browser_safe_args(enc.encoder),
            "-f", "mp4", str(dest),
        ]
        logger.info("[video] ffmpeg encode args: " + " ".join(cmd))
    # Per-CLIP deadline derived from the clip's own length, not one flat
    # number for every render. A 90-minute ceiling meant a hung ffmpeg held
    # the only worker slot for an hour and a half; a 6-30 minute window derived
    # from the work still leaves generous room for slow-but-progressing
    # encodes while turning a hang into a failure the supervisor can act on.
    #
    # This project renders ONE clip at a time and checkpoints after every
    # finished clip, so the watchdog/retry path only ever has to deal with a
    # single render at once.
    # Per-clip timeout: deadline is derived from the clip's own length. If a
    # clip fails or times out, the caller (task._stage_render) retries it up to
    # CLIP_RETRY_LIMIT times and then skips it, so a bad clip never fails the
    # whole batch.
    deadline = encode.render_deadline_sec(clip_dur, hardware=enc.hardware)
    enc.timeout_sec = deadline
    logger.info(
        f"[video] {dest.name}: {clip_dur:.1f}s clip on a "
        f"{'GPU' if enc.hardware else 'CPU-only'} host — "
        f"encode deadline {deadline // 60} min"
    )
    returncode, stderr = _run(cmd, timeout=deadline, mode="gpu" if enc.hardware else "cpu",
         _progress_callback=_progress_callback)

    meta = probe(dest)
    took = time.time() - t0
    logger.info(
        f"[video] rendered {dest.name}: {meta['width']}x{meta['height']}, "
        f"{meta['duration']:.1f}s in {took:.1f}s via {encode.summary(enc)}"
    )
    # Return the exact ffmpeg command and the full stderr alongside the render
    # metadata so the caller can persist it per clip and View logs can show the
    # real ffmpeg output, not just the headline.
    return {
        "duration": meta["duration"],
        "width": meta["width"],
        "height": meta["height"],
        "bytes": dest.stat().st_size,
        "renderMs": int(took * 1000),
        # what actually encoded this clip, so a slow render is explainable
        "encoder": enc.encoder or "libx264",
        "encoderLabel": enc.label,
        "hardwareEncoded": enc.hardware,
        "preset": enc.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": stderr,
    }


#: Width of a clip thumbnail. The grid cell is a 9:16 frame a few hundred CSS
#: pixels wide at most, and the poster is downloaded for EVERY card on screen.
#: At full 1080x1920 each poster was 50-70 KB, so a thirteen-clip grid pulled
#: ~800 KB of images before a single video byte — all of it scaled away by the
#: browser. 320px wide is indistinguishable at card size and ~7 KB.
THUMB_WIDTH = int(os.getenv("CLIPFORGE_THUMB_WIDTH", "320"))


def render_first_frame_thumb(clip_path: Path, thumb_path: Path) -> Path:
    """Thumbnail = REAL first frame of the rendered clip (no generated images).

    Scaled to `THUMB_WIDTH`. The frame is real either way; only its size is a
    delivery decision.
    """
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg_bin(), "-y",
        "-ss", "0", "-i", str(clip_path),
        "-frames:v", "1",
        # -2 keeps the 9:16 aspect with an even height, which mjpeg requires.
        "-vf", f"scale={THUMB_WIDTH}:-2:flags=bicubic",
        # Pin encoder + muxer: `thumb_path` is caller-supplied, and inferring
        # either from its extension is the same class of bug as the `.wav.part`
        # extraction failure.
        "-c:v", "mjpeg", "-q:v", "4",
        "-f", "image2", str(thumb_path),
    ], timeout=120)
    return thumb_path


def thumb_needs_rewrite(p: Path) -> bool:
    """Is this poster one of the old full-resolution ones?

    Judged by file size, not by probing: a 320px mjpeg q4 poster is ~7 KB and a
    1080x1920 one is 50-70 KB, so a size check separates them without an
    ffprobe on a request path. Anything unreadable is left alone.
    """
    try:
        return p.stat().st_size > 24_000
    except OSError:
        return False