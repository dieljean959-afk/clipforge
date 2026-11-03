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
from app import store

from fastapi import HTTPException


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

#: Mean luma (0-255) below which a frame is black rather than "dark".
#:
#: A fade-to-black still passes through ~20 on the luma plane, so this only
#: catches an encode whose pixels are genuinely empty. Used to refuse a clip
#: that would otherwise be presented as rendered while playing pure black —
#: the failure the user sees as "the clip is broken" with no error anywhere.
BLACK_FRAME_LUMA = 8.0


def frame_luma(path: Path, at_sec: float = 0.0) -> float:
    """Mean luma (0-255) of one frame near ``at_sec``. -1.0 when unreadable."""
    seek = max(0.0, float(at_sec))
    args = [ffmpeg_bin(), "-v", "error"]
    if seek > 0:
        # No `-ss 0` for a still image: seeking a one-frame input past t=0
        # yields nothing, which is how every poster luma check returned -1.
        args += ["-ss", f"{seek:.3f}"]
    args += [
        "-i", str(path),
        "-frames:v", "1",
        # Downscale first: the mean of a 32x32 gray frame is the same number as
        # the mean of a 1080x1920 one, for 1/2000th the work.
        "-vf", "scale=32:32:flags=neighbor,format=gray",
        "-f", "rawvideo", "-",
    ]
    try:
        proc = subprocess.run(args, capture_output=True, timeout=60)
    except Exception:
        return -1.0
    data = proc.stdout or b""
    if not data:
        return -1.0
    return sum(data) / len(data)


def black_frame_reason(path: Path, duration: float) -> str | None:
    """Explain why a rendered clip is pitch black, or None when it has picture.

    Samples five points across the clip instead of the first frame only: a clip
    may legitimately start on a fade, and only an encode where EVERY sample is
    empty is a broken render. This is the check that stops "pitch black" clips
    from ever being marked done.
    """
    dur = max(0.0, float(duration or 0.0))
    if dur <= 0:
        return "rendered clip reports no duration"
    fractions = (0.05, 0.25, 0.5, 0.75, 0.92)
    samples = [frame_luma(path, dur * f) for f in fractions]
    readable = [s for s in samples if s >= 0]
    if not readable:
        return "rendered clip has no decodable picture"
    brightest = max(readable)
    if brightest < BLACK_FRAME_LUMA:
        return (
            "rendered clip is pitch black (mean luma "
            f"{brightest:.1f}/255 across {len(readable)} sampled frames)"
        )
    return None


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
    black = black_frame_reason(path, float(meta.get("duration") or 0.0))
    if black:
        raise RenderError(f"{path.name}: {black}")
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

def retry_clip_render(clip_id: str) -> dict:
    """Re-render ONE clip by id, without re-running detection/transcription.

    The repair path for a single clip whose render.state is "failed": it
    re-encodes just that clip from the existing cuts, reusing the cached
    transcript and the (possibly re-translated) caption cues.
    """
    clip = store.get_clip_doc(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    job_id = clip.get("jobId") or ""
    job = store.get_job(job_id) if job_id else None
    if job is None:
        raise HTTPException(404, "job not found")
    if job.get("stage") in {"queued", "grabbing", "transcribing", "finding", "rendering", "exporting"}:
        raise HTTPException(409, "this job is still running — wait for it to finish")
    src = store.find_source_file(job_id)
    if src is None:
        raise HTTPException(400, "source file no longer on disk — this clip cannot be re-rendered")
    # Imported here rather than at module scope: app.stages imports nothing
    # from video, but keeping the dependency lazy avoids an import cycle if it
    # ever does. `StageWorkspace` used to be referenced with no import at all,
    # so this repair path raised NameError the first time anyone called it.
    from app.stages import StageWorkspace

    ws = StageWorkspace(settings.app.paths.data_dir, job_id)
    cuts = ws.read_json("reframe", "cuts.json") or []
    cut = next((c for c in cuts if f"clip_{c.get('order', 0):03d}" == clip_id), None)
    if cut is None:
        try:
            order = int(clip_id.split("_")[-1])
        except ValueError:
            raise HTTPException(400, "could not map clip id to a cut")
        tmp = [c for c in cuts if int(c.get("order", -1)) == order]
        cut = tmp[0] if tmp else None
        if cut is None:
            raise HTTPException(400, "no cut found for this clip")
    if clip.get("renderState") == "done" and clip.get("videoPath"):
        raise HTTPException(409, "this clip is already rendered")
    dest = settings.app.paths.outputs_dir / job_id / f"clip_{int(cut.get('order', 0)):03d}.mp4"
    try:
        meta = encode.render_clip(src, cut, dest, settings, on_log=lambda line: store.append_job_log(job_id, line))
    except RenderError as exc:
        store.write_clip(clip_id, {"renderState": "failed", "mediaError": str(exc), "jobId": job_id}, job_id=job_id)
        raise HTTPException(400, str(exc))
    # The poster has to be regenerated too: a re-render overwrites the MP4, so
    # the old poster is a frame from the previous encode. It also used to be
    # written as "000.thumb.jpg" (no `clip_` prefix), a name nothing serves.
    thumb_path = dest.parent / f"clip_{int(cut.get('order', 0)):03d}.thumb.jpg"
    try:
        render_first_frame_thumb(dest, thumb_path)
    except Exception as exc:  # noqa: BLE001 - the clip matters, the poster is best-effort
        logger.warning(f"[video] re-render poster failed for {clip_id}: {exc}")
    store.write_clip(
        clip_id,
        {
            "videoPath": meta["videoPath"],
            "thumbPath": str(thumb_path),
            "mediaBytes": meta.get("bytes") or int(dest.stat().st_size),
            "mediaWidth": meta["width"],
            "mediaHeight": meta["height"],
            "mediaDurationSec": round(float(meta["duration"]), 3),
            "fps": meta.get("fps", 30.0),
            "renderState": "done",
            "jobId": job_id,
        },
        job_id=job_id,
    )
    return {"queued": True, "clipId": clip_id}

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
    position. When a sample is marked "cut": True (a detected shot
    boundary) the x SNAPS to the new shot's position instead of interpolating
    across the cut — the subject jumps at a cut, so a smooth pan through it
    would be a lie.

    This is the 8-arg entry point the render path calls: (src_w, src_h, fps,
    crop_x, crop_y, crop_w, crop_h, sample_windows). It builds the per-sample
    window list from the cropBox + optional sampleWindows. The crop geometry
    is normalized from the source (center crop, 9:16 window) before the
    expression is built.
    """
    # Normalize the crop geometry from the source frame so the 9:16 window
    # always fits: the source's height is the limiting dimension for a
    # vertical frame, so the window width is scaled proportionally and
    # centered.
    crop_w = min(int(src_h * crop_w / crop_h), src_w)
    crop_x = int((src_w - crop_w) / 2)
    crop_y = 0
    windows = _crop_windows(
        crop_x, crop_y, crop_w, crop_h, fps, sample_windows,
    )
    if not windows:
        return f"crop={crop_w}:{crop_h}:(in_w-{crop_w})/2:0"

    # Multiple sample windows: join each window's filter pair and drop the
    # final "crop=0:0" the old code produced after the last sample.
    parts = []
    for i, w in enumerate(windows):
        parts.append(
            f"crop={w['w']}:{w['h']}:({w['x']}):({w['y']}),"
            f"scale={w['w']}:{w['h']}:force_original_aspect_ratio=increase"
        )
    return ",".join(parts)


def _crop_windows(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    fps: float,
    sample_windows: list[dict] | None,
) -> list[dict]:
    """Build the per-sample crop windows from the cropBox + sampleWindows.

    sampleWindows (when present) is the reframe-stage trace with clip-RELATIVE
    sample timestamps, where t=0 is the start of the CLIP (after the -ss/-t
    seek), not necessarily the start of the source. We keep those timestamps
    exactly as-is so the filter expression is consistent with the clip-relative
    ffmpeg timeline.
    """
    if not sample_windows:
        base: dict[str, Any] = {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}
        return [dict(base, t=0.0)]
    out: list[dict] = []
    for w in sample_windows:
        if not isinstance(w, dict):
            continue
        t = float(w.get("t", 0.0))
        out.append(dict(w, t=t))
        out[-1].setdefault("x", crop_x)
        out[-1].setdefault("y", crop_y)
        out[-1].setdefault("w", crop_w)
        out[-1].setdefault("h", crop_h)
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

    The caller's 1080x1920 target is a VERTICAL frame, so it first SCALES the
    horizontal source UP (smearing it to cover the whole frame) and then
    CROPS the CENTRE to 9:16. This is the opposite of the legacy 'scale-down
    + pad' chain, which left black bars above and below a letterboxed subject.

    * The first `scale` uses `force_original_aspect_ratio=increase`, which
      scatters source pixels beyond the target box and then `crop` trims the
      overflow. No pad/no bars — the frame is 100% filled by stretched pixels.
    * The `crop` is the CENTRE window, so the subject stays framed, not
      nudged to one edge.
    * `setsar=1` pins the sample aspect ratio to square pixels so the player
      can't letterbox the already-full-bleed stream.
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
        # Crop away the source's excess height before we ever scale to the
        # target frame, so the whole thing stays full-bleed.
        crop_y = max(0, (src_h - crop_w * 1920 // 1080) // 2)
        chain.append(f"crop={crop_w}:{crop_w * 1920 // 1080}:{src_w / 2 - crop_w / 2}:{crop_y}")
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
            "-filter_complex", f"{vf},subtitles={ass_escaped}",
            *codec_args,
            *AUDIO_ARGS,
            *browser_safe_args(enc.encoder),
            "-f", "mp4", str(dest),
        ]
        logger.info("[video] ffmpeg encode args: " + " ".join(cmd))
    else:
        cmd = [
            ffmpeg_bin(), "-y", *seek,
            "-filter_complex", vf,
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

def _thumb_at(clip_path: Path, thumb_path: Path, at_sec: str, *, thumbnail_filter: bool) -> None:
    """Write one poster frame at ``at_sec`` (a str so it can be empty)."""
    vf = f"scale={THUMB_WIDTH}:-2:flags=bicubic"
    args = [ffmpeg_bin(), "-y"]
    if at_sec:
        args += ["-ss", at_sec]
    args += ["-i", str(clip_path)]
    if thumbnail_filter:
        # analyse the opening seconds and let ffmpeg pick the most
        # representative frame (histogram-based) rather than frame 0
        args += ["-vf", f"thumbnail=100,{vf}"]
    else:
        args += ["-vf", vf]
    args += [
        "-frames:v", "1",
        # -2 keeps the 9:16 aspect with an even height, which mjpeg requires.
        # Pin encoder + muxer: `thumb_path` is caller-supplied, and inferring
        # either from its extension is the same class of bug as the `.wav.part`
        # extraction failure.
        "-c:v", "mjpeg", "-q:v", "4",
        "-f", "image2", str(thumb_path),
    ]
    _run(args, timeout=120)


def render_first_frame_thumb(clip_path: Path, thumb_path: Path) -> Path:
    """Poster for a rendered clip — a REAL frame, chosen so it is not black.

    Frame 0 was the whole rule, and a clip that opens on a fade from black (or
    a dark scene) got a pure-black poster. The card then looked like a failed
    render even though the MP4 played fine. ffmpeg's `thumbnail` filter picks
    the most representative frame of the opening batch, which skips the fade;
    if the result is still black (a genuinely dark clip) the poster is taken a
    couple of seconds in, where the subject is usually lit.
    """
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    _thumb_at(clip_path, thumb_path, "", thumbnail_filter=True)
    if frame_luma(thumb_path) < BLACK_FRAME_LUMA:
        _thumb_at(clip_path, thumb_path, "2.0", thumbnail_filter=False)
    return thumb_path

def thumb_needs_rewrite(p: Path) -> bool:
    """Should this thumbnail poster be rewritten?

    A thumbnail is stale when it is smaller than what a real 320px-desktop
    poster should be, because every card in the grid is scaled down to a few
    hundred CSS pixels and the 7 KB holder is a leftover 1080x1920 full-frame
    render from a pre-sizing pass. A file that is 1080x1920 already rendered
    (proper geometry, non-trivial mtime) is never stale and is returned as-is,
    so vertical 9:16 previews never re-burn a fresh frame after the first few
    seconds of playback.
    """
    try:
        st = p.stat()
    except OSError:
        return False
    if st.st_size <= 24_000:
        return True
    # Always keep a properly sized geometry: size alone is the old check, and
    # geometry is the only reliable way to tell a full-frame 1080x1920 poster
    # from a 320px desktop placeholder (the two overlap on file size).
    try:
        fm = probe(p)
    except Exception:
        return True
    w = int(fm.get("width") or 0)
    h = int(fm.get("height") or 0)
    if w == 1080 and h == 1920:
        return False
    return True