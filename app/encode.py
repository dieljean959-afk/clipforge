"""Encode capability + graceful degradation for the render stage.

The render step is the one place where a small host is most likely to fall
over: a 1080p60 source encoded with libx264 on a single vCPU runs at roughly
0.1x realtime, so a 15-second clip takes minutes and a long source effectively
never        finishes. Nothing here changes output CORRECTNESS — it changes how hard
we push the CPU, and it is explicit about the trade rather than silently
degrading.

Three independent levers, in order of preference:

  1. Hardware encoder (NVENC / VAAPI / QSV) when ffmpeg and the host have
     one. Offload is ~free, so the GPU-host quality settings are kept as-is.
  2. A faster x264 preset on CPU-only hosts. `veryfast` → `ultrafast` costs
     compression efficiency, not correctness, and is worth ~2-3x on one core.
  3. A render deadline sized to the mode, so a CPU-only host gets the long
     deadline it needs and a GPU host is not made to wait for it.

Detection is cached: `ffmpeg -encoders` costs a process spawn, and the
render stage asks once per clip.
"""

from __future__ import annotations

import functools
import os
import re
import shlex as _shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app import hosting
from app import hostlimits
from app.config import Settings, ffmpeg_bin

# --------------------------------------------------------------------------- #
# timeouts — deliberately NOT shared
# --------------------------------------------------------------------------- #

RENDER_TIMEOUT_GPU_SEC = 20 * 60
#: A single-CPU host can legitimately spend minutes per 9:16 clip. The old code
#: reused the GPU deadline here, which is exactly what produced "the render
#: just stops" on small hosts. This one is sized for slow-but-progressing.
#:
#: It was then set to 90 minutes, which was worse: a hung ffmpeg held the ONLY
#: worker slot for an hour and a half, and every other job sat at 0% behind it
#: with a "~any moment" estimate. The deadline is now derived from the actual
#: clip length by `render_deadline_sec()` — generous for slow-but-progressing
#: work, but bounded, so a hang costs minutes rather than an hour.
RENDER_TIMEOUT_CPU_SEC = 12 * 60
#: Floor for the derived deadline, so a short clip still gets room for the
#: process spawn, scene analysis and the burn-in pass.
RENDER_DEADLINE_FLOOR_SEC = 6 * 60
#: Ceiling, so a very long clip can never hold a worker slot indefinitely.
RENDER_DEADLINE_CEILING_SEC = 30 * 60
#: Wall-seconds of encode allowed per second of output. Measured on this host:
#: a 1080x1920 60fps clip encodes at roughly 1x realtime with the ultrafast
#: fallback, so 6x is a wide margin that still catches a real hang.
RENDER_REALTIME_FACTOR = 6
#: Probing ffmpeg for encoders/filters is quick but not instant on a loaded box.
PROBE_TIMEOUT_SEC = 30


def render_deadline_sec(clip_seconds: float, *, hardware: bool) -> int:
    """Wall-clock budget for ONE clip render.

    A flat deadline is the wrong shape in both directions. Too short and a
    legitimately slow clip on a 1-CPU host is killed mid-encode; too long and
    a hung ffmpeg pins the only worker slot for over an hour while the queue
    behind it claims to be "any moment away".

    So the budget scales with the work: a floor for        fixed overhead, a
    per-second-of-output allowance for the encode itself, and a hard ceiling.
    """
    if hardware:
        # GPU encodes an order of magnitude faster than realtime; keep the
        # historical, tighter deadline rather than the CPU budget.
        return RENDER_TIMEOUT_GPU_SEC
    raw = RENDER_DEADLINE_FLOOR_SEC + max(0.0, clip_seconds) * RENDER_REALTIME_FACTOR
    return int(min(RENDER_DEADLINE_CEILING_SEC, max(RENDER_DEADLINE_FLOOR_SEC, round(raw))))    #: x264 presets ordered fastest-first. CPU-only hosts get one of the        first two.
_PRESET_RANK = (
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
)
#: What we drop to on a CPU-only host, and the absolute floor we never pass.
_CPU_PRESET = "veryfast"
_CPU_PRESET_FLOOR = "veryfast"


@dataclass
class EncoderChoice:
    """What the render will actually use, and why."""

    #: "" = libx264 (software). Otherwise e.g. "h264_nvenc", "h264_vaapi".
    encoder: str = ""
    #: Label safe to show in a log line / health payload.
    label: str = "x264 (software)"
    #: True when a hardware encoder is doing the work.
    hardware: bool = False
    preset: str = "veryfast"
    crf: int = 23
    hw_accel: str = ""
    timeout_sec: int = RENDER_TIMEOUT_GPU_SEC
    #: Populated when the requested preset had to be lowered.
    degraded_from: str = ""
    notes: list[str] =        field(default_factory=list)
    #: Adaptive VBR ceiling for this output, in kbps. 0 = no ceiling (CRF
    #: alone, the old behaviour). Set by `for_output()`.
    maxrate_kbps: int = 0
    bufsize_kbps: int = 0
    #: Thread count handed to the encoder.
    #: 0 = ffmpeg auto-detects all available cores (uses -threads 0).
    #: >0 = pinned to that count (used when cgroup quota is known).
    #:
    #: This project renders ONE clip at a time and checkpoints after every
    #:        finished clip, so ffmpeg should not compete with itself across multiple
    #: encode threads. Pin to 1 for deterministic CPU behavior on the target
    #: host; 0 (autodetect) is still the default elsewhere.
    threads: int = 0

    def threads_args(self) -> list[str]:
        """`-threads` for the encoder and the        filter graph.

        -threads 0 tells ffmpeg to auto-detect and use all available cores.
        On modern kernels this respects cgroup limits. We always pass it
        explicitly so the behavior is deterministic across ffmpeg versions.
        """
        return [
            "-threads", str(self.threads),
            "-filter_threads", str(self.threads),
            "-filter_complex_threads", str(self.threads),
        ]

    def emoji_threads_args(self) -> list[str]:
        """Threads for a filter_complex pass that carries emoji overlays.

        A 1 CPU / 2 GiB host has nothing to parallelise: letting ffmpeg spawn
        one filter thread per core it *thinks* it has spends the whole budget
        on scheduling and, on this container, is the difference between the
        emoji pass costing 10 % and 80 % more than the plain one. One thread,
        deterministically, whenever the host reports a single usable core.
        """
        try:
            from app import hostlimits

            cpus = int(hostlimits.limits().get("cpus") or 0)
        except Exception:  # noqa: BLE001 - never fail a render over a thread hint
            cpus = 0
        n = 1 if cpus <= 1 else max(1, self.threads)
        return [
            "-threads", str(n),
            "-filter_threads", str(n),
            "-filter_complex_threads", str(n),
        ]

    def for_output(self, width: int, height: int, fps: float) -> "EncoderChoice":
        """Size the encode to the output it is actually producing.

        A flat CRF is not a size policy. On a CPU-only host the preset may be
        lowered (superfast), which disables lookahead/CABAC, so the same crf
        produces a larger        file. The VBR ceiling is derived from output geometry
        (bits per pixel per frame) so a 30fps clip gets half the ceiling of a
        60fps one at the same resolution.

        The CRF is adapted to the preset: a degraded preset gets a higher CRF
        since the cap does the rate limiting, and a capable preset keeps or
        lowers it.

        Returns self so it can be chained onto `choose()`.
        """
        self.maxrate_kbps = target_bitrate_kbps(width, height, fps)
        self.bufsize_kbps = self.maxrate_kbps * 2
        if self.preset == "superfast":
            self.crf = min(28, self.crf + 2)
        elif self.preset == "veryfast":
            self.crf = max(16, self.crf - 1)
        elif _preset_rank(self.preset) >= _preset_rank("medium"):
            self.crf = max(16, self.crf - 1)
        return self

    def ffmpeg_codec_args(self) -> list[str]:
        """The ``-c:v``/quality args for the chosen encoder.

        When `for_output()` has set a ceiling, the encode becomes constrained
        VBR: CRF still decides quality per frame, and the ceiling stops a hard
        scene from producing an enormous        file. Easy content stays well under
        the cap, so this costs nothing on simple footage.

        libx264 and QSV get -movflags +faststart for web playback; NVENC and
        VAAPI get their equivalent muxer flags.
        """
        cap: list[str] = []
        if self.maxrate_kbps > 0:
            cap = ["-maxrate", f"{self.maxrate_kbps}k", "-bufsize", f"{self.bufsize_kbps}k"]
        thr = self.threads_args()
        if self.encoder == "h264_nvenc":
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p4" if self.preset == "ultrafast" else "p5",
                "-cq", str(self.crf),
                "-b:v", "0",
                *thr,
                *cap,
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
        if self.encoder == "h264_vaapi":
            # VAAPI takes a bitrate, not a CRF; prefer the adaptive ceiling,
            # and fall back to scaling the CRF when there is no output geometry.
            kbps = self.maxrate_kbps or max(1200, int(self.crf * 180))
            return [
                "-c:v", "h264_vaapi",
                "-b:v", f"{kbps}k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
        if self.encoder == "h264_qsv":
            return [
                "-c:v", "h264_qsv",
                "-preset", "veryfast" if self.preset == "ultrafast" else "medium",
                "-global_quality", str(self.crf),
                "-look_ahead", "0",
                *thr,
                *cap,
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
        return [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-threads", "0", "-movflags", "+faststart",
        ]

    def hw_accel_args(self) -> list[str]:
        """Decoder-side acceleration args (empty when not applicable)."""
        if not self.hw_accel:
            return []
        return ["-hwaccel", self.hw_accel]


def _preset_rank(name: str) -> int:
    try:
        return _PRESET_RANK.index(name)
    except ValueError:
        # Unknown/veryfast-family name: treat as mid-pack rather than fastest.
        return _PRESET_RANK.index("veryfast")


def _presets_available() -> set[str]:
    """Encoder names ffmpeg was built with."""
    try:
        out = subprocess.run(
            [ffmpeg_bin(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SEC,
        ).stdout or ""
    except Exception as exc:  # noqa: BLE001 - any failure means "no hw"
        logger.debug(f"[encode] encoder probe failed: {exc}")
        return set()
    found: set[str] = set()
    for line in out.splitlines():
        m = re.match(r"\s*V[.A-Z0-9]{5}\s+(\S+)", line)
        if m:
            found.add(m.group(1))
    return found


def _device_present(kind: str) -> bool:
    if kind == "nvenc":
        nvidia_smi = hosting.shutil_which('nvidia-smi') or hostlimits.shutil_which('nvidia-smi')
        return os.path.exists('/dev/nvidia0') or os.path.exists('/dev/nvidia-uvm') or bool(nvidia_smi)
    if kind == "vaapi":
        return os.path.exists("/dev/dri/renderD128") or os.path.exists("/dev/dri/card0")
    if kind == "qsv":
        return os.path.exists("/dev/dri")
    return False


def _encoder_works(encoder: str, hw_accel: str) -> bool:
    """Actually try to encode ONE frame with `encoder`.

    A build listing `h264_vaapi` plus a `/dev/dri` node is NOT proof of a
    usable GPU: containers routinely expose the device node while the driver
    is absent or the session is headless. Trusting the listing would turn a
    merely-slow software render into a hard "unknown encoder"/"no such device"
    failure, so we pay one cheap encode to        find out for real.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="clipforge-hwprobe-") as td:
        out = f"{td}/probe.mp4"
        cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        if hw_accel and encoder != "h264_nvenc":
            # VAAPI needs the device explicitly; QSV/others auto-negotiate.
            if hw_accel == "vaapi":
                cmd += ["-vaapi_device", "/dev/dri/renderD128"]
        cmd += [
            "-f", "lavfi", "-i", "color=c=black:size=64x64:rate=1",
            "-frames:v", "1",
            "-c:v", encoder,
            "-f", "mp4", out,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SEC
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[encode] {encoder} probe raised: {exc}")
            return False
        if proc.returncode != 0:
            err_lines = (proc.stderr or "").strip().splitlines()
            reason = (err_lines[-1] if err_lines else "no diagnostic")[:120]
            logger.info(
                f"[encode] {encoder} is listed but not usable here ({reason}) "
                f"— staying on x264"
            )
            return False
        return os.path.exists(out) and os.path.getsize(out) > 0


@functools.lru_cache(maxsize=1)


def hardware_encoder() -> tuple[str, str, str]:
    """Pick the best usable encoder for this host.

    Returns (encoder_name, label, hw_accel). encoder_name may be '' for libx264.
    """
    # CPU-only host: never attempt GPU paths.
    if not hosting.shutil_which('nvidia-smi') and not Path('/dev/nvidia0').exists() and not Path('/dev/nvidia-uvm').exists():
        return '', 'x264 (software)', ''
    # If a GPU device is present, still only claim what ffmpeg can actually use.
    for enc in ('h264_nvenc', 'h264_vaapi', 'h264_qsv'):
        if _encoder_works(enc, enc.split('_')[1].lower()):
            return enc, enc, enc.split('_', 1)[1]
    return '', 'x264 (software)', ''


def reset_cache() -> None:
    """Drop the memoised probe (tests only)."""
    hardware_encoder.cache_clear()


#: Bits per pixel per frame for the VBR ceiling. 0.06 puts 1080x1920 at 60fps
#: near 7.5 Mbps, which is comfortably above what any social platform keeps
#: after its own re-encode, and is a third of what a flat crf 20 produced on
#: this host. Clamped so a tiny or a huge output still gets a sane ceiling.
BITS_PER_PIXEL = 0.06
MIN_MAXRATE_KBPS = 2_500
MAX_MAXRATE_KBPS = 14_000


def target_bitrate_kbps(
    width: int,
    height: int,
    fps: float,
    *,
    bpp: float = BITS_PER_PIXEL,
) -> int:
    """The VBR ceiling for an output of this geometry, in kbps.

    Derived from the OUTPUT rather than pinned to a constant, which is what
    makes the cap adaptive: the same policy gives a 30fps clip half the
    ceiling of a 60fps one at the same resolution, and scales with resolution
    instead of assuming 1080x1920 forever.
    """
    safe_fps = fps if fps and fps > 1 else 30.0
    raw = width * height * safe_fps * bpp / 1000.0
    return int(max(MIN_MAXRATE_KBPS, min(MAX_MAXRATE_KBPS, round(raw))))


def resolve_preset(requested: str) -> tuple[str, str]:
    """Clamp `requested` to something a CPU-only host can        finish.

    Returns (preset, note). The note is empty when nothing was changed.
    """
    req = (requested or "veryfast").strip()
    if _preset_rank(req) <= _preset_rank(_CPU_PRESET_FLOOR):
        return req, ""
    if _preset_rank(req) > _preset_rank(_CPU_PRESET):
        return _CPU_PRESET, (
            f"preset {req} → {_CPU_PRESET} on this CPU-only host "
            f"(~2-3x faster, slightly larger        files)"
        )
    return req, ""


def choose(settings: Settings, *, force_software: bool = False) -> EncoderChoice:
    """Decide the encoder, preset, quality and deadline for this render."""
    requested = settings.video.preset
    crf = settings.video.crf

    if force_software:
        encoder, label, hw = "", "x264 (software, forced)", ""
    else:
        encoder, label, hw = hardware_encoder()

    choice = EncoderChoice(
        encoder=encoder,
        label=label,
        hardware=bool(encoder),
        hw_accel=hw,
        crf=crf,
    )

    # Use -threads 0 so ffmpeg auto-detects and uses all available cores.
    # ffmpeg's internal auto-detection respects cgroup limits on modern kernels,
    # so this gives the best utilization without manual thread pinning.
    choice.threads = 1

    if encoder:
        # Hardware encodes fast enough that a slow preset is affordable, so the
        # user's quality settings are kept. Deadline stays the strict one.
        choice.preset = requested
        choice.timeout_sec = RENDER_TIMEOUT_GPU_SEC
        return choice

    # --- software path: degrade deliberately and say so ---------------------
    preset, note = resolve_preset(requested)
    if note:
        choice.degraded_from = requested
        choice.notes.append(note)
    choice.preset = preset
    choice.timeout_sec = RENDER_TIMEOUT_CPU_SEC  # replaced per clip by render_deadline_sec()
    choice.notes.append(
        f"CPU-only render deadline: {RENDER_TIMEOUT_CPU_SEC // 60} min "
        f"(hardware encoder absent — a single-vCPU host encodes 9:16 at ~0.1x realtime)"
    )
    return choice


def analysis_sample_hz(source_fps: float, *, hardware: bool) -> float:
    """How often to sample frames for reframing/scene analysis.

    On a hardware encoder the CPU is free, so keep the accurate dense sampling.
    On a CPU-only host the analysis competes with the encode for the same core,
    so drop to a coarser rate: fewer MediaPipe detections and fewer scdet
    frames, at the cost of slightly coarser crop tracking. Shot boundaries are
    still detected because scdet runs on sampled frames, not on motion, and
    cuts are far more frequent than 1 Hz.
    """
    if hardware:
        return 2.0
    return 0.75


def scdet_sample_filter(threshold: int, sample_hz: float, source_fps: float) -> str:
    """scdet        filtergraph that only inspects every Nth frame.

    `select='not(mod(n\\,N))'` drops frames before scdet sees them, so the
    decoder still runs but the per-frame score/compare work shrinks by N.
    N is derived from the source frame rate so the *sampling rate in seconds*
    stays roughly constant regardless of 30 vs 60 fps input.
    """
    stride = max(1, int(round(source_fps / max(0.1, sample_hz))))
    if stride <= 1:
        return f"scdet=threshold={threshold}:metadata=print"
    return f"select='not(mod(n\\,{stride}))',scdet=threshold={threshold}:metadata=print"


def summary(choice: EncoderChoice) -> str:
    """One-line, greppable description of the render configuration."""
    bits = [choice.label, f"preset {choice.preset}", f"crf {choice.crf}"]
    if choice.maxrate_kbps:
        bits.append(f"maxrate {choice.maxrate_kbps // 1000} Mbps")
    if choice.degraded_from:
        bits.append(f"(downgraded from {choice.degraded_from})")
    bits.append(f"deadline {choice.timeout_sec // 60} min")
    return " | ".join(bits)


def as_dict(choice: EncoderChoice) -> dict[str, Any]:
    return {
        "encoder": choice.encoder or "libx264",
        "label": choice.label,
        "hardware": choice.hardware,
        "preset": choice.preset,
        "crf": choice.crf,
        "maxrateKbps": choice.maxrate_kbps,
        "degradedFrom": choice.degraded_from,
        "timeoutSec": choice.timeout_sec,
        "notes": choice.notes,
    }


def render_clip(
    src: Path,
    cut: dict[str, Any],
    dest: Path,
    settings: Settings,
    *,
    cues: list[dict[str, Any]] | None = None,
    preset_key: str | None = None,
    emoji_selection: list[dict[str, Any]] | None = None,
    title: str | None = None,
    on_log: Callable[[str], None] | None = None,
    audio_src: str | None = None,
    probe_meta: dict[str, Any] | None = None,
    start_offset: float | None = None,
) -> dict[str, Any]:
    """Render one reframe cut into a 9:16 MP4.

    This is the per-clip render entry point the export stage calls. It picks the
    encoder for this host, builds the crop/pan filter graph from the cut's cropBox
    and sampleWindows, burns the subtitle cues into the frame, and writes the
    finished clip with web-playback flags.

    `cut` is a dict produced by ``reframe.find_cuts`` (one per clip). It must
    contain at least ``start``/``end`` and ``cropBox``; ``sampleWindows`` is
    optional and improves crop tracking when present.

    Returns the render metadata dict (so the export stage can persist real
    per-clip status + the ffmpeg command/stderr for View logs).
    """
    from app import video as _video

    # A source can now be a presigned URL (the bucket is the library; the disk
    # only ever holds the clip being written) or a local Path (uploads). The
    # render is otherwise identical — same graph, same captions, same title
    # bar — because ffmpeg reads both through the same `-i` argument.
    remote = isinstance(src, str) and src.startswith(("http://", "https://"))
    src = str(src) if remote else Path(src)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if on_log:
        on_log(f"Rendering clip {cut.get('order', '?')} — {cut.get('start')}s to {cut.get('end')}s")

    # Source-file guard: fail loud with the real file reason before we try to seek.
    # A remote source has no local file to stat — its integrity was proven by
    # the HEAD check when it was uploaded, and the HEAD check again below.
    if not remote:
        if not src.exists() or not src.is_file():
            raise _video.RenderError(
                f"source file for clip {cut.get('order', '?')} is missing: {src}"
            )
        try:
            src_stat = src.stat()
        except OSError as exc:
            raise _video.RenderError(
                f"source file for clip {cut.get('order', '?')} is unreadable: {exc}"
            ) from exc
        if src_stat.st_size == 0:
            raise _video.RenderError(
                f"source file for clip {cut.get('order', '?')} is empty (0 bytes): {src.name}"
            )

    # Probing a presigned URL costs a round trip per clip; the fetch stage
    # already probed this exact stream, so reuse its geometry.
    if remote and probe_meta and int(probe_meta.get("width") or 0) > 0:
        probe = {"duration": 0.0, "fps": 30.0, **probe_meta}
    else:
        probe = _video.probe(src)
    width = int(probe["width"])
    height = int(probe["height"])
    fps = float(probe.get("fps", 30.0) or 30.0)
    duration = float(probe["duration"])

    start = float(cut.get("start", 0.0))
    end = float(cut.get("end", duration))
    clip_seconds = max(0.1, end - start)

    # The deliverable geometry, not the source's. The clip is cropped to a
    # 9:16 window around the subject and then SCALED up to this frame; using
    # the source height for the bitrate ceiling (and the source dimensions for
    # the caption layout) produced 404x720 files while the UI advertised
    # 1080x1920.
    out_w = int(settings.video.width)
    out_h = int(settings.video.height)

    choice = choose(settings).for_output(out_w, out_h, fps)
    deadline = render_deadline_sec(clip_seconds, hardware=choice.hardware)
    choice.timeout_sec = deadline

    crop = cut.get("cropBox", {})
    crop_x = int(crop.get("x", 0))
    crop_y = int(crop.get("y", 0))
    crop_w = int(crop.get("w", width))
    crop_h = int(crop.get("h", height))

    samples = cut.get("sampleWindows") or []
    # TARGET A: per-shot framing mode. The reframe stage decides FIT vs FILL
    # per shot segment and records the choice (and the reason) on the cut.
    # FILL keeps today's tight 9:16 crop; FIT scales the FULL 16:9 frame to
    # 1080px wide, centers it vertically over a blurred copy of the same
    # frame (scaled to cover, gaussian sigma ~30). Segments come as a list:
    # [{"start": 0.0, "end": 2.1, "mode": "fit", "reason": "..."}, ...]
    # (absolute source seconds). Segments disappear (or never arrive) →
    # every clip is FILL exactly as before, so existing batches don't change.
    fit_segments = cut.get("framingSegments") or []

    if fit_segments:
        clip_segments = [
            {
                "start": max(0.0, float(s.get("start", 0.0)) - start),
                "end": min(clip_seconds, float(s.get("end", 0.0)) - start),
                "mode": str(s.get("mode", "fill")),
            }
            for s in fit_segments
            if float(s.get("end", 0.0)) > start and float(s.get("start", 0.0)) < end
        ]
    else:
        clip_segments = [{"start": 0.0, "end": clip_seconds, "mode": "fill"}]

    def _fit_graph() -> str:
        """FULL-frame FIT: source shown whole over its blurred self.

        The blurred pad is the SAME frame scaled to COVER 1080x1920 (it must
        be bigger than the frame, not letterboxed inside it), gaussian-blurred
        sigma 30, then the sharp copy is scaled to 1080 wide (~608 high for a
        16:9 source) and centered. ``setsar=1`` pins square pixels on both
        branches the same way build_output_chain does.
        """
        return (
            f"split=2[bg][fg];"
            f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},gblur=sigma=30,setsar=1[bgb];"
            f"[fg]scale={out_w}:-2:flags=bicubic,setsar=1[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
        )

    has_fit = any(s["mode"] == "fit" for s in clip_segments)
    fills = [s for s in clip_segments if s["mode"] != "fit"]
    if has_fit and not fills:
        # whole clip FIT: full frame shown over its blurred self
        filtergraph = _fit_graph()
    elif has_fit:
        # mixed FILL/FIT within one clip. Both branches render the FULL
        # 1080x1920 frame; the switch is a hard cut done by ``overlay`` with
        # ``enable=between(...)`` — when enabled it covers the base with the
        # FIT frame (same size), when disabled the FILL base shows through.
        # No frames are dropped or re-timed, so audio stays in sync and the
        # switch lands exactly on the reframe stage's chosen boundaries
        # (guaranteed >=1s apart and never mid-word).
        fill_chain = _video.build_output_chain(
            samples, width, height, crop_w, out_w, out_h, fps,
        )
        fit_cond = "+".join(
            f"between(t,{s['start']:.3f},{s['end']:.3f})"
            for s in clip_segments if s["mode"] == "fit"
        )
        filtergraph = (
            f"split=2[modesrc][fitref];"
            f"[modesrc]{fill_chain},setsar=1[fillbase];"
            f"[fitref]{_fit_graph()},setsar=1[fitscale];"
            f"[fillbase][fitscale]overlay=0:0:enable='{fit_cond}'"
        )
    else:
        # build_output_chain's no-windows branch centre-crops at src height;
        # honour the cut's own cropBox x/y so the subject stays framed.
        filtergraph = (
            _video.crop_pan_filter(
                width, height, fps, crop_x, crop_y, crop_w, crop_h, None,
            )
            + f",scale={out_w}:{out_h}:force_original_aspect_ratio=increase"
            + f",crop={out_w}:{out_h},setsar=1"
        )

    # Burn the captions into the frames: an ASS file is written per clip from
    # the cue list (clip-relative timings), then applied via ffmpeg's
    # subtitles filter AFTER the crop/pan graph. Clips keep captions even
    # when played on platforms that never read sidecar SRTs.
    ass_path: Path | None = None
    #: Where the caption sits is a LAYOUT question, not a taste one. A FIT
    #: clip shows the source at frame width in a band in the middle of the
    #: 9:16 frame, so a bottom-margin caption lands in the blur UNDER the
    #: video. Anchor it to the foreground's own lower edge instead, computed
    #: from the real crop aspect. Mixed fit/fill clips keep the safe-area
    #: margin: the foreground moves mid-clip, so one anchor would be wrong for
    #: part of it. Computed BEFORE the caption block because the emoji layer
    #: hangs off the same anchor.
    anchor_y: int | None = None
    if has_fit and not fills:
        from app import captions_animated as _anim

        box = cut.get("cropBox") or {}
        try:
            bw, bh = float(box.get("w") or 0.0), float(box.get("h") or 0.0)
        except (TypeError, ValueError):
            bw, bh = 0.0, 0.0
        anchor_y = _anim.foreground_bottom(
            bw if bw > 0 and bh > 0 else width,
            bh if bw > 0 and bh > 0 else height,
            out_w,
            out_h,
        )

    #: Timed emoji PNGs. Two sources: the caller's context-aware selection
    #: (any preset) and, for the animated styles only, the keyword overlays the
    #: style itself ships. Either way they are PNG overlays — libass cannot
    #: draw colour emoji.
    emoji: list[dict] = []
    if emoji_selection:
        from app import emoji as _emoji

        emoji = _emoji.overlays_for_render(
            emoji_selection, settings,
            frame_w=out_w, frame_h=out_h, anchor_y=anchor_y,
            preset_key=preset_key,
        )
    if cues:
        from app import subtitle as _subtitle

        ass_path = dest.with_suffix(".ass")
        # The preset the JOB chose, not None: writing with None resolved every
        # clip to the default look, so a bold-stack job burned the wrong style
        # (and the wrong font dir) while the dashboard said otherwise.
        _subtitle.write_ass(
            cues, settings, ass_path, out_w, out_h,
            preset_key=preset_key,
            anchor_y=anchor_y,
        )

        # Escape for ffmpeg's filter parser: backslash, colon and quote are
        # all meaningful inside a filter argument.
        esc = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        # Per-preset fonts dir: the Viral look ships Montserrat inside app/fonts
        # (host has no system Montserrat), so libass must be pointed at it.
        font_dir = _subtitle.ass_font_dir(settings, preset_key)
        if font_dir:
            font_path = str(font_dir).replace(":", "\\:")
            ass_arg = "subtitles=" + esc + ":fontsdir=" + font_path
        else:
            ass_arg = "subtitles=" + esc
        filtergraph = filtergraph + "," + ass_arg

        # No hook title layer: a second subtitles= filter (one Dialogue line per
        # clip titled "Hook N") burned an unwrapped, full-width black-on-white
        # bar across the top of the frame on top of the real captions. The
        # target's examples keep the hook only while the first words speak, so
        # a caller can attach it with its own Dialogue timing if it wants a
        # heading; the render path keeps the caption layer alone.

    # White title bar (LOCKED_SPEC §2): a full-width white plate near the top
    # with the clip's hook title. Drawn to a PNG by app.titlebar, which
    # measures every line with the real font and shrinks/wraps so nothing can
    # ever run off an edge; it is then overlaid for the whole clip. One extra
    # input and one overlay — no second libass pass, no guesswork.
    title_png: Path | None = None
    title_top = 0
    if title and str(title).strip():
        from app import titlebar as _titlebar

        plan = _titlebar.plan(str(title), frame_w=out_w)
        title_png = _titlebar.render_png(plan, dest.with_suffix(".title.png"))
        title_top = plan.top

    # Emoji layer (animated styles only) is emitted above the caption as timed
    # PNG overlays — it is the only second layer the render is allowed to
    # produce, and it carries the same SafeArea/insets as the text below.
    # The keyword emoji layer belonged to the animated styles, which no longer
    # exist (LOCKED_SPEC §1). The Twemoji overlay that remains is the
    # context-aware selection above — chosen once per clip in the captions
    # stage and burned in this same single pass.

    # Encode to a temp file and only then replace the destination. A render
    # that is killed mid-write (host restart, stall watchdog, a re-render
    # overlapping the previous one) used to leave a PARTIALLY WRITTEN MP4 at
    # the real path: it still probes with the right duration, but decodes to
    # garbage/black frames later in the clip — the exact "the clip is broken
    # but the app says done" failure. `os.replace` is atomic on one filesystem,
    # so a reader either sees the previous complete file or the new complete
    # one, never a half-written one.
    tmp_dest = dest.with_name(dest.stem + ".part.mp4")

    # ---- ffmpeg execution -------------------------------------------------
    # The command is rebuilt per attempt so a failed first pass can retry with
    # a different filter chain / encoder instead of aborting the clip. Both
    # re-invocations use the same seek semantics, so the output geometry and
    # captions are unchanged.
    def _build(graph: str, enc: Any) -> list[str]:
        # Emoji: an animated style can attach a Twemoji PNG to a chunk's key
        # word. PNGs cannot be drawn by libass, so each one becomes a looped
        # input and an overlay windowed with enable=between(). Only styles that
        # ask for emoji take this path; everything else keeps the plain -vf
        # chain, and the static-crop fallback renders without emoji rather than
        # failing the clip over decoration.
        # Inputs: the video stream, and — for a bucket source — the separate
        # audio object. Both get their OWN `-ss` so ffmpeg range-fetches only
        # the clip's bytes from each, instead of reading the file from 0.
        # `seek` is the position INSIDE the input: for a sectioned source it is
        # the clip's start minus the section's start, which is why it is not
        # simply `start`.
        seek = start if start_offset is None else start_offset
        inputs = ["-ss", f"{seek:.3f}", "-i", str(src)]
        audio_map = "0:a:0?"
        if audio_src:
            inputs += ["-ss", f"{seek:.3f}", "-i", audio_src]
            audio_map = "1:a:0?"

        if emoji or title_png:
            # ONE pass, ONE input per DISTINCT emoji image.
            #
            # The old chain opened the same PNG again for every occurrence and
            # animated it with `scale=...:eval=frame`, which re-runs the scaler
            # on every decoded frame — on a 1 CPU / 2 GiB host that was most of
            # the render time. Now each distinct PNG is opened once, fanned out
            # with `split`, pre-scaled ONCE, and the pop is a y-offset over the
            # first 180 ms plus a short alpha fade (no per-frame scaling).
            chain = [f"[0:v]{graph}[cv0]"]
            extra: list[str] = []
            by_path: dict[str, list[dict]] = {}
            for item in emoji:
                by_path.setdefault(str(item["path"]), []).append(item)
            src_idx = 2 if audio_src else 1
            layer = 0
            for path, items in by_path.items():
                size = int(items[0].get("size") or 140)
                extra += ["-loop", "1", "-t", f"{clip_seconds:.3f}", "-i", path]
                n = len(items)
                if n == 1:
                    chain.append(
                        f"[{src_idx}:v]format=rgba,scale={size}:{size}[em{layer}]"
                    )
                    outs = [f"[em{layer}]"]
                else:
                    # split=1 fan-out: one decode, one scale, N timed overlays.
                    chain.append(
                        f"[{src_idx}:v]format=rgba,scale={size}:{size},split={n}"
                        + "".join(f"[e{layer}_{k}]" for k in range(n))
                    )
                    outs = [f"[e{layer}_{k}]" for k in range(n)]
                for k, item in enumerate(items):
                    s0 = float(item["start"])
                    s1 = float(item["end"])
                    fade_in = min(0.10, max(0.03, (s1 - s0) / 4))
                    fade_out = min(0.12, max(0.03, (s1 - s0) / 4))
                    lift = max(6, int(size * 0.09))
                    chain.append(
                        f"{outs[k]}fade=t=in:st={s0:.3f}:d={fade_in:.3f}:alpha=1,"
                        f"fade=t=out:st={max(0.0, s1 - fade_out):.3f}:d={fade_out:.3f}:alpha=1"
                        f"[em{layer}f{k}]"
                    )
                    emoji_x = int(item.get("x") or (out_w - size) // 2)
                    # Rises `lift` px into place over 180 ms. Quoted, because
                    # the expression contains commas that the filter parser
                    # would otherwise read as filter separators.
                    emoji_y = int(item.get("y") or 0)
                    y_expr = (
                        f"'{emoji_y}+{lift}*(1-min(1,(t-{s0:.3f})/0.18))'"
                    )
                    label = f"cv{layer + 1}"
                    chain.append(
                        f"[cv{layer}][em{layer}f{k}]overlay=x={emoji_x}:y={y_expr}:"
                        f"enable='between(t,{s0:.3f},{s1:.3f})'[{label}]"
                    )
                    layer += 1
                src_idx += 1
            # The title plate goes on LAST, above every other layer, for the
            # whole clip (no enable window).
            out_label = f"cv{layer}"
            if title_png is not None:
                extra += ["-loop", "1", "-t", f"{clip_seconds:.3f}", "-i", str(title_png)]
                chain.append(
                    f"[{out_label}][{src_idx}:v]overlay=x=0:y={title_top}:format=auto[ttl]"
                )
                out_label = "ttl"
            return [
                ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                *enc.hw_accel_args(),
                *inputs,
                *extra,
                *enc.emoji_threads_args(),
                "-filter_complex", ";".join(chain),
                *enc.ffmpeg_codec_args(),
                "-map", f"[{out_label}]",
                "-map", audio_map,
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                # MUST sit next to the output. `-t` placed before another `-i`
                # is read as that INPUT's duration, which left the looped emoji
                # input unbounded and the render running forever.
                "-t", f"{clip_seconds:.3f}",
                str(tmp_dest),
            ]
        return [
            ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            *enc.hw_accel_args(),
            # Seek on the INPUT: `-ss` before `-i` jumps straight to the segment,
            # while `-ss` after `-i` makes ffmpeg decode and discard every frame
            # from 0. On a 21-minute source that turned a 25s clip into minutes of
            # decoding before the first frame reached the encoder. Crop windows
            # and ASS cues are clip-relative (t=0 at the seek point), so this does
            # not shift the reframe or the captions.
            "-ss", f"{seek:.3f}",
            "-i", str(src),
            *(["-ss", f"{seek:.3f}", "-i", audio_src] if audio_src else []),
            "-t", f"{clip_seconds:.3f}",
            *enc.threads_args(),
            "-vf", graph,
            *enc.ffmpeg_codec_args(),
            "-map", "0:v:0",
            "-map", audio_map,
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            str(tmp_dest),
        ]

    def _attempt(graph: str, enc: Any) -> tuple[list[str], Any]:
        cmd = _build(graph, enc)
        if on_log:
            on_log(f"ffmpeg {' '.join(_shlex.quote(a) for a in cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=enc.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise _video.RenderError(
                f"render of clip {cut.get('order', '?')} timed out after "
                f"{enc.timeout_sec}s — the host may be overloaded; Retry will "
                f"resume from the last finished clip"
            ) from exc
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            reason = err[-1] if err else "no diagnostic"
            raise _video.RenderError(
                f"ffmpeg failed on clip {cut.get('order', '?')}: {reason}"
            )
        if not tmp_dest.exists() or tmp_dest.stat().st_size == 0:
            raise _video.RenderError(
                f"ffmpeg exited 0 but produced no output for clip {cut.get('order', '?')}"
            )
        return cmd, proc

    # Attempt 1: the real chain (face-tracked crop/pan, hardware encoder if
    # this host has one). Attempt 2 exists because a render can come back
    # either broken or black on a source the first path cannot handle — a
    # face-tracking expression the decoder seeks into oddly, a hardware
    # encoder that writes empty frames, an AV1/VP9 source that decodes to
    # zeros. In that case re-run with a static crop and the software encoder
    # rather than surfacing a black clip as "done".
    static_w = max(2, min(crop_w, width))
    static_h = max(2, min(crop_h, height))
    static_x = max(0, min(crop_x, width - static_w))
    static_y = max(0, min(crop_y, height - static_h))
    static_graph = (
        f"crop={static_w}:{static_h}:{static_x}:{static_y}"
        f",scale={out_w}:{out_h}:force_original_aspect_ratio=increase"
        f",crop={out_w}:{out_h},setsar=1"
    )

    if cues:
        static_graph = static_graph + "," + ass_arg

    safe_choice = choose(settings, force_software=True).for_output(out_w, out_h, fps)
    safe_choice.timeout_sec = render_deadline_sec(clip_seconds, hardware=safe_choice.hardware)
    plans: list[tuple[str, str, Any]] = [("primary", filtergraph, choice)]
    if static_graph != filtergraph or choice.encoder != safe_choice.encoder:
        plans.append(("static-crop/software", static_graph, safe_choice))

    last_error: Exception | None = None
    cmd: list[str] = []
    proc: Any = None
    meta: dict[str, Any] | None = None
    for label, graph, enc in plans:
        try:
            cmd, proc = _attempt(graph, enc)
            meta = _video.verify_output(tmp_dest, expected_sec=clip_seconds)
            if label != "primary" and on_log:
                on_log(f"clip {cut.get('order', '?')}: recovered with {label}")
            break
        except _video.RenderError as exc:
            last_error = exc
            # A timeout means the work is too slow, not that the graph is
            # wrong — retrying would just burn another full deadline.
            if "timed out" in str(exc):
                raise
            if label == plans[-1][0]:
                raise
            if on_log:
                on_log(
                    f"clip {cut.get('order', '?')} attempt '{label}' failed "
                    f"({str(exc)[:200]}) — retrying with a static crop"
                )
            tmp_dest.unlink(missing_ok=True)
    if meta is None:
        assert last_error is not None
        raise last_error

    # Verified — publish it. Nothing else ever sees a partial encode.
    try:
        os.replace(tmp_dest, dest)
    except OSError as exc:
        raise _video.RenderError(
            f"could not publish rendered clip {dest.name}: {exc}"
        ) from exc

    if on_log:
        on_log(f"Rendered clip {cut.get('order', '?')} -> {dest.name} ({dest.stat().st_size} bytes)")

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }

    probed = _video.probe(str(dest))
    return {
        "clipOrder": int(cut.get("order", 0)),
        "start": round(start, 3),
        "end": round(end, 3),
        "durationSec": round(float(meta["duration"]), 3),
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "bytes": int(dest.stat().st_size),
        "videoPath": str(dest),
        "fps": float(probed.get("fps", 30.0)),
        "encoder": choice.encoder or "libx264",
        "encoderLabel": choice.label,
        "hardwareEncoded": bool(choice.hardware),
        "preset": choice.preset,
        "ffmpegCommand": " ".join(cmd),
        "ffmpegStderr": (proc.stderr or "").strip(),
    }


# fmt: off
# fmt: on

