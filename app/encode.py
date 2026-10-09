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
import shutil
import subprocess
from dataclasses import dataclass,        field
from typing import Any

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
        return os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidia-uvm") or bool(hosting.shutil_which("nvidia-smi")) or bool(hostlimits.shutil_which("nvidia-smi"))
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
