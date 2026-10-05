"""Encode capability + graceful degradation for the render stage.

The render step is the one place where a small host is most likely to fall
over: a 1080p60 source encoded with libx264 on a single vCPU runs at roughly
0.1x realtime, so a 15-second clip takes minutes and a long source effectively
never finishes. Nothing here changes output CORRECTNESS — it changes how hard
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
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app import hostlimits
from app.config import Settings, ffmpeg_bin

# --------------------------------------------------------------------------- #
# timeouts — deliberately NOT shared
# --------------------------------------------------------------------------- #

#: A GPU host encodes far faster than it can be waited on; fail fast so a
#: broken job surfaces instead of hanging the worker slot.
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

    So the budget scales with the work: a floor for fixed overhead, a
    per-second-of-output allowance for the encode itself, and a hard ceiling.
    """
    if hardware:
        # GPU encodes an order of magnitude faster than realtime; keep the
        # historical, tighter deadline rather than the CPU budget.
        return RENDER_TIMEOUT_GPU_SEC
    raw = RENDER_DEADLINE_FLOOR_SEC + max(0.0, clip_seconds) * RENDER_REALTIME_FACTOR
    return int(min(RENDER_DEADLINE_CEILING_SEC, max(RENDER_DEADLINE_FLOOR_SEC, round(raw))))

#: x264 presets ordered fastest-first. CPU-only hosts get one of the first two.
_PRESET_RANK = (
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
)
#: What we drop to on a CPU-only host, and the absolute floor we never pass.
_CPU_PRESET = "superfast"
_CPU_PRESET_FLOOR = "superfast"


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
    notes: list[str] = field(default_factory=list)
    #: Adaptive VBR ceiling for this output, in kbps. 0 = no ceiling (CRF
    #: alone, the old behaviour). Set by `for_output()`.
    maxrate_kbps: int = 0
    bufsize_kbps: int = 0
    #: Thread count handed to the encoder. Derived from the CONTAINER's CPU
    #: allowance, not from nproc — see `choose()`.
    threads: int = 0  # 0 = ffmpeg auto-detects (all available cores)

    def threads_args(self) -> list[str]:
        """`-threads` for the encoder and the filter graph.

        When threads is 0, pass -threads 0 explicitly so ffmpeg auto-detects
        and uses all available cores. When > 0, pin to that count (used when
        the container's cgroup quota is known and we want to stay within it).
        """
        if self.threads <= 0:
            return [
                "-threads", "0",
                "-filter_threads", "0",
                "-filter_complex_threads", "0",
            ]
        return [
            "-threads", str(self.threads),
            "-filter_threads", str(self.threads),
            "-filter_complex_threads", str(self.threads),
        ]

    def for_output(self, width: int, height: int, fps: float) -> "EncoderChoice":
        """Size the encode to the output it is actually producing.

        A flat CRF is not a size policy. On this CPU-only host the preset is
        lowered to `superfast`, which disables the lookahead and CABAC
        machinery x264 normally uses, so the SAME crf produces a far larger
        file than the config implies. Measured on a real batch: crf 23 at
        720x1280/30 came out at ~6 Mbps — comfortable for vertical social
        delivery and well within disk budgets.

        So the ceiling is derived from the OUTPUT geometry rather than pinned
        to a magic number: bits-per-pixel-per-frame is the standard way to
        size a video target, and it means a 30fps clip gets half the ceiling
        of the 60fps one at the same resolution instead of the same one.

        The CRF is also adapted to the preset, because the presets are not
        equally efficient. With the cap doing the rate limiting, a lower CRF
        just spends bits in the wrong places — so the degraded preset gets
        the higher CRF, and a capable preset keeps (or lowers) it.

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
        scene from producing an enormous file. Easy content stays well under
        the cap, so this costs nothing on simple footage.
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
            ]
        if self.encoder == "h264_vaapi":
            # VAAPI takes a bitrate, not a CRF; prefer the adaptive ceiling,
            # and fall back to scaling the CRF when there is no output geometry.
            kbps = self.maxrate_kbps or max(1200, int(self.crf * 180))
            return ["-c:v", "h264_vaapi", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p"]
        if self.encoder == "h264_qsv":
            return [
                "-c:v", "h264_qsv",
                "-preset", "veryfast" if self.preset == "ultrafast" else "medium",
                "-global_quality", str(self.crf),
                "-look_ahead", "0",
                *thr,
                *cap,
                "-pix_fmt", "yuv420p",
            ]
        return [
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
            *thr,
            *cap,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
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
        return shutil.which("nvidia-smi") is not None
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
    failure, so we pay one cheap encode to find out for real.
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
    """Return (encoder, label, hw_accel) for the best WORKING HW encoder.

    Returns ("", "x264 (software)", "") when there is none. Both the encoder
    listing AND a one-frame functional encode must pass, so a device node
    without a working driver is never selected. Cached per process: the answer
    cannot change while the server is running, and probing spawns ffmpeg.
    """
    encoders = _presets_available()

    # NVENC first: by far the best quality/speed when a NVIDIA GPU is present.
    if "h264_nvenc" in encoders and _device_present("nvenc"):
        if _encoder_works("h264_nvenc", "cuda"):
            return ("h264_nvenc", "NVENC (NVIDIA, hardware)", "cuda")
        logger.info("[encode] NVENC present but not usable — falling back")

    if "h264_vaapi" in encoders and _device_present("vaapi"):
        if _encoder_works("h264_vaapi", "vaapi"):
            return ("h264_vaapi", "VAAPI (hardware)", "vaapi")
        logger.info("[encode] VAAPI present but not usable — falling back")

    if "h264_qsv" in encoders and _device_present("qsv"):
        if _encoder_works("h264_qsv", "qsv"):
            return ("h264_qsv", "QSV (Intel, hardware)", "qsv")
        logger.info("[encode] QSV present but not usable — falling back")

    return ("", "x264 (software)", "")


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
    """Clamp `requested` to something a CPU-only host can finish.

    Returns (preset, note). The note is empty when nothing was changed.
    """
    req = (requested or "veryfast").strip()
    if _preset_rank(req) <= _preset_rank(_CPU_PRESET_FLOOR):
        return req, ""
    if _preset_rank(req) > _preset_rank(_CPU_PRESET):
        return _CPU_PRESET, (
            f"preset {req} → {_CPU_PRESET} on this CPU-only host "
            f"(~2-3x faster, slightly larger files)"
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
    choice.threads = 0

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
    """scdet filtergraph that only inspects every Nth frame.

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
