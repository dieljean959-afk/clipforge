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
RENDER_TIMEOUT_CPU_SEC = 90 * 60
#: Probing ffmpeg for encoders/filters is quick but not instant on a loaded box.
PROBE_TIMEOUT_SEC = 30

#: x264 presets ordered fastest-first. CPU-only hosts get one of the first two.
_PRESET_RANK = (
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
)
#: What we drop to on a CPU-only host, and the absolute floor we never pass.
_CPU_PRESET = "ultrafast"
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
    crf: int = 20
    hw_accel: str = ""
    timeout_sec: int = RENDER_TIMEOUT_GPU_SEC
    #: Populated when the requested preset had to be lowered.
    degraded_from: str = ""
    notes: list[str] = field(default_factory=list)

    def ffmpeg_codec_args(self) -> list[str]:
        """The ``-c:v``/quality args for the chosen encoder."""
        if self.encoder == "h264_nvenc":
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p4" if self.preset == "ultrafast" else "p5",
                "-cq", str(self.crf),
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
            ]
        if self.encoder == "h264_vaapi":
            # VAAPI takes a bitrate, not a CRF; scale the CRF we were given.
            kbps = max(1200, int(self.crf * 180))
            return ["-c:v", "h264_vaapi", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p"]
        if self.encoder == "h264_qsv":
            return [
                "-c:v", "h264_qsv",
                "-preset", "veryfast" if self.preset == "ultrafast" else "medium",
                "-global_quality", str(self.crf),
                "-look_ahead", "0",
                "-pix_fmt", "yuv420p",
            ]
        return [
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
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
    choice.timeout_sec = RENDER_TIMEOUT_CPU_SEC
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
        "degradedFrom": choice.degraded_from,
        "timeoutSec": choice.timeout_sec,
        "notes": choice.notes,
    }
