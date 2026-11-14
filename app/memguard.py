# LOCKED: do not modify — see LOCKED_SPEC.md §4 (memory admission).
"""ONE memory-requirement helper. Bytes in, megabytes out, nowhere else.

The failure this exists to prevent, seen on the user's phone:

    Host is low on memory (1992 MB usable, needs ~1000000000 MB to start)

That number is bytes printed as "MB": a 1 GB requirement compared against a
value already divided by 1024, so the guard refused a host with 1.9 GB free.
The rule now lives in exactly one place and it is written in the two units
that were confused: ``bytes`` and ``MB``, with no third unit and no implicit
conversion anywhere else.

The requirement is per-model because Whisper's RSS is per-model. int8 `small`
needs about 900 MB; admitting it on 1.9 GB is correct and must never be
refused. A host is refused ONLY when usable memory is below the requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

MB = 1024 * 1024

#: Peak RSS for faster-whisper int8, in MB. `small` is the value the spec pins.
MODEL_REQUIREMENT_MB: dict[str, float] = {
    "tiny": 300.0,
    "tiny.en": 300.0,
    "base": 400.0,
    "base.en": 400.0,
    "small": 900.0,
    "small.en": 900.0,
    "medium": 1800.0,
    "medium.en": 1800.0,
    "large-v3": 3200.0,
    "large": 3200.0,
}

#: ffmpeg (decode + x264) rides alongside the model in the same job process.
RENDER_OVERHEAD_MB = 300.0

#: Fallback when the configured model is not in the table: the largest small
#: class model rather than a guess that would admit something that cannot fit.
DEFAULT_MODEL = "small"


def bytes_to_mb(n: float | int | None) -> float:
    """Bytes -> MB (binary megabyte). The ONLY conversion in the codebase."""
    return 0.0 if n is None else float(n) / float(MB)


def mb_to_bytes(mb: float | int | None) -> int:
    """MB (binary) -> bytes. Inverse of :func:`bytes_to_mb`."""
    return 0 if mb is None else int(round(float(mb) * MB))


def requirement_mb(model: str | None, *, stage: str | None = None) -> float:
    """MB this job needs, from the model name. Unknown models use `small`.

    `transcribing` loads the model (the table value IS the requirement for
    int8 `small`: 900 MB). Every stage that only runs ffmpeg needs
    ``RENDER_OVERHEAD_MB`` and no model at all.
    """
    name = str(model or DEFAULT_MODEL).strip().lower()
    base = MODEL_REQUIREMENT_MB.get(name, MODEL_REQUIREMENT_MB[DEFAULT_MODEL])
    if str(stage or "").lower() in {"rendering", "export", "exporting", "captions"}:
        return RENDER_OVERHEAD_MB
    return base


def available_bytes() -> int | None:
    """Usable RAM right now, from /proc/meminfo MemAvailable (bytes)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


@dataclass(frozen=True)
class Admission:
    ok: bool
    available_mb: float | None
    required_mb: float
    message: str | None = None


def check(
    model: str | None = None,
    *,
    stage: str | None = None,
    avail_bytes: int | None = None,
) -> Admission:
    """Decide whether a job may start. ``available >= required`` always passes."""
    required = requirement_mb(model, stage=stage)
    if avail_bytes is None:
        avail_bytes = available_bytes()
    if avail_bytes is None:
        # An unreadable /proc is not evidence of a shortage.
        return Admission(True, None, required)
    avail_mb = bytes_to_mb(avail_bytes)
    if avail_mb >= required:
        return Admission(True, avail_mb, required)
    return Admission(
        False,
        avail_mb,
        required,
        f"Host is low on memory ({avail_mb:.0f} MB usable, needs "
        f"~{required:.0f} MB to start). Free memory and retry — finished "
        f"stages are cached, so it resumes where it stopped.",
    )
