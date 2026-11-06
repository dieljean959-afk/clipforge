"""Memory budget for one job, read from the CGROUP in bytes, reported in MB.

The failure this replaces: `mkl_malloc: failed to allocate memory` inside
faster-whisper on a 2 GiB container. Nothing was over budget on paper — the
process thought it had the host's 3 GiB, the cgroup only allowed 2 GiB, and no
step asked "does the next chunk fit?" before starting it.

Rules:

* The limit comes from the CGROUP (`memory.max`), never from `MemTotal`. Host
  totals are a lie inside a container; this host reports 3 GiB while allowing 2.
* `memory.current` — not `MemAvailable` — is what is already spent in this
  cgroup, page cache included, because the kernel counts it too.
* Headroom is reserved for the API, the worker daemon and ffmpeg before any
  transcription budget is handed out.
* Units are explicit and shared with :mod:`app.memguard`: bytes in, MB out.
  There is no third unit and no implicit conversion in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.memguard import MB, bytes_to_mb, mb_to_bytes

CGROUP_ROOT = Path("/sys/fs/cgroup")

#: Reserved for processes that must survive the job: API (~90 MB), worker
#: daemon (~30 MB), ffmpeg/x264 spikes (~150 MB). Never handed to ASR.
HEADROOM_MB = 320.0

#: One 30 s faster-whisper window needs roughly this much for int8 `small`
#: (mel + encoder activations + decoder KV), measured on this host.
ASR_MB_PER_30S = 55.0

#: Chunk lengths the planner is allowed to pick, in seconds.
MIN_CHUNK_SEC = 30
MAX_CHUNK_SEC = 120


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw in {"max", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def cgroup_limit_bytes() -> int | None:
    """`memory.max` in bytes, or None when this cgroup is unlimited."""
    return _read_int(CGROUP_ROOT / "memory.max")



def cgroup_peak_bytes() -> int | None:
    return _read_int(CGROUP_ROOT / "memory.peak")


def cgroup_stat() -> dict[str, int]:
    """`memory.stat` — the breakdown behind `memory.current`."""
    out: dict[str, int] = {}
    try:
        for line in (CGROUP_ROOT / "memory.stat").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            if key and value.strip().lstrip("-").isdigit():
                out[key] = int(value.strip())
    except OSError:
        pass
    return out


def cgroup_used_bytes() -> int | None:
    """Working set: `memory.current` minus page cache the kernel can reclaim.

    Page cache is charged to the cgroup but is NOT the reason a malloc fails:
    the kernel drops clean file pages before it refuses an anonymous mapping.
    Counting it as "used" made a host with 1.4 GB of cache look like it had
    300 MB left and skipped every ladder rung. `inactive_file` is the part the
    kernel reclaims first, so that is what is subtracted here.
    """
    current = _read_int(CGROUP_ROOT / "memory.current")
    if current is None:
        return None
    reclaimable = cgroup_stat().get("inactive_file", 0)
    return max(0, current - max(0, reclaimable))


def cgroup_events() -> dict[str, int]:
    """Counters from `memory.events` (low/high/max/oom/oom_kill)."""
    out: dict[str, int] = {}
    try:
        for line in (CGROUP_ROOT / "memory.events").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            if key and value.strip().isdigit():
                out[key] = int(value.strip())
    except OSError:
        pass
    return out


def host_available_bytes() -> int | None:
    """MemAvailable, only as a fallback when the cgroup is unlimited."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


@dataclass(frozen=True)
class Budget:
    """What one job may use, all values in MB except the byte fields."""

    limit_mb: float | None
    used_mb: float
    available_mb: float
    headroom_mb: float
    source: str  # "cgroup" | "proc" | "unknown"

    @property
    def usable_mb(self) -> float:
        """Memory this job may actually spend: available minus headroom."""
        return max(0.0, self.available_mb - self.headroom_mb)

    def fits(self, need_mb: float) -> bool:
        return self.usable_mb >= float(need_mb)

    def shortfall_mb(self, need_mb: float) -> float:
        return max(0.0, float(need_mb) - self.usable_mb)

    def describe(self) -> str:
        limit = "unlimited" if self.limit_mb is None else f"{self.limit_mb:.0f} MB"
        return (
            f"limit={limit} used={self.used_mb:.0f} MB (reclaimable cache excluded) "
            f"available={self.available_mb:.0f} MB "
            f"headroom={self.headroom_mb:.0f} MB "
            f"usable={self.usable_mb:.0f} MB ({self.source})"
        )


def budget_from(
    limit_bytes: int | None,
    used_bytes: int | None,
    available_bytes: int | None,
    *,
    headroom_mb: float = HEADROOM_MB,
) -> Budget:
    """Pure constructor — the unit of every test, no /sys needed."""
    if limit_bytes is not None:
        limit_mb = bytes_to_mb(limit_bytes)
        used_mb = bytes_to_mb(used_bytes or 0)
        avail_mb = max(0.0, limit_mb - used_mb)
        return Budget(limit_mb, used_mb, avail_mb, float(headroom_mb), "cgroup")
    if available_bytes is not None:
        return Budget(None, 0.0, bytes_to_mb(available_bytes), float(headroom_mb), "proc")
    return Budget(None, 0.0, 0.0, float(headroom_mb), "unknown")


def current_budget(*, headroom_mb: float = HEADROOM_MB) -> Budget:
    """Live budget for this cgroup (falls back to MemAvailable if unlimited)."""
    return budget_from(
        cgroup_limit_bytes(),
        cgroup_used_bytes(),
        host_available_bytes(),
        headroom_mb=headroom_mb,
    )


def asr_requirement_mb(model: str, seconds: float) -> float:
    """Peak MB for decoding `seconds` of audio with `model` and no VAD/beam cost."""
    from app.memguard import MODEL_REQUIREMENT_MB  # local: keeps the import graph flat

    base = MODEL_REQUIREMENT_MB.get(str(model).lower(), MODEL_REQUIREMENT_MB["small"])
    # The model is resident; the decode of `seconds` adds activations that grow
    # with the window (bounded by the encoder's 30 s window inside the engine).
    windows = max(1.0, float(seconds) / 30.0)
    return base + ASR_MB_PER_30S * windows


def plan_chunk_seconds(
    model: str,
    budget: Budget,
    *,
    preferred: float = 60.0,
    minimum: float = float(MIN_CHUNK_SEC),
) -> float:
    """Longest chunk ≤ `preferred` whose peak still fits the budget.

    Never returns less than `minimum`: at that point the caller's ladder has to
    drop to a smaller model or VAD-free decoding, because no chunk length makes
    the current one fit.
    """
    limit_ok = float(preferred)
    for candidate in (float(MAX_CHUNK_SEC), 60.0, float(MIN_CHUNK_SEC)):
        if candidate > preferred:
            continue
        if budget.fits(asr_requirement_mb(model, candidate)):
            limit_ok = candidate
            break
    else:
        limit_ok = float(minimum)
    return max(float(minimum), limit_ok)


def fits_next_chunk(model: str, seconds: float, *, headroom_mb: float = HEADROOM_MB) -> bool:
    """One call the callers use: does this chunk fit RIGHT NOW?"""
    return current_budget(headroom_mb=headroom_mb).fits(asr_requirement_mb(model, seconds))


def reclaim_hint() -> str:
    """Human sentence for the job log when a budget check fails."""
    b = current_budget()
    return (
        f"{b.describe()} — a {b.usable_mb:.0f} MB budget cannot hold another "
        "full decode; switching to a lighter attempt"
    )


__all__ = [
    "HEADROOM_MB",
    "Budget",
    "asr_requirement_mb",
    "budget_from",
    "cgroup_events",
    "cgroup_limit_bytes",
    "cgroup_peak_bytes",
    "cgroup_used_bytes",
    "current_budget",
    "fits_next_chunk",
    "host_available_bytes",
    "mb_to_bytes",
    "plan_chunk_seconds",
    "reclaim_hint",
    "MB",
]
