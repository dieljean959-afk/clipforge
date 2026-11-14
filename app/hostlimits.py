"""Container-aware CPU/memory limits.

``os.cpu_count()`` and ``os.sysconf`` report the HOST's resources. Inside a
container that can be wildly wrong: the Freebuff sandbox reports 48 CPUs while
its cgroup CPU quota is effectively **1 CPU**, and 377 GB of RAM while its
cgroup memory limit is **2 GB**. Trusting the host numbers causes exactly the
failures users report as "transcription is slow / the job hangs forever":

  * CTranslate2 spawns one inference thread per "available" CPU. On a 1-CPU
    quota that is 48x oversubscription — every thread competes for the same
    core and each one allocates its own working buffers, so inference gets
    *slower* and memory use balloons until the OOM killer takes the process.
  * A worker pool sized for 48 cores starts several transcription jobs, each
    loading its own model copy. At ~1.7 GB per `medium` model inside a 2 GB
    limit the container is killed mid-job — the "worker exited unexpectedly /
    stuck at 0%" report.

Everything in ClipForge that sizes threads, batch memory or worker pools goes
through this module instead of ``os.cpu_count()``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

_CGROUP_V2 = Path("/sys/fs/cgroup")
_CGROUP_V1 = Path("/sys/fs/cgroup")

# Measured resident-set cost of one loaded CTranslate2/Whisper model, per size
# (int8). Measured, not guessed: on this sandbox `small` loads at 618 MB and
# `medium` at 1828 MB. Used to decide which model FITS and how many parallel
# jobs fit — not how many the user asked for.
MODEL_RSS_BYTES = {
    "tiny": 150 * 1024**2,
    "base": 260 * 1024**2,
    "small": 660 * 1024**2,
    "medium": 1800 * 1024**2,
    "large-v3": 3400 * 1024**2,
    "large-v2": 3400 * 1024**2,
}

# Order used when falling back to a model that fits.
MODEL_ORDER = ["tiny", "base", "small", "medium", "large-v3"]

# Non-model overhead per running job: ffmpeg decode/encode buffers, the decoded
# audio chunk, CTranslate2 workspaces, Python runtime. Transcription itself is
# where this peaks (encoder workspace scales with the audio window).
PER_JOB_OVERHEAD_BYTES = 400 * 1024**2

# Fraction of the limit a single job may plan to use. The rest is headroom for
# the API process, the OS page cache and ffmpeg children.
SAFETY = 0.9


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except Exception:
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def cpu_quota_cpus() -> float | None:
    """CPU quota in whole-CPU equivalents, or None when unrestricted."""
    # cgroup v2: "<quota> <period>"
    try:
        parts = (_CGROUP_V2 / "cpu.max").read_text(encoding="ascii").split()
        if len(parts) == 2 and parts[0] != "max":
            return int(parts[0]) / int(parts[1])
    except Exception:
        pass
    # cgroup v1: quota / period
    try:
        quota = int((_CGROUP_V1 / "cpu" / "cpu.cfs_quota_us").read_text(encoding="ascii"))
        period = int((_CGROUP_V1 / "cpu" / "cpu.cfs_period_us").read_text(encoding="ascii"))
        if quota > 0 and period > 0:
            return quota / period
    except Exception:
        pass
    return None


def memory_limit_bytes() -> int | None:
    """Hard memory limit for this container, or None when unrestricted."""
    for candidate in (
        _CGROUP_V2 / "memory.max",
        _CGROUP_V1 / "memory" / "memory.limit_in_bytes",
    ):
        value = _read_int(candidate)
        if value is not None and value < 1 << 60:  # v1 uses a huge sentinel
            return value
    return None


def host_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def effective_cpus() -> int:
    """Usable CPUs after honouring cgroup quota and CPU affinity."""
    usable = host_cpu_count()
    quota = cpu_quota_cpus()
    if quota is not None:
        usable = min(usable, max(1, int(quota)))
    return max(1, usable)


def effective_memory_bytes() -> int | None:
    """Usable memory: the smaller of the cgroup limit and physical RAM."""
    limit = memory_limit_bytes()
    physical: int | None = None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            physical = pages * page_size
    except (ValueError, OSError, AttributeError):
        physical = None
    if limit is None:
        return physical
    if physical is None:
        return limit
    return min(limit, physical)


def inference_threads() -> int:
    """Threads for one CTranslate2/Whisper inference.

    Capped at 8: beyond that a single inference gains little and the per-thread
    working buffers become the memory bottleneck.
    """
    return max(1, min(8, effective_cpus()))


def batch_size_for_host(requested: int) -> int:
    """Clamp a requested decode batch to what this host can hold in memory."""
    limit = effective_memory_bytes()
    if limit is None:
        return requested
    if limit <= 2 * 1024**3:
        return max(1, min(requested, 4))
    if limit <= 4 * 1024**3:
        return max(1, min(requested, 8))
    return requested


def per_job_bytes(model: str) -> int:
    return int(MODEL_RSS_BYTES.get(model, MODEL_RSS_BYTES["medium"]) * 1.15) + PER_JOB_OVERHEAD_BYTES


def recommended_concurrency(model: str, requested: int) -> int:
    """How many parallel jobs actually FIT on this host.

    The configured value is an upper bound; memory decides the real number, so a
    pool never OOM-kills the container. Returns at least 1 (a single job always
    gets to run even on the smallest host).
    """
    if requested <= 1:
        return 1
    limit = effective_memory_bytes()
    if limit is None:
        return max(1, min(requested, effective_cpus()))
    budget = int(limit * SAFETY)
    fits = max(1, budget // max(1, per_job_bytes(model)))
    return max(1, min(requested, fits, effective_cpus() + 1))


def model_fits(model: str) -> bool:
    """True when one transcription with `model` fits inside this host's memory."""
    limit = effective_memory_bytes()
    if limit is None:
        return True
    return per_job_bytes(model) <= int(limit * SAFETY)


def resolved_model(configured: str) -> tuple[str, str | None]:
    """Return (model to actually load, note explaining any change).

    A model that cannot fit is not "slow", it is an OOM kill: the process dies
    mid-transcription with nothing the user can act on, which is exactly the
    "jobs stick at 0% / the worker exited unexpectedly" report. So when the
    configured model does not fit this host we fall back to the largest smaller
    model that does, and say so everywhere (boot banner, /api/health,
    /api/doctor, job log). On any host with room for the configured model this
    is a no-op.
    """
    if model_fits(configured):
        return configured, None
    limit = effective_memory_bytes()
    if limit is None:
        return configured, None
    try:
        below = MODEL_ORDER[: MODEL_ORDER.index(configured)]
    except ValueError:
        below = MODEL_ORDER[:3]
    for candidate in reversed(below):
        if model_fits(candidate):
            return candidate, (
                f"model '{configured}' cannot run on this host: one transcription with it "
                f"needs about {per_job_bytes(configured) / 1024**3:.1f} GiB but only "
                f"{limit / 1024**3:.1f} GiB is available, so it is OOM-killed instead of "
                f"finishing. Using '{candidate}' ({per_job_bytes(candidate) / 1024**3:.1f} GiB). "
                "Set CLIPFORGE_WHISPER_MODEL or [whisper] model to override."
            )
    return "tiny", (
        f"model '{configured}' does not fit this host (only {limit / 1024**3:.1f} GiB "
        "available) — using 'tiny'. Transcription quality will be reduced."
    )


def limits() -> dict[str, Any]:
    """Machine-readable summary for /api/health and /api/doctor."""
    quota = cpu_quota_cpus()
    limit = memory_limit_bytes()
    physical = None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical = pages * page_size if pages > 0 and page_size > 0 else None
    except (ValueError, OSError, AttributeError):
        physical = None
    return {
        "hostCpus": host_cpu_count(),
        "cpus": effective_cpus(),
        "cpuQuota": quota,
        "memoryLimitBytes": limit,
        "memoryEffectiveBytes": effective_memory_bytes(),
        "physicalMemoryBytes": physical,
        "constrained": (quota is not None and quota < host_cpu_count())
        or (limit is not None and physical is not None and limit < physical),
        "inferenceThreads": inference_threads(),
    }


def describe() -> str:
    """One-line human summary for boot logs."""
    info = limits()
    parts = [
        f"{info['cpus']} CPU (host reports {info['hostCpus']})"
        if info["cpus"] != info["hostCpus"]
        else f"{info['cpus']} CPU"
    ]
    eff = info["memoryEffectiveBytes"]
    phys = info["physicalMemoryBytes"]
    if eff is not None:
        gib = eff / 1024**3
        parts.append(
            f"{gib:.1f} GiB RAM (host has {phys / 1024**3:.0f} GiB)"
            if phys is not None and phys > eff * 1.1
            else f"{gib:.1f} GiB RAM"
        )
    parts.append(f"inference threads {info['inferenceThreads']}")
    if info["constrained"]:
        parts.append("container-limited")
    return " | ".join(parts)


def model_warning(model: str) -> str | None:
    """Honest warning when the configured model does not fit this host."""
    _resolved, note = resolved_model(model)
    return note


def oom_hint(exit_code: int | None, stage: str | None) -> str | None:
    """Explain a signalled worker (SIGKILL -9, SIGTERM -15).

    Two different things produce a signalled child, and conflating them sends
    operators down the wrong path:

      -9  SIGKILL — typically the OOM killer. Needs a smaller footprint.
     -15  SIGTERM — an EXTERNAL termination: a host restart, redeploy, container
          stop or scale-down. This is NOT memory pressure, and lowering the
          model size will not help. Before this branch existed, -15 fell through
          to a bare "exited unexpectedly" with no cause at all, which is the
          least actionable failure in the system.
    """
    if exit_code == -15:
        where = f" during '{stage}'" if stage else ""
        return (
            f"The worker was terminated by SIGTERM{where} — something outside the "
            "pipeline killed it; it did not fail on its own. This is NOT an "
            "out-of-memory kill (that reports code -9). Typical causes are the host "
            "restarting or redeploying, a container stop/scale-down, or a manual "
            "restart. Retry the job; if it keeps happening, check whether a deploy "
            "or restart time lines up with the failure."
        )
    if exit_code != -9:
        return None
    limit = effective_memory_bytes()
    where = f" during '{stage}'" if stage else ""
    size = f" (host memory limit {limit / 1024**3:.1f} GiB)" if limit else ""
    return (
        f"The worker process was killed by the system{where} — this is an "
        f"out-of-memory kill, not a code error{size}. Lower the memory footprint: use a "
        "smaller whisper model (CLIPFORGE_WHISPER_MODEL=small), process a shorter source, "
        "or dispatch rendering to a bigger host with WORKER_URL."
    )


def log_summary(model: str = "medium") -> None:
    logger.info(f"[host] limits: {describe()}")
    warn = model_warning(model)
    if warn:
        logger.warning(f"[host] {warn}")
