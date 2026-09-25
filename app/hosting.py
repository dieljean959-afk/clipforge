"""Hosted-runtime helpers: system stats, remote-worker dispatch, cleanup.

The FastAPI process IS the hosted worker (one process, API + in-process
pipeline thread). This module adds the pieces a hosted deployment needs:

- ``system_stats()`` — real CPU/RAM for the status card (psutil when
  installed, /proc + os otherwise). Never returns NaN; unknown values are
  None and the UI renders "unknown" honestly.
- ``remote_probe()`` / ``dispatch_job()`` — when WORKER_URL is set, hand jobs
  to the user's home machine over Tailscale (HTTP) with honest errors.
- ``RetentionCleaner`` — background thread deleting downloads/outputs after
  the retention window (hosted storage is finite; videos are deleted after
  the user has had the chance to download them).
- ``tailnet_ip()`` — this process's Tailscale 100.x.x.x address when
  tailscaled is present (logged at boot, shown in the status card).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import ROOT, Settings
from app.stages import remove_workspace
from app.store import LocalStore

# --------------------------------------------------------------------------- #
# system stats (never NaN — None means "unknown", the UI renders it honestly)
# --------------------------------------------------------------------------- #

_STATS_CACHE: dict[str, Any] = {}
_STATS_CACHE_AT = 0.0
_STATS_LOCK = threading.Lock()


def _proc_stat_cpu_percent() -> float | None:
    """Sample /proc/stat twice (0.15s apart) → busy percentage, or None."""
    try:
        def read() -> tuple[int, int]:
            with open("/proc/stat", encoding="ascii") as fh:
                line = fh.readline()
            parts = [int(p) for p in line.split()[1:9]]
            idle = parts[3] + parts[4]  # idle + iowait
            return sum(parts), idle

        total0, idle0 = read()
        time.sleep(0.15)
        total1, idle1 = read()
        dt, di = total1 - total0, idle1 - idle0
        if dt <= 0:
            return None
        return round(max(0.0, min(100.0, (1 - di / dt) * 100)), 1)
    except Exception:
        return None


def system_stats() -> dict[str, Any]:
    """CPU/RAM of THIS server process's host. Cached 2s to keep /api/health snappy."""
    global _STATS_CACHE, _STATS_CACHE_AT
    with _STATS_LOCK:
        if time.time() - _STATS_CACHE_AT < 2.0 and _STATS_CACHE:
            return dict(_STATS_CACHE)

        stats: dict[str, Any] = {
            "cpuPercent": None,
            "cores": os.cpu_count(),
            "memTotalBytes": None,
            "memUsedBytes": None,
            "load1": None,
            "pid": os.getpid(),
        }
        try:
            import psutil  # optional dependency

            vm = psutil.virtual_memory()
            stats["cpuPercent"] = round(psutil.cpu_percent(interval=0.1), 1)
            stats["cores"] = psutil.cpu_count() or stats["cores"]
            stats["memTotalBytes"] = vm.total
            stats["memUsedBytes"] = vm.total - vm.available
        except ImportError:
            # stdlib fallback (Linux): /proc/stat + /proc/meminfo + getloadavg
            stats["cpuPercent"] = _proc_stat_cpu_percent()
            if hasattr(os, "getloadavg"):
                try:
                    stats["load1"] = round(os.getloadavg()[0], 2)
                except OSError:
                    pass
            try:
                meminfo = {}
                with open("/proc/meminfo", encoding="ascii") as fh:
                    for line in fh:
                        key, _, val = line.partition(":")
                        meminfo[key.strip()] = int(val.strip().split()[0]) * 1024
                if meminfo.get("MemTotal"):
                    stats["memTotalBytes"] = meminfo["MemTotal"]
                    stats["memUsedBytes"] = meminfo["MemTotal"] - meminfo.get(
                        "MemAvailable", 0
                    )
            except Exception:
                pass
        except Exception as exc:  # psutil present but failed — say unknown
            logger.warning(f"[hosting] psutil failed: {exc}")

        # final guard: no NaN/inf can ever leak to JSON
        for k, v in stats.items():
            if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                stats[k] = None

        _STATS_CACHE = stats
        _STATS_CACHE_AT = time.time()
        return dict(stats)


def disk_stats() -> dict[str, Any]:
    """Disk free/total for the deployment sandbox (the volumes jobs write to).

    Real values via shutil.disk_usage on the repo root — never fabricated.
    Unknown values stay None and the UI renders them honestly.
    """
    stats: dict[str, Any] = {"freeBytes": None, "totalBytes": None}
    try:
        import shutil as _shutil

        usage = _shutil.disk_usage(ROOT)
        stats["freeBytes"] = usage.free
        stats["totalBytes"] = usage.total
    except Exception:
        pass  # unknown stays None
    return stats


def tailnet_ip() -> str | None:
    """This process's Tailscale IP (100.64.0.0/10), or None when absent."""
    exe = os.path.expanduser(shutil_which("tailscale") or "")
    if exe:
        try:
            import subprocess

            out = subprocess.run(
                [exe, "ip", "-4"], capture_output=True, text=True, timeout=4
            )
            for line in out.stdout.splitlines():
                ip = line.strip()
                if _is_tailnet(ip):
                    return ip
        except Exception:
            pass
    try:
        import socket

        import psutil  # optional

        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family == socket.AF_INET and _is_tailnet(a.address):
                    return a.address
    except Exception:
        pass
    return None


def _is_tailnet(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4 or parts[0] != "100":
        return False
    try:
        return 64 <= int(parts[1]) <= 127
    except ValueError:
        return False


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


# --------------------------------------------------------------------------- #
# remote worker dispatch (freebuff dashboard + home-renderer over Tailscale)
# --------------------------------------------------------------------------- #


class DispatchError(Exception):
    """Job could not be handed to the remote worker — the UI shows this."""


def _auth_headers(settings: Settings) -> dict[str, str]:
    if settings.hosted.worker_token:
        return {"Authorization": f"Bearer {settings.hosted.worker_token}"}
    return {}


def remote_probe(settings: Settings) -> dict[str, Any]:
    """GET {WORKER_URL}/agent/ping → honest online/offline with reason."""
    if not settings.hosted.worker_url:
        return {"online": False, "detail": "WORKER_URL not configured"}
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{settings.hosted.worker_url}/agent/ping",
            headers={**_auth_headers(settings), "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        return {
            "online": True,
            "detail": str(data.get("version", "worker-agent")),
            "queueDepth": int(data.get("queueDepth", 0)),
        }
    except Exception as exc:
        return {"online": False, "detail": f"{type(exc).__name__}: {exc}"}


def _multipart_stream(
    fields: dict[str, str], file_field: str, file_path: Path, chunk: int = 1024 * 1024
):
    """Yield a multipart/form-data body in chunks without loading the file."""
    boundary = f"----clipforge{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ).encode("utf-8")
        )
    head = b"".join(parts) + (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{file_path.name}\"\r\nContent-Type: video/mp4\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    size = file_path.stat().st_size
    total = len(head) + size + len(tail)

    def gen():
        yield head
        with file_path.open("rb") as fh:
            while True:
                data = fh.read(chunk)
                if not data:
                    break
                yield data
        yield tail

    return gen, total, boundary


def dispatch_job(settings: Settings, job: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Upload the source video + job spec to the remote worker over the tailnet.

    The heavy pipeline (whisper + ffmpeg) then runs on the user's machine;
    this server only tracks state. Raises DispatchError with the exact reason.
    """
    if not settings.hosted.worker_url:
        raise DispatchError("WORKER_URL is not configured — cannot dispatch")
    if not source_path.exists():
        raise DispatchError(f"source file missing: {source_path.name}")

    import urllib.request

    options = job.get("options", {}) or {}
    fields = {
        "jobId": job["_id"],
        "sourceType": str(job.get("sourceType", "upload")),
        "source": str(job.get("source", ""))[:2048],
        "minClipSec": str(options.get("minClipSec") or ""),
        "maxClipSec": str(options.get("maxClipSec") or ""),
        "maxClips": str(options.get("maxClips") or ""),
    }
    gen, total, boundary = _multipart_stream(fields, "file", source_path)

    req = urllib.request.Request(
        f"{settings.hosted.worker_url}/agent/jobs",
        data=gen(),  # type: ignore[arg-type] — urllib accepts iterables of bytes
        headers={
            **_auth_headers(settings),
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(total),
        },
        method="POST",
    )
    logger.info(f"[hosting] dispatching job {job['_id']} to {settings.hosted.worker_url} ({total} bytes)")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            ack = json.loads(resp.read().decode("utf-8") or "{}")
    except Exception as exc:
        raise DispatchError(
            f"remote worker at {settings.hosted.worker_url} rejected the job: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not ack.get("accepted"):
        raise DispatchError(f"remote worker did not accept the job: {ack}")
    logger.info(f"[hosting] job {job['_id']} accepted by remote worker")
    return ack


# --------------------------------------------------------------------------- #
# retention: hosted disks are finite — delete finished work after the window
# --------------------------------------------------------------------------- #


class RetentionCleaner:
    """Daemon thread deleting completed jobs' artifacts after retention_hours."""

    def __init__(self, settings: Settings, store: LocalStore, interval_min: int = 30) -> None:
        self.settings = settings
        self.store = store
        self.interval_min = interval_min
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="clipforge-retention", daemon=True).start()
        logger.info(
            f"[hosting] retention cleaner online — outputs deleted "
            f"{self.settings.hosted.retention_hours}h after completion"
        )

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_min * 60):
            try:
                self.sweep()
            except Exception as exc:
                logger.error(f"[hosting] retention sweep failed: {exc}")

    def sweep(self) -> int:
        cutoff = time.time() * 1000 - self.settings.hosted.retention_hours * 3600 * 1000
        removed = 0
        for job in self.store.list_jobs():
            done_at = job.get("completedAt")
            stage = job.get("stage")
            expired = (stage in {"complete", "failed"} and done_at and done_at < cutoff) or (
                stage not in {"complete", "failed"}
                and job.get("createdAt", 0) < cutoff - 24 * 3600 * 1000
            )
            if expired:
                job_id = job["_id"]
                self.store.delete_job(job_id)
                import shutil as _shutil

                for d in (
                    self.settings.app.paths.downloads_dir / job_id,
                    self.settings.app.paths.outputs_dir / job_id,
                ):
                    _shutil.rmtree(d, ignore_errors=True)
                # per-stage mirrors: extracted audio WAVs + per-chunk transcript
                # caches are the biggest artifacts a long job leaves behind
                remove_workspace(self.settings.app.paths.data_dir, job_id)
                removed += 1
                logger.info(f"[hosting] retention: removed job {job_id} artifacts")
        return removed
