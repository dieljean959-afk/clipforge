from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from loguru import logger
from app.config import ROOT, Settings
from app.store import LocalStore

class RetentionCleaner:
    """Background thread that deletes finished work after the retention window.

    Kept intentionally thin: it only deletes files/dirs the server already
    decided are expired, and it never touches anything the API or worker is
    currently reading.  ``sweep()`` returns how many artefacts were removed so
    the dashboard can report it without re-walking the disk.
    """

    def __init__(self, settings: Settings, store: LocalStore, interval_min: int = 30) -> None:
        self.settings = settings
        self.store = store
        self.interval_min = interval_min
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_min * 60 + 5)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_min * 60):
            try:
                swept = self.sweep()
                logger.info(f"[retention] sweep removed {swept} artefacts")
            except Exception:
                logger.exception("[retention] sweep failed")

    def sweep(self) -> int:
        """Delete batches past the retention window. Returns jobs removed.

        The POLICY lives in ``reclaim.sweep_expired`` (bucket first, local
        second, only terminal batches, and an explicit "retention 0 means never"
        rule). This method is only the periodic driver, so there is no second
        implementation to drift. An earlier copy of this method re-implemented
        the policy with ``retention_hours == 0`` interpreted as "expire
        everything", which is how a 0-hour window turned into a disk wipe.
        """
        from app import reclaim

        hours = int(self.settings.hosted.retention_hours or 0)
        if hours <= 0:
            return 0
        out = reclaim.sweep_expired(self.store, self.settings.app.paths, hours)
        return int(out.get("jobsRemoved") or 0)


def _multipart_stream(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    chunk: int = 1 << 20,
):
    """Build a multipart/form-data body as an iterator of bytes.

    ``fields`` are the small text parts (jobId, source…), ``file_path`` is
    streamed from disk in ``chunk`` slices, so a 60 MB source never has to fit
    in memory. Returns ``(gen, total_bytes, boundary)``: the caller needs the
    exact length up front for Content-Length, and urllib will not compute it
    from an iterator.
    """
    boundary = f"----clipforge{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )
    head = b"".join(parts) + (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
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








# --------------------------------------------------------------------------- #
# system stats (never NaN — None means "unknown", the UI renders it honestly)
# --------------------------------------------------------------------------- #

_MB = 1024 * 1024
_STATS_CACHE: dict[str, Any] = {}
_STATS_CACHE_AT = 0.0
_STATS_LOCK = threading.Lock()
_DISK_CACHE: dict[str, Any] = {}
_DISK_CACHE_AT = 0.0
_DISK_CACHE_KEY = ""


def _proc_stat_cpu_percent() -> float | None:
    """Sample /proc/stat twice (0.05 s apart) → busy percentage, or None."""
    try:
        def read() -> tuple[int, int]:
            with open("/proc/stat", encoding="ascii") as fh:
                line = fh.readline()
            parts = [int(p) for p in line.split()[1:9]]
            idle = parts[3] + parts[4]  # idle + iowait
            return sum(parts), idle

        total0, idle0 = read()
        time.sleep(0.05)
        total1, idle1 = read()
        dt, di = total1 - total0, idle1 - idle0
        if dt <= 0:
            return None
        return round(100.0 * (dt - di) / dt, 1)
    except Exception:
        return None


def system_stats() -> dict[str, Any]:
    """CPU/RAM for the status card. Unknown is None, never NaN or 0."""
    global _STATS_CACHE, _STATS_CACHE_AT
    with _STATS_LOCK:
        if _STATS_CACHE and time.time() - _STATS_CACHE_AT < 2.0:
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

        for k, v in stats.items():
            if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                stats[k] = None

        _STATS_CACHE = stats
        _STATS_CACHE_AT = time.time()
        return dict(stats)


def mb_from_bytes(value: int | float | None) -> float | None:
    """Bytes in, MB out — the one conversion (1 MB = 1024*1024 bytes)."""
    if value is None:
        return None
    return round(float(value) / _MB, 1)


def _measure_tree_bytes(paths: Iterable[Path]) -> int:
    total = 0
    for root in paths:
        try:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                try:
                    if p.is_file() and not p.is_symlink():
                        total += p.stat().st_size
                except OSError:
                    continue
        except OSError:
            continue
    return total


def invalidate_disk_cache() -> None:
    """Forget the cached tree size (after a delete/offload)."""
    global _DISK_CACHE, _DISK_CACHE_AT, _DISK_CACHE_KEY
    _DISK_CACHE = {}
    _DISK_CACHE_AT = 0.0
    _DISK_CACHE_KEY = ""


def disk_stats(path: Path | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """Honest free/total space for the filesystem holding `path`.

    ``freeBytes`` is the CONSERVATIVE of two real measurements: what statvfs
    says is free, and what is actually reachable given the bytes our own tree
    occupies. It is never more optimistic than the filesystem, and when the
    filesystem cannot be read it is None (unknown), never 0 — a confident zero
    refuses every job at submit time.
    """
    global _DISK_CACHE, _DISK_CACHE_AT, _DISK_CACHE_KEY
    probe = Path(path) if path is not None else Path(ROOT)
    # Walk up to the nearest existing ancestor: a not-yet-created job dir must
    # report the filesystem it will live on, not "0 free".
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    key = str(probe)
    with _STATS_LOCK:
        if _DISK_CACHE and _DISK_CACHE_KEY == key and time.time() - _DISK_CACHE_AT < 5.0:
            return dict(_DISK_CACHE)
        out: dict[str, Any] = {
            "path": str(probe),
            "totalBytes": None,
            "freeBytes": None,
            "statvfsFreeBytes": None,
            "treeBytes": None,
            "totalMb": None,
            "freeMb": None,
        }
        try:
            usage = shutil.disk_usage(str(probe))
            out["totalBytes"] = int(usage.total)
            out["statvfsFreeBytes"] = int(usage.free)
        except Exception as exc:  # noqa: BLE001 - unknown, not zero
            logger.debug(f"[hosting] disk_usage({probe}) failed: {exc}")
        try:
            if settings is not None:
                p = settings.app.paths
                roots = [p.data_dir, p.outputs_dir, p.downloads_dir]
            else:
                roots = [probe]
            tree = _measure_tree_bytes(roots)
            out["treeBytes"] = tree
            if out["totalBytes"] is not None:
                out["freeBytes"] = min(
                    out["statvfsFreeBytes"],
                    max(0, out["totalBytes"] - tree),
                )
            else:
                out["freeBytes"] = out["statvfsFreeBytes"]
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[hosting] tree measurement failed: {exc}")
            out["freeBytes"] = out["statvfsFreeBytes"]
        out["totalMb"] = mb_from_bytes(out["totalBytes"])
        out["freeMb"] = mb_from_bytes(out["freeBytes"])
        _DISK_CACHE = out
        _DISK_CACHE_AT = time.time()
        _DISK_CACHE_KEY = key
        return dict(out)


def wait_for_space(
    paths: Any,
    need_bytes: int,
    *,
    cap_bytes: int | None = None,
    timeout_sec: float = 1800.0,
    poll_sec: float = 5.0,
    on_wait: Any = None,
) -> tuple[bool, str]:
    """Block until the host has room for `need_bytes`, cleaning as it waits.

    Returns ``(ok, reason)``. ``on_wait(reason, free, scratch)`` is called on
    every tick so the job can report "Waiting for disk space" with the numbers
    instead of going silent. A caller that never gets space gets ``(False,
    reason)`` and can decide — it is not an exception here.
    """
    data_dir = Path(getattr(paths, "data_dir", paths))
    deadline = time.time() + timeout_sec
    last_reason = ""
    while True:
        stats = disk_stats(data_dir)
        free = stats.get("freeBytes")
        if free is None:
            return True, "free space unreadable — proceeding"
        scratch = stats.get("treeBytes") or 0
        over_cap = bool(cap_bytes) and scratch + int(need_bytes) > int(cap_bytes)
        if free >= need_bytes and not over_cap:
            return True, ""
        last_reason = (
            f"{mb_from_bytes(free)} MB free, need {mb_from_bytes(need_bytes)} MB"
            if free < need_bytes
            else f"scratch would pass the {mb_from_bytes(cap_bytes)} MB cap"
        )
        if time.time() >= deadline:
            return False, last_reason
        if on_wait is not None:
            try:
                on_wait(last_reason, free, scratch)
            except Exception:  # noqa: BLE001 - a progress hook must not break the wait
                pass
        time.sleep(poll_sec)


def shutil_which(name: str) -> str | None:
    return shutil.which(name)


class DispatchError(Exception):
    """A remote worker rejected or could not accept a job."""


def _auth_headers(settings: Settings) -> dict[str, str]:
    token = getattr(settings.hosted, "worker_token", "") or ""
    return {"Authorization": f"Bearer {token}"} if token else {}


def tailnet_ip() -> str | None:
    """This process's Tailscale 100.x address when tailscaled is reachable."""
    try:
        import subprocess

        out = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                ip = line.strip()
                if _is_tailnet(ip):
                    return ip
    except Exception:
        pass
    try:
        for _name, addrs in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = addrs[0]
            if _is_tailnet(ip):
                return ip
    except Exception:
        pass
    return None


def _is_tailnet(ip: str) -> bool:
    return ip.startswith("100.") and 64 <= int(ip.split(".")[1] or 0) <= 127


def remote_probe(settings: Settings) -> dict[str, Any]:
    """Is the configured remote worker answering? Never invents success."""
    url = getattr(settings.hosted, "worker_url", "") or ""
    if not url:
        return {"enabled": False, "detail": "no WORKER_URL — running locally"}
    import urllib.request

    try:
        req = urllib.request.Request(f"{url}/agent/health", headers=_auth_headers(settings))
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        return {
            "enabled": True,
            "online": True,
            "detail": str(data.get("version", "worker-agent")),
        }
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {
            "enabled": True,
            "online": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }
