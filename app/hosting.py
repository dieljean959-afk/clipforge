from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

#: Hosted release version, e.g. "1.1.0".
HOSTED_VERSION: str = "1.1.0"

#: The major version component of `_HOSTED_VERSION`.
NODE_VERSION_MAJOR: int = int(HOSTED_VERSION.split(".")[0])

from app.config import ROOT, Settings
from app.config import load_config
from app.store import LocalStore
from app.stages import remove_workspace
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
        """Delete expired artefacts. Returns count removed."""
        import shutil
        from app import reclaim

        paths = reclaim.load_paths()
        cutoff = time.time() - (self.settings.hosted.retention_hours or 0) * 3600
        deleted = 0

        for root in (paths.downloads, paths.outputs):
            if root is None or not root.exists():
                continue
            for p in root.rglob("*"):
                try:
                    if p.is_symlink() or not p.exists():
                        continue
                    if p.stat().st_mtime < cutoff and reclaim.is_deletable(p):
                        shutil.rmtree(p.parent if p.is_dir() else p, ignore_errors=True)
                        deleted += 1
                except Exception:
                    pass
        return deleted



from app import config as app_config
#: Hosted release version, e.g. "1.1.0". The dashboard reads this so the UI
#: can surface the exact build even if the worker/daemon restarts.



from app.stages import remove_workspace

# --------------------------------------------------------------------------- #
# system stats (never NaN — None means "unknown", the UI renders it honestly)
# --------------------------------------------------------------------------- #

_STATS_CACHE: dict[str, Any] = {}
_STATS_CACHE_AT = 0.0
_STATS_LOCK = threading.Lock()


def _proc_stat_cpu_percent() -> float | None:
    """Sample /proc/stat twice (0.05s apart) → busy percentage, or None.

    The gap is a blocking sleep inside /api/health (2s cache). 150ms was long
    enough to push a health poll past the supervisor's probe timeout on a busy
    1-vCPU host, so it is kept short — the delta is still measurable, and
    jiffies are 10ms, so 50ms resolves the percentage. """
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


#: How long a measured tree size stays fresh. Walking every job artefact costs
#: ~230 ms for ~2,000 files, and /api/health is polled every 10 s by the
#: dashboard, so measuring on every call would spend a quarter of a core
#: continuously to print a number that barely moves.
_TREE_CACHE_TTL_SEC = 60.0
#: node_modules + .venv are ~1 GB and take ~1.8 s to walk, but they are
#: installed before the app boots and never change while it runs. Re-walking
#: them every minute would burn ~3% of a core on a 1-vCPU sandbox to produce a
#: number that cannot have moved. Six hours outlives the sandbox itself.
_STATIC_CACHE_TTL_SEC = 6 * 3600.0
_tree_cache: dict[str, tuple[float, int]] = {}
_static_cache: tuple[float, int] | None = None
_tree_cache_lock = threading.Lock()


def invalidate_disk_cache() -> None:
    """Forget the measured tree sizes because a delete just changed them.

    /api/health is polled every few seconds and the reader expects "free" to
    move the moment a batch is removed. The tree measurement is cached for a
    minute, so without this the card kept showing the pre-delete number and a
    Clean up looked like it freed nothing.
    """
    with _tree_cache_lock:
        _tree_cache.clear()


def _cached_measure(key: str, roots: tuple[Path, ...], ttl: float) -> int:
    """Measure `roots`, reusing the last answer while it is still fresh."""
    now = time.monotonic()
    with _tree_cache_lock:
        hit = _tree_cache.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
    # Measured OUTSIDE the lock: this is filesystem I/O and holding a lock
    # across it would block /api/health for everyone while one caller walks
    # 45,000 files. A concurrent caller just does the same work twice.
    value = _measure_tree_bytes(roots)
    with _tree_cache_lock:
        _tree_cache[key] = (now, value)
    return value


def _static_tree_bytes(root: Path) -> int:
    """Bytes held by the app's own dependencies, cached for hours."""
    global _static_cache
    now = time.monotonic()
    with _tree_cache_lock:
        if _static_cache is not None and now - _static_cache[0] < _STATIC_CACHE_TTL_SEC:
            return _static_cache[1]
    value = _measure_tree_bytes(
        (root / "node_modules", root / ".venv", root / "dist", root / ".git")
    )
    with _tree_cache_lock:
        _static_cache = (now, value)
    return value


def _measure_tree_bytes(paths: Iterable[Path]) -> int:
    """Sum the real size of the directories ClipForge writes media into.

    Deliberately counts the FILES rather than trusting the filesystem's own
    block accounting. On this sandbox the root is virtiofs and `statvfs`
    reports 72 MB used while `du` finds 1.2 GB — the kernel's numbers simply
    do not describe what is on disk, so anything derived from them is a
    fiction. Walking the tree is the only measurement here that can be
    believed.
    """
    total = 0
    for root in paths:
        if not root or not root.exists():
            continue
        try:
            for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
                for name in filenames:
                    p = Path(dirpath) / name
                    try:
                        # st_size, not st_blocks: this is what the app will
                        # read and write, and it is the number the user can
                        # reason about. Symlinks are skipped so a link into the
                        # tree is not counted twice.
                        if p.is_symlink():
                            continue
                        total += p.stat().st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def disk_stats(path: Path | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """Disk free/total for the volume jobs ACTUALLY write to.

    Measured on the DATA dir, not the repo root. CLIPFORGE_DATA_DIR and
    CLIPFORGE_STORAGE_DIR exist precisely so every job artefact can be moved
    onto a larger mounted volume — and reporting the sandbox root in that case
    describes a filesystem nothing is written to, so the Doctor card would
    show a full disk that is in fact empty. Unknown stays None rather than a
    confident wrong number.

    `freeBytes` is the CONSERVATIVE figure, not the filesystem's own: it is
    the smaller of what statvfs claims and what the filesystem size minus the
    bytes ClipForge can actually see on disk. On a host whose block accounting
    is broken the second term is far smaller and correct, and a preflight
    built on the first term would wave through a job that dies at clip 9 of 10.
    Under-estimating free space is the safe direction for a gate; it never
    admits a job that could not have fit.
    """
    stats: dict[str, Any] = {
        "freeBytes": None, "totalBytes": None, "path": None,
        "treeBytes": None, "statvfsFreeBytes": None,
        "mediaBytes": None, "depsBytes": None,
    }
    target = Path(path) if path is not None else ROOT
    try:
        import shutil as _shutil

        # A data dir pointing at a volume that is not mounted must not read as
        # "0 bytes free" (which refuses every job at submit time) — walk up to
        # the nearest parent that does exist and report the real filesystem.
        probe = target
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = _shutil.disk_usage(probe)
        stats["totalBytes"] = usage.total
        stats["statvfsFreeBytes"] = usage.free
        stats["path"] = str(probe)

        # Only worth measuring when something can actually be measured.
        if settings is None:
            try:

                settings = load_config(quiet=True)
            except Exception:
                settings = None
        if settings is not None:
            p = settings.app.paths
            media = _cached_measure(
                "media:" + str(probe),
                (p.data_dir, p.outputs_dir, p.downloads_dir, p.uploads_dir),
                _TREE_CACHE_TTL_SEC,
            )
            deps = _static_tree_bytes(ROOT)
            tree = media + deps
            stats["treeBytes"] = tree
            stats["mediaBytes"] = media
            stats["depsBytes"] = deps
            # Everything we can SEE on this volume is now counted, so
            # total-minus-tree really is the space a new job has to fit in.
            # The min with statvfs keeps the conservative property even on a
            # host whose accounting is sane but reports a smaller number.
            visible_free = usage.total - tree
            stats["freeBytes"] = min(usage.free, max(0, visible_free))
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






