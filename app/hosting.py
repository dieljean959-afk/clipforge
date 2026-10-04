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


def disk_stats(path: Path | None = None) -> dict[str, Any]:
    """Disk free/total for the volume jobs ACTUALLY write to.

    Measured on the DATA dir, not the repo root. CLIPFORGE_DATA_DIR and
    CLIPFORGE_STORAGE_DIR exist precisely so every job artefact can be moved
    onto a larger mounted volume — and reporting the sandbox root in that case
    describes a filesystem nothing is written to, so the Doctor card would
    show a full disk that is in fact empty. Unknown stays None rather than a
    confident wrong number.
    """
    stats: dict[str, Any] = {"freeBytes": None, "totalBytes": None, "path": None}
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
        stats["freeBytes"] = usage.free
        stats["totalBytes"] = usage.total
        stats["path"] = str(probe)
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
            "[hosting] retention cleaner online — outputs deleted "
            + (
                "never (retention disabled)"
                if self.settings.hosted.retention_hours <= 0
                else f"{self.settings.hosted.retention_hours}h after completion"
            )
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
        removed = self.sweep_orphans()
        if self.settings.hosted.retention_hours <= 0:
            # retention_hours = 0 means "never expire anything". A library
            # that silently empties itself while nobody is looking is far
            # worse than a disk that fills up, so the off switch has to be
            # reachable from config alone. The orphan sweep above still runs:
            # it only removes artefacts whose job record is GONE, which is not
            # "deleting the user's history".
            return removed
        cutoff = time.time() * 1000 - self.settings.hosted.retention_hours * 3600 * 1000
        for job in self.store.list_jobs():
            done_at = job.get("completedAt")
            stage = job.get("stage")
            # ONLY a finished job is ever expired. The old second clause
            # ("any stage, if createdAt is old enough") could delete a job that
            # was still queued or mid-render: a long source sitting in the queue
            # behind another job ages past the clock, and cleanup then removed
            # the source out from under a pipeline about to read it. Retention
            # is for finished work; orphans (the `sweep_orphans` path) already
            # handle records that no longer exist at all.
            if stage not in {"complete", "failed"}:
                continue
            expired = bool(done_at and done_at < cutoff)
            if expired:
                job_id = job["_id"]
                self.store.delete_job(job_id)
                import shutil as _shutil

                for d in self.settings.app.paths.job_source_dirs(job_id) + (
                    self.settings.app.paths.outputs_dir / job_id,
                ):
                    _shutil.rmtree(d, ignore_errors=True)
                # per-stage mirrors: extracted audio WAVs + per-chunk transcript
                # caches are the biggest artifacts a long job leaves behind
                remove_workspace(self.settings.app.paths.data_dir, job_id)
                removed += 1
                logger.info(f"[hosting] retention: removed job {job_id} artifacts")
        return removed

    # ------------------------------------------------------------------ #
    # orphans: disk nobody owns
    # ------------------------------------------------------------------ #

    #: How long an orphan must sit untouched before it is removed. Long enough
    #: that a job mid-restart — where the record and the folder are briefly out
    #: of step — is never mistaken for one, and short enough that the disk does
    #: not carry yesterday's debris forever.
    ORPHAN_MAX_AGE_SEC = float(os.getenv("CLIPFORGE_ORPHAN_MAX_AGE_SEC", str(7 * 86400)))

    def sweep_orphans(self) -> int:
        """Delete artefacts whose job record no longer exists, plus stale temps.

        Every job writes a lot of bytes nobody tracks: a source download, a
        per-stage workspace of extracted WAVs and transcript chunks, rendered
        clips and their posters. When a job is deleted, tombstoned, or its
        record is lost to a host restart, all of that stays behind forever. On
        a 4 GB volume a handful of abandoned 700 MB sources is the difference
        between finishing a job and failing with "disk almost full".

        Only things with NO owning job record are touched. A live job's
        workspace is never a candidate, whatever its age.
        """
        known = {j["_id"] for j in self.store.list_jobs()}
        # Belt and braces alongside the `known` check: a job that is queued or
        # running right now must never be swept even if its record somehow
        # fails to load. The sweeper runs from a daemon thread in the web
        # process, so "the record exists" is checked against a snapshot taken
        # microseconds earlier than the delete.
        active = {
            j["_id"] for j in self.store.list_jobs()
            if j.get("stage") in {"queued", "grabbing", "transcribing", "finding",
                                  "rendering", "captions", "export"}
        }
        cutoff = time.time() - self.ORPHAN_MAX_AGE_SEC
        removed = 0
        paths = self.settings.app.paths

        for root in (paths.downloads_dir, paths.uploads_dir or (paths.data_dir / "uploads"),
                   paths.outputs_dir, paths.data_dir / "work"):
            if not root.exists():
                continue
            for child in root.iterdir():
                if not child.is_dir() or child.name in known or child.name in active:
                    continue
                try:
                    if child.stat().st_mtime > cutoff:
                        continue
                except OSError:
                    continue
                import shutil as _shutil

                _shutil.rmtree(child, ignore_errors=True)
                removed += 1
                logger.info(f"[hosting] orphan sweep: removed {child}")

        # Clip documents for jobs that are gone, and half-written temp files
        # left by a process that was killed mid-write.
        clips_dir = paths.data_dir / "clips"
        if clips_dir.exists():
            for f in clips_dir.glob("*.json"):
                try:
                    # OLDER than the cutoff is the orphan; a fresh one is left
                    # alone. This comparison was inverted (`<= cutoff` to
                    # continue), so the sweeper deleted clip documents the
                    # instant they appeared and kept the stale ones forever —
                    # the exact opposite of the 24h rule every other branch
                    # here obeys, and the one place it could quietly discard a
                    # record that a job was about to re-adopt.
                    if f.stat().st_mtime > cutoff:
                        continue
                    clip = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if clip.get("jobId") not in known:
                    f.unlink(missing_ok=True)
                    removed += 1

        for root in (paths.downloads_dir, paths.uploads_dir or (paths.data_dir / "uploads"),
                   paths.outputs_dir):
            if not root.exists():
                continue
            for f in root.rglob("*"):
                if not f.is_file() or f.suffix.lower() not in {".part", ".tmp", ".bin"}:
                    continue
                try:
                    if f.stat().st_mtime <= cutoff:
                        f.unlink(missing_ok=True)
                        removed += 1
                except OSError:
                    continue

        # Deleted jobs are recoverable from the trash for a day, then they are
        # just disk. Same rule, same clock as everything else here.
        trash = getattr(paths, "trash", None) or (paths.data_dir / "trash")
        if trash.exists():
            for child in trash.iterdir():
                try:
                    if child.stat().st_mtime <= cutoff:
                        import shutil as _shutil

                        _shutil.rmtree(child, ignore_errors=True)
                        removed += 1
                except OSError:
                    continue
        return removed
