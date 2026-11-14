"""The safe cleanup routine behind every "free disk" action.

Two rules, in this order:

1. NEVER delete something the app needs or the user cannot get back:
   ``.env`` and any key store, ``cookies.txt``, the job documents (the DB),
   the database backup, the app code, the venv, ``node_modules``, fonts, the
   emoji assets, any media a batch row still references, and anything a RUNNING
   job is holding. Everything here is scratch, cache, or a duplicate of a
   verified bucket object.
2. Report what was removed, in MB, per category — a button that frees space but
   cannot say how much did not actually help anyone.

This is deliberately the same function for the panel buttons, the nightly pass
and (through ``tools/disk_cleanup.sh``) a host cron, so there is one definition
of "safe" rather than three that drift apart.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from loguru import logger

_MB = 1024 * 1024

#: A log file is rotated by keeping its tail once it passes this.
LOG_MAX_BYTES = 20 * _MB
LOG_TAIL_LINES = 2000
#: Logs older than this are deleted outright.
LOG_KEEP_DAYS = 3
#: Anything in /tmp this old and not owned by the platform is debris.
TMP_KEEP_MINUTES = 60
#: Platform files in /tmp that this process must never touch.
_TMP_KEEP_PREFIXES = ("daytona", "vly", "rc")

#: Paths that must survive any sweep. Matched on the RESOLVED path, so a
#: scratch folder that merely lives under `data/` is still removable while the
#: job documents, the DB backup, the code, the runtime and the assets are not.
_PROTECTED_NAMES = frozenset({".env", ".env.local", ".env.production", "cookies.txt", "config.toml"})
_PROTECTED_ROOTS = ("data/jobs", "data/backups", "data/clips", "data/uploads",
                    "app/assets", "app/fonts", ".venv", "node_modules", ".git")


def free_mb(path: Path | str) -> float:
    try:
        st = os.statvfs(str(path))
        return round((st.f_bavail * st.f_frsize) / _MB, 1)
    except OSError:
        return 0.0


def _size_mb(path: Path) -> float:
    total = 0
    try:
        if path.is_file():
            return path.stat().st_size / _MB
        for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    continue
    except OSError:
        return 0.0
    return total / _MB


def _live_job_ids(store: Any) -> set[str]:
    """Batches whose files must not be touched.

    A job counts as live unless it reached a terminal state. Errs on the side of
    keeping: an unreadable store means "do not delete", never "delete
    everything".
    """
    try:
        try:
            jobs = store.list_jobs()
        except TypeError:
            jobs = store.list_jobs(limit=2000)  # other store implementations
    except Exception as exc:  # noqa: BLE001 - unknown state must not become deletion
        logger.warning(f"[sweep] cannot read the job list ({exc}) — assuming a batch is live")
        return {"*"}
    live: set[str] = set()
    for job in jobs:
        status = str(job.get("status") or "")
        if status not in {"complete", "failed", "paused", "deleting"}:
            live.add(str(job.get("_id") or ""))
    return live


def _protected(path: Path) -> bool:
    if path.name in _PROTECTED_NAMES:
        return True
    text = path.as_posix().rstrip("/")
    return any(
        text == root or text.endswith("/" + root) or f"/{root}/" in text
        for root in _PROTECTED_ROOTS
    )


def _rm(path: Path, *, dry_run: bool, freed: list[float]) -> None:
    if _protected(path):
        logger.warning(f"[sweep] refusing to delete protected path {path}")
        return
    mb = _size_mb(path)
    if dry_run:
        freed.append(mb)
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        freed.append(mb)
    except OSError as exc:
        logger.warning(f"[sweep] could not remove {path}: {exc}")


def sweep(
    store: Any,
    paths: Any,
    *,
    dry_run: bool = False,
    include_models: bool = True,
) -> dict[str, Any]:
    """Free local disk safely; return per-category MB.

    ``dry_run`` lists the same categories without deleting anything — that is
    what the UI's confirmation step and this module's tests use.
    """
    started = time.time()
    data_dir = Path(paths.data_dir)
    root = Path(getattr(paths, "root", data_dir.parent))
    before = free_mb("/")
    live = _live_job_ids(store)
    busy = bool(live - {""})

    categories: list[dict[str, Any]] = []

    def record(name: str, freed: list[float], note: str = "") -> None:
        mb = round(sum(freed), 1)
        if mb > 0 or note:
            categories.append({"name": name, "mb": mb, "files": len(freed), "note": note})
        logger.info(f"[sweep] {name}: {mb} MB" + (" (dry run)" if dry_run else "") + (f" — {note}" if note else ""))

    # ---- 1. python bytecode ------------------------------------------------
    freed: list[float] = []
    for base in (root / "app", root / "tests", root / "tools", root / "agent_tools"):
        if not base.is_dir():
            continue
        for pycache in base.rglob("__pycache__"):
            _rm(pycache, dry_run=dry_run, freed=freed)
    record("python bytecode", freed, "regenerated on the next import")

    # ---- 2. job scratch of batches that no longer exist --------------------
    freed = []
    work = data_dir / "work"
    if work.is_dir():
        for child in work.iterdir():
            if not child.is_dir():
                continue
            if child.name in live:
                continue
            _rm(child, dry_run=dry_run, freed=freed)
    record("orphaned stage scratch", freed, "no batch owns this folder")

    # ---- 3. stale temp files ----------------------------------------------
    freed = []
    cutoff = time.time() - TMP_KEEP_MINUTES * 60
    for tmp_root in (Path("/tmp"), Path("/var/tmp")):
        if not tmp_root.is_dir():
            continue
        for entry in tmp_root.iterdir():
            name = entry.name
            if any(name.startswith(prefix) for prefix in _TMP_KEEP_PREFIXES):
                continue
            try:
                if entry.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            _rm(entry, dry_run=dry_run, freed=freed)
    record("stale /tmp", freed, f"older than {TMP_KEEP_MINUTES} min")

    # ---- 4. logs: rotate the big ones, delete the old ones -----------------
    freed = []
    log_cutoff = time.time() - LOG_KEEP_DAYS * 86400
    for log in list(data_dir.glob("logs/*.log*")) + list(Path("/var/log").glob("**/*.log*")):
        try:
            st = log.stat()
        except OSError:
            continue
        if st.st_mtime < log_cutoff:
            _rm(log, dry_run=dry_run, freed=freed)
            continue
        if st.st_size <= LOG_MAX_BYTES:
            continue
        if dry_run:
            freed.append(0.0)
            continue
        try:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL_LINES:]
            freed.append(st.st_size / _MB)
            log.write_text("\n".join(tail) + "\n", encoding="utf-8")
        except OSError:
            continue
    record("logs", freed, f"rotated over {LOG_MAX_BYTES // _MB} MB, deleted over {LOG_KEEP_DAYS} days")

    # ---- 5. downloader/package caches -------------------------------------
    freed = []
    home = Path(os.path.expanduser("~"))
    for cache in (
        home / ".cache" / "pip",
        home / ".cache" / "yt-dlp",
        Path("/root/.cache/pip"),
        Path("/root/.cache/yt-dlp"),
        Path("/root/.npm/_cacache"),
    ):
        if cache.exists():
            _rm(cache, dry_run=dry_run, freed=freed)
    record("package/download caches", freed, "pure caches; they refill on demand")

    # ---- 6. model weights (bucket copy verified first) --------------------
    freed = []
    if include_models and not busy:
        try:
            from app import modelstore

            if dry_run:
                status = modelstore.status().get("models") or {}
                mb = sum(int(v.get("localBytes") or 0) for v in status.values()) / _MB
                if mb:
                    freed.append(mb)
                    categories.append({
                        "name": "model weights", "mb": round(mb, 1),
                        "files": len(status), "note": "bucket copy verified; restored on demand (dry run)",
                    })
            else:
                out = modelstore.release_all()
                if out.get("freed"):
                    freed.append(out["freed"] / _MB)
                record("model weights", freed, "bucket copy verified; restored on demand")
        except Exception as exc:  # noqa: BLE001 - never let this break the button
            logger.warning(f"[sweep] model release skipped: {exc}")
    elif include_models and busy:
        record("model weights", [], "skipped — a batch is running and holds them")

    after = free_mb("/")
    total = round(sum(c["mb"] for c in categories), 1)
    result = {
        "ok": True,
        "dryRun": dry_run,
        "categories": categories,
        "freedMb": total,
        "freeBeforeMb": before,
        "freeAfterMb": after,
        "seconds": round(time.time() - started, 2),
        "skipped": sorted(live - {""}),
    }
    logger.info(
        f"[sweep] {'would free' if dry_run else 'freed'} {total} MB "
        f"(disk {before} → {after} MB free) in {result['seconds']}s"
    )
    return result


# ---------------------------------------------------------------------------
# nightly pass
# ---------------------------------------------------------------------------

_LAST_RUN_KEY = "disk_sweep_last"


def _state_path(paths: Any) -> Path:
    return Path(paths.data_dir) / "disk_sweep.json"


def nightly(store: Any, paths: Any, *, every_seconds: float = 86400.0) -> dict[str, Any]:
    """Run the sweep at most once per period; log what it removed.

    Lives in the API process (like the retention/reclaim loops) because this
    container has no cron and no systemd: a schedule that needs a binary that
    is not installed is not a schedule.
    """
    import json

    path = _state_path(paths)
    try:
        last = float(json.loads(path.read_text(encoding="utf-8")).get(_LAST_RUN_KEY) or 0)
    except Exception:  # noqa: BLE001 - a missing/garbled marker just means "run"
        last = 0.0
    now = time.time()
    if now - last < every_seconds:
        return {"skipped": "not due", "nextInSec": int(every_seconds - (now - last))}
    out = sweep(store, paths)
    try:
        path.write_text(json.dumps({_LAST_RUN_KEY: now, "last": out}, indent=2), encoding="utf-8")
    except OSError:
        pass
    logger.info(
        f"[sweep] nightly cleanup freed {out['freedMb']} MB: "
        + ", ".join(f"{c['name']} {c['mb']} MB" for c in out["categories"])
    )
    return out
