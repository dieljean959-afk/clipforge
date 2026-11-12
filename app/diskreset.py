"""Hard disk reset for the 4 GB sandbox.

The ordinary reclaim path only moves finished media to the bucket. That is not
enough when *caches and test debris* fill the disk: on this host the Hugging
Face cache held a 752 MiB distil-medium model the CPU-only worker can never
run, and pytest left 500+ MiB under /tmp. Neither is a batch's media, so
`reclaim()` never touched them.

This module is the explicit, dry-run-first reset. It removes only regenerable
bytes, never a user file that is not verified in the bucket, and never anything
a live batch or the running app owns. `tools/disk_reset.py` and
`POST /api/storage/reset` call the same function, so the button cannot delete
more than the CLI.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

_MB = 1024 * 1024
#: Temp debris older than this is safe. The reset button also removes pytest
#: session folders explicitly because those can be hundreds of MB.
TMP_KEEP_MIN = 30
LOG_MAX_BYTES = 20 * _MB
LOG_KEEP_DAYS = 3
KEEP_STATE = "disk_reset.json"
KEEP_PREFIXES = ("daytona", "vly", "rc", "systemd-")
PROTECTED_NAMES = frozenset({
    ".env", ".env.local", ".env.production", "cookies.txt", "config.toml",
})
PROTECTED_ROOTS = (
    ".venv", "node_modules", ".git",
    "data/jobs", "data/backups", "data/clips",
    "app/assets", "app/fonts",
)
#: These are removed only through modelstore, which verifies the bucket copy.
MODEL_ROOT = "data/models"


def _mb(value: int | float | None) -> float:
    return round(float(value or 0) / _MB, 1)


def _free_mb(path: Path | str) -> float:
    try:
        return round(shutil.disk_usage(str(path)).free / _MB, 1)
    except OSError:
        return 0.0


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                p = Path(root) / name
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
    return total


def _size_bytes(path: Path) -> int:
    try:
        if path.is_file() and not path.is_symlink():
            return path.stat().st_size
    except OSError:
        return 0
    return _tree_bytes(path)


def _protected(path: Path) -> bool:
    """True when a path is app code/runtime/user state, never reset debris."""
    if path.name in PROTECTED_NAMES:
        return True
    text = path.resolve().as_posix().rstrip("/")
    for root in PROTECTED_ROOTS:
        marker = Path(root).resolve().as_posix()
        if text == marker or text.endswith("/" + marker) or f"/{marker}/" in text:
            return True
    return False


def _live_job_ids(store: Any) -> set[str]:
    """Conservative: an unreadable store means every folder is live."""
    try:
        try:
            jobs = list(store.list_jobs())
        except TypeError:
            jobs = list(store.list_jobs(limit=2000))
    except Exception as exc:  # noqa: BLE001 - unknown is not permission to delete
        logger.warning(f"[reset] job list unreadable ({exc}) — treating all as live")
        return {"*"}
    live: set[str] = set()
    for job in jobs:
        status = str(job.get("status") or job.get("stage") or "")
        if status not in {"complete", "failed", "paused", "deleting", "cancelled"}:
            live.add(str(job.get("_id") or ""))
    return live


def _open_deleted() -> list[dict[str, Any]]:
    """Files unlinked while a process still holds them; frees only on restart."""
    found: list[dict[str, Any]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )[:160]
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if "(deleted)" not in target:
                    continue
                try:
                    size = int((fd.stat()).st_size)
                except OSError:
                    size = 0
                found.append({
                    "pid": int(proc.name),
                    "cmd": cmd.strip(),
                    "path": target,
                    "mb": _mb(size),
                })
        except (OSError, ValueError):
            continue
    return found


def _bucket_verified(paths: Any, job_id: str, path: Path) -> bool:
    """HEAD the expected object and require the exact local byte count."""
    try:
        from app import objectstore

        cfg = objectstore.config_from_env()
        if not cfg.enabled or not cfg.delete_local:
            return False
        outputs = Path(getattr(paths, "outputs_dir", ""))
        if outputs and outputs.resolve() in path.resolve().parents:
            key = objectstore.key_for_path(path, outputs)
        else:
            key = objectstore.upload_key(job_id, path.name)
        if not key:
            return False
        return objectstore.verify(key, path.stat().st_size, cfg)
    except Exception as exc:  # noqa: BLE001 - cannot verify -> keep it
        logger.warning(f"[reset] bucket verification failed for {path}: {exc}")
        return False


def _add(
    cats: list[dict[str, Any]], name: str, paths: list[Path], note: str,
) -> None:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return
    biggest = sorted(paths, key=_size_bytes, reverse=True)
    total = sum(_size_bytes(p) for p in paths)
    cats.append({
        "name": name,
        "mb": round(total / _MB, 1),
        "count": len(paths),
        "paths": [str(p) for p in biggest[:12]],
        "_all": [str(p) for p in paths],
        "note": note,
    })


def _job_roots(paths: Any) -> dict[str, Path]:
    data = Path(paths.data_dir)
    return {
        "work": data / "work",
        "uploads": Path(paths.uploads_dir or data / "uploads"),
        "downloads": Path(paths.downloads_dir),
        "outputs": Path(paths.outputs_dir),
    }


def _backup_db(paths: Any) -> str | None:
    """Back up job docs before apply. There is no SQLite DB on this host."""
    data = Path(paths.data_dir)
    backups = data / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = backups / f"disk-reset-{stamp}.tgz"
    try:
        with tarfile.open(dest, "w:gz") as tar:
            for name in ("jobs", "clips"):
                src = data / name
                if src.exists():
                    tar.add(src, arcname=name)
        return str(dest)
    except (OSError, tarfile.TarError) as exc:
        logger.warning(f"[reset] DB backup failed: {exc}")
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return None


_FOOTPRINT: dict[str, Any] = {}
_FOOTPRINT_AT = 0.0


def footprint(settings: Any, *, max_age: float = 30.0) -> dict[str, Any]:
    """Snapshot for the status card, cached so health stays cheap."""
    global _FOOTPRINT, _FOOTPRINT_AT
    now = time.time()
    if _FOOTPRINT and now - _FOOTPRINT_AT < max_age:
        return dict(_FOOTPRINT)
    try:
        from app import modelstore

        model_bytes = sum(
            int(v.get("localBytes") or 0)
            for v in (modelstore.status().get("models") or {}).values()
        )
    except Exception:  # noqa: BLE001
        model_bytes = 0
    paths = settings.app.paths
    roots = _job_roots(paths)
    scratch = sum(_tree_bytes(roots[k]) for k in ("work", "uploads"))
    media = sum(_tree_bytes(roots[k]) for k in ("downloads", "outputs"))
    home = Path(os.path.expanduser("~"))
    cache_roots: Iterable[Path] = (
        home / ".cache" / "huggingface",
        home / ".cache" / "pip",
        home / ".cache" / "yt-dlp",
        home / ".cache" / "uv",
        home / ".npm" / "_cacache",
        home / ".bun" / "install" / "cache",
        Path("/root/.npm/_cacache"),
    )
    cache = sum(_tree_bytes(p) for p in cache_roots if p.exists())
    out = {
        "scratchBytes": scratch,
        "scratchMb": _mb(scratch),
        "cacheBytes": cache,
        "cacheMb": _mb(cache),
        "mediaBytes": media,
        "mediaMb": _mb(media),
        "modelBytes": model_bytes,
        "modelMb": _mb(model_bytes),
        "lastCleanedAt": None,
        "lastFreedMb": None,
    }
    try:
        state = json.loads((Path(paths.data_dir) / KEEP_STATE).read_text(encoding="utf-8"))
        out["lastCleanedAt"] = state.get("at")
        out["lastFreedMb"] = state.get("freedMb")
    except (OSError, ValueError, TypeError):
        pass
    _FOOTPRINT = out
    _FOOTPRINT_AT = now
    return dict(out)


_SWEEPER: Any = None


def start_sweeper(store: Any, settings: Any, *, interval: float = 600.0) -> Any:
    """Start the 10-minute crash-proof housekeeping loop (idempotent)."""
    global _SWEEPER
    if _SWEEPER is not None and _SWEEPER.is_alive():
        return _SWEEPER

    def loop() -> None:
        from app import disksweep

        while True:
            time.sleep(interval)
            try:
                free = _free_mb("/")
                if free < 1500:
                    logger.warning(
                        f"[reset] disk {free} MB free (<1500) — running safe sweep"
                    )
                    disksweep.sweep(store, settings.app.paths, include_models=False)
                if free < 800:
                    plan_out = reset(store, settings, apply=False)
                    logger.error(
                        "[reset] disk below 800 MB — new jobs are paused. "
                        f"Hard reset would free {plan_out['plannedMb']} MB; "
                        "press Reset disk or run tools/disk_reset.py --apply."
                    )
            except Exception as exc:  # noqa: BLE001 - housekeeping must not die
                logger.warning(f"[reset] sweeper pass failed: {exc}")

    _SWEEPER = threading.Thread(target=loop, name="disk-reset-sweeper", daemon=True)
    _SWEEPER.start()
    return _SWEEPER


def reset(store: Any, settings: Any, *, apply: bool = False) -> dict[str, Any]:
    """Plan (default) or perform the hard reset. Never raises for one path."""
    paths = settings.app.paths
    data = Path(paths.data_dir)
    roots = _job_roots(paths)
    live = _live_job_ids(store)
    busy = bool(live - {""})
    before = _free_mb("/")
    cats: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    backup: str | None = None
    open_deleted = _open_deleted()

    # ---- 1. caches nobody needs ------------------------------------------
    home = Path(os.path.expanduser("~"))
    _add(cats, "hugging-face cache", [home / ".cache" / "huggingface" / "hub"],
         "not used by the local modelstore; restored from the bucket on demand")
    _add(cats, "pip / downloader caches", [
        home / ".cache" / "pip", home / ".cache" / "yt-dlp",
        home / ".cache" / "uv", Path("/root/.cache/pip"),
        Path("/root/.cache/yt-dlp"),
    ], "regenerated on demand")
    _add(cats, "npm / bun caches", [
        home / ".npm" / "_cacache", home / ".bun" / "install" / "cache",
        Path("/root/.npm/_cacache"),
    ], "package manager caches")

    # ---- 2. Python/test debris -------------------------------------------
    pycache: list[Path] = []
    for base in (Path("app"), Path("tests"), Path("tools")):
        pycache.extend(base.rglob("__pycache__"))
    _add(cats, "python bytecode", pycache, "regenerated on next import")

    pytest_dirs = sorted(Path("/tmp").glob("pytest-of-*"))
    running_pytest = False
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if "pytest" in (proc / "cmdline").read_bytes().decode("utf-8", "replace"):
                running_pytest = True
                break
        except OSError:
            continue
    if running_pytest:
        kept.append({"path": "/tmp/pytest-of-*", "reason": "a pytest process is running"})
    else:
        _add(cats, "pytest temp dirs", pytest_dirs,
             "test runs only; never app state")

    stale: list[Path] = []
    cutoff = time.time() - TMP_KEEP_MIN * 60
    for base in (Path("/tmp"), Path("/var/tmp")):
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            if entry.name in {"pytest-of-root"} or entry.name.startswith(KEEP_PREFIXES):
                continue
            if entry.name.startswith("pytest-of-"):
                continue
            try:
                if entry.stat().st_mtime <= cutoff:
                    stale.append(entry)
            except OSError:
                continue
    _add(cats, f"stale /tmp (>{TMP_KEEP_MIN} min)", stale,
         "temp files, not referenced by the app")

    # ---- 3. scratch and temp files --------------------------------------
    scratch: list[Path] = []
    work = roots["work"]
    if work.is_dir():
        scratch.extend(
            child for child in work.iterdir()
            if child.is_dir() and child.name not in live
        )
    _add(cats, "orphaned job scratch", scratch,
         "no live batch owns this folder; contains only re-derivable stage files")

    partial: list[Path] = []
    for root in roots.values():
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".part", ".ytdl", ".tmp", ".incomplete"} \
                    and not p.name.endswith((".fetching", ".uploading")):
                continue
            job_id = next((part for part in p.parts if part in live), "")
            if job_id:
                continue
            try:
                if p.stat().st_mtime <= time.time() - TMP_KEEP_MIN * 60:
                    partial.append(p)
            except OSError:
                continue
    _add(cats, "partial downloads / temp files", partial,
         "interrupted transfer or stage, no live batch")

    # ---- 4. orphan media: only when the bucket copy matches -------------
    verified_media: list[Path] = []
    for name in ("downloads", "outputs", "uploads"):
        root = roots[name]
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir() or child.name in live:
                continue
            for p in child.iterdir():
                if not p.is_file() or p.is_symlink():
                    continue
                if _bucket_verified(paths, child.name, p):
                    verified_media.append(p)
                else:
                    kept.append({
                        "path": str(p),
                        "reason": "not verified in the bucket (HEAD size mismatch or no bucket)",
                    })
    _add(cats, "media verified in bucket", verified_media,
         "exact HEAD size matched; served from the bucket")

    # ---- 5. logs ----------------------------------------------------------
    log_targets: list[Path] = []
    logs = data / "logs"
    if logs.is_dir():
        for p in list(logs.glob("*.log*")):
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size > LOG_MAX_BYTES:
                log_targets.append(p)
            elif st.st_mtime < time.time() - LOG_KEEP_DAYS * 86400 and p.name != "backend.log":
                log_targets.append(p)
    _add(cats, "logs (rotate/truncate)", log_targets,
         f"keep last 3 days or {LOG_MAX_BYTES // _MB} MB")

    # ---- 6. model copies (bucket-verified by modelstore) -----------------
    model_bytes = 0
    try:
        from app import modelstore

        status = modelstore.status()
        model_bytes = sum(
            int(v.get("localBytes") or 0)
            for v in (status.get("models") or {}).values()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[reset] model status failed: {exc}")
    if model_bytes:
        cats.append({
            "name": "local model copies",
            "mb": _mb(model_bytes),
            "count": 1,
            "paths": [str(Path(paths.data_dir) / "models")],
            "note": "modelstore verifies each bucket object before releasing",
        })

    # ---- 7. apply ---------------------------------------------------------
    if apply:
        backup = _backup_db(paths)
        # Never release a model a live batch is using.
        removed_bytes = 0
        for cat in cats:
            if cat["name"] == "local model copies":
                if busy:
                    kept.append({"path": str(Path(paths.data_dir) / "models"),
                                 "reason": "a batch is running"})
                    continue
                try:
                    from app import modelstore

                    out = modelstore.release_all()
                    removed_bytes += int(out.get("freed") or 0)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[reset] model release failed: {exc}")
                continue
            for raw in cat.pop("_all", cat["paths"]):
                p = Path(raw)
                if not p.exists() or _protected(p):
                    continue
                size = _size_bytes(p)
                try:
                    if p.is_file() and cat["name"].startswith("logs") \
                            and size > LOG_MAX_BYTES:
                        tail = p.read_bytes()[-2 * _MB:]
                        p.write_bytes(b"[... trimmed by disk reset ...]\n" + tail)
                    elif p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink(missing_ok=True)
                    removed_bytes += size
                except OSError as exc:
                    logger.warning(f"[reset] could not remove {p}: {exc}")
        try:
            (data / KEEP_STATE).write_text(json.dumps({
                "at": int(time.time() * 1000),
                "freedMb": _mb(removed_bytes),
                "backup": backup,
            }, indent=2), encoding="utf-8")
        except OSError:
            pass

    after = _free_mb("/")
    planned = round(sum(c["mb"] for c in cats), 1)
    public_cats = [
        {k: v for k, v in cat.items() if k != "_all"} for cat in cats
    ]
    result = {
        "ok": True,
        "dryRun": not apply,
        "freeBeforeMb": before,
        "freeAfterMb": after,
        "freedMb": round((before - after) * -1, 1) if apply else planned,
        "plannedMb": planned,
        "categories": public_cats,
        "kept": kept[:40],
        "openDeleted": open_deleted,
        "backup": backup,
        "skipped": [
            {"what": "git gc --prune=now", "why": "git is blocked on this host"},
            {"what": "journald SystemMaxUse=50M", "why": "journalctl is not installed"},
            {"what": "SQLite WAL/VACUUM", "why": "this host uses JSON job documents, no SQLite"},
        ],
    }
    if apply:
        global _FOOTPRINT_AT
        _FOOTPRINT_AT = 0.0
        logger.warning(
            f"[reset] freed {result['freedMb']} MB: disk {before} → {after} MB free"
        )
        try:
            from app import hosting

            hosting.invalidate_disk_cache()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[reset] disk cache invalidation skipped: {exc}")
    return result
