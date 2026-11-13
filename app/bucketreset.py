"""Dry-run-first bucket cleanup for finished ClipMake batches.

The local reset frees the 4 GB sandbox disk. This module frees the *plan*: the
source and audio of batches that are over, outputs whose clip documents were
deleted, objects whose job document is gone, and incomplete multipart uploads
that S3 counts against the bucket without ever showing them in a listing.

Safety rules, in order:
  * a live job's prefixes are never touched;
  * a terminal job is only touched after its DB documents are backed up;
  * an object with no DB reference must be at least an hour old (an upload in
    progress has no job document yet and would otherwise look like an orphan);
  * models/ is out of scope;
  * plan is the default; deletion happens only with ``apply=True``.
"""

from __future__ import annotations

import re
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

_MB = 1024 * 1024
_ORPHAN_MIN_AGE_SEC = 3600
_MULTIPART_MIN_AGE_SEC = 3600
_TERMINAL = {"complete", "failed", "paused", "cancelled", "deleted"}
_OUTPUT_RE = re.compile(r"/clip_(\d{3})(?:\.|$)")
_SHARE_PREFIXES = ("shares/", "share/", "tmp/", "presign/", "temp/")


def _mb(value: int | float | None) -> float:
    return round(float(value or 0) / _MB, 1)


def _ts(value: Any) -> float:
    """S3 LastModified / job timestamps → epoch seconds (0 when unknown)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    if isinstance(value, (int, float)):
        # Job documents use milliseconds; S3 sometimes gives seconds.
        return float(value) / 1000.0 if value > 10_000_000_000 else float(value)
    return 0.0


def _all_jobs(store: Any) -> list[dict[str, Any]]:
    try:
        return list(store.list_jobs())
    except TypeError:
        return list(store.list_jobs(limit=5000))


def _live_job_ids(store: Any) -> set[str]:
    """A job is live while any stage can still write to its prefixes."""
    live: set[str] = set()
    for job in _all_jobs(store):
        status = str(job.get("status") or job.get("stage") or "")
        if status not in _TERMINAL:
            live.add(str(job.get("_id") or ""))
    return {j for j in live if j}


def _backup_db(settings: Any) -> str | None:
    """Back up job + clip documents before a single bucket delete."""
    data = Path(settings.app.paths.data_dir)
    backups = data / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    dest = backups / f"bucket-reset-{time.strftime('%Y%m%d-%H%M%S')}.tgz"
    try:
        with tarfile.open(dest, "w:gz") as tar:
            for name in ("jobs", "clips"):
                src = data / name
                if src.exists():
                    tar.add(src, arcname=name)
        return str(dest)
    except (OSError, tarfile.TarError) as exc:
        logger.warning(f"[bucket-reset] DB backup failed: {exc}")
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _add_category(
    cats: dict[str, dict[str, Any]], name: str, reason: str,
    obj: dict[str, Any] | None = None, *,
    key: str = "", size: int = 0, last_modified: Any = None,
) -> None:
    cat = cats.setdefault(name, {
        "name": name, "reason": reason, "mb": 0.0, "bytes": 0, "count": 0,
        "keys": [], "items": [],
    })
    cat["count"] += 1
    cat["bytes"] += int(size)
    cat["mb"] = _mb(cat["bytes"])
    if len(cat["items"]) < 40:
        cat["items"].append({
            "key": key,
            "mb": _mb(size),
            "lastModified": last_modified,
        })
    cat["keys"].append(key)


def _output_order(key: str) -> int | None:
    m = _OUTPUT_RE.search(key)
    return int(m.group(1)) if m else None


def _clip_orders(store: Any, job_id: str) -> set[int]:
    try:
        clips = store.list_clips(job_id)
    except Exception:  # noqa: BLE001 - unknown is not permission to delete
        return set()
    return {int(c.get("order") or 0) for c in clips}


def plan(store: Any, settings: Any) -> dict[str, Any]:
    """List every bucket object that is safe to delete, grouped by reason."""
    from app import objectstore

    cfg = objectstore.config_from_env()
    if not cfg.enabled:
        raise RuntimeError("bucket is not configured")
    objs = objectstore.list_prefix(cfg=cfg, strict=True)
    total_bytes = sum(int(o.get("size") or 0) for o in objs)
    jobs = _all_jobs(store)
    by_id = {str(j.get("_id") or ""): j for j in jobs}
    live = _live_job_ids(store)
    now = time.time()
    retention_h = int(getattr(settings.hosted, "retention_hours", 0) or 0)
    cats: dict[str, dict[str, Any]] = {}
    protected: list[dict[str, Any]] = []

    for obj in objs:
        key = str(obj.get("key") or "")
        size = int(obj.get("size") or 0)
        modified = obj.get("lastModified")
        age = now - _ts(modified) if _ts(modified) else 0.0
        parts = key.split("/")
        # <prefix>/uploads/<job>/... or <prefix>/outputs/<job>/...
        if len(parts) < 4 or parts[1] not in {"uploads", "outputs"}:
            if any(parts[1].startswith(p.rstrip("/")) for p in _SHARE_PREFIXES if len(parts) > 1):
                if age >= _ORPHAN_MIN_AGE_SEC:
                    _add_category(
                        cats, "stale share/presign artifacts",
                        "temporary share or presigned object older than an hour",
                        obj, key=key, size=size, last_modified=modified,
                    )
            continue
        kind, job_id = parts[1], parts[2]
        if job_id in live:
            protected.append({"key": key, "reason": "live batch reference"})
            continue
        job = by_id.get(job_id)
        if job is None:
            if age < _ORPHAN_MIN_AGE_SEC:
                protected.append({"key": key, "reason": "young upload with no DB reference yet"})
                continue
            _add_category(
                cats, "no DB reference (orphans)",
                "batch document is gone; no dashboard can reach these bytes",
                obj, key=key, size=size, last_modified=modified,
            )
            continue

        status = str(job.get("status") or job.get("stage") or "")
        if status not in _TERMINAL:
            protected.append({"key": key, "reason": "batch is still running"})
            continue

        if kind == "uploads":
            # Audio and video of a batch that is over. The clips are separate
            # output objects. Keep the source while any clip is failed: Retry
            # renders it from this object, and deleting it would strand the one
            # clip the user is told to retry.
            try:
                clips = store.list_clips(job_id)
            except Exception:  # noqa: BLE001 - unknown is not permission to delete
                clips = []
            if any(str(c.get("renderState") or "") == "failed" for c in clips):
                protected.append({"key": key, "reason": "batch has a failed clip that can be retried"})
                continue
            if age >= _ORPHAN_MIN_AGE_SEC:
                _add_category(
                    cats, "source/audio of finished batches",
                    "the batch is terminal; its source/audio is no longer being read",
                    obj, key=key, size=size, last_modified=modified,
                )
            else:
                protected.append({"key": key, "reason": "recently written source/audio"})
            continue

        # outputs/<job>/...
        orders = _clip_orders(store, job_id)
        order = _output_order(key)
        if order is not None and order not in orders:
            _add_category(
                cats, "thumbnails/captions of deleted clips",
                "no clip document owns this output any more",
                obj, key=key, size=size, last_modified=modified,
            )
            continue
        if retention_h > 0:
            done_at = _ts(job.get("completedAt") or job.get("finishedAt") or job.get("updatedAt"))
            if done_at and (now - done_at) > retention_h * 3600:
                _add_category(
                    cats, "expired clips past retention",
                    f"batch finished more than {retention_h}h ago",
                    obj, key=key, size=size, last_modified=modified,
                )

    # Incomplete multipart uploads are not in list_objects at all. They still
    # consume space until aborted, which is why the bucket showed 0.96 GB in
    # the UI but the objects summed lower.
    multipart: list[dict[str, Any]] = []
    try:
        s3 = objectstore.client(cfg)
        if s3 is not None:
            resp = s3.list_multipart_uploads(Bucket=cfg.bucket, Prefix=f"{cfg.prefix}/")
            for up in resp.get("Uploads", []) or []:
                key = str(up.get("Key") or "")
                initiated = up.get("Initiated")
                age = now - _ts(initiated) if _ts(initiated) else 0.0
                if age < _MULTIPART_MIN_AGE_SEC:
                    protected.append({"key": key, "reason": "multipart upload is still recent"})
                    continue
                multipart.append({
                    "key": key,
                    "uploadId": up.get("UploadId"),
                    "initiated": initiated,
                    "ageHours": round(age / 3600, 1),
                })
    except Exception as exc:  # noqa: BLE001 - a listing failure is reported, not hidden
        logger.warning(f"[bucket-reset] multipart listing failed: {exc}")

    # Snapshot the keys BEFORE stripping them from the public categories:
    # `_keys` is what apply() deletes, and building it afterwards produced an
    # empty list while the UI showed 49 planned objects (a no-op apply).
    plan_keys = [key for cat in cats.values() for key in cat.get("keys", [])]
    public_cats = []
    for cat in cats.values():
        cat.pop("keys", None)
        public_cats.append(cat)
    public_cats.sort(key=lambda c: -c["bytes"])
    return {
        "ok": True,
        "scope": "bucket",
        "dryRun": True,
        "bucket": cfg.bucket,
        "prefix": cfg.prefix,
        "usageBeforeBytes": total_bytes,
        "usageBeforeMb": _mb(total_bytes),
        "plannedBytes": sum(c["bytes"] for c in public_cats),
        "plannedMb": _mb(sum(c["bytes"] for c in public_cats)),
        "objects": sum(c["count"] for c in public_cats),
        "categories": public_cats,
        "multipart": multipart,
        "multipartCount": len(multipart),
        "kept": protected[:80],
        "backup": None,
        "_keys": plan_keys,
        "_multipart": multipart,
        "_usageBefore": total_bytes,
    }


def apply(store: Any, settings: Any, plan_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Back up the DB, delete the planned keys, abort stale uploads."""
    from app import objectstore

    cfg = objectstore.config_from_env()
    if not cfg.enabled:
        raise RuntimeError("bucket is not configured")
    result = plan_result or plan(store, settings)
    backup = _backup_db(settings)
    if backup is None:
        raise RuntimeError("could not back up job documents; refusing to delete bucket objects")
    keys = list(result.get("_keys") or [])
    deleted = 0
    bytes_deleted = 0
    failures: list[str] = []
    for key in keys:
        try:
            meta = objectstore.head(key, cfg)
            size = int((meta or {}).get("size") or 0)
            if objectstore.delete(key, cfg):
                deleted += 1
                bytes_deleted += size
            else:
                failures.append(key)
        except Exception as exc:  # noqa: BLE001 - continue through the list
            failures.append(f"{key}: {type(exc).__name__}: {exc}")

    aborted = 0
    s3 = objectstore.client(cfg)
    for up in result.get("_multipart") or []:
        if not up.get("uploadId"):
            continue
        try:
            s3.abort_multipart_upload(
                Bucket=cfg.bucket, Key=up["key"], UploadId=up["uploadId"]
            )
            aborted += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"multipart {up['key']}: {type(exc).__name__}: {exc}")

    objectstore.invalidate_usage()
    after = objectstore.usage(cfg, refresh=True)
    return {
        "ok": not failures,
        "scope": "bucket",
        "dryRun": False,
        "bucket": cfg.bucket,
        "backup": backup,
        "freedMb": _mb(bytes_deleted),
        "deletedObjects": deleted,
        "abortedUploads": aborted,
        "failures": failures[:50],
        "usageBeforeBytes": int(result.get("_usageBefore") or 0),
        "usageBeforeMb": _mb(result.get("_usageBefore") or 0),
        "usageAfterBytes": int(after.get("bytes") or 0),
        "usageAfterMb": _mb(after.get("bytes") or 0),
        "objectsAfter": int(after.get("objects") or 0),
    }
