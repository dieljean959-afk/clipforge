"""Free local disk by moving FINISHED batches' media into the bucket.

Why this exists: the sandbox has a 4 GB disk and about 1.6 GB of it is the
virtualenv plus node_modules. Durable storage was wired as an archive — an
uploaded source was pushed and its local copy dropped, but a URL source stayed
on disk for the life of the batch, and serving a clip whose local copy was gone
pulled it back onto the disk. So the 5 GB bucket could never relieve the 4 GB
disk: every read refilled what a cleanup had freed.

This module is the missing direction. It walks the *terminal* batches (never a
running one — those files are in use), pushes each media file under the
bucket's own key, and only after a confirmed upload drops the local copy. Every
dropped file stays reachable two ways: `app/objectstore.open_stream` serves
clip video straight from the bucket without touching the disk, and
`store.find_source_file` / `_restore_media` pull a source back for a re-render.

Nothing here can lose an upload: `drop_local_after_push` refuses unless the
push really succeeded and offload is configured with delete-local on.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

#: Files worth moving. Anything else in a job's folders (a lock, a partial
#: download, a probe artefact) is left exactly where it is.
MEDIA_SUFFIXES = {
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v",
    ".jpg", ".jpeg", ".png", ".webp",
    ".srt", ".vtt", ".txt", ".json",
}

#: A transfer that keeps dying on one file must not hold the whole cleanup.
SKIP_MARKERS = (".uploading", ".fetching", ".tmp", ".part")

_MB = 1024 * 1024


def _free_bytes(paths: Any) -> int | None:
    try:
        return shutil.disk_usage(Path(paths.data_dir)).free
    except OSError:
        return None


def _job_dirs(paths: Any, job_id: str) -> list[Path]:
    """The three folders a job's media can live in (missing ones are fine)."""
    from . import store as store_mod

    try:
        source_dirs = tuple(store_mod._job_source_dirs(paths, job_id))
    except Exception:
        source_dirs = ()
    return [Path(paths.outputs_dir) / job_id, *source_dirs]


def _wanted(p: Path) -> bool:
    if p.suffix.lower() not in MEDIA_SUFFIXES:
        return False
    if any(p.name.endswith(m) for m in SKIP_MARKERS):
        return False
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def key_for(paths: Any, job_id: str, p: Path) -> str | None:
    """The bucket key a local file already has, or the one it should get.

    Outputs round-trip through `objectstore.key_for_path`, which is the same
    mapping the serving routes use — so a file pushed here is found again by
    name. Sources use `objectstore.upload_key`, which is what
    `store._materialize_source` pulls back.
    """
    from . import objectstore

    outputs = Path(getattr(paths, "outputs_dir", None))
    if outputs == p.parent.parent:
        return objectstore.key_for_path(p, outputs)
    for d in _job_dirs(paths, job_id):
        try:
            p.resolve().relative_to(d.resolve())
        except (ValueError, OSError):
            continue
        return objectstore.upload_key(job_id, p.name)
    return None


def reclaim(
    store: Any,
    paths: Any,
    *,
    target_free_bytes: int = 0,
    job_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Push finished batches' media to the bucket and drop the local copies.

    `target_free_bytes` stops the walk once the disk is comfortable; 0 means
    "offload everything finished" (what the panel's button asks for). Returns a
    dict the endpoint can hand straight to the UI, with a `reason` whenever it
    could not do anything — never a silent no-op.
    """
    from . import jobstatus, objectstore

    result: dict[str, Any] = {
        "ok": False,
        "freedBytes": 0,
        "freedMb": 0.0,
        "uploaded": 0,
        "scanned": 0,
        "jobs": 0,
        "freeBytes": _free_bytes(paths),
        "reason": "",
    }
    cfg = objectstore.config_from_env()
    if not cfg.enabled:
        result["reason"] = "media offload is off — no bucket configured"
        return result
    if not cfg.delete_local:
        result["reason"] = (
            "CLIPFORGE_OFFLOAD_DELETE_LOCAL=0 keeps every local copy, so there is "
            "nothing to reclaim; set it to 1 to let finished media live only in "
            "the bucket"
        )
        return result
    s3 = objectstore.client(cfg)
    if s3 is None:
        result["reason"] = "media offload is configured but the S3 client is not ready"
        return result

    wanted_ids = set(job_ids) if job_ids else None
    free = result["freeBytes"]
    for job in store.list_jobs():
        if target_free_bytes and isinstance(free, int) and free >= target_free_bytes:
            break
        jid = str(job.get("_id") or "")
        if not jid or (wanted_ids is not None and jid not in wanted_ids):
            continue
        if job.get("deleting") or not jobstatus.is_terminal(job):
            # A running batch is writing those files right now; deleting them
            # mid-render is indistinguishable from a corrupt disk.
            continue
        result["jobs"] += 1
        keep_source = bool(job.get("keepSource"))
        source_prefixes = tuple(
            str(d) for d in _job_dirs(paths, jid)[1:]
        )
        for d in _job_dirs(paths, jid):
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if not _wanted(p):
                    continue
                if keep_source and any(str(p).startswith(pre) for pre in source_prefixes):
                    continue
                result["scanned"] += 1
                key = key_for(paths, jid, p)
                if not key:
                    continue
                size = p.stat().st_size
                if not objectstore.upload_file(p, key, cfg):
                    continue
                if not objectstore.drop_local_after_push(True, p, cfg):
                    continue
                result["uploaded"] += 1
                result["freedBytes"] += size
                free = _free_bytes(paths)
                if target_free_bytes and isinstance(free, int) and free >= target_free_bytes:
                    break
    result["freeBytes"] = _free_bytes(paths)
    result["freedMb"] = round(result["freedBytes"] / _MB, 1)
    result["ok"] = True
    if not result["uploaded"]:
        result["reason"] = "nothing to reclaim — no finished batch keeps media on local disk"
    log.info(
        f"[reclaim] offloaded {result['uploaded']} file(s), freed "
        f"{result['freedMb']} MB from {result['jobs']} finished batch(es)"
    )
    return result
