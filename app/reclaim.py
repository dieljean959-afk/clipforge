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
import threading
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

#: Which job stages may be moved into the bucket. Deliberately NOT
#: `jobstatus.is_terminal`: that set includes `deleting` (being removed) and
#: `paused` (resumable, and a resume reads the source from disk).
RECLAIMABLE_STAGES = frozenset({"complete", "failed"})


def _all_jobs(store: Any) -> list[dict[str, Any]]:
    """Every job document, whatever the store's signature.

    `LocalStore.list_jobs()` takes NO arguments while the remote stores take a
    limit. Calling the wrong one raised TypeError, every caller below caught
    it and carried on — which quietly turned retention and the bucket budget
    into no-ops, and turned `sweep_startup`'s job map into `{}`. With no job
    list every work folder looks orphaned, so a boot sweep deleted the scratch
    of jobs that were running: a 20-minute transcribe died with
    "source file not found: .../transcribe/audio.m4a" because the API restarted
    under it. Raises if the store cannot be read at all — the callers below
    treat that as "delete nothing", never as "delete everything".
    """
    try:
        return list(store.list_jobs())
    except TypeError:
        return list(store.list_jobs(limit=1000))

#: Free space below which the watchdog offloads finished batches on its own.
#: The submit gate wants 100 MB and the worker's preflight wants 400 MB, so this
#: sits just above the worker floor: reclaim while there is still room to work,
#: not after a render has already failed for lack of it.
LOW_DISK_BYTES = 480 * _MB
#: How far a watchdog pass frees: comfortably above the floor, so one pass lasts
#: several batches instead of firing on every tick.
TARGET_FREE_BYTES = 1200 * _MB
#: Seconds between watchdog passes. Slow on purpose — this is a safety net under
#: a manual button, not a hot loop.
WATCH_INTERVAL = 60.0


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
        # Only a settled batch may be moved. A running one is writing those
        # files right now; `deleting` is on its way out and must not be raced,
        # and `paused` can be resumed (a resume expects its source on disk).
        status = jobstatus.status_of(job)
        if status not in RECLAIMABLE_STAGES or job.get("deleting"):
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


# --------------------------------------------------------------------------- #
# the moment a batch finishes
# --------------------------------------------------------------------------- #

_offload_lock = threading.Lock()


def offload_job(store: Any, paths: Any, job_id: str) -> None:
    """Move one just-finished batch into the bucket and free its local bytes.

    Called from the pipeline's completion path, not from a request. When the
    disk is already tight it runs inline — a batch that just finished is holding
    several hundred MB it no longer needs, and waiting for the watchdog (up to a
    minute) or for the user to press a button is how the NEXT batch fails for
    lack of space. With room to spare it runs in the background so completion
    is never held up by an upload.
    """
    free = _free_bytes(paths)
    tight = isinstance(free, int) and free < LOW_DISK_BYTES

    def run() -> None:
        with _offload_lock:
            try:
                res = reclaim(store, paths, job_ids=[job_id])
                if res.get("uploaded"):
                    logger.info(
                        f"[reclaim] job {job_id}: {res['uploaded']} file(s) "
                        f"({res['freedMb']} MB) moved to the bucket after completion"
                    )
                # The source is only needed while the batch runs: once every clip
                # is in the bucket, its copy is dead weight in a 5 GB plan. The
                # rules keep it while a batch is running/paused so Retry and Run
                # again still work, and a later re-render re-fetches from the
                # URL (telling the user first) when it is gone.
                job = store.get_job(job_id) or {}
                if job.get("status") == "complete":
                    dropped = delete_source_objects(job_id)
                    if dropped:
                        logger.info(
                            f"[reclaim] job {job_id}: {dropped} source object(s) "
                            "deleted now that the batch is finished"
                        )
            except Exception as exc:  # noqa: BLE001 - never fail a finished job
                logger.warning(f"[reclaim] post-completion offload for {job_id} failed: {exc}")

    if tight:
        run()
        return
    threading.Thread(target=run, name="offload-finished", daemon=True).start()


# --------------------------------------------------------------------------- #
# watchdog: make the bucket the default, not a button nobody presses
# --------------------------------------------------------------------------- #

_thread: threading.Thread | None = None
_stop = threading.Event()


def tick(store: Any, paths: Any, low_mb: int = LOW_DISK_BYTES // _MB) -> dict[str, Any] | None:
    """One watchdog pass: reclaim only if the disk is actually tight.

    Returns the reclaim result, or None when there was room and nothing was
    attempted. Separate from the loop so it can be tested without threads.
    """
    free = _free_bytes(paths)
    if not isinstance(free, int) or free >= low_mb * _MB:
        return None
    out = reclaim(
        store,
        paths,
        target_free_bytes=max(TARGET_FREE_BYTES, low_mb * _MB * 2),
    )
    # A running batch cannot be reclaimed, so "low and nothing to do" is the
    # normal state while a render is in flight — say it once per pass at info
    # level instead of warning every minute about a situation that resolves
    # itself when the batch finishes (see `offload_job`).
    if out.get("uploaded"):
        logger.warning(
            f"[reclaim] only {round(free / _MB)} MB free — moved "
            f"{out['freedMb']} MB of finished batches into the bucket"
        )
    else:
        logger.info(
            f"[reclaim] only {round(free / _MB)} MB free, but nothing finished "
            "to reclaim yet — the running batch still needs its source on disk"
        )
    return out


def _loop(store: Any, paths: Any, interval: float, low_mb: int) -> None:
    while not _stop.is_set():
        try:
            tick(store, paths, low_mb)
        except Exception as exc:  # noqa: BLE001 - the loop must not die
            logger.warning(f"[reclaim] watchdog pass failed: {exc}")
        try:
            housekeeping(store, paths)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[reclaim] housekeeping pass failed: {exc}")
        if _stop.wait(interval):
            break


def start_watchdog(
    store: Any,
    paths: Any,
    *,
    interval: float = WATCH_INTERVAL,
    low_mb: int = LOW_DISK_BYTES // _MB,
) -> threading.Thread:
    """Keep the 4 GB disk useful without anyone pressing anything.

    Runs in the API process like the purge loop: off every request path, so a
    slow PUT to the bucket can never become a proxy timeout on a page load.
    Does nothing at all while there is free space.
    """
    global _thread
    if _thread is not None and _thread.is_alive():
        return _thread
    _stop.clear()
    _thread = threading.Thread(
        target=_loop,
        args=(store, paths, interval, low_mb),
        name="media-reclaim",
        daemon=True,
    )
    _thread.start()
    return _thread


def stop_watchdog() -> None:
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=5.0)


# --------------------------------------------------------------------------- #
# the bucket as the library: verified pushes, retention, budget, crash sweep
# --------------------------------------------------------------------------- #

_GB = 1024**3


def push_verified(path: Path, key: str) -> dict[str, Any]:
    """Upload one file and require a HEAD with the exact size before trusting it.

    The rule the whole offload rests on: nothing local is deleted, and nothing
    is reported as stored, until the bucket confirms the byte count. A PUT that
    was accepted but truncated (common on a flaky link) is caught here.
    """
    from . import objectstore

    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return {"ok": False, "key": key, "error": "missing locally"}
    if size <= 0:
        return {"ok": False, "key": key, "error": "empty file"}
    if not objectstore.upload_file(p, key):
        return {"ok": False, "key": key, "error": "upload failed"}
    if not objectstore.verify(key, size):
        return {"ok": False, "key": key, "error": "HEAD size mismatch after upload"}
    return {"ok": True, "key": key, "bytes": size}


def push_clip_and_thumb(job_id: str, clip_path: Path, thumb_path: Path | None) -> dict[str, Any]:
    """Verified push for one finished clip (+ its thumbnail), then drop local.

    Returns the clip's keys and whether the local copy was removed. On ANY
    failure the local files stay and the caller marks the clip \"upload
    pending\" and retries with backoff — a finished clip is never lost.
    """
    from . import objectstore

    video = push_verified(clip_path, objectstore.output_key(job_id, clip_path.name))
    if not video["ok"]:
        return {"ok": False, "video": video, "thumb": None, "dropped": False}
    thumb_res = None
    if thumb_path is not None and Path(thumb_path).exists():
        thumb_res = push_verified(thumb_path, objectstore.output_key(job_id, Path(thumb_path).name))
    dropped = False
    if video["ok"]:
        cfg = objectstore.config_from_env()
        if objectstore.drop_local_after_push(True, clip_path, cfg):
            dropped = True
        if thumb_res and thumb_res.get("ok"):
            objectstore.drop_local_after_push(True, thumb_path, cfg)
    return {"ok": True, "video": video, "thumb": thumb_res, "dropped": dropped}


def delete_source_objects(job_id: str) -> int:
    """Drop a finished batch's source from the bucket (its clips stay)."""
    from . import objectstore

    cfg = objectstore.config_from_env()
    if not cfg.enabled:
        return 0
    return objectstore.delete_prefix(f"{cfg.prefix}/uploads/{job_id}/", cfg)


def delete_job_objects(job_id: str) -> int:
    """Remove every bucket object belonging to one batch (source + outputs)."""
    from . import objectstore

    cfg = objectstore.config_from_env()
    if not cfg.enabled:
        return 0
    removed = 0
    for prefix in (
        f"{cfg.prefix}/uploads/{job_id}/",
        f"{cfg.prefix}/outputs/{job_id}/",
    ):
        removed += objectstore.delete_prefix(prefix, cfg)
    return removed


def retention_countdown(job: dict[str, Any], retention_hours: int) -> int | None:
    """Hours until this finished batch expires, or None when it never will."""
    if retention_hours <= 0 or job.get("status") not in RECLAIMABLE_STAGES:
        return None
    done_ms = int(job.get("completedAt") or 0)
    if not done_ms:
        return None
    import time as _time

    left = done_ms / 1000.0 + retention_hours * 3600 - _time.time()
    return max(0, int(left // 3600))


def sweep_expired(store: Any, paths: Any, retention_hours: int) -> dict[str, Any]:
    """Delete batches finished longer than the retention window.

    Bucket first, local second, and only for COMPLETE/FAILED batches — a
    running, paused or deleting job keeps everything it is using.
    """
    import time as _time

    out = {"expired": [], "objectsRemoved": 0, "jobsRemoved": 0}
    if retention_hours <= 0:
        return out
    cutoff = _time.time() - retention_hours * 3600
    try:
        jobs = _all_jobs(store)
    except Exception:  # noqa: BLE001
        return out
    for job in jobs:
        if job.get("status") not in RECLAIMABLE_STAGES:
            continue
        done_ms = int(job.get("completedAt") or 0)
        if not done_ms or done_ms / 1000.0 > cutoff:
            continue
        job_id = str(job.get("_id") or "")
        if not job_id:
            continue
        removed = delete_job_objects(job_id)
        for d in _job_dirs(paths, job_id):
            shutil.rmtree(d, ignore_errors=True)
        try:
            store.delete_job(job_id)
            out["jobsRemoved"] += 1
        except Exception:  # noqa: BLE001
            pass
        out["expired"].append(job_id)
        out["objectsRemoved"] += removed
        logger.info(
            f"[retention] job {job_id} finished more than {retention_hours}h ago — "
            f"removed {removed} object(s) and its local media"
        )
    return out


def refresh_status() -> dict[str, Any]:
    """Bucket usage as GB/percent of the plan, for every status payload."""
    from . import objectstore

    use = objectstore.usage()
    budget = int(use.get("budgetBytes") or 0)
    return {
        "usageBytes": use["bytes"],
        "usageGb": round(use["bytes"] / _GB, 2),
        "usageObjects": use["objects"],
        "budgetGb": round(budget / _GB, 2),
        "usedPercent": round(100.0 * use["bytes"] / budget, 1) if budget else 0.0,
    }


def enforce_budget(
    store: Any, paths: Any, *, warn_at: float = 0.8, prune_at: float = 0.9,
    retention_hours: int = 48,
) -> dict[str, Any]:
    """Keep the 5 GB plan from becoming a failed render.

    * >=80%: say so and run the expiry sweep early (that is free space the user
      already agreed to give up).
    * >=90%: delete the OLDEST finished batches until back under the line —
      never a job that is queued, running, paused or deleting.
    """
    from . import objectstore

    out = {"usedPercent": 0.0, "warned": False, "pruned": [], "swept": None}
    use = objectstore.usage()
    budget = int(use.get("budgetBytes") or 0)
    if not budget:
        return out
    pct = 100.0 * use["bytes"] / budget
    out["usedPercent"] = round(pct, 1)
    if pct < warn_at * 100:
        return out
    out["warned"] = True
    logger.warning(
        f"[budget] bucket is {out['usedPercent']}% of the {round(budget / _GB, 2)} GB plan "
        f"({round(use['bytes'] / _GB, 2)} GB used)"
    )
    swept = sweep_expired(store, paths, retention_hours)
    out["swept"] = swept
    if swept["expired"]:
        objectstore.invalidate_usage()
        use = objectstore.usage(refresh=True)
        pct = 100.0 * use["bytes"] / budget
        out["usedPercent"] = round(pct, 1)
    if pct < prune_at * 100:
        return out
    finished = [
        j for j in _all_jobs(store)
        if j.get("status") in RECLAIMABLE_STAGES and j.get("completedAt")
    ]
    finished.sort(key=lambda j: int(j.get("completedAt") or 0))
    for job in finished:
        if pct < prune_at * 100:
            break
        job_id = str(job.get("_id") or "")
        removed = delete_job_objects(job_id)
        for d in _job_dirs(paths, job_id):
            shutil.rmtree(d, ignore_errors=True)
        try:
            store.delete_job(job_id)
        except Exception:  # noqa: BLE001
            pass
        out["pruned"].append(job_id)
        logger.warning(
            f"[budget] over 90% — removed oldest finished batch {job_id} "
            f"({removed} object(s)); running jobs untouched"
        )
        use = objectstore.usage(refresh=True)
        pct = 100.0 * use["bytes"] / budget
        out["usedPercent"] = round(pct, 1)
    return out


HOUSEKEEPING_INTERVAL = 300.0
_last_housekeeping = 0.0


def housekeeping(
    store: Any, paths: Any, *, retention_hours: int | None = None, force: bool = False,
) -> dict[str, Any]:
    """The periodic "keep the plan tidy" pass: retention + 80/90% budget.

    Runs off the watchdog thread every few minutes, because a rule that only
    fires inside a request is a rule that never fires: with no job starting,
    nothing was ever expiring and the 5 GB plan would fill up silently.

    It refuses to do anything unless the bucket answers a listing — deleting a
    batch because it is "expired" while the bucket is unreachable is how media
    disappears without ever being stored anywhere.
    """
    global _last_housekeeping
    import time as _time

    from . import objectstore

    now = _time.time()
    if not force and now - _last_housekeeping < HOUSEKEEPING_INTERVAL:
        return {"skipped": "not due"}
    if not objectstore.configured():
        return {"skipped": "no bucket"}
    if not objectstore.reachable():
        return {"skipped": "bucket unreachable"}
    _last_housekeeping = now
    if retention_hours is None:
        try:
            from .config import load_config

            retention_hours = int(load_config().hosted.retention_hours)
        except Exception:  # noqa: BLE001
            retention_hours = 48
    swept = sweep_expired(store, paths, retention_hours)
    budget = enforce_budget(store, paths, retention_hours=retention_hours)
    # The scheduled half of the disk story: once a day, the SAME safe routine
    # the panel's button calls removes scratch, caches, old logs and the
    # bucket-backed model copies. This container has no cron or systemd, so the
    # watchdog thread is the scheduler.
    try:
        from . import disksweep

        nightly = disksweep.nightly(store, paths)
    except Exception as exc:  # noqa: BLE001 - housekeeping must never raise
        logger.warning(f"[sweep] nightly cleanup skipped: {exc}")
        nightly = {"skipped": str(exc)[:200]}
    if swept["jobsRemoved"] or budget["pruned"]:
        objectstore.invalidate_usage()
    return {
        "retentionHours": retention_hours,
        "expired": swept["expired"],
        "jobsRemoved": swept["jobsRemoved"],
        "usedPercent": budget["usedPercent"],
        "pruned": budget["pruned"],
        "diskSweep": nightly,
    }


def sweep_startup(store: Any, paths: Any) -> dict[str, Any]:
    """Clean what a crash or a kill -9 left behind, before any job sees the disk.

    Two kinds of leftovers: scratch folders whose job no longer exists (or is
    already finished), and half-written temp files (`.fetching`, `.part`,
    `.uploading`). Bucket objects are never dropped here — a push is verified
    before its local copy is removed, so anything still local is the only copy.
    """
    out = {"dirsRemoved": 0, "filesRemoved": 0, "bytesFreed": 0}
    try:
        jobs = {str(j.get("_id")): j for j in _all_jobs(store)}
    except Exception as exc:  # noqa: BLE001 - unreadable store must not become deletion
        logger.warning(
            f"[sweep] cannot read the job list ({exc}) — sweeping nothing: every "
            "work folder would look orphaned, including a running job's"
        )
        return out
    roots = [
        Path(paths.outputs_dir),
        Path(paths.data_dir) / "work",
        Path(paths.data_dir) / "uploads",
        Path(paths.data_dir) / "downloads",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            try:
                if not child.is_dir():
                    continue
                job = jobs.get(child.name)
                finished = job is None or job.get("status") in RECLAIMABLE_STAGES
                if not finished:
                    continue
                size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
                shutil.rmtree(child, ignore_errors=True)
                out["dirsRemoved"] += 1
                out["bytesFreed"] += size
            except OSError:
                continue
    # A RUNNING job writes `.tmp` files on purpose: the ASR checkpoints each
    # window through `win_000NN.json.tmp` and then renames it, so sweeping
    # every `.tmp` under data/ pulls the file out from under a live decoder
    # (FileNotFoundError mid-transcribe). Only temp files in folders that do
    # not belong to a live job are debris.
    live = {
        str(j.get("_id")) for j in jobs.values()
        if str(j.get("status")) not in RECLAIMABLE_STAGES
    }
    for pattern in ("**/*.fetching", "**/*.part", "**/*.uploading", "**/*.tmp"):
        for stale in list(Path(paths.data_dir).glob(pattern)):
            try:
                owner = next(
                    (p.name for p in stale.parents if p.name in live), None
                )
                if owner is not None:
                    continue
                out["bytesFreed"] += stale.stat().st_size
                stale.unlink(missing_ok=True)
                out["filesRemoved"] += 1
            except OSError:
                continue
    if out["dirsRemoved"] or out["filesRemoved"]:
        logger.info(
            f"[sweep] crash leftovers removed: {out['dirsRemoved']} folder(s), "
            f"{out['filesRemoved']} temp file(s), {round(out['bytesFreed'] / _MB, 1)} MB"
        )
    return out


def unused_batches(store: Any) -> dict[str, Any]:
    """Bucket objects whose batch no longer exists, grouped per batch.

    A batch whose job document is gone cannot appear in the dashboard, so its
    media is invisible there while still counting against the plan. That is the
    only thing this returns: `models/` and other non-batch prefixes are never
    considered, and a batch that is still live is never listed.

    Reading the job list is load-bearing. If the store throws, the answer must
    be "cannot tell" rather than "everything is unused" — returning an empty
    list is the safe failure, because the delete path re-checks anyway.
    """
    from . import objectstore

    cfg = objectstore.config_from_env()
    if not cfg.enabled:
        return {"items": [], "totalBytes": 0, "objects": 0, "enabled": False}
    try:
        live = {str(j.get("_id")) for j in _all_jobs(store)}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[reclaim] unused scan: cannot read jobs ({exc})")
        return {"items": [], "totalBytes": 0, "objects": 0, "enabled": True}
    groups: dict[str, dict[str, Any]] = {}
    for obj in objectstore.list_prefix():
        key = str(obj.get("key") or "")
        parts = key.split("/")
        if len(parts) < 4 or parts[1] not in {"uploads", "outputs"}:
            continue
        job_id = parts[2]
        if not job_id or job_id in live:
            continue
        g = groups.setdefault(job_id, {
            "jobId": job_id, "bytes": 0, "objects": 0, "kinds": set(),
        })
        g["bytes"] += int(obj.get("size") or 0)
        g["objects"] += 1
        g["kinds"].add(parts[1])
    items = sorted(groups.values(), key=lambda g: -g["bytes"])
    for g in items:
        g["kinds"] = sorted(g["kinds"])
    return {
        "items": items,
        "totalBytes": sum(g["bytes"] for g in items),
        "objects": sum(g["objects"] for g in items),
        "enabled": True,
    }


def delete_unused_batches(store: Any, job_ids: list[str] | None = None) -> dict[str, Any]:
    """Delete bucket media for batches that are provably not live any more.

    `job_ids=None` means "every unused batch right now". Whatever the caller
    asked for, the live set is read again HERE and every id is filtered through
    it, so a list rendered a minute ago cannot delete a batch that has since
    been re-queued. Returns what was removed and what was skipped, with bytes,
    so the caller can say exactly how much plan was reclaimed.
    """
    from . import objectstore

    cfg = objectstore.config_from_env()
    if not cfg.enabled:
        return {"deleted": [], "skipped": [], "removed": 0, "bytesFreed": 0}
    try:
        live = {str(j.get("_id")) for j in _all_jobs(store)}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[reclaim] unused delete: cannot read jobs ({exc})")
        return {"deleted": [], "skipped": sorted(job_ids or []), "removed": 0, "bytesFreed": 0}

    wanted = sorted(set(job_ids)) if job_ids else None
    scan = unused_batches(store)
    known = {str(g["jobId"]): g for g in scan["items"]}
    targets = wanted if wanted is not None else sorted(known)
    deleted: list[str] = []
    skipped: list[str] = []
    removed = 0
    bytes_freed = 0
    for job_id in targets:
        if job_id in live:
            skipped.append(job_id)
            continue
        group = known.get(job_id)
        if group is None:
            skipped.append(job_id)
            continue
        try:
            gone = delete_job_objects(job_id)
        except Exception as exc:  # noqa: BLE001 - one batch must not stop the rest
            logger.warning(f"[reclaim] unused delete {job_id} failed: {exc}")
            skipped.append(job_id)
            continue
        if gone:
            deleted.append(job_id)
            removed += gone
            bytes_freed += int(group["bytes"])
        else:
            skipped.append(job_id)
    if deleted:
        objectstore.invalidate_usage()
        logger.info(
            f"[reclaim] removed {removed} unused object(s) from {len(deleted)} "
            f"dead batch(es), {round(bytes_freed / _MB, 1)} MB"
        )
    return {
        "deleted": deleted,
        "skipped": skipped,
        "removed": removed,
        "bytesFreed": bytes_freed,
    }


def reconcile(store: Any) -> dict[str, Any]:
    """Bucket vs DB: objects whose batch no longer exists, and batches with none."""
    from . import objectstore

    live = {str(j.get("_id")): j for j in _all_jobs(store)}
    orphans: list[dict[str, Any]] = []
    by_job: dict[str, int] = {}
    for obj in objectstore.list_prefix():
        key = str(obj.get("key") or "")
        parts = key.split("/")
        job_id = parts[2] if len(parts) > 2 and parts[1] in {"uploads", "outputs"} else ""
        if not job_id:
            continue
        by_job[job_id] = by_job.get(job_id, 0) + obj["size"]
        if job_id not in live:
            orphans.append({"key": key, "size": obj["size"]})
    empty = [
        jid for jid in live
        if jid not in by_job and live[jid].get("status") == "complete"
    ]
    return {
        "bucketObjects": sum(by_job.values()),
        "orphanObjects": len(orphans),
        "orphanBytes": sum(o["size"] for o in orphans),
        "orphans": orphans[:200],
        "completedBatchesWithNoObjects": empty,
    }
