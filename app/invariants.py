"""Job/clip invariants: derive the batch from its clips, never the reverse.

The bug this exists for
-----------------------
A batch finished 15 renders, its export stage then failed a disk self-check, and
the job document was written back as ``clips: [], progress: 0, isTerminal:
false, errorKind: fail_fast``. The clips were fine and in the bucket; the record
was wrong, so the dashboard showed an empty batch with no way to act on it and
the job was not resumable either.

Two rules follow from that:

1. **Clips are the source of truth for what a batch contains.** A job's
   ``clips`` list, ``clipCount`` and ``progress`` are a projection of the clip
   documents on disk. When they disagree, the clip documents win and the job is
   repaired.
2. **A terminal verdict is never invented by accounting.** If clips rendered and
   verified, the batch is at least partially successful and must be shown as
   such ("completed with 2 failed clips"), never as a bare failure.

The checker runs at boot (a boot phase) and every 5 minutes on the worker's
housekeeping tick. It only writes when it finds a real inconsistency, so it is
cheap and idempotent.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from loguru import logger

from app.jobstatus import _RUNNING_ONLY_FIELDS

#: renderState values that mean "this clip has a playable master".
RENDERED_STATES = frozenset({"done", "complete", "rendered", "uploaded"})
#: renderState values that mean "this clip will not render without a retry".
FAILED_STATES = frozenset({"failed", "error"})

_REPAIR_LOCK = threading.Lock()
_LAST_RUN: dict[str, Any] = {"at": 0.0, "report": None}


def _clips_for(store: Any, job: dict[str, Any]) -> list[dict[str, Any]]:
    """Clip documents for a job: the doc store first, the job's projection second."""
    jid = str(job.get("_id") or "")
    docs: list[dict[str, Any]] = []
    try:
        docs = [d for d in store.list_clips(jid) if isinstance(d, dict)]
    except Exception as exc:  # noqa: BLE001 - a broken doc store must not stop the check
        logger.warning(f"[invariants] could not list clips for {jid}: {exc}")
    if docs:
        return docs
    # Fall back to the projected list: for old jobs the documents were never
    # written, but the projection still names the files and their state.
    projected = job.get("clips")
    if isinstance(projected, list):
        return [c for c in projected if isinstance(c, dict)]
    return []


def _clip_state(clip: dict[str, Any]) -> str:
    state = str(
        clip.get("renderState")
        or clip.get("status")
        or clip.get("state")
        or ""
    ).lower()
    if state:
        return state
    # No explicit state: infer from the media, which is what the UI does.
    if clip.get("videoPath") or clip.get("videoKey") or clip.get("mediaKey"):
        return "done"
    return "pending"


#: A render is treated as live only while its last real progress is this
#: recent. The pipeline's own stall watchdog fires at 120 s; anything older is
#: a dead process, not a slow encode.
LIVE_RENDER_SEC = 180.0


def _is_live_render(job: dict[str, Any]) -> bool:
    """Is a process actually working on this job right now?

    Progress timestamps only: a stage string is a claim, and the claim is what
    was wrong. An old job with no stamps is dead and may be rebuilt.
    """
    stage = str(job.get("stage") or "")
    if stage not in {"rendering", "export", "exporting", "reframe", "captions"}:
        return False
    now = time.time() * 1000
    stamps = [
        job.get("lastProgressAt"),
        job.get("heartbeatAt"),
        job.get("unitsChangedAt"),
    ]
    fresh = [
        float(s) for s in stamps
        if isinstance(s, (int, float)) and s > 0 and (now - float(s)) < LIVE_RENDER_SEC * 1000
    ]
    return bool(fresh)


def _clip_rank(clip: dict[str, Any]) -> int:
    """How complete a clip document is: a verified remote render wins.

    Same ranking the store uses when two documents claim one order. It is
    repeated here so a repair picks the deliverable, not the older twin that
    still has a local path but no bucket key.
    """
    return (
        (4 if _clip_state(clip) == "done" else 0)
        + (2 if clip.get("videoKey") else 0)
        + (1 if clip.get("videoPath") else 0)
    )


def restore_clip_docs(store: Any, job: dict[str, Any]) -> int:
    """Re-materialise clip documents from the job's clip projection.

    The projection on the job document is a full clip record (media size,
    geometry, share id, render state). When a re-detection pass or a crash
    deletes the documents but the media is safely in the bucket, the projection
    is the only surviving copy of "what this batch contains" — and losing it
    makes 15 delivered clips look like an empty batch. Writing the documents
    back is a metadata-only operation: no fetch, no transcription, no render.

    Returns the number of documents restored.
    """
    restored = 0
    projected = job.get("clips")
    if not isinstance(projected, list):
        return 0
    for clip in projected:
        if not isinstance(clip, dict):
            continue
        cid = str(clip.get("_id") or "")
        if not cid:
            continue
        if store.clip_path(cid).exists():
            continue
        doc = dict(clip)
        doc.setdefault("jobId", job.get("_id"))
        # A projected record with media is a finished clip, even if the old
        # document never wrote an explicit render state.
        if not doc.get("renderState"):
            doc["renderState"] = "done" if doc.get("videoPath") else "pending"
        store.clip_path(cid).parent.mkdir(parents=True, exist_ok=True)
        tmp = store.clip_path(cid).with_name(
            f"{cid}.json.{__import__('os').getpid()}.restore.tmp"
        )
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(store.clip_path(cid))
        restored += 1
    return restored


def _counts(clips: list[dict[str, Any]]) -> dict[str, int]:
    done = failed = pending = 0
    for clip in clips:
        state = _clip_state(clip)
        if state in RENDERED_STATES:
            done += 1
        elif state in FAILED_STATES:
            failed += 1
        else:
            pending += 1
    return {"done": done, "failed": failed, "pending": pending, "total": len(clips)}


def partial_success_label(counts: dict[str, int]) -> str:
    """"15 clips ready" / "14 clips ready, 1 failed" — the honest summary."""
    done, failed = counts["done"], counts["failed"]
    if failed and done:
        return f"{done} clip{'s' if done != 1 else ''} ready, {failed} failed"
    if failed and not done:
        return f"{failed} clip{'s' if failed != 1 else ''} failed"
    return f"{done} clip{'s' if done != 1 else ''} ready"


def _projection_stale(job: dict[str, Any], clips: list[dict[str, Any]]) -> bool:
    """Does the job's embedded clip list still match the clip documents?

    The projection is a cache the dashboard and export read; after a single
    clip is re-rendered (caption cleanup, fps repair) the document changes but
    the old projection can keep the stale captionCues/mediaBytes forever. That
    is exactly how the fixed Burpees clip could still show [INAUDIBLE] in one
    view and clean captions in another.
    """
    projected = job.get("clips")
    if not isinstance(projected, list) or len(projected) != len(clips):
        return True
    by_id = {str(c.get("_id") or ""): c for c in projected if isinstance(c, dict)}
    for doc in clips:
        cid = str(doc.get("_id") or "")
        old = by_id.get(cid)
        if old is None:
            return True
        for key in (
            "order", "renderState", "videoPath", "thumbPath", "mediaBytes",
            "mediaDurationSec", "fps", "mediaCodec", "captionCues",
        ):
            if old.get(key) != doc.get(key):
                return True
    return False


def _needs_repair(
    job: dict[str, Any], counts: dict[str, int], clips: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Which invariants this document breaks (empty list = consistent)."""
    if counts["total"] == 0:
        return []
    broken: list[str] = []
    clips_field = job.get("clips")
    if not isinstance(clips_field, list) or len(clips_field) < counts["total"]:
        broken.append("clips-list-shorter-than-clip-docs")
    elif str(job.get("stage") or "") in ("complete", "failed", "cancelled"):
        if clips and _projection_stale(job, clips):
            broken.append("clips-projection-stale")
    if counts["done"] and int(job.get("clipCount") or 0) < counts["done"]:
        broken.append("clipCount-below-rendered")
    stage = str(job.get("stage") or "")
    if counts["done"] and stage in ("failed", "grabbing", "", "fetch", "fetching"):
        broken.append("failed-or-unstarted-stage-with-rendered-clips")
    # A batch cannot be complete while candidate clips have no terminal state.
    # The old repair wrote stage=complete whenever ANY clip was done; a crash
    # after clip 1 then hid the other 20 pending candidates behind a green
    # "1 clip ready" card and no way to continue.
    if counts["pending"] > 0 and stage == "complete":
        broken.append("complete-with-pending-clips")
    if counts["done"] and not job.get("isTerminal"):
        broken.append("non-terminal-with-rendered-clips")
    # A terminal batch whose finished clips are already durable in the bucket is
    # settled, not stale: its rendered deliverables live remotely and should not be
    # used as a reason to re-open the batch for resume/re-render.
    finished_remote = any(
        isinstance(c, dict) and str(c.get("renderState") or "").lower() == "done"
        and bool(c.get("videoKey")) for c in (clips or [])
    )
    if not (stage in ("failed", "complete", "cancelled") and not job.get("retryable") and finished_remote):
        pass
    if counts["done"] and not job.get("progress"):
        broken.append("zero-progress-with-rendered-clips")
    if counts["done"] and counts["pending"] == 0 and str(job.get("errorKind") or "") == "fail_fast":
        broken.append("fail-fast-on-a-delivered-batch")
    # A partial-success warning that outlived the retry is a lie in the UI:
    # after the failed clip is repaired the batch is simply complete.
    if job.get("warning") and counts["failed"] == 0:
        broken.append("stale-partial-success-warning")
    # A terminal batch without a finish stamp reads as "still running" to every
    # consumer that sorts or times batches by when they ended.
    if stage in ("complete", "failed", "cancelled") and not job.get("finishedAt"):
        broken.append("terminal-without-finishedAt")
    return broken


def repair_job(
    store: Any,
    job: dict[str, Any],
    counts: dict[str, int],
    *,
    reason: str,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Rebuild the batch-level fields from the clip documents.

    Deliberately narrow: it only writes what is derived from the clips (list,
    count, progress, the partial-success summary and the terminal verdict). It
    never deletes clips or rewrites the source/settings of a batch.
    """
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    jid = str(job.get("_id") or "")
    clips = _clips_for(store, job)
    label = partial_success_label(counts)
    done, failed, pending = counts["done"], counts["failed"], counts["pending"]
    total = counts["total"]

    # Keep the projected list in the same order as the documents, one document
    # per ORDER (a pre-fix detection pass can leave a rendered twin).
    by_order: dict[int, dict[str, Any]] = {}
    for c in sorted(clips, key=lambda c: (int(c.get("order") or 0), str(c.get("_id")))):
        order = int(c.get("order") or 0)
        prev = by_order.get(order)
        if prev is None or _clip_rank(c) > _clip_rank(prev):
            by_order[order] = c
    ordered = [by_order[k] for k in sorted(by_order)]
    clips = ordered
    done = sum(1 for c in ordered if _clip_state(c) == "done")
    failed = sum(1 for c in ordered if _clip_state(c) == "failed")
    pending = len(ordered) - done - failed
    total = len(ordered)
    label = partial_success_label({"done": done, "failed": failed, "pending": pending, "total": total})
    job["clips"] = ordered
    job["clipCount"] = done
    if total:
        job["clipTarget"] = max(int(job.get("clipTarget") or 0), total)
    # The card reads readyClips/failedClips, not clipCount. A repair that wrote
    # only clipCount left a batch with three verified clips showing zero ready.
    job["readyClips"] = done
    job["failedClips"] = failed
    job["renderFailures"] = failed
    if pending:
        # Not finished: leave a real, resumable checkpoint instead of a green
        # batch with invisible pending candidates. 99 is deliberate — the
        # progress bar must not claim 100 while work remains.
        prior_stage = str(job.get("stage") or "")
        job["progress"] = max(1, min(99, round(100 * done / total))) if total else 0
        job["progressDetail"] = (
            f"{label}; {pending} clip{'s' if pending != 1 else ''} still pending — "
            "Resume continues from the saved checkpoints."
        )
        job["stage"] = "failed"
        job["status"] = "failed"
        job["isTerminal"] = True
        job["interrupted"] = True
        job["retryable"] = True
        job["errorKind"] = "interrupted"
        job["errorAction"] = "resume"
        job["error"] = job["progressDetail"]
        job["errorDetail"] = None
        job["failedStage"] = job.get("failedStage") or prior_stage or "exporting"
        job["completedAt"] = job.get("completedAt") or now
        job["finishedAt"] = job.get("finishedAt") or job["completedAt"]
        job.pop("warning", None)
    else:
        job["progress"] = 100
        job["progressDetail"] = label + "."
        job["isTerminal"] = True
        job["stage"] = "complete" if done else ("failed" if failed else str(job.get("stage") or "complete"))
        job["status"] = job["stage"]
        job["completedAt"] = job.get("completedAt") or now
        job["finishedAt"] = job.get("finishedAt") or job["completedAt"]
        job["interrupted"] = False
        job["errorKind"] = None
        job["failedStage"] = None
        if failed and done:
            job["error"] = None
            job["warning"] = label + " — the failed clip can be retried from the batch."
            job["errorDetail"] = None
        elif failed and not done:
            job["error"] = label
        else:
            job["error"] = None
            job["errorDetail"] = None
            job.pop("warning", None)
    store.write_job(job)
    try:
        store.append_job_log(jid, f"[invariants] repaired: {reason}; {label}")
    except Exception as exc:  # noqa: BLE001 - the repair already landed
        logger.debug(f"[invariants] could not append log for {jid}: {exc}")
    logger.warning(f"[invariants] job {jid} repaired ({reason}) → {label}")
    return job


def rebuild_batches(store: Any, *, apply: bool = True) -> dict[str, Any]:
    """Rebuild missing batch records from the clip documents that exist.

    The failure this repairs: clips ARE rendered and uploaded (the bucket grew
    from 0.88 GB to 1.57 GB) but the batch-level document was never finalised,
    so "Your batches" listed nothing while the videos sat in the bucket.

    Rules, in order:
      * the clip ROWS are the truth, and ALL of them are read — the job's own
        projection is exactly what is missing when a batch was never finalised;
      * only delivered rows become the batch's clip list: cards for candidates
        that never rendered are not part of a finished batch (they are reported
        as `unmetReason` instead);
      * the write is bookkeeping only — nothing is re-rendered, re-downloaded
        or deleted;
      * a batch that is actively rendering is never touched.
    """
    report: dict[str, Any] = {"checked": 0, "rebuilt": 0, "jobs": []}
    try:
        jobs = list(store.list_jobs())
    except Exception as exc:  # noqa: BLE001 - a repair pass must not die
        logger.warning(f"[invariants] rebuild: could not list jobs: {exc}")
        return report
    now_ms = int(time.time() * 1000)
    for job in jobs:
        jid = str(job.get("_id") or "")
        if not jid:
            continue
        report["checked"] += 1
        try:
            docs = store.list_clip_docs_any_authority(jid) if hasattr(
                store, "list_clip_docs_any_authority"
            ) else _clips_for(store, job)
            delivered = [c for c in docs if _clip_state(c) == "done"]
            if not delivered:
                continue
            delivered.sort(key=lambda c: (int(c.get("order") or 0), str(c.get("_id"))))
            before = {
                "status": job.get("status"),
                "stage": job.get("stage"),
                "readyClips": int(job.get("readyClips") or 0),
                "clipCount": int(job.get("clipCount") or 0),
                "isTerminal": job.get("isTerminal"),
                "finishedAt": job.get("finishedAt"),
            }
            if (
                before["readyClips"] >= len(delivered)
                and before["clipCount"] >= len(delivered)
                and before["isTerminal"]
                and before["finishedAt"]
            ):
                continue  # already a correct, listed batch
            # Only a LIVE render is off limits. "stage says rendering" is not
            # the same thing as "a process is working": the batch that started
            # this whole repair was parked on `rendering` with no worker behind
            # it, and refusing to rebuild it left its clips listed nowhere.
            if _is_live_render(job):
                report["jobs"].append({"jobId": jid, "skipped": "job is actively rendering"})
                continue
            entry: dict[str, Any] = {
                "jobId": jid,
                "before": before,
                "delivered": len(delivered),
                "candidates": len(docs),
                "applied": bool(apply),
            }
            if apply:
                failed = [c for c in docs if _clip_state(c) == "failed"]
                counts = {
                    "done": len(delivered),
                    "failed": len(failed),
                    "pending": max(0, len(docs) - len(delivered) - len(failed)),
                    "total": len(docs),
                }
                labels = partial_success_label(counts)
                job["clips"] = delivered
                job["clipIds"] = [str(c.get("_id")) for c in delivered]
                job["clipCount"] = len(delivered)
                job["readyClips"] = len(delivered)
                job["failedClips"] = len(failed)
                job["renderFailures"] = len(failed)
                job["clipTarget"] = max(int(job.get("clipTarget") or 0), counts["total"])
                job["progress"] = 100
                job["progressMax"] = 100
                job["stage"] = "complete"
                job["status"] = "complete"
                job["isTerminal"] = True
                job["interrupted"] = False
                job["retryable"] = False
                job["errorKind"] = None
                job["errorAction"] = None
                job["error"] = None
                job["errorDetail"] = None
                job["failedStage"] = None
                job["finishedAt"] = job.get("finishedAt") or job.get("completedAt") or now_ms
                job["completedAt"] = job.get("completedAt") or job["finishedAt"]
                job["batchState"] = (
                    "finished_with_failures" if counts["failed"] else "finished"
                )
                job["finishedWithFailures"] = bool(counts["failed"])
                job["progressDetail"] = (
                    labels
                    if counts["failed"]
                    else f"{len(delivered)} clip{'s' if len(delivered) != 1 else ''} ready"
                ) + "."
                if counts["pending"]:
                    job["unmetReason"] = (
                        f"{counts['pending']} candidate clip(s) never rendered in the "
                        "attempt that produced this batch. The delivered clips are safe "
                        "and downloadable."
                    )
                # A rebuilt batch is settled: no running-only field may survive
                # or the card renders a spinner beside a finished batch.
                for field in _RUNNING_ONLY_FIELDS:
                    job.pop(field, None)
                store.write_job(job)
                entry["after"] = {
                    "status": job.get("status"),
                    "stage": job.get("stage"),
                    "readyClips": job.get("readyClips"),
                    "clipCount": job.get("clipCount"),
                    "isTerminal": job.get("isTerminal"),
                    "finishedAt": job.get("finishedAt"),
                    "batchState": job.get("batchState"),
                }
                report["rebuilt"] += 1
                logger.warning(
                    f"[invariants] rebuilt batch {jid}: {len(delivered)} delivered clip(s), "
                    f"{counts['pending']} candidate(s) never rendered "
                    f"(was readyClips={before['readyClips']}, finishedAt={before['finishedAt']})"
                )
                try:
                    store.append_job_log(
                        jid,
                        "[invariants] batch record rebuilt from verified clip rows "
                        f"({len(delivered)} ready) — nothing was re-rendered",
                    )
                except Exception:  # noqa: BLE001
                    pass
            report["jobs"].append(entry)
        except Exception as exc:  # noqa: BLE001 - one bad job must not stop the pass
            logger.warning(f"[invariants] rebuild failed for {jid}: {exc}")
            report["jobs"].append({"jobId": jid, "error": str(exc)[:300]})
    return report


def check_and_repair(
    store: Any,
    paths: Any,
    *,
    log: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Scan every job, repair the inconsistent ones, return a small report."""
    if not _REPAIR_LOCK.acquire(blocking=False):
        return {"checked": 0, "repaired": 0, "skipped": "already running"}
    try:
        checked = repaired = 0
        details: list[dict[str, Any]] = []
        for job in store.list_jobs():
            checked += 1
            restored_docs = restore_clip_docs(store, job)
            if restored_docs:
                details.append({
                    "jobId": job.get("_id"),
                    "restoredDocs": restored_docs,
                })
                clips = _clips_for(store, job)
            else:
                clips = _clips_for(store, job)
            counts = _counts(clips)
            broken = _needs_repair(job, counts, clips)
            if not broken and counts["total"] == 0:
                # A batch that died before any clip existed (bad fetch, bad
                # source) still needs the terminal flag written: the document
                # said stage=failed, isTerminal=false, so every raw reader
                # (not just the API normaliser) saw a job that would never end.
                stage = str(job.get("stage") or "")
                if stage in ("complete", "failed", "cancelled") and not job.get("isTerminal"):
                    job["isTerminal"] = True
                    job["finishedAt"] = job.get("finishedAt") or job.get("completedAt") or int(time.time() * 1000)
                    store.write_job(job)
                    repaired += 1
                    details.append({
                        "jobId": job.get("_id"),
                        "broken": ["terminal-stage-without-terminal-flag"],
                        "counts": counts,
                    })
                    logger.warning(
                        f"[invariants] job {job.get('_id')} repaired "
                        f"(terminal stage {stage!r} was not marked terminal)"
                    )
                continue
            if not broken:
                continue
            jid = str(job.get("_id") or "")
            # Never touch a batch a worker is actively rendering: it will write
            # its own consistent state when the current clip lands.
            if str(job.get("stage") or "") in ("rendering", "export", "exporting") and not job.get("isTerminal"):
                continue
            repair_job(store, job, counts, reason=", ".join(broken))
            repaired += 1
            details.append({"jobId": jid, "broken": broken, "counts": counts})
        report = {
            "checked": checked,
            "repaired": repaired,
            "restoredDocs": sum(d.get("restoredDocs", 0) for d in details),
            "details": details,
        }
        _LAST_RUN["at"] = time.time()
        _LAST_RUN["report"] = report
        if log and repaired:
            log(f"[invariants] {repaired}/{checked} job(s) repaired")
        return report
    finally:
        _REPAIR_LOCK.release()


def last_report() -> dict[str, Any] | None:
    return _LAST_RUN.get("report")


def start_periodic(
    store: Any,
    paths: Any,
    interval_sec: float = 300.0,
    log: Callable[[str], Any] | None = None,
) -> threading.Thread:
    """Run the checker every `interval_sec` (5 min) on a daemon thread."""

    def _loop() -> None:
        while True:
            time.sleep(interval_sec)
            try:
                # Same order as boot: rebuild missing batch records from the
                # clip rows first, then reconcile. A finished batch that fell
                # out of the list because its record was never written comes
                # back on the next pass, without anyone pressing a button.
                try:
                    rebuild_batches(store, apply=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[invariants] periodic rebuild failed: {exc}")
                check_and_repair(store, paths, log=log)
            except Exception as exc:  # noqa: BLE001 - a checker must never die loudly here
                logger.warning(f"[invariants] periodic check failed: {exc}")

    th = threading.Thread(target=_loop, name="invariants", daemon=True)
    th.start()
    return th
