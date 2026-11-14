"""Cheap, continuous proof that a 'ready' clip still is.

The render contract is checked once, on the machine that produced the file.
That proof is stale the moment the sandbox is recycled, a bucket object is
deleted by hand, or an offload is interrupted. This monitor re-checks every
ready clip at boot and every ten minutes:

  * the bucket object exists and its size matches the clip document;
  * a ranged GET really returns 206 (not a 200 full body, not a sign error);
  * the clip has an fps and a duration, so the UI badge is not guessing.

It never downloads a clip to the 4 GB disk to answer this. A clip that fails
is flagged with `healthState` + `healthIssue` and the dashboard's existing
Retry button can re-render it; a local verified copy is re-uploaded first.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from loguru import logger

_REPORT: dict[str, Any] = {"at": 0.0, "checked": 0, "bad": 0, "items": []}
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None


def check(store: Any, settings: Any, *, limit: int = 512) -> dict[str, Any]:
    """HEAD + ranged 206 + stored fps for every done clip (cheap)."""
    from app import objectstore

    started = time.time()
    items: list[dict[str, Any]] = []
    try:
        clips = store.list_all_clips(limit=limit)
    except Exception as exc:  # noqa: BLE001 - a monitor must not break the app
        return {"ok": False, "error": f"cannot list clips: {exc}", "checked": 0, "bad": 0, "items": []}
    for clip in clips:
        if str(clip.get("renderState") or "") not in {"done", "complete", "rendered", "uploaded"}:
            continue
        cid = str(clip.get("_id") or "")
        issues: list[str] = []
        path = clip.get("videoPath")
        local = None
        try:
            from pathlib import Path

            local = Path(str(path)) if path else None
        except Exception:  # noqa: BLE001
            local = None
        if not clip.get("fps"):
            issues.append("no stored fps")
        if not clip.get("mediaDurationSec"):
            issues.append("no stored duration")
        key = objectstore.key_for_path(str(path), settings.app.paths.outputs_dir) if path else None
        head = objectstore.head(key) if key else None
        if local is not None and local.exists() and local.stat().st_size > 0:
            if int(clip.get("mediaBytes") or 0) and local.stat().st_size != int(clip.get("mediaBytes") or 0):
                issues.append("local size differs from clip document")
            if key and not head:
                # The local copy is the verified one; restore the durable copy.
                if objectstore.upload_file(local, key):
                    head = objectstore.head(key)
                    logger.info(f"[clip-health] {cid}: re-uploaded missing bucket object")
                else:
                    issues.append("bucket object missing and re-upload failed")
        elif key:
            if not head:
                issues.append("bucket object missing")
            elif int(clip.get("mediaBytes") or 0) and int(head.get("size") or 0) != int(clip.get("mediaBytes") or 0):
                issues.append(
                    f"bucket size {head.get('size')} != clip document {clip.get('mediaBytes')}"
                )
            opened = objectstore.open_stream(key, "bytes=0-1023")
            if opened is None:
                issues.append("ranged GET failed")
            else:
                status, _headers, body = opened
                try:
                    data = body.read()
                finally:
                    try:
                        body.close()
                    except Exception:  # noqa: BLE001
                        pass
                if status != 206:
                    issues.append(f"ranged GET returned {status}, expected 206")
                if not data:
                    issues.append("ranged GET returned no bytes")
        elif not (local and local.exists()):
            issues.append("no local file and no bucket key")

        state = "ok" if not issues else "failed"
        if issues:
            try:
                store.write_clip(cid, {
                    "healthState": state,
                    "healthIssue": "; ".join(issues),
                    "healthCheckedAt": int(time.time() * 1000),
                }, job_id=str(clip.get("jobId") or "") or None)
            except Exception as exc:  # noqa: BLE001 - keep checking the rest
                logger.warning(f"[clip-health] could not flag {cid}: {exc}")
            items.append({"clipId": cid, "order": clip.get("order"), "issues": issues})
        else:
            # Clear a previous flag once the clip is healthy again; otherwise a
            # repaired file would keep its Retry badge forever.
            if clip.get("healthState") or clip.get("healthIssue"):
                try:
                    store.write_clip(cid, {
                        "healthState": "ok",
                        "healthIssue": None,
                        "healthCheckedAt": int(time.time() * 1000),
                    }, job_id=str(clip.get("jobId") or "") or None)
                except Exception as exc:  # noqa: BLE001 - keep checking the rest
                    logger.warning(f"[clip-health] could not clear {cid}: {exc}")
            logger.debug(f"[clip-health] {cid}: head + ranged 206 + fps ok")
    report = {
        "ok": not items,
        "at": int(started * 1000),
        "checked": sum(
            1 for c in clips
            if str(c.get("renderState") or "") in {"done", "complete", "rendered", "uploaded"}
        ),
        "bad": len(items),
        "items": items[:40],
    }
    with _LOCK:
        _REPORT.clear()
        _REPORT.update(report)
    return report


def report() -> dict[str, Any]:
    with _LOCK:
        return dict(_REPORT)


def start_monitor(store: Any, settings: Any, *, interval: float = 600.0) -> threading.Thread:
    """Boot + every 10 min. Idempotent."""
    global _THREAD
    if _THREAD is not None and _THREAD.is_alive():
        return _THREAD

    def loop() -> None:
        while True:
            try:
                result = check(store, settings)
                if result.get("bad"):
                    logger.warning(
                        f"[clip-health] {result['bad']} of {result['checked']} ready clip(s) "
                        "failed the cheap contract — flagged for Retry"
                    )
            except Exception as exc:  # noqa: BLE001 - housekeeping must not die
                logger.warning(f"[clip-health] pass failed: {exc}")
            time.sleep(interval)

    _THREAD = threading.Thread(target=loop, name="clip-health", daemon=True)
    _THREAD.start()
    return _THREAD
