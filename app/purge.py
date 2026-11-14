"""Delete a batch in the background, in small steps that survive a restart.

Why the endpoint no longer deletes anything itself
--------------------------------------------------
Deleting a batch used to run entirely inside the request: ask the worker to
cancel over its unix socket (30 s timeout, retried once — up to a minute when
the daemon is wedged), unlink the job and clip documents, walk and delete the
media tree, then paginate the bucket and delete its objects. Each step is
harmless alone; together, on a 1 vCPU host whose worker daemon was restarting
every few minutes, the request could sit past the proxy's timeout and the edge
answered with its own HTML 502 page. The user saw "the delete failed" while
nothing had been deleted, and the batch stayed in the list.

So the request now only marks the batch `deleting` (one locked document write)
and answers 202. This module owns the rest, on a daemon thread:

  cancel  — stop the runner (idempotent; "absent" is a fine answer)
  files   — remove a few files per tick, never a whole tree in one go
  objects — delete the batch's objects from durable storage
  records — remove the clip documents and the job document, last

Phase and counters live in the job document (`deleting.phase`, `.removed`,
`.attempts`, `.error`), so a host restart resumes where it stopped, and a job is
only removed once every file it owned is gone. A failure is recorded on the
batch — `deleting` plus a reason — instead of leaving it half-deleted and
invisible.

Safety: every path is resolved and must be strictly inside a media root AND
named after the job id. A job id that could escape the media folder is refused
before a single byte is touched, and one batch's purge can never walk into
another's directory.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from loguru import logger

#: The stage/status a batch wears while it is being removed.
STATUS = "deleting"

#: Files unlinked per tick. Small enough that a purge of a 15-clip batch never
#: holds the disk (or this thread) long enough to matter to the API.
FILES_PER_TICK = 8

#: How often the loop looks for work.
TICK_SEC = 0.25

#: Retry backoff, in seconds, indexed by attempt-1 (last value repeats).
BACKOFF_SEC: tuple[float, ...] = (2, 5, 15, 45, 120, 300)

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PurgeError(RuntimeError):
    """A purge step failed for a reason worth recording on the batch."""


def valid_job_id(job_id: str) -> bool:
    """True for ids this module is willing to turn into filesystem paths."""
    return bool(job_id) and bool(_JOB_ID_RE.match(str(job_id)))


def _now_ms() -> int:
    return int(time.time() * 1000)


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #

def media_roots(paths: Any) -> tuple[Path, ...]:
    """Every root a batch's files may live under.

    Sources are the exception: they can sit on a separate volume from the
    workspace (`downloads_dir`/`uploads_dir`), which is exactly why they are
    listed here instead of being assumed to be under `data_dir`.
    """
    out: list[Path] = []
    for root in (
        getattr(paths, "data_dir", None),
        getattr(paths, "outputs_dir", None),
        getattr(paths, "downloads_dir", None),
        getattr(paths, "uploads_dir", None),
    ):
        if root is None:
            continue
        try:
            out.append(Path(root).resolve())
        except OSError:
            continue
    return tuple(out)


def safe_target(target: Path, roots: Iterable[Path], job_id: str) -> bool:
    """True only for a job-scoped path strictly inside a media root.

    The id check is the first line of defence (no `.`, `/`, `\\`, no `..`), and
    the resolve check is the second: even a symlinked root cannot make
    `.../<job_id>` land outside the media folder, because resolution happens
    before the containment test.
    """
    if not valid_job_id(job_id):
        return False
    if Path(target).name != job_id:
        return False
    try:
        resolved = Path(target).resolve()
    except OSError:
        return False
    for root in roots:
        root = Path(root).resolve()
        # Strictly inside a media root — `data/work/<job>`, never the root
        # itself (`rmtree(data)` would be catastrophic) and never anything a
        # `..` could resolve above it.
        if resolved != root and resolved.is_relative_to(root):
            return True
    return False


def targets(paths: Any, job_id: str, *, keep_source: bool) -> list[Path]:
    """Every directory a deleted batch owns. Never a directory it does not."""
    data = Path(paths.data_dir)
    out = Path(paths.outputs_dir)
    roots = media_roots(paths)
    candidates: list[Path] = [
        data / "work" / job_id,
        data / "models" / job_id,
        data / "layers" / job_id,
    ]
    if not keep_source:
        job_sources = getattr(paths, "job_source_dirs", None)
        if callable(job_sources):
            candidates.extend(job_sources(job_id))
    candidates.append(out / job_id)
    good: list[Path] = []
    for cand in candidates:
        if safe_target(cand, roots, job_id):
            good.append(cand)
        else:
            logger.error(
                f"[purge] refusing unsafe path {cand} for job {job_id!r} — "
                "outside the media folder or not named after the job"
            )
    return good


def _iter_files(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    else:
                        yield Path(entry.path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PurgeError(f"could not read {d}: {exc}") from exc


def _prune(root: Path, budget: int) -> tuple[int, bool]:
    """Remove up to ``budget`` files under ``root``.

    Returns ``(removed, finished)``. ``finished`` is True only when the whole
    tree is gone (files first, then the empty directories), so the caller can
    advance to the next phase without ever claiming a half-deleted batch is
    done.
    """
    if budget <= 0:
        return 0, not Path(root).exists()
    if not Path(root).exists():
        return 0, True
    files = sorted(_iter_files(root))
    removed = 0
    for p in files[:budget]:
        try:
            os.unlink(p)
            removed += 1
        except FileNotFoundError:
            # Another process or an earlier interrupted run removed it.
            pass
        except OSError as exc:
            raise PurgeError(f"could not remove {p}: {exc}") from exc
    if len(files) > budget:
        return removed, False
    # No files left. Drop the empty shell (retries, sleeps, ignored errors:
    # only directories are left, so this cannot fail on a busy file).
    shutil.rmtree(root, ignore_errors=True)
    return removed, not Path(root).exists()


# --------------------------------------------------------------------------- #
# job document
# --------------------------------------------------------------------------- #

#: A batch may only be DELETED once it has settled. Anything else is work in
#: flight: its worker is decoding, its files are being written, and the purge
#: below cancels that worker and unlinks what it was writing.
DELETABLE_STATUSES = frozenset({"complete", "failed"})


class JobNotDeletable(ValueError):
    """Refused: the batch is still running (or paused), not finished."""


def deletable_reason(job: dict[str, Any]) -> str | None:
    """None when this batch may be deleted, else a sentence for the caller."""
    status = str(job.get("status") or job.get("stage") or "")
    if status in DELETABLE_STATUSES or status == STATUS:
        return None
    label = str(job.get("stage") or status or "unknown")
    return (
        f"this batch is still {label} — it has not finished, so deleting it would "
        "cancel work in progress. Wait for it (or press Pause), then delete."
    )


def mark(
    store: Any, job_id: str, *, keep_source: bool = False, require_finished: bool = False,
) -> dict[str, Any] | None:
    """Mark a batch as deleting. The ONLY work the API does synchronously.

    Re-marking an already-deleting batch (a second tap after a failure) keeps
    the phase and the removed counter and clears the retry backoff, so the
    retry is a nudge rather than a restart.

    `require_finished` is what the MASS path ("Clean up dead batches") passes.
    That dialog asks the server for junk, then sends every id it did not
    classify — which on a busy dashboard included a batch that was still
    grabbing. Marking it deleting cancelled its worker and removed 35 files
    mid-run, so a bulk delete now refuses anything that has not settled while
    the single, explicitly confirmed delete keeps its old behaviour.
    """
    if not valid_job_id(job_id):
        raise ValueError(f"invalid job id: {job_id!r}")
    if require_finished:
        job = store.get_job(job_id) or {}
        if not job:
            return None
        reason = deletable_reason(job)
        if reason:
            raise JobNotDeletable(reason)

    def mutate(job: dict[str, Any]) -> None:
        prev = dict(job.get("deleting") or {})
        job["stage"] = STATUS
        job["status"] = STATUS
        job["error"] = None
        job["errorDetail"] = None
        for field in (
            "queuePosition", "stagePlan", "unitsDone", "unitsTotal", "unitLabel",
            "stageEtaSec", "totalEtaSec", "activeStage", "stageLabel",
        ):
            job.pop(field, None)
        job["progress"] = 100
        job["progressDetail"] = "Deleting — removing this batch's files in the background."
        job["deleting"] = {
            "startedAt": prev.get("startedAt") or _now_ms(),
            "keepSource": bool(keep_source),
            "phase": prev.get("phase") or "cancel",
            "removed": int(prev.get("removed") or 0),
            "attempts": 0,
            "error": None,
            "nextAttemptAt": 0,
            "updatedAt": _now_ms(),
        }

    return store.update_job(job_id, mutate)


def _patch(store: Any, job_id: str, **fields: Any) -> None:
    def mutate(job: dict[str, Any]) -> None:
        state = dict(job.get("deleting") or {})
        state.update(fields)
        state["updatedAt"] = _now_ms()
        job["deleting"] = state
        job["stage"] = STATUS
        job["status"] = STATUS

    store.update_job(job_id, mutate)


def pending(store: Any) -> list[str]:
    """Ids still waiting to be purged — used by the boot log and the tests."""
    out: list[str] = []
    for job in store.list_jobs():
        if job.get("stage") == STATUS or job.get("status") == STATUS:
            out.append(str(job.get("_id")))
    return out


# --------------------------------------------------------------------------- #
# the work
# --------------------------------------------------------------------------- #

def advance(
    store: Any,
    paths: Any,
    job: dict[str, Any],
    *,
    cancel: Callable[[str], Any] | None = None,
    remote: Callable[[str], Any] | None = None,
) -> None:
    """Run one step of a deleting batch's purge. Safe to call repeatedly."""
    job_id = str(job.get("_id") or "")
    if not valid_job_id(job_id):
        logger.error(f"[purge] refusing job id {job_id!r} — not a safe path component")
        return
    state = dict(job.get("deleting") or {})
    phase = str(state.get("phase") or "cancel")

    if phase == "cancel":
        if cancel is not None:
            try:
                what = cancel(job_id)
                logger.info(f"[purge] {job_id}: worker cancel -> {what}")
            except Exception as exc:  # noqa: BLE001 - purge must go on
                logger.warning(f"[purge] {job_id}: worker cancel failed: {exc}")
        _patch(store, job_id, phase="files")
        return

    if phase == "files":
        removed = 0
        finished = True
        for target in targets(paths, job_id, keep_source=bool(state.get("keepSource"))):
            got, done = _prune(target, FILES_PER_TICK - removed)
            removed += got
            finished = finished and done
            if not done:
                break
        total = int(state.get("removed") or 0) + removed
        if not finished:
            if removed:
                logger.info(
                    f"[purge] {job_id}: removed {removed} file(s) "
                    f"(total {total}) — more to go"
                )
            _patch(store, job_id, removed=total)
            return
        logger.info(f"[purge] {job_id}: media gone ({total} file(s) removed)")
        _patch(store, job_id, removed=total, phase="objects")
        return

    if phase == "objects":
        if remote is not None:
            try:
                remote(job_id)
            except Exception as exc:  # noqa: BLE001 - recorded, retried
                raise PurgeError(f"bucket purge failed: {exc}") from exc
        else:
            # No object store configured: nothing to remove remotely.
            logger.info(f"[purge] {job_id}: no object store configured — skipping bucket")
        _patch(store, job_id, phase="records")
        return

    if phase == "records":
        # Last, so the batch stays visible (as `deleting`) until nothing is
        # left behind it. `delete_job` removes the clip documents too.
        store.delete_job(job_id)
        # A read path can recreate an empty `data/work/<id>/<stage>` directory
        # while the purge runs (constructing a StageWorkspace mkdirs it), so
        # the documents go first and the now-fileless tree is dropped after.
        # Nothing but this batch's own directories can be reached: `targets`
        # already refused anything else.
        for target in targets(paths, job_id, keep_source=bool(state.get("keepSource"))):
            shutil.rmtree(target, ignore_errors=True)
        try:
            from app import hosting  # local: keeps this module importable alone

            hosting.invalidate_disk_cache()
        except Exception:  # noqa: BLE001 - the number is cosmetic
            pass
        logger.info(f"[purge] {job_id}: batch removed (records + media)")
        return

    logger.warning(f"[purge] {job_id}: unknown phase {phase!r} — restarting at files")
    _patch(store, job_id, phase="files")


def tick(
    store: Any,
    paths: Any,
    *,
    cancel: Callable[[str], Any] | None = None,
    remote: Callable[[str], Any] | None = None,
) -> int:
    """Advance every deleting batch once. Returns how many were advanced."""
    now = time.time()
    handled = 0
    for job in store.list_jobs():
        # The `deleting` document is the authority, not `stage`: a runner that
        # was mid-stage when the user tapped delete can write its own stage
        # back for a moment after the mark, and that must not orphan the purge.
        if not job.get("deleting") and job.get("stage") != STATUS and job.get("status") != STATUS:
            continue
        state = dict(job.get("deleting") or {})
        try:
            due = float(state.get("nextAttemptAt") or 0)
        except (TypeError, ValueError):
            due = 0.0
        if due > now:
            continue
        handled += 1
        try:
            advance(store, paths, job, cancel=cancel, remote=remote)
        except Exception as exc:  # noqa: BLE001 - every step is retried
            attempts = int(state.get("attempts") or 0) + 1
            delay = BACKOFF_SEC[min(attempts, len(BACKOFF_SEC)) - 1]
            reason = str(exc) or type(exc).__name__
            _patch(
                store,
                str(job.get("_id")),
                attempts=attempts,
                error=reason[:300],
                nextAttemptAt=time.time() + delay,
            )
            logger.warning(
                f"[purge] {job.get('_id')}: {reason} — retrying in {delay:g}s "
                f"(attempt {attempts})"
            )
    return handled


# --------------------------------------------------------------------------- #
# daemon loop
# --------------------------------------------------------------------------- #

_thread: threading.Thread | None = None
_stop = threading.Event()


def _loop(
    store: Any,
    paths: Any,
    cancel: Callable[[str], Any] | None,
    remote: Callable[[str], Any] | None,
    interval: float,
) -> None:
    while not _stop.is_set():
        try:
            tick(store, paths, cancel=cancel, remote=remote)
        except Exception as exc:  # noqa: BLE001 - the loop must not die
            logger.warning(f"[purge] pass failed: {exc}")
        if _stop.wait(interval):
            break


def start(
    store: Any,
    paths: Any,
    *,
    cancel: Callable[[str], Any] | None = None,
    remote: Callable[[str], Any] | None = None,
    interval: float = TICK_SEC,
) -> threading.Thread:
    """Start the purge loop, finishing anything a restart left in `deleting`."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return _thread
    waiting = pending(store)
    if waiting:
        logger.info(
            f"[boot] resume delete: {len(waiting)} batch(es) left in `deleting` "
            f"({', '.join(waiting[:4])}{'…' if len(waiting) > 4 else ''})"
        )
    _stop.clear()
    _thread = threading.Thread(
        target=_loop,
        args=(store, paths, cancel, remote, interval),
        name="batch-purge",
        daemon=True,
    )
    _thread.start()
    return _thread


def stop() -> None:
    """Stop the loop (tests, shutdown)."""
    _stop.set()
    global _thread
    _thread = None
