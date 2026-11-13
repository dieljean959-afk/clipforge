"""ClipForge local API — FastAPI backend + in-process worker, one process.

Endpoints serve the React dashboard and the pipeline artifacts on ./downloads
and ./outputs. Boot sequence: config validation → ffmpeg check → Whisper model
pre-download (explicit, with progress) → worker thread starts.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from app import config

from app import (
    clipcount,
    diskreset,
    objectstore,
    purge,
    reclaim,
    share_video,
    sharelink,
    worker,
)


from uuid import uuid4 as uuid4


SHARE_CACHE_CONTROL = "public, max-age=31536000, immutable"
MEDIA_CACHE_CONTROL = "public, max-age=300"
_MIN_FREE_BYTES = 100 * 1024 * 1024  # 100 MB free-disk floor before HTTP 507
#: Container types the pipeline can probe and render. Anything else is refused
#: at the door instead of failing later inside ffmpeg.
ALLOWED_UPLOAD_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


_STARTED_AT = time.time()


from app.config import (
    ConfigError,
    boot_banner,
    ffmpeg_bin,
    ffprobe_bin,
    load_config,
)
from app.store import LocalStore
from app.social.accounts import AccountStore
from app.hosting import RetentionCleaner
from app import video as video_svc
from app.social import PLATFORMS, describe_all, redirect_uri
from app.social import base as social_base
from app.social import post_video as social_post
from app.social.base import (
    NeedsReconnect,
    NotConfigured,
    PostRejected,
    SocialError,
)
# Module handle, distinct from the `worker` INSTANCE below. The boot recovery
# pass needs `unrecoverable_reason` / `INTERRUPT_MAX_RESUMES` from the module;
# without this import that pass raised NameError at import time and uvicorn
# exited before binding port 8000, so the proxy answered a bodiless 500.
from app import worker as worker_mod
_lock: InstanceLock | None = None

from app.boot import (
    Boot,
    InstanceLock,
    clear_failure,
    failure_payload,
    last_failure,
    record_fatal,
)
from app.worker import InProcessWorker
from app.workerd import WorkerSupervisor
# The API process started; used by GET /api/health to report true uptime.
HOST_STARTUP_MS = int(time.time() * 1000)


def _purge_remote_job(job_id: str) -> None:
    """Flag one in-flight big-job for deletion so the sync worker finishes it.
    (Placeholder used by the purge boot path; the real delete runs in purge.py.)
    """
    pass


def _job_list(store=None):
    """List jobs the same way _active_job_for does (durable records, sorted)."""
    from app.store import LocalStore as _Store
    store = _Store(settings.app.paths)
    jobs = store.list_jobs() if store is not None else store.list_jobs()
    return sorted(jobs, key=lambda j: j.get("createdAt") or 0, reverse=True)


def _describe_job(job):
    """Normalise one job the way jobstatus.describe() does for the dashboard."""
    from app import jobstatus
    return jobstatus.describe(job or {})


def _share_backfill() -> dict[str, int]:
    """Give every pre-existing clip a share id, and report what it did.

    Runs in the BACKGROUND. A deployment that gains share links overnight has
    hundreds of clips with no id; minting them is a few file writes, but doing
    it inline would delay the boot that has to re-queue interrupted jobs, and
    those jobs are the urgent thing. So: one cheap pass inline is enough for a
    fresh install, and the thread finishes the backlog behind the UI.
    """
    started = sharelink.backfill(store)
    if started["minted"] or started.get("tokens"):
        logger.info(
            f"[share] minted {started['minted']} share id(s) and "
            f"{started.get('tokens', 0)} public token(s) for existing clips"
        )

    def _finish() -> None:
        try:
            rest = sharelink.backfill(store, limit=2000)
            if rest["minted"] or rest.get("tokens"):
                logger.info(
                    f"[share] backfilled {rest['minted']} more share id(s) and "
                    f"{rest.get('tokens', 0)} token(s) "
                    f"({rest['scanned']} clips scanned)"
                )
        except Exception as exc:  # noqa: BLE001 - telemetry only
            logger.warning(f"[share] background backfill failed: {exc}")

    threading.Thread(target=_finish, name="share-backfill", daemon=True).start()
    return started


#: Repository root. Used for state that is keyed by path rather than by
#: config (the live-key refreshers). Derived, never hardcoded, so the app keeps
#: working when it is started from another directory.
ROOT = Path(__file__).resolve().parent.parent


#: Post-listen work. Created in `_boot` (cheap) and filled in by
#: `_boot_phases()` below, which runs on a daemon thread started from the
#: lifespan startup handler. Nothing here is allowed to delay the port.
boot = Boot()


def _boot() -> None:
    """Cheap, synchronous setup: config, store handles, log files, instance lock.

    Everything that reads a lot of disk, talks to the network or waits on a
    process runs AFTER the port is open (see `_boot_phases`). The rule is that
    this function must stay under ~500 ms on a cold cache, because uvicorn
    cannot bind its port until import finishes.
    """
    global settings, store, worker, retention, accounts, _lock
    try:
        settings = load_config()
    except ConfigError as exc:
        # A config error is a real boot failure: print one line with the cause
        # and the fix, and record it so the preview can show it.
        payload = failure_payload(
            "config",
            str(exc),
            "Fix config.toml / the environment variables named above, then restart.",
            code="bad_config",
        )
        record_fatal(payload, _data_dir_hint())
        print(f"[boot] FATAL config: {exc} → FIX: {payload['fix']}", flush=True)
        raise SystemExit(2) from exc
    boot_banner(settings)

    # fail fast on missing ffmpeg/ffprobe (a ffmpeg check is two stats, not a
    # subprocess: keep it here so a broken host fails in one clear line)
    logger.info(f"[boot] ffmpeg: {ffmpeg_bin()}")
    logger.info(f"[boot] ffprobe: {ffprobe_bin()}")

    store = LocalStore(settings.app.paths)
    worker = WorkerSupervisor(settings, store)
    retention = RetentionCleaner(settings, store)
    accounts = AccountStore(settings.app.paths.data_dir)

    # --- one instance only ------------------------------------------------
    # Two uvicorns on one store corrupt job documents; two workers race for the
    # same clip files. The lock is a pid file, reclaimed when that pid is gone.
    _lock = InstanceLock(
        settings.app.paths.data_dir / "api.lock", "api", pid=os.getpid()
    )
    if not _lock.acquire():
        holder = _lock.holder or "another process"
        payload = failure_payload(
            "instance-lock",
            f"another API instance (pid {holder}) already owns this data dir",
            "Stop the other server (or delete data/api.lock if that pid is gone) "
            "and restart. Only one API may own the store.",
            code="port_or_lock_in_use",
        )
        # A SERVER must not start a second copy (the port bind also refuses it,
        # but this is the one that protects the store from two writers). A
        # test/tool importing this module while the server runs is not a second
        # server: record it and keep going, or the suite could never import.
        serving = any("uvicorn" in str(arg) for arg in sys.argv)
        if serving:
            record_fatal(payload, settings.app.paths.data_dir)
            print(
                f"[boot] FATAL instance-lock: {payload['cause']} → FIX: {payload['fix']}",
                flush=True,
            )
            raise SystemExit(3)
        logger.warning(
            f"[boot] {payload['cause']} — continuing read-only (this process is "
            "not the server)"
        )

    # --- durable logs + native-crash tracebacks ---------------------------
    try:
        log_dir = settings.app.paths.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "api.log",
            rotation="10 MB",
            retention=5,
            enqueue=False,
            level="INFO",
            backtrace=True,
            diagnose=False,
        )
        import faulthandler

        _crash_fh = open(log_dir / "crash.log", "a", encoding="utf-8")
        faulthandler.enable(_crash_fh)
    except Exception as exc:  # noqa: BLE001 - logging must never block boot
        logger.warning(f"[boot] file logging unavailable: {exc}")

    # The port is about to open; record how long the synchronous path took.
    boot.listen_ms = int((time.time() - boot.started_at) * 1000)
    clear_failure(settings.app.paths.data_dir)


def _data_dir_hint() -> Path | None:
    """The data dir for failure reports, available even when config failed."""
    try:
        return settings.app.paths.data_dir  # type: ignore[name-defined]
    except Exception:
        return None


def _boot_phases(b: Boot) -> None:
    """Everything heavy, after the port is open. Each phase has its own budget.

    A phase that fails or times out is logged, shown by /api/status and never
    aborts the ones after it: one broken dependency must not cost the dashboard.
    """
    data_dir = settings.app.paths.data_dir

    # ORDER MATTERS: recovery decides what happens to in-flight jobs before the
    # worker daemon starts. The daemon adopts any job that looks in-flight, so a
    # recovery pass that runs second finds every job "owned" and writes no
    # decision at all — the batch then sat at "exporting" forever.
    def _share() -> dict:
        started = _share_backfill()
        return {"detail": f"minted {started.get('minted', 0)} share id(s)"}

    b.run_phase("share_backfill", _share, timeout_ms=30_000)

    def _recover() -> dict:
        resumed, restored = _recover_jobs()
        return {"detail": f"{resumed} resumed, {restored} restored from clips"}

    b.run_phase("job_recovery", _recover, timeout_ms=45_000)

    # --- worker daemon ---------------------------------------------------
    def _spawn() -> dict:
        start_ms = time.perf_counter()
        worker.start()
        st = worker.status()
        return {
            "detail": (
                f"online={st.get('online')} pid={worker.daemon_pid} "
                f"concurrency={st.get('concurrency')}"
            ),
            "spawnMs": int((time.perf_counter() - start_ms) * 1000),
        }

    b.run_phase("worker_spawn", _spawn, timeout_ms=20_000)


    def _purge() -> dict:
        purge.start(
            store,
            settings.app.paths,
            cancel=worker.cancel,
            remote=_purge_remote_job,
        )
        return {"detail": "delete purge started"}

    def _reap_daemons() -> dict:
        from app import workerd as _wd

        # A daemon whose lock file is gone cannot be adopted or stopped; it only
        # holds RAM. Reap before spawning anything else.
        killed = _wd.reap_stray_daemons(settings, keep_pids={worker.daemon_pid} if worker.daemon_pid else set())
        return {"detail": f"{len(killed)} stray daemon(s) reaped"}

    b.run_phase("worker_dedupe", _reap_daemons, timeout_ms=10_000)

    b.run_phase("worker_purge", _purge, timeout_ms=5_000)

    def _reclaim() -> dict:
        reclaim.start_watchdog(store, settings.app.paths)
        return {"detail": "reclaim watchdog started"}

    b.run_phase("reclaim_watchdog", _reclaim, timeout_ms=5_000)

    def _crash_sweep() -> dict:
        swept = reclaim.sweep_startup(store, settings.app.paths)
        mb = round((swept.get("bytesFreed") or 0) / 1048576, 1)
        return {
            "detail": (
                f"{swept.get('dirsRemoved', 0)} folder(s), "
                f"{swept.get('filesRemoved', 0)} temp file(s), {mb} MB freed"
            )
        }

    b.run_phase("crash_sweep", _crash_sweep, timeout_ms=30_000)

    def _disk_loop() -> dict:
        diskreset.start_sweeper(store, settings)
        return {"detail": "10-minute crash-proof disk sweeper started"}

    b.run_phase("disk_reset_watchdog", _disk_loop, timeout_ms=5_000)

    def _clip_health() -> dict:
        from app import cliphealth

        cliphealth.start_monitor(store, settings)
        return {"detail": "clip HEAD/range/fps monitor started (boot + every 10 min)"}

    b.run_phase("clip_health", _clip_health, timeout_ms=5_000)

    def _warm() -> dict:
        _do_warmup()
        return {"detail": warmup.get("current") or "unknown"}

    # A model download can legitimately take minutes; it must never gate boot.
    b.run_phase("model_preload", _warm, timeout_ms=180_000)

    def _retention() -> dict:
        retention.start()
        swept = retention.sweep()
        return {"detail": f"{swept} expired/orphaned item(s) removed"}

    b.run_phase("retention_sweep", _retention, timeout_ms=30_000)

    def _logs() -> dict:
        """Bound the log files and release model weights no job needs.

        These are the two things that actually grew during a long render
        (27 MB supervisor log + rotated api logs + model copies). Doing it in
        one measured phase means a reboot leaves the disk where it found it,
        instead of inheriting the previous run's debris.
        """
        from app import loghygiene

        return loghygiene.sweep(settings)

    b.run_phase("log_hygiene", _logs, timeout_ms=20_000)

    def _invariants() -> dict:
        from app import invariants

        report = invariants.check_and_repair(store, settings.app.paths, log=logger.info)
        return {
            "detail": (
                f"{report['checked']} job(s) checked, {report['repaired']} repaired"
            )
        }

    b.run_phase("invariants", _invariants, timeout_ms=60_000)

    def _updates() -> dict:
        # yt-dlp's own update check is a network call. It is allowed to be slow,
        # never to be un-timed, and never to run before the port is open.
        from app import download as _dl

        if not hasattr(_dl, "check_for_update"):
            return {"skipped": True, "detail": "no update check in this build"}
        res = _dl.check_for_update(timeout=8)
        return {"detail": str(res)}

    b.run_phase("update_checks", _updates, timeout_ms=15_000)

    # The dashboard polls /api/health; its expensive blocks live in this thread
    # so the route itself stays a memory read.
    b.run_phase("health_snapshot", start_health_snapshot, timeout_ms=5_000)

    b.extra.append  # keep the phase list stable for /api/status
    clear_failure(data_dir)


def _recover_jobs() -> tuple[int, int]:
    """Re-queue interrupted jobs and restore records whose clips are on disk.

    Returns ``(resumed, restored)``. Split out of the old import-time boot so it
    can be timed and bounded; the body is unchanged.
    """
    resumed = 0
    restored = 0
    recoverable = {"grabbing", "transcribing", "finding", "rendering", "queued"}
    # Jobs a runner is busy with RIGHT NOW. Recovery must not touch these.
    try:
        running_now = set(worker.active_pids() or {})
    except Exception:  # noqa: BLE001 - no daemon answer means nothing of ours is running
        running_now = set()
    for job in store.list_jobs():
        if job.get("_id") in running_now:
            continue
        rendered, promised = store.clips_delivered(job["_id"], job)
        if InProcessWorker.has_delivered_output(job, rendered, promised):
            logger.info(
                f"[boot] job {job['_id']} is marked failed but all {rendered} clips are on "
                f"disk — restoring stage=complete (the failure cost no output)"
            )
            job.update({
                "stage": "complete",
                "progress": 100,
                "progressDetail": f"{rendered} clips ready.",
                "clipCount": rendered,
                "completedAt": job.get("completedAt") or int(time.time() * 1000),
                "failedStage": None,
                "error": None,
                "errorDetail": None,
            })
            store.write_job(job)
            store.append_job_log(
                job["_id"],
                f"marked complete at boot: all {rendered} clips were already rendered, "
                "so the recorded failure had no output to show for it",
            )
            restored += 1
            continue
        # A finished batch can still carry `interrupted: true` from an earlier
        # attempt that was killed and then resumed successfully.
        if job.get("stage") == "complete" and job.get("interrupted"):
            job["interrupted"] = False
            store.write_job(job)
            continue
        reason = None
        if job.get("stage") in recoverable:
            reason = "mid-flight"
        elif job.get("interrupted"):
            reason = "flagged interrupted"
        elif InProcessWorker.has_signal_failure(job):
            reason = "recorded as failed, but the cause was an external kill"
        if reason is None:
            continue
        dead_end = worker_mod.unrecoverable_reason(store, job)
        if dead_end is not None:
            logger.warning(
                f"[boot] job {job['_id']} cannot be resumed ({reason}): "
                "source file is gone — failing instead of re-queueing"
            )
            job.update({
                "stage": "failed",
                "error": dead_end,
                "errorDetail": None,
                "interrupted": False,
                "completedAt": job.get("completedAt") or int(time.time() * 1000),
            })
            store.write_job(job)
            store.append_job_log(job["_id"], f"NOT RESUMABLE: {dead_end}")
            continue
        resumes = int(job.get("interruptCount") or 0)
        if resumes >= worker_mod.INTERRUPT_MAX_RESUMES:
            logger.warning(
                f"[boot] job {job['_id']} exhausted its resume budget "
                f"({resumes}) — leaving it failed instead of looping"
            )
            continue
        was = job.get("stage")
        if was == "failed":
            job.update({
                "error": None,
                "completedAt": None,
                "interrupted": True,
                "interruptReason": "Recovered after an external kill — resuming from cached stages.",
            })
            store.write_job(job)
        job.update({
            "stage": "queued",
            "progress": 0,
            "progressDetail": "Queued — resuming from cached stages after a restart.",
            "completedAt": None,
            "error": None,
            "errorDetail": None,
            "unitLabel": None,
            "unitsDone": None,
            "unitsTotal": None,
        })
        store.write_job(job)
        logger.info(f"[boot] re-queuing job {job['_id']} ({reason}, was {was})")
        worker.enqueue(job["_id"])
        resumed += 1
    return resumed, restored


def _whisper_model_cached(model: str) -> bool:
    """Is this faster-whisper model already in the HuggingFace cache?

    A directory check only — deliberately NOT a model load. The API process
    must never hold a Whisper model: on a 2 GiB host that is ~1 GB of RSS in
    the same process that has to answer /api/health, and it is the reason a
    job could starve the API. The model is loaded once inside the per-job
    runner child process instead, and reused there for every clip.
    """
    from pathlib import Path

    from app.asr import model_cache_dir

    base = Path(model_cache_dir())
    if not base.is_dir():
        return False
    needle = model.replace("/", "--")
    return any(
        p.is_dir() and p.name.startswith("models--") and needle in p.name
        for p in base.iterdir()
    )


def _do_warmup() -> None:
    """Report model readiness without loading anything in the API process."""
    global warmup

    try:
        from app.asr import resolved_model

        model, note = resolved_model(settings)
        cached = _whisper_model_cached(model)
        warmup = {
            "current": "already downloaded" if cached else "downloads on first job",
            "configured": settings.whisper.model,
            "model": model,
            "engine": "faster-whisper",
        }
        if note:
            warmup["note"] = note
    except Exception as exc:
        warmup = {
            "current": "unknown",
            "reason": f"{type(exc).__name__}: {exc}",
            "configured": settings.whisper.model,
        }

warmup = {}


# --------------------------------------------------------------------------- #
# /api/health — the dashboard's whole status card, answered from a snapshot
# --------------------------------------------------------------------------- #
#
# The card reads one document: worker, warmup, whisper, asr, system, disk,
# tools, cookies, storage, dispatch, detection, limits, tailnet. Building all of
# that on every poll is exactly what made this route time out (a version probe
# per poll, a bucket HEAD, a tailnet round-trip). The EXPENSIVE blocks are
# rebuilt every 2.5 s in one background thread; the request merges the snapshot
# with the cheap live bits (worker state, warmup, uptime) and returns in ~1 ms.
# A block that fails is named in `degraded` — never silently dropped.

_HEALTH_SLOW: dict[str, Any] = {}
_HEALTH_SLOW_AT = 0.0
_HEALTH_LOCK = threading.Lock()
_HEALTH_THREAD: threading.Thread | None = None
HEALTH_SNAPSHOT_SEC = 2.5
_MODEL_BLOCK: dict[str, Any] | None = None
_MODEL_BLOCK_AT = 0.0
_DISPATCH_BLOCK: dict[str, Any] | None = None
_DISPATCH_BLOCK_AT = 0.0
_STORAGE_BLOCK: dict[str, Any] | None = None
_STORAGE_BLOCK_AT = 0.0


def _tool_version(cmd: list[str] | None) -> str | None:
    """First line of `<tool> -version`, or None. Bounded, background only."""
    if not cmd:
        return None
    import subprocess

    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3, check=False
        )
        text = (out.stdout or out.stderr or "").strip().splitlines()
        return text[0].strip() if text else None
    except Exception:  # noqa: BLE001 - "could not run it" is the honest answer
        return None


def _cookie_state() -> dict[str, Any] | None:
    """The yt-dlp cookie jar: present?, age, count, and whether it is stale."""
    import shutil

    raw = str(getattr(settings.download, "cookies_file", "") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        return {"present": False, "path": str(path), "problem": "file not found"}
    try:
        age_days = round((time.time() - path.stat().st_mtime) / 86400, 1)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"present": False, "path": str(path), "problem": str(exc)}
    cookies = [ln for ln in lines if ln and not ln.startswith("#")]
    session_only = all(ln.split("\t")[-1].strip() == "" for ln in cookies) if cookies else False
    stale = age_days > 30
    return {
        "present": True,
        "path": str(path),
        "ageDays": age_days,
        "cookieCount": len(cookies),
        "sessionOnly": session_only,
        "stale": stale,
        "problem": (
            "cookies are session-only or older than 30 days; YouTube will bounce"
            if stale or session_only
            else None
        ),
        "shutilWhich": shutil.which("yt-dlp"),
    }


def _key_status(provider: str, names: tuple[str, ...]) -> dict[str, Any]:
    """Presence only. A key's VALUE is never read into a response or a log."""
    present = next((n for n in names if os.environ.get(n)), None)
    return {
        "provider": provider,
        "keyLoaded": bool(present),
        "source": present or "env",
        "problem": None if present else f"set one of {', '.join(names)}",
        "lastCheckedAt": int(time.time() * 1000),
        "aliases": list(names),
    }


def _build_health_slow() -> dict[str, Any]:
    """Every expensive block of the status card. Called off the request path."""
    import shutil

    from app import asr as _asr
    from app import hosting as _hosting

    out: dict[str, Any] = {}
    degraded: list[str] = []
    since = time.time()

    def block(name: str, fn) -> None:
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001 - one broken block, not a broken card
            degraded.append(name)
            out[name] = None
            logger.warning(f"[health] {name} block failed: {type(exc).__name__}: {exc}")

    def _tools() -> dict[str, Any]:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        try:
            import yt_dlp  # noqa: PLC0415

            ytdlp = getattr(getattr(yt_dlp, "version", None), "__version__", None)
        except Exception:  # noqa: BLE001 - not installed is a real state
            ytdlp = None
        return {
            "ffmpeg": _tool_version([ffmpeg, "-version"]) if ffmpeg else None,
            "ffprobe": ffprobe,
            "ytdlp": ytdlp,
            "mediaDir": str(settings.app.paths.outputs_dir),
            "downloadsDir": str(settings.app.paths.downloads_dir),
        }

    def _model() -> dict[str, Any]:
        info = _asr.model_status(settings)
        engine = _asr.status(settings)
        return {
            "whisper": {
                "model": info.get("model"),
                "compute": info.get("computeType"),
                "configuredModel": info.get("configured"),
                "engine": info.get("engine"),
                "isWhisperx": str(info.get("engine") or "").endswith("whisperx"),
                "label": info.get("label"),
                "sizeMb": info.get("sizeMb"),
                "localPresent": info.get("localPresent"),
                "bucketPresent": info.get("bucketPresent"),
                "mismatch": info.get("mismatch"),
            },
            "asr": {
                "engine": getattr(engine, "engine", None),
                "model": info.get("model"),
                "device": info.get("device"),
                "detail": info.get("label"),
                "estimatedRtf": getattr(engine, "estimated_rtf", None),
                "rtfSource": getattr(engine, "rtf_source", None),
            },
        }

    def _storage() -> dict[str, Any]:
        info = objectstore.status()
        ready = bool(info.get("clientReady"))
        # A listing that failed is UNKNOWN, not zero. Rendering an error as
        # "0.00 GB" made a healthy, model-filled bucket look wiped.
        unknown = bool(info.get("usageUnknown"))
        return {
            "enabled": bool(info.get("enabled")),
            "clientReady": ready,
            "bucket": info.get("bucket"),
            "endpoint": info.get("endpoint"),
            "prefix": info.get("prefix"),
            "deleteLocal": info.get("deleteLocal"),
            "verified": ready,
            "ready": ready,
            "hasAccessKey": info.get("hasAccessKey"),
            "hasSecret": info.get("hasSecret"),
            "accessKeyFirst4": info.get("accessKeyFirst4"),
            "usageBytes": None if unknown else info.get("usageBytes"),
            "usageGb": None if unknown else round((info.get("usageBytes") or 0) / 1024**3, 2),
            "budgetGb": round((info.get("budgetBytes") or 0) / 1024**3, 2),
            "usedPercent": (
                None
                if unknown
                else round(100 * (info.get("usageBytes") or 0) / info["budgetBytes"], 1)
                if info.get("budgetBytes")
                else None
            ),
        }

    def _dispatch() -> dict[str, Any]:
        probe = _hosting.remote_probe(settings)
        return {
            "enabled": bool(getattr(settings.hosted, "worker_url", "")),
            "workerUrl": getattr(settings.hosted, "worker_url", "") or None,
            "remote": probe,
            "detail": probe.get("detail"),
        }

    block("system", _hosting.system_stats)

    def _disk() -> dict[str, Any]:
        info = _hosting.disk_stats(settings.app.paths.data_dir, settings)
        try:
            info.update(diskreset.footprint(settings))
        except Exception as exc:  # noqa: BLE001 - the card must still render
            logger.debug(f"[health] disk footprint skipped: {exc}")
        return info

    block("disk", _disk)
    block("tools", _tools)
    # The model block heads the bucket when no local copy exists, so it runs on
    # its own slower clock (30 s) instead of every 2.5 s snapshot.
    def _model_cached() -> dict[str, Any]:
        global _MODEL_BLOCK, _MODEL_BLOCK_AT
        if _MODEL_BLOCK and (time.time() - _MODEL_BLOCK_AT) < 30:
            return _MODEL_BLOCK
        value = _model()
        _MODEL_BLOCK, _MODEL_BLOCK_AT = value, time.time()
        return value

    block("model", _model_cached)
    block("cookies", _cookie_state)
    def _storage_cached() -> dict[str, Any]:
        global _STORAGE_BLOCK, _STORAGE_BLOCK_AT
        if _STORAGE_BLOCK and (time.time() - _STORAGE_BLOCK_AT) < 20:
            return _STORAGE_BLOCK
        value = _storage()
        _STORAGE_BLOCK, _STORAGE_BLOCK_AT = value, time.time()
        return value

    block("storage", _storage_cached)
    # A remote worker probe is a tailnet round-trip; once every 20 s is plenty.
    def _dispatch_cached() -> dict[str, Any]:
        global _DISPATCH_BLOCK, _DISPATCH_BLOCK_AT
        if _DISPATCH_BLOCK and (time.time() - _DISPATCH_BLOCK_AT) < 20:
            return _DISPATCH_BLOCK
        value = _dispatch()
        _DISPATCH_BLOCK, _DISPATCH_BLOCK_AT = value, time.time()
        return value

    block("dispatch", _dispatch_cached)
    block("tailnetIp", _hosting.tailnet_ip)

    out["llm"] = _key_status("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    out["llmGroq"] = _key_status("groq", ("GROQ_API_KEY",))
    out["llmNvidia"] = _key_status("nvidia", ("NVIDIA_API_KEY",))
    out["llmDeepSeek"] = _key_status("deepseek", ("DEEPSEEK_API_KEY",))
    out["detection"] = {
        "provider": "deepseek",
        "label": "DeepSeek",
        "tradeoff": "hosted, fast; the key is the only thing sent to the provider",
        "options": [],
        "models": {},
        "transcriptionEngine": "faster-whisper",
        "ready": out["llmDeepSeek"]["keyLoaded"],
        "detail": None if out["llmDeepSeek"]["keyLoaded"] else "DEEPSEEK_API_KEY is not set",
    }
    out["degraded"] = degraded
    out["builtAt"] = int(time.time() * 1000)
    out["buildMs"] = int((time.time() - since) * 1000)
    return out


def _health_slow_loop() -> None:
    """Refresh the snapshot forever, one block at a time, never raising."""
    global _HEALTH_SLOW_AT
    while True:
        try:
            snap = _build_health_slow()
            with _HEALTH_LOCK:
                _HEALTH_SLOW.clear()
                _HEALTH_SLOW.update(snap)
                _HEALTH_SLOW_AT = time.time()
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            logger.warning(f"[health] snapshot refresh failed: {type(exc).__name__}: {exc}")
        time.sleep(HEALTH_SNAPSHOT_SEC)


def start_health_snapshot() -> dict[str, Any]:
    """Start the refresher once (idempotent: the boot phase may re-run)."""
    global _HEALTH_THREAD
    with _HEALTH_LOCK:
        if _HEALTH_THREAD is not None and _HEALTH_THREAD.is_alive():
            return {"detail": "already running"}
        # One synchronous build in the BOOT phase (budgeted), so the first
        # dashboard poll sees a complete card instead of nulls.
        try:
            first = _build_health_slow()
            _HEALTH_SLOW.clear()
            _HEALTH_SLOW.update(first)
            globals()["_HEALTH_SLOW_AT"] = time.time()
        except Exception as exc:  # noqa: BLE001 - the thread will retry
            logger.warning(f"[health] first snapshot failed: {type(exc).__name__}: {exc}")
        th = threading.Thread(target=_health_slow_loop, name="health-snapshot", daemon=True)
        _HEALTH_THREAD = th
        th.start()
    return {"detail": f"refresh every {HEALTH_SNAPSHOT_SEC}s"}

# `_boot()` runs at import time, so an exception escaping it makes the MODULE
# unimportable: uvicorn prints a traceback and exits before it ever binds
# 127.0.0.1:8000. Nothing in the browser can distinguish that from "the host is
# down" — the Vite proxy just answers 500 with an empty body, which is the
# "API unreachable" banner. Config errors still exit on purpose (SystemExit),
# but recovery/bookkeeping bugs must cost queued jobs, never the API.
try:
    _boot()
except SystemExit:
    raise
except Exception:  # noqa: BLE001 - availability beats bookkeeping
    logger.exception(
        "[boot] startup tasks failed — serving the API anyway "
        "(check the traceback above; jobs were left as they are)"
    )

app = FastAPI(
    title="ClipForge",
    version="1.1.0",
    description="Hosted video → viral clips pipeline (MoneyPrinterTurbo heritage).",
)


def _rss_mb(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


async def _resource_monitor() -> None:
    """Log API RSS, runner-child RSS and event-loop lag every 30 s.

    Event-loop lag is the number that separates "the API is busy" from "the API
    is hung": a starved loop is what makes /api/health time out while the
    process is technically alive, and the platform proxy turns exactly that
    into an empty 500. These lines land in data/logs/api.log.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    baseline = 0.0
    while True:
        loop.time()
        t0 = loop.time()
        await asyncio.sleep(30)
        lag = max(0.0, (loop.time() - t0) - 30.0)
        api_rss = _rss_mb(os.getpid())
        try:
            kids = worker.active_pids()
        except Exception:  # noqa: BLE001 - telemetry only
            kids = {}
        kid_rss = {jid: _rss_mb(pid) for jid, pid in kids.items()}
        logger.info(
            f"[monitor] api_rss={api_rss}MB loop_lag={lag:.2f}s "
            f"children={kid_rss or '{}'} uptime={int(time.time() - _STARTED_AT)}s"
        )
        if api_rss is not None and baseline and api_rss > baseline * 1.5 and api_rss - baseline > 200:
            logger.warning(
                f"[monitor] API RSS grew {baseline:.0f}MB → {api_rss:.0f}MB — "
                "heavy work in the API process would explain a proxy 500"
            )
        baseline = api_rss or baseline


@app.get("/api/health")
def health() -> dict:
    """The dashboard's status card, answered from a prebuilt snapshot (~1 ms).

    It must never walk the disk, spawn ffmpeg, call the worker daemon over a
    socket, head the bucket or round-trip to the tailnet on the REQUEST path:
    each of those has made the card "dead-looking" while the API was fine. The
    expensive blocks are refreshed every 2.5 s by `_health_slow_loop`; this
    merges them with the cheap live bits and never omits `worker`.
    """
    boot.mark_health()
    with _HEALTH_LOCK:
        slow = dict(_HEALTH_SLOW)
        age_ms = int((time.time() - _HEALTH_SLOW_AT) * 1000) if _HEALTH_SLOW_AT else None

    degraded = list(slow.get("degraded") or [])
    try:
        worker_state = worker.status()
    except Exception as exc:  # noqa: BLE001 - the card must render anyway
        degraded.append("worker")
        logger.warning(f"[health] worker status failed: {type(exc).__name__}: {exc}")
        worker_state = {
            "online": False,
            "currentJob": None,
            "queueDepth": 0,
            "activeCount": 0,
        }

    # Both live in the snapshot: the first disk read walks the data dir (60 ms
    # here), and a request-path walk is exactly what made this route time out.
    system = slow.get("system")
    disk = slow.get("disk")

    # warmup is the memory-only readiness record from _do_warmup(); map it onto
    # the stage/detail pair the card renders.
    current = str(warmup.get("current") or "")
    if warmup.get("reason"):
        warm_stage, warm_detail = "failed", str(warmup["reason"])
    elif current == "already downloaded":
        warm_stage, warm_detail = "ready", current
    else:
        warm_stage, warm_detail = "skipped", current or "unknown"

    payload: dict[str, Any] = {
        "ok": True,
        "degraded": sorted(set(degraded)),
        "version": getattr(config, "HOSTED_VERSION", "1.1.0"),
        "apiUptimeMs": int(time.time() * 1000 - HOST_STARTUP_MS),
        "appVersion": getattr(config, "HOSTED_VERSION", "1.1.0"),
        "worker": worker_state,
        "stallTimeoutSec": worker_state.get("stallTimeoutSec"),
        "warmup": {"stage": warm_stage, "detail": warm_detail,
                   "configured": warmup.get("configured"),
                   "model": warmup.get("model"), "note": warmup.get("note")},
        "system": system,
        "disk": disk,
        "limits": {
            "maxSourceMin": int(getattr(settings.hosted, "max_source_min", 120)),
            "retentionHours": int(getattr(settings.hosted, "retention_hours", 0)),
        },
        "whisper": (slow.get("model") or {}).get("whisper") or {
            "model": settings.whisper.model,
            "compute": settings.whisper.compute_type,
            "configuredModel": settings.whisper.model,
        },
        "asr": (slow.get("model") or {}).get("asr"),
        "tools": slow.get("tools"),
        "cookies": slow.get("cookies"),
        "storage": slow.get("storage"),
        "dispatch": slow.get("dispatch"),
        "detection": slow.get("detection"),
        "llm": slow.get("llm"),
        "llmGroq": slow.get("llmGroq"),
        "llmNvidia": slow.get("llmNvidia"),
        "llmDeepSeek": slow.get("llmDeepSeek"),
        "tailnetIp": slow.get("tailnetIp"),
        "snapshotAgeMs": age_ms,
        "boot": {
            "listenMs": boot.listen_ms,
            "firstHealthMs": boot.first_health_ms,
            "phasesDone": boot.done,
            "failedPhases": boot.summary()["failedPhases"],
        },
    }
    return payload


@app.get("/api/status")
def status() -> dict:
    """Per-dependency health, each probed separately with its own timeout.

    One slow dependency must not make the whole card red, and a failing one must
    name itself: the dashboard shows the list, the doctor asserts on it.
    """
    from app import doctor

    boot.mark_health()
    deps = doctor.dependency_status(settings, store, boot)
    return {
        "ok": all(d.get("ok") for d in deps),
        "boot": boot.summary(),
        "dependencies": deps,
        "fatal": last_failure(settings.app.paths.data_dir),
    }


@app.get("/api/doctor")
def doctor_report() -> dict:
    """The "Doctor" button: one pass/fail report, everything the UI needs.

    Same payload the CLI (tools/doctor.py) prints, so a report pasted from the
    dashboard and one from a terminal are the same document.
    """
    from app import doctor as doctor_mod
    from app import invariants

    report = doctor_mod.run_doctor(settings)
    report["dependencies"] = doctor_mod.dependency_status(settings, store, boot)
    report["boot"] = boot.summary()
    try:
        report["invariants"] = invariants.last_report() or invariants.check_and_repair(
            store, settings.app.paths
        )
    except Exception as exc:  # noqa: BLE001 - a report must still render
        report["invariants"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["ok"] = bool(report.get("ok")) and all(
        d.get("ok") for d in report["dependencies"]
    )
    return report


@app.get("/api/boot")
def boot_report() -> dict:
    """The boot phase table alone (used by tools/doctor.py)."""
    return boot.summary()


@app.on_event("startup")
async def _start_post_listen_work() -> None:
    """Open the port first, then do the heavy work on a daemon thread.

    uvicorn binds its socket only after lifespan startup returns, so nothing in
    this handler is allowed to block. Every phase has its own timeout and lands
    in /api/status; a failure is logged and shown, never fatal.
    """
    import asyncio

    asyncio.create_task(_resource_monitor())
    boot.start(_boot_phases)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Every unhandled exception elsewhere in the API returns JSON and is logged with its request id, so a 500 can never escape as HTML or a blank screen."""
    rid = request.headers.get("x-request-id") or uuid4().hex
    logger.exception(f"[api] {request.method} {request.url.path} id={rid} raised")
    _record_api_error(request, 500)
    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}: {exc}", "request_id": rid},
        headers={"x-request-id": rid},
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    rid = request.headers.get("x-request-id") or uuid4().hex
    logger.warning(f"[api] {request.method} {request.url.path} id={rid} validation error")
    _record_api_error(request, 422)
    return JSONResponse(
        status_code=422,
        content={"error": f"validation error: {exc.errors()}", "request_id": rid},
        headers={"x-request-id": rid},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # same-origin UI; permissive for API tooling
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "ETag"],
)


#: Last 50 non-2xx API calls, newest last. The browser only ever saw
#: "HTTP 400" — a code with no body, from a hop we do not control — and the
#: platform web-process log is the one place the real cause existed, which the
#: person hitting the bug cannot reach. This ring buffer puts that evidence
#: back inside the app: GET /api/errors/recent shows what the server actually
#: saw, including which hop looked like the caller.
_RECENT_ERROR_MAX = 50
_recent_api_errors: list[dict] = []


def _record_api_error(request: Request, status: int) -> None:
    _recent_api_errors.append({
        "at": int(time.time() * 1000),
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "userAgent": (request.headers.get("user-agent") or "")[:120],
        "origin": (request.headers.get("origin") or "")[:120],
        "contentLength": request.headers.get("content-length"),
    })
    if len(_recent_api_errors) > _RECENT_ERROR_MAX:
        del _recent_api_errors[: len(_recent_api_errors) - _RECENT_ERROR_MAX]


@app.middleware("http")
async def _log_errors(request: Request, call_next):
    """Log every non-2xx API response with its exception — the dashboard only
    shows "HTTP 400"-style codes, so the server must keep the real evidence."""
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"[api] {request.method} {request.url.path} raised")
        _record_api_error(request, 500)
        raise
    if response.status_code >= 400 and request.url.path.startswith("/api"):
        logger.warning(f"[api] {request.method} {request.url.path} -> {response.status_code}")
        _record_api_error(request, response.status_code)
    return response


# --------------------------------------------------------------------------- #
# tailscale-only gate (opt-in): the pipeline API is private to the tailnet
# --------------------------------------------------------------------------- #


def _is_tailnet_ip(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4 or parts[0] != "100":
        return False
    try:
        return 64 <= int(parts[1]) <= 127
    except ValueError:
        return False


@app.middleware("http")
async def _tailscale_gate(request: Request, call_next):
    if settings.hosted.tailscale_only:
        client = request.client.host if request.client else ""
        # loopback covers `tailscale serve` / local reverse proxies;
        # 100.64/10 covers direct tailnet connections.
        allowed = client in {"127.0.0.1", "::1", "localhost", "testclient"} or _is_tailnet_ip(client)
        if not allowed:
            logger.warning(f"[gate] rejected non-tailnet request from {client} {request.url.path}")
            return JSONResponse(
                {"detail": "ClipForge API is tailnet-only (TAILSCALE_ONLY=true) — connect over Tailscale."},
                status_code=401,
            )
    return await call_next(request)


@app.on_event("shutdown")
def _mark_interrupted() -> None:
    """Best-effort: record in-flight jobs as interrupted before we go away.

    The host can SIGKILL us at any moment, so this cannot be relied on — but
    when the runtime does grant a graceful stop, flagging the running jobs is
    what makes the next boot RESUME them instead of leaving them frozen at
    their last stage with no terminal state.
    """
    try:
        worker.mark_in_flight_interrupted()
    except Exception as exc:  # noqa: BLE001 — never block shutdown
        logger.warning(f"[shutdown] could not flag in-flight jobs: {exc}")


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #

def source_ident(url: str) -> str:
    """A stable identity for a video URL, ignoring per-visit tracking params.

    YouTube hands out ``?si=`` share tokens, so the same video arrives under a
    different string every time it is copied from the app. Comparing raw URLs
    therefore never matched, and every share created another batch.
    """
    try:
        parsed = urlsplit(str(url).strip())
    except ValueError:
        return str(url).strip().lower()
    host = (parsed.netloc or "").lower().removeprefix("www.").removeprefix("m.")
    path = parsed.path.rstrip("/")
    if host.endswith("youtu.be"):
        return f"youtube:{path.lstrip('/')}"
    if "youtube.com" in host:
        vid = parse_qs(parsed.query).get("v")
        if vid:
            return f"youtube:{vid[0]}"
        if path.startswith("/shorts/") or path.startswith("/embed/"):
            return f"youtube:{path.rsplit('/', 1)[-1]}"
    keep = {k: v for k, v in parse_qs(parsed.query).items() if k not in {"si", "feature", "t", "utm_source", "utm_medium", "utm_campaign"}}
    query = "&".join(f"{k}={v[0]}" for k, v in sorted(keep.items()))
    return f"{host}{path}" + (f"?{query}" if query else "")


DISK_PAUSE_BYTES = 800 * 1024 * 1024


def _require_disk_headroom() -> None:
    """Pause NEW submissions below 800 MB; never touch a running batch."""
    import shutil

    try:
        free = int(shutil.disk_usage(str(settings.app.paths.data_dir)).free)
    except OSError:
        return
    if free >= DISK_PAUSE_BYTES:
        return
    logger.warning(
        f"[disk] new batch paused: {free // 1048576} MB free "
        f"(< {DISK_PAUSE_BYTES // 1048576} MB)"
    )
    raise HTTPException(
        507,
        f"Waiting for disk space — only {free // 1048576} MB free. "
        "Press Reset disk or run tools/disk_reset.py --apply, then retry.",
    )


class CreateJobIn(BaseModel):
    sourceType: str = Field(pattern="^(url|upload)$")
    source: str = Field(min_length=1, max_length=2048)
    minClipSec: int | None = Field(default=None, ge=5, le=180)
    maxClipSec: int | None = Field(default=None, ge=10, le=180)
    maxClips: int | None = Field(default=None, ge=1, le=clipcount.HARD_CEILING)
    captionPreset: str | None = Field(default=None, pattern="^(bold_stack)$")
    language: str = Field(default="auto", pattern="^(auto|en|es)$")
    emojiEnabled: bool | None = None
    emojiDensity: str | None = Field(default=None, pattern="^(low|medium|high)$")


def _active_job_for(body: "CreateJobIn") -> dict | None:
    """The non-terminal job already running this exact source + settings."""
    want_source = source_ident(body.source)
    want = (
        body.minClipSec, body.maxClipSec, body.maxClips, body.language,
        body.captionPreset, body.emojiEnabled, body.emojiDensity,
    )
    terminal = {"complete", "failed", "cancelled", "deleting"}
    for other in store.list_jobs():
        if str(other.get("stage") or "") in terminal:
            continue
        if other.get("sourceType") != "url":
            continue
        if source_ident(str(other.get("source") or other.get("url") or "")) != want_source:
            continue
        opts = other.get("options") or {}
        have = (
            opts.get("minClipSec"), opts.get("maxClipSec"), opts.get("maxClips"),
            other.get("language"), other.get("captionPreset"),
            other.get("emojiEnabled"), other.get("emojiDensity"),
        )
        if all(w is None or h is None or str(w) == str(h) for w, h in zip(want, have)):
            return other
    return None


def _master_path(clip: dict[str, Any]) -> Path | None:
    """The clip's rendered master. Never goes through the share copy."""
    p = Path(clip.get("videoPath") or "")
    return p if clip.get("videoPath") and p.exists() else None


def _clip_media(clip: dict[str, Any], kind: str) -> Path | None:
    """Path to one of a clip's media files, or None when it does not exist.

    This used to return the SHARE copy for ``kind="video"``, and the share copy
    was built by calling this function again — infinite recursion, so every
    thumbnail and download of a shared clip answered HTTP 500 with
    ``RecursionError``. Each path now has exactly one meaning: this returns what
    the clip document names.
    """
    key = {"video": "videoPath", "thumb": "thumbPath"}.get(kind)
    if not key:
        return None
    p = Path(clip.get(key) or "")
    return p if p.exists() else None


def _clip_for_share_id(raw: str) -> dict[str, Any] | None:
    """Resolve a share id back to the clip document that owns it."""
    for clip in store.list_all_clips():
        if clip.get("shareId") == raw:
            return clip
    return None


def _clip_for_share_token(raw: str) -> dict[str, Any] | None:
    """Resolve a 7-day share token back to its clip."""
    for clip in store.list_all_clips():
        if clip.get("shareToken") == raw:
            return clip
    return None


def _media_etag(p: Path) -> str:
    """A stable etag from real bytes (size + mtime), so changed files 412."""
    try:
        return f"{p.stat().st_size}-{int(p.stat().st_mtime)}"
    except OSError:
        return ""


def _ascii_filename(text: str, fallback: str = "clip", ext: str = "mp4") -> str:
    """A header-safe download name (HTTP headers are latin-1 only).

    Titles are Spanish/emoji in this app, and interpolating one directly into
    Content-Disposition produced ``SER� CON`` on the wire. Starlette encodes
    Unicode filenames as RFC 5987 ``filename*``; doing it here keeps a plain
    ASCII name for old clients too.
    """
    import unicodedata

    raw = unicodedata.normalize("NFKD", str(text or ""))
    raw = raw.encode("ascii", "ignore").decode("ascii")
    clean = "".join(ch for ch in raw if ch.isalnum() or ch in " -_").strip()
    clean = "-".join(clean.split())[:60].strip("-_") or fallback
    return f"{clean}.{ext}"


def _bucket_key_for(clip: dict[str, Any]) -> str | None:
    """The bucket key of this clip's master, or None when unknown."""
    from app import objectstore

    raw = clip.get("videoPath")
    if not raw:
        return None
    return objectstore.key_for_path(raw, settings.app.paths.outputs_dir)


def _body_iterator(body: Any) -> Any:
    """Yield an S3 StreamingBody without ever holding the whole clip."""
    if hasattr(body, "iter_chunks"):
        try:
            for chunk in body.iter_chunks(chunk_size=256 * 1024):
                if chunk:
                    yield chunk
        finally:
            try:
                body.close()
            except Exception:  # noqa: BLE001 - a closed socket is not an error
                pass
    else:
        data = body.read() if hasattr(body, "read") else bytes(body)
        if data:
            yield data


def _bucket_media_response(
    clip: dict[str, Any],
    request: Request,
    *,
    filename: str | None = None,
    attachment: bool = False,
) -> Response | None:
    """Serve a clip master from the bucket with real Range support.

    The old version passed the whole ``Request`` object as the S3 Range header
    and then handed the ``(status, headers, body)`` tuple to StreamingResponse
    as if it were an iterator. The result was a 200 with no body and Range
    ignored — previews and downloads both failed while the route said OK.
    """
    from app import objectstore

    key = _bucket_key_for(clip)
    if not key:
        return None
    range_header = request.headers.get("range")
    opened = objectstore.open_stream(key, range_header)
    if opened is None:
        return None
    status, headers, body = opened
    headers = dict(headers)
    headers["Accept-Ranges"] = "bytes"
    headers["Content-Type"] = "video/mp4"
    headers["Cache-Control"] = MEDIA_CACHE_CONTROL
    if filename:
        disposition = "attachment" if attachment else "inline"
        headers["Content-Disposition"] = f'{disposition}; filename="{filename}"'
    if request.method == "HEAD":
        try:
            total = int(headers.get("Content-Length") or 0)
            if total and range_header and status == 200:
                headers["Content-Length"] = str(total)
        finally:
            try:
                body.close()
            except Exception:  # noqa: BLE001
                pass
        return Response(status_code=status, headers=headers, media_type="video/mp4")
    return StreamingResponse(
        _body_iterator(body),
        status_code=status,
        headers=headers,
        media_type="video/mp4",
    )


def _clip_bucket_stream(
    clip: dict[str, Any], request: Request, filename: str
) -> Response | None:
    """Stream a clip's master from the bucket when local media was reclaimed."""
    if clip.get("videoPath") and Path(clip["videoPath"]).exists():
        return None
    return _bucket_media_response(clip, request, filename=filename, attachment=True)


def _restore_media(p: Path) -> bool:
    """Pull one reclaimed source/media file back from the bucket."""
    from app import objectstore
    if not p.exists():
        key = objectstore.key_for_path(str(p), settings.app.paths.outputs_dir)
        if not key:
            return False
        if not objectstore.download_to(key, str(p)):
            return False
    return True



def _share_copy_for(clip: dict) -> Path | None:
    """Where this clip's share-sized MP4 lives, building it if needed."""
    master = _master_path(clip)
    if master is None:
        return None
    try:
        return _JOB_SHARE.get(master)
    except Exception as exc:
        # A transient path/lock issue must never turn a share request into a
        # 503: degrade to the link, and let the lock reap the copy later.
        logger.warning(f"[share] _share_copy_for failed for {clip.get('_id')}: {exc}")
        return None


@app.api_route("/s/{share_id}.mp4", methods=["GET", "HEAD"])
def public_share_mp4(share_id: str, request: Request) -> Response:
    """The public MP4 for a share id.

    Public on purpose: no login, no expiry, no session. That is the whole
    point of a share link, and it is why it serves the share-SIZED copy
    (720x1280, a few MB) rather than the 1080x1920 master.

    Range requests come from Starlette's FileResponse, which answers `Range:`
    with 206 + Content-Range. CORS is explicit because these bytes are fetched
    by pages on other origins — an embed, a social unfurler — and the
    app-wide middleware already allows any origin for /api but not for /s.
    """
    raw = share_id[:-4] if share_id.lower().endswith(".mp4") else share_id
    # New links use a 7-day token; old links use the 10-char share id. Both
    # resolve to public bytes, and the token branch still enforces revocation.
    try:
        clip, _tok = _resolve_public_share(raw)
    except HTTPException as exc:
        raise exc
    share = _share_copy_for(clip)
    if share is None or not share.exists():
        # Older links predate the bucket-backed share route. They must not
        # stay on "Preparing" forever: stream the master with Range instead.
        return _share_stream_response(clip, request)
    return FileResponse(
        share,
        media_type="video/mp4",
        headers={
            "Cache-Control": SHARE_CACHE_CONTROL,
            "Accept-Ranges": "bytes",
            "ETag": _media_etag(share),
            **_SHARE_CORS,
        },
    )


def _og_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_SHARE_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "Range, Content-Type",
    "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
}


def _share_stream_response(
    clip: dict[str, Any], request: Request, *, attachment: bool = False
) -> Response:
    """Public clip bytes, local or bucket-backed, with Range and CORS."""
    filename = _ascii_filename(
        f"{clip.get('order', 0) + 1:02d}-{clip.get('title') or 'clip'}"
    )
    p = _clip_media(clip, "video")
    if p is not None:
        return FileResponse(
            p,
            media_type="video/mp4",
            filename=filename,
            content_disposition_type="attachment" if attachment else "inline",
            headers={
                "Cache-Control": SHARE_CACHE_CONTROL,
                "Accept-Ranges": "bytes",
                "ETag": _media_etag(p),
                **_SHARE_CORS,
            },
        )
    streamed = _bucket_media_response(
        clip, request, filename=filename, attachment=attachment
    )
    if streamed is None:
        raise HTTPException(404, "this clip's video is no longer available")
    streamed.headers.update(_SHARE_CORS)
    streamed.headers["Cache-Control"] = SHARE_CACHE_CONTROL
    return streamed


@app.api_route("/s/{token}", methods=["GET", "HEAD"])
def public_share_token(token: str, request: Request) -> Response:
    """The public share page /s/<token>. No login, no cookie, no expiry leak.

    This is an HTML page with Open Graph and Twitter Card tags because that is
    what a social app fetches when the link is pasted. The playable bytes are
    at /s/<token>.mp4, not here: serving MP4 at this URL made every social
    unfurl show a bare URL (and made the copied link look like a download).

    Older 10-character share ids still resolve to the same page; their direct
    MP4 remains /s/<id>.mp4.
    """
    clip, tok = _resolve_public_share(token)
    return _share_page_html(clip, request, token=tok)


@app.api_route("/s/{token}/download", methods=["GET", "HEAD"])
def public_share_download(token: str, request: Request) -> Response:
    """The download option on a share link (token or legacy share id)."""
    clip, _tok = _resolve_public_share(token)
    return _share_stream_response(clip, request, attachment=True)


def _public_origin(request: Request) -> str:
    """The origin a recipient will actually use.

    The API is reached through the Vite/production proxy, so ``request.base_url``
    can describe the upstream socket rather than the browser origin. Honour the
    standard forwarded headers first and fall back to the Host header. A copied
    link must never contain 127.0.0.1 when the page came from a public domain.
    """
    proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
    host = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
    if not host:
        host = (request.headers.get("host") or request.url.netloc or "").strip()
    if not proto:
        proto = request.url.scheme
    return f"{proto}://{host}".rstrip("/")


def _resolve_public_share(raw: str) -> tuple[dict[str, Any], str | None]:
    """Resolve a token or a legacy share id, enforcing token expiry/revocation."""
    token_clip = _clip_for_share_token(raw)
    if token_clip is not None:
        state = sharelink.token_state(token_clip)
        if state["state"] != "ready":
            raise HTTPException(410, state["detail"])
        return token_clip, raw
    legacy = _clip_for_share_id(raw)
    if legacy is not None:
        return legacy, None
    raise HTTPException(404, "this share link does not exist")


def _share_page_html(
    clip: dict[str, Any], request: Request, *, token: str | None = None,
) -> HTMLResponse:
    """A public, self-contained share page with the tags social apps read.

    This is the URL people paste into X/WhatsApp/Facebook. The raw MP4 lives at
    a separate ``.mp4`` URL so a crawler gets HTML metadata here instead of
    immediately downloading bytes. Every URL is absolute: a relative
    ``og:video`` is ignored by most unfurlers.
    """
    base = _public_origin(request)
    sid = token or str(clip.get("shareId") or "")
    if not sid:
        raise HTTPException(404, "this clip has no share link")
    page_url = f"{base}/s/{sid}"
    video_url = f"{base}/s/{sid}.mp4"
    download_url = f"{base}/s/{sid}/download"
    embed_url = f"{base}/s/{sid}/embed"
    thumb_path = clip.get("thumbPath")
    image_url = f"{base}/api/clips/{clip['_id']}/thumb" if thumb_path else ""
    title = _og_escape(clip.get("title") or clip.get("hookText") or "ClipForge clip")
    hook = _og_escape(clip.get("hookText") or clip.get("title") or "Made with ClipForge")
    w = int(clip.get("mediaWidth") or 1080)
    h = int(clip.get("mediaHeight") or 1920)
    secure = f'<meta property="og:video:secure_url" content="{_og_escape(video_url)}">' \
        if video_url.startswith("https://") else ""
    image_tags = ""
    if image_url:
        image_tags = (
            f'<meta property="og:image" content="{_og_escape(image_url)}">'
            f'<meta name="twitter:image" content="{_og_escape(image_url)}">'
        )
    tags = "".join([
        '<meta property="og:type" content="video.other">',
        '<meta property="og:site_name" content="ClipForge">',
        f'<meta property="og:title" content="{title}">',
        f'<meta property="og:description" content="{hook}">',
        f'<meta property="og:url" content="{_og_escape(page_url)}">',
        f'<meta property="og:video" content="{_og_escape(video_url)}">',
        f'<meta property="og:video:url" content="{_og_escape(video_url)}">',
        secure,
        '<meta property="og:video:type" content="video/mp4">',
        f'<meta property="og:video:width" content="{w}">',
        f'<meta property="og:video:height" content="{h}">',
        '<meta name="twitter:card" content="player">',
        f'<meta name="twitter:title" content="{title}">',
        f'<meta name="twitter:description" content="{hook}">',
        f'<meta name="twitter:player" content="{_og_escape(embed_url)}">',
        f'<meta name="twitter:player:width" content="{w}">',
        f'<meta name="twitter:player:height" content="{h}">',
        f'<meta name="twitter:player:stream" content="{_og_escape(video_url)}">',
        '<meta name="twitter:player:stream:content_type" content="video/mp4">',
        image_tags,
    ])
    poster = f' poster="{_og_escape(image_url)}"' if image_url else ""
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="canonical" href="{_og_escape(page_url)}">
{tags}
<style>
 body{{margin:0;background:#0b0b0d;color:#f4f4f5;font:16px/1.5 system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      min-height:100vh;gap:18px;padding:24px;text-align:center}}
 video{{max-width:min(420px,92vw);max-height:78vh;border-radius:14px;background:#000}}
 a{{color:#7dd3a0}}
 .download{{display:inline-block;padding:10px 16px;border-radius:999px;background:#f4f4f5;color:#111;text-decoration:none;font-weight:700}}
 small{{color:#a1a1aa}}
</style></head>
<body>
  <video src="{_og_escape(video_url)}" controls playsinline preload="metadata"{poster}></video>
  <div>
    <p style="font-size:18px;font-weight:700;margin:0 0 6px">{title}</p>
    <p style="color:#a1a1aa;margin:0 0 14px">{hook}</p>
    <a class="download" href="{_og_escape(download_url)}">Download MP4</a>
  </div>
  <small>Made with ClipForge</small>
</body></html>"""
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "public, max-age=300",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/s/{token}/page")
def public_share_token_page(token: str, request: Request) -> HTMLResponse:
    """Legacy page URL; serves the same canonical share page."""
    clip, tok = _resolve_public_share(token)
    return _share_page_html(clip, request, token=tok)


@app.get("/s/{token}/embed")
def public_share_embed(token: str, request: Request) -> HTMLResponse:
    """Minimal iframe player used by Twitter's player card."""
    clip, tok = _resolve_public_share(token)
    base = _public_origin(request)
    sid = tok or str(clip.get("shareId") or "")
    video_url = f"{base}/s/{sid}.mp4"
    poster_path = clip.get("thumbPath")
    poster = f"{base}/api/clips/{clip['_id']}/thumb" if poster_path else ""
    poster_attr = f' poster="{_og_escape(poster)}"' if poster else ""
    return HTMLResponse(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<style>html,body,video{margin:0;width:100%;height:100%;background:#000}</style>"
        f"</head><body><video src=\"{_og_escape(video_url)}\" controls playsinline "
        f"autoplay muted loop{poster_attr}></video></body></html>",
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.api_route("/s/{token}/thumb", methods=["GET", "HEAD"])
def public_share_thumb(token: str) -> Response:
    """Public poster for a share page; no login required for the unfurler."""
    clip, _tok = _resolve_public_share(token)
    resp = clip_thumb(str(clip["_id"]))
    resp.headers.update(_SHARE_CORS)
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp


@app.get("/s/{share_id}")
def public_share_page(share_id: str, request: Request) -> HTMLResponse:
    """A public page for a share id, with the tags a social app unfurls.

    When a link is pasted into X, WhatsApp or Slack, the app fetches this page
    and reads the og:* tags. Without them the link unfurls as a bare URL.

    The absolute base comes from the request, so the tags are correct on the
    generated *.freebuff.dev domain, on localhost, and behind any proxy —
    there is no configured public origin to drift out of date.
    """
    clip = _clip_for_share_id(share_id)
    if clip is None:
        raise HTTPException(404, "this share link does not exist")
    base = str(request.base_url).rstrip("/")
    vid = f"{base}/s/{clip['shareId']}.mp4"
    thumb_path = clip.get("thumbPath")
    img = f"{base}/api/clips/{clip['_id']}/thumb" if thumb_path else ""
    title = _og_escape(clip.get("title") or clip.get("hookText") or "ClipForge clip")
    hook = _og_escape(clip.get("hookText") or clip.get("title") or "")
    w = int(clip.get("mediaWidth") or 1080)
    h = int(clip.get("mediaHeight") or 1920)

    tags = [
        '<meta property="og:type" content="video.other">',
        f'<meta property="og:title" content="{title}">',
        f'<meta property="og:description" content="{hook}">',
        f'<meta property="og:url" content="{base}/s/{clip["shareId"]}">',
        f'<meta property="og:video" content="{vid}">',
        '<meta property="og:video:type" content="video/mp4">',
        f'<meta property="og:video:width" content="{w}">',
        f'<meta property="og:video:height" content="{h}">',
        '<meta name="twitter:card" content="player">',

        f'<meta name="twitter:player" content="{vid}">',
        f'<meta name="twitter:player:width" content="{w}">',
        f'<meta name="twitter:player:height" content="{h}">',
        f'<meta name="twitter:title" content="{title}">',
    ]
    if img:
        tags.append(f'<meta property="og:image" content="{img}">')
        tags.append(f'<meta name="twitter:image" content="{img}">')
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{chr(10).join(tags)}
<style>
 body{{margin:0;background:#0b0b0d;color:#f4f4f5;font:16px/1.5 system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      min-height:100vh;gap:18px;padding:24px;text-align:center}}
 video{{max-width:min(420px,92vw);max-height:78vh;border-radius:14px;background:#000}}
 a{{color:#7dd3a0}}
 small{{color:#a1a1aa}}
</style></head>
<body>
  <video src="{vid}" controls playsinline preload="metadata"
         poster="{img}"></video>
  <div>
    <p style="font-size:18px;font-weight:700;margin:0 0 6px">{title}</p>
    <p style="color:#a1a1aa;margin:0 0 10px">{hook}</p>
    <a href="{vid}" download>Download MP4</a>
  </div>
  <small>Made with ClipForge</small>
</body></html>"""
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": SHARE_CACHE_CONTROL,
            "Access-Control-Allow-Origin": "*",
        },
    )


class DiskResetIn(BaseModel):
    apply: bool = False
    scope: str = "local"  # local | bucket | all


@app.post("/api/storage/reset")
def storage_reset(body: DiskResetIn | None = None) -> dict:
    """Dry-run by default; `{"apply": true}` performs the reset.

    ``scope=local`` is the existing protects-everything disk reset,
    ``scope=bucket`` plans/applies only provably dead bucket objects, and
    ``all`` runs both. Nothing is deleted without ``apply``. The bucket plan
    is computed once and handed to apply (it carries its exact key list).
    """
    apply = bool(body and body.apply)
    scope = str((body.scope if body else "local") or "local").lower()
    if scope not in {"local", "bucket", "all"}:
        raise HTTPException(400, "scope must be local, bucket or all")
    try:
        local = None
        bucket = None
        if scope in {"local", "all"}:
            local = diskreset.reset(store, settings, apply=apply)
            local["scope"] = "local"
            # Unlinked-but-open bytes only return when the holder exits. The
            # worker is the only long-lived child that holds media, and the
            # supervisor owns its restart; do it only when the scan found one.
            if apply and local.get("openDeleted"):
                try:
                    local["workerRestart"] = worker.restart()
                except Exception as exc:  # noqa: BLE001 - report, do not fail reset
                    local["workerRestart"] = {"ok": False, "error": str(exc)}
        if scope in {"bucket", "all"}:
            from app import bucketreset

            planned = bucketreset.plan(store, settings)
            if apply:
                bucket = bucketreset.apply(store, settings, planned)
            else:
                planned.pop("_keys", None)
                planned.pop("_multipart", None)
                bucket = planned
        if scope == "local":
            return local or {}
        if scope == "bucket":
            return bucket or {}
        return {
            "ok": bool((local or {}).get("ok")) and bool((bucket or {}).get("ok")),
            "scope": "all",
            "dryRun": not apply,
            "local": local,
            "bucket": bucket,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - report the real reason
        logger.warning(f"[storage] reset failed: {type(exc).__name__}: {exc}")
        raise HTTPException(503, f"could not reset {scope}: {exc}") from exc


@app.get("/api/dev/verify-clips")
def verify_clips(jobId: str) -> dict:
    """Prove every rendered clip of one batch is really playable.

    The render path already runs `video.verify_output` on each encode, but that
    proof dies with the process that ran it: a later reboot cannot tell a good
    clip from a 0-byte remnant. This exposes the SAME check on demand — bytes,
    a decodable video stream, 1080x1920, a duration that matches the planned
    cut, and a decoded picture that is not black.

    Media that was offloaded is pulled to a temp file first, so the answer does
    not depend on what is left on local disk. Read-only: nothing here edits or
    re-renders a clip.
    """
    import shutil

    job = store.get_job(jobId) or {}
    if not job:
        raise HTTPException(404, "job not found")
    clips = store.list_clips(jobId)
    out: list[dict[str, Any]] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="verify-", dir=settings.app.paths.data_dir))
    try:
        for clip in clips:
            clip_id = str(clip.get("_id") or "")
            order = clip.get("order")
            result: dict[str, Any] = {
                "clipId": clip_id,
                "order": order if isinstance(order, int) else None,
                "title": clip.get("title") or "",
                "valid": False,
                "reason": "",
            }
            path = Path(str(clip.get("videoPath") or ""))
            try:
                if not (path.exists() and path.stat().st_size > 0):
                    # Offloaded: pull the object back just for the probe, then
                    # delete the copy. Never leave a second local copy behind.
                    key = objectstore.key_for_path(path, settings.app.paths.outputs_dir)
                    local = tmp_root / (path.name or f"{clip_id}.mp4")
                    if not key or not objectstore.materialize(key, local):
                        result["reason"] = "media is neither on disk nor in the bucket"
                        out.append(result)
                        continue
                    path = local
                expected = None
                try:
                    start, end = float(clip.get("start") or 0), float(clip.get("end") or 0)
                    if end > start > 0:
                        expected = end - start
                except (TypeError, ValueError):
                    expected = None
                meta = video_svc.verify_output(path, expected_sec=expected)
                result.update({
                    "valid": True,
                    "durationSec": round(float(meta.get("duration") or 0.0), 3),
                    "width": int(meta.get("width") or 0),
                    "height": int(meta.get("height") or 0),
                    "bytes": path.stat().st_size,
                })
            except video_svc.RenderError as exc:
                result["reason"] = str(exc)
            except Exception as exc:  # noqa: BLE001 - one bad clip must not 500 the report
                result["reason"] = f"{type(exc).__name__}: {exc}"
            out.append(result)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    valid = sum(1 for r in out if r["valid"])
    return {
        "total": len(out),
        "valid": valid,
        "invalid": len(out) - valid,
        "clips": out,
    }


@app.get("/api/clips/health")
def clips_health() -> dict:
    """The last cheap contract pass over every ready clip."""
    from app import cliphealth

    return cliphealth.report()


@app.get("/api/storage/unused")
def storage_unused() -> dict:
    """Bucket media whose batch no longer exists, grouped per batch.

    These batches cannot appear anywhere in the dashboard — their job documents
    are gone, so the page has nothing to list — yet their source and clips still
    count against the plan. Reads the bucket once and compares it to the live
    job ids. Deletes nothing.
    """
    try:
        return reclaim.unused_batches(store)
    except Exception as exc:  # noqa: BLE001 - the card must render anyway
        logger.warning(f"[storage] unused scan failed: {type(exc).__name__}: {exc}")
        raise HTTPException(503, f"could not read the bucket: {exc}") from exc


class DeleteUnusedIn(BaseModel):
    jobIds: list[str] = Field(default_factory=list)


@app.post("/api/storage/unused/delete")
def storage_unused_delete(body: DeleteUnusedIn | None = None) -> dict:
    """Delete unused bucket media. Ids are re-checked against live jobs.

    The client may name the batches it was shown, or send nothing for "every
    batch that is unused right now". Either way the server re-derives the live
    set at delete time, so a stale list can never remove a batch that has just
    been re-queued.
    """
    ids = [str(j) for j in ((body.jobIds if body else []) or [])] or None
    try:
        return reclaim.delete_unused_batches(store, ids)
    except Exception as exc:  # noqa: BLE001 - report, never half-answer
        logger.warning(f"[storage] unused delete failed: {type(exc).__name__}: {exc}")
        raise HTTPException(503, f"could not delete unused media: {exc}") from exc


@app.get("/api/clips")
def list_clips(jobId: str = "") -> list[dict]:
    """The clips grid's data source — one job's clips, or one clip by id.

    This route was MISSING while the dashboard's grid fetched it on every
    render, so a finished batch showed an empty grid (404) even though its clip
    documents and media were fine. ``jobId`` is the contract the frontend uses;
    a clip id is also accepted because one caller passes an id in that slot.
    """
    if not jobId:
        return _attach_share_state(store.list_clip_docs_for_list())
    clips = store.list_clips(jobId)
    if not clips:
        single = store.get_clip_doc(jobId)
        if single:
            clips = [single]
    return _attach_share_state(clips)


@app.post("/api/clips/{clip_id}/share")
def create_share_link(clip_id: str, request: Request) -> dict:
    """Mint a public, expiring, revocable share link and return it NOW.

    This is deliberately not a transcode. The old flow waited for a separate
    share-sized MP4 to be built and left the button on "Preparing…" forever
    when that build never ran (which is every offloaded clip). The public
    route streams the master directly from the bucket with Range, so a link is
    valid the instant this returns.
    """
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    if clip.get("renderState") != "done" or not clip.get("videoPath"):
        raise HTTPException(409, "this clip has no rendered video yet")
    token = sharelink.ensure_token(clip)
    clip["shareState"] = sharelink.STATE_READY
    store.write_clip(clip)
    base = _public_origin(request)
    expires = int(clip.get("shareExpiresAt") or 0)
    return {
        "ok": True,
        "token": token,
        # The canonical URL to paste into a social app: an HTML page with OG
        # tags. `videoUrl` is the playable Range endpoint; downloadUrl forces
        # an attachment.
        "url": f"{base}/s/{token}",
        "pageUrl": f"{base}/s/{token}",
        "videoUrl": f"{base}/s/{token}.mp4",
        "downloadUrl": f"{base}/s/{token}/download",
        "expiresAt": expires,
        "expiresInDays": 7,
        "revocable": True,
        "shareState": "ready",
    }


@app.api_route("/api/clips/{clip_id}/share", methods=["GET", "HEAD"])
def clip_share_bytes(clip_id: str, request: Request) -> Response:
    """The MP4 behind the Share button's file path (same bytes as preview)."""
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    filename = _ascii_filename(
        f"{clip.get('order', 0) + 1:02d}-{clip.get('title') or 'clip'}"
    )
    p = _clip_media(clip, "video")
    if p is not None:
        return FileResponse(
            p,
            media_type="video/mp4",
            filename=filename,
            content_disposition_type="inline",
            headers={
                "Cache-Control": MEDIA_CACHE_CONTROL,
                "Accept-Ranges": "bytes",
                "ETag": _media_etag(p),
            },
        )
    streamed = _bucket_media_response(clip, request, filename=filename)
    if streamed is not None:
        return streamed
    raise HTTPException(404, "clip file missing from disk and bucket")


@app.delete("/api/clips/{clip_id}/share")
def revoke_share_link(clip_id: str) -> dict:
    """Revoke a share link immediately; the token is dead on the next request."""
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    token = clip.get("shareToken")
    if not token:
        return {"ok": True, "revoked": False, "detail": "no share link to revoke"}
    clip["shareRevoked"] = True
    clip["shareExpiresAt"] = 0
    clip["shareState"] = "revoked"
    store.write_clip(clip)
    return {"ok": True, "revoked": True, "detail": "share link revoked"}


@app.post("/api/clips/{clip_id}/share-link")
def ensure_share_link(clip_id: str) -> dict:
    """Mint/verify this clip's public link and report its real state.

    Kept for older clients. It now mints the same 7-day token the new
    ``/api/clips/{id}/share`` route uses and reports ready as soon as the
    master is streamable — no separate share transcode, no endless
    "Preparing…".
    """
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    token = sharelink.ensure_token(clip)
    clip["shareState"] = sharelink.STATE_READY
    store.write_clip(clip)
    tok = sharelink.token_state(clip)
    return {
        **sharelink.share_state(clip, _share_copy_for(clip)),
        "shareState": "ready" if tok["state"] == "ready" else tok["state"],
        "shareDetail": tok["detail"],
        "shareReady": tok["state"] == "ready",
        "shareToken": token,
        "shareUrl": f"/s/{token}",
        "shareVideoUrl": f"/s/{token}.mp4",
        "sharePageUrl": f"/s/{token}",
        "shareExpiresAt": tok.get("expiresAt"),
    }


def _attach_share_state(clips: list[dict]) -> list[dict]:
    """Add the share facts to each clip, without building anything.

    A clip's master is streamable from disk OR the bucket; a share copy is an
    optional upgrade, not a prerequisite. Saying "preparing" until a separate
    transcode exists left every offloaded clip on that label forever. The
    token is the source of truth now, and the URL is valid immediately.
    """
    out = []
    for clip in clips:
        tok = sharelink.token_state(clip)
        has_media = bool(clip.get("videoPath")) and (
            Path(str(clip.get("videoPath"))).exists()
            or bool(_bucket_key_for(clip))
        )
        state = "ready" if tok["state"] == "ready" and has_media else tok["state"]
        if tok["state"] == "ready" and not has_media:
            state = "failed"
            tok["detail"] = "the rendered file is missing from disk and bucket"
        token = tok.get("token")
        sid = token or (str(clip.get("shareId") or "") or None)
        share_page = f"/s/{sid}" if sid else None
        out.append({
            **clip,
            **sharelink.share_state(clip, None),
            "shareState": state if state in {"ready", "preparing", "failed"} else "missing",
            "shareDetail": tok["detail"],
            "shareReady": state == "ready",
            "shareToken": token,
            # `/s/<id>` is now an HTML page with OG tags. The playable MP4 has
            # its own explicit URL; do not hand a raw MP4 to a social app.
            "shareUrl": share_page,
            "sharePageUrl": share_page,
            "shareVideoUrl": f"/s/{sid}.mp4" if sid else None,
            "shareExpiresAt": tok.get("expiresAt"),
        })
    return out

def _attach_clip_captions(store: LocalStore, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge the rendered caption cues for one clip doc into the response.

    The detection pass writes ``captionCues: []`` as a placeholder and the
    render stage writes the real cues to a per-clip ``captions/cues_000.json``
    sidecar. The clips list endpoint has been returning the placeholder, so
    every card's subtitle line was empty while the burn-in itself was fine.
    Read the sidecar and merge it in (the least-complete document wins the id,
    so a pending twin may legitimately still hold the placeholder).
    """
    cues = doc.get("captionCues") or []
    if cues and cues[0] != "":
        return cues
    job_id = str(doc.get("jobId") or "")
    order = doc.get("order")
    if not job_id or not isinstance(order, int):
        return cues
    # ONLY this clip's own sidecar. The old loop walked cues_000..003 and
    # returned the first that existed, so after a re-order or a partial
    # workspace clip 1 could display clip 0's captions — worse than none.
    cue_path = store._paths.outputs_dir / job_id / "captions" / f"cues_{order:03d}.json"
    if not cue_path.exists():
        return cues
    try:
        payload = json.loads(cue_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return cues
    owner = payload.get("clipId")
    if owner and str(owner) != str(doc.get("_id")):
        return cues
    merged = payload.get("cues") or []
    if merged and merged[0] != "":
        return merged
    return cues

def _list_clips_with_captions(store: LocalStore, job_id: str | None, limit: int = 96, with_cues: bool = False) -> list[dict]:
    """Return clip metadata with the rendered caption cues attached.

    ``list_clips_for_list`` is the canonical list; for the all-jobs path we
    deduplicate by (job, order) and let the caption merger handle the rest.
    The ``with_cues`` flag repeats the per-clip JSON sidecar lookup that the
    single-clip endpoint already does, so the shared preview URL gets the same
    cue shape the detail dialog reads. The job-scoped path is the one the
    dashboard grid is actually watching, so it runs the sidecar merge on every
    request in that hot path regardless of the flag.
    """
    base = store.list_clips_for_list(job_id) if job_id else store.list_all_clips(limit=limit)
    out = []
    seen: set[tuple[str, int]] = set()
    for doc in base:
        key = (str(doc.get("jobId") or ""), int(doc.get("order") or 0))
        if key in seen:
            continue
        seen.add(key)
        merged = {**doc}
        merged["captionCues"] = _attach_clip_captions(store, doc)
        out.append(merged)
    return out


@app.api_route("/api/clips/{clip_id}/thumb", methods=["GET", "HEAD"])
def clip_thumb(clip_id: str) -> FileResponse:
    """Real first frame of the rendered clip — inline image, no generated art.

    Posters written before thumbnails were sized were full-resolution 1080x1920
    JPEGs of 50-70 KB, one per clip. Rewritten here on first request rather
    than by a migration script, so every batch that already exists gets small
    posters the moment anyone looks at it, and the rewrite is a single ffmpeg
    call guarded by a file-size test.
    """
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "thumbnail not available")
    p = _clip_media(clip, "thumb")
    if p is None:
        # Local media is reclaimed to the bucket after verification, so a
        # missing poster is usually "archived", not "never made". Pull it back
        # (and, failing that, the master) before giving up.
        if clip.get("thumbPath"):
            _restore_media(Path(clip["thumbPath"]))
            p = _clip_media(clip, "thumb")
    if p is None:
        # No poster on disk: make one from the clip rather than showing a black
        # card. This is also what heals posters rendered before the
        # non-black-frame selection existed.
        master = _master_path(clip)
        if master is None and clip.get("videoPath"):
            if _restore_media(Path(clip["videoPath"])):
                master = _master_path(clip)
        if master is None:
            raise HTTPException(404, "thumbnail file missing")
        p = Path(clip["thumbPath"]) if clip.get("thumbPath") else (
            settings.app.paths.outputs_dir / str(clip.get("jobId"))
            / f"clip_{int(clip.get('order') or 0):03d}.thumb.jpg"
        )
        try:
            video_svc.render_first_frame_thumb(master, p)
            store.write_clip(
                str(clip["_id"]), {"thumbPath": str(p)}, job_id=str(clip.get("jobId") or "") or None
            )
        except Exception as exc:  # noqa: BLE001 - a poster is never worth a 500
            raise HTTPException(404, f"thumbnail could not be created: {exc}") from exc
    master_for_thumb = _master_path(clip)
    if master_for_thumb is not None and video_svc.thumb_needs_rewrite(p):
        try:
            video_svc.render_first_frame_thumb(master_for_thumb, p)
        except Exception as exc:  # noqa: BLE001 - a poster is never worth a 500
            logger.warning(f"[thumb] could not rewrite {p.name} small: {exc}")
    return FileResponse(
        p,
        media_type="image/jpeg",
        headers={
            "Cache-Control": MEDIA_CACHE_CONTROL,
            "ETag": _media_etag(p),
        },
    )


@app.api_route("/api/clips/{clip_id}/video", methods=["GET", "HEAD"])
def clip_video(clip_id: str, request: Request) -> Response:
    """The clip the preview plays — same origin, Range, inline.

    This route did not exist: the dashboard's ``<video src>`` asked for
    ``/api/clips/{id}/video`` and got a 404, so every preview was a black box
    while downloads worked. Serving from the bucket here also means a
    reclaimed local file is streamed, never downloaded back to the 4 GB disk.
    """
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    filename = _ascii_filename(f"{clip.get('order', 0) + 1:02d}-{clip.get('title') or 'clip'}")
    p = _clip_media(clip, "video")
    if p is not None:
        return FileResponse(
            p,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Cache-Control": MEDIA_CACHE_CONTROL,
                "Accept-Ranges": "bytes",
                "ETag": _media_etag(p),
            },
        )
    streamed = _bucket_media_response(clip, request, filename=filename)
    if streamed is not None:
        return streamed
    raise HTTPException(404, "clip file missing from disk and bucket")


@app.api_route("/api/clips/{clip_id}/download", methods=["GET", "HEAD"])
def clip_download(clip_id: str, request: Request) -> Response:
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    filename = _ascii_filename(
        f"{clip.get('order', 0) + 1:02d}-{clip.get('title') or 'clip'}"
    )
    streamed = _clip_bucket_stream(clip, request, filename=filename)
    if streamed is not None:
        return streamed
    p = _clip_media(clip, "video")
    if p is None:
        raise HTTPException(404, "clip file missing on disk")
    return FileResponse(
        p,
        media_type="video/mp4",
        filename=filename,
        content_disposition_type="attachment",
        headers={
            "Cache-Control": MEDIA_CACHE_CONTROL,
            "Accept-Ranges": "bytes",
            "ETag": _media_etag(p),
        },
    )


# --------------------------------------------------------------------------- #
# "Post to..." -- purely additive. Nothing in this section can affect, delay or
# block a download; the existing MP4/SRT/captions.txt paths above are untouched.
# --------------------------------------------------------------------------- #

#: CSRF state -> the PKCE verifier it was minted with. In-process and
#: short-lived by design: a verifier is only needed for the few seconds between
#: the redirect out and the callback back, and keeping it out of a file means
#: there is no on-disk secret to leak or clean up.
_OAUTH_PENDING: dict[str, dict] = {}
_OAUTH_TTL_SEC = 600
_oauth_lock = threading.Lock()


def _prune_oauth(now: float) -> None:
    stale = [k for k, v in _OAUTH_PENDING.items() if now - v["createdAt"] > _OAUTH_TTL_SEC]
    for key in stale:
        _OAUTH_PENDING.pop(key, None)


@app.get("/api/social/accounts")
def social_accounts() -> dict:
    """Per-platform readiness + the approval reality, for the Settings panel.

    Returns no tokens -- only whether an account is connected and what it is
    called. The UI uses ``gate`` to decide whether to offer Connect, explain the
    approval requirement, or fall back to download.
    """
    with _oauth_lock:
        _prune_oauth(time.time())
    return {"platforms": describe_all(settings.social, accounts)}


@app.post("/api/social/{slug}/connect")
def social_connect(slug: str) -> dict:
    if slug not in ("youtube", "x", "tiktok", "instagram"):
        raise HTTPException(404, f"unknown platform {slug!r}")
    cfg = settings.social
    mod = PLATFORMS[slug]
    status, reason = mod.readiness(cfg, accounts)
    if status == "needs_app":
        # 409, not 404: the platform is real, the credentials are what is missing.
        raise HTTPException(409, reason)
    state = social_base.new_state()
    verifier, _ = social_base.pkce_pair()
    with _oauth_lock:
        _prune_oauth(time.time())
        _OAUTH_PENDING[state] = {
            "slug": slug,
            "verifier": verifier,
            "createdAt": time.time(),
        }
    url = mod.authorize_url(
        cfg, redirect_uri=redirect_uri(cfg, slug), state=state, code_verifier=verifier
    )
    return {"authUrl": url, "state": state}


@app.get("/api/social/{slug}/callback")
def social_callback(slug: str, request: Request) -> RedirectResponse:
    """OAuth redirect target. Lands the user back in Settings either way."""
    dest = f"{settings.social.public_url.rstrip('/')}/?settings=1&social={slug}"
    params = dict(request.query_params)
    code = params.get("code")
    state = params.get("state", "")
    if params.get("error"):
        return RedirectResponse(f"{dest}&social_error={params['error']}")
    with _oauth_lock:
        pending = _OAUTH_PENDING.pop(state, None)
    if not pending or pending["slug"] != slug:
        # No/expired/mismatched state: treat as a CSRF failure, do not exchange.
        return RedirectResponse(f"{dest}&social_error=invalid_state")
    mod = PLATFORMS[slug]
    try:
        record = mod.exchange_code(
            settings.social,
            code=code or "",
            redirect_uri=redirect_uri(settings.social, slug),
            code_verifier=pending["verifier"],
        )
    except SocialError as exc:
        return RedirectResponse(f"{dest}&social_error={str(exc)[:160]}")
    accounts.save(slug, record)
    return RedirectResponse(f"{dest}&social_ok={slug}")


@app.post("/api/social/{slug}/disconnect")
def social_disconnect(slug: str) -> dict:
    if slug not in ("youtube", "x", "tiktok", "instagram"):
        raise HTTPException(404, f"unknown platform {slug!r}")
    removed = accounts.delete(slug)
    return {"disconnected": removed, "platforms": describe_all(settings.social, accounts)}


def social_error_status(exc: Exception) -> int:
    """Map a SocialError to the HTTP status the UI branches on.

    The three cases need different words in the toast, so they get different
    statuses: "needs an app / needs review" (offer download), "reconnect
    account" (offer a connect button), and "the platform refused this clip"
    (offer an edit). Collapsing them into one 500 would make all three read as
    the same useless failure.
    """
    if isinstance(exc, NotConfigured):
        return 409
    if isinstance(exc, NeedsReconnect):
        return 401
    if isinstance(exc, PostRejected):
        return 422
    return 502


class PostClipBody(BaseModel):
    platform: str
    caption: str | None = None
    title: str = ""
    privacy: str = ""


@app.post("/api/clips/{clip_id}/post")
def clip_post(clip_id: str, body: PostClipBody) -> dict:
    """Publish one rendered clip to one connected platform.

    Status mapping is deliberate so the UI can say something true:
      409 not configured  -> "needs an app / needs review", offer download
      401 needs reconnect -> "Failed -- reconnect account"
      422 rejected       -> "the platform refused this clip"
    """
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    if not clip.get("videoPath") or not Path(clip["videoPath"]).exists():
        raise HTTPException(409, "this clip has no rendered MP4 to post yet")
    if body.platform not in ("youtube", "x", "tiktok", "instagram"):
        raise HTTPException(404, f"unknown platform {body.platform!r}")

    # The caption is the same short line + hashtag block the clipboard copies.
    from app.caption_format import build_post_caption

    caption = (
        body.caption.strip()
        if body.caption is not None
        else build_post_caption(clip.get("hookText", ""), clip.get("hashtags", []))
    )
    try:
        return social_post(
            body.platform,
            settings.social,
            accounts,
            video_path=clip["videoPath"],
            caption=caption,
            title=body.title or clip.get("title", ""),
            privacy=body.privacy,
        )
    except SocialError as exc:
        raise HTTPException(social_error_status(exc), str(exc)) from exc


@app.post("/api/clips/{clip_id}/retry-render")
def retry_clip_render_route(clip_id: str, force: bool = False) -> dict:
    """Re-render ONE clip (the "Retry clip" button).

    ``force=true`` re-renders a done clip too, which is how a caption-text fix
    reaches an existing deliverable without re-running detection. Light enough
    to run inline: it renders a single clip from the cached cut and caption
    cues, and the result is pushed to the bucket before returning.
    """
    from app import video

    if store.get_clip_doc(clip_id) is None:
        raise HTTPException(404, "clip not found")
    return video.retry_clip_render(clip_id, force=force)


@app.api_route("/api/clips/{clip_id}/srt", methods=["GET", "HEAD"])
def clip_srt(clip_id: str) -> Response:
    """Download the clip's SRT with a real attachment filename.

    This used to be ``text/plain`` with no Content-Disposition, so a phone
    either rendered it inline or saved it as ``srt``. Subtitle players look at
    the extension, and the download button advertises a .srt file.
    """
    from app import subtitle as subtitle_svc

    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    lines: list[str] = []
    for i, cue in enumerate(clip.get("captionCues", []), 1):
        text = subtitle_svc.clean_caption_text(str(cue.get("text") or ""))
        if not text:
            continue
        lines.append(
            f"{len(lines) + 1}\n{_srt_time(cue['start'])} --> {_srt_time(cue['end'])}\n{text}\n"
        )
    filename = _ascii_filename(
        f"{clip.get('order', 0) + 1:02d}-{clip.get('title') or 'clip'}", ext="srt"
    )
    return Response(
        "\n".join(lines),
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/clips/{clip_id}/tracks/{filename}")
def clip_track(clip_id: str, filename: str) -> FileResponse:
    """Download one translated caption track (.srt/.vtt) for a clip.

    The requested name is MATCHED against the paths this clip recorded, never
    joined onto a directory — so a crafted filename like ``../../.env`` cannot
    escape the output folder. Only files the pipeline itself wrote are served.
    """
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    tracks = clip.get("translationTracks") or []
    if not tracks:
        raise HTTPException(
            404,
            "no caption tracks for this clip — enable [translate] in config.toml and re-render",
        )
    wanted = Path(filename).name
    for track in tracks:
        for ext, recorded in (track.get("files") or {}).items():
            if Path(recorded).name != wanted:
                continue
            p = Path(recorded)
            if not _restore_media(p):
                raise HTTPException(404, f"track file missing on disk: {p.name}")
            media = "text/vtt" if ext == "vtt" else "application/x-subrip"
            return FileResponse(
                p, media_type=media, filename=p.name,
                headers={"Content-Disposition": f'attachment; filename="{p.name}"'},
            )
    if any(t.get("error") for t in tracks):
        raise HTTPException(404, f"caption track '{wanted}' failed to translate for this clip")
    raise HTTPException(404, f"no such caption track on this clip: {wanted}")


def _srt_time(sec: float) -> str:
    ms = max(0, round(sec * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"



@app.get("/api/jobs")
def list_jobs() -> dict:
    """Every durable batch. `GET /api/jobs/{id}` gives one up to the client."""
    return {"jobs": [_describe_job(j) for j in _job_list()]}


#: Batches one bulk-delete request may name. The dashboard chunks its list at
#: this size; the cap keeps a malformed request from turning into a very long
#: synchronous loop of document writes.
BULK_DELETE_MAX = 200


class BulkDeleteIn(BaseModel):
    jobIds: list[str] = Field(default_factory=list)


@app.get("/api/jobs/junk")
def junk_jobs() -> dict:
    """Batches that settled without delivering a single clip.

    The dashboard's "Clean up" dialog has always asked for this route, but the
    route did not exist: the request fell through to `/api/jobs/{job_id}`, was
    read as a job called "junk", and answered a fabricated `queued` record.
    The dialog therefore showed nothing and the button could never work.

    Declared BEFORE `/api/jobs/{job_id}` on purpose: FastAPI matches routes in
    definition order, and the other way round would keep swallowing it. The
    classification matches what the dialog can safely delete (`complete` with
    no clips, or `failed`); a batch that produced clips is never listed here.
    """
    out: list[dict[str, Any]] = []
    for raw in _job_list():
        job = _describe_job(raw)
        status = str(job.get("status") or "")
        try:
            clips = int(job.get("clipCount") or 0)
        except (TypeError, ValueError):
            clips = 0
        if status != "failed" and not (status == "complete" and clips == 0):
            continue
        reason = (
            str(job.get("error") or job.get("failedStage") or "failed")
            if status == "failed"
            else "no clips rendered"
        )
        out.append({
            "_id": job["_id"],
            "title": job.get("title") or job.get("source") or job["_id"],
            "stage": job.get("stage") or status,
            "status": status,
            "clipCount": clips,
            "createdAt": job.get("createdAt"),
            "reason": reason[:120],
        })
    return {"jobs": out}


@app.post("/api/jobs/bulk-delete")
def bulk_delete_jobs(body: BulkDeleteIn) -> dict:
    """Mark many settled batches for deletion in one call.

    This is the mass path behind "Clean up dead batches". `require_finished=True`
    is the safety that matters: without it a stale dashboard could mark a batch
    that was still downloading, which cancelled a live worker and removed 35 of
    its files mid-run. Each id is checked on its own, so one bad entry cannot
    fail the whole sweep, and anything refused is named in the answer instead of
    disappearing silently.

    Deletion itself stays asynchronous (app/purge.py): this only marks the
    batches, exactly like the single DELETE route.
    """
    ids = [str(j) for j in (body.jobIds or [])][:BULK_DELETE_MAX]
    deleted: list[str] = []
    missing: list[str] = []
    refused: list[dict[str, str]] = []
    for job_id in ids:
        if not purge.valid_job_id(job_id):
            missing.append(job_id)
            continue
        try:
            marked = purge.mark(store, job_id, require_finished=True)
        except purge.JobNotDeletable as exc:
            refused.append({"jobId": job_id, "reason": str(exc)})
            continue
        except purge.PurgeError as exc:
            refused.append({"jobId": job_id, "reason": str(exc)})
            continue
        if marked is None:
            missing.append(job_id)
        else:
            deleted.append(job_id)
    if deleted:
        logger.info(f"[jobs] bulk delete marked {len(deleted)} batch(es) for purge")
    return {
        "deleting": deleted,
        "deleted": deleted,
        "missing": missing,
        "refused": refused,
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _describe_job(job)


@app.post("/api/jobs")
def create_job(body: CreateJobIn) -> dict:
    """Create and queue a new batch (idempotent for an identical source+settings).

    This is the API behind the "New batch" button. `POST /api/jobs/upload`
    stays the only way to ingest a file, and `/api/jobs/{id}/retry` stays the
    only way to re-run a failed batch.
    """
    _require_disk_headroom()
    # One ACTIVE batch per video + settings. A finished/failed batch already has
    # its rendered clips on disk, so it is not an active run and must not steal
    # a new submission for the same video.
    duplicate = _active_job_for(body)
    if duplicate is not None:
        logger.info("[jobs] %s is already being processed as %s — returning that job",
                    source_ident(body.source), duplicate.get("_id"))
        return {"jobId": duplicate["_id"], "title": duplicate.get("title"),
                "duration": duplicate.get("durationSec"), "deduped": True}

    job = store.create_job(
        source_type=body.sourceType,
        source=body.source,
        title=body.source.split("?")[0][:80] or "Untitled clip",
        min_clip_sec=body.minClipSec,
        max_clip_sec=body.maxClipSec,
        max_clips=body.maxClips,
        caption_preset=body.captionPreset,
        language=body.language,
        duration=None,
    )
    store.append_job_log(job["_id"], f"job created ({body.sourceType}: {body.source[:80]})")
    store.write_job(job)
    worker.enqueue(job["_id"])
    return {"jobId": job["_id"], "title": job.get("title"), "duration": job.get("durationSec")}


@app.post("/api/jobs/upload")
async def upload_job(
    file: UploadFile = File(...),
    captionPreset: str | None = Form(None),
    maxClips: int | None = Form(None),
    minClipSec: int | None = Form(None),
    maxClipSec: int | None = Form(None),
    language: str = Form("auto"),
) -> dict:
    """Create a batch from an ATTACHED file.

    The dashboard posts multipart here (api.uploadJob); the route had been lost,
    so every "Attach file" submission answered 404 while URL batches worked.
    The bytes are streamed to `<uploads>/<job_id>/source.<ext>` — the exact path
    the pipeline's upload branch resolves — with the same size cap as a URL
    source, and a failed write never leaves a half job behind.
    """
    _require_disk_headroom()
    import shutil

    name = Path(file.filename or "").name
    # No suffix is NOT "probably mp4": guessing stored /etc/hostname as an .mp4
    # and failed minutes later inside ffmpeg instead of at the door.
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            415,
            f"unsupported file type {ext or '(none)'!r}; use one of "
            f"{', '.join(sorted(ALLOWED_UPLOAD_EXT))}",
        )

    cap_mb = int(getattr(settings.hosted, "max_source_mb", 0) or 0)
    cap = cap_mb * 1024 * 1024 if cap_mb > 0 else 0

    # Stream to a STAGING directory first, and only create the job document
    # once the bytes are complete. Persisting the job first meant a slow or
    # aborted upload left a queued job with no file, which boot recovery would
    # adopt and fail ("the uploaded file is no longer on the server").
    staging = settings.app.paths.uploads_dir / f".staging-{uuid4().hex}"
    staged = staging / f"source{ext}"
    try:
        staging.mkdir(parents=True, exist_ok=True)
        written = 0
        with staged.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if cap and written > cap:
                    raise HTTPException(413, f"file exceeds the {cap_mb} MB limit")
                out.write(chunk)
        if written == 0:
            raise HTTPException(400, "the uploaded file is empty")
    except HTTPException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001 - report write failures honestly
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(500, f"could not store the upload: {type(exc).__name__}: {exc}") from exc

    job = store.create_job(
        source_type="upload",
        source=name,
        title=Path(name).stem[:80] or "Untitled clip",
        min_clip_sec=minClipSec,
        max_clip_sec=maxClipSec,
        max_clips=maxClips,
        caption_preset=captionPreset,
        language=language or "auto",
        duration=None,
    )
    job_id = job["_id"]
    dest_dir = settings.app.paths.uploads_dir / job_id
    try:
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(dest_dir)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        store.delete_job(job_id)
        raise HTTPException(500, f"could not place the upload: {exc}") from exc

    job["bytes"] = written
    store.append_job_log(job_id, f"upload stored: {name} ({written} bytes)")
    store.write_job(job)
    worker.enqueue(job_id)
    return {"jobId": job_id, "file": name, "bytes": written}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    if not purge.valid_job_id(job_id):
        raise HTTPException(400, "invalid job id")
    marked = purge.mark(store, job_id, keep_source=False)
    if marked is None:
        raise HTTPException(404, "job not found")
    return {
        "deleting": True,
        "jobId": job_id,
        "status": purge.STATUS,
        "keptSource": False,
    }


@app.post("/api/jobs/{job_id}/verify-duration")
def verify_job_duration(job_id: str) -> dict:
    """Re-measure a job's source from yt-dlp and ffprobe.

    Fixes the old 120:00 card for a 20:37 video without re-fetching media.
    """
    from app import source_stream

    if store.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    try:
        return source_stream.reconcile_source_duration(store, job_id)
    except Exception as exc:  # noqa: BLE001 - report the real reason
        raise HTTPException(400, f"could not verify the source duration: {exc}") from exc


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    """Resume a failed batch from its last checkpointed stage + cached clips."""
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.get("stage") != "failed":
        raise HTTPException(400, "only failed jobs can be retried")
    dead_end = worker_mod.unrecoverable_reason(store, job)
    if dead_end is not None:
        job.update({"stage": "failed", "error": dead_end, "errorDetail": None,
                    "interrupted": False, "interruptReason": None})
        store.write_job(job)
        raise HTTPException(400, dead_end)
    try:
        depth = worker.retry_from_cache(job_id)
    except KeyError:
        raise HTTPException(404, "job not found") from None
    return {"queued": True, "queueDepth": depth}


@app.get("/api/jobs/{job_id}/captions.txt")
def captions_txt(job_id: str) -> Response:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    text = job.get("captionsTxt") or ""
    return Response(text, media_type="text/plain")


class ZipRequest(BaseModel):
    clipIds: list[str] = Field(min_length=1, max_length=32)


# Export used to build the whole archive in an io.BytesIO and return
# buf.getvalue() — which also copies it. A full batch is 10 clips at ~80MB, so
# a single "Export all" held ~800MB resident, and the 32-clip request ceiling
# implied 2.5GB. This host has 2.0 GiB total and a Whisper model loaded, so
# exporting a batch could OOM the server and take every running job with it.
# The archive is now written to disk and streamed, and oversized requests are
# refused with an honest limit instead of being attempted.
MAX_EXPORT_BYTES = 1_500_000_000  # 1.5 GB of source MP4s


@app.post("/api/export/zip")
def export_zip(body: ZipRequest):
    """ZIP of REAL rendered MP4s + captions.txt — no metadata-only exports."""
    import zipfile

    members: list[tuple[Path | None, str, str]] = []
    total = 0
    skipped: list[str] = []
    for cid in body.clipIds:
        clip = store.get_clip(cid)
        if clip is None or not clip.get("videoPath"):
            skipped.append(cid)
            continue
        p = Path(clip["videoPath"])
        key = objectstore.key_for_path(p, settings.app.paths.outputs_dir)
        if p.exists() and p.stat().st_size > 0:
            size = p.stat().st_size
        elif key:
            # In the bucket and NOT pulled back: `_restore_media` here would
            # refill the disk this export is supposed to leave alone. The bytes
            # go bucket -> ZIP directly, one chunk at a time.
            meta = objectstore.head(key)
            if meta is None:
                skipped.append(cid)
                continue
            size = int(meta.get("size") or 0)
        else:
            skipped.append(cid)
            continue
        if total + size > MAX_EXPORT_BYTES:
            raise HTTPException(
                413,
                f"Export is {total // (1024 * 1024)} MB of MP4s and the next clip "
                f"would pass the {MAX_EXPORT_BYTES // (1024 * 1024)} MB limit — "
                "export fewer clips at a time, or download them individually.",
            )
        total += size
        title = str(clip.get("title") or "clip")[:40]
        members.append((p if p.exists() else None, f"clips/{int(clip.get('order', 0)) + 1:02d}-{title}.mp4", key or ""))

    if not members:
        raise HTTPException(404, "no rendered clips to export")

    first_clip = store.get_clip(body.clipIds[0]) if body.clipIds else None
    job_id = first_clip["jobId"] if first_clip else None
    job = store.get_job(job_id) if job_id else None

    # ZIP_STORED: H.264 is already compressed, so deflating would burn CPU for
    # ~0% gain. The archive itself is spooled (memory up to 32 MB, then disk) so
    # the clip files are never staged: each one is either read from local disk
    # or streamed straight out of the bucket into the archive.
    spool = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024, suffix=".zip")
    try:
        with zipfile.ZipFile(spool, "w", zipfile.ZIP_STORED) as zf:
            for src, arcname, key in members:
                if src is not None:
                    zf.write(src, arcname)
                    continue
                opened = objectstore.open_stream(key) if key else None
                if opened is None:
                    skipped.append(arcname)
                    continue
                _status, _headers, body = opened
                chunks = body.iter_chunks(256 * 1024) if hasattr(body, "iter_chunks") else [body.read()]
                with zf.open(arcname, "w") as dst:
                    for chunk in chunks:
                        dst.write(chunk)
            zf.writestr("captions.txt", (job or {}).get("captionsTxt") or "")
            zf.writestr(
                "README.txt",
                "ClipForge export — 9:16 1080x1920 H.264/AAC\n"
                "Only clip content you own or have permission to use.\n",
            )
        spool.seek(0, os.SEEK_END)
        archive_size = spool.tell()
        spool.seek(0)
        return StreamingResponse(
            iter(lambda: spool.read(256 * 1024), b""),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="clipforge-export.zip"',
                "Content-Length": str(archive_size),
            },
        )
    except BaseException:
        spool.close()
        raise


def _unlink_quietly(path: Path) -> None:
    """Remove the temp export once it has been streamed to the client."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("[export] could not remove temp archive %s", path)


# --------------------------------------------------------------------------- #
# static UI (built frontend) — mounted last so /api wins
# --------------------------------------------------------------------------- #

static_dir = Path(__file__).resolve().parent.parent / "webui"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="webui")
else:
    @app.get("/")
    def root() -> dict:
        return {
            "name": "ClipForge API",
            "docs": "/docs",
            "ui": "The hosted dashboard is served by the platform's web process.",
        }


#: One share copy per job, built the first time anything in the job asks for
#: it, so several clips of the same batch never re-encode the same video.
_JOB_SHARE = share_video.JobShareLocker()


def main() -> None:
    """Entry point for the `clipforge` console script (pyproject.toml)."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        log_level="info",
    )
