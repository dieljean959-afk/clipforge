"""ClipForge local API — FastAPI backend + in-process worker, one process.

Endpoints serve the React dashboard and the pipeline artifacts on ./downloads
and ./outputs. Boot sequence: config validation → ffmpeg check → Whisper model
pre-download (explicit, with progress) → worker thread starts.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, HTTPException, Request, UploadFile
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
from starlette.background import BackgroundTask
from loguru import logger
from pydantic import BaseModel, Field

from app import (
    caption_style,
    clipcount,
    download,
    eta,
    hosting,
    hostlimits,
    jobstatus,
    objectstore,
    purge,
    reclaim,
    share_video,
    sharelink,
    subtitle,
)
from app import download as download_mod
from app import encode as encode_mod
from app.hosting import disk_stats
from app import llm as llm_mod


from uuid import uuid4 as uuid4


SHARE_CACHE_CONTROL = "public, max-age=31536000, immutable"
MEDIA_CACHE_CONTROL = "public, max-age=300"
_SHARE_CACHE_CONTROL = "public, max-age=31536000, immutable"
MEDIA_CACHE_CONTROL = "public, max-age=300"
_MIN_FREE_BYTES = 100 * 1024 * 1024  # 100 MB free-disk floor before HTTP 507


_STARTED_AT = time.time()


from app.config import (
    S3_CREDENTIAL_VARS,
    ConfigError,
    Settings,
    boot_banner,
    delete_env_var,
    ffmpeg_bin,
    ffprobe_bin,
    live_deepseek_key,
    live_gemini_key,
    live_groq_key,
    live_nvidia_key,
    load_config,
    sync_deepseek_env_after_save,
    sync_deepseek_env_for_delete,
    sync_gemini_env_after_save,
    sync_gemini_env_for_delete,
    sync_groq_env_after_save,
    sync_groq_env_for_delete,
    sync_nvidia_env_after_save,
    sync_nvidia_env_for_delete,
    sync_s3_env_after_save,
    sync_s3_env_for_delete,
    write_env_var,
    delete_env_var,
)
from app.store import LocalStore
from app.social.accounts import AccountStore
from app.hosting import RetentionCleaner
from app import subtitle as subtitle_svc
from app import video as video_svc
from app import hosting as hosting_mod
from app import social as social_mod
from app.social import PLATFORMS, describe_all, describe, redirect_uri
from app.social import base as social_base
from app.social import post_video as social_post
from app.config import _LiveKeyState
from app.social.base import (
    NeedsReconnect,
    NotConfigured,
    PostRejected,
    SocialError,
)
from app.stages import StageWorkspace, remove_workspace
from app import disksweep
from app.slices.cleanup import unlink_model_dirs
from starlette.background import BackgroundTask
# Module handle, distinct from the `worker` INSTANCE below. The boot recovery
# pass needs `unrecoverable_reason` / `INTERRUPT_MAX_RESUMES` from the module;
# without this import that pass raised NameError at import time and uvicorn
# exited before binding port 8000, so the proxy answered a bodiless 500.
from app import worker as worker_mod
from app.worker import InProcessWorker
from app.workerd import WorkerSupervisor


def _purge_remote_job(job_id: str) -> None:
    """Flag one in-flight big-job for deletion so the sync worker finishes it.
    (Placeholder used by the purge boot path; the real delete runs in purge.py.)
    """
    pass


def _job_list(store=None):
    """List jobs the same way _active_job_for does (durable records, sorted)."""
    from app import store as _store
    return sorted(
        [j for j in (_store.list_jobs() if store is None else store.list_jobs())],
        key=lambda j: j.get("createdAt") or 0,
        reverse=True,
    )


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
    if started["minted"]:
        logger.info(
            f"[share] minted {started['minted']} share link(s) for existing clips"
        )

    def _finish() -> None:
        try:
            rest = sharelink.backfill(store, limit=2000)
            if rest["minted"]:
                logger.info(
                    f"[share] backfilled {rest['minted']} more share link(s) "
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


def _boot() -> None:
    global settings, store, worker, retention, accounts
    try:
        settings = load_config()
    except ConfigError as exc:
        logger.error(f"[boot] {exc}")
        raise SystemExit(2) from exc
    boot_banner(settings)

    # fail fast on missing ffmpeg/ffprobe
    logger.info(f"[boot] ffmpeg: {ffmpeg_bin()}")
    logger.info(f"[boot] ffprobe: {ffprobe_bin()}")

    store = LocalStore(settings.app.paths)
    # LOCKED_SPEC §4: the pipeline lives in its own process. The supervisor
    # spawns that daemon (nice 15), adopts it if the API itself was restarted,
    # and restarts it if it dies. The API only proxies control calls.
    worker = WorkerSupervisor(settings, store)
    retention = RetentionCleaner(settings, store)
    accounts = AccountStore(settings.app.paths.data_dir)

    # --- durable logs + native-crash tracebacks ---------------------------
    # The platform only keeps the web process's stdout for the life of that
    # process, so a traceback from the failure itself is gone by the time anyone
    # looks. Write both the app log and the faulthandler dump to files that
    # survive a restart, under the data dir (never inside the watched app/).
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

    # --- crash recovery ---------------------------------------------------
    # Four things are recoverable, and they are not the same thing:
    #   1. mid-flight — the process died before it could write a terminal state,
    #      so the job is simply frozen at its last stage.
    #   2. interrupted — we knew a host restart killed it.
    #   3. a FAILURE whose message is a SIGNAL banner, i.e. recorded as failure
    #      before the interrupted/resumed distinction existed. Those jobs are
    #      sitting at stage=failed with an intact stage cache, and used to be
    #      lost permanently.
    #   4. a job whose clips are ALL on disk. Handled first and separately: it
    #      is not resumable work, it is a record that is simply wrong (see
    #      below), and re-queueing it would burn another detection call to
    #      re-render clips that already exist.
    # Bounded by INTERRUPT_MAX_RESUMES so a job that cannot survive a restart
    # loop eventually stops instead of cycling forever.
    # Share ids are minted at clip creation, so this only mints for clips that
    # predate the feature. It is cheap (no transcoding) and it runs BEFORE the
    # job reconciliation below so a restored batch already has its links.
    try:
        _share_backfill()
    except Exception as exc:  # noqa: BLE001 - never block boot
        logger.warning(f"[boot] share-link backfill skipped: {exc}")

    recoverable = {"grabbing", "transcribing", "finding", "rendering", "queued"}
    # Jobs a runner is busy with RIGHT NOW. Recovery must not touch these: this
    # loop rewrites the job to stage=queued / progress=0 and enqueues it again,
    # so every API restart turned a live 15-clip export into "Queued… 0%" for
    # as long as that stage went between progress writes, and left a duplicate
    # queue entry that could never start while the only worker slot was still
    # busy with that very job. Ask the daemon which pids are alive, don't guess
    # from the stage word: "exporting" looks exactly like a dead job.
    try:
        running_now = set(worker.active_pids() or {})
    except Exception:  # noqa: BLE001 - no daemon answer means nothing of ours is running
        running_now = set()
    for job in store.list_jobs():
        if job.get("_id") in running_now:
            continue
        # A job can be stamped stage=failed by a LATER attempt after a good
        # batch was already delivered — e.g. redetect re-running detection,
        # getting rate-limited, and overwriting a finished job. The clips are
        # on disk, so the record is the thing that is wrong. Restoring
        # stage=complete here (rather than leaving it red) is the difference
        # between "here are your 10 clips" and a permanent phantom failure
        # with no action the user can take.
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
            continue
        # A finished batch can still carry `interrupted: true` from an earlier
        # attempt that was killed and then resumed successfully. Re-queueing on
        # that stale flag flipped a COMPLETE batch back to "queued" after every
        # restart and sent it through the pipeline again for no output.
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
        # A job we cannot possibly resume must not be re-queued. Its upload is
        # gone, so every restart spent a full pipeline launch rediscovering
        # that and failing again — the reported "resuming (attempt 2)" loop.
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
            # Clear the false terminal state so the resume can proceed; the
            # stage markers on disk, not this field, decide where it restarts.
            job.update({
                "error": None,
                "completedAt": None,
                "interrupted": True,
                "interruptReason": "Recovered after an external kill — resuming from cached stages.",
            })
            store.write_job(job)
        # Now in EVERY recovery path the job is waiting its turn like any other
        # queued job, so say so. Leaving `stage` at the stage it died in made the
        # dashboard render a job that was merely in the queue as "Stage 3 of 4
        # — Detecting, 99%", indistinguishable from real work: the stage markers
        # on disk (not this field) decide where the runner actually restarts.
        job.update({
            "stage": "queued",
            "progress": 0,
            "progressDetail": "Queued — resuming from cached stages after a restart.",
            "completedAt": None,
            # The attempt that died left its error behind. A resumed job is not
            # failing any more, and a queued job advertising "AI detection
            # failed: HTTP 429" is a contradiction the dashboard would repeat.
            "error": None,
            "errorDetail": None,
            # Unit counters describe work in the CURRENT attempt. Left over
            # from the run that died, a queued job rendered "10 of 10 clips"
            # next to a 0% progress bar.
            "unitLabel": None,
            "unitsDone": None,
            "unitsTotal": None,
        })
        store.write_job(job)
        logger.info(f"[boot] re-queuing job {job['_id']} ({reason}, was {was})")
        worker.enqueue(job["_id"])

    # Whisper model pre-download happens in a background thread so /health
    # responds immediately while the model loads.
    threading.Thread(target=_do_warmup, name="clipforge-warmup", daemon=True).start()


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


@app.on_event("startup")
async def _start_resource_monitor() -> None:
    import asyncio

    asyncio.create_task(_resource_monitor())

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


@app.on_event("startup")
def _start_worker() -> None:
    # The worker daemon is a separate process now: if it cannot come up, the
    # API must still serve (every route reports the worker as offline, and the
    # supervisor keeps retrying). Failing startup here took the whole dashboard
    # down for a worker-only problem.
    try:
        worker.start()
        st = worker.status()
        logger.info(
            f"[boot] worker daemon: online={st.get('online')} "
            f"pid={worker.daemon_pid} concurrency={st.get('concurrency')} "
            "(separate process, nice 15 — this API only proxies)"
        )
    except Exception as exc:  # noqa: BLE001 - never make the API unavailable
        logger.error(f"[boot] worker daemon did not start: {exc}")
    # Finishes any batch a restart interrupted mid-delete. Runs on its own
    # thread and in small steps, so a half-deleted batch is completed at boot
    # instead of sitting in the list forever — and never in a request.
    try:
        purge.start(
            store,
            settings.app.paths,
            cancel=worker.cancel,
            remote=_purge_remote_job,
        )
    except Exception as exc:  # noqa: BLE001 - a purge must not keep the API down
        logger.warning(f"[boot] delete purge did not start: {exc}")
    # Keeps the 4 GB disk usable without anyone pressing a button: when free
    # space drops toward the worker's 400 MB floor, finished batches move into
    # the bucket and their local copies go. Off the request path on purpose —
    # a slow PUT inside a page load is what turned deletes into CDN 502s.
    try:
        reclaim.start_watchdog(store, settings.app.paths)
    except Exception as exc:  # noqa: BLE001 - never let cleanup break boot
        logger.warning(f"[boot] media reclaim watchdog did not start: {exc}")
    # Crash leftovers: a kill -9 (or a sandbox snapshot) leaves scratch folders
    # for jobs that no longer exist and half-written `.fetching` files. Clear
    # them before any job can measure the disk, so one bad crash does not need
    # a human with a terminal.
    try:
        swept = reclaim.sweep_startup(store, settings.app.paths)
        if swept.get("dirsRemoved") or swept.get("filesRemoved"):
            logger.info(
                f"[boot] crash sweep: {swept['dirsRemoved']} folder(s), "
                f"{swept['filesRemoved']} temp file(s), "
                f"{round(swept['bytesFreed'] / (1024 * 1024), 1)} MB freed"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[boot] crash sweep did not run: {exc}")
    retention.start()
    # Sweep ONCE at boot, not just on the daemon's first 30-minute tick.
    #
    # The daemon loop is `while not self._stop.wait(interval_min * 60)`, so
    # the first pass was 30 minutes after startup — meaning every sandbox
    # restart inherited the previous run's debris and then sat on it for half
    # an hour. With retention at 0 that accumulated indefinitely: 353 MB of
    # trashed batches on a 4 GB disk at the time of writing. A boot sweep is
    # the one moment guaranteed to see the whole picture, so it runs here. It
    # is the same `sweep()` the daemon calls, so there is no second code path
    # to keep honest.
    try:
        swept = retention.sweep()
        if swept:
            logger.info(f"[boot] startup sweep removed {swept} expired/orphaned item(s)")
    except Exception as exc:  # noqa: BLE001 - never block startup on cleanup
        logger.warning(f"[boot] startup sweep failed: {exc}")
    # Clip documents whose job is gone are invisible in the UI but still on
    # disk (a previous build's deletes left them behind). Reconcile at boot so
    # the dashboard starts from a consistent set instead of showing phantom
    # "queued" cards until the 7-day orphan window elapses.
    try:
        stats = store.reconcile_clips()
        if any(stats.values()):
            logger.info(
                "[boot] clip reconcile: "
                + ", ".join(f"{v} {k}" for k, v in stats.items() if v)
            )
    except Exception as exc:  # noqa: BLE001 - never block startup on cleanup
        logger.warning(f"[boot] clip reconcile failed: {exc}")
    # Clip documents a killed worker left mid-flight: the MP4 is on disk but the
    # document still says pending/running, so the dashboard shows “queued ·
    # no fps” and blocks play/download/share on a clip that is actually fine.
    try:
        fixed = video_svc.repair_render_states(store)
        if fixed.get("repaired"):
            logger.info(
                f"[boot] render state repaired for {fixed['repaired']} clip(s) "
                f"({fixed['scanned']} scanned, {fixed['skipped']} skipped)"
            )
    except Exception as exc:  # noqa: BLE001 - never block startup on a repair
        logger.warning(f"[boot] render state repair failed: {exc}")
    tailnet = hosting.tailnet_ip()
    logger.info(
        f"[boot] API up on http://{settings.app.host}:{settings.app.port} "
        f"(worker: {'remote dispatch' if settings.hosted.worker_url else 'local in-process'})"
    )
    if tailnet:
        logger.info(f"[boot] tailnet IP: {tailnet}")
    elif settings.hosted.tailscale_only or settings.hosted.worker_url:
        logger.info("[boot] tailnet IP: not detected (tailscaled not reachable from this process)")


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


def _clip_media(clip: dict[str, Any], kind: str) -> Path | None:
    """Path to one of a clip's media files, or None when it does not exist."""
    if clip.get("shareId") and kind == "video":
        return _share_copy_for(clip)
    key = {"video": "videoPath", "thumb": "thumbPath"}.get(kind)
    if not key:
        return None
    p = Path(clip.get(key) or "")
    return p if p.exists() else None


def _clip_for_share_id(raw: str) -> dict[str, Any] | None:
    """Resolve a share id back to the clip document that owns it."""
    from app import store
    for clip in store.list_all_clips():
        if clip.get("shareId") == raw:
            return clip
    return None


def _share_copy_for(clip: dict[str, Any]) -> Path | None:
    """Where this clip's share-sized MP4 lives, building it if needed."""
    master = _clip_media(clip, "video")
    if master is None:
        return None
    try:
        return _JOB_SHARE.get(master)
    except Exception as exc:
        # A transient path/lock issue must never turn a share request into a
        # 503: degrade to the link, and let the lock reap the copy later.
        logger.warning(f"[share] _share_copy_for failed for {clip.get('_id')}: {exc}")
        return None


def _media_etag(p: Path) -> str:
    """A stable etag from real bytes (size + mtime), so changed files 412."""
    try:
        return f"{p.stat().st_size}-{int(p.stat().st_mtime)}"
    except OSError:
        return ""


def _clip_bucket_stream(clip: dict[str, Any], request: Request, filename: str) -> StreamingResponse:
    """Stream a clip's master from the bucket when local media was reclaimed."""
    from app import objectstore, store
    if not clip.get("videoPath") or Path(clip["videoPath"]).exists():
        return clip_download(clip, request, filename)
    key = objectstore.key_for_path(clip["videoPath"], settings.app.paths.outputs_dir)
    if not key:
        return clip_download(clip, request, filename)
    return StreamingResponse(
        objectstore.open_stream(key, request),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Disposition": f'attachment; filename="{filename}"'},
    )


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



def _clip_media(clip: dict[str, Any], kind: str) -> Path | None:
    """Path to one of a clip's media files, or None when it does not exist."""
    if clip.get("shareId") and kind == "video":
        return _share_copy_for(clip)
    key = {"video": "videoPath", "thumb": "thumbPath"}.get(kind)
    if not key:
        return None
    p = Path(clip.get(key) or "")
    return p if p.exists() else None


def _clip_for_share_id(raw: str) -> dict[str, Any] | None:
    """Resolve a share id back to the clip document that owns it."""
    from app import store
    for clip in store.list_all_clips():
        if clip.get("shareId") == raw:
            return clip
    return None


def _share_copy_for(clip: dict[str, Any]) -> Path | None:
    """Where this clip's share-sized MP4 lives, building it if needed."""
    master = _clip_media(clip, "video")
    if master is None:
        return None
    try:
        return _JOB_SHARE.get(master)
    except Exception as exc:
        # A transient path/lock issue must never turn a share request into a
        # 503: degrade to the link, and let the lock reap the copy later.
        logger.warning(f"[share] _share_copy_for failed for {clip.get('_id')}: {exc}")
        return None


def _media_etag(p: Path) -> str:
    """A stable etag from real bytes (size + mtime), so changed files 412."""
    try:
        return f"{p.stat().st_size}-{int(p.stat().st_mtime)}"
    except OSError:
        return ""


def _clip_bucket_stream(clip: dict[str, Any], request: Request, filename: str) -> StreamingResponse:
    """Stream a clip's master from the bucket when local media was reclaimed."""
    from app import objectstore, store
    if not clip.get("videoPath") or Path(clip["videoPath"]).exists():
        return clip_download(clip, request, filename)
    key = objectstore.key_for_path(clip["videoPath"], settings.app.paths.outputs_dir)
    if not key:
        return clip_download(clip, request, filename)
    return StreamingResponse(
        objectstore.open_stream(key, request),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
    master = _clip_media(clip, "video")
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
def public_share_mp4(share_id: str) -> FileResponse:
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
    clip = _clip_for_share_id(raw)
    if clip is None:
        raise HTTPException(404, "this share link does not exist")
    share = _share_copy_for(clip)
    if share is None or not share.exists():
        raise HTTPException(503, "this share link is still being prepared")
    return FileResponse(
        share,
        media_type="video/mp4",
        headers={
            "Cache-Control": SHARE_CACHE_CONTROL,
            "Accept-Ranges": "bytes",
            "ETag": _media_etag(share),
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Content-Type",
            "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
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
    share = _share_copy_for(clip)
    if share is None or not share.exists():
        raise HTTPException(503, "this share link is still being prepared")
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


@app.post("/api/clips/{clip_id}/share-link")
def ensure_share_link(clip_id: str) -> dict:
    """Mint/verify this clip's public link and report its real state.

    This is the endpoint behind the Share button's "Preparing… -> ✓ Ready".
    It is idempotent: the same clip returns the same id forever, and calling
    it on an already-ready clip costs one stat().
    """
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    sid = sharelink.ensure_id(clip)
    if not sharelink.valid_id(sid):
        raise HTTPException(500, "could not mint a share id for this clip")
    share = _share_copy_for(clip)
    state = sharelink.share_state(clip, share)
    if state["shareState"] != sharelink.STATE_READY:
        clip["shareState"] = state["shareState"]
        store.write_clip(clip)
    return state


def _attach_share_state(clips: list[dict]) -> list[dict]:
    """Add the share facts to each clip, without building anything.

    A HEAD check the UI can trust has to answer from the filesystem, so this
    probes the share copy that already exists and says "preparing" for the rest.
    It never transcodes — that happens at the end of a render, or on the first
    request to the public URL.
    """
    out = []
    for clip in clips:
        share = None
        master = clip.get("videoPath")
        if master:
            try:
                share = share_video.share_path_for(Path(master))
            except Exception:  # noqa: BLE001 - never fail a list over a poster
                share = None
        out.append({**clip, **sharelink.share_state(clip, share)})
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
    order = int(doc.get("order") or 0)
    if not job_id:
        return cues
    clips_dir = store._paths.data_dir / "clips"
    json_path = clips_dir / f"{doc.get('_id')}.json"
    if not json_path.exists():
        return cues
    try:
        doc_data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return cues
    for key in ("cues_000.json", "cues_001.json", "cues_002.json", "cues_003.json"):
        cue_path = store._paths.outputs_dir / job_id / "captions" / key
        if cue_path.exists():
            try:
                payload = json.loads(cue_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
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
        # No poster on disk: make one from the clip rather than showing a black
        # card. This is also what heals posters rendered before the
        # non-black-frame selection existed.
        master = _clip_media(clip, "video") if clip.get("videoPath") else None
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
    master_for_thumb = _clip_media(clip, "video") if clip.get("videoPath") else None
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


@app.api_route("/api/clips/{clip_id}/download", methods=["GET", "HEAD"])
def clip_download(clip_id: str, request: Request) -> Response:
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    safe = "".join(ch for ch in clip.get("title", "clip") if ch.isalnum() or ch in " -_")[:40] or "clip"
    filename = f"{clip['order'] + 1:02d}-{safe}.mp4"
    streamed = _clip_bucket_stream(clip, request, filename=filename)
    if streamed is not None:
        return streamed
    p = _clip_media(clip, "video")
    if p is None:
        raise HTTPException(404, "clip file missing on disk")
    return FileResponse(
        p,
        media_type="video/mp4",
        filename=f"{clip['order'] + 1:02d}-{safe}.mp4",
        headers={
            "Content-Disposition": f'attachment; filename="{clip["order"] + 1:02d}-{safe}.mp4"',
            "Cache-Control": MEDIA_CACHE_CONTROL,
            "Accept-Ranges": "bytes",
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


@app.get("/api/clips/{clip_id}/srt")
def clip_srt(clip_id: str) -> Response:
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    lines = []
    for i, cue in enumerate(clip.get("captionCues", []), 1):
        lines.append(f"{i}\n{_srt_time(cue['start'])} --> {_srt_time(cue['end'])}\n{cue['text']}\n")
    return Response("\n".join(lines), media_type="text/plain")


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
