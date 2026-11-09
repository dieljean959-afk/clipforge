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

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
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
    share_video,
    sharelink,
    subtitle,
)
from app import download as download_mod
from app import encode as encode_mod
from app.hosting import disk_stats
from app import llm as llm_mod


from uuid import uuid4 as uuid4


MIN_FREE_BYTES = 100 * 1024 * 1024  # 100 MB free-disk floor before HTTP 507
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
from app.slices.cleanup import unlink_model_dirs
from starlette.background import BackgroundTask
# Module handle, distinct from the `worker` INSTANCE below. The boot recovery
# pass needs `unrecoverable_reason` / `INTERRUPT_MAX_RESUMES` from the module;
# without this import that pass raised NameError at import time and uvicorn
# exited before binding port 8000, so the proxy answered a bodiless 500.
from app import worker as worker_mod
from app.worker import InProcessWorker
from app.workerd import WorkerSupervisor


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

class CreateJobIn(BaseModel):
    sourceType: str = Field(pattern="^(url|upload)$")
    source: str = Field(min_length=1, max_length=2048)
    minClipSec: int | None = Field(default=None, ge=5, le=180)
    maxClipSec: int | None = Field(default=None, ge=10, le=180)
    maxClips: int | None = Field(default=None, ge=1, le=clipcount.HARD_CEILING)
    # ONE caption style (LOCKED_SPEC §1). The field stays for older clients, but
    # `bold_stack` is the only key any render can wear.
    captionPreset: str | None = Field(default=None, pattern="^(bold_stack)$")
    language: str = Field(default="auto", pattern="^(auto|en|es)$")
    # Emoji per batch: on/off and how dense. Omitted = the config defaults.
    emojiEnabled: bool | None = None
    emojiDensity: str | None = Field(default=None, pattern="^(low|medium|high)$")


class EmojiEditIn(BaseModel):
    """One chunk's emoji, edited by hand in the caption editor.

    ``emoji`` is a whitelisted glyph to add/swap, or null to remove it. The
    clip is re-rendered after the edit unless ``rerender`` is false (the UI
    sends false while the user is still picking).
    """

    i: int = Field(ge=0)
    emoji: str | None = None
    rerender: bool = True


class RequeueIn(BaseModel):
    jobId: str


class DeleteJobBody(BaseModel):
    """Body for POST /api/jobs/{id}/delete.

    `keepSource` mirrors the checkbox in the confirm dialog: keep the
    downloaded/uploaded source video, delete the record + outputs only.
    Sent as an empty JSON object (or no body at all) for normal deletes.
    """

    keepSource: bool = False


class BulkDeleteIn(BaseModel):
    """An explicit list of ids. Never a filter — see /api/jobs/bulk-delete.

    The length is checked by the handler, not by the field: a `min_length=1`
    here answered a 422 whose `detail` is a list of pydantic objects, which the
    dashboard then had to render as a bare "HTTP 422". The handler's own
    "no job ids given" is the message a person can act on.
    """
    jobIds: list[str] = Field(default_factory=list, max_length=200)


class InternalProgressIn(BaseModel):
    stage: str
    progress: float = Field(ge=0, le=100)
    progressDetail: str | None = None
    unitLabel: str | None = None
    unitsDone: float | None = None
    unitsTotal: float | None = None


class InternalLogIn(BaseModel):
    lines: list[str] = Field(max_length=50)


class InternalFailedIn(BaseModel):
    error: str
    detail: str | None = None


class InternalCompleteIn(BaseModel):
    clipCount: int = Field(ge=0)
    totalClipsSec: float = Field(ge=0)
    captionsTxt: str | None = None


def _require_worker_token(request: Request) -> None:
    expected = settings.hosted.worker_token
    if not expected:
        raise HTTPException(503, "worker token not configured — set WORKER_TOKEN to accept remote updates")
    got = request.headers.get("authorization", "")
    # constant-time-ish compare; token leakage is not a realism concern here
    if not got.startswith("Bearer ") or got[7:] != expected:
        raise HTTPException(401, "invalid worker token")


# ---------------------------------------------------------------------------
# dev/diagnostics: verify every clip on disk with ffprobe — honest pass/fail
# ---------------------------------------------------------------------------


@app.get("/api/dev/verify-clips")
def verify_clips(jobId: str | None = None) -> dict:
    """ffprobe every rendered clip and report valid/invalid with reasons."""
    clips = store.list_clips(jobId) if jobId else store.list_all_clips(limit=512)
    results = []
    for clip in clips:
        entry: dict = {"clipId": clip["_id"], "order": clip.get("order"), "title": clip.get("title")}
        vp = clip.get("videoPath")
        if not vp:
            entry.update({"valid": False, "reason": "no rendered file"})
        elif not Path(vp).exists():
            entry.update({"valid": False, "reason": f"file missing on disk: {Path(vp).name}"})
        else:
            try:
                meta = video_svc.probe(Path(vp))
                if meta["duration"] <= 0:
                    entry.update({"valid": False, "reason": "ffprobe reports 0s duration"})
                else:
                    entry.update({
                        "valid": True,
                        "reason": "ok",
                        "durationSec": meta["duration"],
                        "width": meta["width"],
                        "height": meta["height"],
                        "bytes": Path(vp).stat().st_size,
                    })
            except Exception as exc:
                entry.update({"valid": False, "reason": f"ffprobe failed: {exc}"})
        results.append(entry)
    valid = sum(1 for r in results if r["valid"])
    return {"total": len(results), "valid": valid, "invalid": len(results) - valid, "clips": results}


# --------------------------------------------------------------------------- #
# detection backend (DeepSeek = hosted default, NVIDIA/Groq = hosted alts, Gemini, Ollama = local)
# --------------------------------------------------------------------------- #

# Shown verbatim in the UI so the tradeoff is visible BEFORE the user picks.
# Deliberately no copy here calls any backend "the DEFAULT": the provider is
# switchable from the dashboard without a redeploy, so a label baked at import
# time goes stale the moment anyone picks a different backend.
DETECTION_TRADEOFFS = {
    "deepseek": "Hosted, OpenAI-compatible chat completions. The DeepSeek platform "
                "bills a balance rather than a fixed tokens-per-DAY ceiling like "
                "Groq's free tier, so a long source cannot run out of quota half way through "
                "detection — and its rates are the cheapest of the hosted options. Needs a "
                "DeepSeek API key, and a topped-up balance.",
    "nvidia": "Hosted, selectable without a redeploy. NVIDIA NIM is the same wire "
              "format and the same code path, and hosts its own `deepseek-*` models — "
              "metered on NVIDIA credits, not a DeepSeek balance. Needs an NVIDIA API key.",
    "groq": "Hosted BACKUP. Same wire format and the same code path as DeepSeek, but its "
            "free tier is ~30 requests per minute with a 200k tokens-per-DAY ceiling "
            "per model — the app paces calls below the request limit and retries 429s, "
            "but a spent daily quota can only be waited out. Needs a Groq API key.",
    "gemini": "Hosted alternative, selectable without a redeploy. Needs a Gemini API key.",
    "ollama": "Fully local, free and private: your transcript never leaves the machine. "
              "Slower — it shares the same CPU as Whisper, so it runs sequentially "
              "after the transcript exists instead of alongside it.",
}


def detection_status(probe: bool = True) -> dict:
    """Which backend runs AI highlight detection, and is it actually usable.

    `probe=False` answers from configuration and key presence only, making NO
    network call. `/api/health` uses it, so a Groq or Ollama outage can never
    stall or break our own health endpoint — a third-party hiccup must not be
    able to take down the page that tells the user anything is wrong.
    """
    active = llm_mod.provider(settings)
    out: dict = {
        "provider": active,
        "label": llm_mod.provider_label(settings),
        "tradeoff": DETECTION_TRADEOFFS.get(active, ""),
        "options": [
            {"value": p, "label": p, "tradeoff": DETECTION_TRADEOFFS.get(p, "")}
            for p in llm_mod.PROVIDERS
        ],
        "models": {
            "deepseek": settings.deepseek.model,
            "nvidia": settings.nvidia.model,
            "groq": settings.groq.model,
            "gemini": settings.gemini.model,
            "ollama": settings.llm.ollama_model,
        },
        # The backup is reported alongside the primary so the UI can say what
        # happens when the primary runs out, rather than leaving the user to
        # discover it from a failed job.
        "fallbackProvider": llm_mod.settings_llm_fallback(settings),
        # Transcription is unaffected by this choice — say so explicitly so the
        # UI never implies the ASR engine changes with the detection backend.
        "transcriptionEngine": subtitle_svc.engine_status(settings).engine,
    }
    if active == "ollama":
        if not probe:
            # No network call: report the configured target and let the
            # dashboard / doctor do the actual reachability check.
            out["ready"] = None
            out["probed"] = False
            out["detail"] = f"ollama configured at {settings.llm.ollama_url} (not probed by /api/health)"
        else:
            ok, reason = llm_mod.ollama_status(settings)
            out["ready"] = ok
            out["probed"] = True
            out["detail"] = reason
    elif active in llm_mod.OPENAI_PROVIDER_KEYS:
        live = llm_mod._LIVE_KEY_RESOLVERS[active]()
        var = llm_mod.OPENAI_PROVIDER_KEYS[active][1]
        # force=True re-reads the env each call, which is what makes a key
        # saved after boot appear on the next poll. It is a local env lookup,
        # never a request to the provider, so it is safe in the health path.
        live.refresh(force=probe)
        out["ready"] = live.key is not None
        out["probed"] = False
        out["detail"] = live.problem or (
            f"{var} loaded from {live.source}" if live.key
            else f"{var} missing — add it in the Keys/API keys tab."
        )
        backup = llm_mod.settings_llm_fallback(settings)
        if out["ready"] and backup:
            out["detail"] += f" Falls back to {backup} if a chunk fails."
    else:
        out["ready"] = settings.gemini_api_key is not None
        out["probed"] = False
        out["detail"] = (
            f"GEMINI_API_KEY loaded from {live_gemini_key().source}"
            if settings.gemini_api_key else "GEMINI_API_KEY missing."
        )
    return out


class DetectionIn(BaseModel):
    provider: str


@app.post("/api/settings/detection")
def set_detection_backend(body: DetectionIn) -> dict:
    """Switch the AI-detection backend. Transcription engine is untouched."""
    if body.provider not in llm_mod.PROVIDERS:
        raise HTTPException(
            400, f"unknown detection backend {body.provider!r} (expected one of {llm_mod.PROVIDERS})"
        )
    settings.llm.provider = body.provider
    logger.info(f"[llm] detection backend switched to {body.provider}")
    return detection_status()


# --------------------------------------------------------------------------- #
# health / status
# --------------------------------------------------------------------------- #

# Spawning `<binary> -version` costs ~300ms on this 1-vCPU host, and /api/health
# asked for two of them on every poll — that alone pushed health past the
# supervisor's probe timeout and restarted a healthy API in a loop. A binary's
# version cannot change while this process runs, so the answer is cached.
_TOOL_VERSION_CACHE: dict[tuple[str, str], str | None] = {}


def _tool_version(binary: str, flag: str = "-version") -> str | None:
    """First line of `<binary> -version`, or None when it cannot run.

    The status card claims which tools the pipeline will actually use, so the
    number has to come from the binary on PATH — not from a constant. ffmpeg
    4.4 prints its version and still exits 1 on some builds, so the exit code
    is not treated as failure when stdout has text.
    """
    import subprocess

    cache_key = (binary, flag)
    if cache_key in _TOOL_VERSION_CACHE:
        return _TOOL_VERSION_CACHE[cache_key]

    try:
        proc = subprocess.run(
            [binary, flag], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        # Not cached: a missing binary may be installed later (the UI offers
        # that), and a negative answer is cheap to retry.
        return None
    out = (proc.stdout or "") + (proc.stderr or "")
    line = out.strip().splitlines()[0] if out.strip() else ""
    _TOOL_VERSION_CACHE[cache_key] = line or None
    return line or None


def _ytdlp_version() -> str | None:
    try:
        import yt_dlp
    except Exception:
        return None
    # `yt_dlp.version` is a MODULE, so `getattr(yt_dlp, "version")` is not a
    # string — the module's `__version__` is.
    text = getattr(getattr(yt_dlp, "version", None), "__version__", None)
    if not isinstance(text, str):
        text = getattr(yt_dlp, "__version__", None)
    return text if isinstance(text, str) else None


@app.get("/api/health")
def health() -> dict:
    """Liveness + status for the dashboard.

    Two rules, both learned from a real "API unreachable: HTTP 400" banner:

    1. **This endpoint never fails because a SUB-COMPONENT failed.** Every
       block is built through `_safe`, so a disk probe, a shell stat, an RTF
       read or a key lookup that raises degrades that one field to
       `{"error": ...}` and still returns 200 with everything else. Before,
       any single one of those could take down the whole payload, and the
       dashboard's only way to describe "I could not read one number" was to
       claim the entire API was unreachable.
    2. **It does not depend on any third party being up.** In particular the
       AI provider is consulted WITHOUT a network probe (`probe=False`).
       `detection_status()` otherwise calls `ollama_status()`, which opens an
       HTTP connection with a 5s timeout — so a third-party hiccup could stall
       or break our own health endpoint. Groq/Gemini readiness here is key
       PRESENCE only (an env lookup); whether the vendor is actually answering
       is `/api/doctor`'s job, and detection reports it honestly when it runs.

    A health check answers "is this process serving?", so the status is 200
    whenever we got far enough to answer at all. It reports `ok: false` plus
    `degraded` reasons when something is wrong, which is more useful than a
    bare non-2xx — and 400 is never a valid response for this route.
    """
    def _safe(name: str, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            logger.warning("[health] block %r degraded: %s: %s", name, type(exc).__name__, exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

    degraded: list[str] = []

    def _block(name: str, fn, *args, **kwargs):
        out = _safe(name, fn, *args, **kwargs)
        if isinstance(out, dict) and "error" in out and len(out) == 1:
            degraded.append(name)
        return out

    remote = None
    if settings.hosted.worker_url:
        remote = _safe("remoteWorker", hosting.remote_probe, settings)
    worker_status = _block("worker", worker.status)
    engine = _block("whisper", subtitle_svc.engine_status, settings)
    rtf_value, rtf_source = 0.0, "unavailable"
    if not (isinstance(engine, dict) and "error" in engine):
        rtf_store = _safe("rtfStore", eta.store, settings.app.paths.data_dir)
        if rtf_store is not None and not (isinstance(rtf_store, dict) and "error" in rtf_store):
            eff = _safe(
                "rtf", rtf_store.effective_rtf,
                engine.engine, engine.device, engine.model, engine.compute_type,
            )
            if eff and not (isinstance(eff, dict) and "error" in eff):
                rtf_value, rtf_source = eff
    return {
        # The process answered, so it IS up. `ok` is false only when a part of
        # the picture is missing, and `degraded` says which part.
        "ok": not degraded,
        "degraded": degraded,
        "version": "1.1.0",
        "worker": worker_status,
        "warmup": _block(
            "warmup",
            lambda: dict(warmup),
        ),
        "llm": _block(
            "llm",
            lambda: {
                "provider": llm_mod.provider(settings),
                # LIVE re-resolution — a key saved in the platform after boot is
                # picked up on the next health poll without restarting anything.
                **live_gemini_key().status(),
            },
        ),
        "llmGroq": _block(
            "llmGroq",
            lambda: {
                "provider": llm_mod.PROVIDER_GROQ,
                **live_groq_key().status(),
            },
        ),
        "llmNvidia": _block(
            "llmNvidia",
            lambda: {
                "provider": llm_mod.PROVIDER_NVIDIA,
                **live_nvidia_key().status(),
            },
        ),
        "llmDeepSeek": _block(
            "llmDeepSeek",
            lambda: {
                "provider": llm_mod.PROVIDER_DEEPSEEK,
                **live_deepseek_key().status(),
            },
        ),
        # probe=False: key presence only, never a vendor round trip. A Groq or
        # Ollama outage must not be able to fail our own health endpoint.
        "detection": _block("detection", detection_status, probe=False),
        "whisper": _block(
            "whisperDetail",
            lambda: {
                # The RESOLVED engine/model/compute, not what config.toml asks
                # for. The host-limit downgrade (medium -> small) has to be
                # visible in the status label, otherwise the UI advertises a
                # model that never loads.
                "model": engine.model,
                "compute": engine.compute_type,
                "configuredModel": settings.whisper.model,
                "engine": engine.engine,
                "isWhisperx": engine.is_whisperx,
            },
        ),
        "asr": _block(
            "asr",
            lambda: {
                # Which engine is really transcribing, which model actually
                # loads, and the RTF this host has measured — the same numbers
                # the ETA is derived from.
                **engine.as_dict(),
                "estimatedRtf": round(rtf_value, 4),
                "rtfSource": rtf_source,
                "rtfHistory": (
                    rtf_store.snapshot()
                    if rtf_store is not None and not (isinstance(rtf_store, dict) and "error" in rtf_store)
                    else []
                ),
                "host": hostlimits.limits(),
                "hostDetail": hostlimits.describe(),
            },
        ),
        "system": _block("system", hosting.system_stats),
        # Whether clips are pushed to durable storage. `enabled` needs all four
        # CLIPFORGE_S3_* values, and config_from_env() re-reads per call, so this
        # turns ON by itself once the keys land — no restart, nothing to guess at.
        "storage": _block("storage", objectstore.status),
        "encode": _block("encode", lambda: encode_mod.as_dict(encode_mod.choose(settings))),
        "disk": _block("disk", lambda: disk_stats(settings.app.paths.data_dir)),
        # The status card reports the pipeline's real toolchain and where it
        # writes. Versions are read from the binaries on PATH per poll, so a
        # rebuilt sandbox shows up without redeploying the frontend.
        "tools": _block(
            "tools",
            lambda: {
                "ffmpeg": _tool_version(ffmpeg_bin()),
                "ffprobe": _tool_version(ffprobe_bin()),
                "ytdlp": _ytdlp_version(),
                "mediaDir": str(settings.app.paths.outputs_dir),
                "downloadsDir": str(settings.app.paths.downloads_dir),
            },
        ),
        "tailnetIp": hosting.tailnet_ip(),
        # The cookie jar's real state, so an expiring export is a visible
        # warning before it becomes a URL that will not download. Reading a
        # small text file is cheap; a yt-dlp probe is not, and is not done here.
        "cookies": _block(
            "cookies",
            lambda: download_mod.cookie_health(
                getattr(settings.download, "cookies_file", "") or None
            ),
        ),
        "limits": {
            "maxSourceMin": settings.hosted.max_source_min,
            "retentionHours": settings.hosted.retention_hours,
        },
        "stallTimeoutSec": worker_status.get("stallTimeoutSec"),
        "dispatch": (
            {"enabled": True, "workerUrl": settings.hosted.worker_url, "remote": remote}
            if settings.hosted.worker_url
            else {"enabled": False}
        ),
    }


@app.post("/api/sandbox/restart")
def restart_sandbox() -> dict:
    """Stop the worker daemon and start a fresh one.

    This is the dashboard's "Restart sandbox" button. It deliberately replaces
    the process instead of waiting for a boot: the daemon owns the queue and
    every worker thread, so a stage wedged on a dead child only clears when that
    process goes away. Stages are cached on disk, so a queued job resumes from
    its last checkpoint.

    Never raises: the button shows the reason inline, and a 500 here would read
    as "the API is down" when the API is exactly what answered.
    """
    try:
        result = worker.restart()
    except Exception as exc:  # noqa: BLE001 - report, do not 500
        logger.warning("[sandbox] restart failed: %s: %s", type(exc).__name__, exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    pid = result.get("pid")
    if pid is None:
        return {
            "ok": False,
            "error": "the worker daemon did not come back — see data/logs/workerd.out",
        }
    return {"ok": True, "pid": pid, "previousPid": result.get("previousPid")}


@app.get("/api/status")
def api_status() -> dict:
    """Per-dependency status: one broken dependency never 500s the endpoint.

    `/api/health` answers "is the API process serving?" and is cheap on purpose.
    This route answers the next question — "which dependency is broken?" — and
    therefore touches the disk, the bucket config and the yt-dlp/ffmpeg binaries.
    Every probe runs through its own try/except: a failing bucket probe, a
    missing binary or an unreadable data dir each degrade ONE entry to
    ``{"ok": false, "message": ...}`` while the remaining entries still answer,
    and the response is always 200. A status page that dies because a dependency
    died is useless exactly when it is needed.
    """
    def _probe(name: str, fn, *args, **kwargs) -> dict:
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            logger.warning(f"[status] {name} probe failed: {type(exc).__name__}: {exc}")
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
        if isinstance(out, dict) and out.get("ok") is False:
            return {"ok": False, "message": out.get("message") or out.get("error") or "not ready"}
        message = out.get("message") if isinstance(out, dict) else None
        return {"ok": True, "message": message or "ok"}

    def _db() -> dict:
        # Job records are JSON files, so the real question is "can we read and
        # write the data dir?" — not a socket connect.
        probe = settings.app.paths.data_dir / ".status-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        jobs = store.list_jobs()
        return {"ok": True, "message": f"readable · {len(jobs)} job(s)"}

    def _config() -> dict:
        if not settings.whisper.model:
            return {"ok": False, "message": "no whisper model configured"}
        return {"ok": True, "message": f"loaded · model {settings.whisper.model}"}

    def _bucket() -> dict:
        # Fail SOFT. Object storage is optional: without keys, and without the
        # boto3 client, media simply stays on local disk. That is "offload
        # unavailable", not "the API is broken".
        st = objectstore.status()
        enabled = bool(st.get("enabled"))
        ready = bool(st.get("clientReady"))
        if not enabled:
            return {"ok": True, "message": "offload unavailable — no bucket configured (local disk only)"}
        if not ready:
            return {"ok": True, "message": "offload unavailable — bucket configured but client not ready (local disk only)"}
        return {"ok": True, "message": f"offload on · {st.get('bucket')}"}

    def _worker() -> dict:
        st = worker.status()
        if not st.get("online"):
            return {"ok": False, "message": "worker queue is not running"}
        return {
            "ok": True,
            "message": (
                f"online · {st.get('activeCount', 0)} active, "
                f"{st.get('queueDepth', 0)} queued"
            ),
        }

    def _binary(name: str, getter, flag: str) -> dict:
        path = getter()
        line = _tool_version(path, flag)
        if not line:
            return {"ok": False, "message": f"{name} not found or not runnable at {path}"}
        return {"ok": True, "message": line}

    def _disk() -> dict:
        st = disk_stats(settings.app.paths.data_dir)
        free = st.get("freeBytes") or 0
        if free < MIN_FREE_BYTES:
            return {
                "ok": False,
                "message": f"only {free // (1024 * 1024)} MB free — new jobs are refused below {MIN_FREE_BYTES // (1024 * 1024)} MB",
            }
        return {"ok": True, "message": f"{free // (1024 * 1024)} MB free"}

    deps = {
        "db": _probe("db", _db),
        "config": _probe("config", _config),
        "bucket": _probe("bucket", _bucket),
        "worker": _probe("worker", _worker),
        "ytdlp": _probe("ytdlp", lambda: (
            {"ok": True, "message": _ytdlp_version()}
            if _ytdlp_version()
            else {"ok": False, "message": "yt-dlp python package is not installed"}
        )),
        "ffmpeg": _probe("ffmpeg", lambda: _binary("ffmpeg", ffmpeg_bin, "-version")),
        "disk": _probe("disk", _disk),
    }
    failing = [name for name, d in deps.items() if not d["ok"]]
    return {
        # Always 200: this endpoint's job is to REPORT, so a down dependency is
        # data, not an HTTP error.
        "ok": not failing,
        "failing": failing,
        "deps": deps,
        "pid": os.getpid(),
        "uptimeSec": int(time.time() - _STARTED_AT),
    }


@app.get("/api/doctor")
def doctor() -> dict:
    """Actionable environment diagnostics (ffmpeg build, GPU, model, API, disk)."""
    from app.doctor import run_doctor

    try:
        return run_doctor(settings)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never 500
        # These probes shell out to ffmpeg and touch the network, so any one of
        # them can raise on a loaded or throttled host. A diagnostics page that
        # returns a bare 500 tells the user nothing, which is exactly when they
        # need it most: report the failure as a check instead of an exception.
        logger.exception("[doctor] diagnostics failed")
        return {
            "ok": False,
            "summary": {"fail": 1, "warn": 0, "ok": 0},
            "checks": [
                {
                    "name": "doctor",
                    "status": "fail",
                    "detail": f"Diagnostics could not complete: {exc}",
                    "fix": "Re-check once the host is idle, or run "
                           "`.venv/bin/python -m app.doctor` for the full report.",
                }
            ],
        }


@app.get("/api/caption-presets")
def caption_presets() -> list[dict]:
    from app.subtitle import CAPTION_PRESETS

    return [
        {"key": p.key, "label": p.label}
        for p in CAPTION_PRESETS.values()
    ]


@app.get("/api/usage")
def usage() -> dict:
    """Today's token budget for the ACTIVE detection provider.

    Read-only and dependency-free: the counter is written as calls happen, so
    this never calls out to the provider. If the key is absent, or nothing has
    been spent today, it still answers with a zeroed but well-formed shape so
    the UI has no empty state to handle.

    The denominator follows the ACTIVE provider, because the budgets are not
    comparable: Groq states a real per-model tokens-per-day ceiling in its
    429 body, while DeepSeek and NIM bill a balance or credits and send no
    equivalent number, so their limits are whatever the operator configured.
    Showing Groq's 200k while running on DeepSeek would report a badge that is
    both wrong and alarming.

    A provider with NO daily token budget at all (Ollama is local and free;
    Gemini bills per token against a key with no published daily ceiling) used
    to borrow Groq's config wholesale, so the badge showed a confident
    "0 / 200,000" and named a Groq model while a completely different backend
    was selected. It now reports `budgeted: false` with no invented numbers,
    and the UI says so instead of drawing a meter.
    """
    from app import groq_usage

    active = llm_mod.provider(settings)
    budgeted = active in llm_mod.OPENAI_PROVIDER_KEYS
    cfg = llm_mod.openai_cfg(settings, active) if budgeted else None
    limit = int(getattr(cfg, "daily_token_limit", 0) or 0) if cfg else 0
    model = str(getattr(cfg, "model", "") or "") if cfg else ""
    snap = groq_usage.store(settings.app.paths.data_dir).snapshot(
        default_limit=limit, model=model, provider=active
    )
    snap["configured"] = bool(model)
    snap["provider"] = active
    snap["budgeted"] = budgeted
    snap["providerLabel"] = str(getattr(cfg, "label", active)) if cfg else active.capitalize()
    return snap


@app.get("/api/warmup")
def warmup_status() -> dict:
    return warmup


@app.post("/api/settings/reload-secrets")
def reload_secrets() -> dict:
    """Force re-read of env files + live key resolution (the Re-check button)."""
    live = live_gemini_key()
    live.refresh(force=True)
    status = live.status()
    groq_live = live_groq_key()
    groq_live.refresh(force=True)
    nvidia_live = live_nvidia_key()
    nvidia_live.refresh(force=True)
    deepseek_live = live_deepseek_key()
    deepseek_live.refresh(force=True)
    logger.info(f"[config] secrets re-checked — key loaded: {status['keyLoaded']} ({status['source']})")
    return {
        **status,
        "groq": groq_live.status(),
        "nvidia": nvidia_live.status(),
        "deepseek": deepseek_live.status(),
        "activeProvider": llm_mod.provider(settings),
    }


class SetKeyIn(BaseModel):
    key: str = Field(min_length=20, max_length=300)


@app.post("/api/settings/gemini-key")
def set_gemini_key(body: SetKeyIn) -> dict:
    """Save/replace GEMINI_API_KEY in .env.local (the platform Keys UI has no
    edit for existing vars — this is the missing Edit, exposed in-app).
    Takes the raw key; stores it; returns the same status shape as Re-check
    plus a verification shape (first 6 / last 4 chars only — never the key)."""
    raw = body.key.strip()
    if raw.lower().startswith("encrypted:"):
        raise HTTPException(400, "That looks like an encrypted blob — paste the raw AIza... key.")
    if " " in raw:
        raise HTTPException(400, "The key contains spaces — paste the raw key with no quotes or spaces.")
    write_env_var("GEMINI_API_KEY", raw)
    # Mirror into the process env: the live resolver prefers OS env, so a stale
    # copy from an older .env.local would otherwise shadow the new value.
    sync_gemini_env_after_save(raw)
    live = live_gemini_key()
    live.refresh(force=True)
    status = live.status()
    logger.info("[config] GEMINI_API_KEY updated via dashboard — verification: "
                f"{raw[:6]}…{raw[-4:]} ({len(raw)} chars)")
    return {
        **status,
        "storedShape": {"first6": raw[:6], "last4": raw[-4:], "length": len(raw)},
    }


def _require_delete_confirmation(confirm: bool, env_name: str) -> None:
    """Refuse to destroy a key without an explicit `?confirm=true`.

    These deletes are irreversible from inside the app: the value is removed
    from .env.local and from the process env, and the platform's own copy is
    not something we can read back. That is a big, sharp button for something a
    stray request, a bad retry, or a careless test can trigger — and a key that
    is gone mid-session takes AI detection offline with it.

    428 (Precondition Required) rather than 400: nothing is wrong with the
    request, it is missing the deliberate step.
    """
    if not confirm:
        raise HTTPException(
            428,
            f"Deleting {env_name} is permanent. Re-send with ?confirm=true once you are "
            "sure — the value cannot be read back out of the server afterwards.",
        )


@app.delete("/api/settings/gemini-key")
def remove_gemini_key(confirm: bool = False) -> dict:
    """Delete GEMINI_API_KEY from .env.local (the missing Delete button)."""
    _require_delete_confirmation(confirm, "GEMINI_API_KEY")
    # Purge the process-env copies FIRST — the live resolver prefers OS env,
    # so a leftover injected value would keep keyLoaded=True after the file
    # line was removed.
    dropped = sync_gemini_env_for_delete()
    removed = delete_env_var("GEMINI_API_KEY")
    live = live_gemini_key()
    live.refresh(force=True)
    logger.info(
        f"[config] GEMINI_API_KEY deleted via dashboard "
        f"({removed} line(s) removed, process env dropped: {dropped or 'none'})"
    )
    return {**live.status(), "linesRemoved": removed, "envDropped": dropped}


# --------------------------------------------------------------------------- #
# caption style
# --------------------------------------------------------------------------- #


@app.post("/api/settings/groq-key")
def set_groq_key(body: SetKeyIn) -> dict:
    """Save/replace GROQ_API_KEY in .env.local.

    Mirrors set_gemini_key so a phone-only operator can paste a key from
    console.groq.com/keys without a terminal. Returns the live status plus a
    verification shape (first 6 / last 4 chars only — never the key).
    """
    raw = body.key.strip()
    if raw.lower().startswith("encrypted:"):
        raise HTTPException(400, "That looks like an encrypted blob — paste the raw gsk_... key.")
    if " " in raw:
        raise HTTPException(400, "The key contains spaces — paste the raw key with no quotes or spaces.")
    write_env_var("GROQ_API_KEY", raw)
    # Mirror into the process env: the live resolver prefers OS env, so a stale
    # copy from an older .env.local would otherwise shadow the new value.
    sync_groq_env_after_save(raw)
    live = live_groq_key()
    live.refresh(force=True)
    status = live.status()
    logger.info("[config] GROQ_API_KEY updated via dashboard — verification: "
                f"{raw[:6]}…{raw[-4:]} ({len(raw)} chars)")
    return {
        **status,
        "storedShape": {"first6": raw[:6], "last4": raw[-4:], "length": len(raw)},
    }


@app.delete("/api/settings/groq-key")
def remove_groq_key(confirm: bool = False) -> dict:
    """Delete GROQ_API_KEY from .env.local."""
    _require_delete_confirmation(confirm, "GROQ_API_KEY")
    dropped = sync_groq_env_for_delete()
    removed = delete_env_var("GROQ_API_KEY")
    live = live_groq_key()
    live.refresh(force=True)
    logger.info(
        f"[config] GROQ_API_KEY deleted via dashboard "
        f"({removed} line(s) removed, process env dropped: {dropped or 'none'})"
    )
    return {**live.status(), "linesRemoved": removed, "envDropped": dropped}


# --------------------------------------------------------------------------- #
# caption style
# --------------------------------------------------------------------------- #


@app.post("/api/settings/nvidia-key")
def set_nvidia_key(body: SetKeyIn) -> dict:
    """Save/replace NVIDIA_API_KEY in .env.local.

    Mirrors set_groq_key so a phone-only operator can paste a key from
    build.nvidia.com without a terminal. Returns the live status plus a
    verification shape (first 6 / last 4 chars only — never the key).
    """
    raw = body.key.strip()
    if raw.lower().startswith("encrypted:"):
        raise HTTPException(400, "That looks like an encrypted blob — paste the raw nvapi-... key.")
    if " " in raw:
        raise HTTPException(400, "The key contains spaces — paste the raw key with no quotes or spaces.")
    write_env_var("NVIDIA_API_KEY", raw)
    # Mirror into the process env: the live resolver prefers OS env, so a stale
    # copy from an older .env.local would otherwise shadow the new value.
    sync_nvidia_env_after_save(raw)
    live = live_nvidia_key()
    live.refresh(force=True)
    status = live.status()
    logger.info("[config] NVIDIA_API_KEY updated via dashboard — verification: "
                f"{raw[:6]}…{raw[-4:]} ({len(raw)} chars)")
    return {
        **status,
        "storedShape": {"first6": raw[:6], "last4": raw[-4:], "length": len(raw)},
    }


@app.delete("/api/settings/nvidia-key")
def remove_nvidia_key(confirm: bool = False) -> dict:
    """Delete NVIDIA_API_KEY from .env.local."""
    _require_delete_confirmation(confirm, "NVIDIA_API_KEY")
    dropped = sync_nvidia_env_for_delete()
    removed = delete_env_var("NVIDIA_API_KEY")
    live = live_nvidia_key()
    live.refresh(force=True)
    logger.info(
        f"[config] NVIDIA_API_KEY deleted via dashboard "
        f"({removed} line(s) removed, process env dropped: {dropped or 'none'})"
    )
    return {**live.status(), "linesRemoved": removed, "envDropped": dropped}


# --------------------------------------------------------------------------- #
# caption style
# --------------------------------------------------------------------------- #


@app.post("/api/settings/deepseek-key")
def set_deepseek_key(body: SetKeyIn) -> dict:
    """Save/replace DEEPSEEK_API_KEY in .env.local.

    Mirrors set_groq_key so a phone-only operator can paste a key from
    platform.deepseek.com without a terminal. Returns the live status plus a
    verification shape (first 6 / last 4 chars only — never the key).
    """
    raw = body.key.strip()
    if raw.lower().startswith("encrypted:"):
        raise HTTPException(400, "That looks like an encrypted blob — paste the raw sk-... key.")
    if " " in raw:
        raise HTTPException(400, "The key contains spaces — paste the raw key with no quotes or spaces.")
    if raw.startswith("nvapi-"):
        # Platform keys are `sk-…`; an `nvapi-…` pasted here is a real NVIDIA
        # key, and accepting it would produce a confusing 401 from DeepSeek
        # rather than pointing at the tab that actually issued it.
        raise HTTPException(
            400,
            "That is an NVIDIA key (nvapi-...). DeepSeek keys start with sk- and come "
            "from platform.deepseek.com/api_keys — save that one under NVIDIA instead.",
        )
    write_env_var("DEEPSEEK_API_KEY", raw)
    # Mirror into the process env: the live resolver prefers OS env, so a stale
    # copy from an older .env.local would otherwise shadow the new value.
    sync_deepseek_env_after_save(raw)
    live = live_deepseek_key()
    live.refresh(force=True)
    status = live.status()
    logger.info("[config] DEEPSEEK_API_KEY updated via dashboard — verification: "
                f"{raw[:6]}…{raw[-4:]} ({len(raw)} chars)")
    return {
        **status,
        "storedShape": {"first6": raw[:6], "last4": raw[-4:], "length": len(raw)},
    }


@app.delete("/api/settings/deepseek-key")
def remove_deepseek_key(confirm: bool = False) -> dict:
    """Delete DEEPSEEK_API_KEY from .env.local."""
    _require_delete_confirmation(confirm, "DEEPSEEK_API_KEY")
    dropped = sync_deepseek_env_for_delete()
    removed = delete_env_var("DEEPSEEK_API_KEY")
    live = live_deepseek_key()
    live.refresh(force=True)
    logger.info(
        f"[config] DEEPSEEK_API_KEY deleted via dashboard "
        f"({removed} line(s) removed, process env dropped: {dropped or 'none'})"
    )
    return {**live.status(), "linesRemoved": removed, "envDropped": dropped}


# --------------------------------------------------------------------------- #
# media download cookies (the platform Keys tab has no Edit/Delete for this
# variable, so the app exposes it in-app — same trap as the AI keys above).
# --------------------------------------------------------------------------- #
_COOKIES_VAR_TO_FIELD: dict[str, str] = {
    "YT_DLP_COOKIES_FILE": "cookies_file",
    "YT_DLP_COOKIES_FROM_BROWSER": "cookies_from_browser",
}


class SetStorageIn(BaseModel):
    """Media-offload (S3/R2) credentials.

    EVERY field is optional and blank means "keep whatever is already saved",
    not "clear it". That is what makes rotating a secret possible: R2 shows
    the access key id in plain text but the secret access key only once, so
    re-pasting all four values every time is how people end up with a typo in
    a field they were not changing. Clearing is an explicit delete.
    """

    endpoint: str | None = None
    bucket: str | None = None
    accessKey: str | None = None
    secretKey: str | None = None
    region: str | None = None
    prefix: str | None = None

    #: Request field -> env var. Only the two non-secret settings plus the four
    #: required ones; CLIPFORGE_OFFLOAD_DELETE_LOCAL is a behaviour switch, not
    #: part of a credential set, so it is not editable here.


#: Request field -> env var for the storage dialog. Module level, not a class
#: attribute: Pydantic v2 turns a leading-underscore class attribute into a
#: private attr, which is not readable off the class the way this needs.
STORAGE_FIELD_TO_VAR: dict[str, str] = {
    "endpoint": "CLIPFORGE_S3_ENDPOINT",
    "bucket": "CLIPFORGE_S3_BUCKET",
    "accessKey": "CLIPFORGE_S3_ACCESS_KEY",
    "secretKey": "CLIPFORGE_S3_SECRET_KEY",
    "region": "CLIPFORGE_S3_REGION",
    "prefix": "CLIPFORGE_S3_PREFIX",
}

#: The four fields objectstore.ObjectStoreConfig.enabled requires, as
#: (env var, ObjectStoreConfig attribute, label for the operator). Kept in one
#: place so the completeness check and config.py's S3_CREDENTIAL_VARS cannot
#: drift apart — a name off by one reports "offload is off" with no clue which
#: field is missing.
STORAGE_REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("CLIPFORGE_S3_ENDPOINT", "endpoint", "endpoint"),
    ("CLIPFORGE_S3_BUCKET", "bucket", "bucket"),
    ("CLIPFORGE_S3_ACCESS_KEY", "access_key", "access key"),
    ("CLIPFORGE_S3_SECRET_KEY", "secret_key", "secret key"),
)


# --------------------------------------------------------------------------- #
# media download cookies — YAML file path (new: cookies_file / cookies_from_browser)
# --------------------------------------------------------------------------- #
def _storage_field(body: SetStorageIn) -> dict[str, str]:
    """The non-blank fields of a save request, stripped and ready to write."""
    out: dict[str, str] = {}
    for name in ("endpoint", "bucket", "accessKey", "secretKey", "region", "prefix"):
        raw = getattr(body, name, None)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            out[name] = value
    return out


@app.post("/api/settings/s3-key")
def set_s3_key(body: SetStorageIn) -> dict:
    """Save/replace the S3/R2 offload credentials in .env.local.

    The LLM backends each got an in-app key manager; object storage did not,
    so the only way to enable offload was the platform Keys tab — which, as
    established for the AI keys, does not deliver values to this process. That
    left media offload unreachable without a terminal, which is the same trap
    this endpoint exists to remove for R2/S3.

    No restart: objectstore.config_from_env() is read per operation and the
    client cache is keyed on the resolved config, so saving here rebuilds the
    client on the next call.
    """
    given = _storage_field(body)
    if not given:
        raise HTTPException(
            400,
            "Nothing to save — every field was blank. Fill in at least the "
            "endpoint, bucket, access key and secret key.",
        )
    # The endpoint is typed far more often than the rest (R2 vs S3 vs a
    # path-style endpoint), so a wrong paste fails here with a clear message
    # instead of much later as an opaque botocore EndpointConnectionError.
    endpoint = given.get("endpoint") or os.getenv("CLIPFORGE_S3_ENDPOINT", "").strip()
    if endpoint and not endpoint.lower().startswith(("http://", "https://")):
        raise HTTPException(
            400,
            "The endpoint must be a full URL including https:// — for example "
            "https://<account-id>.r2.cloudflarestorage.com",
        )
    for name, value in given.items():
        if name in {"endpoint", "bucket"}:
            continue
        # Same guard as the AI keys: an encrypted blob pasted from the Keys tab
        # is unreadable by boto3 and would fail as an opaque 403.
        if value.lower().startswith("encrypted:"):
            raise HTTPException(
                400,
                f"{name} looks like an encrypted blob — paste the raw value from "
                "your S3/R2 provider, not the Keys tab.",
            )
    written: dict[str, str] = {}
    for name, value in given.items():
        env_name = STORAGE_FIELD_TO_VAR[name]
        write_env_var(env_name, value)
        written[env_name] = value
    sync_s3_env_after_save(written)

    cfg = objectstore.config_from_env()
    missing = [label for _var, attr, label in STORAGE_REQUIRED if not getattr(cfg, attr)]
    if missing:
        # Everything given was persisted — this is not a failure to save, it is
        # a still-incomplete set. Say which fields are still missing instead of
        # reporting success with offload quietly off.
        raise HTTPException(
            400,
            f"Saved, but media offload stays off until every field is set — still "
            f"missing: {', '.join(missing)}. Fill those in and save again.",
        )
    logger.info(
        "[config] S3/R2 credentials updated via dashboard "
        f"(bucket={cfg.bucket!r} endpoint={cfg.endpoint!r} "
        f"secret={len(cfg.secret_key)} chars, not logged)"
    )
    status = objectstore.status()
    return {
        **status,
        "savedFields": sorted(given),
        # Shape only. The secret never leaves the server, exactly as the AI
        # key endpoints do — but its length is genuinely useful for spotting a
        # truncated paste, which is the failure this endpoint exists to prevent.
        "storedShape": {
            "secretLength": len(cfg.secret_key),
            "accessKeyFirst4": cfg.access_key[:4],
        },
    }


@app.delete("/api/settings/s3-key")
def remove_s3_key(confirm: bool = False) -> dict:
    """Delete the S3/R2 credentials from .env.local.

    Same deliberate `?confirm=true` gate as the AI keys: this cannot be undone
    from inside the app, and the secret is unrecoverable once it is gone.
    Only the four required credentials are removed — region and prefix are not
    secret and survive, so re-adding a key does not mean retyping the prefix.
    """
    _require_delete_confirmation(confirm, "S3/R2 credentials")
    dropped = sync_s3_env_for_delete(S3_CREDENTIAL_VARS)
    removed = sum(delete_env_var(name) for name in S3_CREDENTIAL_VARS)
    logger.info(
        f"[config] S3/R2 credentials deleted via dashboard ({removed} line(s) "
        f"removed, process env dropped: {dropped or 'none'})"
    )
    status = objectstore.status()
    return {**status, "linesRemoved": removed, "envDropped": dropped}


# ---------------------------------------------------------------------------
# media download cookies — YAML file path (new: cookies_file / cookies_from_browser)
# ---------------------------------------------------------------------------
_COOKIES_VAR_TO_FIELD: dict[str, str] = {
    "YT_DLP_COOKIES_FILE": "cookies_file",
    "YT_DLP_COOKIES_FROM_BROWSER": "cookies_from_browser",
}

#: Request field -> env var for the live cookies dialog. Module level, not a
#: class attribute: Pydantic v2 turns a leading-underscore class attribute into a
#: private attr, which is not readable off the class the way this needs.
_COOKIES_FIELD_TO_VAR: dict[str, str] = {
    "cookies_file": "YT_DLP_COOKIES_FILE",
    "cookies_from_browser": "YT_DLP_COOKIES_FROM_BROWSER",
}

#: Process-wide live cookies-key state (created on first use).
#: Like the AI keys, the pasted cookies.txt is live account session material and
#: is treated as compromised after it is read here.
_live_cookies_key: _LiveKeyState | None = None


def _resolve_cookies_key(env: dict[str, str | None]) -> tuple[str | None, str, str | None]:
    """Find a usable cookies value: env-file contents beat the OS env.

    Returns (key, source, problem).
    """
    key = env.get("YT_DLP_COOKIES_FILE") or env.get("YT_DLP_COOKIES_FROM_BROWSER")
    if key:
        return key, "env-file",
    return None, "not found", "no cookies provided"


def set_cookies_key(body: dict[str, str | None]) -> dict[str, str]:
    """Save/replace the live cookies in .env.local.

    The cookies.txt is live account session material (SID, HSID, SSID, APISID,
    SAPISID, LOGIN_INFO, __Secure-3PSID, OSID, ACCOUNT_CHOOSER), so this is
    treated as compromised: the app exposes it in-app, and after saving it the
    user should sign out of all devices, change their Google password, enable
    two-factor authentication, and check recent security activity.
    """
    file_path = body.get("cookies_file") or ""
    browser_cookies = body.get("cookies_from_browser") or ""
    written: dict[str, str] = {}
    if file_path.strip():
        write_env_var("YT_DLP_COOKIES_FILE", file_path.strip())
        written["YT_DLP_COOKIES_FILE"] = file_path.strip()
    if browser_cookies.strip():
        write_env_var("YT_DLP_COOKIES_FROM_BROWSER", browser_cookies.strip())
        written["YT_DLP_COOKIES_FROM_BROWSER"] = browser_cookies.strip()

    # Refresh the process-level live key so every following request picks up
    # the new value without a restart.
    global _live_cookies_key
    _live_cookies_key = _LiveKeyState(
        ROOT,
        resolver=_resolve_cookies_key,
        aliases=("YT_DLP_COOKIES_FILE", "YT_DLP_COOKIES_FROM_BROWSER"),
        label="YT-DLP cookies",
    )
    _live_cookies_key.refresh(force=True)

    return {
        **written,
        "source": "file" if file_path.strip() else ("browser" if browser_cookies.strip() else "none"),
        "warning": (
            "cookies.txt is live account session material; treat as compromised "
            "and get a new password before the next render"
            if written
            else None
        ),
    }


@app.post("/api/settings/cookies")
def set_cookies_key(body: dict[str, str | None]) -> dict[str, str]:
    """Save/replace the live cookies in .env.local."""
    return set_cookies_key(body)


@app.delete("/api/settings/cookies")
def remove_cookies_key(confirm: bool = False) -> dict[str, str]:
    """Delete the live cookies from .env.local."""
    _require_delete_confirmation(
        confirm,
        "cookies (the pasted cookies.txt itself lives at /home/daytona/clipforge-data/cookies.txt)",
    )
    dropped = [
        name
        for name in ("YT_DLP_COOKIES_FILE", "YT_DLP_COOKIES_FROM_BROWSER")
        if os.environ.pop(name, None) is not None
    ]
    removed = sum(
        delete_env_var(name)
        for name in ("YT_DLP_COOKIES_FILE", "YT_DLP_COOKIES_FROM_BROWSER")
    )
    logger.info(
        f"[config] cookies deleted via dashboard ({removed} line(s) removed, process env dropped: {dropped or 'none'})"
    )
    # Drop the live key so a following request re-reads .env.local.
    global _live_cookies_key
    _live_cookies_key = None
    return {
        "linesRemoved": removed,
        "envDropped": dropped or "none",
    }


@app.get("/api/settings/cookies")
def get_cookies_key() -> dict[str, str]:
    """Read the current cookies values from .env.local (and the live key)."""
    if _live_cookies_key is not None:
        _live_cookies_key.refresh(force=True)
        key, source, problem = (
            _live_cookies_key.key,
            _live_cookies_key.source,
            _live_cookies_key.problem,
        )
    else:
        key = os.getenv("YT_DLP_COOKIES_FILE") or os.getenv("YT_DLP_COOKIES_FROM_BROWSER")
        source = "os-env" if key else "not found"
        problem = None
    return {
        "cookies_file": os.getenv("YT_DLP_COOKIES_FILE", "") or None,
        "cookies_from_browser": os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "") or None,
        "source": source or None,
        "problem": problem,
    }

def _freeze_caption_settings(job: dict) -> None:
    """Store the RESOLVED caption object on the job (preset + sliders).

    Written at submit time so the payload the UI sent is what renders, and
    so the dashboard can show exactly which style a batch wears.
    """
    overrides = dict(caption_style.read(settings.app.paths))
    preset = str(job.get("captionPreset") or "").strip()
    if preset:
        overrides["preset_key"] = preset
    if job.get("emojiEnabled") is not None:
        overrides["emoji_enabled"] = bool(job.get("emojiEnabled"))
    if job.get("emojiDensity"):
        overrides["emoji_density"] = job.get("emojiDensity")
    resolved = caption_style.resolve_caption_settings(settings, overrides)
    job["captionPreset"] = resolved["presetKey"]
    job["captionSettings"] = resolved
    job["captionStyleId"] = resolved["hash"]


class CaptionStyleIn(BaseModel):
    # All optional: the dialog sends the whole set it knows about, and a
    # missing key means "stop overriding this one".
    words_per_cue: float | None = None
    words_per_cue_max: float | None = None
    pause_sec: float | None = None
    highlight_color: str | None = None
    # These two were MISSING from this model while the dialog sent them, so
    # pydantic dropped them silently: "Caption size" and the preset chip never
    # reached the store, the renderer therefore never saw them, and every
    # render came out at the default size wearing the default look. That is
    # the whole "the sliders do nothing" report.
    size_scale: float | None = None
    preset_key: str | None = None
    emoji_enabled: bool | None = None
    emoji_density: str | None = Field(default=None, pattern="^(low|medium|high)$")
    position_mode: str | None = Field(default=None, pattern="^(video_bottom|frame|custom)$")
    position_y: float | None = None
    # "Reset to preset": drop the slider overrides but KEEP the chosen look.
    reset: bool | None = None


@app.get("/api/settings/captions")
def get_caption_style() -> dict:
    """The caption knobs, merged over config.toml, plus which are overridden."""
    return caption_style.effective(settings)


@app.put("/api/settings/captions")
def put_caption_style(body: CaptionStyleIn) -> dict:
    """Save the caption knobs the dashboard exposes.

    Saving does NOT re-encode anything: a clip's captions are burned into its
    pixels, so the change lands on the next render. The style fingerprint
    covers these values, which is what makes the existing "Re-render captions"
    button re-encode instead of skipping a clip it thinks is already styled
    correctly.

    No restart is needed and the running process is not mutated: the render
    stage re-reads the stored file when a job starts (caption_style.resolve).
    """
    payload = body.model_dump(exclude_none=True)
    if payload.pop("reset", False):
        # Keep the identity keys (preset, emoji, position) and drop every
        # slider override + the highlight colour, so the batch falls back to
        # exactly what the preset file says.
        keep = {caption_style.PRESET_KEY, caption_style.EMOJI_ENABLED_KEY,
                caption_style.EMOJI_DENSITY_KEY, caption_style.POSITION_MODE_KEY}
        payload = {k: v for k, v in payload.items() if k in keep}
    stored = caption_style.write(settings.app.paths, payload)
    resolved = caption_style.effective(settings)
    logger.info(f"[caption-style] saved {stored or '(defaults)'} -> styleId {resolved['hash']}")
    return {
        **resolved,
        "saved": stored,
        "styleId": resolved["hash"],
    }


@app.delete("/api/settings/captions")
def reset_caption_style() -> dict:
    """Drop every override and go back to config.toml."""
    caption_style.write(settings.app.paths, {})
    return caption_style.effective(settings)


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #

@app.post("/api/jobs")
def create_job(body: CreateJobIn) -> dict:
    title: str | None = None
    duration: float | None = None
    if body.sourceType == "upload":
        # This route accepts a FILENAME, not file bytes. Accepting it as an
        # upload created a job whose source file was never on disk: it sat at
        # 'queued', the fetch stage looked for downloads/<id>/source.mp4, and
        # the job failed with "uploaded file missing" — then re-queued itself
        # on every server restart, forever. Uploads have exactly one door:
        # POST /api/jobs/upload, which is the only route that receives bytes.
        raise HTTPException(
            400,
            "Uploads must be sent to /api/jobs/upload with the file attached — "
            "this endpoint only creates URL jobs.",
        )
    if body.sourceType == "url":
        info = download.probe_url(body.source)
        if info:
            title = info.get("title")
            duration = info.get("duration")
    if (
        duration is not None
        and duration > settings.hosted.max_source_min * 60
    ):
        raise HTTPException(
            413,
            f"Video is {duration / 60:.0f} min — this plan processes at most "
            f"{settings.hosted.max_source_min} min. Trim the source and try again.",
        )
    # Disk preflight: long videos download multi-GB sources; a full disk would
    # otherwise fail mid-pipeline with a confusing ffmpeg/whisper error.
    free = disk_stats(settings.app.paths.data_dir).get("freeBytes")
    if free is not None and free < MIN_FREE_BYTES:
        # The old message promised "files are deleted automatically 24h after a
        # job completes" as the way out. That is only true when retention is
        # switched ON, and retention_hours defaults to 0 (keep everything), so
        # on a default install the message told the user the disk would drain
        # itself while it never would. Say what is actually configured.
        retention_h = getattr(settings.hosted, "retention_hours", 0) or 0
        cleanup = (
            f"completed jobs are deleted automatically {retention_h}h after they "
            "finish"
            if retention_h > 0
            else "nothing is deleted automatically — delete old batches on the "
            "dashboard to reclaim space"
        )
        raise HTTPException(
            507,
            f"Server disk is almost full ({free // (1024 * 1024)} MB free) — "
            f"{cleanup}, then try again.",
        )
    job = store.create_job(
        source_type=body.sourceType,
        source=body.source,
        title=title,
        min_clip_sec=body.minClipSec,
        max_clip_sec=body.maxClipSec,
        max_clips=body.maxClips,
        caption_preset=body.captionPreset,
        language=body.language,
        duration=duration,
    )
    if body.emojiEnabled is not None:
        job["emojiEnabled"] = bool(body.emojiEnabled)
    if body.emojiDensity:
        job["emojiDensity"] = body.emojiDensity
    # Freeze the resolved caption object on the job. A batch must render the
    # style it was submitted with, not whatever the dashboard is set to when
    # the worker finally picks it up (queues are minutes long).
    _freeze_caption_settings(job)
    job_id = job["_id"]
    store.append_job_log(job_id, f"job created ({body.sourceType}: {body.source[:80]})")
    store.write_job(job)
    worker.enqueue(job_id)
    return {"jobId": job_id, "title": title, "duration": duration}


@app.post("/api/jobs/upload")
async def upload_job(
    file: UploadFile,
    minClipSec: int | None = None,
    maxClipSec: int | None = None,
    maxClips: int | None = None,
    captionPreset: str | None = None,
    language: str = "auto",
    emojiEnabled: bool | None = None,
    emojiDensity: str | None = None,
) -> dict:
    """Manual upload: always works, no URL fetching involved.

    The duration is checked before enqueueing so an oversized upload is rejected
    at submit time rather than failing halfway through transcription.
    """
    from app import download as download_svc

    job = store.create_job(
        source_type="upload",
        source="pending-upload",
        min_clip_sec=minClipSec,
        max_clip_sec=maxClipSec,
        max_clips=maxClips,
        caption_preset=captionPreset,
        language=language,
    )
    if emojiEnabled is not None:
        job["emojiEnabled"] = bool(emojiEnabled)
    if emojiDensity in ("low", "medium", "high"):
        job["emojiDensity"] = emojiDensity
    _freeze_caption_settings(job)
    job_id = job["_id"]
    store.write_job(job)
    dest_dir = store.job_source_dir_for_write(job_id)
    tmp = dest_dir / "upload.bin"
    max_mb = settings.hosted.max_source_mb
    written = 0
    try:
        with tmp.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                # Checked while streaming, so an oversized upload is abandoned
                # rather than written in full and then rejected — on a 4 GB disk
                # that is the difference between a clear error and a full volume.
                if max_mb and written > max_mb * 1024 * 1024:
                    fh.close()
                    tmp.unlink(missing_ok=True)
                    shutil.rmtree(dest_dir, ignore_errors=True)
                    store.delete_job(job_id)
                    raise HTTPException(
                        413,
                        f"That file is larger than the {max_mb} MB limit for this plan. "
                        f"Compress it or trim it and attach again.",
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        # Never leave a job behind for bytes that were not stored: that orphan
        # shows up as "pending-upload — source was lost" on the dashboard.
        tmp.unlink(missing_ok=True)
        shutil.rmtree(dest_dir, ignore_errors=True)
        store.delete_job(job_id)
        raise HTTPException(507, f"Could not store the upload: {exc}") from exc
    final = download_svc.save_upload(tmp, dest_dir, file.filename)
    try:
        uploaded_meta = video_svc.probe(final)
    except Exception as exc:
        final.unlink(missing_ok=True)
        store.delete_job(job_id)
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(400, f"Could not read the uploaded video: {exc}") from exc
    max_seconds = settings.hosted.max_source_min * 60
    if uploaded_meta["duration"] > max_seconds:
        final.unlink(missing_ok=True)
        store.delete_job(job_id)
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(
            413,
            f"Video is {uploaded_meta['duration'] / 60:.0f} min — this plan "
            f"processes at most {settings.hosted.max_source_min} min. Trim the source and try again.",
        )
    job = store.get_job(job_id) or job
    job["source"] = final.name
    # Push to durable storage BEFORE the job is queued, and before the local
    # copy is (optionally) dropped. If the push fails the local file stays and
    # the job runs exactly as it did before — an upload is the one artefact
    # there is no way to recreate, so this is never allowed to be lossy.
    offloaded = _offload_source(job_id, final)
    job["sourceRemote"] = bool(offloaded)
    # The bytes exist now, so the "Waiting for upload" gate is over: the job is
    # runnable and the worker may pick it up (LOCKED_SPEC §4).
    job["stage"] = "queued"
    job["progressDetail"] = "Upload received — queued."
    store.write_job(job)
    store.append_job_log(
        job_id,
        f"upload ingested: {final.name} ({final.stat().st_size if final.exists() else offloaded} bytes)"
        + (" | copied to durable storage" if offloaded else ""),
    )
    worker.enqueue(job_id)
    return {
        "jobId": job_id,
        "file": final.name,
        "bytes": final.stat().st_size if final.exists() else 0,
        "durable": bool(offloaded),
    }


def _offload_source(job_id: str, path: Path) -> bool:
    """Copy an uploaded source to durable storage, then free the local bytes.

    Returns whether the durable copy exists. Never raises: durability is a
    best-effort property, and a storage outage must not reject an upload the
    user already completed.
    """
    from app import objectstore

    cfg = objectstore.config_from_env()
    if not cfg.enabled or not path.exists():
        return False
    key = objectstore.upload_key(job_id, path.name)
    ok = objectstore.upload_file(path, key, cfg)
    if not ok:
        return False
    if objectstore.drop_local_after_push(True, path, cfg):
        # The pipeline reads the source straight back through
        # `find_source_file`, which materialises it from the bucket on demand,
        # so deleting it here changes nothing for the job that is about to run.
        logger.info(f"[api] job {job_id}: source offloaded, local copy dropped")
    return True


@app.get("/api/jobs")
def list_jobs(
    limit: int = 10, before: int | None = None, all: bool = False
) -> dict:
    """One page of batch summaries, newest first.

    Was `-> list[dict]` returning EVERY job with its full document — including
    up to 300 log lines each — and the dashboard re-fetched it every two
    seconds. Seventeen jobs was 285 KB per poll; the clip request the user was
    actually waiting for queued behind that, which is what "Clips are
    loading..." meant in practice.

    Now: a page of 10 (or every job with `?all=1`), a summary projection, and
    a cursor. The full document — logs included — is still what
    GET /api/jobs/{id} returns, and /api/jobs/{id}/logs is unchanged.
    """
    # A job whose worker died mid-stage leaves `stage` on a live value with
    # nothing behind it. Reconciled HERE — at the single point every read goes
    # through — so no view can disagree about whether it is running.
    #
    # Reconciling walks every job document and may re-queue one, and the
    # dashboard polls every two seconds. Doing that on every poll spent most of
    # the request budget proving that nothing had changed; a short floor keeps
    # the guarantee while removing the duplicate work.
    worker.reconcile_stale_running(min_interval_sec=LIST_RECONCILE_FLOOR_SEC)
    # When all=True, fetch every job — use a large limit so the pagination
    # layer returns everything instead of an empty page (limit=0 means "no
    # jobs" in list_job_summaries).
    page_limit = 0 if all else limit
    if all:
        page_limit = 999999  # effectively unlimited
    page, total, next_before = store.list_job_summaries(
        limit=page_limit, before=before
    )
    # live queue positions — a job waiting at 0% must be distinguishable from
    # one that is genuinely stuck
    positions = worker.queue_snapshot()
    for job in page:
        job["queuePosition"] = positions.get(job["_id"])
    # ONE derivation for every consumer, so the list card and the job card
    # cannot read the same document and reach opposite conclusions.
    return {
        "jobs": jobstatus.describe_all(page),
        "total": total,
        "nextBefore": next_before,
        "limit": 0 if all else limit,
    }


#: Shortest gap between two reconciliation passes. The dashboard polls the list
#: every 2s; reconciling on every one of those is wasted work, and the queue
#: snapshot it consults only changes when a job actually starts.
LIST_RECONCILE_FLOOR_SEC = float(os.getenv("CLIPFORGE_LIST_RECONCILE_SEC", "3"))

#: How long a serialized list response may be reused. Job progress is written
#: every few seconds by the runner, so a couple of seconds of staleness is
#: invisible — and it collapses a burst of dashboard polls into one parse.
LIST_CACHE_SEC = float(os.getenv("CLIPFORGE_LIST_CACHE_SEC", "2"))
_list_cache: dict[str, tuple[float, "Response"]] = {}


def _cached(key: str, build):
    now = time.monotonic()
    hit = _list_cache.get(key)
    if hit is not None and now - hit[0] < LIST_CACHE_SEC:
        return hit[1]
    value = build()
    _list_cache[key] = (now, value)
    return value


@app.get("/api/errors/recent")
def recent_api_errors() -> dict:
    """The last API calls the server answered with an error, newest last.

    Exists because "HTTP 400" with no body is what an intermediary looks like,
    and a bug report made of that is a bug report with no information in it.
    """
    return {"errors": list(_recent_api_errors)}


@app.get("/api/jobs/junk")
def list_junk_jobs() -> dict:
    """Failed batches with no delivered clips — the deletable ones.

    MUST be declared above `@app.get("/api/jobs/{job_id}")`: FastAPI matches in
    definition order, so the moment this sat below it, every call answered 404
    "job not found" with a path parameter called `junk`. That was the whole
    cleanup feature, dead on arrival, with an error message that pointed at the
    wrong thing.

    Offered as a suggestion rather than applied automatically: a failed job can
    still hold the only cached transcript and rendered clips of a video the
    user paid to process, and silently emptying their history is not a decision
    this endpoint gets to make for them.
    """
    jobs = store.list_jobs()
    out = []
    for job in jobs:
        if job.get("stage") not in {"failed", "complete"}:
            continue
        rendered, promised = store.clips_delivered(job["_id"], job)
        if rendered:
            continue
        out.append({
            "_id": job["_id"],
            "title": job.get("title") or job.get("source") or job["_id"],
            "stage": job.get("stage"),
            "createdAt": job.get("createdAt"),
            "reason": job.get("error") or "no clips rendered",
        })
    out.sort(key=lambda j: j.get("createdAt") or 0, reverse=True)
    return {"jobs": out}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    # "Check now" hits this route, so it must reconcile too — otherwise the
    # button that exists to tell the truth is the one place that cannot.
    # Forced: a human asked for the truth RIGHT NOW.
    worker.reconcile_stale_running(force=True)
    job = store.get_job(job_id) or job
    job["queuePosition"] = worker.queue_snapshot().get(job_id)
    return jobstatus.describe(job)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, keepSource: bool = False) -> dict:
    """Delete a batch and everything it wrote to disk.

    `keepSource` keeps the downloaded/uploaded source video and only removes
    the job record, workspace artifacts (transcript, detections, cuts) and
    rendered clips. The checkbox in the confirm dialog backs this: the source
    is the one artifact that costs real bytes, so the reader decides.
    """
    # Stop the work FIRST. Deleting the record while a child is still
    # downloading/transcribing/rendering just burns CPU and LLM calls on a job
    # nobody is waiting for — and the child would then write its state back.
    cancelled = worker.cancel(job_id)
    ok = store.delete_job(job_id)
    if not ok:
        raise HTTPException(404, "job not found")
    # Everything that job wrote to disk. The downloaded source (hundreds of MB
    # for a long video) and the rendered clips are NOT in the job store, so
    # without this a manual delete only freed a few KB of JSON and the disk
    # stayed full — the retention sweeper already does exactly this.
    remove_workspace(settings.app.paths.data_dir, job_id)
    unlink_model_dirs(settings.app.paths.data_dir, job_id)
    if keepSource:
        # Remove only the per-stage workspace + outputs, keep the source file.
        # `remove_workspace` already covered data/work/<id>; the source lives
        # under the per-job source dirs, so removing them now would destroy
        # the file the reader asked us to keep.
        shutil.rmtree(settings.app.paths.outputs_dir / job_id, ignore_errors=True)
    else:
        for d in settings.app.paths.job_source_dirs(job_id) + (
            settings.app.paths.outputs_dir / job_id,
        ):
            shutil.rmtree(d, ignore_errors=True)
    hosting_mod.invalidate_disk_cache()
    _purge_remote_job(job_id)
    return {"deleted": True, "cancelled": cancelled, "keptSource": keepSource}


@app.post("/api/jobs/{job_id}/delete")
def delete_job_post(job_id: str, body: DeleteJobBody | None = None) -> dict:
    """The same delete, over POST.

    DELETE /api/jobs/{id} answers 200 when it is called directly, through the
    Vite proxy, and through the preview's edge — all three were verified. What
    the browser actually did was still come back as a bodyless "HTTP 400", and
    a verb is the one thing about that request a proxy is free to disagree
    about. POST is what every other mutation in this API already uses, is
    preflight-free, and is never filtered by a method allowlist, so the UI uses
    it and this route is the one the dashboard calls. The DELETE route stays
    for anything scripted against the API.
    """
    return delete_job(job_id, keepSource=bool(body and body.keepSource))


@app.post("/api/jobs/bulk-delete")
def bulk_delete_jobs(body: BulkDeleteIn) -> dict:
    """Delete many batches in one call — the test-batch janitor.

    Clearing a dashboard down to the batches worth keeping was, until now, one
    DELETE per batch with a confirmation dialog per batch. The dialogs are
    right for a single delete and unbearable for forty, and the junk this is
    for ("source.mp4 · 2 clips", repeated) always arrives as a crowd.

    `jobs` is an explicit id list — there is no "delete everything matching X"
    mode, because a typo in a filter is unrecoverable in a way a typo in a list
    of ids you can see is not.
    """
    if not body.jobIds:
        raise HTTPException(400, "no job ids given")
    if len(body.jobIds) > 200:
        raise HTTPException(400, "too many ids in one request (max 200)")
    deleted: list[str] = []
    missing: list[str] = []
    for job_id in body.jobIds:
        if not worker.cancel(job_id) and store.get_job(job_id) is None:
            missing.append(job_id)
            continue
        if not store.delete_job(job_id):
            missing.append(job_id)
            continue
        remove_workspace(settings.app.paths.data_dir, job_id)
        unlink_model_dirs(settings.app.paths.data_dir, job_id)
        for d in settings.app.paths.job_source_dirs(job_id) + (
            settings.app.paths.outputs_dir / job_id,
        ):
            shutil.rmtree(d, ignore_errors=True)
        hosting_mod.invalidate_disk_cache()
        _purge_remote_job(job_id)
        deleted.append(job_id)
    logger.info(
        f"[api] bulk delete: {len(deleted)} removed, {len(missing)} not found"
    )
    return {"deleted": deleted, "missing": missing}


@app.post("/api/jobs/{job_id}/restore")
def restore_job(job_id: str) -> dict:
    """Undo a delete: job + clip docs + rendered media come back from trash.

    Deletes used to be a one-way door — the retention sweep emptied whole
    libraries with no undo — so this exists to make a mistake recoverable
    without waiting for someone to restore a backup.
    """
    ok = store.restore_job(job_id)
    if not ok:
        raise HTTPException(404, "nothing to restore for this job")
    return {"restored": True, "jobId": job_id}


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str) -> dict:
    """Stop a job at its next safe boundary (never mid-write)."""
    try:
        return worker.pause(job_id)
    except KeyError:
        raise HTTPException(404, "job not found") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict:
    """Resume a paused job from its last completed checkpoint."""
    try:
        depth = worker.resume(job_id)
    except KeyError:
        raise HTTPException(404, "job not found") from None
    return {"resumed": True, "queueDepth": depth}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    """Retry a job, resuming from cached videos and stages.

    The backend now checkpoints after every finished clip and re-renders only
    the unfinished ones, so a Retry from a failed render-stage job resumes from
    the last completed clip instead of starting over. Failed upload jobs with no
    source file are still rejected with a readable message.
    """
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.get("stage") != "failed":
        raise HTTPException(400, "only failed jobs can be retried")
    dead_end = worker_mod.unrecoverable_reason(store, job)
    if dead_end is not None:
        # Record the reason on the job as well as returning it: the card is
        # rendered from the document, so a refusal that only lives in the HTTP
        # response leaves the old, wrong message on screen (this is how a job
        # kept showing a memory-guard message from a build that no longer
        # exists instead of "re-upload the file").
        job.update({
            "stage": "failed",
            "error": dead_end,
            "errorDetail": None,
            "interrupted": False,
            "interruptReason": None,
        })
        store.write_job(job)
        raise HTTPException(400, dead_end)
    try:
        depth = worker.retry_from_cache(job_id)
    except KeyError:
        raise HTTPException(404, "job not found") from None
    return {"queued": True, "queueDepth": depth}


# --------------------------------------------------------------------------- #
# internal worker API — the remote worker-agent (over Tailscale) reports back
# --------------------------------------------------------------------------- #


@app.post("/api/internal/jobs/{job_id}/progress")
def internal_progress(job_id: str, body: InternalProgressIn, request: Request) -> dict:
    _require_worker_token(request)
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job.update({
        "stage": body.stage,
        "progress": body.progress,
        "progressDetail": body.progressDetail or job.get("progressDetail"),
        "unitLabel": body.unitLabel,
        "unitsDone": body.unitsDone,
        "unitsTotal": body.unitsTotal,
        "lastProgressAt": int(time.time() * 1000),
    })
    store.write_job(job)
    return {"ok": True}


@app.post("/api/internal/jobs/{job_id}/logs")
def internal_logs(job_id: str, body: InternalLogIn, request: Request) -> dict:
    _require_worker_token(request)
    if store.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    for line in body.lines:
        store.append_job_log(job_id, line[:500])
    return {"ok": True}


@app.post("/api/internal/jobs/{job_id}/failed")
def internal_failed(job_id: str, body: InternalFailedIn, request: Request) -> dict:
    _require_worker_token(request)
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job.update({
        "stage": "failed",
        "error": body.error[:500],
        "errorDetail": (body.detail or "")[:1500],
        "completedAt": int(time.time() * 1000),
    })
    store.write_job(job)
    store.append_job_log(job_id, f"FAILED (remote worker): {body.error[:200]}")
    return {"ok": True}


@app.post("/api/internal/jobs/{job_id}/complete")
def internal_complete(job_id: str, body: InternalCompleteIn, request: Request) -> dict:
    _require_worker_token(request)
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job.update({
        "stage": "complete",
        "progress": 100,
        "progressDetail": f"{body.clipCount} clips ready (rendered on your worker).",
        "clipCount": body.clipCount,
        "totalClipsSec": body.totalClipsSec,
        "captionsTxt": body.captionsTxt,
        "completedAt": int(time.time() * 1000),
    })
    store.write_job(job)
    store.append_job_log(job_id, f"remote worker finished — {body.clipCount} clips")
    return {"ok": True}


class RegisterWorkerIn(BaseModel):
    workerUrl: str = Field(min_length=8, max_length=300)
    token: str = ""


@app.post("/api/internal/register-worker")
def register_worker(body: RegisterWorkerIn) -> dict:
    """The home worker-agent can self-register its tailnet URL.

    Disabled unless CLIPFORGE_ALLOW_SELF_REGISTER=true (else it would let any
    internet caller repoint the dispatch target).
    """
    import os as _os

    if _os.getenv("CLIPFORGE_ALLOW_SELF_REGISTER", "").lower() not in {"1", "true", "yes"}:
        raise HTTPException(403, "self-registration disabled — set WORKER_URL via the platform env")
    settings.hosted.worker_url = body.workerUrl.rstrip("/")
    if body.token.strip():
        settings.hosted.worker_token = body.token.strip()
    logger.info(f"[hosting] worker self-registered: {settings.hosted.worker_url}")
    return {"ok": True, "workerUrl": settings.hosted.worker_url}


@app.post("/api/jobs/{job_id}/retranscribe")
def retranscribe_job(job_id: str) -> dict:
    """Re-run transcription (and detection) for a finished batch.

    The repair path for anything wrong with WHAT WAS HEARD: a transcription
    made before word confidence was recorded, a better model, a threshold
    change. Unlike /rerender, which reuses the cached words, this invalidates
    the transcript — so every clip is re-encoded afterwards and the .srt files
    are rewritten from the new cues.

    It also re-runs detection, because the cuts are derived from the words:
    keeping candidates scored against a transcript that no longer exists would
    leave word indices pointing at the wrong places. That costs one AI call per
    batch, which the alternative (stale cues over new words) is not worth.
    """
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.get("stage") in {"queued", "grabbing", "transcribing", "finding", "rendering"}:
        raise HTTPException(409, "this job is still running — wait for it to finish")
    if store.find_source_file(job_id) is None:
        raise HTTPException(400, "source file no longer on disk — re-submit the URL to transcribe again")
    if not store.list_clips(job_id):
        raise HTTPException(400, "this batch has no clips")

    # "transcribe" and everything after it: detect, reframe, captions, export.
    # "fetch" stays done, so the downloaded video is not re-fetched.
    ws = StageWorkspace(settings.app.paths.data_dir, job_id)
    # Remove only the words/cues/captions the repaired run must re-derive.
    # The extracted audio cache that transcribe and the render stages both
    # read is NOT part of this clear: on a 1-CPU / 2-GB host the 600s minimum
    # useful window is real, and deleting the scratch before the re-queued
    # attempt can finish turns a stalled attempt into an infinite re-queue
    # loop that leaves the dashboard showing "Transcribing... 0%". The source
    # video is never removed by a repair — that is a user delete.
    ws.clear_from("transcribe", leave_scratch_audio=True)
    for clip in store.list_clips(job_id):
        clip["renderState"] = "pending"
        store.write_clip(clip)
    job.update({
        "stage": "queued",
        "progress": 0,
        "progressDetail": "Re-transcribing — the source is reused, words are re-decoded with confidence scores.",
        "error": None,
        "errorDetail": None,
        "completedAt": None,
        "transcriptWordCount": None,
        "transcriptLabel": None,
        "asrPeakRssMb": None,
        "unitLabel": None,
        "unitsDone": None,
        "unitsTotal": None,
    })
    store.write_job(job)
    store.append_job_log(
        job_id,
        "re-transcribe requested — transcript and detection invalidated; the "
        "downloaded source is reused, so nothing is fetched again",
    )
    worker.enqueue(job_id)
    return {"queued": True, "resumesAt": ws.first_incomplete()}


def _clip_emoji_track(store: LocalStore, clip: dict[str, Any]) -> dict[str, Any]:
    """The stored emoji track for one clip (sidecar first, doc as the fallback)."""
    job_id = str(clip.get("jobId") or "")
    order = int(clip.get("order") or 0)
    sidecar = (
        store._paths.data_dir / "work" / job_id / "captions" / f"emoji_{order:03d}.json"
    )
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "source": clip.get("emojiSource"),
        "tags": clip.get("emojiTags") or [],
        "selected": clip.get("emojiTrack") or [],
    }


def _write_clip_emoji_track(store: LocalStore, clip: dict[str, Any], track: dict[str, Any]) -> None:
    job_id = str(clip.get("jobId") or "")
    order = int(clip.get("order") or 0)
    path = store._paths.data_dir / "work" / job_id / "captions" / f"emoji_{order:03d}.json"
    if path.parent.exists():
        path.write_text(json.dumps(track, ensure_ascii=False, indent=1), encoding="utf-8")
    clip["emojiTrack"] = track.get("selected") or []
    clip["emojiSource"] = track.get("source")
    store.write_clip(clip)


def _rerender_clip(store: LocalStore, clip_id: str) -> None:
    """Re-encode ONE clip with the emoji track exactly as it now stands.

    Runs on a daemon thread: a render is ~45 s, far past any sane HTTP
    timeout. Failure is recorded on the clip doc, never swallowed.
    """
    from dataclasses import replace as _replace

    from app import encode as _encode
    from app import translate as _translate
    from app.config import load_config as _load_config
    from app.stages import StageWorkspace
    from app.subtitle import Transcript, Word

    clip = store.get_clip_doc(clip_id)
    if not clip:
        return
    job_id = str(clip.get("jobId") or "")
    order = int(clip.get("order") or 0)
    try:
        settings = _load_config()
        job = store.get_job(job_id)
        ws = StageWorkspace(settings.app.paths.data_dir, job_id)
        cuts = ws.read_json("reframe", "cuts.json") or []
        cut = next((c for c in cuts if int(c.get("order", -1)) == order), None)
        if cut is None:
            raise RuntimeError(f"no cut for clip order {order}")
        cues = (ws.read_json("captions", f"cues_{order:03d}.json") or {}).get("cues") or []
        if not cues:
            raise RuntimeError("no caption cues stored for this clip")
        track = _clip_emoji_track(store, clip)
        src = settings.app.paths.outputs_dir.parent / "downloads" / job_id / "source.mp4"
        if not src.exists():
            candidate = settings.app.paths.data_dir / "downloads" / job_id / "source.mp4"
            src = candidate if candidate.exists() else src
        dest = settings.app.paths.outputs_dir / job_id / f"clip_{order:03d}.mp4"
        store.write_clip(clip_id, {"renderState": "running", "mediaError": None}, job_id=job_id)
        meta = _encode.render_clip(
            src, cut, dest, settings, cues=cues,
            preset_key=str(job.get("captionPreset") or settings.subtitle.preset or "bold_pop"),
            emoji_selection=track.get("selected") or [],
        )
        store.write_clip(
            clip_id,
            {
                "renderState": "done",
                "mediaError": None,
                "videoPath": meta["videoPath"],
                "mediaBytes": meta["bytes"],
                "mediaWidth": meta["width"],
                "mediaHeight": meta["height"],
                "mediaDurationSec": meta["durationSec"],
                "renderMs": meta.get("renderMs"),
            },
            job_id=job_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface it on the clip, not in a log nobody reads
        store.write_clip(
            clip_id,
            {"renderState": "failed", "mediaError": f"emoji re-render: {exc}"[:500]},
            job_id=job_id or None,
        )


@app.post("/api/clips/{clip_id}/emoji")
def edit_clip_emoji(clip_id: str, body: EmojiEditIn) -> dict:
    """Add, swap or remove ONE chunk's emoji, then re-render that clip.

    The judgement (emotion, intensity, confidence, reason) is kept from the
    original tag: only the emoji changes, so the track still says why the
    chunk was picked. A hand-added emoji is marked ``byHandle`` so the UI never
    presents it as the model's choice.
    """
    from app import emoji as _emoji
    from app.stages import StageWorkspace

    clip = store.get_clip_doc(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    track = _clip_emoji_track(store, clip)
    cues = (StageWorkspace(load_config().app.paths.data_dir, str(clip.get("jobId") or ""))
            .read_json("captions", f"cues_{int(clip.get('order') or 0):03d}.json") or {}).get("cues") or []
    if body.i >= len(cues):
        raise HTTPException(400, f"caption {body.i} does not exist ({len(cues)} cues)")
    if body.emoji is not None and not _emoji.is_allowed(body.emoji):
        raise HTTPException(400, f"{body.emoji!r} is not in the emoji whitelist")

    cue = cues[body.i]
    start = float(cue.get("start") or 0.0)
    cue_end = float(cue.get("end") or start)
    # The model tags coarser chunks than the burned caption lines, so an edit is
    # matched by TIME: any emoji already covering this cue's span is replaced,
    # never stacked. Two overlays in the same second read as a rendering bug.
    selected = [
        s for s in track.get("selected") or []
        if float(s.get("end") or 0.0) <= start + 1e-6
        or float(s.get("start") or 0.0) >= cue_end - 1e-6
    ]
    if body.emoji is not None:
        tag = next((t for t in track.get("tags") or [] if int(t.get("i", -1)) == body.i), {})
        span = min(_emoji.MAX_DURATION, max(_emoji.MIN_DURATION, cue_end - start))
        selected.append({
            "i": body.i,
            "emoji": body.emoji,
            "emotion": tag.get("emotion") or "manual",
            "intensity": tag.get("intensity"),
            "confidence": tag.get("confidence"),
            "reason": (
                f"added by hand (model said: {tag['reason']})" if tag.get("reason")
                else "added by hand"
            ),
            "sensitive": bool(tag.get("sensitive")),
            "start": round(start, 3),
            "end": round(start + span, 3),
            "cueText": str(cue.get("text") or ""),
            "byHandle": True,
        })
    selected.sort(key=lambda s: float(s.get("start") or 0.0))
    track["selected"] = selected
    track["editedByHand"] = True
    _write_clip_emoji_track(store, clip, track)
    if body.rerender:
        threading.Thread(
            target=_rerender_clip, args=(store, clip_id),
            name=f"emoji-rerender-{clip_id[:8]}", daemon=True,
        ).start()
    return {"emojiTrack": selected, "emojiSource": track.get("source"), "rerendering": body.rerender}


@app.get("/api/emoji/whitelist")
def emoji_whitelist() -> dict:
    """The curated list the editor's picker offers (and the renderer accepts)."""
    from app import emoji as _emoji

    wl = _emoji.load_whitelist()
    return {
        "source": wl.source,
        "emojis": [
            {"emoji": e.emoji, "emotions": list(e.emotions), "keywords": list(e.keywords)}
            for e in wl.entries.values()
        ],
    }


@app.post("/api/jobs/{job_id}/rerender")
def rerender_job(job_id: str) -> dict:
    """Re-encode a finished batch with fresh captions, reusing the transcript.

    This is the repair path for anything wrong with the BURNED-IN text — a
    caption-style fix, a threshold change, a bad `.ass` (the karaoke preset
    once welded words together on screen: "bro." + "Did" rendered as
    "bro.Did"). Re-encoding a 10-clip batch costs minutes of CPU and no money:
    the transcript and the detection result are read back from the stage cache,
    so nothing is re-transcribed and no LLM call is made. That is the
    difference between this and Re-detect, which re-scores the video and costs
    a detection call.
    """
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.get("stage") in {"queued", "grabbing", "transcribing", "finding", "rendering"}:
        raise HTTPException(409, "this job is still running — wait for it to finish")
    src = store.find_source_file(job_id)
    if src is None:
        raise HTTPException(400, "source file no longer on disk — this batch cannot be re-rendered")
    clips = store.list_clips(job_id)
    if not clips:
        raise HTTPException(400, "this batch has no clips to re-render")

    # Only the caption/reframe stages are invalidated. "transcribe" and "detect"
    # stay done, so the run resumes with the cached words and the cached cuts.
    ws = StageWorkspace(settings.app.paths.data_dir, job_id)
    ws.clear_from("reframe")
    reset = 0
    for clip in clips:
        if clip.get("renderState") == "done":
            clip["renderState"] = "pending"
            store.write_clip(clip)
            reset += 1
    job.update({
        "stage": "queued",
        "progress": 0,
        "progressDetail": "Re-rendering with fresh captions — transcript and cuts are reused.",
        "error": None,
        "errorDetail": None,
        "completedAt": None,
        "unitLabel": None,
        "unitsDone": None,
        "unitsTotal": None,
    })
    store.write_job(job)
    store.append_job_log(
        job_id,
        f"re-render requested — {reset} clip(s) reset; the transcript and the "
        "detection are reused, so this costs no transcription and no AI call",
    )
    worker.enqueue(job_id)
    return {"queued": True, "clipsReset": reset, "resumesAt": ws.first_incomplete()}


@app.post("/api/jobs/{job_id}/redetect")
def redetect_job(job_id: str) -> dict:
    """Re-run highlight detection on an existing job and re-render the clips."""

    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    # Reuse the same readiness logic the dashboard shows, rather than a
    # Gemini-only check left over from before Groq existed. Asking the user to
    # "add GEMINI_API_KEY" when Groq is the active provider was wrong twice
    # over: it named the wrong key, and it refused outright for anyone who had
    # only ever set a Groq key.
    status = detection_status()
    if not status.get("ready"):
        provider = status.get("provider", "detection")
        raise HTTPException(
            400,
            f"AI detection offline — {status.get('detail') or f'the {provider} backend is not ready'}",
        )
    src = store.find_source_file(job_id)
    if src is None:
        raise HTTPException(400, "source file no longer on disk — re-run the job from a fresh URL/upload")
    # Re-run detection onward. The stage workspace keeps fetch + transcribe
    # artifacts, so the job resumes at 'detect' instead of starting over.
    ws = StageWorkspace(settings.app.paths.data_dir, job_id)
    ws.clear_from("detect")
    job = store.get_job(job_id) or job
    job.update({"stage": "queued", "progress": 0, "error": None, "errorDetail": None})
    store.write_job(job)
    worker.enqueue(job_id)
    return {"queued": True, "resumesAt": ws.first_incomplete()}


@app.get("/api/jobs/{job_id}/logs")
def job_logs(job_id: str) -> list[str]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.get("logs", [])


# --------------------------------------------------------------------------- #
# clips + media
# --------------------------------------------------------------------------- #

@app.get("/api/clips")
def list_clips(jobId: str | None = None, limit: int = 96, withCues: bool = False) -> Response:
    """Clip documents for ONE batch, cached for a couple of seconds.

    Only ever called for the batch the user has actually opened — the grid for
    every other batch is never rendered — so the answer is short and stable
    between polls. `list_clips` walks the clip directory and matches on
    `jobId`, which is the index this store lacks; the per-job result is
    memoised here instead.

    The shared preview URL is served out of the same list cache as the rest of
    the dashboard. The job-scoped path always carries the rendered caption cues
    now, because the preview overlay reads them directly from the same response
    the card uses — no second per-clip request is needed before a video can show
    its burned-in text.
    """
    if jobId:
        return _cached(
            f"clips:{jobId}",
            lambda: JSONResponse(
                _attach_share_state(
                    _list_clips_with_captions(store, jobId, with_cues=True),
                )
            ),
        )
    return _cached(
        f"clips-all:{limit}",
        lambda: JSONResponse(_list_clips_with_captions(store, None, limit, with_cues=False)),
    )


#: Cache policy for rendered media.
#:
#: `no-cache` still allows caching but forces revalidation, which is exactly
#: what is wanted for a URL whose contents change under the same id: a clip is
#: re-rendered in place after "Re-detect"/"Re-transcribe", and with a long
#: `max-age` the browser keeps showing the previous encode — the classic "I
#: re-rendered and it still plays the old clip". `must-revalidate` stops any
#: intermediary (the preview proxy included) from serving a stale body without
#: asking. Range requests are unaffected: Starlette answers 206 from the file
#: on disk, with Content-Length, Content-Range and Accept-Ranges.
MEDIA_CACHE_CONTROL = "no-cache, must-revalidate"

#: The share copy is immutable and content-addressed by its ETag, so it is safe
#: — and necessary — to let a CDN and the browser keep it for a year. Without
#: this the 3 MB share file is re-fetched on every page load, which is the exact
#: cost the prefetch cache exists to avoid.
SHARE_CACHE_CONTROL = "public, max-age=31536000, immutable"


@app.api_route("/api/clips/{clip_id}/video", methods=["GET", "HEAD"])
def clip_video(clip_id: str) -> FileResponse:
    """Stream the rendered clip for <video> playback.

    Content-Disposition stays inline — 'attachment' makes browsers refuse to
    play the stream. Downloads go through /download instead.

    Range support is what makes seeking and progressive playback work at all:
    Starlette's FileResponse answers a `Range:` header with 206 Partial
    Content plus Content-Range/Accept-Ranges, and a bare GET with a real
    Content-Length. It is asserted rather than assumed in
    tools/playback_test.py, because a regression here (a hand-rolled
    StreamingResponse, or a proxy that strips Range) is invisible until every
    clip in the UI stops playing.

    `immutable` + an ETag derived from the file's size and mtime is what lets
    a CDN and the browser cache this for a year. That is only safe because the
    ETag changes when the bytes do: a re-render produces a new mtime, so the
    old URL is never reused for new footage.
    """
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    p = _clip_media(clip, "video")
    if p is None:
        raise HTTPException(404, "clip file missing on disk")
    src = p
    cut = {
        "start": float(clip.get("start") or 0.0),
        "end": float(clip.get("end") or 0.0),
        "cropBox": {
            "x": int(clip.get("crop_x") or 0),
            "y": int(clip.get("crop_y") or 0),
            "w": int(clip.get("crop_w") or 0),
            "h": int(clip.get("crop_h") or 0),
        },
        "sampleWindows": [
            {
                "t": float(w.get("t") or 0.0),
                "x": int(w.get("x") or 0),
                "y": int(w.get("y") or 0),
                "w": int(w.get("w") or 0),
                "h": int(w.get("h") or 0),
                "cut": bool(w.get("cut") or False),
            }
            for w in (clip.get("sampleWindows") or [])
        ],
    }
    return FileResponse(
        p,
        media_type="video/mp4",
        headers={
            "Cache-Control": MEDIA_CACHE_CONTROL,
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{p.name}"',
            "ETag": _media_etag(p),
        },
    )





def _purge_remote_job(job_id: str) -> None:
    """Remove a deleted job's objects from durable storage too.

    Deleting a batch that was offloaded has to mean deleted. Without this the
    bucket would keep the source video and every clip forever, which is both a
    privacy problem and a bill that grows with every test batch.
    """
    from app import objectstore

    cfg = objectstore.config_from_env()
    if not cfg.enabled:
        return
    s3 = objectstore.client(cfg)
    if s3 is None:
        return
    prefix = f"{cfg.prefix}"
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=cfg.bucket, Prefix=f"{prefix}/"):
            keys = [
                {"Key": o["Key"]}
                for o in page.get("Contents", [])
                if f"/{job_id}/" in o["Key"]
            ]
            for i in range(0, len(keys), 1000):
                s3.delete_objects(
                    Bucket=cfg.bucket, Delete={"Objects": keys[i:i + 1000]}
                )
    except Exception as exc:  # noqa: BLE001 - a stale object is not worth a 500
        logger.warning(f"[objectstore] could not purge remote objects for {job_id}: {exc}")


def _clip_media(clip: dict, kind: str) -> Path | None:
    """Locate a clip's video/poster on disk, healing a drifted stored path.

    Clip documents carry an absolute ``videoPath``/``thumbPath`` written when
    the clip was rendered. Those paths stop resolving when the media tree is
    relocated (the hosted storage move leaves documents pointing at the old
    root) or when a job is restored onto a new filesystem. Falling straight
    through to 404 is what made a perfectly good batch look like a wall of
    black, unplayable cards.

    So: try the recorded path, then the canonical
    ``outputs/<job>/clip_<order>.mp4`` name, then the recorded basename inside
    the job's output directory. The first hit is written back to the document,
    so the repair costs one lookup, not one per request.
    """
    field = "videoPath" if kind == "video" else "thumbPath"
    suffix = ".mp4" if kind == "video" else ".thumb.jpg"
    order = int(clip.get("order") or 0)
    job_id = str(clip.get("jobId") or "")
    raw = clip.get(field)
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw))
    if job_id:
        out_dir = settings.app.paths.outputs_dir / job_id
        candidates.append(out_dir / f"clip_{order:03d}{suffix}")
        if raw:
            candidates.append(out_dir / Path(raw).name)
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if not _restore_media(cand):
            continue
        if raw != key:
            try:
                store.write_clip(
                    str(clip.get("_id")), {field: key}, job_id=job_id or None
                )
                logger.info(f"[api] clip {clip.get('_id')}: {field} healed to {key}")
            except Exception as exc:  # noqa: BLE001 - serving matters, healing is a bonus
                logger.warning(f"[api] could not persist healed {field}: {exc}")
        return cand
    return None


def _media_etag(p: Path) -> str:
    """Weak-ish ETag from size+mtime: changes exactly when the bytes change."""
    try:
        st = p.stat()
    except OSError:
        return '"0"'
    return f'"{int(st.st_mtime)}-{st.st_size}"'


def _restore_media(p: Path) -> bool:
    """Pull an offloaded media file back onto disk before serving it.

    Rendered clips are offloaded once they are finished, so a sandbox rebuild
    leaves the job documents pointing at paths that no longer exist. Every
    serving route calls this first, which means the routes themselves — and
    therefore Range support, ETags, caching and the frontend — are unchanged:
    by the time a FileResponse is built the file is local again. Returns False
    when there is nothing local and nothing in durable storage, which leaves
    each route's existing 404 exactly as it was.
    """
    if p.exists() and p.stat().st_size > 0:
        return True
    from app import objectstore

    key = objectstore.key_for_path(p, settings.app.paths.outputs_dir)
    if not key:
        return False
    if objectstore.materialize(key, p):
        logger.info(f"[api] restored {p.name} from durable storage")
        return True
    return False


@app.api_route("/api/clips/{clip_id}/share", methods=["GET", "HEAD"])
def clip_share(clip_id: str) -> FileResponse:
    """A small, share-sized copy of the clip, built on first request.

    This is what the Web Share sheet consumes. It exists because
    `navigator.share({files})` needs the whole file as a `File` synchronously
    inside the click handler, and the 1080p master is 6-10 MB — long enough on
    a phone that the user-activation window expires and `share()` fails.

    The copy is 720x1280 H.264 + AAC 96k with +faststart, capped at 3 MB (see
    app/share_video.py). It is built lazily and cached on disk, keyed by the
    master's size+mtime, so a re-render rebuilds rather than serving a stale
    lighter version.

    Cache-Control is `immutable` and safe here for the same reason as on
    /video: the ETag is derived from the file, so new bytes are a new identity.

    A transcode failure is NOT a failed request. The client falls back to
    sharing the caption plus the clip URL, which always works, so this returns
    503 with a readable reason instead of an opaque 500.
    """
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "rendered video not available for this clip")
    if not clip.get("videoPath"):
        # 409, matching /post: the clip EXISTS, it just has no render yet. A 404
        # would tell the client the clip is gone and offer no way forward, when
        # the honest answer is "not rendered yet".
        raise HTTPException(409, "this clip has no rendered MP4 to share yet")
    master = Path(clip["videoPath"])
    if not _restore_media(master):
        raise HTTPException(404, f"clip file missing on disk: {master.name}")
    try:
        share = share_video.build_share_version(master)
    except share_video.ShareRenderError as exc:
        # 503 + a reason the UI can act on. The whole point of the link
        # fallback is that a missing share copy must never be a dead end.
        logger.warning(f"[share] {master.name}: {exc}")
        raise HTTPException(
            503, f"share-sized copy unavailable ({exc}); share the link instead",
        ) from exc
    return FileResponse(
        share,
        media_type="video/mp4",
        headers={
            "Cache-Control": SHARE_CACHE_CONTROL,
            "Accept-Ranges": "bytes",
            "ETag": _media_etag(share),
        },
    )


# --------------------------------------------------------------------------- #
# public share links: /s/<id>.mp4 and /s/<id>
# --------------------------------------------------------------------------- #


def _clip_for_share_id(share_id: str) -> dict | None:
    """The clip a public share id belongs to, or None.

    A short public id in a URL is an enumeration surface: the id has to be
    matched EXACTLY against what was minted, never used to build a path. The
    regex in app.sharelink already rejects anything that is not a 10-character
    id from the fixed alphabet, so this only ever compares against a literal.
    """
    if not sharelink.valid_id(share_id):
        return None
    for clip in store.list_all_clips(limit=1000):
        if clip.get("shareId") == share_id:
            return clip
    return None


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
def clip_download(clip_id: str) -> FileResponse:
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    p = _clip_media(clip, "video")
    if p is None:
        raise HTTPException(404, "clip file missing on disk")
    safe = "".join(ch for ch in clip.get("title", "clip") if ch.isalnum() or ch in " -_")[:40] or "clip"
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

    members: list[tuple[Path, str]] = []
    total = 0
    skipped: list[str] = []
    for cid in body.clipIds:
        clip = store.get_clip(cid)
        if clip is None or not clip.get("videoPath"):
            skipped.append(cid)
            continue
        p = Path(clip["videoPath"])
        if not _restore_media(p):
            skipped.append(cid)
            continue
        size = p.stat().st_size
        if total + size > MAX_EXPORT_BYTES:
            raise HTTPException(
                413,
                f"Export is {total // (1024 * 1024)} MB of MP4s and the next clip "
                f"would pass the {MAX_EXPORT_BYTES // (1024 * 1024)} MB limit — "
                "export fewer clips at a time, or download them individually.",
            )
        total += size
        title = str(clip.get("title") or "clip")[:40]
        members.append((p, f"clips/{int(clip.get('order', 0)) + 1:02d}-{title}.mp4"))

    if not members:
        raise HTTPException(404, "no rendered clips to export")

    first_clip = store.get_clip(body.clipIds[0]) if body.clipIds else None
    job_id = first_clip["jobId"] if first_clip else None
    job = store.get_job(job_id) if job_id else None

    # ZIP_STORED: H.264 is already compressed, so deflating would burn CPU for
    # ~0% gain. The point of writing to disk is that peak memory stays flat no
    # matter how large the batch is.
    fd, tmp_name = tempfile.mkstemp(prefix="clipforge-export-", suffix=".zip")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf:
            for src, arcname in members:
                zf.write(src, arcname)
            zf.writestr("captions.txt", (job or {}).get("captionsTxt") or "")
            zf.writestr(
                "README.txt",
                "ClipForge export — 9:16 1080x1920 H.264/AAC\n"
                "Only clip content you own or have permission to use.\n",
            )
        return FileResponse(
            tmp_path,
            media_type="application/zip",
            filename="clipforge-export.zip",
            background=BackgroundTask(_unlink_quietly, tmp_path),
        )
    except BaseException:
        _unlink_quietly(tmp_path)
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
