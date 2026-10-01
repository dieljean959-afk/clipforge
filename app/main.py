"""ClipForge local API — FastAPI backend + in-process worker, one process.

Endpoints serve the React dashboard and the pipeline artifacts on ./downloads
and ./outputs. Boot sequence: config validation → ffmpeg check → Whisper model
pre-download (explicit, with progress) → worker thread starts.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from app import caption_style, clipcount, eta, hosting, hostlimits, jobstatus
from app import encode as encode_mod
from app import llm as llm_mod
from app import subtitle as subtitle_svc
from app import video as video_svc
from app import worker as worker_mod
from app.config import (
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
    write_env_var,
)
from app.download import probe_url
from app.hosting import RetentionCleaner, disk_stats
from app.social import (
    PLATFORMS,
    AccountStore,
    NeedsReconnect,
    NotConfigured,
    PostRejected,
    SocialError,
    describe_all,
    redirect_uri,
)
from app.social import base as social_base
from app.social import post_video as social_post
from app.stages import StageWorkspace, remove_workspace
from app.store import LocalStore
from app.worker import InProcessWorker

# --------------------------------------------------------------------------- #
# boot
# --------------------------------------------------------------------------- #

settings: Settings
store: LocalStore
worker: InProcessWorker
retention: RetentionCleaner | None = None
warmup: dict = {"stage": "pending", "detail": "", "startedAt": None}

# Disk preflight floor for new jobs: source download + caption/work files need
# headroom (a 2h 1080p source is multi-GB). Below this, submits are refused
# with a readable 507 instead of failing mid-pipeline.
MIN_FREE_BYTES = 1.5 * 1024**3


def _do_warmup() -> None:
    """Pre-download / load the model once, with honest progress."""
    if settings.hosted.worker_url:
        # dispatch mode: rendering happens on the home machine — do not burn
        # sandbox RAM/disk on a model this process will never use.
        warmup["stage"] = "skipped"
        warmup["detail"] = "dispatch mode — the remote worker loads its own model"
        logger.info("[boot] Whisper warmup skipped (remote worker dispatch mode)")
        return
    # Tight-memory hosts: holding the model in the API process would leave too
    # little for the job that has to load its own copy, so the API process would
    # be the thing that OOMs. Jobs still load the model themselves on first use.
    headroom_need = hostlimits.per_job_bytes(settings.whisper.model) * 2
    limit = hostlimits.effective_memory_bytes()
    if limit is not None and headroom_need > limit:
        warmup["stage"] = "skipped"
        warmup["detail"] = (
            f"skipped to save memory — {hostlimits.describe()}; each job loads its own model "
            "(the first job therefore starts a little slower)"
        )
        logger.info(f"[boot] Whisper warmup skipped: {warmup['detail']}")
        return
    warmup["startedAt"] = time.time()
    warmup["stage"] = "loading_model"
    t0 = time.time()
    try:
        def on_log(line: str) -> None:
            warmup["detail"] = line
            logger.info(f"[warmup] {line}")

        transcript = subtitle_svc.predownload_model(settings, on_log=on_log)
        warmup["stage"] = "ready"
        warmup["detail"] = f"{transcript.engine} / {transcript.model_size} loaded in {time.time() - t0:.1f}s"
        logger.info(f"[boot] ASR model loaded: {warmup['detail']}")
    except Exception as exc:
        warmup["stage"] = "failed"
        warmup["detail"] = str(exc)
        logger.exception("[boot] Whisper warmup failed")


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
    worker = InProcessWorker(settings, store)
    retention = RetentionCleaner(settings, store)
    accounts = AccountStore(settings.app.paths.data_dir)

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
    recoverable = {"grabbing", "transcribing", "finding", "rendering", "queued"}
    for job in store.list_jobs():
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
        reason = None
        if job.get("stage") in recoverable:
            reason = "mid-flight"
        elif job.get("interrupted"):
            reason = "flagged interrupted"
        elif InProcessWorker.has_signal_failure(job):
            reason = "recorded as failed, but the cause was an external kill"
        if reason is None:
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


_boot()

app = FastAPI(
    title="ClipForge",
    version="1.1.0",
    description="Hosted video → viral clips pipeline (MoneyPrinterTurbo heritage).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # same-origin UI; permissive for API tooling
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _log_errors(request: Request, call_next):
    """Log every non-2xx API response with its exception — the dashboard only
    shows "HTTP 400"-style codes, so the server must keep the real evidence."""
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"[api] {request.method} {request.url.path} raised")
        raise
    if response.status_code >= 400 and request.url.path.startswith("/api"):
        logger.warning(f"[api] {request.method} {request.url.path} -> {response.status_code}")
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
    worker.start()
    retention.start()
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
    captionPreset: str | None = Field(default=None, pattern="^(bold_pop|karaoke|lower_third|boxed)$")


class RequeueIn(BaseModel):
    jobId: str


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
DETECTION_TRADEOFFS = {
    "deepseek": "Hosted, OpenAI-compatible chat completions, and the DEFAULT. The DeepSeek "
                "platform bills a balance rather than a fixed tokens-per-DAY ceiling like "
                "Groq's free tier, so a long source cannot run out of quota half way through "
                "detection — and its rates are the cheapest of the hosted options. Needs a "
                "DeepSeek API key.",
    "nvidia": "Hosted alternative, selectable without a redeploy. NVIDIA NIM is the same "
              "wire format and the same code path, and hosts its own `deepseek-*` models — "
              "but on an NVIDIA quota, not a DeepSeek balance. Needs an NVIDIA API key.",
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
        "warmup": warmup,
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
        "encode": _block("encode", lambda: encode_mod.as_dict(encode_mod.choose(settings))),
        "disk": _block("disk", disk_stats),
        "tailnetIp": hosting.tailnet_ip(),
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

class CaptionStyleIn(BaseModel):
    # All optional: the dialog sends the whole set it knows about, and a
    # missing key means "stop overriding this one".
    words_per_cue: float | None = None
    words_per_cue_max: float | None = None
    pause_sec: float | None = None
    highlight_color: str | None = None


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
    stored = caption_style.write(
        settings.app.paths, body.model_dump(exclude_none=True)
    )
    return {
        **caption_style.effective(settings),
        "saved": stored,
        "styleId": subtitle_svc.style_fingerprint(
            caption_style.resolve(settings), settings.subtitle.preset
        ),
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
    if body.sourceType == "url":
        info = probe_url(body.source)
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
    free = disk_stats().get("freeBytes")
    if free is not None and free < MIN_FREE_BYTES:
        raise HTTPException(
            507,
            f"Server disk is almost full ({free // (1024 * 1024)} MB free) — "
            "files are deleted automatically 24h after a job completes, or delete "
            "old jobs on the dashboard and try again.",
        )
    job = store.create_job(
        source_type=body.sourceType,
        source=body.source,
        title=title,
        min_clip_sec=body.minClipSec,
        max_clip_sec=body.maxClipSec,
        max_clips=body.maxClips,
        caption_preset=body.captionPreset,
    )
    job_id = job["_id"]
    store.append_job_log(job_id, f"job created ({body.sourceType}: {body.source[:80]})")
    worker.enqueue(job_id)
    return {"jobId": job_id, "title": title, "duration": duration}


@app.post("/api/jobs/upload")
async def upload_job(
    file: UploadFile,
    minClipSec: int | None = None,
    maxClipSec: int | None = None,
    maxClips: int | None = None,
    captionPreset: str | None = None,
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
    )
    job_id = job["_id"]
    dest_dir = settings.app.paths.downloads_dir / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / "upload.bin"
    with tmp.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            fh.write(chunk)
    final = download_svc.save_upload(tmp, dest_dir)
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
    store.write_job(job)
    store.append_job_log(job_id, f"upload ingested: {final.name} ({final.stat().st_size} bytes)")
    worker.enqueue(job_id)
    return {"jobId": job_id, "file": final.name, "bytes": final.stat().st_size}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    # A job whose worker died mid-stage leaves `stage` on a live value with
    # nothing behind it. Reconciled HERE — at the single point every read goes
    # through — so no view can disagree about whether it is running.
    worker.reconcile_stale_running()
    jobs = store.list_jobs()
    # live queue positions — a job waiting at 0% must be distinguishable from
    # one that is genuinely stuck
    positions = worker.queue_snapshot()
    for job in jobs:
        job["queuePosition"] = positions.get(job["_id"])
    # ONE derivation for every consumer, so the list card and the job card
    # cannot read the same document and reach opposite conclusions.
    return jobstatus.describe_all(jobs)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    # "Check now" hits this route, so it must reconcile too — otherwise the
    # button that exists to tell the truth is the one place that cannot.
    worker.reconcile_stale_running()
    job = store.get_job(job_id) or job
    job["queuePosition"] = worker.queue_snapshot().get(job_id)
    return jobstatus.describe(job)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
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
    for d in (
        settings.app.paths.downloads_dir / job_id,
        settings.app.paths.outputs_dir / job_id,
    ):
        shutil.rmtree(d, ignore_errors=True)
    return {"deleted": True, "cancelled": cancelled}


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
    try:
        depth = worker.retry(job_id)
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
    # "fetch" stays done, so the video is not downloaded a second time.
    ws = StageWorkspace(settings.app.paths.data_dir, job_id)
    ws.clear_from("transcribe")
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
def list_clips(jobId: str | None = None, limit: int = 96) -> list[dict]:
    if jobId:
        return store.list_clips(jobId)
    return store.list_all_clips(limit=limit)


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
    """
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    p = Path(clip["videoPath"])
    if not p.exists():
        raise HTTPException(404, f"clip file missing on disk: {p.name}")
    return FileResponse(
        p,
        media_type="video/mp4",
        headers={
            "Cache-Control": MEDIA_CACHE_CONTROL,
            "Accept-Ranges": "bytes",
        },
    )


@app.api_route("/api/clips/{clip_id}/thumb", methods=["GET", "HEAD"])
def clip_thumb(clip_id: str) -> FileResponse:
    """Real first frame of the rendered clip — inline image, no generated art."""
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("thumbPath"):
        raise HTTPException(404, "thumbnail not available")
    p = Path(clip["thumbPath"])
    if not p.exists():
        raise HTTPException(404, "thumbnail file missing")
    return FileResponse(
        p,
        media_type="image/jpeg",
        headers={"Cache-Control": MEDIA_CACHE_CONTROL},
    )


@app.api_route("/api/clips/{clip_id}/download", methods=["GET", "HEAD"])
def clip_download(clip_id: str) -> FileResponse:
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    p = Path(clip["videoPath"])
    if not p.exists():
        raise HTTPException(404, f"clip file missing on disk: {p.name}")
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
            if not p.exists():
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
        if not p.exists():
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


def main() -> None:
    """Entry point for the `clipforge` console script (pyproject.toml)."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        log_level="info",
    )
