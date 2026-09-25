"""ClipForge local API — FastAPI backend + in-process worker, one process.

Endpoints serve the React dashboard and the pipeline artifacts on ./downloads
and ./outputs. Boot sequence: config validation → ffmpeg check → Whisper model
pre-download (explicit, with progress) → worker thread starts.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from app import eta, hosting, hostlimits
from app import subtitle as subtitle_svc
from app import video as video_svc
from app.config import (
    ConfigError,
    Settings,
    boot_banner,
    delete_env_var,
    ffmpeg_bin,
    ffprobe_bin,
    live_gemini_key,
    load_config,
    sync_gemini_env_after_save,
    sync_gemini_env_for_delete,
    write_env_var,
)
from app.download import probe_url
from app.hosting import RetentionCleaner, disk_stats
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
    global settings, store, worker, retention
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

    # Re-queue any jobs left mid-flight by a previous run (crash recovery).
    for job in store.list_jobs():
        if job.get("stage") in {"grabbing", "transcribing", "finding", "rendering", "queued"}:
            logger.info(f"[boot] re-queuing interrupted job {job['_id']} (was {job['stage']})")
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
    maxClips: int | None = Field(default=None, ge=1, le=15)
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
# health / status
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health() -> dict:
    remote = hosting.remote_probe(settings) if settings.hosted.worker_url else None
    worker_status = worker.status()
    engine = subtitle_svc.engine_status(settings)
    rtf_store = eta.store(settings.app.paths.data_dir)
    rtf_value, rtf_source = rtf_store.effective_rtf(
        engine.engine, engine.device, engine.model, engine.compute_type
    )
    return {
        "ok": True,
        "version": "1.1.0",
        "worker": worker_status,
        "warmup": warmup,
        "llm": {
            "provider": "gemini",
            # LIVE re-resolution — a key saved in the platform after boot is
            # picked up on the next health poll without restarting anything.
            **live_gemini_key().status(),
        },
        "whisper": {
            # The RESOLVED engine/model/compute, not what config.toml asks for.
            # The host-limit downgrade (medium -> small) has to be visible in the
            # status label, otherwise the UI advertises a model that never loads.
            "model": engine.model,
            "compute": engine.compute_type,
            "configuredModel": settings.whisper.model,
            "engine": engine.engine,
            "isWhisperx": engine.is_whisperx,
        },
        "asr": {
            # Which engine is really transcribing, which model actually loads,
            # and the RTF this host has measured — the same numbers the ETA is
            # derived from.
            **engine.as_dict(),
            "estimatedRtf": round(rtf_value, 4),
            "rtfSource": rtf_source,
            "rtfHistory": rtf_store.snapshot(),
            "host": hostlimits.limits(),
            "hostDetail": hostlimits.describe(),
        },
        "system": hosting.system_stats(),
        "disk": disk_stats(),
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

    return run_doctor(settings)


@app.get("/api/caption-presets")
def caption_presets() -> list[dict]:
    from app.subtitle import CAPTION_PRESETS

    return [
        {"key": p.key, "label": p.label}
        for p in CAPTION_PRESETS.values()
    ]


@app.get("/api/warmup")
def warmup_status() -> dict:
    return warmup


@app.post("/api/settings/reload-secrets")
def reload_secrets() -> dict:
    """Force re-read of env files + live key resolution (the Re-check button)."""
    live = live_gemini_key()
    live.refresh(force=True)
    status = live.status()
    logger.info(f"[config] secrets re-checked — key loaded: {status['keyLoaded']} ({status['source']})")
    return status


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


@app.delete("/api/settings/gemini-key")
def remove_gemini_key() -> dict:
    """Delete GEMINI_API_KEY from .env.local (the missing Delete button)."""
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
        raise HTTPException(400, f"Could not read the uploaded video: {exc}") from exc
    max_seconds = settings.hosted.max_source_min * 60
    if uploaded_meta["duration"] > max_seconds:
        final.unlink(missing_ok=True)
        store.delete_job(job_id)
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
    jobs = store.list_jobs()
    # live queue positions — a job waiting at 0% must be distinguishable from
    # one that is genuinely stuck
    positions = worker.queue_snapshot()
    for job in jobs:
        job["queuePosition"] = positions.get(job["_id"])
    return jobs


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job["queuePosition"] = worker.queue_snapshot().get(job_id)
    return job


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    ok = store.delete_job(job_id)
    if not ok:
        raise HTTPException(404, "job not found")
    # stage mirrors (per-chunk transcript caches, the extracted audio WAV) are
    # not part of the job store and would otherwise accumulate forever
    remove_workspace(settings.app.paths.data_dir, job_id)
    return {"deleted": True}


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


@app.post("/api/jobs/{job_id}/redetect")
def redetect_job(job_id: str) -> dict:
    """Re-run Gemini detection on an existing job and re-render the clips."""

    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if settings.gemini_api_key is None:
        live = live_gemini_key()
        live.refresh(force=True)
        raise HTTPException(
            400,
            live.problem
            or "AI detection offline — add GEMINI_API_KEY in the Keys/API keys tab (or .env), then hit Re-check on the dashboard.",
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


@app.get("/api/clips/{clip_id}/video")
def clip_video(clip_id: str) -> FileResponse:
    """Stream the rendered clip for <video> playback.

    Content-Disposition stays inline — 'attachment' makes browsers refuse to
    play the stream. Downloads go through /download instead.
    """
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    p = Path(clip["videoPath"])
    if not p.exists():
        raise HTTPException(404, f"clip file missing on disk: {p.name}")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/clips/{clip_id}/thumb")
def clip_thumb(clip_id: str) -> FileResponse:
    """Real first frame of the rendered clip — inline image, no generated art."""
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("thumbPath"):
        raise HTTPException(404, "thumbnail not available")
    p = Path(clip["thumbPath"])
    if not p.exists():
        raise HTTPException(404, "thumbnail file missing")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/clips/{clip_id}/download")
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
        headers={"Content-Disposition": f'attachment; filename="{clip["order"] + 1:02d}-{safe}.mp4"'},
    )


@app.get("/api/clips/{clip_id}/srt")
def clip_srt(clip_id: str) -> Response:
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    lines = []
    for i, cue in enumerate(clip.get("captionCues", []), 1):
        lines.append(f"{i}\n{_srt_time(cue['start'])} --> {_srt_time(cue['end'])}\n{cue['text']}\n")
    return Response("\n".join(lines), media_type="text/plain")


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


@app.post("/api/export/zip")
def export_zip(body: ZipRequest) -> Response:
    """ZIP of REAL rendered MP4s + captions.txt — no metadata-only exports."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        included = 0
        for cid in body.clipIds:
            clip = store.get_clip(cid)
            if clip is None or not clip.get("videoPath"):
                continue
            p = Path(clip["videoPath"])
            if not p.exists():
                continue
            zf.write(p, f"clips/{clip['order'] + 1:02d}-{clip.get('title', 'clip')[:40] or 'clip'}.mp4")
            included += 1
        first_clip = store.get_clip(body.clipIds[0]) if body.clipIds else None
        job_id = first_clip["jobId"] if first_clip else None
        job = store.get_job(job_id) if job_id else None
        zf.writestr("captions.txt", (job or {}).get("captionsTxt") or "")
        zf.writestr(
            "README.txt",
            "ClipForge export — 9:16 1080x1920 H.264/AAC\n"
            "Only clip content you own or have permission to use.\n",
        )
    if included == 0:
        raise HTTPException(404, "no rendered clips to export")
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="clipforge-export.zip"'},
    )


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
