"""ClipForge local API — FastAPI backend + in-process worker, one process.

Endpoints serve the React dashboard and the pipeline artifacts on ./downloads
and ./outputs. Boot sequence: config validation → ffmpeg check → Whisper model
pre-download (explicit, with progress) → worker thread starts.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from loguru import logger

from app import subtitle as subtitle_svc
from app import video as video_svc
from app.config import (
    ConfigError,
    Settings,
    boot_banner,
    ffmpeg_bin,
    ffprobe_bin,
    load_config,
)
from app.download import probe_url
from app.store import LocalStore
from app.worker import InProcessWorker

# --------------------------------------------------------------------------- #
# boot
# --------------------------------------------------------------------------- #

settings: Settings
store: LocalStore
worker: InProcessWorker
warmup: dict = {"stage": "pending", "detail": "", "startedAt": None}


def _do_warmup() -> None:
    """Pre-download / load the Whisper model once, with honest progress."""
    warmup["startedAt"] = time.time()
    warmup["stage"] = "loading_model"
    t0 = time.time()
    try:
        def on_log(line: str) -> None:
            warmup["detail"] = line
            logger.info(f"[warmup] {line}")

        subtitle_svc.predownload_model(settings, on_log=on_log)
        warmup["stage"] = "ready"
        warmup["detail"] = f"loaded in {time.time() - t0:.1f}s"
        logger.info(f"[boot] Whisper model loaded: {settings.whisper.model} in {time.time() - t0:.1f}s")
    except Exception as exc:
        warmup["stage"] = "failed"
        warmup["detail"] = str(exc)
        logger.exception("[boot] Whisper warmup failed")


def _boot() -> None:
    global settings, store, worker
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
    title="ClipForge Local",
    version="1.0.0",
    description="Local video → viral clips pipeline (MoneyPrinterTurbo heritage).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local app; the UI may be served from a different port in dev
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _start_worker() -> None:
    worker.start()
    logger.info("[boot] in-process worker online, consuming queue: clipforge-local")


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #

class CreateJobIn(BaseModel):
    sourceType: str = Field(pattern="^(url|upload)$")
    source: str = Field(min_length=1, max_length=2048)
    minClipSec: int | None = Field(default=None, ge=5, le=180)
    maxClipSec: int | None = Field(default=None, ge=10, le=180)
    maxClips: int | None = Field(default=None, ge=1, le=15)


class RequeueIn(BaseModel):
    jobId: str


# --------------------------------------------------------------------------- #
# health / status
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "version": "1.0.0",
        "worker": worker.status(),
        "warmup": warmup,
        "llm": {
            "provider": "gemini",
            "keyLoaded": settings.gemini_api_key is not None,
        },
        "whisper": {
            "model": settings.whisper.model,
            "compute": settings.whisper.resolved_compute,
        },
    }


@app.get("/api/warmup")
def warmup_status() -> dict:
    return warmup


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
    job = store.create_job(
        source_type=body.sourceType,
        source=body.source,
        title=title,
        min_clip_sec=body.minClipSec,
        max_clip_sec=body.maxClipSec,
        max_clips=body.maxClips,
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
) -> dict:
    """Manual upload: always works, no URL fetching involved."""
    from app import download as download_svc

    job = store.create_job(
        source_type="upload",
        source="pending-upload",
        min_clip_sec=minClipSec,
        max_clip_sec=maxClipSec,
        max_clips=maxClips,
    )
    job_id = job["_id"]
    dest_dir = settings.app.paths.downloads_dir / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / "upload.bin"
    with tmp.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            fh.write(chunk)
    final = download_svc.save_upload(tmp, dest_dir)
    job = store.get_job(job_id) or job
    job["source"] = final.name
    store.write_job(job)
    store.append_job_log(job_id, f"upload ingested: {final.name} ({final.stat().st_size} bytes)")
    worker.enqueue(job_id)
    return {"jobId": job_id, "file": final.name, "bytes": final.stat().st_size}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return store.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    ok = store.delete_job(job_id)
    if not ok:
        raise HTTPException(404, "job not found")
    return {"deleted": True}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    try:
        depth = worker.retry(job_id)
    except KeyError:
        raise HTTPException(404, "job not found") from None
    return {"queued": True, "queueDepth": depth}


@app.post("/api/jobs/{job_id}/redetect")
def redetect_job(job_id: str) -> dict:
    """Re-run Gemini detection on an existing job and re-render the clips."""
    from app.task import Pipeline

    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if settings.gemini_api_key is None:
        raise HTTPException(
            400,
            "AI detection offline — add GEMINI_API_KEY to .env and restart.",
        )
    src = store.find_source_file(job_id)
    if src is None:
        raise HTTPException(400, "source file no longer on disk — re-run the job from a fresh URL/upload")
    # reset and re-run the pipeline from the find stage: simplest correct path
    # is a full re-run; the source is already local so grab is instant.
    job = store.get_job(job_id) or job
    job.update({"stage": "queued", "progress": 0, "error": None, "errorDetail": None})
    store.write_job(job)
    worker.enqueue(job_id)
    return {"queued": True}


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
    clip = store.get_clip(clip_id)
    if clip is None or not clip.get("videoPath"):
        raise HTTPException(404, "rendered video not available for this clip")
    p = Path(clip["videoPath"])
    if not p.exists():
        raise HTTPException(404, f"clip file missing on disk: {p.name}")
    return FileResponse(p, media_type="video/mp4", filename=p.name)


@app.get("/api/clips/{clip_id}/thumb")
def clip_thumb(clip_id: str) -> FileResponse:
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
        job_id = store.get_clip(body.clipIds[0])["jobId"] if body.clipIds else None
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
        buf.getvalue(),
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
            "name": "ClipForge Local API",
            "docs": "/docs",
            "ui": "run `bun run build` and restart to serve the built dashboard here",
        }
