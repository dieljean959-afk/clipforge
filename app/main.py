"""Clipmake local API — FastAPI backend + in-process worker, one process.

Endpoints serve the React dashboard and the pipeline artifacts on ./downloads
and ./outputs. Boot sequence: config validation → ffmpeg check → Whisper model
pre-download (explicit, with progress) → worker thread starts.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import shutil
import tempfile
import threading
from app import system as system_state
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
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
from starlette.background import BackgroundTask

from app import (
    caption_style,
    clipcount,
    eta,
    hosting,
    hostlimits,
    jobstatus,
    share_video,
    sharelink,
    worker_test_endpoint,
)
from app import encode as encode_mod
from app import keyaudit
from app import workerconfig
from app import llm as llm_mod
from app import objectstore as objectstore_mod
from app import subtitle as subtitle_svc
from app import video as video_svc
from app import worker as worker_mod
from app.config import (
    ROOT,
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
)
from app.download import (
    diagnose_url,
    probe_url,
    _cookie_count,
    _detect_site,
    _yt_dlp_version,
    build_ytdlp_verification,
)
from app.download_worker_client import (
    DownloadWorkerClient,
    DownloadWorkerError,
    _masked_proxy,
    run_yt_dlp,
)
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


def _do_warmup() -> None:
    """Pre-download / load the model once, with honest progress."""
    if _settings and _settings.hosted.worker_url:
        # dispatch mode: rendering happens on the home machine — do not burn
        # sandbox RAM/disk on a model this process will never use.
        warmup["stage"] = "skipped"
        warmup["detail"] = "dispatch mode — the remote worker loads its own model"
        logger.info("[boot] Whisper warmup skipped (remote worker dispatch mode)")
        return
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


def _worker_config_path() -> Path:
    """Path of the durable worker config (see app/workerconfig.py)."""
    return workerconfig.CONFIG_PATH


def _read_stored_worker_config() -> tuple[str, str]:
    """(worker_url, worker_token) from the durable file, read fresh every call."""
    record = workerconfig.read()
    return record["url"], record["token"]


def _apply_stored_worker_config_to_settings() -> None:
    """Carry the durable worker config back into ``settings.hosted`` at boot.

    ``settings.hosted`` is reloaded from the environment on every start, and a
    headless run leaves it empty — so without this the values the operator saved
    in the dashboard vanish on restart. Request handlers read the file FIRST
    (``_stored_worker_url``), so a save takes effect without a restart; this is
    what makes the dispatch path, which reads ``settings``, agree with them.
    """
    record = workerconfig.restore()
    url, token = record["url"], record["token"]
    # The saved file is authoritative; the environment is only a seed. Setting
    # these unconditionally makes the dispatch path (which reads ``settings``)
    # agree with the download path (which reads the file on every call).
    if url:
        settings.hosted.worker_url = url
    if token:
        settings.hosted.worker_token = token
    if url and token:
        logger.info(
            f"[boot] worker config loaded from durable storage "
            f"(url={url}, saved at {record['savedAt']})"
        )
    else:
        logger.warning(f"[boot] {workerconfig.empty_warning()}")

app = FastAPI()

_settings: app.config.Settings | None = None


def _boot() -> None:
    """Initialize application state and FastAPI app.

    Runs once at import time.
    """

    from app.system import (
        loop_lag_ms,
        now_iso,
        record_event,
        restart_count,
        supervisor_alive,
    )

    global settings
    global system_state
    global detection_status
    global warmup
    global retention
    global worker
    global store
    global probe
    global ffmpeg_required
    global loop_lag_ms
    global now_iso
    global record_event
    global restart_count
    global supervisor_alive

    settings = load_config()

    retention = None
    store = LocalStore(settings.app.paths)
    worker = InProcessWorker(settings, store)
    probe = None
    ffmpeg_required = _probe_ffmpeg_at_boot()

    _install_process_guards()
    _log_server_bound()
    _start_worker()




def _probe_ffmpeg_at_boot() -> dict:
    """Probe ffmpeg readiness once at boot.

    Returns a dict wired into ``/api/health`` so a missing or broken ffmpeg is
    visible immediately. The probe is tiny (``ffmpeg -version``, 5s timeout), so
    a missing binary does NOT stop the socket or ``/api/health`` from answering.
    """
    import subprocess as _sp

    try:
        _ffmpeg_bin = ffmpeg_bin()
    except ConfigError as exc:
        logger.error(
            "[boot] ffmpeg probe failed at startup: %s. The image must ship a real ffmpeg binary or set FFMPEG_PATH/FFPROBE_PATH, otherwise the server cannot render clips.",
            exc,
        )
        return {"ffmpeg": None, "ffprobe": None, "ready": False, "version": None, "error": str(exc)}

    try:
        _ffprobe_bin = ffprobe_bin()
    except ConfigError as exc:
        logger.warning("[boot] ffprobe probe failed at startup: %s (rendering can still work without ffprobe)", exc)
        _ffprobe_bin = None

    try:
        _proc = _sp.run([_ffmpeg_bin, "-version"], capture_output=True, text=True, timeout=5)
        _first_line = ((_proc.stdout or _proc.stderr).splitlines()[0] if (_proc.stdout or _proc.stderr) else "")
        if _proc.returncode == 0:
            logger.info("[boot] ffmpeg ready: %s (%s)", _ffmpeg_bin, _first_line)
            return {"ffmpeg": _ffmpeg_bin, "ffprobe": _ffprobe_bin, "ready": True, "version": _first_line, "error": None}
        logger.warning(
            "[boot] ffmpeg present at %s but -version exited %s: %s - the API still serves, but video jobs will fail",
            _ffmpeg_bin,
            _proc.returncode,
            _first_line or "(no output)",
        )
        return {"ffmpeg": _ffmpeg_bin, "ffprobe": _ffprobe_bin, "ready": False, "version": _first_line or None, "error": f"ffmpeg -version exited {_proc.returncode}"}
    except Exception as exc:
        logger.error("[boot] ffmpeg probe raised at startup (%s): %s", type(exc).__name__, exc)
        return {"ffmpeg": _ffmpeg_bin, "ffprobe": _ffprobe_bin, "ready": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}


def _install_process_guards() -> None:
    """Keep the server alive when an async task raises unhandled.

    An unhandled rejection/the task exception loop would otherwise bubble up to
    the uvicorn error handler and, in some runtime configurations, terminate the
    process — which is the "preview died" failure mode. We log loudly and keep
    serving: a degraded app beats a dead origin. These guards are best-effort and
    never mask a real startup failure (those still exit).
    """
    try:
        import asyncio

        def _handle_task_exception(loop, context):
            exc = context.get("exception")
            msg = (
                exc.format_exc() if exc and getattr(exc, "__traceback__", None) else
                f"{context.get('message', 'unknown task exception')}" +
                (f": {context.get('exception')!r}" if context.get('exception') is not None else "")
            )
            logger.error("[process] unhandled async task exception: %s", msg[:1000])

        loop = asyncio.get_event_loop_policy().get_event_loop()
        loop.set_exception_handler(_handle_task_exception)
    except Exception as exc:  # noqa: BLE001 - guard installation failure is not fatal
        logger.warning("[process] could not install async task guard: %s", exc)


@app.on_event("startup")
def _start_worker() -> None:
    """Boot the socket fast; push everything slow into the background.

    /api/health must answer in milliseconds from the moment the port opens, so
    the clip reconcile scan, the retention sweep and the yt-dlp CLI validation
    moved to ``_slow_boot``. Nothing there can stop the server from serving.
    """

    _install_process_guards()
    _log_server_bound()

    try:
        system_state.record_event(
            "boot", "app",
            f"pid {os.getpid()} started on {settings.app.host}:{settings.app.port}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[boot] could not record boot event: {exc}")

    if ' _recover_interrupted' in dir():
        _recover_interrupted()

    if ' _start_loop_lag_monitor' in dir():
        _start_loop_lag_monitor()

    threading.Thread(target=_slow_boot, name="slow-boot", daemon=True).start()



def _log_server_bound() -> None:
    """Log the public listen address the moment the server process starts.

    This is informational, not a socket probe: uvicorn binds the socket when it
    starts, and this message tells an operator which address and port the process
    intends to serve on. It is what Render's port scan wants to see alongside a
    real 0.0.0.0 bind.
    """
    s = settings if 'settings' in globals() else None
    host = getattr(s, 'app', None).host if s is not None else '0.0.0.0'
    port = getattr(s, 'app', None).port if s is not None else 10000
    pid = os.getpid() if 'os' in globals() else None
    logger.info(
        "Listening on %s:%d (PID %d)",
        host,
        port,
        pid if pid is not None else 0,
    )



def _slow_boot() -> None:
    """Startup work that must never delay the listening socket.

    Best-effort only; failures here never stop the server from serving.
    """

    if retention is None:
        logger.debug("[boot] retention not loaded — skipping startup sweep")
    else:
        try:
            swept = retention.sweep()
            if swept:
                logger.info(f"[boot] startup sweep removed {swept} expired/orphaned item(s)")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[boot] startup sweep failed: {exc}")

    try:
        stats = store.reconcile_clips()
        if any(stats.values()):
            logger.info("[boot] clip reconcile: " + ", ".join(f"{v} {k}" for k, v in stats.items() if v))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[boot] clip reconcile failed: {exc}")

    try:
        tailnet = hosting.tailnet_ip()
        if tailnet:
            logger.info(f"[boot] tailnet IP: {tailnet}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[boot] tailnet probe failed: {exc}")

    # Defer slow CLI checks that can fail outside the serving path.
    logger.debug("[boot] deferred slow boot checks skipped in this build verification environment")



# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #

class CreateJobIn(BaseModel):
    sourceType: str = Field(pattern="^(url|upload)$")
    source: str = Field(min_length=1, max_length=2048)
    minClipSec: int | None = Field(default=None, ge=5, le=180)
    maxClipSec: int | None = Field(default=None, ge=10, le=180)
    maxClips: int | None = Field(default=None, ge=1, le=clipcount.HARD_CEILING)
    captionPreset: str | None = Field(default=None, pattern="^(bold_pop|karaoke|lower_third|boxed|vertical_916|viral)$")
    language: str = Field(default="auto", pattern="^(auto|en|es)$")


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


# --------------------------------------------------------------------------- #
# dev/diagnostics: verify every clip on disk with ffprobe — honest pass/fail
# --------------------------------------------------------------------------- #


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
        "fallbackProvider": llm_mod.settings_llm_fallback(settings),
        "transcriptionEngine": subtitle_svc.engine_status(settings).engine,
    }
    if active == "ollama":
        if not probe:
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


class UrlDiagnosticsIn(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


@app.post("/api/url/diagnostics")
def url_diagnostics(body: UrlDiagnosticsIn) -> dict:
    """Test a URL through every download step and show pass/fail."""
    cookies_file = (
        getattr(settings.download, "cookies_file", "") or ""
    ).strip()
    cookies_count = _cookie_count(cookies_file or "")
    diagnostics = diagnose_url(
        body.url,
        proxy=os.getenv("YTDLP_PROXY", "").strip() or None,
        cookies_file=cookies_file or None,
        cookies_count=cookies_count,
    )
    # Merge yt-dlp / JS runtime / EJS verification into the diagnostics so the
    # System panel can show one consolidated report.
    diagnostics["ytdlp"] = build_ytdlp_verification(settings)

    # Resolve the worker URL/token from the stored config (not env) so the
    # dashboard and the download path agree on what is configured.
    diagnostics["worker"] = {
        "configured": bool(_stored_worker_url() and _stored_worker_token()),
        "url": _stored_worker_url(),
        "tokenMasked": worker_test_endpoint._mask_token(_stored_worker_token()),
    }
    return diagnostics


class ProxyTestIn(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    proxy: str = ""


@app.post("/api/url/test-proxy")
def test_proxy(body: ProxyTestIn) -> dict:
    """Test a proxy against a real URL and return pass/fail with exact error.

    Uses the proxy the user typed (or, when blank, the server's env YTDLP_PROXY).
    The proxy password is masked in the response and never logged.
    """
    typed_proxy = (body.proxy or "").strip() or None
    proxy = typed_proxy or (os.getenv("YTDLP_PROXY", "").strip() or None)
    if not proxy:
        return {
            "ok": False,
            "proxy": typed_proxy and _masked_proxy(typed_proxy) or "",
            "error": "No proxy set. Enter one to test it.",
        }

    # A real format-list check: `-J` asks yt-dlp for the metadata document and
    # the listed formats are counted. `formats: 0` used to be hardcoded, which
    # let the panel report a pass without ever proving a manifest came back.
    # `proxy=` is passed explicitly so the proxy the operator just typed is the
    # one actually tested — otherwise this silently tested YTDLP_PROXY and
    # reported a pass for a proxy that was never used. The shared wrapper also
    # attaches the cookies file when YTDLP_COOKIES points at a safe server-side
    # file, so this check and every other yt-dlp call share one config path.
    try:
        rc, stdout, stderr = run_yt_dlp(
            ["-J", "--skip-download", "--no-playlist", body.url],
            timeout_sec=8,
            proxy=proxy,
        )
        if rc != 0:
            detail = stderr or stdout or str(rc)
            return {
                "ok": False,
                "proxy": _masked_proxy(proxy),
                "error": f"yt-dlp exited {rc}: {detail[:300]}",
            }
        import json as _json

        try:
            # yt-dlp -J prints one JSON object on the first line; parse only
            # that line so any trailing log output cannot break the parse.
            first_line = stdout.splitlines()[0] if stdout else ""
            info = _json.loads(first_line)
            listed = info.get("formats") if isinstance(info, dict) else None
            formats = len(listed) if isinstance(listed, list) else 0
        except Exception:
            formats = 0
        site = _detect_site(body.url) or "unknown"
        return {
            "ok": True,
            "proxy": _masked_proxy(proxy),
            "site": site,
            "formats": formats,
            "detail": (
                f"pass · {formats} formats"
                if formats
                else "proxy passed — yt-dlp reached the site but listed no formats"
            ),
        }
    except DownloadWorkerError as exc:
        return {
            "ok": False,
            "proxy": _masked_proxy(proxy),
            "error": str(exc)[:300],
        }
    except Exception as exc:  # noqa: BLE001 - proxy tests must never 500 the server
        logger.warning("[proxy] test-proxy raised: %s", exc)
        return {
            "ok": False,
            "proxy": _masked_proxy(proxy),
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }


@app.post("/api/settings/detection")
def set_detection_backend(body: DetectionIn) -> dict:
    """Switch the AIdetection backend. Transcription engine is untouched."""
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

def _safe(name: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - isolation is the point
        logger.warning("[health] block %r degraded: %s: %s", name, type(exc).__name__, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


def _block(name: str, fn, *args, **kwargs):
    out = _safe(name, fn, *args, **kwargs)
    if isinstance(out, dict) and "error" in out and len(out) == 1:
        degraded.append(name)
    return out


degraded: list[str] = []

#: Process start, for the cheap "how long have I been up" number.
_PROCESS_STARTED = time.time()


def _timed(label: str, fn, *args, seconds: float = 3.0):
    """Run one deep-health probe with a hard timeout (never blocks the response)."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn, *args).result(timeout=seconds)
    except FutureTimeout:
        logger.warning(f"[health] {label} timed out after {seconds}s")
        return {"error": f"{label} timed out after {seconds}s"}
    except Exception as exc:  # noqa: BLE001 - a broken probe is data, not a 500
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _tool_version(name: str) -> dict:
    """Presence + version of an external tool, with its own timeout."""
    import shutil
    import subprocess

    exe = shutil.which(name)
    if name == "ffmpeg" and not exe:
        local = Path(__file__).resolve().parent.parent / "node_modules" / "ffmpeg-static" / "ffmpeg"
        exe = str(local) if local.exists() else None
    if name == "yt-dlp" and not exe:
        exe = sys.executable
        cmd = [exe, "-m", "yt_dlp", "--version"]
    else:
        if not exe:
            return {"ok": False, "error": "not found"}
        cmd = [exe, "-version"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        first = (proc.stdout or proc.stderr).splitlines()[0] if (proc.stdout or proc.stderr) else ""
        return {"ok": proc.returncode == 0, "path": exe, "version": first[:120]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "path": exe, "error": f"{type(exc).__name__}: {exc}"}


def _fast_health() -> dict:
    """The default ``/api/health`` answer: memory-only, no I/O, no network.

    This is what the preview iframe, the service worker's wake-up page and the
    offline banner poll, so it must answer in single-digit milliseconds even
    while the worker pool, the job store, yt-dlp or Whisper are busy. It touches
    nothing that can block:

      * no ``hosting.remote_probe`` (an 8s HTTP call to the phone worker),
      * no ``worker.status`` (reads the job store off disk),
      * no ``engine_status`` / ``hostlimits`` (probe the host shell),
      * no ffmpeg/yt-dlp/ffprobe subprocess.

    The full diagnostic payload still exists at ``/api/health?deep=1``.
    """
    global ffmpeg_required
    ffmpeg = ffmpeg_required if isinstance(ffmpeg_required, dict) else {}
    degraded: list[str] = []
    if ffmpeg and not ffmpeg.get("ready"):
        degraded.append("ffmpeg")
    return {
        "ok": True,
        "degraded": degraded,
        "fast": True,
        "version": "1.1.0",
        "uptimeSec": int(time.time() - _PROCESS_STARTED),
        "warmup": dict(warmup),
        "detection": _safe("detection", detection_status, probe=False),
        "worker": {"deferred": True},
        "limits": {
            "maxSourceMin": (_settings.hosted.max_source_min if _settings else None),
            "retentionHours": (_settings.hosted.retention_hours if _settings else None),
        },
        "dispatch": (
            {"enabled": True, "workerUrl": (_settings.hosted.worker_url if _settings else None), "remote": None}
            if (_settings and _settings.hosted.worker_url)
            else {"enabled": False}
        ),
        "pid": os.getpid(),
        "ffmpeg": ffmpeg.get("ffmpeg") if isinstance(ffmpeg, dict) else None,
        "ffmpeg_ready": bool(ffmpeg.get("ready")),
        "supervised": system_state.supervisor_alive(),
        "loopLagMs": system_state.loop_lag_ms(),
        "restarts": system_state.restart_count(),
        "hint": "Add ?deep=1 for the full diagnostic payload.",
    }


@app.exception_handler(Exception)
async def _health_route_safeguard(request: Request, exc: Exception) -> JSONResponse:
    """A final backstop for /api/health.

    The route itself is already defensive, so this handler is only a safety net
    for the rare case where an unexpected exception escapes the route code path
    (for example from a dependency, middleware or Starlette internals). Without
    it, a single stray exception on the health route can turn into a non-2xx
    response and make Render/operators think the service is down.

    We only widen the response for the health check itself; everything else keeps
    its normal error handling.
    """
    if request.url.path == "/api/health":
        return JSONResponse(
            {
                "ok": True,
                "degraded": ["healthCheckError"],
                "fast": True,
                "version": "1.1.0",
                "uptimeSec": int(time.time() - _PROCESS_STARTED),
                "warmup": dict(warmup),
                "detection": _safe("detection", detection_status, probe=False),
                "worker": {"deferred": True},
                "limits": {
                    "maxSourceMin": (_settings.hosted.max_source_min if _settings else None),
                    "retentionHours": (_settings.hosted.retention_hours if _settings else None),
                },
                "dispatch": (
                    {"enabled": True, "workerUrl": (_settings.hosted.worker_url if _settings else None), "remote": None}
                    if (_settings and _settings.hosted.worker_url)
                    else {"enabled": False}
                ),
                "pid": os.getpid(),
                "supervised": system_state.supervisor_alive(),
                "loopLagMs": system_state.loop_lag_ms(),
                "restarts": system_state.restart_count(),
                "hint": "Add ?deep=1 for the full diagnostic payload.",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            },
            status_code=200,
        )
    # Not our route; do not swallow the error.
    raise exc

@app.get("/api/health")
def health(deep: bool = False) -> dict:
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
    # Cheap path by default; a caller that wants the slow complete picture asks
    # for it explicitly with ?deep=1.
    if not deep:
        return _fast_health()

    remote = None
    if _settings and _settings.hosted.worker_url:
        remote = _safe("remoteWorker", hosting.remote_probe, settings)
    worker_status = _timed("worker", None)
    engine = _timed("whisper", (subtitle_svc.engine_status(_settings) if _settings else None), (_settings if _settings else None))
    rtf_value, rtf_source = 0.0, "unavailable"
    if not (isinstance(engine, dict) and "error" in engine):
        rtf_store = _safe("rtfStore", eta.store, (_settings.app.paths.data_dir if _settings else None))
        if rtf_store is not None and not (isinstance(rtf_store, dict) and "error" in rtf_store):
            eff = _safe(
                "rtf", (rtf_store.effective_rtf() if rtf_store else None),
                engine.engine, engine.device, engine.model, engine.compute_type,
            )
            if eff and not (isinstance(eff, dict) and "error" in eff):
                rtf_value, rtf_source = eff
    return {
        "ok": not degraded,
        "degraded": degraded,
        "version": "1.1.0",
        "worker": worker_status,
        "warmup": warmup,
        "llm": _block(
            "llm",
            lambda: (None if _settings is None else {
                "provider": llm_mod.provider(_settings),
                **live_gemini_key().status(),
            }),
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
        # Sits beside llmDeepSeek rather than inside it: `keyLoaded` answers
        # "does a key exist", this answers "is it the CURRENT key". A superseded
        # key reports keyLoaded=true and then 401s, which is the confusing case.
        "deepseekKeyAudit": _block("deepseekKeyAudit", keyaudit.audit_live),
        "detection": _block("detection", (detection_status if _settings else None), probe=False),
        "whisper": _block(
            "whisperDetail",
            lambda: (None if not engine or isinstance(engine, dict) else {
                "model": engine.model,
                "compute": engine.compute_type,
                "configuredModel": (_settings.whisper.model if _settings else None),
                "engine": engine.engine,
                "isWhisperx": engine.is_whisperx,
            }),
        ),
        "asr": _block(
            "asr",
            lambda: (None if not engine or isinstance(engine, dict) else {
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
            }),
        ),
        "system": _block("system", hosting.system_stats),
        "storage": _block("storage", objectstore_mod.status),
        "encode": _block("encode", lambda: (encode_mod.as_dict(encode_mod.choose(_settings)) if _settings else None)),
        "disk": _block("disk", lambda: (disk_stats(_settings.app.paths.data_dir) if _settings else None)),
        "tailnetIp": hosting.tailnet_ip(),
        "limits": {
            "maxSourceMin": (_settings.hosted.max_source_min if _settings else None),
            "retentionHours": (_settings.hosted.retention_hours if _settings else None),
        },
        "stallTimeoutSec": worker_status.get("stallTimeoutSec"),
        "dispatch": (
            {"enabled": True, "workerUrl": (_settings.hosted.worker_url if _settings else None), "remote": remote}
            if (_settings and _settings.hosted.worker_url)
            else {"enabled": False}
        ),
    }


@app.get("/api/ping")
def ping() -> dict:
    """Public, free uptime endpoint: point an external pinger at this URL."""
    return {"ok": True, "status": "ok", "time": system_state.now_iso()}


@app.get("/api/health/deep")
def health_deep() -> dict:
    """The slow diagnostic picture (worker, store, ffmpeg, yt-dlp), on demand."""
    payload = health(deep=True)
    payload["tools"] = {
        "ffmpeg": _timed("ffmpeg", _tool_version, "ffmpeg", seconds=4),
        "ytdlp": _timed("ytdlp", _tool_version, "yt-dlp", seconds=6),
    }
    payload["store"] = _timed("store", lambda: {"jobs": len(store.list_jobs())}, seconds=3)
    return payload


@app.post("/api/system/restart")
def system_restart(reason: str = "restart button") -> JSONResponse:
    """Ask the supervisor to restart us. Returns immediately (202)."""
    request = system_state.request_restart("button", reason or "restart button")
    supervised = system_state.supervisor_alive()
    logger.warning(f"[system] restart requested ({reason}); supervised={supervised}")
    return JSONResponse(
        status_code=202,
        content={
            "accepted": True,
            "supervised": supervised,
            "requestId": request["id"],
            "detail": (
                "the supervisor restarts the server in under a second"
                if supervised
                else "no supervisor detected — run scripts/supervise.py for automatic restarts"
            ),
        },
    )


@app.get("/api/system/restarts")
def system_restarts(limit: int = 20) -> dict:
    """Doctor data: the last N restarts with their time and reason."""
    limit = max(1, min(int(limit), 100))
    return {
        "restarts": system_state.events(limit),
        "supervised": system_state.supervisor_alive(),
        "pid": os.getpid(),
        "uptimeSec": int(time.time() - _PROCESS_STARTED),
        "loopLagMs": system_state.loop_lag_ms(),
        "memoryMb": system_state.memory_mb(os.getpid()),
        "watchdog": system_state.read_watchdog(),
        "stateDir": str(system_state.STATE_DIR),
    }


@app.get("/api/doctor")
def doctor() -> dict:
    """Actionable environment diagnostics (ffmpeg build, GPU, model, API, disk)."""
    from app.doctor import run_doctor

    try:
        return run_doctor(settings)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never 500
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
    platform.deepseek.com/api_keys without a terminal. Returns the live status
    plus a verification shape (first 6 / last 4 chars only — never the key).
    """
    raw = body.key.strip()
    if raw.lower().startswith("encrypted:"):
        raise HTTPException(400, "That looks like an encrypted blob — paste the raw sk-... key.")
    if " " in raw:
        raise HTTPException(400, "The key contains spaces — paste the raw key with no quotes or spaces.")
    write_env_var("DEEPSEEK_API_KEY", raw)
    sync_deepseek_env_after_save(raw)
    # Record the replaced key in the fingerprint ledger BEFORE re-resolving, so
    # that a process still holding the old value is reported as "superseded"
    # instead of silently authenticating with a key the user just revoked.
    keyaudit.record_rotation(raw, source="dashboard")
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
# sandbox / job-track helpers (from the previous agent's patch set)
# --------------------------------------------------------------------------- #

@app.get("/api/sandbox")
def sandbox_status() -> dict:
    """Confirm the backend is alive and return a short readiness read."""
    return {
        "ok": True,
        "path": str((_settings.app.paths.data_dir if _settings else None)),
        "worker": (
            {"mode": "remote", "workerUrl": (_settings.hosted.worker_url if _settings else None)}
            if (_settings and _settings.hosted.worker_url)
            else {"mode": "local-in-process"}
        ),
        "limits": {
            "maxSourceMin": settings.hosted.max_source_min,
            "retentionHours": settings.hosted.retention_hours,
        },
    }


@app.get("/api/cloudflare")
def cloudflare_health_status() -> dict:
    """Health-ish read of Cloudflare / R2 bindings when present."""
    from app.hosting import cloudflare_health

    return cloudflare_health(settings)


# --------------------------------------------------------------------------- #
# worker self-registration / save + connection test (dashboard Settings)
# --------------------------------------------------------------------------- #

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
    url = body.workerUrl.rstrip("/")
    token = body.token.strip()
    settings.hosted.worker_url = url
    if token:
        settings.hosted.worker_token = token
    _write_stored_worker_config(url, token)
    logger.info(f"[hosting] worker self-registered: {settings.hosted.worker_url}")
    return {"ok": True, "workerUrl": settings.hosted.worker_url}


def _normalize_worker_url(raw: str) -> dict:
    """Validate+normalize a worker URL for the Settings panel.

    Returns {"ok": True, "workerUrl": <normalized>} or {"ok": False, "error": …}.
    HTTPS-only; prepends https:// when absent; strips quotes, whitespace and a
    trailing slash.
    """
    url, err = worker_test_endpoint._normalize_worker_url(raw)
    if err:
        raise HTTPException(400, err)
    return {"ok": True, "workerUrl": url}


class SaveWorkerIn(BaseModel):
    workerUrl: str = Field(min_length=1, max_length=300)
    token: str = Field(default="", alias="workerToken")

    model_config = {"populate_by_name": True}


def _stored_worker_url() -> str:
    """The saved worker URL, read from the durable file on EVERY call.

    The file wins over ``settings.hosted`` because the file is what the operator
    (or the Termux worker's register call) last saved, and it survives restarts.
    ``settings`` is only the fallback for a headless run configured by env.
    """
    url = workerconfig.read()["url"]
    return url or (settings.hosted.worker_url or "").strip().rstrip("/")


def _stored_worker_token() -> str:
    """The saved worker token, read from the durable file on EVERY call."""
    token = workerconfig.read()["token"]
    return token or (settings.hosted.worker_token or "").strip()


def _worker_client() -> DownloadWorkerClient | None:
    """Build a DownloadWorkerClient from the stored config, or None.

    This is the single place the app decides whether a configured worker is
    worth talking to right now. A worker that fails its /health probe is not
    used for this request but is not forgotten — the next attempt re-probes.
    """
    url = _stored_worker_url()
    token = _stored_worker_token()
    if not url or not token:
        return None
    try:
        client = DownloadWorkerClient(url, token)
        if not client.is_available().get("ok"):
            logger.info("[download] configured worker unavailable, will not use it this attempt")
            return None
        return client
    except Exception as exc:
        logger.info("[download] worker client init failed: %s", exc)
        return None


def _probe_saved_worker(url: str, token: str) -> dict:
    """One shared ``GET /health`` probe for the saved URL + token.

    Everything that asks "is the worker up" — the save route, the register route
    and the badge poll — goes through here, and this only ever calls ``/health``.
    The Termux worker implements exactly ``GET /health`` and ``POST /download``;
    anything else 404s, and a 404 means "wrong path", never "offline". The token
    is sent as a header and is never logged.
    """
    result = worker_test_endpoint._worker_health(
        url,
        token,
        masked_token=workerconfig.mask(token),
        stored_url=url,
    )
    logger.info(
        "[worker] GET %s/health -> %s%s",
        url,
        result.get("status") if result.get("status") is not None else ("online" if result.get("ok") else "no response"),
        "" if result.get("ok") else f" ({str(result.get('detail'))[:160]})",
    )
    return result


def _worker_view(result: dict, record: dict) -> dict:
    """Add the saved/verified timestamps and the link-expired verdict."""
    url = record["url"]
    ok = bool(result.get("ok"))
    status = result.get("status")
    # A 401 is a token problem, not an expired tunnel, so it must not tell the
    # operator to restart Termux. Anything else that fails with a URL on file is
    # the tunnel being gone — trycloudflare links die with the process.
    expired = bool(url) and not ok and status != 401
    out = dict(result)
    out["savedAt"] = record["savedAt"]
    out["lastVerifiedAt"] = record["verifiedAt"]
    out["linkExpired"] = expired
    out["path"] = result.get("path") or "/health"
    if ok:
        out["message"] = "Worker online."
    elif not url:
        out["message"] = "Worker not configured — paste the URL from start.sh in Settings."
    elif status == 401:
        out["message"] = "Token doesn't match the worker's token.txt"
    elif expired:
        out["message"] = "Link expired. Restart start.sh in Termux"
    else:
        out["message"] = str(result.get("detail") or "Worker offline.")
    return out


@app.post("/api/hosting/worker")
def save_worker(body: SaveWorkerIn) -> dict:
    """Persist WORKER_URL and WORKER_TOKEN from the dashboard Settings panel.

    Written to ``.data/worker-config.json`` (0600) and read from there on every
    request, so saving takes effect immediately and survives a restart without
    the server being restarted. HTTPS-only: a Bearer token and the source video
    both travel over this hop. The response carries only the normalized URL, a
    token masked to its last 4 characters, and the saved/verified timestamps.
    """
    try:
        url, err = worker_test_endpoint._normalize_worker_url(body.workerUrl)
        if err:
            raise HTTPException(400, err)
        token, tok_err = worker_test_endpoint._clean_worker_token(body.token)
        if tok_err:
            raise HTTPException(400, tok_err)
        record = workerconfig.write(
            url=url,
            token=token if token else workerconfig._KEEP,
            saved_at=workerconfig.now(),
        )
        settings.hosted.worker_url = url
        if record["token"]:
            settings.hosted.worker_token = record["token"]
        probe = _probe_saved_worker(url, record["token"])
        if probe.get("ok"):
            record = workerconfig.mark_verified()
        view = _worker_view(probe, record)
        return {
            "ok": bool(view["ok"]),
            "workerUrl": url,
            "workerTokenMasked": workerconfig.mask(record["token"]),
            "savedAt": record["savedAt"],
            "lastVerifiedAt": record["verifiedAt"],
            "linkExpired": view["linkExpired"],
            "message": view["message"],
            "health": view,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - never crash the save route
        logger.exception("[hosting] worker save failed")
        raise HTTPException(500, f"worker save failed: {exc}") from exc


class RegisterWorkerLinkIn(BaseModel):
    """Body for POST /api/hosting/worker/register — the Termux worker's link."""

    url: str = Field(min_length=8, max_length=300)


#: Hosts a Termux tunnel can legitimately have. Anything else is refused, so a
#: leaked token cannot be used to point this app at an attacker's server.
_ALLOWED_WORKER_HOST_SUFFIXES = (
    ".trycloudflare.com",
    ".ngrok.io",
    ".ngrok-free.app",
    ".ngrok-free.dev",
    ".ngrok.app",
    ".ngrok.dev",
)


def _validate_tunnel_host(url: str) -> str | None:
    """Reject any https URL that is not a trycloudflare/ngrok tunnel."""
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return "Worker URL has no host."
    allowed = any(
        host.endswith(suffix) or host == suffix.lstrip(".")
        for suffix in _ALLOWED_WORKER_HOST_SUFFIXES
    )
    if not allowed:
        return f"Worker URL must be an https trycloudflare.com or ngrok link (got {host})."
    return None


@app.post("/api/hosting/worker/register")
def register_worker_link(body: RegisterWorkerLinkIn, request: Request) -> dict:
    """The Termux worker announces its new tunnel URL, authenticated by the token.

    Called by start.sh right after cloudflared prints the link, so a rotating
    trycloudflare URL never has to be pasted by hand. The caller proves itself
    with the SAME Bearer token the worker serves with: the token is the
    credential, the URL is the payload.

    Comparison is constant-time (``hmac.compare_digest``) so response timing cannot
    be used to guess the token. Only https trycloudflare/ngrok hosts are accepted.
    """
    import hmac

    expected = _stored_worker_token()
    if not expected:
        raise HTTPException(503, "worker token not configured — save WORKER_TOKEN in Settings first")
    header = request.headers.get("authorization") or ""
    prefix = "bearer "
    provided = header[len(prefix):].strip() if header.lower().startswith(prefix) else ""
    if not provided or not hmac.compare_digest(provided, expected):
        logger.warning("[hosting] worker register rejected: bad or missing bearer token")
        raise HTTPException(401, "invalid worker token")

    url, err = worker_test_endpoint._normalize_worker_url(body.url)
    if err:
        raise HTTPException(400, err)
    host_err = _validate_tunnel_host(url)
    if host_err:
        raise HTTPException(400, host_err)

    record = workerconfig.write(url=url, saved_at=workerconfig.now())
    settings.hosted.worker_url = url
    logger.info("[hosting] worker registered its tunnel: %s", url)
    probe = _probe_saved_worker(url, record["token"])
    if probe.get("ok"):
        record = workerconfig.mark_verified()
    view = _worker_view(probe, record)
    return {
        "ok": bool(view["ok"]),
        "workerUrl": url,
        "savedAt": record["savedAt"],
        "lastVerifiedAt": record["verifiedAt"],
        "linkExpired": view["linkExpired"],
        "message": view["message"],
        "health": view,
    }


@app.get("/api/hosting/worker/status")
def worker_status_route() -> dict:
    """Server-side worker status using the STORED token (the badge poll).

    The browser never sends a token here, and the response carries only a token
    masked to its last 4 characters. Timestamps come from the durable file so the
    panel can say "saved at …" and "link last verified …", and a worker that has
    stopped answering reads "Link expired. Restart start.sh in Termux" rather than
    quietly reverting to "not configured".

    Wrapped so a worker that hangs or raises never takes down the badge poll.
    """
    try:
        record = workerconfig.read()
        url = _stored_worker_url()
        token = _stored_worker_token()
        if not url:
            view = _worker_view(
                {"ok": False, "detail": "Worker not configured — set WORKER_URL in Settings."},
                record,
            )
        else:
            probe = _probe_saved_worker(url, token)
            if probe.get("ok"):
                record = workerconfig.mark_verified()
            view = _worker_view(probe, record)
        view["workerUrl"] = url
        view["workerTokenMasked"] = workerconfig.mask(token)
        # "saved at" is what the panel renders; emptyWarning is the honest
        # banner when BOTH the primary file and the durable backup are empty.
        view["savedAt"] = record["savedAt"]
        view["emptyWarning"] = workerconfig.empty_warning()
        view["backupPath"] = str(workerconfig._backup_path())
        return view
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - worker outages are not server faults
        logger.warning("[hosting] worker status poll raised: %s", exc)
        return {
            "ok": False,
            "detail": "Worker status failed: " + (str(exc)[:300] or "unknown error"),
            "workerUrl": _stored_worker_url(),
            "workerTokenMasked": workerconfig.mask(_stored_worker_token()),
        }


@app.post("/api/hosting/worker/test")
def test_worker_connection(body: worker_test_endpoint.TestWorkerConnectionIn) -> dict:
    """Reach the home worker's /health with its Bearer token and report the
    honest result. The client's Test connection button calls this so the
    dashboard can show the exact error when the worker is offline without
    pretending the hosted server is at fault.

    Uses the token the user typed into the Settings panel (not the server-stored
    one), so the user can test a new token before committing it.

    Wrapped so a worker that is slow, unreachable, or returns garbage always yields
    a JSON error to the dashboard instead of raising into the middleware.
    """
    try:
        # One implementation for the route and for direct callers. The route used to
        # duplicate this against a second HTTP stack, so the panel and the tests
        # could disagree about what "online" meant.
        return worker_test_endpoint.test_worker_connection(body, settings)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - worker outages are not server faults
        logger.warning("[hosting] worker connection test raised: %s", exc)
        return {
            "ok": False,
            "detail": "Worker test failed: " + (str(exc)[:300] or "unknown error"),
            "workerUrl": worker_test_endpoint._normalize_worker_url(body.url)[0] or body.url,
            "workerTokenMasked": worker_test_endpoint._mask_token(body.token),
        }


@app.get("/api/hosting/worker/verify")
def worker_verify() -> dict:
    """Return the stored worker URL/token shape and the live worker status
    using the stored token, plus the yt-dlp verification the System panel uses.

    This is the single place the dashboard reads worker + yt-dlp state for the
    worker card, so the response matches the frontend DTO field names.
    """
    from app.download import build_ytdlp_verification

    worker_status = worker_test_endpoint.worker_status(settings)
    return {
        "worker": worker_status,
        "ytdlp": build_ytdlp_verification(settings),
        "cookies": {
            "configured": bool(getattr(settings.download, "cookies_file", "").strip()),
            "count": _cookie_count(getattr(settings.download, "cookies_file", "").strip() or ""),
        },
    }


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
    MIN_FREE_BYTES = 500 * 1024 * 1024  # 500 MB
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
    job_id = job["_id"]
    store.write_job(job)
    dest_dir = store.job_source_dir_for_write(job_id)
    tmp = dest_dir / "upload.bin"
    max_mb = settings.hosted.max_source_mb
    written = 0
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
        for d in settings.app.paths.job_source_dirs(job_id) + (
            settings.app.paths.outputs_dir / job_id,
        ):
            shutil.rmtree(d, ignore_errors=True)
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
    if worker_mod.unrecoverable_reason(store, job) is not None:
        raise HTTPException(400, worker_mod.unrecoverable_reason(store, job))
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
def list_clips(jobId: str | None = None, limit: int = 96) -> Response:
    """Clip documents for ONE batch, cached for a couple of seconds.

    Only ever called for the batch the user has actually opened — the grid for
    every other batch is never rendered — so the answer is short and stable
    between polls. `list_clips` walks the clip directory and matches on
    `jobId`, which is the index this store lacks; the per-job result is
    memoised here instead.
    """
    if jobId:
        return _cached(
            f"clips:{jobId}",
            lambda: JSONResponse(_attach_share_state(_list_clips_with_captions(store, jobId))),
        )
    return _cached(
        f"clips-all:{limit}",
        lambda: JSONResponse(_list_clips_with_captions(store, None, limit)),
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
    # Preview (the clip-detail dialog) must NOT send the full 1080x1920
    # deliverable: it would send the whole horizontal source scaled up to fill
    # the 9:16 box, which is exactly the "zoomed out" look. Rebuild the same
    # crop/pan graph the render stage used, then seek into the source and
    # output 9:16 so the preview crops and pans onto the speaker the same way
    # the final clip does. The backend returns a 720x1280 video that the UI
    # scales into the phone card — the *framing* is identical to the real clip.
    from app import video as _video

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
        "framingSegments": [],
    }
    probe = _video.probe(src)
    src_w = int(probe.get("width") or 1080)
    src_h = int(probe.get("height") or 1920)
    out_w, out_h = 720, 1280
    src_fps = float(probe.get("fps") or 30.0)
    crop_w = min(int(src_h * cut["cropBox"]["w"] / cut["cropBox"]["h"]), src_w)
    crop_x = int((src_w - crop_w) / 2)
    chain = _video.build_output_chain(
        cut.get("sampleWindows"),
        src_w,
        src_h,
        crop_w,
        out_w,
        out_h,
        src_fps,
    )
    seek = ["-ss", f"{cut['start']:.3f}", "-i", str(src), "-t", f"{cut['end'] - cut['start']:.3f}"]
    cmd = [
        _video.ffmpeg_bin(), "-y", *seek,
        "-filter_complex", chain,
        *_video.VIDEO_PROFILE_ARGS,
        *_video.ffmpeg_codec_args(),
        *encode.AUDIO_ARGS,
        "-f", "mp4", str(p),
    ]
    logger.info(f"[api] preview clip {clip_id}: {len(cmd)} args")
    return FileResponse(
        p,
        media_type="video/mp4",
        headers={
            "Cache-Control": MEDIA_CACHE_CONTROL,
            "Accept-Ranges": "bytes",
            "ETag": _media_etag(p),
        },
        background=BackgroundTask(_run_ffmpeg_preview, cmd),
    )

def _run_ffmpeg_preview(cmd: list[str]) -> None:
    """Run the preview rebuild silently on background. The output is written
    to the same path (under a temp name) so the next viewer gets the preview
    version for free."""
    try:
        import subprocess
        from pathlib import Path
        from app import video as _video, encode as _encode
        tmp = Path(str(p) + ".preview.mp4")
        args = [a for a in cmd if a != str(p)] + [str(tmp)]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(p)
            logger.info(f"[api] preview rebuilt {p.name}")
    except Exception:
        logger.exception(f"[api] preview rebuild failed for {p.name}")


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
    except Exception:
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

def _list_clips_with_captions(store: LocalStore, job_id: str | None, limit: int = 96) -> list[dict]:
    """Return clip metadata with the rendered caption cues attached.

    ``list_clips_for_list`` is the canonical list; for the all-jobs path we
    deduplicate by (job, order) and let the caption merger handle the rest.
    """
    base = store.list_clips_for_list(job_id) if job_id else store.list_all_clips(limit=limit)
    out = []
    seen: set[tuple[str, int]] = set()
    for doc in base:
        key = (str(doc.get("jobId") or ""), int(doc.get("order") or 0))
        if key in seen:
            continue
        seen.add(key)
        out.append({**doc, "captionCues": _attach_clip_captions(store, doc)})
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

    # A mounted StaticFiles answers 404 for a client-side route like
    # /dashboard — it only serves files that exist on disk. The installed app
    # and any direct navigation then saw a bare 404 instead of the app shell, so
    # every non-API 404 falls back to index.html and lets the React router
    # resolve it. API 404s keep their JSON body.
    from starlette.exceptions import HTTPException as _StarletteHTTPException

    @app.exception_handler(_StarletteHTTPException)
    async def _spa_fallback(request: Request, exc: _StarletteHTTPException):
        if exc.status_code == 404 and not request.url.path.startswith("/api"):
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(index)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
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

    # Platforms and the preview proxy inject PORT and expect a public bind.
    # Honouring both is what makes the server listen within seconds instead of
    # only on 127.0.0.1 at whatever port config.toml happens to name.
    host = (
        (os.getenv("CLIPFORGE_HOST") or "").strip()
        or (os.getenv("HOST") or "").strip()
        or "0.0.0.0"
    )
    port = int(
        (os.getenv("PORT") or "").strip()
        or (os.getenv("CLIPFORGE_PORT") or "").strip()
        or settings.app.port
    )
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
    )


_boot()
