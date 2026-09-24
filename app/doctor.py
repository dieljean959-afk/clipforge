"""doctor: actionable environment diagnostics.

Replaces vague errors ("API unreachable: HTTP 400") with specific, fixable
messages. Every check returns ok/warn/fail plus a `fix` hint.

    python -m app.doctor            # human-readable report, exit 1 on FAIL
    GET /api/doctor                 # same checks as JSON
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, ffmpeg_bin, ffprobe_bin, live_gemini_key


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str
    fix: str = ""


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except Exception as exc:  # pragma: no cover
        return 1, str(exc)


def check_ffmpeg(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    try:
        ff = ffmpeg_bin()
        ver = _run([ff, "-version"])
        first = (ver[1].splitlines() or ["?"])[0]
        checks.append(Check("ffmpeg", "ok", first))
    except Exception as exc:
        checks.append(Check(
            "ffmpeg", "fail", f"not usable: {exc}",
            "install ffmpeg (apt install ffmpeg / brew install ffmpeg) or set FFMPEG_PATH",
        ))
        return checks

    # build capabilities that the pipeline silently depends on
    try:
        filters = _run([ff, "-hide_banner", "-filters"], timeout=30)[1]
    except Exception:
        filters = ""
    for filt, why in (("libx264", "H.264 clip encoding"), ("subtitles", "libass caption burn")):
        if filt == "subtitles":
            ok = " subtitles " in filters
        else:
            ok = "libx264" in filters or "libx264" in _run([ff, "-hide_banner", "-encoders"])[1]
        checks.append(Check(
            f"ffmpeg filter {filt}", "ok" if ok else "fail",
            f"{why}: {'available' if ok else 'MISSING'}",
            "" if ok else "this ffmpeg build lacks it; captions/encoding will fail",
        ))
    checks.append(Check(
        "ffmpeg scdet (cut detection)", "ok" if " scdet " in filters else "warn",
        "shot-aware reframing " + ("available" if " scdet " in filters else "unavailable"),
        "" if " scdet " in filters else "optional: install a full ffmpeg build for cut-aware crops",
    ))
    try:
        checks.append(Check("ffprobe", "ok", ffprobe_bin()))
    except Exception as exc:
        checks.append(Check("ffprobe", "fail", str(exc), "install ffmpeg (includes ffprobe) or set FFPROBE_PATH"))
    return checks


def check_gpu() -> list[Check]:
    checks: list[Check] = []
    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        checks.append(Check(
            "GPU (ctranslate2)", "ok" if n > 0 else "warn",
            f"{n} CUDA device(s)" if n > 0 else "no CUDA device — Whisper runs on CPU (slower on long videos)",
            "" if n > 0 else "expected on CPU-only hosts; nothing to fix",
        ))
    except Exception as exc:
        checks.append(Check("GPU (ctranslate2)", "warn", f"not available: {exc}", ""))
    return checks


def check_whisper(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    try:
        import faster_whisper  # noqa: F401

        checks.append(Check("whisper backend", "ok", f"faster-whisper {getattr(faster_whisper, '__version__', '?')}"))
    except Exception as exc:
        checks.append(Check("whisper backend", "fail", f"faster-whisper missing: {exc}",
                            "pip install faster-whisper"))
        return checks
    device = settings.whisper.device
    compute = settings.whisper.resolved_compute
    if device == "cuda" or (device == "auto" and compute == "float16"):
        checks.append(Check("whisper compute", "warn", f"CUDA requested ({compute})",
                            "no GPU detected — set [whisper] device = \"cpu\""))
    else:
        checks.append(Check("whisper compute", "ok", f"{device}/{compute} (model={settings.whisper.model})"))
    checks.append(Check("whisper model", "ok", settings.whisper.model, ""))
    return checks


def check_llm(settings: Settings) -> list[Check]:
    live = live_gemini_key()
    live.refresh(force=True)
    status = live.status()
    if not status.get("keyLoaded"):
        return [Check("gemini key", "fail", status.get("problem") or status.get("source", "not found"),
                      "set GEMINI_API_KEY in the Keys tab (or .env)")]
    checks = [Check("gemini key", "ok", f"loaded from {status.get('source')} (checked {status.get('lastCheckedAt')})")]
    if os.getenv("CLIPFORGE_DOCTOR_LIVE", "").lower() in {"1", "true", "yes"}:
        try:
            from google import genai

            client = genai.Client(api_key=settings.gemini_api_key)
            client.models.list()
            checks.append(Check("gemini API", "ok", f"live call ok (model={settings.gemini.model})"))
        except Exception as exc:
            checks.append(Check("gemini API", "fail", f"live call failed: {exc}",
                                "check the key + network; the backend self-heals retired models"))
    else:
        checks.append(Check("gemini API", "warn", "not tested (set CLIPFORGE_DOCTOR_LIVE=1 to test)",
                            "run: CLIPFORGE_DOCTOR_LIVE=1 python -m app.doctor"))
    return checks


def check_storage(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    try:
        usage = shutil.disk_usage(settings.app.paths.data_dir)
        free_gb = usage.free / 1e9
        status = "ok" if free_gb >= 1.5 else ("warn" if free_gb >= 0.5 else "fail")
        checks.append(Check("disk", status, f"{free_gb:.1f} GB free of {usage.total / 1e9:.1f} GB",
                            "" if status == "ok" else "delete finished jobs; artifacts auto-delete after "
                            f"{settings.hosted.retention_hours}h"))
    except Exception as exc:
        checks.append(Check("disk", "warn", f"unknown: {exc}", ""))
    checks.append(Check("limits", "ok", f"max source {settings.hosted.max_source_min} min, "
                                       f"retention {settings.hosted.retention_hours}h"))
    return checks


def check_network() -> list[Check]:
    code, _ = _run(["curl", "-sI", "-m", "5", "https://www.youtube.com"], timeout=10)
    if code == 0:
        return [Check("network (yt-dlp fetch)", "ok", "youtube reachable")]
    return [Check("network (yt-dlp fetch)", "warn", "could not reach youtube.com — URL jobs will fail, uploads still work", "")]


def run_doctor(settings: Settings) -> dict[str, Any]:
    checks: list[Check] = []
    checks += check_ffmpeg(settings)
    checks += check_gpu()
    checks += check_whisper(settings)
    checks += check_llm(settings)
    checks += check_storage(settings)
    checks += check_network()
    payload = {
        "ok": all(c.status != "fail" for c in checks),
        "summary": {
            "fail": sum(1 for c in checks if c.status == "fail"),
            "warn": sum(1 for c in checks if c.status == "warn"),
            "ok": sum(1 for c in checks if c.status == "ok"),
        },
        "checks": [asdict(c) for c in checks],
    }
    return payload


def main() -> int:
    from app.config import load_config

    settings = load_config(quiet=True)
    report = run_doctor(settings)
    print("== ClipForge doctor ==")
    for c in report["checks"]:
        mark = {"ok": "PASS", "warn": "WARN", "fail": "FAIL"}[c["status"]]
        print(f"  [{mark}] {c['name']}: {c['detail']}")
        if c["fix"]:
            print(f"         fix: {c['fix']}")
    print(f"\nsummary: {report['summary']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
