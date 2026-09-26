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
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from app import hostlimits
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
    """Report the ASR engine that will actually transcribe, and why.

    WhisperX is the preferred engine; when its torch/pyannote stack is absent
    the batched faster-whisper fallback still runs (same CTranslate2 engine and
    the same batching speed-up), so this is a WARN with the exact install
    command, never a FAIL.
    """
    from app import asr as asr_mod

    checks: list[Check] = []
    try:
        import faster_whisper  # noqa: F401

        checks.append(Check(
            "faster-whisper", "ok", f"{getattr(faster_whisper, '__version__', '?')}"
        ))
    except Exception as exc:
        checks.append(Check("faster-whisper", "fail", f"missing: {exc}",
                            "pip install faster-whisper"))
        return checks

    status = asr_mod.status(settings)
    ok, reason = asr_mod.whisperx_available()
    if ok:
        checks.append(Check("asr engine", "ok", f"whisperx — {reason}"))
        checks.append(Check(
            "word alignment", "ok",
            "wav2vec2 forced alignment available (accurate word-level timestamps)",
        ))
    else:
        checks.append(Check(
            "asr engine", "warn",
            f"faster-whisper batched pipeline — {reason}",
            "optional upgrade: pip install 'clipforge[whisperx]' adds wav2vec2 "
            "forced-alignment word timestamps; batched decoding is already active",
        ))
        checks.append(Check(
            "word alignment", "warn",
            "engine-native word timestamps (no forced-alignment pass)",
            "install the whisperx extra for phoneme-level word timing accuracy",
        ))

    checks.append(Check(
        "asr device",
        "ok" if status.device == "cpu" else "warn",
        f"{status.device}/{status.compute_type}, batch {status.batch_size}, model={status.model}",
        "" if status.device == "cpu" else "GPU path selected — verify torch/CUDA is present",
    ))
    # Host fit (container CPU/memory caps vs. the configured model) is checked
    # separately by check_host().
    try:
        import ctranslate2

        checks.append(Check("ctranslate2", "ok", ctranslate2.__version__))
    except Exception as exc:
        checks.append(Check("ctranslate2", "warn", f"version unknown: {exc}", ""))
    return checks


def check_llm(settings: Settings) -> list[Check]:
    from app import llm as llm_mod  # noqa: PLC0415 - avoids an import cycle at boot

    if llm_mod.provider(settings) == llm_mod.PROVIDER_OLLAMA:
        # Local backend: no key exists or is needed; the server IS the check.
        ok, reason = llm_mod.ollama_status(settings)
        return [Check(
            "detection backend", "ok" if ok else "fail", reason,
            "" if ok else "run `ollama serve` + `ollama pull <model>`, or switch "
                          "[llm] provider back to groq",
        )]
    if llm_mod.provider(settings) == llm_mod.PROVIDER_GROQ:
        return _check_groq(settings)
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


def _check_groq(settings: Settings) -> list[Check]:
    """Groq key presence + optional live probe of the models endpoint.

    The live probe is deliberately a GET /models rather than a chat call: it
    costs no quota and still proves both the key AND network path work.
    """
    from app.config import live_groq_key  # noqa: PLC0415
    from app.llm import GROQ_USER_AGENT  # noqa: PLC0415 - same edge-block guard

    live = live_groq_key()
    live.refresh(force=True)
    status = live.status()
    if not status.get("keyLoaded"):
        return [Check("groq key", "fail", status.get("problem") or status.get("source", "not found"),
                      "set GROQ_API_KEY in the Keys tab (or .env)")]
    checks = [Check("groq key", "ok", f"loaded from {status.get('source')}")]
    if os.getenv("CLIPFORGE_DOCTOR_LIVE", "").lower() in {"1", "true", "yes"}:
        base = str(settings.groq.base_url).rstrip("/")
        req = urllib.request.Request(  # noqa: S310 - https base from config
            f"{base}/models",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key or ''}",
                # Same Cloudflare 1010 signature ban as the chat call — without
                # this the probe reports a bogus "bad key" for a good key.
                "User-Agent": GROQ_USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
            want = settings.groq.model
            note = ""
            if ids and want not in ids:
                note = f" (CONFIGURED MODEL NOT IN CATALOG — pick one of: {', '.join(ids[:6])})"
            checks.append(Check("groq API", "ok" if not note else "warn",
                                f"live call ok, {len(ids)} model(s) visible{note}"))
        except urllib.error.HTTPError as exc:
            checks.append(Check("groq API", "fail", f"live call failed: HTTP {exc.code}",
                                "check the key + network"))
        except Exception as exc:  # noqa: BLE001 - any transport error is a fail
            checks.append(Check("groq API", "fail", f"live call failed: {exc}",
                                "check the key + network"))
    else:
        checks.append(Check("groq API", "warn", "not tested (set CLIPFORGE_DOCTOR_LIVE=1 to test)",
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


def check_host(settings: Settings) -> list[Check]:
    """What this host can actually do — with the honest numbers.

    Container CPU/memory caps decide both the inference thread count and how
    many jobs may run at once, so they belong in diagnostics rather than being
    silently mis-detected from os.cpu_count().
    """
    info = hostlimits.limits()
    model = settings.whisper.model
    resolved, note = hostlimits.resolved_model(model)
    per_job = hostlimits.per_job_bytes(resolved) / 1024**3
    fit_status = "fail" if note else "ok"
    return [
        Check(
            "host resources",
            "warn" if info["constrained"] else "ok",
            f"{info['cpus']} CPU (host reports {info['hostCpus']}), "
            f"{(info['memoryEffectiveBytes'] or 0) / 1024**3:.1f} GiB RAM usable, "
            f"inference threads {info['inferenceThreads']}",
            ""
            if not info["constrained"]
            else "container-capped host: the worker pool and inference threads are sized "
            "from these limits, not from the host's core count",
        ),
        Check(
            "whisper model fits host",
            fit_status,
            note or f"'{model}' (~{per_job:.2f} GiB per job) fits the available memory",
            "set CLIPFORGE_WHISPER_MODEL (or [whisper] model) to a smaller model" if note else "",
        ),
        Check(
            "parallel jobs fit host",
            "ok",
            f"configured {settings.hosted.worker_concurrency}, "
            f"memory allows {hostlimits.recommended_concurrency(resolved, 8)}",
            "",
        ),
    ]


def check_worker(settings: Settings) -> list[Check]:
    """Remote worker reachability — decides where renders actually run.

    Once WORKER_URL is set, dispatch is unconditional (app/task.py hands the
    job over and returns; there is no local fallback). An unreachable worker
    therefore fails every job instead of degrading, which is why this is a
    FAIL and not a WARN. The token itself is never printed — only whether one
    is configured.
    """
    from app import hosting  # noqa: PLC0415 - avoids an import cycle

    url = settings.hosted.worker_url
    if not url:
        return [Check(
            "remote worker", "ok",
            "not configured — the pipeline runs in this process",
            "set WORKER_URL to render on a bigger machine (see README, Tailscale mode B)",
        )]
    if not settings.hosted.worker_token:
        return [Check(
            "remote worker", "fail",
            f"{url} is set but WORKER_TOKEN is missing",
            "set WORKER_TOKEN to the same secret the agent was started with; "
            "progress/complete callbacks would be rejected as 401",
        )]
    probe = hosting.remote_probe(settings)
    if probe.get("online"):
        return [Check(
            "remote worker", "ok",
            f"{url} reachable — {probe.get('detail')}, "
            f"queue depth {probe.get('queueDepth', 0)}",
            "",
        )]
    return [Check(
        "remote worker", "fail",
        f"{url} unreachable — {probe.get('detail')}",
        "start the agent (python3 tools/worker_agent.py --server <this host>), and "
        "confirm this server can reach that address: a 100.x tailnet IP needs "
        "tailscale on BOTH ends, and a public IP needs a forwarded port",
    )]


def check_network() -> list[Check]:
    code, _ = _run(["curl", "-sI", "-m", "5", "https://www.youtube.com"], timeout=10)
    if code == 0:
        return [Check("network (yt-dlp fetch)", "ok", "youtube reachable")]
    return [Check("network (yt-dlp fetch)", "warn", "could not reach youtube.com — URL jobs will fail, uploads still work", "")]


def check_encode(settings: Settings) -> list[Check]:
    """Which video encoder this host will actually use, and what it costs.

    A missing hardware encoder is a WARN, not a FAIL: software rendering works,
    it is just slow on a small host. Saying so explicitly is the point — the
    symptom users report is "the render just stops", and the real cause is that
    a 9:16 software encode on one vCPU runs at ~0.1x realtime.
    """
    from app import encode as encode_mod  # noqa: PLC0415 - avoids an import cycle

    choice = encode_mod.choose(settings)
    status = "ok" if choice.hardware else "warn"
    fix = "" if choice.hardware else (
        "expected on a CPU-only host: 9:16 software encoding runs at ~0.1x realtime. "
        + (
            "Lower [video] width/height, or wait for the remote worker check below."
            if settings.hosted.worker_url
            else "Move rendering to a GPU worker (WORKER_URL), or lower [video] width/height."
        )
    )
    checks = [Check("video encoder", status, encode_mod.summary(choice), fix)]
    if choice.degraded_from:
        checks.append(Check(
            "x264 preset", "warn",
            f"{choice.degraded_from} → {choice.preset} (CPU-only fallback)",
            "raise it back to veryfast once a hardware encoder is available",
        ))
    checks.append(Check(
        "render deadline", "ok",
        f"{choice.timeout_sec // 60} min "
        f"({'hardware' if choice.hardware else 'CPU-only fallback'} mode)",
        "",
    ))
    sample_hz = encode_mod.analysis_sample_hz(30.0, hardware=choice.hardware)
    checks.append(Check(
        "reframe sampling", "ok",
        f"{sample_hz:.2f} samples/s",
        "" if choice.hardware else "coarse on CPU-only hosts to leave cycles for the encode",
    ))
    return checks


def run_doctor(settings: Settings) -> dict[str, Any]:
    checks: list[Check] = []
    checks += check_ffmpeg(settings)
    checks += check_encode(settings)
    checks += check_worker(settings)
    checks += check_gpu()
    checks += check_whisper(settings)
    checks += check_llm(settings)
    checks += check_storage(settings)
    checks += check_host(settings)
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
