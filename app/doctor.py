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
    if llm_mod.provider(settings) in llm_mod.OPENAI_PROVIDER_KEYS:
        return _check_hosted(settings)
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


def _check_hosted(settings: Settings) -> list[Check]:
    """Hosted-provider key presence + optional live probe of the models endpoint.

    Covers DeepSeek, NVIDIA NIM and Groq, which are the same OpenAI-compatible
    shape, so the probe is identical: a GET /models rather than a chat call,
    because it costs no quota and still proves both the key AND the network
    path work.
    """
    from app import llm as llm_mod  # noqa: PLC0415 - avoids an import cycle at boot
    from app.llm import GROQ_USER_AGENT  # noqa: PLC0415 - same edge-block guard

    which = llm_mod.provider(settings)
    cfg = llm_mod.openai_cfg(settings, which)
    env_name = llm_mod.OPENAI_PROVIDER_KEYS[which][1]
    api_key = llm_mod.openai_api_key(which, settings)

    live = llm_mod._LIVE_KEY_RESOLVERS[which]()
    live.refresh(force=True)
    status = live.status()
    if not status.get("keyLoaded"):
        return [Check(f"{which} key", "fail",
                      status.get("problem") or status.get("source", "not found"),
                      f"set {env_name} in the Keys tab (or .env)")]
    checks = [Check(f"{which} key", "ok", f"loaded from {status.get('source')}")]
    # DeepSeek is the app's detector, and its balance is the difference between
    # a batch running and every batch failing at detection. A 1-token chat is
    # cheap proof, so it always runs for this provider (the key stays masked).
    # Other providers keep the opt-in live probe.
    live_probe = which == "deepseek" or os.getenv("CLIPFORGE_DOCTOR_LIVE", "").lower() in {
        "1", "true", "yes"
    }
    if live_probe:
        base = str(cfg.base_url).rstrip("/")
        req = urllib.request.Request(  # noqa: S310 - https base from config
            f"{base}/models",
            headers={
                "Authorization": f"Bearer {api_key or ''}",
                # Same Cloudflare 1010 signature ban as the Groq chat call —
                # without this the probe reports a bogus "bad key" for a good key.
                "User-Agent": GROQ_USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
            want = cfg.model
            note = ""
            if ids and want not in ids:
                note = f" (CONFIGURED MODEL NOT IN CATALOG — pick one of: {', '.join(ids[:6])})"
            checks.append(Check(f"{which} API", "ok" if not note else "warn",
                                f"live call ok, {len(ids)} model(s) visible{note}"))
            if which == "deepseek":
                try:
                    payload = json.dumps({
                        "model": cfg.model,
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "max_tokens": 1,
                        "temperature": 0,
                    }).encode("utf-8")
                    chat = urllib.request.Request(  # noqa: S310 - configured https base
                        f"{base}/chat/completions",
                        data=payload,
                        headers={
                            "Authorization": f"Bearer {api_key or ''}",
                            "Content-Type": "application/json",
                            "User-Agent": GROQ_USER_AGENT,
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(chat, timeout=20) as resp:  # noqa: S310
                        json.loads(resp.read().decode("utf-8"))
                    checks.append(Check(
                        "deepseek 1-token", "ok",
                        "live chat completion returned (key masked, 1 token spent)",
                    ))
                except urllib.error.HTTPError as exc:
                    # Report the provider's status/message, never the request
                    # headers or the key. A 402 here means "out of credit",
                    # which is the real reason detection would fail.
                    raw = exc.read()[:500].decode("utf-8", "replace")
                    try:
                        why = json.loads(raw).get("error", {}).get("message") or raw
                    except Exception:  # noqa: BLE001
                        why = raw
                    checks.append(Check(
                        "deepseek 1-token", "fail",
                        f"live chat failed: HTTP {exc.code} — {str(why)[:180]}",
                        "check quota/billing; the key itself was not printed",
                    ))
                except Exception as exc:  # noqa: BLE001 - never print the key
                    checks.append(Check(
                        "deepseek 1-token", "fail",
                        f"live chat failed: {type(exc).__name__}",
                        "check quota/billing; the key itself was not printed",
                    ))
        except urllib.error.HTTPError as exc:
            checks.append(Check(f"{which} API", "fail", f"live call failed: HTTP {exc.code}",
                                "check the key + network"))
        except Exception as exc:  # noqa: BLE001 - any transport error is a fail
            checks.append(Check(f"{which} API", "fail", f"live call failed: {exc}",
                                "check the key + network"))
    else:
        checks.append(Check(f"{which} API", "warn",
                            "not tested (set CLIPFORGE_DOCTOR_LIVE=1 to test)",
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


def check_routes(settings: Settings) -> list[Check]:
    """Hit the real routes on the running API, not a mock."""
    base = f"http://127.0.0.1:{settings.app.port}"
    checks: list[Check] = []
    probes = (
        ("/api/health", "GET", None, "route /api/health"),
        ("/api/clips/health", "GET", None, "route /api/clips/health"),
        ("/api/storage/reset", "POST", b'{"scope":"bucket","apply":false}',
         "route /api/storage/reset (bucket dry-run)"),
    )
    for path, method, data, name in probes:
        try:
            req = urllib.request.Request(
                base + path, method=method, data=data,
                headers={"Content-Type": "application/json"} if data else {},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                checks.append(Check(name, "ok", f"HTTP {resp.status}"))
        except urllib.error.HTTPError as exc:
            checks.append(Check(name, "fail", f"HTTP {exc.code}"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(name, "warn", f"API not reachable: {exc}",
                                "start the app, then run doctor again"))
            break
    return checks


def check_bucket(settings: Settings) -> list[Check]:
    """A real put/head/ranged-get/delete against the configured bucket."""
    from app import objectstore

    cfg = objectstore.config_from_env()
    if not cfg.enabled:
        return [Check("bucket round-trip", "warn", "bucket disabled (local-only media)", "")]
    try:
        result = objectstore.test_roundtrip(cfg)
        steps = ", ".join(
            f"{s['step']}={'ok' if s.get('ok') else 'FAIL'}" for s in result.get("steps", [])
        )
        return [Check(
            "bucket round-trip", "ok" if result.get("ok") else "fail",
            f"{cfg.bucket}: {steps}",
            "" if result.get("ok") else "check endpoint/bucket/keys in the Keys tab",
        )]
    except Exception as exc:  # noqa: BLE001
        return [Check("bucket round-trip", "fail", f"{type(exc).__name__}: {exc}", "check bucket settings")]


def check_clip_health(settings: Settings, store: Any = None) -> list[Check]:
    """HEAD + ranged 206 + stored fps for every ready clip."""
    from app import cliphealth
    from app.store import LocalStore

    st = store or LocalStore(settings.app.paths)
    try:
        result = cliphealth.check(st, settings)
    except Exception as exc:  # noqa: BLE001
        return [Check("clip health", "fail", f"{type(exc).__name__}: {exc}", "inspect the bucket")]
    if not result.get("checked"):
        return [Check("clip health", "warn", "no ready clips to check", "")]
    return [Check(
        "clip health", "ok" if not result.get("bad") else "fail",
        f"{result['checked']} ready clip(s), {result['bad']} failed HEAD/range/fps",
        "" if not result.get("bad") else "open the clip and press Retry",
    )]


def check_audio_sync(settings: Settings, store: Any = None) -> list[Check]:
    """Audio placement + duplicate results for the newest gated batch.

    The gate runs locally at render time and stores only numbers/reasons. This
    checker does not download media or contact an ASR provider; it reports the
    newest batch's recorded placement lag/correlation and re-runs the pure
    duplicate comparison against the stored fingerprints.
    """
    from app import audiosync
    from app.store import LocalStore

    st = store or LocalStore(settings.app.paths)
    try:
        clips = st.list_all_clips(limit=250)
    except Exception as exc:  # noqa: BLE001
        return [Check("audio sync", "fail", f"cannot list clips: {exc}", "inspect data/clips")]
    gated = [c for c in clips if isinstance(c.get("audioSync"), dict)]
    if not gated:
        return [Check(
            "audio sync", "warn", "no clips carry an audioSync result yet",
            "render a batch with the new sync gate",
        )]
    latest_job = max(gated, key=lambda c: int(c.get("createdAt") or 0)).get("jobId")
    batch = [c for c in gated if c.get("jobId") == latest_job]
    placement = [c["audioSync"].get("placement") or {} for c in batch]
    lags = [float(p["lagMs"]) for p in placement if p.get("lagMs") is not None]
    corrs = [float(p["correlation"]) for p in placement if p.get("correlation") is not None]
    failed = [c for c in batch if not (c["audioSync"] or {}).get("passed")]
    max_lag = max((abs(v) for v in lags), default=0.0)
    min_corr = min(corrs, default=0.0)
    detail = (
        f"job {latest_job}: {len(batch)} gated, {len(failed)} failed; "
        f"max |lag| {max_lag:.0f} ms, min correlation {min_corr:.3f}"
    )
    fix = "" if not failed else "open the failed clip and press Retry render"
    checks = [Check("audio sync", "ok" if not failed else "fail", detail, fix)]

    seen: list[dict[str, Any]] = []
    dupes = 0
    for c in batch:
        fp = (c["audioSync"] or {}).get("fingerprint") or {}
        if not fp:
            continue
        found = audiosync.duplicate_of(fp, seen)
        if not found.get("ok"):
            dupes += 1
        seen.append({"clipId": c.get("_id"), "fingerprint": fp})
    checks.append(Check(
        "duplicate clips", "ok" if dupes == 0 else "fail",
        f"job {latest_job}: {len(seen)} fingerprinted, {dupes} duplicate(s)",
        "" if dupes == 0 else "reject the duplicate candidate and re-render the batch",
    ))
    return checks


def check_share(settings: Settings) -> list[Check]:
    """A real ranged GET on a share token, from a cookieless request."""
    from app.store import LocalStore

    st = LocalStore(settings.app.paths)
    try:
        clips = st.list_all_clips(limit=50)
    except Exception as exc:  # noqa: BLE001
        return [Check("share link", "warn", f"cannot list clips: {exc}", "")]
    clip = next((c for c in clips if c.get("shareToken") and not c.get("shareRevoked")), None)
    if clip is None:
        return [Check("share link", "warn", "no live share token yet", "open a clip and press Share")]
    # `/s/<token>` is the HTML page (for social unfurlers); the ranged bytes
    # live on the explicit .mp4 endpoint.
    url = f"http://127.0.0.1:{settings.app.port}/s/{clip['shareToken']}.mp4"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-99"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - localhost
            body = resp.read(100)
            ok = resp.status == 206 and len(body) == 100
            return [Check("share link", "ok" if ok else "fail",
                          f"HTTP {resp.status}, {len(body)} bytes ranged", "")]
    except Exception as exc:  # noqa: BLE001
        return [Check("share link", "fail", f"{type(exc).__name__}: {exc}", "check the share route")]


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
    checks += check_routes(settings)
    checks += check_bucket(settings)
    checks += check_clip_health(settings)
    checks += check_audio_sync(settings)
    checks += check_share(settings)
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


# --------------------------------------------------------------------------- #
# /api/status — per-dependency probes, each independent and time-bounded
# --------------------------------------------------------------------------- #

def _probe(name: str, fn, timeout_sec: float = 2.0) -> dict:
    """Run one dependency probe in a helper thread; never block the response."""
    import threading

    box: dict = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported as not-ok
            box["error"] = exc

    th = threading.Thread(target=_run, name=f"probe-{name}", daemon=True)
    th.start()
    th.join(timeout_sec)
    if th.is_alive():
        return {"name": name, "ok": False, "detail": f"no answer in {timeout_sec}s", "ms": int(timeout_sec * 1000)}
    if "error" in box:
        exc = box["error"]
        return {"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    value = box.get("value")
    if isinstance(value, dict):
        out = {"name": name, "ok": bool(value.get("ok", True))}
        out.update({k: v for k, v in value.items() if k != "ok"})
        return out
    return {"name": name, "ok": bool(value), "detail": "" if value else "unavailable"}


def dependency_status(settings: Settings, store=None, boot=None) -> list[dict]:
    """One row per dependency. A slow/failed one names itself, never reddens all.

    The dashboard, the Doctor button and tools/doctor.py all read this, so the
    row order is stable and each row is JSON-safe.
    """
    import time

    from app import asr, objectstore
    from app.hosting import disk_stats
    from app.store import LocalStore

    rows: list[dict] = []

    def _state() -> dict:
        t0 = time.perf_counter()
        st = store if store is not None else LocalStore(settings.app.paths)
        jobs = st.list_jobs()
        return {
            "ok": True,
            "detail": f"{len(jobs)} job(s) readable",
            "ms": int((time.perf_counter() - t0) * 1000),
        }

    rows.append(_probe("state", _state))

    def _bucket() -> dict:
        cfg = objectstore.config_from_env()
        if not cfg.enabled:
            return {"ok": True, "detail": "disabled (local-only media)"}
        return {
            "ok": True,
            "detail": f"{cfg.bucket} @ {cfg.endpoint}",
            "durable": True,
        }

    rows.append(_probe("bucket", _bucket))

    def _model() -> dict:
        info = asr.model_status(settings)
        return {
            "ok": bool(info.get("filesPresent")),
            "detail": info.get("label") or info.get("model"),
            "mismatch": info.get("mismatch"),
            "model": info.get("model"),
            "computeType": info.get("computeType"),
            "cpuThreads": info.get("cpuThreads"),
            "sizeMb": info.get("sizeMb"),
        }

    rows.append(_probe("whisper", _model, timeout_sec=3.0))

    def _llm() -> dict:
        # Presence only: never read, compare or log a key's value. A 1-token
        # call belongs in doctor.py, not in a poll the dashboard runs every 10 s.
        import os as _os

        names = ("DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "NVIDIA_API_KEY")
        present = [n for n in names if _os.environ.get(n)]
        return {
            "ok": bool(present),
            "detail": (
                f"{', '.join(present)} set" if present
                else "no detection key (jobs fail at detection)"
            ),
        }

    rows.append(_probe("deepseek", _llm))

    def _ytdlp() -> dict:
        try:
            import yt_dlp

            ver = getattr(getattr(yt_dlp, "version", None), "__version__", None)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        return {"ok": bool(ver), "detail": str(ver or "importable, no version")}

    rows.append(_probe("yt-dlp", _ytdlp))

    def _ffmpeg() -> dict:
        from app.config import ffmpeg_bin, ffprobe_bin

        return {"ok": True, "detail": f"{ffmpeg_bin()} · ffprobe {ffprobe_bin()}"}

    rows.append(_probe("ffmpeg", _ffmpeg))

    def _disk() -> dict:
        stats = disk_stats(settings.app.paths.data_dir)
        free = stats.get("freeMb")
        return {
            "ok": (free is None) or free > 200,
            "detail": f"{free} MB free" if free is not None else "unknown",
            "freeMb": free,
        }

    rows.append(_probe("disk", _disk))

    def _worker() -> dict:
        from app import worker as worker_mod

        live = getattr(worker_mod, "_active_worker", None)
        if live is None:
            # The supervisor lives on app.main; importing it here would be a
            # cycle, so the API hands it over at boot.
            from app import main as main_mod

            live = getattr(main_mod, "worker", None)
        st = live.status() if live is not None else {}
        return {
            "ok": bool(st.get("online")),
            "detail": (
                f"online={st.get('online')} "
                f"pid={getattr(live, 'daemon_pid', None)} "
                f"concurrency={st.get('concurrency')}"
            ),
        }

    rows.append(_probe("worker", _worker))

    if boot is not None:
        failed = boot.summary()["failedPhases"]
        rows.append({
            "name": "boot",
            "ok": not failed,
            "detail": "all phases ok" if not failed else "failed: " + ", ".join(failed),
        })
    return rows
