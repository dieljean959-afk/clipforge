#!/usr/bin/env python3
"""ASR engine, queue-pool and ETA checks (plus an opt-in speed benchmark).

Always runs (fast, no model download):
  1. engine resolution — which backend transcribes, on which device, and why
  2. queue pool — 3+ jobs are dispatched, run CONCURRENTLY, release their slots
     and report a real queue position
  3. ETA — RTF priors, measured-progress extrapolation, persistence, and the
     value the pipeline actually writes onto a job

Opt-in benchmark (CLIPFORGE_BENCH=1) — batches vs one-decode-per-window on the
SAME audio through the SAME CTranslate2 engine, reporting the real-time factor
before and after. That A/B is the engine-level speed change; see README for how
WhisperX layers forced alignment on top.

    python3 tools/asr_engine_test.py
    CLIPFORGE_BENCH=1 CLIPFORGE_BENCH_SECONDS=120 python3 tools/asr_engine_test.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'OK ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# 1. engine resolution
# --------------------------------------------------------------------------- #

def section_engine(settings) -> None:
    from app import asr

    print("== 1. engine resolution")
    status = asr.status(settings)
    ok, reason = asr.whisperx_available()
    check("engine resolved to a known backend", status.engine in asr.ENGINES, status.engine)
    check("device/compute resolved (never 'auto' at runtime)",
          status.device in {"cpu", "cuda"} and status.compute_type in {"int8", "float16"},
          f"{status.device}/{status.compute_type}")
    check("batching on (the whole point of the swap)", status.batched is True)
    check("batch size in range", 1 <= status.batch_size <= 64, str(status.batch_size))
    check("engine choice is explained, never silent", bool(status.detail), status.detail)
    check("whisperx availability reported honestly", isinstance(ok, bool), f"{ok} — {reason}")
    if not ok:
        check(
            "fallback detail names the install command",
            "whisperx" in status.detail.lower(),
            status.detail,
        )


def section_whisperx_compat(settings) -> None:
    """Exercise the API boundaries that changed across WhisperX releases."""
    from app import asr

    print("== 1b. WhisperX API compatibility")
    captured: dict[str, dict] = {}

    def new_load(name, device, compute_type, language, asr_options=None,
                 vad_method=None, threads=4):
        captured["new"] = {
            "name": name, "device": device, "compute_type": compute_type,
            "language": language, "asr_options": asr_options,
            "vad_method": vad_method, "threads": threads,
        }
        return "new-model"

    wx_new = SimpleNamespace(load_model=new_load)
    with patch.object(asr.importlib, "import_module", return_value=wx_new):
        result = asr._load_whisperx("medium", "cpu", "int8", 4, settings, None)
    check("new WhisperX load uses Silero VAD", captured["new"]["vad_method"] == "silero")
    check(
        "decoder safeguards go through load_model(asr_options=...)",
        captured["new"]["asr_options"] == {
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.2,
            "no_speech_threshold": 0.6,
        },
        str(captured["new"]["asr_options"]),
    )
    check(
        "new WhisperX load returns the loaded engine handle",
        result.engine == asr.ENGINE_WHISPERX and result.model == "new-model",
        str(result),
    )

    def old_load(name, device, compute_type, language, asr_options=None, threads=4):
        captured["old"] = {"asr_options": asr_options, "threads": threads}
        return "old-model"

    wx_old = SimpleNamespace(load_model=old_load)
    with patch.object(asr.importlib, "import_module", return_value=wx_old):
        asr._load_whisperx("medium", "cpu", "int8", 2, settings, None)
    check(
        "3.3.x-compatible load keeps safeguards without vad_method",
        captured["old"]["asr_options"] == captured["new"]["asr_options"]
        and captured["old"]["threads"] == asr.hostlimits.inference_threads(),
        str(captured["old"]),
    )

    segment = {
        "start": 0.0, "end": 1.0, "text": "hello",
        "words": [{"word": "hello", "start": 0.1, "end": 0.8}],
    }

    def align(segments, model, metadata, audio, device, return_char_alignments=False):
        return {"segments": segments}

    class OldModel:
        def transcribe(self, audio, batch_size=None, language=None, task=None,
                       print_progress=False, combined_progress=False, verbose=False):
            return {"segments": [segment], "language": language or "en"}

    class NewModel:
        def transcribe(self, audio, batch_size=None, language=None, task=None,
                       print_progress=False, combined_progress=False, verbose=False,
                       progress_callback=None):
            if progress_callback is not None:
                progress_callback(50.0)
            return {"segments": [segment], "language": language or "en"}

    for label, model in (("old", OldModel()), ("new", NewModel())):
        wx = SimpleNamespace(align=align)
        loaded = asr.LoadedEngine(
            engine=asr.ENGINE_WHISPERX, device="cpu", compute_type="int8",
            model_name="medium", batch_size=1, model=model, whisperx=wx,
        )
        progress: list[float] = []
        words, detected = loaded.asr_words([0.0] * 16000, "en", progress.append)
        check(
            f"{label} WhisperX words use aligned timing",
            [(w.word, round(w.start, 2), round(w.end, 2)) for w in words]
            == [("hello", 0.1, 0.8)] and detected == "en",
            str(words),
        )
        check(f"{label} WhisperX reports progress", bool(progress), str(progress))
        # A segment with no avg_logprob must NOT be treated as doubtful: the
        # rule exists to label real uncertainty, and a missing metric is not
        # evidence of any.
        check(
            f"{label} WhisperX words default to full confidence when nothing was measured",
            all(w.conf == 1.0 for w in words),
            str([w.conf for w in words]),
        )

    # ---- per-token confidence plumbing ---------------------------------
    print("== 1c. confidence plumbing (low-confidence captions)")
    import math

    from app import subtitle  # noqa: PLC0415 - only this section needs it

    obj_seg = SimpleNamespace(avg_logprob=-1.2, words=[
        SimpleNamespace(word="clear", start=0.0, end=0.5, probability=0.98),
        SimpleNamespace(word="huh", start=0.5, end=1.0, probability=0.21),
        # An engine build that does not fill probability at all.
        SimpleNamespace(word="nopro", start=1.0, end=1.4),
    ])
    check(
        "exp(avg_logprob) is the segment fallback (0.30)",
        abs(asr._segment_confidence(obj_seg) - math.exp(-1.2)) < 1e-6,
        str(asr._segment_confidence(obj_seg)),
    )
    check(
        "WhisperX's dict segments are read too",
        abs(asr._segment_confidence({"avg_logprob": -1.2}) - math.exp(-1.2)) < 1e-6,
    )
    check(
        "a missing avg_logprob means 'not measured', not 'doubtful'",
        asr._segment_confidence(SimpleNamespace()) == 1.0,
    )
    check(
        "a per-word probability wins over the segment value",
        abs(asr._word_confidence(obj_seg.words[0], 0.3) - 0.98) < 1e-6,
    )
    check(
        "a word with no probability falls back to its segment",
        abs(asr._word_confidence(obj_seg.words[2], 0.3) - 0.3) < 1e-6,
    )
    check(
        "a zero probability is treated as 'not filled in', not as certain-zero",
        abs(asr._word_confidence(SimpleNamespace(probability=0.0), 0.44) - 0.44) < 1e-6,
    )
    check(
        "AsrWord without a conf reads as unmeasured (1.0), never as a guess",
        asr.AsrWord(start=0.0, end=1.0, word="x").conf == 1.0,
    )
    check(
        "as_dict() carries the confidence for the caption layer",
        "conf" in subtitle.Word(start=0.0, end=1.0, word="x", conf=0.5).as_dict(),
    )


# --------------------------------------------------------------------------- #
# 2. queue pool: 3+ simultaneous jobs
# --------------------------------------------------------------------------- #

def section_queue(tmp: Path) -> None:
    print("== 2. queue pool (3 simultaneous jobs, concurrency 3)")
    import app.worker as worker_mod
    from app.config import load_config
    from app.store import LocalStore
    from app.worker import InProcessWorker

    os.environ["CLIPFORGE_DOWNLOADS_DIR"] = str(tmp / "downloads")
    os.environ["CLIPFORGE_OUTPUTS_DIR"] = str(tmp / "outputs")
    os.environ["CLIPFORGE_DATA_DIR"] = str(tmp / "data")
    settings = load_config(quiet=True)
    # 3 jobs, 3 slots — the pool must not be narrower than the queue
    settings.hosted.worker_concurrency = 3
    store = LocalStore(settings.app.paths)
    worker = InProcessWorker(settings, store)
    # The deployment's real limit may be one slot. This focused dispatcher test
    # overrides it explicitly so queue ordering/release is exercised with 3 jobs
    # without pretending this constrained host can run three model instances.
    worker.concurrency = 3

    job_ids = [store.create_job("upload", "missing.mp4")["_id"] for _ in range(3)]
    for job_id in job_ids:
        worker.enqueue(job_id)

    # positions are written at enqueue time, BEFORE any worker picks them up
    positions = [store.get_job(j)["queuePosition"] for j in job_ids]
    check("queue positions assigned at enqueue", positions == [0, 1, 2], str(positions))
    detail = store.get_job(job_ids[2])["progressDetail"] or ""
    check("queue position explained in the job detail", "waiting behind 2" in detail.lower(), detail)
    check("enqueue reports the real depth", worker.status()["queueDepth"] == 3,
          str(worker.status()["queueDepth"]))

    worker.start()
    deadline = time.time() + 60
    peak_active = 0
    while time.time() < deadline:
        status = worker.status()
        peak_active = max(peak_active, status["activeCount"])
        if all((store.get_job(j) or {}).get("stage") in {"complete", "failed"} for j in job_ids):
            break
        time.sleep(0.3)

    stages = [(store.get_job(j) or {}).get("stage") for j in job_ids]
    check("every queued job reached a terminal state (none left hanging)",
          all(s in {"complete", "failed"} for s in stages), str(stages))
    check("jobs actually ran in PARALLEL (pool > 1)", peak_active >= 2,
          f"peak concurrent jobs = {peak_active}")

    final = worker.status()
    check("all worker slots released after the jobs finished",
          final["activeCount"] == 0, str(final["activeCount"]))
    check("queue drained", final["queueDepth"] == 0, str(final["queueDepth"]))
    check("queue positions cleared once picked up",
          all(store.get_job(j)["queuePosition"] is None for j in job_ids))
    check("failed jobs carry a readable error (missing upload)",
          all((store.get_job(j) or {}).get("error") for j in job_ids),
          str([(store.get_job(j) or {}).get("error") for j in job_ids]))

    worker.stop(timeout=3)
    for job_id in job_ids:
        store.delete_job(job_id)
    worker_mod.STALL_TIMEOUT_SEC = worker_mod.STALL_TIMEOUT_SEC  # unchanged, keeps import used

    # Retry must kill a live old process before enqueueing a new attempt. This
    # uses a fake process so the check never starts a second real model load.
    retry_job = store.create_job("upload", "missing-retry.mp4")["_id"]
    class FakeProcess:
        pid = 424242
        def poll(self):
            return None
    old_process = FakeProcess()
    worker._active[retry_job] = old_process
    worker._current_job = retry_job
    worker._current_proc = old_process
    killed: list[object] = []
    original_kill = worker._kill_process
    worker._kill_process = lambda proc: killed.append(proc)  # type: ignore[method-assign]
    try:
        worker.retry(retry_job)
    finally:
        worker._kill_process = original_kill  # type: ignore[method-assign]
        worker._active.pop(retry_job, None)
        worker._current_job = None
        worker._current_proc = None
    check("retry kills the old process before fresh enqueue", killed == [old_process], str(killed))
    retried = store.get_job(retry_job)
    check("retry resets the job to queued", retried.get("stage") == "queued", str(retried.get("stage")))
    store.delete_job(retry_job)


# --------------------------------------------------------------------------- #
# 3. ETA
# --------------------------------------------------------------------------- #

def section_eta(tmp: Path) -> None:
    print("== 3. ETA estimation")
    from app import eta

    check("RTF prior is faster on GPU than CPU",
          eta.prior_rtf("faster-whisper", "cuda", "medium") < eta.prior_rtf("faster-whisper", "cpu", "medium"))
    check("RTF prior grows with model size",
          eta.prior_rtf("faster-whisper", "cpu", "tiny") < eta.prior_rtf("faster-whisper", "cpu", "medium")
          < eta.prior_rtf("faster-whisper", "cpu", "large-v3"))
    check("whisperx prior is faster than the unaligned engine on CPU",
          eta.prior_rtf("whisperx", "cpu", "medium") < eta.prior_rtf("faster-whisper", "cpu", "medium"),
          f"{eta.prior_rtf('whisperx', 'cpu', 'medium')} vs {eta.prior_rtf('faster-whisper', 'cpu', 'medium')}")

    check("no ETA before there is enough signal",
          eta.eta_from_progress(5.0, 0.01) is None)
    # 100s elapsed at 50% → another ~100s to go
    eta50 = eta.eta_from_progress(100.0, 0.5)
    check("measured ETA = elapsed/done * remaining", eta50 is not None and abs(eta50 - 100.0) < 1.0,
          str(eta50))
    check("ETA hits zero at completion", eta.eta_from_progress(100.0, 1.0) == 0.0)

    seconds, source = eta.estimate(elapsed_sec=1.0, fraction=0.0, prior_remaining_sec=240.0)
    check("falls back to the RTF prior before progress", seconds == 240.0 and source == "rtf-prior",
          f"{seconds} ({source})")
    seconds, source = eta.estimate(elapsed_sec=60.0, fraction=0.5, prior_remaining_sec=240.0)
    check("switches to the measured estimate once progress exists",
          source == "measured" and seconds is not None and abs(seconds - 60.0) < 1.0,
          f"{seconds} ({source})")
    check("audio duration x RTF is the prior shape",
          abs(eta.eta_from_prior(600.0, 0.5) - 300.0) < 1e-6)
    check("tail estimate grows with clip count",
          eta.tail_estimate_sec(10) > eta.tail_estimate_sec(2))

    store = eta.RtfStore(tmp / "eta")
    key = eta.rtf_key("faster-whisper", "cpu", "medium")
    check("no measurement yet → documented prior",
          store.effective_rtf("faster-whisper", "cpu", "medium")[1].startswith("documented"))
    store.observe(key, 0.30, audio_seconds=100.0)
    store.observe(key, 0.10, audio_seconds=100.0)
    check("observation recorded", store.get(key) is not None, str(store.get(key)))
    check("EMA moves toward the faster measurement", store.get(key) < 0.30, str(store.get(key)))
    check("nonsense RTF values ignored", (store.observe(key, float("nan")), store.get(key) is not None)[1])
    check("measurement is preferred over the prior",
          store.effective_rtf("faster-whisper", "cpu", "medium")[1].startswith("measured"))
    reloaded = eta.RtfStore(tmp / "eta")
    check("RTF stats persist across restarts", reloaded.get(key) is not None, str(reloaded.get(key)))

    # the value the pipeline actually writes onto a job
    from app.config import load_config
    from app.store import LocalStore
    from app.task import Pipeline

    os.environ["CLIPFORGE_DOWNLOADS_DIR"] = str(tmp / "downloads")
    os.environ["CLIPFORGE_OUTPUTS_DIR"] = str(tmp / "outputs")
    os.environ["CLIPFORGE_DATA_DIR"] = str(tmp / "data")
    settings = load_config(quiet=True)
    pipeline = Pipeline(settings, LocalStore(settings.app.paths))
    job = pipeline.store.create_job("upload", "missing.mp4")
    job["durationSec"] = 600.0
    pipeline.store.write_job(job)
    pipeline._progress(job["_id"], "transcribing", 0.0, "test",
                       prior_eta_sec=eta.eta_from_prior(600.0, 0.2))
    stored = pipeline.store.get_job(job["_id"])
    check("job gets stageEtaSec from the prior", stored["stageEtaSec"] == 120.0, str(stored["stageEtaSec"]))
    check("job gets a whole-job ETA", (stored["totalEtaSec"] or 0) > stored["stageEtaSec"],
          f"{stored['totalEtaSec']} vs {stored['stageEtaSec']}")
    check("ETA is labelled with its derivation", stored["etaSource"] == "rtf-prior", stored["etaSource"])
    check("stageStartedAt recorded for elapsed-time maths", bool(stored.get("stageStartedAt")))
    check("no measured ETA yet at 0% (nothing to extrapolate from)",
          stored["stageEtaSec"] == 120.0, str(stored["stageEtaSec"]))
    # simulate 40s spent reaching 50%: the ETA must switch to the measured form
    stored["stageStartedAt"] = int(time.time() * 1000) - 40_000
    pipeline.store.write_job(stored)
    pipeline._progress(job["_id"], "transcribing", 0.5, "test", prior_eta_sec=120.0)
    stored = pipeline.store.get_job(job["_id"])
    check("later progress switches the ETA to measured", stored["etaSource"] == "measured", stored["etaSource"])
    check("measured ETA is ~40s of remaining work at 50% done",
          abs((stored["stageEtaSec"] or 0) - 40.0) < 3.0, str(stored["stageEtaSec"]))
    check("stage timing recorded when a stage completes",
          stored.get("stageTimings") is not None)
    pipeline.store.delete_job(job["_id"])


# --------------------------------------------------------------------------- #
# 4. benchmark (opt-in)
# --------------------------------------------------------------------------- #

def _synthetic_wav(dest: Path, seconds: int) -> Path:
    from app.config import ffmpeg_bin

    subprocess.run(
        [
            ffmpeg_bin(), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
            "-f", "lavfi", "-i", f"anoisesrc=color=white:duration={seconds}:amplitude=0.05",
            "-filter_complex", "amix=inputs=2",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest),
        ],
        check=True, capture_output=True, timeout=300,
    )
    return dest


def section_benchmark(tmp: Path, seconds: int) -> None:
    """Per-window decoding (the old shape) vs one batched pass (the new shape).

    Same audio, same model, same engine — this isolates exactly the change the
    WhisperX swap brings at the CTranslate2 level. VAD is off so the comparison
    measures decode throughput rather than how much silence VAD strips.
    """
    print(f"== 4. benchmark: {seconds}s of audio, per-30s-window vs batched")
    import wave

    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import BatchedInferencePipeline

    from app import asr as asr_mod
    from app import hostlimits
    from app.config import load_config

    settings = load_config(quiet=True)
    # Benchmark what the pipeline would ACTUALLY load. A configured model that
    # does not fit is downgraded at runtime; benchmarking the configured name
    # instead measures a model production never uses — or OOMs on a small host.
    configured = settings.whisper.model
    model_name, downgrade = hostlimits.resolved_model(configured)
    if downgrade:
        print(f"   host limit: '{configured}' -> '{model_name}' ({downgrade.split('.')[0]}.)")
    # Honour [whisper] device/compute_type the way app.asr does. Hardcoding
    # "cpu" here reported CPU RTF for a machine that has a GPU, which is the
    # one number this benchmark exists to inform.
    device, compute = asr_mod._resolve_device(settings)
    threads = hostlimits.inference_threads()
    batch = settings.whisper.batch_size
    wav = _synthetic_wav(tmp / "bench.wav", seconds)
    with wave.open(str(wav), "rb") as fh:
        audio = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    sr = 16000

    try:
        t0 = time.time()
        model = WhisperModel(model_name, device=device, compute_type=compute, cpu_threads=threads)
        load_s = time.time() - t0
    except Exception as exc:
        # Previously a bare `return`, so the run ended with a clean pass and no
        # numbers — the most dangerous possible failure for a benchmark.
        print(f"   SKIP benchmark — could not load '{model_name}' on {device}/{compute}: {exc}")
        check("benchmark produced RTF numbers", False, f"model load failed: {exc}")
        return
    print(f"   model '{model_name}' loaded in {load_s:.1f}s on {device}/{compute} ({threads} threads)")
    if device == "cpu":
        print("   NOTE: no CUDA device visible — these RTF figures are CPU-only.")

    common = dict(
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=False,
        compression_ratio_threshold=2.2,
        no_speech_threshold=0.6,
    )

    # --- BEFORE: one decode per 30s window (the previous engine shape) ------
    t0 = time.time()
    for start in range(0, seconds, 30):
        chunk = audio[start * sr : min(seconds, start + 30) * sr]
        if len(chunk) < sr // 2:
            continue
        segments, _ = model.transcribe(chunk, language="en", **common)
        list(segments)  # generators are lazy — consume or nothing is measured
    before_wall = time.time() - t0

    # --- AFTER: one batched pass over the whole buffer ---------------------
    # BatchedInferencePipeline raises "No clip timestamps found" for any audio
    # >= 30s when vad_filter is False and clip_timestamps is unset, so the
    # default 120s buffer used to crash here regardless of host. We are
    # measuring decode throughput rather than VAD, so hand it the same uniform
    # 30s window split the BEFORE loop walks — identical audio on both sides.
    windows = [{"start": s, "end": min(s + 30, seconds)} for s in range(0, seconds, 30)]
    print(f"   batched pass: {len(windows)} x 30s window(s), batch_size={batch} "
          f"(fills {min(len(windows), batch)})")
    pipeline = BatchedInferencePipeline(model=model)
    t0 = time.time()
    segments, _ = pipeline.transcribe(
        audio, language="en", batch_size=batch, clip_timestamps=windows, **common
    )
    list(segments)
    after_wall = time.time() - t0

    before_rtf = before_wall / seconds
    after_rtf = after_wall / seconds
    speedup = before_wall / after_wall if after_wall > 0 else 0.0
    print(f"   BEFORE (per-window)  : {before_wall:.1f}s wall, RTF {before_rtf:.3f}")
    print(f"   AFTER  (batch {batch:>2})    : {after_wall:.1f}s wall, RTF {after_rtf:.3f}")
    print(f"   speed-up             : {speedup:.2f}x  (RTF {before_rtf:.3f} → {after_rtf:.3f})")
    check("batched decoding is not slower than per-window decoding", after_wall <= before_wall,
          f"{after_wall:.1f}s vs {before_wall:.1f}s")
    check("RTF improved or held", after_rtf <= before_rtf + 1e-6,
          f"{before_rtf:.3f} → {after_rtf:.3f}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cf-asr-"))
    try:
        from app.config import load_config

        settings = load_config(quiet=True)
        section_engine(settings)
        section_whisperx_compat(settings)
        section_queue(tmp)
        section_eta(tmp)
        if os.getenv("CLIPFORGE_BENCH", "").strip() in {"1", "true", "yes"}:
            section_benchmark(tmp, int(os.getenv("CLIPFORGE_BENCH_SECONDS", "120")))
        else:
            print("== 4. benchmark SKIPPED (set CLIPFORGE_BENCH=1 to run it)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"ASR ENGINE TEST: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("ASR ENGINE TEST: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
