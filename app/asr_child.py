"""Isolated ASR process: one model, one memory limit, death contained.

Why a process and not a thread: `mkl_malloc: failed to allocate memory` is
fatal to the process that hits it. Transcribing inside the job runner meant a
bad allocation took the runner with it, and the retry re-ran the same 300 s
chunk with the same budget — three identical failures, one dead job.

This process:

* applies the parent's memory limit as RLIMIT_AS **before** loading anything,
  so a decode that does not fit dies here and nowhere else,
* sets the MKL/OpenMP arena knobs before importing ctranslate2 (they are read
  at library init; setting them later does nothing),
* serves chunk WAVs over newline-JSON on stdin/stdout, keeping the model
  resident between chunks of the SAME job,
* unloads the model after `IDLE_UNLOAD_SEC` with no request, so a worker that
  is idling between jobs is not holding 900 MB,
* reports its own peak RSS so the parent can log what each chunk cost.

Never prints secrets: requests carry a path and decode settings only.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time

# MUST run before ctranslate2 / OpenMP initialise. One core, one arena family:
# the default per-thread arenas are what pushed int8 `small` past the cgroup.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

IDLE_UNLOAD_SEC = 300.0
MB = 1024 * 1024


def _trim() -> None:
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - best effort, per-allocator
        pass


def peak_rss_mb() -> float:
    try:
        for line in open("/proc/self/status", encoding="utf-8"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _apply_limit(limit_mb: float | None) -> int | None:
    """RLIMIT_AS = the parent's budget. Returns the bytes actually applied."""
    if not limit_mb:
        return None
    import resource

    limit = int(limit_mb * MB)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        # A hard limit below what is already mapped cannot be applied; the
        # parent's budget checks still gate the work.
        return None
    return limit


def _read_wav(path: str):
    """16 kHz mono PCM WAV -> float32, using the stdlib only.

    `soundfile` is not a hard dependency of faster-whisper's CPU wheel, so the
    child cannot assume it (it was missing on this host and every attempt died
    with ModuleNotFoundError).
    """
    import wave

    import numpy as np

    with wave.open(path, "rb") as fh:
        rate = fh.getframerate()
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        frames = fh.readframes(fh.getnframes())
    if rate != 16000 or channels != 1 or width != 2:  # pragma: no cover
        raise ValueError(f"chunk WAV is {rate}Hz/{channels}ch/{width*8}bit, expected 16kHz mono 16-bit")
    return np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0


def _words(segments) -> list[dict]:
    from app.asr import _segment_confidence, _word_confidence  # noqa: PLC0415

    out: list[dict] = []
    for seg in segments:
        seg_conf = _segment_confidence(seg)
        for w in seg.words or []:
            text = (w.word or "").strip()
            if not text:
                continue
            out.append({
                "start": float(w.start),
                "end": float(w.end),
                "word": text,
                "conf": _word_confidence(w, seg_conf),
            })
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(json.dumps({"ok": False, "error": "no config"}))
        return 2
    cfg = json.loads(argv[0])
    limit = _apply_limit(cfg.get("limit_mb"))
    model_name = str(cfg.get("model") or "small")
    compute = str(cfg.get("compute") or "int8")
    threads = int(cfg.get("threads") or 1)
    try:
        from faster_whisper import WhisperModel

        from app import modelstore  # noqa: PLC0415

        # Loaded from the bucket-backed copy when one exists; the disk copy is
        # transient and is released when the batch ends.
        model = WhisperModel(
            modelstore.resolve(model_name), device="cpu", compute_type=compute,
            cpu_threads=max(1, threads), num_workers=1,
        )
    except Exception as exc:  # noqa: BLE001 - reported, the parent escalates
        print(json.dumps({
            "ok": False, "memory": True,
            "error": f"model {model_name} could not load under {limit or 0} bytes: {exc}",
        }), flush=True)
        return 3
    last_used = time.time()

    def _idle() -> None:
        nonlocal model
        while True:
            time.sleep(15)
            if time.time() - last_used < IDLE_UNLOAD_SEC:
                continue
            if model is None:
                continue
            model = None
            _trim()
            sys.stderr.write(
                f"[asr-child] model unloaded after {IDLE_UNLOAD_SEC:.0f}s idle "
                f"(peak {peak_rss_mb():.0f} MB)\n"
            )
            sys.stderr.flush()
            return

    threading.Thread(target=_idle, daemon=True).start()
    print(json.dumps({
        "ok": True, "ready": True, "model": model_name, "compute": compute,
        "threads": threads, "limit_bytes": limit, "pid": os.getpid(),
    }), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        op = req.get("op") or "decode"
        if op == "quit":
            break
        if op == "unload":
            model = None
            _trim()
            print(json.dumps({"ok": True, "unloaded": True}), flush=True)
            continue
        if model is None:
            print(json.dumps({"ok": False, "memory": True,
                              "error": "model unloaded while idle"}), flush=True)
            return 3
        last_used = time.time()
        try:
            audio = _read_wav(str(req["wav"]))
            kwargs = {
                "language": req.get("language"),
                "task": "transcribe",
                "beam_size": int(req.get("beam_size") or 5),
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.2,
                "no_speech_threshold": 0.6,
                "vad_filter": bool(req.get("vad", True)),
            }
            if kwargs["vad_filter"]:
                kwargs["vad_parameters"] = {"min_silence_duration_ms": 300, "speech_pad_ms": 60}
            segments, info = model.transcribe(audio, **kwargs)
            words = _words(segments)
            print(json.dumps({
                "ok": True,
                "words": words,
                "language": getattr(info, "language", None),
                "peak_rss_mb": round(peak_rss_mb(), 1),
            }), flush=True)
            _trim()
        except (MemoryError, OSError, RuntimeError) as exc:
            text = f"{type(exc).__name__}: {exc}"
            memory = any(
                token in text.lower()
                for token in ("memory", "mkl", "alloc", "segment", "oom")
            )
            print(json.dumps({"ok": False, "memory": memory, "error": text}), flush=True)
            # A process that hit the allocator limit is not trustworthy: the
            # parent starts a fresh one for the next (lighter) attempt.
            return 3 if memory else 1
        except Exception as exc:  # noqa: BLE001 - decoder bug: report, don't hang
            print(json.dumps({"ok": False, "memory": False,
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
