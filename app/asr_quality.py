"""Local Whisper decoding with the accuracy settings, in a guarded child.

Differences from :mod:`app.asr_child` (which stays as it is, because the legacy
ladder still uses it): every setting the accuracy plan asks for is exposed per
request — initial prompt, temperature fallback, log-probability threshold,
compression ratio — and the parent reports the probability of each WORD, not
just of each segment, because that is the granularity the "check this" flag
needs.

Memory discipline is the reason this is a separate process at all:

* the child reports its own peak RSS, and the parent SAMPLES its VmRSS;
* the parent kills it the moment it crosses the cap, so a model that does not
  fit this host dies here and not in the API process;
* the cap is derived from the cgroup (`memory.max` - 400 MB for the API),
  not from `MemTotal`.

Measured on this host (tools/asr_model_probe.py, 30 s of real speech):

    large-v3-turbo   OOM-killed at 1581 MB anon-RSS during load
    medium           OOM-killed during load
    distil-medium.en 1142 MB peak, fits
    small             625 MB peak, fits

RLIMIT_AS is deliberately NOT used as the gate: CTranslate2 maps far more
address space than the RSS it needs, so the legacy ladder refuses models that
run fine (medium was refused by RLIMIT_AS and then OOM-killed at a real 1.5 GB,
which is the truth of it).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MB = 1024 * 1024

#: Measured peak RSS per model on this host (see the module docstring). The
#: ladder uses these to decide what to try; a model that is not listed falls
#: back to the conservative table in app.membudget.
MEASURED_PEAK_MB: dict[str, float] = {
    "distil-medium.en": 1142.0,
    "distil-small.en": 780.0,
    "small": 625.0,
    "small.en": 640.0,
    "base": 455.0,
    "tiny": 355.0,
    "medium": 1855.0,      # OOM-killed here; kept so the ladder skips it
    "large-v3-turbo": 1900.0,
    "large-v3": 3200.0,
    "distil-large-v3": 1900.0,
}

#: The API + worker + ffmpeg must survive the decode.
HEADROOM_MB = 400.0

DECODE_DEFAULTS: dict[str, Any] = {
    "beam_size": 5,
    "temperature": [0.0, 0.2, 0.4],
    "condition_on_previous_text": False,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "word_timestamps": True,
    "vad_filter": True,
}


def rss_cap_mb() -> float:
    """How much RSS a decode child may use on THIS host, right now."""
    limit = None
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
        if raw and raw != "max":
            limit = int(raw) / MB
    except OSError:
        pass
    cap = (limit - HEADROOM_MB) if limit else 1200.0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                cap = min(cap, int(line.split()[1]) / 1024.0 - 150.0)
    except (OSError, ValueError):
        pass
    return max(400.0, cap)


def fits(model: str, cap_mb: float | None = None) -> bool:
    cap = rss_cap_mb() if cap_mb is None else cap_mb
    need = MEASURED_PEAK_MB.get(str(model).lower())
    if need is None:
        from app import membudget

        need = membudget.asr_requirement_mb(model, 30.0)
    return need <= cap


def peak_for(model: str) -> float:
    need = MEASURED_PEAK_MB.get(str(model).lower())
    if need is not None:
        return need
    from app import membudget

    return membudget.asr_requirement_mb(model, 30.0)


class DecodeError(RuntimeError):
    def __init__(self, message: str, *, memory: bool = False) -> None:
        super().__init__(message)
        self.memory = memory


@dataclass
class Window:
    """One decoded window, before alignment."""

    start: float
    end: float
    words: list[dict] = field(default_factory=list)
    language: str | None = None
    segments: list[dict] = field(default_factory=list)
    model: str = ""
    peak_rss_mb: float = 0.0
    settings: dict[str, Any] = field(default_factory=dict)


def _rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


class DecodeProcess:
    """A model-resident child with an RSS ceiling and a hard kill."""

    def __init__(
        self,
        model: str,
        *,
        cap_mb: float | None = None,
        compute: str = "int8",
        threads: int = 1,
        log_path: Path | None = None,
    ) -> None:
        self.model = model
        self.cap_mb = rss_cap_mb() if cap_mb is None else float(cap_mb)
        cfg = {"model": model, "compute": compute, "threads": max(1, int(threads))}
        self._stderr = open(log_path, "ab") if log_path else subprocess.DEVNULL
        env = dict(os.environ)
        env.update({
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        })
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "app.asr_quality", "child", json.dumps(cfg)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1, env=env,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        self._killed = False
        self._peak = 0.0
        threading.Thread(target=self._watch, daemon=True).start()
        ready = self._read(timeout=300.0)
        if not ready or not ready.get("ok"):
            err = (ready or {}).get("error") or "decode process did not start"
            self.kill()
            raise DecodeError(f"model {model} could not be loaded here: {err}", memory=True)

    def _watch(self) -> None:
        pid = self.proc.pid
        while self.proc.poll() is None:
            mb = _rss_mb(pid)
            self._peak = max(self._peak, mb)
            if mb > self.cap_mb:
                self._killed = True
                self.kill()
                return
            time.sleep(0.05)

    def _read(self, timeout: float) -> dict | None:
        import select

        assert self.proc.stdout is not None
        end = time.time() + timeout
        while time.time() < end:
            ready, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if not ready:
                if self.proc.poll() is not None:
                    return None
                continue
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except ValueError:
                continue
        return None

    def decode(self, wav: Path, **settings: Any) -> dict:
        if self.proc.poll() is not None:
            raise DecodeError(
                f"decode process exited (code {self.proc.returncode})"
                + (" — killed at the RSS ceiling" if self._killed else ""),
                memory=True,
            )
        payload = {"op": "decode", "wav": str(wav), **settings}
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DecodeError(f"decode pipe closed: {exc}", memory=True) from exc
        reply = self._read(timeout=3600.0)
        if reply is None:
            raise DecodeError(
                f"decode process died mid-window (exit {self.proc.poll()})"
                + (" — RSS ceiling" if self._killed else ""),
                memory=True,
            )
        if not reply.get("ok"):
            raise DecodeError(str(reply.get("error") or "decode failed"),
                              memory=bool(reply.get("memory")))
        reply["peak_rss_mb"] = max(float(reply.get("peak_rss_mb") or 0.0), self._peak)
        return reply

    def kill(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        try:
            if self.proc.poll() is None and self.proc.stdin is not None:
                self.proc.stdin.write('{"op": "quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.kill()
        finally:
            if self._stderr not in (subprocess.DEVNULL,) and hasattr(self._stderr, "close"):
                try:
                    self._stderr.close()
                except Exception:  # noqa: BLE001
                    pass

    def __enter__(self) -> "DecodeProcess":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# child process
# --------------------------------------------------------------------------- #

def _child(argv: list[str]) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    # `python -m app.asr_quality child {json}`: argv is [module, "child", json]
    rest = [a for a in argv[1:] if a != "child"]
    cfg = json.loads(rest[0]) if rest else {}
    model_name = str(cfg.get("model") or "small")
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            model_name, device="cpu", compute_type=str(cfg.get("compute") or "int8"),
            cpu_threads=max(1, int(cfg.get("threads") or 1)), num_workers=1,
        )
    except Exception as exc:  # noqa: BLE001 - the parent decides what to do
        print(json.dumps({"ok": False, "memory": True,
                          "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 3

    from app.speech_vad import read_wav16k_mono

    print(json.dumps({"ok": True, "ready": True, "model": model_name, "pid": os.getpid()}),
          flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if req.get("op") == "quit":
            break
        try:
            audio = read_wav16k_mono(Path(str(req["wav"])))
            kwargs = dict(DECODE_DEFAULTS)
            for key in ("beam_size", "temperature", "condition_on_previous_text",
                        "compression_ratio_threshold", "log_prob_threshold",
                        "no_speech_threshold", "word_timestamps"):
                if key in req:
                    kwargs[key] = req[key]
            kwargs["task"] = "transcribe"
            kwargs["language"] = req.get("language")
            prompt = req.get("initial_prompt")
            if prompt:
                kwargs["initial_prompt"] = str(prompt)
            if kwargs.get("vad_filter"):
                kwargs["vad_parameters"] = {"min_silence_duration_ms": 400, "speech_pad_ms": 60}
            segments, info = model.transcribe(audio, **kwargs)
            out_segments: list[dict] = []
            words: list[dict] = []
            for seg in segments:
                seg_words = []
                for w in seg.words or []:
                    text = (w.word or "").strip()
                    if not text:
                        continue
                    item = {
                        "word": text,
                        "start": float(w.start),
                        "end": float(w.end),
                        "prob": round(float(getattr(w, "probability", 0.0) or 0.0), 3),
                    }
                    seg_words.append(item)
                    words.append(item)
                out_segments.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": (seg.text or "").strip(),
                    "avg_logprob": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                    "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                    "compression_ratio": float(getattr(seg, "compression_ratio", 0.0) or 0.0),
                    "temperature": float(getattr(seg, "temperature", 0.0) or 0.0),
                    "words": seg_words,
                })
            _trim()  # hand the window's activations back before the next one
            print(json.dumps({
                "ok": True,
                "words": words,
                "segments": out_segments,
                "language": getattr(info, "language", None),
                "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
                "duration": float(getattr(info, "duration", 0.0) or 0.0),
                "peak_rss_mb": round(_own_peak_mb(), 1),
            }), flush=True)
        except (MemoryError, OSError, RuntimeError) as exc:
            text = f"{type(exc).__name__}: {exc}"
            memory = any(t in text.lower() for t in ("memory", "mkl", "alloc", "oom"))
            print(json.dumps({"ok": False, "memory": memory, "error": text}), flush=True)
            if memory:
                return 3
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "memory": False,
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 0


def _trim() -> None:
    """Release the allocator's free arenas between windows.

    The model stays resident (loading it per window would cost more than the
    decode), but the per-window activations must not accumulate: on a 2 GiB
    cgroup the difference is the next window fitting or being OOM-killed.
    """
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - best effort, per-allocator
        pass


def _own_peak_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


if __name__ == "__main__":
    raise SystemExit(_child(sys.argv))
