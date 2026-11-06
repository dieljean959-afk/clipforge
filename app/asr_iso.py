"""Transcribe a span through the LADDER, each attempt in its own process.

Order (each step strictly lighter than the one before, and every step logged
and recorded in the job metadata):

    1. configured model · 60 s window · beam 5 · VAD on
    2. configured model · 30 s window · beam 5 · VAD on
    3. configured model · 30 s window · beam 1 · 1 thread · int8
    4. configured model · 30 s window · beam 1 · VAD OFF
    5. smaller model (base) · 30 s window · beam 1 · VAD OFF
    6. smallest model (tiny) · 30 s window · beam 1 · VAD OFF
    7. the source's own captions, when it has word-timed ones

Nothing is silent: the model that produced the words is returned with the
words, and the job stores "transcribed with base (low memory)" or "using
YouTube captions" so the UI can say exactly what happened.

The budget is re-read before EVERY chunk (:mod:`app.membudget`, cgroup bytes →
MB). A step whose peak cannot fit is skipped instead of being attempted and
failing — that is the difference between a ladder and three identical retries.
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
from typing import Any, Callable

from loguru import logger

from app import membudget

MB = 1024 * 1024


@dataclass(frozen=True)
class Step:
    """One rung. `model=None` means "whatever the job configured"."""

    window: float
    beam: int
    vad: bool
    model: str | None
    threads: int = 1
    compute: str = "int8"

    def label(self, configured: str) -> str:
        model = self.model or configured
        bits = [f"{int(self.window)}s", f"beam {self.beam}", "vad off" if not self.vad else "vad on"]
        if model != configured:
            bits.append(f"model {model}")
        return f"{model} · " + " · ".join(bits)


@dataclass
class Attempt:
    step: Step
    model: str
    ok: bool
    error: str = ""
    memory: bool = False
    peak_rss_mb: float = 0.0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": getattr(self, "start", None),
            "end": getattr(self, "end", None),
            "window": self.step.window,
            "beam": self.step.beam,
            "vad": self.step.vad,
            "model": self.model,
            "ok": self.ok,
            "memory": self.memory,
            "error": self.error[:300],
            "peakRssMb": self.peak_rss_mb,
            "seconds": round(self.seconds, 1),
        }


@dataclass
class DecodedSpan:
    words: list[Any]
    model: str
    language: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    peak_rss_mb: float = 0.0

    @property
    def downshifted(self) -> bool:
        return any(a.ok and a.model != self.attempts[0].model for a in self.attempts) if self.attempts else False


_MODEL_ORDER = ("large-v3", "large", "medium", "small", "base", "tiny")


def smaller_models(configured: str) -> list[str]:
    """Models strictly smaller than the configured one, largest first."""
    name = str(configured or "small").lower()
    try:
        start = _MODEL_ORDER.index(name)
    except ValueError:
        start = _MODEL_ORDER.index("small")
    return [m for m in _MODEL_ORDER[start + 1:]]


def ladder_steps(configured: str, preferred_window: float = 60.0) -> list[Step]:
    """The ordered ladder for a job. Order is part of the contract (tested)."""
    steps = [
        Step(window=min(60.0, preferred_window), beam=5, vad=True, model=None),
        Step(window=min(30.0, preferred_window), beam=5, vad=True, model=None),
        Step(window=min(30.0, preferred_window), beam=1, vad=True, model=None),
        Step(window=min(30.0, preferred_window), beam=1, vad=False, model=None),
    ]
    for name in smaller_models(configured)[:2]:
        steps.append(Step(window=30.0, beam=1, vad=False, model=name))
    return steps


class AsrProcess:
    """A persistent, memory-limited ASR child. One model per process."""

    def __init__(self, model: str, *, compute: str = "int8", threads: int = 1,
                 limit_mb: float | None = None, log_path: Path | None = None) -> None:
        cfg = {
            "model": model,
            "compute": compute,
            "threads": max(1, int(threads)),
            "limit_mb": round(float(limit_mb), 1) if limit_mb else None,
        }
        self.model = model
        self.cfg = cfg
        self._stderr = open(log_path, "ab") if log_path else subprocess.DEVNULL
        env = dict(os.environ)
        env.update({
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        })
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "app.asr_child", json.dumps(cfg)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1, env=env, cwd=str(Path(__file__).resolve().parent.parent),
        )
        ready = self._read_line(timeout=180.0)
        if not ready or not ready.get("ok"):
            error = (ready or {}).get("error") or "ASR process did not start"
            self.kill()
            raise RuntimeError(error)

    # -- plumbing --------------------------------------------------------

    def _read_line(self, timeout: float = 600.0) -> dict | None:
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

    def decode(self, wav_path: Path, *, language: str | None, beam: int, vad: bool) -> dict:
        if self.proc.poll() is not None:
            return {"ok": False, "memory": True,
                    "error": f"ASR process exited with code {self.proc.returncode}"}
        payload = {
            "op": "decode", "wav": str(wav_path), "language": language,
            "beam_size": beam, "vad": vad,
        }
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            return {"ok": False, "memory": True, "error": f"ASR process pipe closed: {exc}"}
        reply = self._read_line()
        if reply is None:
            code = self.proc.poll()
            return {"ok": False, "memory": True,
                    "error": f"ASR process died while decoding (exit {code})"}
        return reply

    def close(self) -> None:
        try:
            if self.proc.poll() is None and self.proc.stdin is not None:
                self.proc.stdin.write('{"op": "quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - closing must never raise
            self.kill()
        finally:
            if isinstance(self._stderr, type(sys.stderr)) or hasattr(self._stderr, "close"):
                try:
                    if self._stderr not in (subprocess.DEVNULL,):
                        self._stderr.close()
                except Exception:
                    pass

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


_SHARED: dict[tuple, "AsrProcess"] = {}
_SHARED_LOCK = threading.Lock()


def shared_process(model: str, *, compute: str, threads: int, limit_mb: float | None,
                   log_path: Path | None) -> "AsrProcess":
    """One model instance per worker: reuse the child across chunks/jobs.

    Reloading a 900 MB model per chunk would cost more than the decode on a
    small host and would re-hit the allocator peak every 60 s. The child is
    keyed by its exact settings, so a ladder rung always gets a process with
    ITS limit; the child unloads itself after 5 idle minutes.
    """
    key = (model, compute, int(threads), round(float(limit_mb or 0))
           if limit_mb else 0)
    with _SHARED_LOCK:
        proc = _SHARED.get(key)
        if proc is not None and proc.proc.poll() is None:
            return proc
        if proc is not None:
            proc.kill()
        proc = AsrProcess(model, compute=compute, threads=threads,
                          limit_mb=limit_mb, log_path=log_path)
        _SHARED[key] = proc
        return proc


def drop_shared() -> None:
    """Forget the cached process — after a failure it may be dead or dirty."""
    with _SHARED_LOCK:
        for proc in _SHARED.values():
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        _SHARED.clear()


def shutdown_shared() -> None:
    """Close the worker's model process (called after every job)."""
    with _SHARED_LOCK:
        for proc in _SHARED.values():
            try:
                proc.close()
            except Exception:  # noqa: BLE001
                proc.kill()
        _SHARED.clear()


def trim_parent() -> None:
    """Give the freed bytes of the last chunk back to the allocator."""
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def decode_span(
    write_wav: Callable[[Path, float, float], None],
    start: float,
    end: float,
    *,
    configured_model: str,
    language: str | None,
    tmp_dir: Path,
    on_log: Callable[[str], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
    preferred_window: float = 60.0,
    budget_fn: Callable[..., membudget.Budget] = membudget.current_budget,
) -> DecodedSpan:
    """Decode [start, end) with the first ladder step that fits and works.

    `write_wav(path, a, b)` writes a 16 kHz mono WAV of the source between the
    two absolute timestamps — the parent never holds the whole source in RAM.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log = on_log or (lambda line: logger.info(f"[asr] {line}"))
    steps = ladder_steps(configured_model, preferred_window)
    attempts: list[Attempt] = []
    last_error = "no attempt could run"

    for index, step in enumerate(steps, start=1):
        model = step.model or configured_model
        need_mb = membudget.asr_requirement_mb(model, step.window)
        budget = budget_fn()
        if not budget.fits(need_mb):
            line = (
                f"attempt {index}/{len(steps)} skipped ({step.label(configured_model)}): "
                f"needs ~{need_mb:.0f} MB, {budget.describe()}"
            )
            log(line)
            trim_parent()
            continue
        line = f"attempt {index}/{len(steps)}: {step.label(configured_model)} — {budget.describe()}"
        log(line)
        if on_chunk:
            # Just the rung's model: the caller knows the chunk numbers and
            # composes the chip text ("chunk 3 of 9 (model small)").
            on_chunk(model)

        piece_paths: list[tuple[float, float, Path]] = []
        offset = start
        piece_no = 0
        while offset < end - 0.05:
            piece_end = min(offset + step.window, end)
            piece_no += 1
            wav = tmp_dir / f"span_{int(start)}_{int(end)}_a{index}_p{piece_no}.wav"
            write_wav(wav, offset, piece_end)
            piece_paths.append((offset, piece_end, wav))
            offset = piece_end

        proc: AsrProcess | None = None
        started = time.time()
        words: list[Any] = []
        detected: str | None = None
        peak = 0.0
        error = ""
        memory = False
        ok = True
        try:
            proc = shared_process(
                model, compute=step.compute, threads=step.threads,
                limit_mb=max(512.0, budget.usable_mb + 64.0),
                log_path=tmp_dir / "asr_child.log",
            )
            for piece_start, _piece_end, wav in piece_paths:
                reply = proc.decode(wav, language=language, beam=step.beam, vad=step.vad)
                if not reply.get("ok"):
                    ok = False
                    error = str(reply.get("error") or "decode failed")
                    memory = bool(reply.get("memory"))
                    break
                peak = max(peak, float(reply.get("peak_rss_mb") or 0.0))
                detected = detected or reply.get("language")
                for w in reply.get("words") or []:
                    words.append(_shift(w, piece_start))
                try:
                    wav.unlink()
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001 - startup/allocation failure
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            memory = True
        finally:
            if not ok:
                # A dead or dirty child must not be reused for the next rung.
                drop_shared()
                if proc is not None:
                    proc.kill()
            trim_parent()

        attempt = Attempt(step=step, model=model, ok=ok, error=error,
                          memory=memory, peak_rss_mb=peak, seconds=time.time() - started)
        attempt.start = round(start, 1)
        attempt.end = round(end, 1)
        attempts.append(attempt)
        if ok:
            if len(attempts) > 1:
                log(f"decoded {start:.0f}-{end:.0f}s with a lighter setting: {step.label(configured_model)}")
            return DecodedSpan(words=words, model=model, language=detected,
                               attempts=attempts, peak_rss_mb=peak)
        last_error = error
        log(f"attempt {index}/{len(steps)} FAILED ({'memory' if memory else 'parse'}): {error[:200]}")
        for _s, _e, wav in piece_paths:
            try:
                wav.unlink()
            except OSError:
                pass

    raise AsrLadderError(
        f"could not transcribe {start:.0f}-{end:.0f}s after {len(attempts)} lighter "
        f"attempts; last error: {last_error}",
        attempts=attempts,
    )


def _shift(word: dict, offset: float) -> Any:
    from app.asr import AsrWord

    return AsrWord(
        start=float(word.get("start", 0.0)) + offset,
        end=float(word.get("end", 0.0)) + offset,
        word=str(word.get("word") or ""),
        conf=float(word.get("conf") or 0.0),
    )


class AsrLadderError(RuntimeError):
    def __init__(self, message: str, attempts: list[Attempt] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


_MEMORY_TOKENS = (
    "mkl_malloc", "failed to allocate", "out of memory", "memoryerror",
    "std::bad_alloc", "bad_alloc", "cannot allocate", "oom",
    "failed to map segment", "not enough memory",
)


def is_memory_error(text: str) -> bool:
    """Does this failure text name memory as the cause?"""
    low = str(text or "").lower()
    return any(token in low for token in _MEMORY_TOKENS)


def plain_memory_reason(attempts_tried: int = 0, total_steps: int = 0) -> str:
    """The sentence the job shows instead of a traceback (details keep it)."""
    step = f" (attempt {attempts_tried} of {total_steps})" if attempts_tried else ""
    return (
        "Ran out of memory while transcribing. Retrying with a lighter "
        f"setting{step}. Your finished work is saved, so a retry resumes from "
        "the next unfinished chunk."
    )


def transcript_label(configured: str, used: str) -> str | None:
    """What the UI shows when the transcript is not the configured quality."""
    if not used or used == configured:
        return None
    return f"transcribed with {used} (low memory)"
