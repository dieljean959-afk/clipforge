"""Real-time-factor (RTF) statistics and live ETA estimation.

RTF = wall-seconds / audio-seconds. Lower is faster: WhisperX on a GPU lands
around 0.02-0.05, batched faster-whisper on CPU int8 around 0.1-0.5 for the
`medium` model. The prompt's "video duration x historical average RTF" prior is
exactly `audio_seconds * rtf`.

Two ETA sources, in order of trust:

  1. **measured progress** — elapsed / fraction * (1 - fraction). Accurate once
     enough of the stage has run.
  2. **the RTF prior** — before progress accumulates. The RTF comes from what
     THIS host has actually measured for the active (engine, model, device),
     persisted in ``data/asr_stats.json`` so it improves across runs and
     survives restarts. Only the very first run on a brand-new host falls back
     to the documented table below.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

# Documented priors — used ONLY until this host has measured its own RTF.
# Keyed by (engine, device); scaled by model size. Conservative on purpose:
# an over-optimistic ETA is worse than a slow one.
_PRIOR_BASE = {
    ("whisperx", "cuda"): 0.020,
    ("whisperx", "cpu"): 0.110,
    ("faster-whisper", "cuda"): 0.050,
    ("faster-whisper", "cpu"): 0.300,
}
_MODEL_FACTOR = {
    "tiny": 0.20,
    "base": 0.30,
    "small": 0.55,
    "medium": 1.0,
    "large-v3": 2.6,
    "large-v2": 2.6,
}
# The `medium` CPU prior above already assumes int8; float16 CPU is slower.
_COMPUTE_FACTOR = {"int8": 1.0, "int8_float16": 1.0, "float16": 1.6, "float32": 2.4}

# EMA weight for a new observation. 0.35 reacts within a few chunks without
# letting one anomalously slow chunk dominate the estimate.
_ALPHA = 0.35


def rtf_key(engine: str, device: str, model: str) -> str:
    return f"{engine}|{device}|{model}"


def prior_rtf(engine: str, device: str, model: str, compute_type: str = "") -> float:
    base = _PRIOR_BASE.get((engine, device))
    if base is None:
        base = _PRIOR_BASE[("faster-whisper", "cpu")]
    value = base * _MODEL_FACTOR.get(model, 1.0) * _COMPUTE_FACTOR.get(compute_type, 1.0)
    return round(max(0.005, value), 4)


class RtfStore:
    """Rolling per-host RTF observations, persisted as JSON."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / "asr_stats.json"
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._read()

    def _read(self) -> dict[str, Any]:
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded
        except Exception as exc:
            logger.warning(f"[eta] unreadable RTF stats ({exc}) — starting fresh")
        return {}

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as exc:
            logger.warning(f"[eta] could not persist RTF stats: {exc}")

    def observe(
        self,
        key: str,
        rtf: float,
        *,
        audio_seconds: float = 0.0,
        engine: str = "",
        device: str = "",
        model: str = "",
    ) -> None:
        """Record one measured RTF (ignoring nonsense values)."""
        if not (0.001 <= rtf <= 10.0):
            return
        with self._lock:
            entry = self._data.get(key) or {}
            samples = int(entry.get("samples", 0))
            prev = entry.get("rtf")
            new = rtf if prev is None else (1 - _ALPHA) * float(prev) + _ALPHA * rtf
            self._data[key] = {
                "rtf": round(new, 4),
                "lastRtf": round(rtf, 4),
                "samples": samples + 1,
                "engine": engine or entry.get("engine", ""),
                "device": device or entry.get("device", ""),
                "model": model or entry.get("model", ""),
                "lastAudioSeconds": round(audio_seconds, 1),
                "updatedAt": int(time.time() * 1000),
            }
            self._write()
            logger.info(
                f"[eta] RTF observed {rtf:.3f} ({audio_seconds:.0f}s audio) — "
                f"host average for {key} is now {new:.3f}"
            )

    def get(self, key: str) -> float | None:
        entry = self._data.get(key) or {}
        value = entry.get("rtf")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def effective_rtf(self, engine: str, device: str, model: str, compute_type: str = "") -> tuple[float, str]:
        """(rtf, source) — measured when we have it, documented prior otherwise."""
        key = rtf_key(engine, device, model)
        measured = self.get(key)
        if measured:
            return measured, f"measured on this host ({int((self._data.get(key) or {}).get('samples', 0))} run(s))"
        return prior_rtf(engine, device, model, compute_type), "documented prior for this engine/device"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {k: dict(v) for k, v in self._data.items() if isinstance(v, dict)}


_STORE: RtfStore | None = None
_STORE_LOCK = threading.Lock()


def store(data_dir: Path) -> RtfStore:
    """Process-wide RTF store for a given data dir."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None or _STORE.path.parent != Path(data_dir):
            _STORE = RtfStore(Path(data_dir))
        return _STORE


# --------------------------------------------------------------------------- #
# ETA maths
# --------------------------------------------------------------------------- #

def eta_from_progress(elapsed_sec: float, fraction: float) -> float | None:
    """Linear extrapolation: elapsed / done * remaining. None until meaningful."""
    if fraction < 0.05 or elapsed_sec <= 0.5:
        return None
    if fraction >= 0.999:
        return 0.0
    projected_total = elapsed_sec / fraction
    return max(0.0, projected_total - elapsed_sec)


def eta_from_prior(audio_seconds: float, rtf: float) -> float:
    return max(0.0, float(audio_seconds) * float(rtf))


def estimate(
    *,
    elapsed_sec: float,
    fraction: float,
    prior_remaining_sec: float | None = None,
) -> tuple[float | None, str]:
    """Best available remaining-time estimate → (seconds, how it was derived)."""
    measured = eta_from_progress(elapsed_sec, fraction)
    if measured is not None:
        return measured, "measured"
    if prior_remaining_sec is not None and prior_remaining_sec > 0:
        return prior_remaining_sec, "rtf-prior"
    return None, "unknown"


def tail_estimate_sec(clip_count: int | None = None, clip_seconds: float = 0.0) -> float:
    """Rough cost of the stages AFTER transcription (detection + renders).

    Deliberately coarse and labelled as an estimate in the UI: one Gemini call
    plus one ffmpeg render per clip. Encoding dominates the render, so it scales
    with the total clip runtime, not the source length.
    """
    clips = max(1, int(clip_count or 8))
    return 20.0 + clips * 12.0 + clip_seconds * 0.6
