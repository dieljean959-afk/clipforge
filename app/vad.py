"""Silence detection with Silero VAD (optional dependency).

If silero-vad / torch are not installed, silence trimming is skipped with a
readable log line — the pipeline still runs. No fake data is produced.
"""

from __future__ import annotations

from loguru import logger

_SILERO_AVAILABLE: bool | None = None


def _load_silero():
    global _SILERO_AVAILABLE
    if _SILERO_AVAILABLE is False:
        return None
    try:
        import torch  # noqa: F401
        from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore
        model = load_silero_vad()
        _SILERO_AVAILABLE = True
        return model, get_speech_timestamps
    except Exception as exc:
        logger.info(f"[vad] Silero VAD unavailable ({type(exc).__name__}) — silence trim disabled")
        _SILERO_AVAILABLE = False
        return None


def detect_speech_windows(wav_path, settings) -> list[dict]:
    """Return speech windows in seconds: [{start, end}]."""
    loaded = _load_silero()
    if loaded is None:
        return []
    model, get_speech_timestamps = loaded
    import wave

    import numpy as np

    with wave.open(str(wav_path), "rb") as wav:
        sr = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    ts = get_speech_timestamps(
        torch_tensor(audio), model, sampling_rate=sr,
        min_silence_duration_ms=350,
        speech_pad_ms=120,
    )
    return [{"start": t["start"] / sr, "end": t["end"] / sr} for t in ts]


def torch_tensor(audio_f32):
    import torch

    return torch.from_numpy(audio_f32)


def silence_cuts_for_clip(
    speech: list[dict],
    clip_start: float,
    clip_end: float,
    settings,
) -> list[dict]:
    """Windows INSIDE [clip_start, clip_end] where no speech occurs and the
    gap is long enough to trim."""
    if not speech:
        return []
    cuts: list[dict] = []
    prev_end = clip_start
    for w in speech:
        if w["end"] <= clip_start:
            continue
        if w["start"] >= clip_end:
            break
        gap_start = max(prev_end, clip_start)
        gap_end = min(w["start"], clip_end)
        if gap_end - gap_start >= 0.45:  # only trim meaningful dead air
            cuts.append({"start": gap_start - clip_start, "end": gap_end - clip_start})
        prev_end = max(prev_end, w["end"])
    tail_start = max(prev_end, clip_start)
    if clip_end - tail_start >= 0.45:
        cuts.append({"start": tail_start - clip_start, "end": clip_end - clip_start})
    return cuts
