"""Silero VAD: where the speech actually is, as a pure data plane.

Every timing decision in the accuracy pipeline needs one thing the transcript
cannot give it: the ground truth of WHEN somebody is talking. Whisper's own
`no_speech_prob` is the model's opinion about its own decode; this module asks
the question acoustically and independently.

The model is the Silero VAD that ships inside faster-whisper's wheel
(``faster_whisper/assets/silero_encoder_v5.onnx``), so there is no download, no
torch, and no second implementation to drift: ``faster_whisper.vad`` runs it
through onnxruntime at 16 kHz with the v5 state pair.

All functions take/return seconds. Nothing here mutates audio it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

#: Silero's own defaults, tuned for speech-in-video rather than dictation:
#: 30 ms frames, a 250 ms minimum speech run, and 400 ms of silence before a
#: region is allowed to end (a breath mid-sentence must not split a word).
MIN_SILENCE_MS = 400
MIN_SPEECH_MS = 250
SPEECH_PAD_MS = 60


@dataclass(frozen=True)
class Region:
    """One contiguous speech region, in absolute seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def read_wav16k_mono(path: Path) -> np.ndarray:
    """16 kHz mono 16-bit PCM WAV -> float32 in [-1, 1] (stdlib only)."""
    import wave

    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        raw = fh.readframes(fh.getnframes())
    if rate != 16000 or channels != 1 or width != 2:
        raise ValueError(f"{path} is {rate} Hz / {channels} ch / {width * 8} bit, expected 16k mono 16-bit")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def speech_regions(
    samples: np.ndarray,
    *,
    min_silence_ms: int = MIN_SILENCE_MS,
    min_speech_ms: int = MIN_SPEECH_MS,
    pad_ms: int = SPEECH_PAD_MS,
    limit: float | None = None,
) -> list[Region]:
    """Silero speech regions for a mono 16 kHz float signal.

    `limit` clamps every region end to the clip the caller is about to caption
    (a region that runs past the last frame would otherwise count as speech
    with no captions and fail the sync gate for a reason that is not a defect).
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(
        threshold=0.5,
        min_speech_duration_ms=int(min_speech_ms),
        max_speech_duration_s=float("inf"),
        min_silence_duration_ms=int(min_silence_ms),
        speech_pad_ms=int(pad_ms),
    )
    stamps = get_speech_timestamps(samples, opts, sampling_rate=16000)
    out: list[Region] = []
    for st in stamps:
        start = float(st["start"]) / 16000.0
        end = float(st["end"]) / 16000.0
        if end - start <= 0.02:
            continue
        if limit is not None:
            if start >= limit:
                continue
            end = min(end, limit)
        out.append(Region(start, end))
    return out


def speech_seconds(regions: Iterable[Region]) -> float:
    return sum(r.duration for r in regions)


def overlap_seconds(regions: list[Region], start: float, end: float) -> float:
    """How much of [start, end) is covered by any region (no double counting)."""
    if end <= start:
        return 0.0
    total = 0.0
    for r in regions:
        lo = max(start, r.start)
        hi = min(end, r.end)
        if hi > lo:
            total += hi - lo
    return total


def voicing_ratio(regions: list[Region], start: float, end: float) -> float:
    """Fraction of [start, end) that Silero calls speech (0..1)."""
    span = end - start
    if span <= 0:
        return 0.0
    return min(1.0, overlap_seconds(regions, start, end) / span)


def snap_to_speech(
    start: float,
    end: float,
    regions: list[Region],
    *,
    tolerance: float = 0.12,
) -> tuple[float, float]:
    """Move a boundary onto the nearest real speech edge within `tolerance`.

    This is the part of timing refinement that does NOT need a model: a word
    whose start sits in the silence before the syllable it belongs to is
    captioned early, and nothing downstream can see that. Snapping is bounded
    (120 ms) so it can only correct a boundary that is provably inside a
    silence gap — it can never move a word into the neighbouring one.
    """
    new_start, new_end = start, end
    if tolerance <= 0:
        return new_start, new_end
    best = tolerance + 1e-9
    for r in regions:
        for edge in (r.start, r.end):
            d = abs(edge - start)
            # Only snap across a silence: the edge must be on the side the word
            # is drifting from, and the span between must be silent.
            if d < best and (abs(start - edge) <= tolerance):
                if overlap_seconds(regions, min(edge, start), max(edge, start)) < 0.02:
                    best = d
                    new_start = edge
    best = tolerance + 1e-9
    for r in regions:
        for edge in (r.start, r.end):
            d = abs(edge - end)
            if d < best and (abs(end - edge) <= tolerance):
                if overlap_seconds(regions, min(edge, end), max(edge, end)) < 0.02:
                    best = d
                    new_end = edge
    if new_end <= new_start:
        return start, end
    return new_start, new_end
