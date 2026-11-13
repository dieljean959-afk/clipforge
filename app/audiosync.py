"""Independent audio/video sync gate for a rendered clip.

The renderer is not allowed to grade its own homework. This module asks two
questions about the FINAL file using the REAL source:

  1. Placement: does the clip's audio land at the planned source second?
  2. Duplicate: does this clip repeat another clip in the same batch?
  3. Speech: do 10 s of the rendered audio and the burned-in caption words
     describe the same speech?

Only local media/ffmpeg and the configured local Whisper are used. No signed
URL or key is ever written into a result (or logged), and no hosted ASR is
contacted.
"""

from __future__ import annotations

import math
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from app.config import Settings, ffmpeg_bin

SR = 16_000
ENVELOPE_HOP = 160  # 100 frames/s
PLACEMENT_TOLERANCE_MS = 40
PLACEMENT_MIN_CORR = 0.55
AUDIO_DUP_MIN = 0.985
VIDEO_DUP_MAX_DIFF = 6.0


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def _decode_audio(
    path_or_url: Path | str,
    *,
    start: float = 0.0,
    duration: float | None = None,
    sr: int = SR,
) -> np.ndarray:
    """Decode a range to mono float32 PCM, through ffmpeg ranged reads."""
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{max(0.0, float(start)):.3f}",
    ]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{float(duration):.3f}"]
    cmd += [
        "-i", str(path_or_url),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        reason = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(reason[-1] if reason else "ffmpeg could not decode audio")
    if not proc.stdout:
        raise RuntimeError("ffmpeg returned no audio samples")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def _envelope(samples: np.ndarray, *, hop: int = ENVELOPE_HOP) -> np.ndarray:
    """Short-time RMS envelope at 100 Hz, zero-padded on the final frame."""
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    n = int(math.ceil(samples.size / hop))
    padded = np.pad(samples, (0, n * hop - samples.size))
    frames = padded.reshape(n, hop)
    env = np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float64)).astype(np.float32)
    # Remove the DC-ish floor so silence cannot dominate normalisation.
    env -= float(np.median(env))
    np.maximum(env, 0.0, out=env)
    return env


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #
def placement(
    clip_path: Path,
    source_audio: Path | str,
    *,
    plan_start: float,
    source_start_time: float = 0.0,
    max_clip_sec: float | None = None,
) -> dict[str, Any]:
    """Cross-correlate the rendered audio with the REAL source around the plan.

    Returns the best lag in milliseconds (positive = audio content is later in
    the source than planned), the normalised correlation at that lag, and
    whether the peak is inside the ±40 ms gate.
    """
    plan_start = max(0.0, float(plan_start))
    source_start_time = max(0.0, float(source_start_time or 0.0))
    clip_duration = float(max_clip_sec or 0.0)
    if clip_duration <= 0:
        clip_duration = 38.0
    window_start = max(0.0, plan_start - 2.0)
    # The rendered clip is at most this long; the source window carries two
    # seconds of lead-in and two of tail so a small offset is still in range.
    source_duration = clip_duration + 4.0
    source_seek = max(0.0, window_start - source_start_time)
    try:
        source = _decode_audio(
            source_audio, start=source_seek, duration=source_duration
        )
        clip = _decode_audio(clip_path)
    except Exception as exc:  # noqa: BLE001 - the gate result names the reason
        return {
            "ok": False,
            "reason": f"could not decode audio for placement: {exc}"[:500],
            "lagMs": None,
            "correlation": None,
            "toleranceMs": PLACEMENT_TOLERANCE_MS,
        }
    if clip.size < SR // 2 or source.size < SR // 2:
        return {
            "ok": False,
            "reason": "clip or source audio has less than 0.5 s of samples",
            "lagMs": None,
            "correlation": None,
            "toleranceMs": PLACEMENT_TOLERANCE_MS,
        }

    a = _envelope(source)
    b = _envelope(clip)
    if a.size <= b.size:
        return {
            "ok": False,
            "reason": "source audio window is shorter than the clip",
            "lagMs": None,
            "correlation": None,
            "toleranceMs": PLACEMENT_TOLERANCE_MS,
        }
    nfft = 1 << int(a.size + b.size - 1).bit_length()
    corr = np.fft.irfft(
        np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft
    )[: a.size - b.size + 1]
    best = int(np.argmax(corr))
    segment = a[best : best + b.size]
    denom = float(np.linalg.norm(segment) * np.linalg.norm(b))
    corr_value = float(corr[best] / denom) if denom > 1e-9 else 0.0
    expected = int(round((plan_start - window_start) * (SR / ENVELOPE_HOP)))
    lag_ms = round((best - expected) * (ENVELOPE_HOP / SR) * 1000.0, 1)
    ok = abs(lag_ms) <= PLACEMENT_TOLERANCE_MS and corr_value >= PLACEMENT_MIN_CORR
    return {
        "ok": bool(ok),
        "lagMs": lag_ms,
        "correlation": round(corr_value, 4),
        "toleranceMs": PLACEMENT_TOLERANCE_MS,
        "threshold": PLACEMENT_MIN_CORR,
        "sourceWindowStartSec": round(window_start, 3),
        "clipSeconds": round(b.size * ENVELOPE_HOP / SR, 3),
        "reason": (
            "audio placement is inside the sync gate"
            if ok
            else (
                f"audio lag {lag_ms:.0f} ms is outside ±{PLACEMENT_TOLERANCE_MS} ms"
                if abs(lag_ms) > PLACEMENT_TOLERANCE_MS
                else f"correlation {corr_value:.3f} is below {PLACEMENT_MIN_CORR}"
            )
        ),
    }


# --------------------------------------------------------------------------- #
# fingerprints / duplicate detection
# --------------------------------------------------------------------------- #
def _read_video_signature(clip_path: Path) -> list[float] | None:
    """16x16 grey average of one frame per second; compact duplicate signal."""
    try:
        proc = subprocess.run(
            [
                ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", str(clip_path),
                "-vf", "fps=1,scale=16:16,format=gray",
                "-f", "rawvideo", "-pix_fmt", "gray", "-",
            ],
            capture_output=True,
            timeout=120,
        )
        if proc.returncode != 0 or len(proc.stdout) < 256:
            return None
        frames = np.frombuffer(proc.stdout, dtype=np.uint8)
        usable = (frames.size // 256) * 256
        if usable < 256:
            return None
        sig = frames[:usable].reshape(-1, 256).mean(axis=0)
        return [round(float(v), 2) for v in sig]
    except Exception:  # noqa: BLE001 - a missing signal is not a false duplicate
        return None


def _read_audio_signature(clip_path: Path) -> list[float] | None:
    """Normalised 100 Hz envelope over the first 20 s, compact audio signal."""
    try:
        samples = _decode_audio(clip_path, duration=20.0)
        env = _envelope(samples)
        if env.size < 8:
            return None
        # 128 points is enough to identify the recording and small enough to
        # keep the clip document readable.
        idx = np.linspace(0, env.size - 1, 128).astype(int)
        sig = env[idx].astype(float)
        norm = float(np.linalg.norm(sig))
        if norm <= 1e-9:
            return None
        return [round(float(v / norm), 5) for v in sig]
    except Exception:  # noqa: BLE001
        return None


def fingerprints(clip_path: Path) -> dict[str, Any]:
    return {
        "video": _read_video_signature(clip_path),
        "audio": _read_audio_signature(clip_path),
    }


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


def _video_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 255.0
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def duplicate_of(
    fingerprint: dict[str, Any], previous: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return the first earlier clip whose frames AND audio both match."""
    mine_v, mine_a = fingerprint.get("video"), fingerprint.get("audio")
    if not mine_v or not mine_a:
        return {"ok": True, "duplicateOf": None, "reason": "fingerprint unavailable"}
    for prior in previous:
        other = prior.get("fingerprint") or {}
        other_v, other_a = other.get("video"), other.get("audio")
        if not other_v or not other_a:
            continue
        vdiff = _video_diff(mine_v, other_v)
        acorr = _cosine(mine_a, other_a)
        if vdiff <= VIDEO_DUP_MAX_DIFF and acorr >= AUDIO_DUP_MIN:
            previously = prior.get("clipId")
            return {
                "ok": False,
                "duplicateOf": previously,
                "videoDiff": round(vdiff, 2),
                "audioSimilarity": round(acorr, 4),
                "reason": f"duplicate of clip {previously} (frames + audio match)",
            }
    return {"ok": True, "duplicateOf": None, "reason": "no duplicate in the batch"}


# --------------------------------------------------------------------------- #
# speech / captions
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _caption_words(cues: list[dict], start: float, end: float) -> list[str]:
    out: list[str] = []
    for cue in cues or []:
        try:
            c0 = float(cue.get("start") or 0.0)
            c1 = float(cue.get("end") or c0)
        except (TypeError, ValueError):
            continue
        if c1 < start or c0 > end:
            continue
        words = cue.get("words") or []
        if words:
            for w in words:
                try:
                    w0 = float(w.get("start") or c0)
                except (TypeError, ValueError):
                    w0 = c0
                if start <= w0 <= end:
                    out.extend(_tokens(str(w.get("word") or w.get("text") or "")))
        else:
            out.extend(_tokens(str(cue.get("text") or "")))
    return out


def _speech_check(
    clip_path: Path,
    cues: list[dict],
    settings: Settings,
    *,
    duration: float,
    cache_dir: Path,
) -> dict[str, Any]:
    """Transcribe 10 s locally and compare the words with the clip's captions."""
    ref_all = _caption_words(cues, 0.0, duration)
    if not ref_all:
        return {"ok": None, "reason": "clip has no caption words to compare"}
    # Put the 10 s window on the first caption that has words, but leave room
    # for the end of the clip.
    cue_start = 0.0
    for cue in cues or []:
        if _caption_words([cue], 0.0, duration):
            try:
                cue_start = max(0.0, float(cue.get("start") or 0.0))
            except (TypeError, ValueError):
                cue_start = 0.0
            break
    win_start = min(max(0.0, cue_start - 0.2), max(0.0, duration - 10.0))
    win_end = min(duration, win_start + 10.0)
    ref = _caption_words(cues, win_start, win_end)
    if not ref:
        return {"ok": None, "reason": "no caption words in the 10 s speech window"}
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav = cache_dir / f"speech-{time.time_ns()}.wav"
    try:
        proc = subprocess.run(
            [
                ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", f"{win_start:.3f}", "-t", f"{max(0.5, win_end - win_start):.3f}",
                "-i", str(clip_path),
                "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
                "-f", "wav", str(wav),
            ],
            capture_output=True,
            timeout=120,
        )
        if proc.returncode != 0 or not wav.exists():
            raise RuntimeError(
                (proc.stderr or b"").decode("utf-8", "replace").strip()[-300:]
                or "ffmpeg could not extract the speech window"
            )
        from app import subtitle

        transcript = subtitle._transcribe_legacy(
            wav, settings, cache_dir=cache_dir / "asr",
        )
        # The WAV is already the clip-relative 10 s window, so its words
        # start at 0. Filtering them by the SOURCE window start dropped every
        # word except the last one and made a correct clip look mismatched.
        hyp = []
        for word in transcript.words:
            if 0.0 <= float(word.start) <= (win_end - win_start) + 0.25:
                hyp.extend(_tokens(word.word))
        from app import asr_accurate

        error = asr_accurate.wer(" ".join(ref), " ".join(hyp))
        overlap = (
            sum((Counter(ref) & Counter(hyp)).values()) / max(1, len(ref))
            if ref else 0.0
        )
        ok = overlap >= 0.25 or (error is not None and error <= 0.65)
        return {
            "ok": bool(ok),
            "windowStartSec": round(win_start, 3),
            "windowEndSec": round(win_end, 3),
            "referenceWords": len(ref),
            "heardWords": len(hyp),
            "overlap": round(float(overlap), 3),
            "wer": round(float(error), 3) if error is not None else None,
            "heard": " ".join(hyp[:40]),
            "reason": (
                "spoken words match the burned-in captions"
                if ok
                else "spoken words do not match the burned-in captions"
            ),
        }
    except Exception as exc:  # noqa: BLE001 - an unavailable model is not a mismatch
        return {
            "ok": None,
            "error": f"local speech check unavailable: {exc}"[:500],
            "reason": "speech check could not run; placement and duplicate checks still apply",
        }
    finally:
        wav.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# one entry point
# --------------------------------------------------------------------------- #
def gate_clip(
    clip_path: Path,
    source_audio: Path | str,
    *,
    plan_start: float,
    plan_end: float,
    cues: list[dict],
    settings: Settings,
    cache_dir: Path,
    previous: list[dict[str, Any]] | None = None,
    source_start_time: float = 0.0,
) -> dict[str, Any]:
    """Run every independent check on one finished clip.

    `previous` is the batch's already-ready [{clipId, fingerprint}] list. The
    returned dict is safe to store: it contains numbers and reasons, never a
    presigned URL, key or credential.
    """
    duration = max(0.1, float(plan_end) - float(plan_start))
    place = placement(
        clip_path,
        source_audio,
        plan_start=plan_start,
        source_start_time=source_start_time,
        max_clip_sec=duration,
    )
    fp = fingerprints(clip_path)
    dup = duplicate_of(fp, previous or [])
    # Do not pay for a local model load when the cheap numeric check already
    # failed; the clip is already rejected and will be re-rendered/replaced.
    speech = (
        _speech_check(
            clip_path, cues, settings, duration=duration,
            cache_dir=cache_dir,
        )
        if place.get("ok") and dup.get("ok")
        else {"ok": None, "reason": "not run: placement or duplicate check failed"}
    )
    reasons: list[str] = []
    if not place.get("ok"):
        reasons.append(str(place.get("reason") or "audio placement failed"))
    if not dup.get("ok"):
        reasons.append(str(dup.get("reason") or "duplicate clip"))
    if speech.get("ok") is False:
        reasons.append(str(speech.get("reason") or "speech mismatch"))
    result = {
        "passed": not reasons,
        "reason": "; ".join(reasons)[:500] if reasons else "audio, speech and duplicate checks passed",
        "placement": place,
        "speech": speech,
        "duplicate": dup,
        "fingerprint": fp,
        "checkedAt": int(time.time() * 1000),
    }
    if not result["passed"]:
        logger.warning(f"[audiosync] {clip_path.name}: {result['reason']}")
    return result
