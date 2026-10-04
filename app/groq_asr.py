"""Groq's hosted Whisper, as a transcription engine.

Why this exists next to `app/asr.py`: the local engine is faster-whisper on
CPU, and on a 1-vCPU sandbox `medium` decodes at roughly 0.3x realtime — a
one-hour source spends over an hour and a half in transcription alone, long
enough that the sandbox is usually restarted mid-job. Groq's hosted
`whisper-large-v3` is both faster AND a different (larger) acoustic model.

Everything network-shaped is isolated in `_post_chunk`; the chunk planning and
the merge are pure functions so the interesting logic — timestamp offsets and
overlap de-duplication — is testable without a key or a network.

Three facts from Groq's docs shape this file:

  * free tier rejects an upload over **25 MB**, so chunks are sized to FIT,
    not merely to be resumable;
  * the Whisper `prompt` is capped at **224 tokens** (see `app.dialect`);
  * api.groq.com sits behind Cloudflare, which 403s the default
    `Python-urllib/3.x` User-Agent with error code 1010. `GROQ_USER_AGENT`
    already exists in `app.llm` for exactly this reason and is reused here —
    without it every call fails looking like a bad key.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from app import dialect as dialect_mod

# --------------------------------------------------------------------------- #
# Limits and policy. Named constants so the test suite asserts the real
# documented numbers rather than literals typed twice.
# --------------------------------------------------------------------------- #

#: Free tier. The dev tier allows 100 MB, but this project runs on free keys
#: and an upload that is rejected at 26 MB would fail the whole job.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: Deliberately under the 25 MB ceiling. A file that lands exactly on the
#: limit has been observed to be rejected, and there is no useful distinction
#: between a 24.9 MB chunk and a 21 MB one — only one extra request.
TARGET_UPLOAD_BYTES = 21 * 1024 * 1024

#: 10 minutes, per the design. Long enough that request overhead is noise,
#: short enough that one failed chunk costs 10 minutes, not an hour.
CHUNK_SEC = 600.0

#: Short overlap so a word spanning a boundary is heard whole by at least one
#: chunk. 3s is comfortably more than a Spanish word plus its trailing breath.
OVERLAP_SEC = 3.0

#: Seconds of silence at each end still get sent. Keeps a chunk's estimate
#: honest when the audio is sparse.
MIN_CHUNK_SEC = 20.0

#: Assumed FLAC bits-per-second for a 16 kHz mono speech track, used to pick a
#: chunk length that fits the upload ceiling. Measured on the sandbox's own
#: clips, FLAC of speech lands around 300-500 kbps; 450 is deliberately
#: pessimistic so the estimate errs toward smaller chunks.
FLAC_BITS_PER_SEC = 450_000

#: Exponential backoff. 429 and 5xx are retried; a 400 (a bad prompt, a bad
#: key shape) is NOT, because retrying a rejected request just spends money.
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SEC = 2.0
RETRY_BACKOFF_CAP_SEC = 30.0

#: Segments whose `no_speech_prob` exceeds this are dropped. Groq returns the
#: same metadata faster-whisper does, and without this filter the model
#: hallucinates over silence: a 700s tone with no speech came back as
#: "Subtitulado por Jnkoil" repeated across all three chunks. On real footage
#: that is a silent intro or a gap between speakers, and the hallucination
#: lands in the transcript and then in burned-in captions. The local engine
#: already guards this with no_speech_threshold=0.6; Groq needs the same gate.
NO_SPEECH_THRESHOLD = 0.6

#: How many back-to-back repetitions of the SAME text are kept. Whisper's
#: signature hallucination is a loop — a silent or near-silent span becomes
#: "Subtitulado por Jnkoil" over and over. Crucially, `no_speech_prob` does
#: NOT catch it: measured on a 700s pure tone, Groq returned the invented
#: segment with no_speech_prob=0.21 and avg_logprob=-0.70, i.e. it was
#: CONFIDENT it had heard speech. Repetition is the only reliable signal.
#: One copy is kept so a legitimate one-off line still appears; the rest go.
MAX_CONSECUTIVE_REPEATS = 1


class GroqAsrError(Exception):
    """Groq could not transcribe. Callers fall back to the local engine."""


class GroqHardError(GroqAsrError):
    """A failure retrying cannot fix (auth, bad request, unsupported file)."""


@dataclass(frozen=True)
class GroqConfig:
    """Everything the Groq path needs, resolved once."""

    model: str = "whisper-large-v3"
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_sec: int = 300
    language: str = "es"
    max_upload_bytes: int = TARGET_UPLOAD_BYTES
    chunk_sec: float = CHUNK_SEC
    overlap_sec: float = OVERLAP_SEC
    retries: int = RETRY_ATTEMPTS

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/audio/transcriptions"


def config_from_settings(settings: Any, model: str = "whisper-large-v3") -> GroqConfig:
    groq = getattr(settings, "groq", None)
    base = getattr(groq, "base_url", None) or "https://api.groq.com/openai/v1"
    # Groq's `base_url` already carries /openai/v1, and so does our endpoint
    # builder, so only the host part is meaningful here.
    return GroqConfig(
        model=model,
        base_url=base,
        timeout_sec=int(getattr(groq, "timeout_sec", 300) or 300),
        language=(getattr(settings.whisper, "language", "auto") or "auto"),
    )


# --------------------------------------------------------------------------- #
# Pure planning/merging — no network, no ffmpeg.
# --------------------------------------------------------------------------- #


def plan_chunks(
    duration: float,
    chunk_sec: float = CHUNK_SEC,
    overlap_sec: float = OVERLAP_SEC,
) -> list[tuple[float, float]]:
    """Windows covering [0, duration], each overlapping the previous one.

    The LAST window always ends at `duration`: trimming it to a whole chunk
    would silently drop the final seconds of the source, which for a clip
    pipeline means losing the ending hook.
    """
    if duration <= 0:
        return [(0.0, 0.0)]
    step = max(1.0, chunk_sec - overlap_sec)
    out: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + chunk_sec)
        out.append((start, end))
        if end >= duration:
            break
        start += step
    return out


def chunk_seconds_for_budget(
    duration: float,
    max_upload_bytes: int = TARGET_UPLOAD_BYTES,
    full_chunk_sec: float = CHUNK_SEC,
    overlap_sec: float = OVERLAP_SEC,
) -> tuple[float, float]:
    """The largest chunk length whose FLAC upload stays inside the ceiling.

    Returns ``(chunk_sec, overlap_sec)``. Groq rejects an oversize upload with
    an opaque 400, so this has to be conservative BEFORE the request rather
    than interpreted after it.
    """
    fits = max(1.0, (max_upload_bytes * 8) / FLAC_BITS_PER_SEC)
    chunk = min(full_chunk_sec, fits)
    # A short source needs no chunking at all — one request covers it.
    if duration > 0:
        chunk = min(chunk, duration)
    # The floor is a QUALITY floor, not a safety one. It must never push a
    # chunk past the hard upload ceiling: the first version applied
    # `max(chunk, MIN_CHUNK_SEC)` after the budget clamp, which turned a 1 MB
    # ceiling into an 18.6s-becomes-20s chunk that still did not fit.
    chunk = min(chunk, fits)
    # Keep the overlap sane relative to a shrunken chunk.
    return chunk, min(overlap_sec, chunk / 3.0)


@dataclass
class ChunkResult:
    """One chunk's decoded segments, still relative to the chunk."""

    start: float
    end: float
    segments: list[dict[str, Any]] = field(default_factory=list)
    words: list[dict[str, Any]] = field(default_factory=list)


def _norm(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def merge_chunks(results: list[ChunkResult]) -> list[dict[str, Any]]:
    """Stitch chunk results into one timeline and drop the overlap duplicates.

    The rule is exact rather than heuristic: chunk i is accepted only from the
    END of chunk i-1 onward, because chunk i-1 already decoded everything up
    to that point. Using chunk i's OWN START as the cut — which the first
    version did — is wrong by exactly the overlap, so every overlapped
    sentence was emitted twice.

    Nothing is lost (chunk i-1 covers the discarded span) and no text
    comparison is needed to guess whether two overlapping decodes said the
    same thing, which is exactly where such heuristics go wrong on
    code-switched speech, where the same phrase legitimately repeats.

    Returns dicts with ABSOLUTE start/end, ready to become `subtitle.Word`.
    """
    out: list[dict[str, Any]] = []
    consecutive = 0
    dropped_repeats = 0
    for idx, res in enumerate(results):
        cut = 0.0 if idx == 0 else results[idx - 1].end
        for seg in res.segments:
            if is_silence(seg):
                continue
            try:
                s = float(seg.get("start", 0.0)) + res.start
                e = float(seg.get("end", 0.0)) + res.start
            except (TypeError, ValueError):
                continue
            if e <= s:
                continue
            if s < cut - 1e-6:
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if out and _norm(out[-1]["text"]) == _norm(text) and abs(out[-1]["end"] - s) < 0.75:
                # Same sentence re-decoded across the seam: keep the earlier.
                continue
            if out and _norm(out[-1]["text"]) == _norm(text):
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_REPEATS:
                    # A Whisper repetition loop. The model's own confidence
                    # says this is speech (no_speech_prob ~0.21 on a silent
                    # tone), so the loop itself is the only evidence.
                    dropped_repeats += 1
                    continue
            else:
                consecutive = 0
            out.append({
                "start": round(s, 3),
                "end": round(e, 3),
                "text": text,
                "conf": segment_confidence(seg),
            })
    if dropped_repeats:
        logger.warning(
            f"[groq_asr] dropped {dropped_repeats} repeated segment(s) — "
            f"Whisper repetition loop over silence or noise"
        )
    return out


def is_silence(seg: dict[str, Any]) -> bool:
    """True when Groq itself says this segment is probably not speech.

    A segment with no `no_speech_prob` is KEPT — the field can be absent on
    older API responses, and dropping everything would produce an empty
    transcript from a working model, which is a far worse failure than a
    possible hallucination.
    """
    try:
        prob = float(seg.get("no_speech_prob"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return prob >= NO_SPEECH_THRESHOLD


def segment_confidence(seg: dict[str, Any]) -> float:
    """Groq reports `avg_logprob`; convert it the same way app.asr does.

    A transcript written before confidence existed must read as "never
    measured" rather than as "certain", so a missing value is NOT 1.0 here.
    """
    lp = seg.get("avg_logprob")
    if lp is None:
        return 0.0
    try:
        return max(0.0, min(1.0, math.exp(float(lp))))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def segments_to_words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split merged segments into timed words.

    Word timings are interpolated across the segment because Groq's word
    granularities are requested but not relied on: a chunk that returns only
    segments must still produce a word timeline, since the caption renderer
    works from words.
    """
    words: list[dict[str, Any]] = []
    for seg in segments:
        text = seg.get("text") or ""
        parts = text.split()
        if not parts:
            continue
        s, e = float(seg["start"]), float(seg["end"])
        span = max(0.001, e - s)
        step = span / len(parts)
        conf = float(seg.get("conf", 0.0) or 0.0)
        for i, p in enumerate(parts):
            words.append({
                "start": round(s + i * step, 3),
                "end": round(s + (i + 1) * step, 3),
                "word": p,
                "conf": round(conf, 3),
            })
    return words


# --------------------------------------------------------------------------- #
# Network + ffmpeg
# --------------------------------------------------------------------------- #


def to_flac(src: Path, dest: Path, start: float, end: float, ffmpeg: str = "ffmpeg") -> Path:
    """Mono 16 kHz FLAC for a time range — Groq's own recommended recipe.

    `-ss` before `-i` seeks by keyframe, which is what makes this fast; the
    cost is that the first keyframe's audio before `start` is included, which
    is harmless here because the overlap already covers the seam.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}",
        "-to", f"{max(0.0, end):.3f}",
        "-i", str(src),
        "-vn", "-map", "0:a:0",
        "-ar", "16000", "-ac", "1",
        "-c:a", "flac",
        # `-f flac` is mandatory, not decoration: the temp file ends in
        # ".part", so ffmpeg cannot infer the container from the extension and
        # exits 1 with "Unable to find a suitable output format". That killed
        # the very first chunk of the first live run.
        "-f", "flac",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    tmp.replace(dest)
    return dest


def _user_agent() -> str:
    # Imported lazily: app.llm is a heavy module and this one must stay
    # importable without it.
    try:
        from app.llm import GROQ_USER_AGENT

        return GROQ_USER_AGENT
    except Exception:  # noqa: BLE001
        return "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def _multipart(fields: dict[str, str], file_field: str, filename: str, blob: bytes) -> tuple[bytes, str]:
    b = "----" + uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    out += (
        f"--{b}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: audio/flac\r\n\r\n"
    ).encode()
    out += blob + b"\r\n" + f"--{b}--\r\n".encode()
    return out, f"multipart/form-data; boundary={b}"


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    try:
        v = exc.headers.get("Retry-After")
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


def _post_chunk(
    audio: Path,
    cfg: GroqConfig,
    api_key: str,
    prompt: str,
    language: str,
) -> dict[str, Any]:
    """Upload one chunk and return the decoded JSON, retrying transient faults.

    Retries 429 and 5xx with exponential backoff. A 400/401/403 raises
    `GroqHardError` immediately — those never fix themselves, and the caller
    wants to fall back to local rather than burn four attempts discovering it.
    """
    if not audio.exists() or audio.stat().st_size == 0:
        raise GroqAsrError(f"chunk audio missing or empty: {audio}")
    blob = audio.read_bytes()
    if len(blob) > cfg.max_upload_bytes:
        raise GroqHardError(
            f"chunk is {len(blob) / 1024 / 1024:.1f} MB, over the "
            f"{cfg.max_upload_bytes / 1024 / 1024:.0f} MB upload limit"
        )

    fields = {
        "model": cfg.model,
        "language": language or "es",
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
        "prompt": prompt,
        "temperature": "0",
    }
    body, content_type = _multipart(fields, "file", audio.name, blob)
    last: str = ""
    for attempt in range(cfg.retries):
        req = urllib.request.Request(cfg.endpoint, data=body, method="POST")  # noqa: S310
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", content_type)
        req.add_header("User-Agent", _user_agent())
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:  # noqa: S310
                raw = resp.read()
            return json.loads(raw or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode(errors="replace")
            last = f"HTTP {exc.code}: {detail}"
            if exc.code in {400, 401, 403, 404, 413, 422}:
                raise GroqHardError(f"Groq rejected the request — {last}") from exc
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise GroqHardError(last) from exc
            logger.warning(
                f"[groq_asr] {last} on attempt {attempt + 1}/{cfg.retries}"
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"[groq_asr] {last} on attempt {attempt + 1}/{cfg.retries}"
            )
        if attempt < cfg.retries - 1:
            delay = min(RETRY_BACKOFF_CAP_SEC, RETRY_BACKOFF_SEC * (2 ** attempt))
            time.sleep(delay)
    raise GroqAsrError(f"Groq unavailable after {cfg.retries} attempts — {last}")


def probe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False, timeout=60,
    )
    try:
        return float((out.stdout or "0").strip())
    except ValueError:
        return 0.0


def transcribe_source(
    src: Path,
    work_dir: Path,
    cfg: GroqConfig,
    api_key: str,
    dialect_id: str = dialect_mod.DEFAULT_DIALECT,
    glossary: str = "",
    on_progress=None,
    should_pause=None,
) -> tuple[list[dict[str, Any]], float]:
    """Transcribe a whole source with Groq. Returns (segments, duration).

    Raises `GroqAsrError` on any unrecoverable failure so the caller can fall
    back to local faster-whisper. NEVER returns partial results as if they
    were complete — a half transcript is worse than none, because it looks
    finished.
    """
    duration = probe_duration(src)
    chunk_sec, overlap = chunk_seconds_for_budget(duration, cfg.max_upload_bytes, cfg.chunk_sec, cfg.overlap_sec)
    windows = plan_chunks(duration, chunk_sec, overlap)
    prompt = dialect_mod.whisper_prompt(dialect_id, glossary)
    logger.info(
        f"[groq_asr] {duration:.1f}s -> {len(windows)} chunk(s) of "
        f"{chunk_sec:.0f}s (overlap {overlap:.1f}s, limit "
        f"{cfg.max_upload_bytes / 1024 / 1024:.0f} MB, model {cfg.model})"
    )

    results: list[ChunkResult] = []
    for idx, (start, end) in enumerate(windows):
        if should_pause is not None and should_pause():
            raise GroqAsrError("paused")
        if on_progress is not None:
            on_progress(idx / max(1, len(windows)), f"Groq chunk {idx + 1} of {len(windows)}")
        audio = to_flac(src, work_dir / f"chunk_{idx:04d}.flac", start, end)
        payload = _post_chunk(audio, cfg, api_key, prompt, cfg.language)
        segments = payload.get("segments") or []
        if not segments and not (payload.get("text") or "").strip():
            logger.warning(f"[groq_asr] chunk {idx} returned no speech")
        results.append(ChunkResult(start=start, end=end, segments=segments))
    return merge_chunks(results), duration


def status(settings: Any, api_key: str | None = None) -> dict[str, Any]:
    """A small dict for /api/health and the Doctor. Never includes the key."""
    try:
        from app.llm import PROVIDER_GROQ, openai_api_key

        key = api_key or openai_api_key(PROVIDER_GROQ, settings)
    except Exception:  # noqa: BLE001
        key = api_key
    cfg = config_from_settings(settings)
    return {
        "configured": bool(key),
        "keyTail": (key or "")[-4:] if key else None,
        "model": cfg.model,
        "endpoint": cfg.endpoint,
        "uploadLimitMb": round(cfg.max_upload_bytes / 1024 / 1024, 1),
        "chunkSec": cfg.chunk_sec,
    }

