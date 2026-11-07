"""Local, accuracy-first transcription. No hosted ASR.

The contract with the rest of the app does not change: this returns a
``subtitle.Transcript`` whose ``words`` are ``[start, end, word]`` and nothing
else downstream (caption builder, ASS writer, renderer, UI) learns that the
pipeline behind it is different.

Order of work (each step is skipped with a recorded reason if it cannot run):

    1a. creator-uploaded subtitles -> their TEXT, force-aligned to the audio
    1b. otherwise the best local Whisper that FITS this host (measured)
    2.  30 s windows, 2 s overlap, language detected once then pinned
    3.  BYTE-identical checkpoint per window, so a crash resumes at the next
    4.  second pass over low-confidence windows, disagreements marked "check"
    5.  wav2vec2 CTC forced alignment, then snapping to Silero speech edges
    6.  hallucination removal outside speech, words sorted and non-overlapping
    7.  a no-reference sync report + WER against whatever captions exist

Nothing here invents text: the only source of words is a human subtitle, the
acoustic model, or a re-decode of the same audio. Every discarded word is
counted and reported.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app import asr_quality, ctc_align, speech_vad, subs_source
from app.config import ffmpeg_bin, Settings
from app.subtitle import (
    OVERLAP_SEC as _SUB_OVERLAP,
    Word,
    Transcript,
    TranscribeError,
    _WavReader,
    _dedupe_words,
    _probe_duration,
)

#: 30 s is the Whisper encoder's own window: decoding exactly one encoder
#: window per request keeps the memory ceiling flat and avoids the
#: cross-window context bleed a longer span introduces.
WINDOW_SEC = 30.0
OVERLAP_SEC = 2.0

#: A window whose worst segment is below this is re-decoded with different
#: sampling settings (accuracy plan, step 6).
SECOND_PASS_LOGPROB = -1.0

#: Silero coverage below which a word is treated as a hallucination.
MIN_VOICING = 0.10

#: Sync gate thresholds (step 8).
MAX_START_ERROR = 0.150
MAX_NO_CAPTION = 0.20
MAX_CAPTION_ON_SILENCE = 0.10
MAX_WER = 0.35

PROMPT_MAX_CHARS = 700


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #

def prepare_audio(src: Path, dest: Path, *, force: bool = False) -> Path:
    """16 kHz mono, high-passed at 80 Hz and loudness-normalised.

    Music beds and game audio are what a raw decode turns into invented words;
    the high-pass removes rumble that the encoder reads as speech energy, and
    loudnorm makes a quiet talker as decodable as a loud one. Both are applied
    to the CACHED copy only — the source video is never touched.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    tmp = dest.with_suffix(".wav.part")
    subprocess.run(
        [
            ffmpeg_bin(), "-y", "-loglevel", "error",
            "-i", str(src),
            "-vn",
            "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(tmp),
        ],
        check=True, capture_output=True, timeout=3600,
    )
    tmp.replace(dest)
    return dest


def speech_regions(wav: Path, duration: float) -> list[speech_vad.Region]:
    """Silero regions for the whole file, paged so RAM stays flat."""
    regions: list[speech_vad.Region] = []
    with _WavReader(wav) as reader:
        block = 300.0
        t = 0.0
        while t < duration:
            end = min(duration, t + block)
            samples = reader.samples(t, end)
            for r in speech_vad.speech_regions(samples, limit=end - t):
                regions.append(speech_vad.Region(r.start + t, r.end + t))
            t = end
    return regions


def plan_windows(duration: float) -> list[tuple[float, float]]:
    if duration <= WINDOW_SEC:
        return [(0.0, max(duration, 0.5))]
    out: list[tuple[float, float]] = []
    step = WINDOW_SEC - OVERLAP_SEC
    t = 0.0
    while t < duration - 0.25:
        out.append((t, min(t + WINDOW_SEC, duration)))
        t += step
    return out


def _write_span(reader: _WavReader, path: Path, start: float, end: float) -> Path:
    import numpy as np

    samples = reader.samples(start, end)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(pcm.tobytes())
    return path


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #

def norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", (text or "").lower())


#: Whisper writes "Mr."; a spelled-out reference writes "MISTER". Counting that
#: as an error would make every model look worse than it is, and would punish
#: the model that uses correct abbreviations more.
_WER_EXPANSIONS = {"mr": "mister", "mrs": "missus", "dr": "doctor", "st": "saint", "jr": "junior"}


def wer(reference: str, hypothesis: str) -> float | None:
    """Word error rate, punctuation- and case-insensitive."""
    ref = [_WER_EXPANSIONS.get(norm_token(t), norm_token(t)) for t in (reference or "").split()]
    hyp = [_WER_EXPANSIONS.get(norm_token(t), norm_token(t)) for t in (hypothesis or "").split()]
    ref = [t for t in ref if t]
    hyp = [t for t in hyp if t]
    if not ref:
        return None
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def _prompt(title: str, description: str | None, tail: str) -> str:
    bits = [b for b in (title.strip(), (description or "").strip()[:200]) if b]
    text = " — ".join(bits)
    if tail:
        text = f"{text} {tail}".strip()
    return text[-PROMPT_MAX_CHARS:]


def _params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #

def align_words(
    words: list[dict],
    reader: _WavReader,
    regions: list[speech_vad.Region],
    cache_root: Path,
    *,
    span: tuple[float, float] | None = None,
) -> tuple[list[dict], bool]:
    """Region-anchored CTC alignment; returns (words, did_align).

    Each Silero region is aligned on its own, with only the words whose
    midpoint falls inside it. Aligning a whole 30 s window at once lets the CTC
    lattice distribute tokens across music and silence (measured: it pulled the
    opening words 6.7 s early, into a music bed), while a region is by
    construction pure speech.
    """
    if not words or not regions or not ctc_align.model_ready(cache_root):
        return words, False
    out: list[dict] = []
    by_region: dict[int, list[int]] = {}
    for idx, w in enumerate(words):
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        best = -1
        best_d = 1e9
        for ri, r in enumerate(regions):
            if r.start - 0.30 <= mid <= r.end + 0.30:
                d = 0.0 if r.start <= mid <= r.end else min(abs(mid - r.start), abs(mid - r.end))
                if d < best_d:
                    best, best_d = ri, d
        if best >= 0:
            by_region.setdefault(best, []).append(idx)
    if not by_region:
        return words, False
    moved: dict[int, dict] = {}
    for ri, idxs in by_region.items():
        r = regions[ri]
        lo = max(0.0, r.start - 0.10)
        hi = min(reader.frames / reader.rate, r.end + 0.10)
        text = " ".join(str(words[i]["word"]) for i in idxs)
        samples = reader.samples(lo, hi)
        aligned = ctc_align.align_text(
            samples, text, cache_root=cache_root, offset=lo,
            speech=[(r.start - lo, r.end - lo)],
        )
        if not aligned or len(aligned) != len(idxs):
            continue
        for i, ali in zip(idxs, aligned):
            start = max(lo, ali.start)
            end = min(hi, max(ali.end, start + 0.04))
            moved[i] = {**words[i], "start": round(start, 3), "end": round(end, 3)}
    if not moved:
        return words, False
    for i, w in enumerate(words):
        out.append(moved.get(i, w))
    out.sort(key=lambda w: (w["start"], w["end"]))
    return out, True


def snap_to_speech(words: list[dict], regions: list[speech_vad.Region]) -> list[dict]:
    out = []
    for w in words:
        start, end = speech_vad.snap_to_speech(float(w["start"]), float(w["end"]), regions)
        out.append({**w, "start": round(start, 3), "end": round(end, 3)})
    return out


def enforce_order(words: list[dict]) -> list[dict]:
    """Sorted by start, no overlaps, no zero-length words."""
    ordered = sorted(words, key=lambda w: (float(w["start"]), float(w["end"])))
    out: list[dict] = []
    for w in ordered:
        start = max(0.0, float(w["start"]))
        end = max(start + 0.04, float(w["end"]))
        if out and start < out[-1]["end"]:
            start = out[-1]["end"]
        if end <= start:
            end = start + 0.04
        out.append({**w, "start": round(start, 3), "end": round(end, 3)})
    return out


def collapse_repeats(words: list[dict], *, max_repeat: int = 3, max_gap: float = 8.0) -> tuple[list[dict], int]:
    """Break a Whisper repetition loop by keeping only its first three copies.

    Measured on this app's own clip: one segment came back as "old is kai, kai,
    kai, kai, kai, kai, please, sub" with no_speech_prob 0.03 and avg_logprob
    -0.35 — the model was CONFIDENT it heard six identical words, so none of
    the confidence gates fire. Repetition is the only signal left. Only copies
    beyond the third are removed, so a deliberate "no, no, no" survives intact.
    """
    kept: list[dict] = []
    run = 0
    removed = 0
    for w in words:
        token = norm_token(str(w["word"]))
        prev = kept[-1] if kept else None
        same = bool(token) and prev is not None and norm_token(str(prev["word"])) == token
        if same and float(w["start"]) - float(prev["end"]) <= max_gap:
            run += 1
        else:
            run = 0
        if same and run >= max_repeat:
            removed += 1
            continue
        kept.append(w)
    return kept, removed


def drop_silence_words(words: list[dict], regions: list[speech_vad.Region]) -> tuple[list[dict], int]:
    """Remove words that Silero says are in pure silence — and nothing else.

    The test is "no overlap with any speech region, padded by 150 ms", not a
    voicing RATIO: a rapidly spoken word at the edge of a region can have a low
    ratio while being perfectly real, and dropping real words is the one
    failure a transcription pipeline must not trade for cleanliness. Measured
    on a real 90 s window, a ratio gate at 0.10 removed 22 of 114 words; this
    one removes the ones in the music bed only.
    """
    padded = [(r.start - 0.15, r.end + 0.15) for r in regions]
    kept: list[dict] = []
    dropped = 0
    for w in words:
        start, end = float(w["start"]), float(w["end"])
        if speech_vad.overlap_seconds([speech_vad.Region(s, e) for s, e in padded], start, end) <= 0.0:
            dropped += 1
            continue
        kept.append(w)
    return kept, dropped


# --------------------------------------------------------------------------- #
# sync report
# --------------------------------------------------------------------------- #

#: A pause shorter than this inside speech is a breath between words, not a
#: stretch of dialogue the captions failed to cover.
GAP_TOLERANCE = 0.30
#: A caption may lead or trail the speech it belongs to by this much (the
#: caption builder extends a cue to its last word end + 0.15 s).
SILENCE_TOLERANCE = 0.20


def _expand(spans: list[tuple[float, float]], pad: float) -> list[tuple[float, float]]:
    return [(max(0.0, s - pad), e + pad) for s, e in spans]


def _union(spans: list[tuple[float, float]]) -> float:
    if not spans:
        return 0.0
    total = 0.0
    cur_s, cur_e = sorted(spans)[0]
    for s, e in sorted(spans)[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s)


def sync_report(
    words: list[dict],
    regions: list[speech_vad.Region],
    duration: float,
) -> dict[str, Any]:
    """Compare caption times with what Silero says is speech. No reference text."""
    def clamp(start: float, end: float) -> tuple[float, float] | None:
        lo = min(max(0.0, float(start)), duration)
        hi = min(max(lo, float(end)), duration)
        return (lo, hi) if hi > lo else None

    speech_spans = [s for s in (clamp(r.start, r.end) for r in regions if r.end > r.start) if s]
    cap_spans = [s for s in (clamp(float(w["start"]), float(w["end"])) for w in words) if s]
    speech_total = _union(speech_spans)
    cap_total = _union(cap_spans)
    if speech_total <= 0:
        return {
            "speechSec": 0.0, "captionSec": round(cap_total, 2),
            "speechWithoutCaption": None, "captionOnSilence": None,
            "avgStartErrorMs": None, "passed": False,
            "reason": "no speech detected by Silero VAD",
        }
    covered = min(speech_total, _union(
        _intersections(speech_spans, _expand(cap_spans, GAP_TOLERANCE))
    ))
    no_caption = max(0.0, speech_total - covered) / speech_total
    cap_covered = min(cap_total, _union(
        _intersections(cap_spans, _expand(speech_spans, SILENCE_TOLERANCE))
    ))
    on_silence = max(0.0, cap_total - cap_covered) / cap_total if cap_total > 0 else 0.0

    # Caption start error: the caption that OPENS each utterance should appear
    # where that utterance starts. A burst is the first word after a pause, and
    # it is only compared when it really does open a region — a burst in the
    # middle of speech has no onset to be early or late for.
    errors: list[float] = []
    for i, w in enumerate(words):
        start = float(w["start"])
        prev_end = float(words[i - 1]["end"]) if i else -1.0
        if i and start - prev_end <= 0.4:
            continue
        onset = min(
            (r.start for r in regions if start - 0.6 <= r.start <= start + 0.25),
            key=lambda s: abs(s - start), default=None,
        )
        if onset is None:
            continue
        errors.append(abs(start - onset))
    avg_err = sum(errors) / len(errors) if errors else None
    passed = (
        avg_err is not None
        and avg_err <= MAX_START_ERROR
        and no_caption <= MAX_NO_CAPTION
        and on_silence <= MAX_CAPTION_ON_SILENCE
    )
    return {
        "speechSec": round(speech_total, 2),
        "captionSec": round(cap_total, 2),
        "speechWithoutCaption": round(no_caption, 4),
        "captionOnSilence": round(on_silence, 4),
        "avgStartErrorMs": None if avg_err is None else round(avg_err * 1000.0, 1),
        "regions": len(regions),
        "startSamples": len(errors),
        "passed": bool(passed),
    }


def _intersections(speech: list[tuple[float, float]], caps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for s, e in speech:
        for cs, ce in caps:
            lo, hi = max(s, cs), min(e, ce)
            if hi > lo:
                out.append((lo, hi))
    return out


# --------------------------------------------------------------------------- #
# manual subtitles (ladder 1a)
# --------------------------------------------------------------------------- #

def manual_cues(
    media: Path,
    cache_dir: Path | None,
    source_url: str | None,
    languages: tuple[str, ...],
) -> tuple[list[subs_source.Cue], str]:
    """Creator-uploaded cues for this source, or ([], reason)."""
    if cache_dir is not None:
        for path in subs_source.find_local_subs(cache_dir / "subs") or []:
            try:
                cues = subs_source.parse_file(path)
                if cues:
                    return cues, f"creator subtitles ({path.name})"
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[accurate] unreadable subtitle {path.name}: {exc}")
    for path in subs_source.find_local_subs(media):
        try:
            cues = subs_source.parse_file(path)
            if cues:
                return cues, f"creator subtitles ({path.name})"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[accurate] unreadable subtitle {path.name}: {exc}")
    if not source_url or cache_dir is None:
        return [], "no source URL for creator subtitles"
    files = subs_source.fetch(source_url, cache_dir / "subs", languages=languages)
    best = subs_source.pick_best(files, prefer=languages[0])
    if best is None:
        return [], "no creator-uploaded subtitles on the source"
    cues = subs_source.parse_file(best)
    return cues, f"creator subtitles ({best.name})" if cues else "empty creator subtitles"


def _align_cues(
    cues: list[subs_source.Cue],
    reader: _WavReader,
    regions: list[speech_vad.Region],
    cache_root: Path,
    on_progress: Callable[[float], None] | None,
    duration: float,
) -> list[dict]:
    """Force-align creator text to the audio, cue by cue.

    A cue's own [start, end] is a human's estimate of where the line is spoken,
    which makes the search window tight and the alignment well-conditioned. When
    the aligner refuses (non-Latin text, a cue too short for its characters),
    the cue's words keep the cue's own timing, split in proportion to length —
    never a guess about WHICH words were said, only about when.
    """
    words: list[dict] = []
    for i, cue in enumerate(cues):
        lo = max(0.0, cue.start - 0.20)
        hi = min(reader.frames / reader.rate, cue.end + 0.20)
        tokens = [t for t in cue.text.split() if t.strip()]
        if not tokens:
            continue
        aligned = None
        if hi > lo and ctc_align.model_ready(cache_root):
            span_regions = [
                (max(0.0, r.start - lo), min(hi - lo, r.end - lo))
                for r in regions if r.end > lo and r.start < hi
            ]
            aligned = ctc_align.align_text(
                reader.samples(lo, hi), cue.text, cache_root=cache_root,
                offset=lo, speech=span_regions or None,
            )
        if aligned and len(aligned) == len(tokens):
            for ali in aligned:
                words.append({
                    "word": ali.word, "start": round(min(hi, max(lo, ali.start)), 3),
                    "end": round(min(hi, max(ali.end, ali.start + 0.04)), 3),
                    "conf": 0.9, "check": False,
                })
        else:
            total = sum(len(t) for t in tokens) or 1
            t0 = cue.start
            for tok in tokens:
                share = (len(tok) / total) * max(0.2, cue.end - cue.start)
                words.append({
                    "word": tok, "start": round(t0, 3),
                    "end": round(t0 + share, 3), "conf": 0.7, "check": False,
                })
                t0 += share
        if on_progress and duration > 0:
            on_progress(min(1.0, cue.end / duration))
    return words


# --------------------------------------------------------------------------- #
# ASR (ladder 1b)
# --------------------------------------------------------------------------- #

@dataclass
class Attempt:
    start: float
    end: float
    model: str
    ok: bool
    peak_rss_mb: float = 0.0
    reason: str = ""
    pass_no: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 1), "end": round(self.end, 1),
            "model": self.model, "ok": self.ok,
            "peakRssMb": round(self.peak_rss_mb, 1),
            "reason": self.reason[:200], "pass": self.pass_no,
        }


def model_ladder(language: str | None) -> list[str]:
    """Largest model that this host can hold, then strictly smaller ones."""
    if language and language != "en":
        names = ["small", "base", "tiny"]
    else:
        # small.en is deliberately absent: on this host's own LibriSpeech run it
        # scored WORSE than plain small (10.1% vs 4.7% WER) while costing the
        # same 461 MB of cache, so it was a step that could only waste a
        # download and 461 MB of disk.
        names = ["distil-medium.en", "small", "base"]
    fits = [m for m in names if asr_quality.fits(m)]
    return fits or ["tiny"]


def _decode_windows(
    reader: _WavReader,
    windows: list[tuple[float, float]],
    *,
    languages: list[str],
    prompt_base: tuple[str, str | None],
    cache_dir: Path | None,
    tmp_dir: Path,
    should_pause: Callable[[], bool] | None,
    on_progress: Callable[[float], None] | None,
    on_chunk: Callable[..., None] | None,
    on_detail: Callable[[str], None] | None,
    duration: float,
) -> tuple[list[dict], list[dict], list[Attempt], str, str | None, float]:
    """Decode every window with the largest model that fits; checkpoint each."""
    attempts: list[Attempt] = []
    cycles: list[dict] = []
    language = languages[0] if languages and languages[0] not in ("auto", "") else None
    ladder = model_ladder(language)
    model = ladder[0]
    detected: str | None = None
    peak = 0.0
    proc: asr_quality.DecodeProcess | None = None
    last_sentence = ""
    resumed = 0
    try:
        for idx, (w_start, w_end) in enumerate(windows):
            if should_pause is not None and should_pause():
                raise TranscribeError("paused")
            cache_file = (cache_dir / f"win_{idx:05d}.json") if cache_dir is not None else None
            params = {"window": [w_start, w_end], "beam": 5, "temperature": [0.0, 0.2, 0.4]}
            if cache_file is not None and cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    if data.get("params") == params:
                        cycles.append(data)
                        model = data.get("model") or model
                        # The peak is part of the checkpoint: a resumed job must
                        # still be able to say what the decode cost on this host.
                        peak = max(peak, float(data.get("peak_rss_mb") or 0.0))
                        last_sentence = (data.get("words") or [{}])[-1].get("word", "") if data.get("words") else ""
                        resumed += 1
                        if on_progress and duration > 0:
                            on_progress(min(1.0, w_end / duration))
                        if on_chunk is not None:
                            on_chunk(idx, len(windows), phase="done", resuming=True)
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[accurate] unreadable checkpoint {cache_file.name}: {exc}")

            if on_chunk is not None:
                on_chunk(idx, len(windows), phase="start")
            prompt = _prompt(prompt_base[0], prompt_base[1], last_sentence)
            wav = _write_span(reader, tmp_dir / f"win_{idx:05d}.wav", w_start, w_end)
            reply = None
            start_model = model
            for candidate in ladder[ladder.index(start_model):] if start_model in ladder else ladder:
                try:
                    if proc is None or proc.model != candidate:
                        if proc is not None:
                            proc.close()
                        if on_detail:
                            on_detail(f"chunk {idx + 1}/{len(windows)} (model {candidate})")
                        proc = asr_quality.DecodeProcess(
                            candidate, log_path=tmp_dir / "asr_child.log",
                        )
                        model = candidate
                    reply = proc.decode(
                        wav, language=language, beam_size=5,
                        temperature=[0.0, 0.2, 0.4], initial_prompt=prompt,
                    )
                    attempts.append(Attempt(w_start, w_end, candidate, True,
                                            float(reply.get("peak_rss_mb") or 0.0)))
                    peak = max(peak, float(reply.get("peak_rss_mb") or 0.0))
                    break
                except asr_quality.DecodeError as exc:
                    attempts.append(Attempt(w_start, w_end, candidate, False, 0.0, str(exc)))
                    logger.warning(f"[accurate] chunk {idx + 1}: {candidate} failed: {exc}")
                    if proc is not None:
                        proc.kill()
                        proc = None
                    if not exc.memory:
                        raise
            if reply is None:
                raise TranscribeError(
                    f"no local model could decode {w_start:.0f}-{w_end:.0f}s; "
                    f"tried {', '.join(ladder)}"
                )
            if detected is None and reply.get("language"):
                detected = str(reply["language"])
                if language is not None and detected != language:
                    logger.warning(f"[accurate] detected {detected} but {language} was requested")
                if language is None and detected != "en" and model in asr_quality.MEASURED_PEAK_MB:
                    # distil-*.en cannot decode another language: after the
                    # first window tells us what this audio is, switch the
                    # ladder instead of forcing English onto it.
                    language = detected
                    ladder = model_ladder(language)
                    logger.info(
                        f"[accurate] detected language '{detected}' — ladder is now "
                        f"{', '.join(ladder)}"
                    )
            words = [
                {"word": str(w["word"]), "start": round(float(w["start"]) + w_start, 3),
                 "end": round(float(w["end"]) + w_start, 3),
                 "conf": float(w.get("prob") or 0.0), "check": False}
                for w in reply.get("words") or []
            ]
            segments = [
                {**s, "start": round(float(s["start"]) + w_start, 3),
                 "end": round(float(s["end"]) + w_start, 3)}
                for s in reply.get("segments") or []
            ]
            cycle = {
                "start": w_start, "end": w_end, "model": model, "params": params,
                "words": words, "segments": segments, "language": reply.get("language"),
                "peak_rss_mb": float(reply.get("peak_rss_mb") or 0.0),
            }
            cycles.append(cycle)
            if words:
                tail = " ".join(str(w["word"]) for w in words[-12:])
                last_sentence = tail[-160:]
            if cache_file is not None:
                tmp = cache_file.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(cycle, ensure_ascii=False), encoding="utf-8")
                tmp.replace(cache_file)
            if on_progress and duration > 0:
                on_progress(min(1.0, w_end / duration))
            if on_chunk is not None:
                on_chunk(idx, len(windows), phase="done")
            try:
                wav.unlink(missing_ok=True)
            except OSError:
                pass
    finally:
        if proc is not None:
            proc.close()
    if resumed:
        logger.info(f"[accurate] resumed {resumed}/{len(windows)} window(s) from checkpoint")
    return (
        [w for c in cycles for w in c.get("words") or []],
        cycles,
        attempts,
        model,
        detected,
        peak,
    )


def _second_pass(
    cycles: list[dict],
    reader: _WavReader,
    *,
    tmp_dir: Path,
    language: str | None,
    title: str,
    attempts: list[Attempt],
) -> int:
    """Re-decode low-confidence windows with different sampling settings.

    A window that needed a temperature fallback internally (any segment with
    temperature > 0) or whose worst segment is below -1.0 log-prob is decoded
    again at beam 3 / temperature 0.2-0.4. Words the two passes agree on keep
    their timing; words they disagree on are flagged "check this" — never
    replaced, because neither pass can prove the other wrong.
    """
    flagged = 0
    model = cycles[0].get("model") if cycles else ""
    proc: asr_quality.DecodeProcess | None = None
    try:
        for cycle in cycles:
            segs = cycle.get("segments") or []
            worst = min((float(s.get("avg_logprob", 0.0)) for s in segs), default=0.0)
            hot = any(float(s.get("temperature") or 0.0) > 0.0 for s in segs)
            if not segs or (worst >= SECOND_PASS_LOGPROB and not hot):
                continue
            wav = _write_span(reader, tmp_dir / f"re_{int(cycle['start']):05d}.wav",
                              float(cycle["start"]), float(cycle["end"]))
            try:
                if proc is None or proc.model != model:
                    if proc is not None:
                        proc.close()
                    proc = asr_quality.DecodeProcess(model, log_path=tmp_dir / "asr_child.log")
                reply = proc.decode(
                    wav, language=language, beam_size=3,
                    temperature=[0.2, 0.4], initial_prompt=title[:PROMPT_MAX_CHARS],
                    condition_on_previous_text=False,
                )
            except asr_quality.DecodeError as exc:
                attempts.append(Attempt(float(cycle["start"]), float(cycle["end"]),
                                        model, False, 0.0, f"second pass: {exc}", pass_no=2))
                if proc is not None:
                    proc.kill()
                    proc = None
                continue
            finally:
                try:
                    wav.unlink(missing_ok=True)
                except OSError:
                    pass
            attempts.append(Attempt(float(cycle["start"]), float(cycle["end"]), model, True,
                                    float(reply.get("peak_rss_mb") or 0.0), "second pass", 2))
            second = [
                {"word": str(w["word"]), "start": float(w["start"]) + float(cycle["start"]),
                 "end": float(w["end"]) + float(cycle["start"])}
                for w in reply.get("words") or []
            ]
            if not second:
                continue
            for w in cycle.get("words") or []:
                mid = (float(w["start"]) + float(w["end"])) / 2.0
                near = min(second, key=lambda s: abs((s["start"] + s["end"]) / 2.0 - mid), default=None)
                if near is None:
                    continue
                if abs((near["start"] + near["end"]) / 2.0 - mid) > 0.45:
                    continue
                if norm_token(near["word"]) != norm_token(w["word"]):
                    w["check"] = True
                    flagged += 1
    finally:
        if proc is not None:
            proc.close()
    return flagged


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SourceHint:
    """What the job knew about this source, recovered without changing the caller."""

    url: str | None = None
    title: str = ""
    description: str | None = None


def source_hint(media: Path, cache_dir: Path | None, settings: Settings) -> SourceHint:
    """Recover the source URL/title for the subtitle fetch from the job's own files.

    `transcribe()` is handed a media path and a cache dir, not the job. The job
    document sits beside the work dir (data/jobs/<id>.json) and carries the
    URL, title and description the fetch stage recorded, so the subtitle step
    can use them WITHOUT this module changing its caller's interface.
    """
    candidates: list[Path] = []
    if cache_dir is not None:
        # .../data/work/<job_id>/transcribe/chunks -> <job_id>
        parts = Path(cache_dir).parts
        if "work" in parts:
            idx = len(parts) - 1 - parts[::-1].index("work")
            if idx + 1 < len(parts):
                job_id = parts[idx + 1]
                candidates.append(Path(settings.app.paths.data_dir) / "jobs" / f"{job_id}.json")
    candidates.append(media.with_suffix(".info.json"))
    candidates.append(media.parent / f"{media.stem}.info.json")
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        url = data.get("source") or data.get("url") or data.get("webpage_url")
        if isinstance(url, str) and url.startswith("http"):
            return SourceHint(
                url=url,
                title=str(data.get("title") or ""),
                description=data.get("description") or data.get("summary"),
            )
    return SourceHint()


def transcribe_source(
    path: Path,
    settings: Settings,
    *,
    cache_dir: Path | None = None,
    language: str | None = None,
    on_progress: Callable[[float], None] | None = None,
    on_chunk: Callable[..., None] | None = None,
    on_detail: Callable[[str], None] | None = None,
    should_pause: Callable[[], bool] | None = None,
    source_url: str | None = None,
    title: str = "",
    description: str | None = None,
) -> Transcript:
    started = time.time()
    duration = _probe_duration(path)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = (cache_dir or Path(settings.app.paths.data_dir)) / "windows"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(settings.app.paths.data_dir) / "models"
    langs = [language] if language else ["auto"]

    mode = "asr"
    model_used = ""
    label = ""
    attempts: list[Attempt] = []
    detected: str | None = None
    peak = 0.0
    aligned = False

    wav = prepare_audio(path, (cache_dir or Path(settings.app.paths.data_dir)) / "speech16k.wav")
    regions = speech_regions(wav, duration)
    logger.info(
        f"[accurate] {duration:.1f}s audio, {len(regions)} speech region(s) "
        f"({speech_vad.speech_seconds(regions):.1f}s voiced), {asr_quality.rss_cap_mb():.0f} MB RSS cap"
    )

    cues, cue_reason = manual_cues(
        path, cache_dir, source_url,
        tuple(dict.fromkeys([*( [language] if language else [] ), *subs_source.DEFAULT_LANGS])),
    )
    reader = _WavReader(wav)
    try:
        if cues:
            mode = "manual"
            model_used = "creator-captions+wav2vec2"
            words = _align_cues(cues, reader, regions, cache_root, on_progress, duration)
            aligned = True
            label = cue_reason
            if on_chunk is not None:
                on_chunk(0, 1, phase="done")
        else:
            logger.info(f"[accurate] no creator subtitles ({cue_reason}) — decoding locally")
            windows = plan_windows(duration)
            words, cycles, attempts, model_used, detected, peak = _decode_windows(
                reader, windows, languages=langs, prompt_base=(title, description),
                cache_dir=cache_dir, tmp_dir=tmp_dir, should_pause=should_pause,
                on_progress=on_progress, on_chunk=on_chunk, on_detail=on_detail,
                duration=duration,
            )
            if cycles:
                flagged = _second_pass(
                    cycles, reader, tmp_dir=tmp_dir, language=language or detected,
                    title=title, attempts=attempts,
                )
                words = [w for c in cycles for w in c.get("words") or []]
                if flagged:
                    logger.info(f"[accurate] second pass flagged {flagged} word(s) 'check this'")
            words, aligned = align_words(words, reader, regions, cache_root)
            if model_used != (settings.whisper.model or ""):
                label = f"transcribed with {model_used}"
    finally:
        reader.close()

    words, dropped_silence = drop_silence_words(words, regions)
    words, dropped_repeats = collapse_repeats(words)
    words = snap_to_speech(words, regions)
    words = enforce_order(words)
    words = _dedupe_words(
        [Word(start=float(w["start"]), end=float(w["end"]), word=str(w["word"]),
              conf=float(w.get("conf", 0.0)), check=bool(w.get("check", False)))
         for w in words],
        _SUB_OVERLAP,
    )
    word_dicts = [
        {"word": w.word, "start": w.start, "end": w.end, "conf": w.conf, "check": w.check}
        for w in words
    ]
    report = sync_report(word_dicts, regions, duration)
    if dropped_silence:
        report["wordsDroppedInSilence"] = dropped_silence
    if dropped_repeats:
        report["wordsDroppedInRepeat"] = dropped_repeats
        logger.info(f"[accurate] repetition loop: dropped {dropped_repeats} duplicated word(s)")

    reference = ""
    reference_kind = ""
    if cues:
        reference, reference_kind = subs_source.cues_text(cues), "creator captions"
    if not reference and cache_dir is not None and source_url:
        ref_files = subs_source.fetch(source_url, cache_dir / "ref_subs", auto=True)
        ref_path = subs_source.pick_best(ref_files)
        if ref_path:
            reference, reference_kind = subs_source.cues_text(subs_source.parse_file(ref_path)), "auto captions"
    hypothesis = " ".join(w.word for w in words)
    error_rate = wer(reference, hypothesis) if reference else None
    if error_rate is not None:
        report["wer"] = round(error_rate, 4)
        report["werAgainst"] = reference_kind
        if error_rate > MAX_WER:
            report["passed"] = False
            report["reason"] = f"WER {error_rate:.0%} against {reference_kind} is above {MAX_WER:.0%}"

    flagged_words = sum(1 for w in words if w.check)
    needs_review = not report.get("passed", False)
    transcript = Transcript(
        words=words,
        duration=duration,
        model_size=model_used,
        engine="faster-whisper (local)" if mode == "asr" else "creator captions + wav2vec2 (local)",
        rtf=round((time.time() - started) / duration, 4) if duration > 0 else None,
        language=detected or (language or ""),
        attempts=[a.as_dict() for a in attempts],
        label=label,
        peak_rss_mb=round(peak, 1),
        sync=report,
        check_words=flagged_words,
        needs_review=needs_review,
        aligned=aligned,
    )
    logger.info(
        f"[accurate] {len(words)} words, mode {mode}, model '{model_used}', "
        f"{flagged_words} flagged, sync {'PASS' if report.get('passed') else 'REVIEW'} "
        f"({json.dumps(report, ensure_ascii=False)[:300]})"
    )
    return transcript

