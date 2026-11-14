#!/usr/bin/env python3
"""Before/after for low-confidence captions, on a REAL clip window.

Runs the real cue builder (``subtitle.words_to_cues`` — the same function the
render stage calls) over words decoded from a real job's source, and prints
the caption lines before and after the low-confidence rule is applied.

This is not a simulation: the audio is the job's own download, the model and
decode kwargs are the pipeline's, and the "before" text is what Whisper
actually produced for that window.

Run:  .venv/bin/python tools/uncertainty_demo.py fd961148baa5 1
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (
    asr,  # noqa: E402
    subtitle,  # noqa: E402
)
from app.asr import ANTI_REPETITION  # noqa: E402
from app.config import ffmpeg_bin, load_config  # noqa: E402
from app.store import LocalStore  # noqa: E402


def decode_window(store, cfg, job_id: str, clip_no: int):
    clips = sorted(store.list_clips(job_id), key=lambda c: c.get("order", 0))
    clip = clips[clip_no - 1]
    src = store.find_source_file(job_id)
    if src is None:
        raise SystemExit(f"no source for {job_id}")
    wav = Path(tempfile.mkdtemp(prefix="unc-demo-")) / "w.wav"
    subprocess.run(
        [ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{float(clip['start']):.3f}",
         "-t", f"{float(clip['end']) - float(clip['start']):.3f}", "-i", str(src),
         "-ac", "1", "-ar", "16000", str(wav)],
        check=True, capture_output=True,
    )
    from faster_whisper import WhisperModel

    from app import hostlimits

    model_name, _ = hostlimits.resolved_model(cfg.whisper.model)
    model = WhisperModel(model_name, device=cfg.whisper.device or "cpu",
                         compute_type=cfg.whisper.resolved_compute)
    segments, _info = model.transcribe(
        str(wav), language=cfg.whisper.language or None, task="transcribe",
        beam_size=5, word_timestamps=True, **ANTI_REPETITION,
    )
    words = []
    for seg in segments:
        seg_conf = asr._segment_confidence(seg)
        for w in (seg.words or []):
            text = (w.word or "").strip()
            if not text:
                continue
            prob = float(getattr(w, "probability", 0.0) or 0.0)
            words.append({
                "start": float(w.start), "end": float(w.end), "word": text,
                "conf": min(1.0, max(0.0, prob)) if prob > 0 else min(1.0, seg_conf),
            })
    return clip, words


def main() -> int:
    job_id = sys.argv[1] if len(sys.argv) > 1 else "fd961148baa5"
    clip_no = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    cfg = load_config()
    store = LocalStore(cfg.app.paths)
    clip, words = decode_window(store, cfg, job_id, clip_no)

    # Caption cues are CLIP-RELATIVE, and this window IS the clip, so the
    # window-relative word timings are already correct. Shifting them by the
    # clip start (as an earlier version did) produced negative timestamps.
    cue_words = list(words)
    on = subtitle.words_to_cues(cue_words, words_per_cue=cfg.subtitle.words_per_cue)
    off = subtitle.words_to_cues(
        cue_words, words_per_cue=cfg.subtitle.words_per_cue,
        confidence_threshold=0.0,  # rule disabled, to get the untouched text
    )

    print(f"job {job_id} clip {clip_no} — {clip.get('title')!r}  "
          f"({clip['start']:.1f}s -> {clip['end']:.1f}s, {len(words)} words)")
    print(f"threshold: mean word confidence < {subtitle.LOW_CONFIDENCE} "
          f"(< {subtitle.INAUDIBLE_BELOW} -> {subtitle.PLACEHOLDER_INAUDIBLE})")
    print()
    changed = 0
    for a, b in zip(off, on):
        if a["text"] == b["text"]:
            continue
        changed += 1
        print(f"  {a['start']:6.2f}s  BEFORE: {a['text']}")
        print(f"          {'':6}  AFTER : {b['text']}   "
              f"(mean confidence {b.get('confidence')})")
        print(f"          words kept: {' '.join(w['word'] for w in a['words'])}")
        print()
    print(f"{changed} of {len(on)} cue(s) replaced. {len(on) - changed} kept verbatim.")

    # The decision is made per CUE, so calibrate on the cue distribution, not
    # on per-segment averages — they are not the same population.
    means = sorted(
        subtitle.cue_confidence(c["words"]) for c in off if c.get("words")
    )
    means = [m for m in means if m is not None]
    if means:
        def pct(q: float) -> float:
            return means[min(len(means) - 1, int(len(means) * q))]
        print("\ncue-mean confidence distribution (the unit the rule decides on):")
        print(f"  n={len(means)}  min={means[0]:.3f}  p10={pct(0.10):.3f}  "
              f"p25={pct(0.25):.3f}  median={pct(0.50):.3f}  p75={pct(0.75):.3f}  "
              f"max={means[-1]:.3f}")
        for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            below = sum(1 for m in means if m < t)
            print(f"  cues below {t:.2f}: {below:3d} / {len(means)} ({100*below/len(means):4.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
