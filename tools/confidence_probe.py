#!/usr/bin/env python3
"""Measure Whisper's own confidence on a REAL clip window.

The subtitle fix for unclear speech needs a threshold, and a threshold picked
out of the air is a guess with a number attached. This tool decodes a real
window of a real job's source with the same model and the same decode kwargs
the pipeline uses, then prints what the model actually said about its own
certainty:

  * per segment: ``avg_logprob`` (mean per-token log probability) and
    ``no_speech_prob``
  * per word: ``probability`` — faster-whisper's own token confidence

Run:  .venv/bin/python tools/confidence_probe.py 885ba6a7ac26 2
      (job id, 1-based clip number)
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.asr import ANTI_REPETITION  # noqa: E402
from app.config import ffmpeg_bin, load_config  # noqa: E402
from app.store import LocalStore  # noqa: E402

DEFAULT_JOB = "885ba6a7ac26"
DEFAULT_CLIP = 2


def clip_window(store: LocalStore, job_id: str, clip_no: int) -> tuple[float, float, str, Path]:
    clips = sorted(store.list_clips(job_id), key=lambda c: c.get("order", 0))
    if not 1 <= clip_no <= len(clips):
        raise SystemExit(f"clip {clip_no} not in job {job_id} ({len(clips)} clips)")
    clip = clips[clip_no - 1]
    src = store.find_source_file(job_id)
    if src is None:
        raise SystemExit(f"no source file for job {job_id}")
    return float(clip["start"]), float(clip["end"]), str(clip.get("title")), src


def slice_audio(src: Path, start: float, end: float) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="conf-probe-"))
    wav = tmp / "window.wav"
    subprocess.run(
        [ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
         "-t", f"{end - start:.3f}", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)],
        check=True, capture_output=True,
    )
    return wav


def main() -> int:
    job_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JOB
    clip_no = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CLIP

    cfg = load_config()
    store = LocalStore(cfg.app.paths)
    start, end, title, src = clip_window(store, job_id, clip_no)
    print(f"job {job_id} clip {clip_no} — {title!r}  {start:.1f}s -> {end:.1f}s")
    print(f"source: {src}")

    from faster_whisper import WhisperModel

    from app import hostlimits

    model_name, note = hostlimits.resolved_model(cfg.whisper.model)
    print(f"model: {model_name} (configured: {cfg.whisper.model}) {note or ''}")
    model = WhisperModel(model_name, device=cfg.whisper.device or "cpu",
                         compute_type=cfg.whisper.resolved_compute)

    wav = slice_audio(src, start, end)
    segments, info = model.transcribe(
        str(wav),
        language=cfg.whisper.language or None,
        task="transcribe",
        beam_size=5,
        word_timestamps=True,
        **ANTI_REPETITION,
    )

    rows = []
    for seg in segments:
        avg = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
        nospeech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        words = [
            {
                "word": (w.word or "").strip(),
                "start": round(float(w.start or 0.0), 2),
                "end": round(float(w.end or 0.0), 2),
                "prob": round(float(getattr(w, "probability", 0.0) or 0.0), 3),
            }
            for w in (seg.words or [])
            if (w.word or "").strip()
        ]
        probs = [w["prob"] for w in words if w["prob"] > 0]
        rows.append({
            "start": round(float(seg.start), 2),
            "end": round(float(seg.end), 2),
            "text": (seg.text or "").strip(),
            "avgLogprob": round(avg, 3),
            "segConf": round(math.exp(avg), 3),
            "noSpeechProb": round(nospeech, 3),
            "meanWordProb": round(sum(probs) / len(probs), 3) if probs else None,
            "words": words,
        })

    print(f"\nsegments: {len(rows)}  language: {getattr(info, 'language', None)}")
    print(f"{'start':>7} {'segConf':>8} {'meanWord':>9} {'noSpch':>7}  text")
    for r in rows:
        mwp = f"{r['meanWordProb']:.3f}" if r["meanWordProb"] is not None else "  -  "
        print(f"{r['start']:7.2f} {r['segConf']:8.3f} {mwp:>9} {r['noSpeechProb']:7.3f}  {r['text'][:70]}")

    allp = [w["prob"] for r in rows for w in r["words"] if w["prob"] > 0]
    if allp:
        allp.sort()
        def pct(q: float) -> float:
            return allp[min(len(allp) - 1, int(len(allp) * q))]
        print(f"\nword probability over {len(allp)} tokens:")
        print(f"  min={allp[0]:.3f}  p05={pct(0.05):.3f}  p25={pct(0.25):.3f}  "
              f"median={pct(0.50):.3f}  p75={pct(0.75):.3f}  max={allp[-1]:.3f}")
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            print(f"  tokens below {t:.1f}: {sum(1 for p in allp if p < t):4d} "
                  f"({100 * sum(1 for p in allp if p < t) / len(allp):.1f}%)")

    Path("data/confidence_probe_last.json").write_text(
        json.dumps({"jobId": job_id, "clip": clip_no, "segments": rows}, indent=1),
        encoding="utf-8",
    )
    print("\nfull record: data/confidence_probe_last.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
