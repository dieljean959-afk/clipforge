#!/usr/bin/env python3
"""Which whisper model actually FITS here, and what changes?

The captions in this batch were burned in with the `small` model, because
`medium` needs more memory than the container has. That is the accuracy
ceiling, so this tool measures the alternatives instead of guessing:

  * does the model load, and what is the peak resident set (VmHWM) afterwards
  * how long does a real clip window take
  * what does it actually transcribe for that window

Run:  .venv/bin/python tools/model_accuracy_probe.py small turbo
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.asr import ANTI_REPETITION  # noqa: E402
from app.config import ffmpeg_bin, load_config  # noqa: E402
from app.store import LocalStore  # noqa: E402

JOB = "f3b81a1dd7e6"
CLIP = 2
WINDOW_SEC = 40


def peak_rss_gib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def window_wav(cfg, store) -> Path:
    clip = sorted(store.list_clips(JOB), key=lambda c: c.get("order", 0))[CLIP - 1]
    src = store.find_source_file(JOB)
    wav = Path(tempfile.mkdtemp(prefix="model-probe-")) / "w.wav"
    subprocess.run(
        [ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{float(clip['start']):.3f}",
         "-t", str(WINDOW_SEC), "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)],
        check=True, capture_output=True,
    )
    return wav


def main() -> int:
    models = sys.argv[1:] or ["small", "turbo"]
    cfg = load_config(quiet=True)
    store = LocalStore(cfg.app.paths)
    wav = window_wav(cfg, store)
    budget = 1.8  # GiB: the container limit times the safety factor

    from faster_whisper import WhisperModel

    for name in models:
        t0 = time.time()
        try:
            model = WhisperModel(name, device="cpu", compute_type="int8")
        except Exception as exc:
            print(f"{name:12} LOAD FAILED: {exc}")
            continue
        load_s = time.time() - t0
        t1 = time.time()
        try:
            segments, _ = model.transcribe(
                str(wav), language=cfg.whisper.language or None, task="transcribe",
                beam_size=5, word_timestamps=True, **ANTI_REPETITION,
            )
            text = " ".join((s.text or "").strip() for s in segments).strip()
        except Exception as exc:
            print(f"{name:12} TRANSCRIBE FAILED: {exc}")
            continue
        secs = time.time() - t1
        peak = peak_rss_gib()
        verdict = "FITS" if peak < budget else "TOO BIG"
        print(f"{name:12} load {load_s:5.1f}s  transcribe {secs:5.1f}s  "
              f"peak RSS {peak:4.2f} GiB  [{verdict}]")
        print(f"             {text[:220]}")
        print()
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
