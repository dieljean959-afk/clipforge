#!/usr/bin/env python3
"""Transcription speed profile on a REAL job source slice (no fake data).

Extracts a ~120s slice of an existing download, transcribes it with the
configured (medium) model at ctranslate2's default cpu_threads=4 and again
with cpu_threads=8, and reports wall time + realtime factor. Accuracy-neutral
experiment: identical model and decode kwargs, only CPU thread count changes.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ffmpeg_bin, load_config
from app.subtitle import WHISPER_KWARGS, _extract_media

SRC = Path("downloads/885ba6a7ac26/source.mp4")
SLICE_SEC = 120


def make_slice(dest: Path) -> Path:
    subprocess.run(
        [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(SRC), "-t", str(SLICE_SEC),
         "-c", "copy", str(dest)],
        check=True, capture_output=True,
    )
    return dest


def main() -> int:
    from faster_whisper import WhisperModel

    settings = load_config()
    if not SRC.exists():
        print(f"SKIP: {SRC} not on disk")
        return 0
    slice_path = make_slice(Path("/tmp/profile_slice.mp4"))
    workdir = Path("/tmp/profile_audio")
    wav = _extract_media(slice_path, 0.0, SLICE_SEC, workdir)
    print(f"slice: {SLICE_SEC}s audio at {wav}")

    for threads in (4, 8):
        model = WhisperModel(
            settings.whisper.model,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
        )
        t0 = time.time()
        segments, info = model.transcribe(str(wav), language=None, **WHISPER_KWARGS)
        n_words = 0
        for seg in segments:
            n_words += len(seg.words or [])
        dt = time.time() - t0
        print(
            f"cpu_threads={threads}: {dt:.1f}s wall for {info.duration:.0f}s audio "
            f"({info.duration / dt:.1f}x realtime), {n_words} words"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
