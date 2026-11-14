"""Proof driver: transcribe a long source, resuming from chunk checkpoints.

Run under a hard cap to reproduce the user's host:

    bash -c 'ulimit -v 1536000; .venv/bin/python tools/asr_proof.py MEDIA'

Each invocation decodes what it can inside the terminal's time budget; finished
chunks are checkpointed to disk, so the next invocation skips them. Prints one
line per chunk with the model, peak RSS and wall time, then the word count.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataclasses import replace  # noqa: E402

from app import membudget  # noqa: E402
from app.config import load_config  # noqa: E402
from app.subtitle import transcribe  # noqa: E402

STATE = ROOT / "outputs" / "asr-proof" / "state.json"
CHUNKS = ROOT / "outputs" / "asr-proof" / "chunks"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: asr_proof.py MEDIA")
        return 2
    media = Path(sys.argv[1])
    settings = load_config(quiet=True)
    paths = replace(settings.app.paths, data_dir=ROOT / "outputs" / "asr-proof" / "data")
    settings = replace(settings, app=replace(settings.app, paths=paths))
    budget = membudget.current_budget()
    budget_s = f"BUDGET {budget.describe()}"

    print(budget_s, flush=True)
    cached = len(list(CHUNKS.glob("chunk_*.json"))) if CHUNKS.exists() else 0
    if cached:
        print(f"RESUME {cached} chunk checkpoint(s) on disk", flush=True)

    started = time.time()
    done: list[str] = []

    def on_chunk(index: int, total: int, *, phase: str, resuming: bool = False) -> None:
        if phase == "start":
            print(f"CHUNK {index + 1}/{total} start cached={_cached(index)}", flush=True)

    def _cached(index: int) -> bool:
        return (CHUNKS / f"chunk_{index:05d}.json").exists()

    try:
        transcript = transcribe(
            media, settings,
            cache_dir=CHUNKS,
            on_chunk=on_chunk,
            on_detail=lambda text: print(f"DETAIL {text}", flush=True),
        )
    except Exception as exc:  # noqa: BLE001 - proof driver reports, never hides
        cached_now = len(list(CHUNKS.glob("chunk_*.json"))) if CHUNKS.exists() else 0
        print(f"INCOMPLETE after {time.time() - started:.0f}s — {cached_now} chunk(s) saved", flush=True)
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1

    print("\nPER-CHUNK (span, model, ok, peak RSS, wall)", flush=True)
    for a in transcript.attempts:
        print(
            f"  {a.get('start')}-{a.get('end')}s  {a.get('model'):<8} "
            f"ok={str(a.get('ok')):<5} peak={a.get('peakRssMb'):>6} MB  "
            f"{a.get('seconds')}s"
            + (f"  [{a.get('error')[:60]}]" if not a.get("ok") else ""),
            flush=True,
        )
    peak = max([a.get("peakRssMb") or 0 for a in transcript.attempts] or [0])
    models = sorted({a.get("model") for a in transcript.attempts if a.get("ok")})
    state = {
        "duration": transcript.duration,
        "words": len(transcript.words),
        "model": transcript.model_size,
        "models_used": models,
        "label": transcript.label,
        "peak_rss_mb": peak,
        "attempts": transcript.attempts,
        "budget": budget_s,
        "seconds": round(time.time() - started, 1),
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps({k: state[k] for k in ("duration", "words", "model", "models_used",
                                            "label", "peak_rss_mb", "seconds")}, indent=2))
    print("FIRST:", " ".join(w.word for w in transcript.words[:14]), flush=True)
    print("LAST:", " ".join(w.word for w in transcript.words[-10:]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
