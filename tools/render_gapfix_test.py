"""E2E test render of one Spanish and one English clip through the modified pipeline.

Proves the Ssemble-style framing + caption changes end to end ON REAL FILES:
  * framing stage: per-shot FIT/FILL decisions, logged with reasons
  * caption stage: short-burst bilingual cues with emphasis
  * render stage: FIT blurred-pad ffmpeg graph, Montserrat captions, hook title

The Spanish clip cuts come from the completed 4a2dda05… batch (a real job whose
transcript is recorded); the English test window uses the JFK source
(5f7c01456333) with its own recorded transcript.

Both burned-in outputs land in /tmp/gapfix-test/.

Run:  .venv/bin/python tools/render_gapfix_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_config
from app import subtitle, translate, reframe, encode, video

OUT = Path("/tmp/gapfix-test")
OUT.mkdir(exist_ok=True)


def render_one(label: str, src: Path, start: float, end: float,
               transcript_path: str, lang: str, offset: float = 0.0) -> None:
    print(f"\n=== {label}: {start:.1f}s-{end:.1f}s ===")
    t0 = time.time()
    # 1) framing stage (TARGET A). Log callback wired so the FIT/FILL per-shot
    # decision prints live.
    clip = {"start": start, "end": end, "_id": f"test_{label}", "title": None}
    cuts = reframe.find_cuts(src, None, [clip], None, on_log=print)
    cut = cuts[0]
    segs = cut.get("framingSegments") or []
    print(f"shots: {len(segs)}")
    for s in segs:
        print(f"  {s['start']:.1f}-{s['end']:.1f}s {s['mode'].upper()} — {s['reason']}")

    # 2) caption stage (TARGET B/C) on the REAL transcript
    data = json.load(open(transcript_path))
    class W:  # noqa: N801 - tiny shim
        def __init__(s, d):
            s.start = d["start"]; s.end = d["end"]
            s.word = d["word"]; s.conf = d.get("conf", 1.0)
    class T:
        words = None
        duration = data.get("duration", 0)
        language = data.get("language", lang)
    # words are recorded in SOURCE-absolute time; the segment-relative clip
    # window starts at `offset`, so shift the words into the same space.
    T.words = [W({**d, "start": d["start"] - offset, "end": d["end"] - offset})
               for d in data["words"]]
    cues = translate.build_cues(T, cut, None, language=lang, style="viral")
    print(f"cues: {len(cues)}")
    for c in cues[:10]:
        emph = ",".join(f'{e["word"]}={e["color"]}' for e in c.get("emphasis") or [])
        print(f"  {c['start']:.2f}-{c['end']:.2f} {c['text']!r} {('[' + emph + ']') if emph else ''}")

    # 3) render stage
    dest = OUT / f"{label}.mp4"
    settings = load_config()
    meta = encode.render_clip(
        src, cut, dest, settings,
        cues=cues,
        preset_key="viral",
        on_log=print,
    )
    got = video.probe(dest)
    print(
        f"RENDERED {dest.name}: {got['width']}x{got['height']}, "
        f"{got['duration']:.1f}s in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    # Spanish source is a saved 1170-1210s SEGMENT of the original video, so
    # the real absolute window 1176.36s maps to segment-relative 6.36s.
    ES_OFFSET = 1170.0
    es_src = Path("/tmp/es_source.mp4")
    en_src = Path("/home/daytona/clipforge-data/downloads/5f7c01456333/source.mp4")
    render_one("es", es_src, 1176.36 - ES_OFFSET, 1200.0 - ES_OFFSET,
               "/tmp/test_transcript_es.json", "es", offset=ES_OFFSET)
    render_one("en", en_src, 5.0, 30.0,
               "data/work/5f7c01456333/transcribe/words.json", "en")
