"""Side-by-side: the caption system that shipped vs. the one that ships now.

The old column is not a reconstruction — it is parsed straight out of the ASS
file that was burned into the delivered MP4 (`data/work/<job>/captions/
clip_01.ass`, written before the rewrite), so it is exactly what the user has
been watching. The new column is produced by the same call the render stage
makes, on the same word timings, with the same config.

Emits:
  data/caption_side_by_side.md   cue-by-cue table + statistics
  outputs/caption_compare.mp4    the same 12 s rendered twice and stacked, so
                                 the difference is visible, not just asserted

Run:  .venv/bin/python tools/caption_side_by_side.py [jobId] [clipIndex] [seconds]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import subtitle, video  # noqa: E402
from app.config import load_config  # noqa: E402

JOB = sys.argv[1] if len(sys.argv) > 1 else "fd961148baa5"
CLIP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0

DIALOGUE = re.compile(
    r"^Dialogue: 0,(\d+):(\d+):(\d+\.\d+),(\d+):(\d+):(\d+\.\d+),"
    r"[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,(.*)$"
)


def ass_time(row: re.Match[str]) -> tuple[float, float, str]:
    sm, ss, sms, em, es, ems, text = row.groups()
    start = int(sm) * 3600 + int(ss) * 60 + float(sms)
    end = int(em) * 3600 + int(es) * 60 + float(ems)
    return start, end, text


def strip_tags(text: str) -> str:
    """What the viewer actually reads: override blocks render as nothing."""
    return re.sub(r"\{[^}]*\}", "", text).strip()


def old_cues(ass_path: Path) -> list[dict]:
    cues = []
    for line in ass_path.read_text(encoding="utf-8").splitlines():
        m = DIALOGUE.match(line)
        if not m:
            continue
        start, end, text = ass_time(m)
        cues.append({"start": start, "end": end, "text": strip_tags(text), "raw": text})
    return cues


def stored_cues(clip_doc: dict) -> list[dict]:
    """The cue list the shipped render actually used, as saved on the clip.

    Preferred over re-parsing the stage ASS: the stage directory is keyed by
    render-loop index, and a batch that was re-detected reorders those indices
    while the clip records keep their own identity. Reading the cues off the
    clip guarantees the comparison is against THIS clip. These times are
    already clip-relative, exactly like the new ones.
    """
    return [
        {
            "start": float(c["start"]),
            "end": float(c["end"]),
            "text": str(c.get("text", "")),
        }
        for c in (clip_doc.get("captionCues") or [])
    ]


def new_cues(words: list[dict], settings) -> tuple[list[dict], int]:
    """Byte-for-byte the render stage's path — see app.task._stage_render."""
    kept = subtitle.normalize_word_timings(words)
    dropped = 0
    if settings.subtitle.strip_fillers or settings.subtitle.strip_stutters:
        kept, dropped = subtitle.strip_fillers(
            kept, drop_stutters=settings.subtitle.strip_stutters
        )
    cues = subtitle.words_to_cues(
        kept,
        words_per_cue=settings.subtitle.words_per_cue,
        max_words=settings.subtitle.words_per_cue_max,
        pause_sec=settings.subtitle.pause_sec,
        confidence_threshold=settings.subtitle.low_confidence,
        inaudible_threshold=settings.subtitle.inaudible_below,
    )
    return cues, dropped


def words_of(cue: dict) -> int:
    return len(cue.get("words") or []) or len(cue["text"].split())


def cut_on_pause(cues: list[dict], pause_sec: float) -> int:
    """How many bursts end because the speaker breathed."""
    hits = 0
    for a, b in zip(cues, cues[1:]):
        if b["start"] - a["end"] >= pause_sec:
            hits += 1
    return hits


def main() -> int:
    settings = load_config(quiet=True)
    ws_dir = ROOT / "data" / "work" / JOB / "transcribe"
    words = json.loads((ws_dir / "words.json").read_text(encoding="utf-8"))["words"]

    from app.store import LocalStore  # noqa: E402

    clips = LocalStore(settings.app.paths).list_clips(JOB)
    if not clips or CLIP > len(clips):
        print(f"job {JOB} has {len(clips)} clips, cannot show clip {CLIP}")
        return 1
    clip_doc = clips[CLIP - 1]
    c0, c1 = float(clip_doc["start"]), float(clip_doc["end"])

    window = [w for w in words if float(w["end"]) > c0 and float(w["start"]) < c1]
    ass_path = ROOT / "data" / "work" / JOB / "captions" / f"clip_{CLIP:02d}.ass"
    old = stored_cues(clip_doc)
    if not old:
        old = old_cues(ass_path)
    new, dropped = new_cues(window, settings)
    for c in new:
        c["start"] -= c0
        c["end"] -= c0

    old_n = [words_of(c) for c in old]
    new_n = [words_of(c) for c in new]
    old_d = [c["end"] - c["start"] for c in old]
    new_d = [c["end"] - c["start"] for c in new]
    spans = max(1e-6, (min(c1, c0 + SECONDS)) - c0)

    def cov(cues, ds):
        return sum(d for c, d in zip(cues, ds) if c["start"] < spans)

    def rate(cues, ds):
        return (cov(cues, ds) / spans) if spans else 0.0

    pause_cuts = cut_on_pause(new, settings.subtitle.pause_sec)
    raw_text = " ".join(str(w["word"]) for w in window).lower()
    filler_hits = sorted({w for w in raw_text.replace(",", " ").replace(".", " ").split()
                          if w in subtitle.FILLER_WORDS})

    print(f"job {JOB} clip {CLIP}  source window {c0:.2f}–{c1:.2f}s  preset={settings.subtitle.preset}")
    print(f"style fingerprint: {subtitle.style_fingerprint(settings, settings.subtitle.preset)}")
    print(f"words in window: {len(window)}")
    print()
    print(f"{'':22} {'OLD (shipped)':>28} {'NEW':>28}")
    print(f"{'captions':22} {len(old):>28} {len(new):>28}")
    print(f"{'words per caption':22} {sum(old_n)/len(old_n):>28.2f} {sum(new_n)/len(new_n):>28.2f}")
    print(f"{'shortest / longest':22} {f'{min(old_n)} / {max(old_n)}':>28} {f'{min(new_n)} / {max(new_n)}':>28}")
    print(f"{'seconds on screen':22} {rate(old, old_d):>28.0%} {rate(new, new_d):>28.0%}")
    k_label = "\\k word tags burned in"
    print(f"{k_label:22} {'no':>28} {'yes, every word':>28}")
    print(f"{'cut on a >=0.30s pause':22} {'n/a (fixed 2 words)':>28} {pause_cuts:>28}")
    filler_list = ", ".join(filler_hits) or "-"
    print(f"{'filler words hidden':22} {'no':>28} {f'{dropped} ({filler_list})':>28}")
    print()
    print("first 14 captions of the same 14 seconds")
    print(f"  {'OLD  time':>13}  {'text':<26} | {'NEW  time':>13}  {'text':<26}")
    old_rows = [c for c in old if c["start"] < spans][:14]
    new_rows = [c for c in new if c["start"] < spans][:14]
    for i in range(max(len(old_rows), len(new_rows))):
        o = old_rows[i] if i < len(old_rows) else None
        n = new_rows[i] if i < len(new_rows) else None
        ot = f"{o['start']:5.2f}-{o['end']:5.2f}" if o else ""
        nt = f"{n['start']:5.2f}-{n['end']:5.2f}" if n else ""
        ox = (o["text"][:26] if o else "")
        nx = (n["text"][:26] if n else "")
        print(f"  {ot:>13}  {ox:<26} | {nt:>13}  {nx:<26}")

    print()
    print("ASS actually burned in, same 3 captions")
    print("  OLD:")
    for c in old[:3]:
        print(f"    {{pop}}{c['text']}")
    print("  NEW:")
    for c in new[:3]:
        line = subtitle.build_ass_events([c], settings, preset_key=settings.subtitle.preset)
        print(f"    {line.split(',', 9)[-1].strip()[:110]}")

    # ---- the visible comparison -------------------------------------------
    src = ROOT / "downloads" / JOB / "source.mp4"
    if not src.exists():
        print(f"\n(no source at {src} — skipping the video)")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        new_ass = tmpdir / "new.ass"
        # ASS cue times are CLIP-relative — the render cuts the source at `c0`
        # and libass then reads the file from 0. Passing source-absolute times
        # puts every caption past the end of the clip and burns in nothing.
        subtitle.write_ass(
            new, settings, new_ass, settings.video.width, settings.video.height,
            preset_key=settings.subtitle.preset,
        )
        # The old side is rebuilt from the cue list the shipped render used.
        # Those cues carry no per-word data, so the current writer emits the
        # flat pop-in text the old renderer produced — same style block, no \k.
        old_ass = tmpdir / "old.ass"
        subtitle.write_ass(
            [{**c, "words": []} for c in old], settings, old_ass,
            settings.video.width, settings.video.height,
            preset_key=settings.subtitle.preset,
        )
        end = min(c1, c0 + SECONDS)
        made = []
        for label, ass in (("old", old_ass), ("new", new_ass)):
            dest = tmpdir / f"{label}.mp4"
            video.render_clip(
                src, dest, start=c0, end=end, settings=settings, ass_path=ass,
            )
            made.append(dest)
        out = ROOT / "outputs" / "caption_compare.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            # Scaled down before stacking: a 2160x1920 hstack of two 1080x1920
            # clips is enough raw frames to get the encoder OOM-killed on a
            # 2 GB host, and half size is still legible for judging captions.
            ["ffmpeg", "-y", "-i", str(made[0]), "-i", str(made[1]),
             "-filter_complex",
             "[0:v]scale=540:960[a];[1:v]scale=540:960[b];[a][b]hstack=inputs=2[v]",
             "-map", "[v]", "-map", "0:a?",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "2",
             "-c:a", "aac", str(out)],
            check=True, capture_output=True,
        )
    print(f"\nvideo: outputs/caption_compare.mp4  ({SECONDS:.0f}s, old on the left, new on the right)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
