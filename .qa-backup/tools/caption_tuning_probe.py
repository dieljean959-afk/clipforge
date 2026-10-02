"""Does a style saved from the dashboard actually change the render?

The unit checks prove resolve() and the fingerprint behave. This proves the
whole chain on REAL data: a saved burst length and highlight colour, run
through the same call the render stage makes on a real clip's real words, and
compared against the same clip rendered with the defaults.

Run:  .venv/bin/python tools/caption_tuning_probe.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import caption_style, subtitle  # noqa: E402
from app.config import Paths, load_config  # noqa: E402
from app.store import LocalStore  # noqa: E402

JOB = "fd961148baa5"


def render(words: list[dict], settings, start: float) -> list[dict]:
    """Exactly app.task._stage_render's caption path for one clip."""
    kept = subtitle.normalize_word_timings(words)
    kept, _ = subtitle.strip_fillers(kept, drop_stutters=settings.subtitle.strip_stutters)
    cues = subtitle.words_to_cues(
        kept,
        words_per_cue=settings.subtitle.words_per_cue,
        max_words=settings.subtitle.words_per_cue_max,
        pause_sec=settings.subtitle.pause_sec,
        confidence_threshold=settings.subtitle.low_confidence,
        inaudible_threshold=settings.subtitle.inaudible_below,
    )
    for c in cues:
        c["start"] -= start
        c["end"] -= start
    return cues


def summarise(label: str, cues: list[dict], settings) -> None:
    n = [len(c["words"]) for c in cues]
    print(f"  {label:<26} {len(cues):>3} cues | "
          f"{sum(n) / max(1, len(n)):.2f} words/cue | "
          f"min {min(n, default=0)} max {max(n, default=0)} | "
          f"style {subtitle.style_fingerprint(settings, settings.subtitle.preset)}")


def main() -> int:
    settings = load_config(quiet=True)
    words = json.loads(
        (ROOT / "data" / "work" / JOB / "transcribe" / "words.json").read_text("utf-8")
    )["words"]
    clip = LocalStore(settings.app.paths).list_clips(JOB)[0]
    c0, c1 = float(clip["start"]), float(clip["end"])
    window = [w for w in words if float(w["end"]) > c0 and float(w["start"]) < c1]

    saved = caption_style.read(settings.app.paths)
    tmp = Path(tempfile.mkdtemp(prefix="cf-probe-"))
    try:
        iso_paths = Paths(
            downloads_dir=settings.app.paths.downloads_dir,
            outputs_dir=settings.app.paths.outputs_dir,
            data_dir=tmp / "data",
            static_dir=settings.app.paths.static_dir,
        )
        iso = replace(settings, app=replace(settings.app, paths=iso_paths))
        iso_paths.data_dir.mkdir(parents=True, exist_ok=True)

        print(f"job {JOB} clip 1 — {len(window)} words over {c1 - c0:.1f}s")
        print()
        caption_style.write(iso_paths, {})
        base_settings = caption_style.resolve(iso)
        base = render(window, base_settings, c0)
        summarise("config defaults", base, base_settings)

        trials = [
            ({"words_per_cue": 1, "words_per_cue_max": 1}, "burst: 1 word"),
            ({"words_per_cue": 4, "words_per_cue_max": 4}, "burst: 4 words"),
            ({"pause_sec": 1.0}, "pause: 1.00s"),
            ({"pause_sec": 0.0}, "pause: off (0.00s)"),
            ({"highlight_color": "#FF5C39"}, "highlight: orange"),
        ]
        for overrides, label in trials:
            caption_style.write(iso_paths, overrides)
            tuned = caption_style.resolve(iso)
            cues = render(window, tuned, c0)
            summarise(label, cues, tuned)
            if overrides.get("highlight_color"):
                header = subtitle.build_ass_header(tuned, 1080, 1920, "bold_pop")
                line = [ln for ln in header.splitlines() if ln.startswith("Style")][0]
                fields = line.split(",")
                # fields[0]="Style: ClipForge", 1=font, 2=size, 3=primary,
                # 4=secondary (the sung word), 5=outline.
                print(f"      style line: {line[:96]}")
                print(f"      sung word is orange: {fields[4] == '&H00395CFF'}  "
                      f"and not green: {fields[4] != '&H0014FF39'}  "
                      f"outline still green: {fields[5] == '&H0014FF39'}")

        # The same clip, same words, same encoder call — only the style differs.
        caption_style.write(iso_paths, {})
        after = render(window, caption_style.resolve(iso), c0)
        same = [c["text"] for c in after] == [c["text"] for c in base]
        print()
        print(f"  clearing the overrides reproduces the default render exactly: {same}")
        return 0 if same else 1
    finally:
        caption_style.write(settings.app.paths, saved)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
