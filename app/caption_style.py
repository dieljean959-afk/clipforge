"""Caption knobs the dashboard can change, without editing config.toml.

Three things decide whether the captions feel right, and all three used to be
buried in config.toml with no way to change them but editing a file on a
server: how long a burst is, where a burst boundary may fall, and what colour
the word being spoken lights up. They are exactly the three a person judging a
clip will want to try five values of, so they live here.

Why a file and not just an in-memory value: the setting has to survive a
restart, and the render runs in a child process, so what is persisted is the
only thing both agree on. The file holds ONLY the keys the user has actually
changed, so a value dropped from config.toml reappears the moment the override
is cleared instead of being frozen forever at whatever the UI last set.

Validation happens once, in `clean()`, and `resolve()` clamps rather than
raising: a corrupt or hand-edited file must never be able to take the render
stage down, because that is a 20-minute job per batch.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import Paths, Settings

#: name -> (low, high, step). `step` is what the slider moves by, and it is
#: also the rounding grid, so a saved 0.33 s pause comes back as 0.35 s rather
#: than as a number the slider cannot display.
TUNABLES: dict[str, tuple[float, float, float]] = {
    "font_size": (32, 160, 4),
    "words_per_cue": (1, 4, 1),
    "words_per_cue_max": (1, 6, 1),
    "pause_sec": (0.0, 1.5, 0.05),
    # The bold-stack size knob: 80-140 % of the reference's 54 px / 86 px
    # pitch. Font size, outline and line pitch all move by this one number.
    "size_scale": (0.8, 1.4, 0.05),
}

COLOR_KEY = "highlight_color"


def store_path(paths: Paths) -> Path:
    return paths.data_dir / "caption_style.json"


# --------------------------------------------------------------------------- #
# reading / writing
# --------------------------------------------------------------------------- #


def read(paths: Paths) -> dict[str, Any]:
    """The stored overrides. A missing or unreadable file is not an error."""
    p = store_path(paths)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"[caption-style] ignoring unreadable {p}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def write(paths: Paths, values: dict[str, Any]) -> dict[str, Any]:
    """Persist cleaned overrides and return what was actually stored."""
    cleaned = clean(values)
    p = store_path(paths)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(f"[caption-style] saved {cleaned or '(defaults)'}")
    return cleaned


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def clean(values: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys, clamp numbers to the slider range, keep a valid hex.

    Unknown keys are dropped rather than passed through: this dict ends up as
    a `dataclasses.replace` on SubtitleConfig, and a typo in a key name would
    otherwise raise inside the render stage, minutes into a job.
    """
    out: dict[str, Any] = {}
    for name, (lo, hi, step) in TUNABLES.items():
        if name not in values or values[name] is None:
            continue
        try:
            raw = float(values[name])
        except (TypeError, ValueError):
            continue
        snapped = round(round(raw / step) * step, 4)
        clamped = min(hi, max(lo, snapped))
        out[name] = int(clamped) if step == 1 else clamped
    color = values.get(COLOR_KEY)
    if color is not None:
        text = str(color).strip()
        if text == "":
            out[COLOR_KEY] = ""          # explicitly "back to the preset default"
        elif text.startswith("#") and len(text) in (4, 7):
            out[COLOR_KEY] = text
    # The cap below the target makes max_words dead config — the burst would
    # always stop at the target — so the target is pulled down instead of
    # storing a pair that cannot both be true.
    if out.get("words_per_cue", 1) > out.get("words_per_cue_max", 6):
        out["words_per_cue"] = out["words_per_cue_max"]
    return out


# --------------------------------------------------------------------------- #
# applying
# --------------------------------------------------------------------------- #


def resolve(settings: Settings) -> Settings:
    """Settings with the stored overrides applied.

    Returns a NEW Settings; the caller's object is untouched, so a job already
    rendering cannot have its captions changed underneath it. Returns the
    original object when nothing is stored, which is the common case and costs
    one file read.
    """
    overrides = clean(read(settings.app.paths))
    if not overrides:
        return settings
    sub = replace(settings.subtitle, **overrides)
    return replace(settings, subtitle=sub)


def effective(settings: Settings) -> dict[str, Any]:
    """What the next render will actually use — overrides, else config.toml.

    Read once: the UI needs both the merged value and the list of keys that
    are actually overridden, so it can tell "set to 3" apart from "inherited
    3 from config.toml" and only mark the sliders that are pinned.
    """
    overrides = clean(read(settings.app.paths))
    merged = replace(settings.subtitle, **overrides) if overrides else settings.subtitle
    return {
        "words_per_cue": merged.words_per_cue,
        "words_per_cue_max": merged.words_per_cue_max,
        "pause_sec": round(merged.pause_sec, 2),
        "size_scale": round(float(getattr(merged, "size_scale", 1.0)), 2),
        "highlight_color": merged.highlight_color,
        "overrides": overrides,
        "limits": {k: {"min": v[0], "max": v[1], "step": v[2]} for k, v in TUNABLES.items()},
    }
