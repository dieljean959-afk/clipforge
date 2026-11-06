# LOCKED: do not modify — see LOCKED_SPEC.md §1 (ONE caption style).
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

import hashlib
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
    "words_per_cue": (1, 8, 1),
    "words_per_cue_max": (1, 10, 1),
    "pause_sec": (0.0, 1.5, 0.05),
    # The size knob: 80-140 % of the preset's own size. Font size, outline and
    # line pitch all move by this one number.
    "size_scale": (0.8, 1.4, 0.05),
    # "Custom Y": where the caption block sits, as a fraction of frame height.
    "position_y": (0.30, 0.95, 0.01),
}

COLOR_KEY = "highlight_color"
PRESET_KEY = "preset_key"
EMOJI_ENABLED_KEY = "emoji_enabled"
EMOJI_DENSITY_KEY = "emoji_density"
POSITION_MODE_KEY = "position_mode"

#: "video_bottom" anchors the block to the centred video's lower edge (the
#: blur layout), "frame" uses position_y of the frame, "custom" is an explicit
#: position_y the user dragged. All three stay supported.
POSITION_MODES = ("video_bottom", "frame", "custom")
EMOJI_DENSITIES = ("low", "medium", "high")


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
    # Strings and flags: validated against a closed set so a hand-edited file
    # cannot point the renderer at a preset or a density that does not exist.
    from app.caption_presets import CAPTION_PRESETS  # local: no import cycle

    preset = values.get(PRESET_KEY)
    if preset is not None:
        text = str(preset).strip()
        if text in CAPTION_PRESETS:
            out[PRESET_KEY] = text
    density = values.get(EMOJI_DENSITY_KEY)
    if density is not None:
        text = str(density).strip().lower()
        if text in EMOJI_DENSITIES:
            out[EMOJI_DENSITY_KEY] = text
    mode = values.get(POSITION_MODE_KEY)
    if mode is not None:
        text = str(mode).strip().lower()
        if text in POSITION_MODES:
            out[POSITION_MODE_KEY] = text
    if EMOJI_ENABLED_KEY in values and values[EMOJI_ENABLED_KEY] is not None:
        out[EMOJI_ENABLED_KEY] = bool(values[EMOJI_ENABLED_KEY])
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
    """Settings with the stored overrides applied — for the RENDERER.

    Returns a NEW Settings; the caller's object is untouched, so a job already
    rendering cannot have its captions changed underneath it. Returns the
    original object when nothing is stored, which is the common case and costs
    one file read.
    """
    resolved = resolve_caption_settings(settings)
    sub = replace(
        settings.subtitle,
        words_per_cue=int(resolved["wordsPerCue"]),
        words_per_cue_max=int(resolved["wordsPerCueMax"]),
        pause_sec=float(resolved["pauseSec"]),
        size_scale=float(resolved["sizeScale"]),
        highlight_color=str(resolved["highlightColor"] or ""),
        position_y=float(resolved["positionY"]),
        emoji_enabled=bool(resolved["emojiEnabled"]),
        emoji_density=str(resolved["emojiDensity"]),
        font_size=int(resolved["fontSize"]),
        preset=str(resolved["presetKey"]),
    )
    return replace(settings, subtitle=sub)


def resolve_caption_settings(settings: Settings, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """THE resolved caption object: preset defaults + slider overrides.

    One function, three consumers — the API/DTO, the live preview and the
    renderer. If they ever disagree about what a caption looks like, it is
    because one of them stopped calling this.

    `overrides` lets a caller resolve a DRAFT (the dialog's unsaved state)
    through the same path as the saved one, so the preview cannot show a
    different style from the render.
    """
    from app.caption_presets import CAPTION_PRESETS, PRESET_ORDER

    stored = clean(read(settings.app.paths)) if overrides is None else clean(overrides)
    preset_key = str(stored.get(PRESET_KEY) or settings.subtitle.preset or "bold_stack")
    if preset_key not in CAPTION_PRESETS:
        preset_key = "bold_stack"
    preset = CAPTION_PRESETS[preset_key]

    def pick(name: str, base: Any) -> Any:
        value = stored.get(name)
        return base if value is None else value

    highlight = stored.get(COLOR_KEY)
    if highlight is None:
        highlight = settings.subtitle.highlight_color
    highlight = str(highlight or "").strip()
    accent = highlight or preset.secondary

    size_scale = float(pick("size_scale", 1.0))
    if not (0.8 <= size_scale <= 1.4):
        size_scale = 1.0
    words = int(pick("words_per_cue", preset.words_per_cue))
    words_max = int(pick("words_per_cue_max", preset.words_per_cue_max))
    words = max(1, min(words, words_max))
    position_mode = str(pick(POSITION_MODE_KEY, preset.position_mode))
    position_y = float(pick("position_y", preset.position_y))
    if position_mode == "custom":
        mode = "custom"
    else:
        mode = position_mode if position_mode in POSITION_MODES else preset.position_mode
        if mode == "custom":
            mode = preset.position_mode
    resolved = {
        "presetKey": preset_key,
        "label": preset.label,
        "fontFamily": preset.font_override or settings.subtitle.font,
        "fontFile": preset.font_file,
        "fontSize": int(round(preset.font_size * size_scale)),
        "outline": int(round(preset.outline_width * size_scale)),
        "shadow": int(preset.shadow),
        "box": bool(preset.box),
        "uppercase": bool(preset.uppercase),
        "primaryColor": preset.primary,
        "highlightColor": accent,
        "highlightIsCustom": bool(highlight),
        "colors": list(preset.colors),
        "lines": int(preset.line_layout),
        # A 2-line preset stores its own pitch. A 1-line preset stores 0 (it
        # has no second line), and handing 0 to the preview produced a block
        # with both lines on top of each other — so fall back to the leading
        # the font itself implies.
        "pitch": int(round((preset.pitch or preset.font_size * 1.16) * size_scale)),
        "maxCharsPerLine": int(preset.max_chars_per_line),
        "wordsPerCue": words,
        "wordsPerCueMax": words_max,
        "pauseSec": round(float(pick("pause_sec", preset.pause_sec)), 2),
        "positionY": round(position_y, 4),
        "positionMode": mode,
        "animation": preset.animation,
        "emojiEnabled": bool(pick(EMOJI_ENABLED_KEY, settings.subtitle.emoji_enabled)),
        "emojiDensity": str(pick(EMOJI_DENSITY_KEY, settings.subtitle.emoji_density)),
        "sizeScale": round(size_scale, 2),
    }

    # "custom" means THE VALUE DIFFERS FROM THE PRESET — not "the key is
    # present in the file". Otherwise re-saving an unchanged slider lights up
    # the badge for a look that is still exactly the preset.
    preset_defaults = {
        "words_per_cue": preset.words_per_cue,
        "words_per_cue_max": preset.words_per_cue_max,
        "pause_sec": round(preset.pause_sec, 2),
        "size_scale": 1.0,
        # "not custom" means "the preset's own accent", whether the user left
        # the field empty or typed the preset's colour back in.
        "highlight_color": preset.secondary,
        "position_mode": preset.position_mode,
        "position_y": round(preset.position_y, 4),
        "emoji_enabled": settings.subtitle.emoji_enabled,
        "emoji_density": settings.subtitle.emoji_density,
    }
    current = {
        "words_per_cue": words,
        "words_per_cue_max": words_max,
        "pause_sec": resolved["pauseSec"],
        "size_scale": round(size_scale, 2),
        "highlight_color": accent,
        "position_mode": resolved["positionMode"],
        "position_y": resolved["positionY"],
        "emoji_enabled": resolved["emojiEnabled"],
        "emoji_density": resolved["emojiDensity"],
    }
    custom = [k for k, v in current.items() if v != preset_defaults[k]]

    resolved.update({
        # snake_case mirrors so older call sites keep reading what they read.
        "words_per_cue": resolved["wordsPerCue"],
        "words_per_cue_max": resolved["wordsPerCueMax"],
        "pause_sec": resolved["pauseSec"],
        "size_scale": resolved["sizeScale"],
        "highlight_color": accent,
        "overrides": stored,
        "custom": custom,
        "differFromPreset": bool(custom),
        "presetKeys": list(PRESET_ORDER),
        "limits": {k: {"min": v[0], "max": v[1], "step": v[2]} for k, v in TUNABLES.items()},
        "emojis": {"enabled": resolved["emojiEnabled"], "density": resolved["emojiDensity"]},
        "hash": caption_hash(resolved),
    })
    return resolved


def caption_hash(resolved: dict[str, Any]) -> str:
    """Short hash of everything that changes the burned-in pixels.

    This is the render CACHE KEY. Without it, "Re-render captions" skipped a
    clip whose style had changed (the file was already there and marked done).
    """
    keys = (
        "presetKey", "fontFamily", "fontSize", "outline", "shadow", "box",
        "uppercase", "primaryColor", "highlightColor", "lines", "pitch",
        "maxCharsPerLine", "wordsPerCue", "wordsPerCueMax", "pauseSec",
        "positionY", "positionMode", "animation", "emojiEnabled", "emojiDensity",
    )
    payload = "|".join(f"{k}={resolved.get(k)!r}" for k in keys)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def effective(settings: Settings) -> dict[str, Any]:
    """What the next render will actually use — the resolved object."""
    return resolve_caption_settings(settings)
