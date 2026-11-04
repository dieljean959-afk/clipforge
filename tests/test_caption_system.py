"""Caption settings, cache key and emoji-selection rules.

These are the guarantees the dashboard promises: a slider that changes the
render, a re-render that never serves a stale clip, and an emoji track that
comes from the whitelist with the density/cooldown rules actually applied.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app import caption_style, emoji
from app.caption_presets import CAPTION_PRESETS, PRESET_ORDER
from app.config import load_config


@pytest.fixture(scope="module")
def base():
    return load_config(quiet=True)


# --------------------------------------------------------------------------- #
# resolveCaptionSettings: preset defaults + slider overrides
# --------------------------------------------------------------------------- #

def test_every_preset_resolves_and_they_differ(base):
    assert len(PRESET_ORDER) == 7
    looks = {}
    for key in PRESET_ORDER:
        r = caption_style.resolve_caption_settings(base, {"preset_key": key})
        assert r["presetKey"] == key
        assert r["fontSize"] > 0 and r["outline"] >= 0 and r["pitch"] > 0
        looks[key] = (
            r["fontFamily"], r["fontSize"], r["outline"], r["box"],
            r["lines"], r["uppercase"], tuple(r["colors"]), r["positionMode"],
        )
    # The whole point of seven presets: no two render the same.
    assert len(set(looks.values())) == len(looks)


def test_size_scale_scales_font_outline_and_pitch_together(base):
    small = caption_style.resolve_caption_settings(base, {"preset_key": "bold_stack", "size_scale": 1.0})
    big = caption_style.resolve_caption_settings(base, {"preset_key": "bold_stack", "size_scale": 1.2})
    assert big["fontSize"] > small["fontSize"]
    assert big["pitch"] > small["pitch"]
    assert big["outline"] >= small["outline"]
    # +-1 px from rounding, not from a second formula.
    assert abs(big["fontSize"] / small["fontSize"] - 1.2) < 0.03
    assert abs(big["pitch"] / small["pitch"] - 1.2) < 0.03


def test_custom_badge_only_when_value_differs_from_preset(base):
    preset = CAPTION_PRESETS["bold_stack"]
    clean = caption_style.resolve_caption_settings(base, {"preset_key": "bold_stack"})
    assert clean["custom"] == []
    # Re-saving the preset's own default is NOT custom ...
    same = caption_style.resolve_caption_settings(
        base, {"preset_key": "bold_stack", "words_per_cue": preset.words_per_cue}
    )
    assert same["custom"] == []
    # ... a value that actually differs is.
    moved = caption_style.resolve_caption_settings(
        base, {"preset_key": "bold_stack", "words_per_cue": preset.words_per_cue + 1}
    )
    assert "words_per_cue" in moved["custom"]


def test_highlight_override_beats_the_preset_cycle(base):
    r = caption_style.resolve_caption_settings(
        base, {"preset_key": "bold_stack", "highlight_color": "#FF2E88"}
    )
    assert r["highlightColor"] == "#FF2E88"
    assert r["highlightIsCustom"] is True
    assert "highlight_color" in r["custom"]


# --------------------------------------------------------------------------- #
# cache key
# --------------------------------------------------------------------------- #

def test_hash_changes_with_any_rendered_value(base):
    a = caption_style.resolve_caption_settings(base, {"preset_key": "bold_stack"})["hash"]
    b = caption_style.resolve_caption_settings(
        base, {"preset_key": "bold_stack", "size_scale": 1.2}
    )["hash"]
    c = caption_style.resolve_caption_settings(base, {"preset_key": "viral"})["hash"]
    d = caption_style.resolve_caption_settings(
        base, {"preset_key": "bold_stack", "highlight_color": "#00FF00"}
    )["hash"]
    assert len({a, b, c, d}) == 4, "a cache key that ignores a setting serves a stale render"
    # Same inputs, same key (the resume path depends on it).
    assert a == caption_style.resolve_caption_settings(base, {"preset_key": "bold_stack"})["hash"]


def test_caption_hash_is_stable_across_process_ordering(base):
    one = caption_style.caption_hash({"a": 1, "b": 2})
    two = caption_style.caption_hash({"b": 2, "a": 1})
    assert one == two and len(one) >= 8


# --------------------------------------------------------------------------- #
# emoji: whitelist, cooldown, density
# --------------------------------------------------------------------------- #

def _cues(n: int, step: float = 2.5) -> list[dict]:
    return [
        {"start": round(i * step, 3), "end": round(i * step + 1.2, 3), "text": f"chunk {i}"}
        for i in range(n)
    ]


def _tags(n: int, cycle=("🔥", "💪", "😂", "🚀", None)) -> list[dict]:
    out = []
    for i in range(n):
        e = cycle[i % len(cycle)]
        out.append({
            "i": i, "emoji": e, "emotion": "joy" if e else "neutral",
            "intensity": 0.8, "confidence": 0.9, "sensitive": False,
        })
    return out


def test_whitelist_rejects_off_list_emoji(base):
    tags = emoji.parse_tags('[{"i":0,"emotion":"joy","emoji":"🦄","confidence":0.9}]', 1)
    # 🦄 is not in the bundled whitelist, so it can never reach the renderer.
    assert tags[0].emoji is None
    # parse keeps a whitelisted one
    ok = emoji.parse_tags('[{"i":0,"emotion":"joy","emoji":"🔥","confidence":0.9}]', 1)
    assert ok[0].emoji == "🔥"


def test_compound_emotion_label_keeps_its_first_known_value():
    # Models answer "pride|joy"; collapsing that to neutral silently dropped
    # the strongest chunk of the clip.
    tags = emoji.parse_tags('[{"i":0,"emotion":"pride|joy","emoji":"🔥","confidence":0.95}]', 1)
    assert tags[0].emotion == "pride"


def test_density_sets_the_share_of_chunks(base):
    cues = _cues(14)
    tags = _tags(14)
    counts = {
        d: len(emoji.select(tags, cues, density=d)) for d in ("low", "medium", "high")
    }
    assert counts["low"] < counts["medium"] < counts["high"]
    total = len(cues)
    assert counts["low"] <= round(total * 0.15) + 1
    assert counts["medium"] <= round(total * 0.35) + 1
    assert counts["high"] <= round(total * 0.60) + 1


def test_cooldown_rules(base):
    cues = _cues(10, step=1.0)          # 1 s apart: every other one is too close
    tags = _tags(10, cycle=("🔥", "💪"))
    picked = emoji.select(tags, cues, density="high")
    starts = [float(p["start"]) for p in picked]
    assert len(picked) >= 2
    for a, b in zip(starts, starts[1:]):
        assert b - a >= emoji.COOLDOWN_SEC
    # Never the same emoji twice in a row.
    for a, b in zip(picked, picked[1:]):
        assert a["emoji"] != b["emoji"]
    # And a clip where every chunk wants the SAME emoji gets exactly one use:
    # repeating it back to back is the thing the rule exists to prevent.
    same = emoji.select(_tags(10, cycle=("🔥",)), cues, density="high")
    assert len(same) == 1
    assert len(picked) <= round(len(cues) * 0.60) + 1


def test_low_confidence_and_neutral_are_skipped(base):
    cues = _cues(6)
    tags = _tags(6, cycle=("🔥",))
    tags[0]["confidence"] = 0.2
    tags[1]["emotion"] = "neutral"
    picked = emoji.select(tags, cues, density="high")
    assert all(p["i"] not in (0, 1) for p in picked)


def test_disabled_emoji_yields_nothing(base):
    assert emoji.select(_tags(6), _cues(6), density="high", enabled=False) == []
