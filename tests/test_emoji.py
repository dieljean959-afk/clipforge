"""Unit tests for the caption-emoji selection rules (app/emoji.py).

These cover the rules the prompt alone cannot be trusted to enforce: the
whitelist gate, the density cap, the cooldown, no-repeat / max-repeat, the
sensitive-content ban, and honest labelling when the model call fails.
"""

from __future__ import annotations

import json

import pytest

from app import emoji


def cue(i: int, start: float, text: str = "line", duration: float = 1.0) -> dict:
    return {"start": start, "end": start + duration, "text": text}


def tag(i: int, emoji_char: str | None, **kw) -> dict:
    base = {
        "i": i, "emotion": "joy", "intensity": 0.8, "emoji": emoji_char,
        "reason": "test", "confidence": 0.9, "sensitive": False,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# whitelist enforcement
# --------------------------------------------------------------------------- #

def test_whitelist_accepts_listed_emoji_and_rejects_others():
    assert emoji.is_allowed("😂")
    assert not emoji.is_allowed("🦄")     # not in the curated list
    assert not emoji.is_allowed(None)


def test_parse_tags_drops_off_whitelist_emoji():
    reply = json.dumps([{
        "i": 0, "emotion": "joy", "intensity": 0.9, "emoji": "🦄",
        "reason": "invented", "confidence": 0.99, "sensitive": False,
    }])
    tags = emoji.parse_tags(reply, 1)
    assert tags[0].emoji is None          # rejected, not passed through
    assert tags[0].confidence == 0.99     # the rest of the judgement survives


def test_every_whitelisted_emoji_has_a_bundled_png():
    wl = emoji.load_whitelist()
    missing = [e.emoji for e in wl.entries.values() if not e.path.exists()]
    assert missing == [], f"whitelist entries without a PNG: {missing}"


# --------------------------------------------------------------------------- #
# confidence + neutral
# --------------------------------------------------------------------------- #

def test_low_confidence_and_neutral_are_skipped():
    cues = [cue(0, 0.0), cue(1, 5.0), cue(2, 10.0)]
    tags = [
        tag(0, "😂", confidence=0.59),          # below the 0.6 floor
        tag(1, "😂", emotion="neutral"),        # neutral
        tag(2, "😂", confidence=0.6),           # kept
    ]
    picked = emoji.select(tags, cues, density="high")
    assert [p["i"] for p in picked] == [2]


# --------------------------------------------------------------------------- #
# density cap
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "density,expected_share", [("low", 0.15), ("medium", 0.35), ("high", 0.60)]
)
def test_density_cap(density, expected_share):
    cues = [cue(i, i * 5.0) for i in range(20)]
    # Alternate the emoji so the no-repeat rule cannot be what limits the count.
    tags = [tag(i, "😂" if i % 2 == 0 else "👏") for i in range(20)]
    picked = emoji.select(tags, cues, density=density)
    cap = round(len(cues) * expected_share)
    assert len(picked) <= cap
    assert len(picked) >= 1


def test_density_changes_how_many_survive():
    cues = [cue(i, i * 5.0) for i in range(20)]
    tags = [tag(i, "😂" if i % 2 == 0 else "👏") for i in range(20)]
    low = emoji.select(tags, cues, density="low")
    high = emoji.select(tags, cues, density="high")
    assert len(low) < len(high)


def test_disabled_track_selects_nothing():
    cues = [cue(i, i * 5.0) for i in range(4)]
    tags = [tag(i, "😂") for i in range(4)]
    assert emoji.select(tags, cues, density="high", enabled=False) == []


# --------------------------------------------------------------------------- #
# cooldown and repeats
# --------------------------------------------------------------------------- #

def test_cooldown_keeps_two_emojis_apart():
    cues = [cue(0, 1.0), cue(1, 2.0), cue(2, 3.5), cue(3, 8.0)]
    tags = [tag(0, "😂"), tag(1, "👏"), tag(2, "🔥"), tag(3, "😮")]
    picked = emoji.select(tags, cues, density="high")
    starts = [p["start"] for p in picked]
    for a, b in zip(starts, starts[1:]):
        assert b - a >= emoji.COOLDOWN_SEC - 1e-9
    # Chunk 1 starts 1.0 s after chunk 0, inside the cooldown, so it is skipped
    # even though 4 chunks at "high" allow 2 emoji; the pair picked are the
    # ones 2.5 s apart.
    assert 2.0 not in starts
    assert starts == [1.0, 3.5]


def test_never_same_emoji_twice_in_a_row():
    cues = [cue(0, 0.0), cue(1, 5.0), cue(2, 10.0)]
    tags = [tag(0, "😂"), tag(1, "😂"), tag(2, "😂")]
    picked = emoji.select(tags, cues, density="high")
    assert len(picked) == 1                    # every later repeat was dropped


def test_one_emoji_used_at_most_three_times():
    cues = [cue(i, i * 5.0) for i in range(10)]
    # "😂" on chunks 0,2,4,6,8 separated by a different emoji so no-repeat passes
    tags = [tag(i, "😂" if i % 2 == 0 else "👏") for i in range(10)]
    picked = emoji.select(tags, cues, density="high")
    assert sum(1 for p in picked if p["emoji"] == "😂") <= emoji.MAX_SAME_EMOJI


# --------------------------------------------------------------------------- #
# sensitive content
# --------------------------------------------------------------------------- #

def test_sensitive_chunk_never_gets_party_or_fire():
    cues = [cue(0, 0.0, "he passed away last night"), cue(1, 5.0, "so sad")]
    tags = [
        tag(0, "🎉", sensitive=True, emotion="celebration"),
        tag(1, "🔥", sensitive=True, emotion="joy"),
    ]
    assert emoji.select(tags, cues, density="high") == []


def test_sensitive_chunk_allows_a_serious_emoji():
    cues = [cue(0, 0.0, "he passed away")]
    tags = [tag(0, "😢", sensitive=True, emotion="sadness")]
    picked = emoji.select(tags, cues, density="high")
    assert [p["emoji"] for p in picked] == ["😢"]


def test_non_sensitive_chunk_still_gets_joy():
    cues = [cue(0, 0.0, "that is so funny")]
    tags = [tag(0, "😂", sensitive=False, emotion="laughter")]
    assert [p["emoji"] for p in emoji.select(tags, cues, density="high")] == ["😂"]


# --------------------------------------------------------------------------- #
# fallback labelling
# --------------------------------------------------------------------------- #

def test_failed_model_call_is_labelled_and_uses_keywords():
    cues = [cue(0, 0.0, "that is so funny, I am dead"), cue(1, 4.0, "just talking")]

    def boom(prompt, settings):
        raise RuntimeError("provider exploded")

    result = emoji.tag_chunks(cues, object(), call=boom)
    assert result["source"] == "fallback: keywords"
    assert result["error"] and "provider exploded" in result["error"]
    assert any(t["emoji"] for t in result["tags"])
    assert result["model"] is None


def test_invalid_json_twice_falls_back_but_valid_second_try_is_llm(monkeypatch):
    cues = [cue(0, 0.0, "so funny")]
    calls = {"n": 0}

    def flaky(prompt, settings):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return json.dumps([{
            "i": 0, "emotion": "laughter", "intensity": 0.9, "emoji": "😂",
            "reason": "joke", "confidence": 0.95, "sensitive": False,
        }])

    result = emoji.tag_chunks(cues, object(), call=flaky)
    assert result["source"] == "llm"
    assert result["error"] is None
    assert result["tags"][0]["emoji"] == "😂"
    assert calls["n"] == 2


def test_model_reply_without_array_falls_back():
    cues = [cue(0, 0.0, "so funny")]
    result = emoji.tag_chunks(cues, object(), call=lambda p, s: '{"nope": true}')
    assert result["source"] == "fallback: keywords"
    assert result["error"]


# --------------------------------------------------------------------------- #
# loudness cue
# --------------------------------------------------------------------------- #

def test_loudness_only_raises_intensity_and_upgrades_known_faces():
    tags = [
        tag(0, "😮", intensity=0.5),
        tag(1, "😠", intensity=0.4),
        tag(2, "😢", intensity=0.3),
    ]
    emoji.apply_loudness(tags, [2.0, 1.6, 0.1])
    assert tags[0]["emoji"] == "🤯"
    assert tags[1]["emoji"] == "🤬"
    assert tags[0]["intensity"] == pytest.approx(0.65)
    assert tags[2]["emoji"] == "😢"          # quiet chunk untouched
    assert tags[2]["intensity"] == pytest.approx(0.3)
