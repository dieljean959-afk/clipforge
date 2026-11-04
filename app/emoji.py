"""Context-aware emoji for burned-in captions.

The captions are burned by libass, which cannot draw colour emoji, so an emoji
is a separate animated PNG layer above the caption block (see
``encode.render_clip``'s overlay chain). This module decides WHICH emoji, if
any, a chunk gets — and it is deliberately conservative, because an emoji on
the wrong line reads as a machine sprinkling decoration.

Pipeline, per clip:

    chunks (from translate.build_cues)
        -> prompt with one chunk of context on each side
        -> ONE batched LLM call (app.llm.call_llm: the same client detection uses)
        -> validation against the editable whitelist (assets/emoji/whitelist.json)
        -> optional loudness cue (ffmpeg ebur128) raising intensity
        -> selection rules (density cap, confidence floor, cooldown, no repeats)
        -> timed overlays for the renderer

Nothing here invents a decision. If the model fails twice the tags are built
from the whitelist's own keywords and the track is labelled
``fallback: keywords`` — the clip metadata and the UI show that label, so a
keyword guess is never presented as the model's judgement.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app import jsonfix

# --------------------------------------------------------------------------- #
# whitelist (one editable config file + bundled PNGs)
# --------------------------------------------------------------------------- #

EMOJI_DIR = Path(__file__).resolve().parent / "assets" / "emoji"
WHITELIST_PATH = EMOJI_DIR / "whitelist.json"

#: The emotion vocabulary the model must answer with. Anything else is
#: coerced to "neutral", which is skipped by selection.
EMOTIONS = (
    "joy", "laughter", "surprise", "anger", "sadness", "fear", "love",
    "disgust", "pride", "confusion", "neutral",
)

#: Natural-language emotion words models actually answer with, folded onto the
#: vocabulary above. Without this an honest "excitement" is coerced to
#: "neutral" and selection then DROPS the emoji the model asked for — the whole
#: judgement lost to a vocabulary mismatch, with nothing in the log to show it.
#: (Caught by the NVIDIA wire check: deepseek-v4.1-flash answered "excitement".)
EMOTION_SYNONYMS = {
    "excited": "surprise", "excitement": "surprise", "amazement": "surprise",
    "amazed": "surprise", "awe": "surprise", "shock": "surprise",
    "shocked": "surprise", "hype": "surprise", "hyped": "surprise",
    "happy": "joy", "happiness": "joy", "delight": "joy", "hopeful": "joy",
    "amused": "laughter", "amusement": "laughter", "funny": "laughter",
    "humor": "laughter", "humour": "laughter",
    "rage": "anger", "furious": "anger", "frustration": "anger",
    "frustrated": "anger", "irritated": "anger",
    "sad": "sadness", "grief": "sadness", "disappointed": "sadness",
    "disappointment": "sadness", "heartbreak": "sadness",
    "anxious": "fear", "anxiety": "fear", "nervous": "fear",
    "worried": "fear", "worry": "fear", "tense": "fear",
    "affection": "love", "adoration": "love", "grateful": "love",
    "gratitude": "love", "tender": "love",
    "annoyed": "disgust", "annoyance": "disgust", "cringe": "disgust",
    "disgusted": "disgust", "contempt": "disgust",
    "confident": "pride", "confidence": "pride", "proud": "pride",
    "accomplishment": "pride", "achievement": "pride", "triumph": "pride",
    "confused": "confusion", "puzzled": "confusion", "puzzlement": "confusion",
    "curious": "confusion", "curiosity": "confusion", "conflicted": "confusion",
    "calm": "neutral", "flat": "neutral", "neutral_": "neutral",
}

#: Emoji on this share of a clip's chunks, at most.
DENSITY = {"low": 0.15, "medium": 0.35, "high": 0.60}

MIN_CONFIDENCE = 0.6
#: What a keyword hit reports. It sits exactly ON the floor: a keyword is
#: evidence the word is there, not an opinion about the speaker, so it is
#: allowed through selection but can never outrank a real model judgement.
#: The track is labelled "fallback: keywords" so nobody mistakes it for one.
FALLBACK_CONFIDENCE = 0.6
#: Keyword hits at or above this many matches are slightly stronger evidence.
FALLBACK_CONFIDENCE_MAX = 0.68
COOLDOWN_SEC = 2.0
MAX_SAME_EMOJI = 3
MIN_DURATION = 0.8
MAX_DURATION = 2.0
#: Rendered emoji size in frame pixels. The bundled PNGs are pre-scaled to
#: 144 px so this is a 1:1 blit at the normal setting — no runtime resize, and
#: ffmpeg's scale filter is only used to keep the aspect identical.
EMOJI_SIZE = 140
CAPTION_GAP = 24

#: Directory name of the pre-scaled copies, next to the 72 px originals.
EMOJI_ASSET_SIZE = 144

#: A shouting chunk may only upgrade to a stronger face from this table.
INTENSITY_UPGRADE = {"😮": "🤯", "😲": "🤯", "😠": "🤬", "😨": "😱", "😢": "😭"}


@dataclass(frozen=True)
class EmojiEntry:
    emoji: str
    file: str
    emotions: tuple[str, ...]
    keywords: tuple[str, ...]

    @property
    def path(self) -> Path:
        """The bundled PNG, preferring the pre-scaled copy.

        ``app/assets/emoji/144/`` is produced once by tools/scale_twemoji.py,
        so the renderer blits at its native size instead of asking ffmpeg to
        resize 72 px up to 140 px on every frame. The 72 px file stays as the
        source of truth for the licence and the fallback.
        """
        scaled = EMOJI_DIR / str(EMOJI_ASSET_SIZE) / self.file
        if scaled.exists():
            return scaled
        return EMOJI_DIR / self.file


@dataclass(frozen=True)
class Whitelist:
    entries: dict[str, EmojiEntry]
    blocked_emotions: frozenset[str]
    sensitive_allowed: frozenset[str]
    source: str

    def get(self, emoji: str | None) -> EmojiEntry | None:
        return self.entries.get((emoji or "").strip()) if emoji else None

    def allows_sensitive(self, emoji: str) -> bool:
        return emoji in self.sensitive_allowed


@lru_cache(maxsize=1)
def load_whitelist() -> Whitelist:
    raw = json.loads(WHITELIST_PATH.read_text(encoding="utf-8"))
    entries: dict[str, EmojiEntry] = {}
    for item in raw.get("emojis", []):
        emoji = str(item.get("emoji") or "").strip()
        filename = str(item.get("file") or "").strip()
        if not emoji or not filename:
            continue
        entries[emoji] = EmojiEntry(
            emoji=emoji,
            file=filename,
            emotions=tuple(str(e) for e in item.get("emotions") or ()),
            keywords=tuple(str(k).lower() for k in item.get("keywords") or ()),
        )
    return Whitelist(
        entries=entries,
        blocked_emotions=frozenset(raw.get("sensitive_blocked_emotions") or ()),
        sensitive_allowed=frozenset(raw.get("sensitive_allowed") or ()),
        source=str(raw.get("source") or ""),
    )


def is_allowed(emoji: str | None) -> bool:
    """Whitelist check — the only gate an emoji ever passes through."""
    entry = load_whitelist().get(emoji)
    return bool(entry and entry.path.exists())


# --------------------------------------------------------------------------- #
# tagging: one batched LLM call per clip
# --------------------------------------------------------------------------- #

@dataclass
class Tag:
    i: int
    emotion: str = "neutral"
    intensity: float = 0.0
    emoji: str | None = None
    reason: str = ""
    confidence: float = 0.0
    sensitive: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "i": self.i, "emotion": self.emotion,
            "intensity": round(self.intensity, 2), "emoji": self.emoji,
            "reason": self.reason, "confidence": round(self.confidence, 2),
            "sensitive": self.sensitive,
        }


def _whitelist_prompt_block() -> str:
    """Whitelist as compact lines: emoji | emotions | cue words."""
    lines = []
    for entry in load_whitelist().entries.values():
        lines.append(
            f"{entry.emoji} | {','.join(entry.emotions)} | "
            f"{','.join(entry.keywords[:9])}"
        )
    return "\n".join(lines)


def build_prompt(cues: list[dict[str, Any]]) -> str:
    """The tagging prompt: chunks + one chunk of context each side."""
    texts = [str(c.get("text") or "").strip() for c in cues]
    rows = []
    for i, cue in enumerate(cues):
        prev_text = texts[i - 1] if i > 0 else "(clip start)"
        next_text = texts[i + 1] if i + 1 < len(texts) else "(clip end)"
        rows.append(
            f"index {i} | {float(cue.get('start') or 0):.2f}-"
            f"{float(cue.get('end') or 0):.2f}s | text: {texts[i]}\n"
            f"    before: {prev_text}\n    after: {next_text}"
        )
    return f"""You are a video editor adding ONE emoji to some caption chunks.
You see the whole clip's chunks in order, each with the chunk before and after.

For every chunk decide the speaker's EMOTION and whether an emoji genuinely
helps. Most chunks should get NO emoji: only the strongest emotional beat.

Rules:
- Pick emoji ONLY from this whitelist (emoji | emotions | cue words):
{_whitelist_prompt_block()}
- "I'm dead"/"I'm dying" usually means LAUGHTER, not death.
- Negation matters: "not funny" is not laughter; "I'm not ok" is sadness.
- Sarcasm and exaggeration: read the surrounding chunks before deciding.
- Sensitive content (death, illness, violence, tragedy, abuse, slurs) must
  NEVER get laughing/party/fire/money emoji. Use null, or a serious one
  (😔 😢 🙏) only if the line is clearly sad.
- If you are not sure, return "emoji": null and a low confidence.
- Never use the same emoji twice in a row across chunks.

Answer with JSON ONLY, an array, one object per chunk, same order:
[{{"i": 0, "emotion": "joy|laughter|surprise|anger|sadness|fear|love|disgust|pride|confusion|neutral",
  "intensity": 0.0, "emoji": "😂" or null, "reason": "max 12 words",
  "confidence": 0.0, "sensitive": false}}]

Chunks:
{chr(10).join(rows)}
"""


def parse_tags(reply: str, count: int) -> list[Tag]:
    """Validate the model's JSON against the whitelist and the schema.

    Unknown emoji are REJECTED (set to null), not passed through: the renderer
    can only draw whitelisted PNGs, and a model that invents one is exactly
    the failure this check exists for.
    """
    payload = jsonfix.loads_lenient(jsonfix.extract_payload(reply) or reply)
    items = payload if isinstance(payload, list) else (
        payload.get("chunks") if isinstance(payload, dict) else None
    )
    if not isinstance(items, list):
        raise ValueError("reply is not a JSON array")
    by_index: dict[int, Tag] = {}
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("i", item.get("index", pos)))
        except (TypeError, ValueError):
            i = pos
        if not 0 <= i < count:
            continue
        emotion = str(item.get("emotion") or "neutral").strip().lower()
        # Models answer compound labels ("pride|joy", "joy / pride"). Split and
        # take the first label we know: collapsing the whole string to neutral
        # silently threw away the best chunk of the clip (a confident 🔥 became
        # "neutral" and was then skipped by the selection rule).
        parts = [p.strip() for p in re.split(r"[|/,+&]|\bor\b", emotion) if p.strip()]
        emotion = "neutral"
        for part in parts or [emotion]:
            cand = EMOTION_SYNONYMS.get(part, part)
            if cand in EMOTIONS and cand != "neutral":
                emotion = cand
                break
            if cand in EMOTIONS:
                emotion = "neutral"
        emoji = item.get("emoji")
        emoji = str(emoji).strip() if isinstance(emoji, str) else None
        if emoji in {"", "null", "None", "none"}:
            emoji = None
        if emoji is not None and not is_allowed(emoji):
            logger.info(f"[emoji] rejecting off-whitelist emoji {emoji!r} at chunk {i}")
            emoji = None
        def _num(value: Any, default: float) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return default
        by_index[i] = Tag(
            i=i, emotion=emotion,
            intensity=_num(item.get("intensity"), 0.0),
            emoji=emoji,
            reason=str(item.get("reason") or "")[:120],
            confidence=_num(item.get("confidence"), 0.0),
            sensitive=bool(item.get("sensitive", False)),
        )
    if not by_index:
        raise ValueError("reply contained no usable chunk objects")
    return [by_index.get(i, Tag(i=i)) for i in range(count)]


def keyword_tags(cues: list[dict[str, Any]]) -> list[Tag]:
    """Fallback: whitelist keywords over each chunk's own words.

    Explicitly labelled by the caller as ``fallback: keywords`` — it is a
    lookup table, not an opinion about the speaker.
    """
    entries = list(load_whitelist().entries.values())
    tags: list[Tag] = []
    for i, cue in enumerate(cues):
        text = " " + re.sub(r"[^a-z0-9' ]+", " ", str(cue.get("text") or "").lower()) + " "
        best: tuple[int, EmojiEntry] | None = None
        for entry in entries:
            hits = sum(1 for kw in entry.keywords if f" {kw} " in text)
            if hits and (best is None or hits > best[0]):
                best = (hits, entry)
        if best is None:
            tags.append(Tag(i=i, emotion="neutral", reason="no whitelist keyword"))
            continue
        _hits, entry = best
        words = len(text.split()) or 1
        tags.append(Tag(
            i=i, emotion=entry.emotions[0] if entry.emotions else "neutral",
            intensity=min(0.9, 0.45 + 0.1 * _hits),
            emoji=entry.emoji,
            reason=f"keyword match: {entry.keywords[0]}",
            confidence=min(FALLBACK_CONFIDENCE_MAX, FALLBACK_CONFIDENCE + 0.02 * _hits),
            sensitive=False,
        ))
    return tags


def tag_chunks(
    cues: list[dict[str, Any]],
    settings: Any,
    *,
    call: Callable[[str, Any], str] | None = None,
) -> dict[str, Any]:
    """Tag every chunk of ONE clip. Never raises: failures are labelled.

    Returns ``{"source": "llm"|"fallback: keywords", "model": str|None,
    "tags": [...], "raw": str, "error": str|None}``.
    """
    if not cues:
        return {"source": "llm", "model": None, "tags": [], "raw": "", "error": None}
    if call is None:
        from app import llm as _llm

        # plain_json: our answer is emoji JSON, not the clip-detection schema.
        def call(prompt: str, cfg: Any) -> str:  # noqa: F811 - default caster
            return _llm.call_llm(prompt, cfg, plain_json=True)
    prompt = build_prompt(cues)
    raw = ""
    last_error: str | None = None
    for attempt in (1, 2):
        try:
            raw = call(prompt, settings)
            tags = parse_tags(raw, len(cues))
            return {
                "source": "llm",
                "model": _model_name(settings),
                "tags": [t.as_dict() for t in tags],
                "raw": raw,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - every failure must be labelled
            last_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            logger.warning(f"[emoji] tagging failed ({last_error})")
    return {
        "source": "fallback: keywords",
        "model": None,
        "tags": [t.as_dict() for t in keyword_tags(cues)],
        "raw": raw,
        "error": last_error,
    }


def _model_name(settings: Any) -> str | None:
    """Provider + model that tagged this clip, for the clip's provenance.

    Read through app.llm's accessors: each hosted section ([deepseek], [nvidia],
    [groq]) carries its own model name, and the old version looked for
    `settings.llm.model` / `settings.detection.model` — neither exists, so every
    track recorded `"model": null` and a track could not be traced back to the
    backend that produced it.
    """
    try:
        from app import llm as _llm

        return str(_llm.provider_label(settings))
    except Exception:  # noqa: BLE001 - provenance must never fail the stage
        return None


# --------------------------------------------------------------------------- #
# optional audio cue: is this chunk shouted?
# --------------------------------------------------------------------------- #

def loudness_z(
    src: Path, cut_start: float, duration: float, cues: list[dict[str, Any]],
    *, timeout: float = 90.0,
) -> list[float] | None:
    """Per-chunk loudness, expressed as z-scores.

    One ffmpeg pass over the clip's audio (ebur128 momentary loudness), then
    each chunk's window is averaged. Returns None when ffmpeg is unavailable or
    the parse fails — the caller then keeps the model's own intensity instead
    of pretending it measured anything.
    """
    if not cues:
        return None
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-v", "info",
        "-ss", f"{max(0.0, cut_start):.3f}", "-t", f"{duration:.3f}",
        "-i", str(src),
        "-af", "ebur128=framelog=verbose:metadata=1,ametadata=print",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"[emoji] loudness pass failed: {exc}")
        return None
    samples: list[tuple[float, float]] = []
    t = 0.0
    for line in (proc.stderr or "").splitlines():
        line = line.strip()
        # ffmpeg prefixes each metadata line with the filter instance
        # ("[Parsed_ametadata_1 @ 0x…] frame:3 pts_time:0.3"), so match the
        # timestamp anywhere in the line instead of at its start.
        moment = re.search(r"pts_time:([0-9.]+)", line)
        if moment:
            t = float(moment.group(1))
            continue
        reading = re.search(r"lavfi\.r128\.M=(-?[0-9.]+)", line)
        if reading:
            value = float(reading.group(1))
            if math.isfinite(value):
                # -120 is ebur128's silence sentinel; clamp it to the usual
                # quiet floor so silence contributes instead of vanishing.
                samples.append((t, max(value, -70.0)))
    if len(samples) < 5:
        return None
    values = [v for _t, v in samples]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    sigma = math.sqrt(var) or 1.0
    out: list[float] = []
    for cue in cues:
        start = float(cue.get("start") or 0.0)
        end = float(cue.get("end") or start)
        window = [v for ts, v in samples if start - 0.1 <= ts <= end + 0.1]
        if not window:
            out.append(0.0)
            continue
        out.append(round((sum(window) / len(window) - mean) / sigma, 2))
    return out


def apply_loudness(tags: list[dict[str, Any]], z_scores: list[float] | None) -> None:
    """A shouting chunk gets more intensity, and may upgrade its face."""
    if not z_scores:
        return
    for tag in tags:
        i = int(tag.get("i", -1))
        if not 0 <= i < len(z_scores):
            continue
        z = z_scores[i]
        tag["loudness_z"] = z
        if z < 1.5:
            continue
        tag["intensity"] = round(min(1.0, float(tag.get("intensity") or 0.0) + 0.15), 2)
        upgraded = INTENSITY_UPGRADE.get(str(tag.get("emoji") or ""))
        if upgraded and is_allowed(upgraded):
            tag["emoji_before_loudness"] = tag.get("emoji")
            tag["emoji"] = upgraded


# --------------------------------------------------------------------------- #
# selection rules (code, not prompt)
# --------------------------------------------------------------------------- #

def _sensitive_ok(tag: dict[str, Any], whitelist: Whitelist) -> bool:
    emoji = str(tag.get("emoji") or "")
    entry = whitelist.get(emoji)
    if entry is None:
        return False
    if not tag.get("sensitive"):
        return True
    if set(entry.emotions) & whitelist.blocked_emotions:
        return False
    return whitelist.allows_sensitive(emoji)


def select(
    tags: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    *,
    density: str = "medium",
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """Choose at most ``density`` × chunks emoji, strongest beats first."""
    if not enabled or not tags:
        return []
    whitelist = load_whitelist()
    share = DENSITY.get(str(density).lower(), DENSITY["medium"])
    cap = max(1, int(round(len(cues) * share)))

    candidates = []
    for tag in tags:
        emoji = str(tag.get("emoji") or "")
        if not emoji or not is_allowed(emoji):
            continue
        if float(tag.get("confidence") or 0.0) < MIN_CONFIDENCE:
            continue
        if str(tag.get("emotion") or "neutral") == "neutral":
            continue
        if not _sensitive_ok(tag, whitelist):
            continue
        i = int(tag.get("i", -1))
        if not 0 <= i < len(cues):
            continue
        candidates.append(tag)

    # Strongest beat first, then drop anything that breaks a spacing rule.
    candidates.sort(
        key=lambda t: (float(t.get("intensity") or 0) * 0.6
                       + float(t.get("confidence") or 0) * 0.4),
        reverse=True,
    )
    picked: list[dict[str, Any]] = []
    for tag in candidates:
        if len(picked) >= cap:
            break
        i = int(tag.get("i"))
        start = float(cues[i].get("start") or 0.0)
        if any(abs(start - float(cues[int(p["i"])].get("start") or 0.0)) < COOLDOWN_SEC
               for p in picked):
            continue
        emoji = str(tag["emoji"])
        if sum(1 for p in picked if p["emoji"] == emoji) >= MAX_SAME_EMOJI:
            continue
        picked.append({**tag, "emoji": emoji})

    # Chronological, then enforce "never the same emoji twice in a row".
    picked.sort(key=lambda t: float(cues[int(t["i"])].get("start") or 0.0))
    out: list[dict[str, Any]] = []
    for tag in picked:
        if out and out[-1]["emoji"] == tag["emoji"]:
            continue
        i = int(tag["i"])
        cue = cues[i]
        start = float(cue.get("start") or 0.0)
        cue_end = float(cue.get("end") or start)
        span = min(MAX_DURATION, max(MIN_DURATION, cue_end - start))
        out.append({
            "i": i,
            "emoji": tag["emoji"],
            "emotion": tag.get("emotion"),
            "intensity": tag.get("intensity"),
            "confidence": tag.get("confidence"),
            "reason": tag.get("reason"),
            "sensitive": bool(tag.get("sensitive")),
            "start": round(start, 3),
            "end": round(start + span, 3),
            "cueText": str(cue.get("text") or ""),
        })
    return out


def build_track(
    cues: list[dict[str, Any]],
    settings: Any,
    *,
    enabled: bool = True,
    density: str = "medium",
    src: Path | None = None,
    cut_start: float = 0.0,
    duration: float = 0.0,
    call: Callable[[str, Any], str] | None = None,
    use_loudness: bool = True,
) -> dict[str, Any]:
    """Full per-clip emoji track: tag -> audio cue -> selection."""
    if not enabled or not cues:
        return {"source": "disabled", "tags": [], "selected": [], "error": None}
    result = tag_chunks(cues, settings, call=call)
    tags = result["tags"]
    if use_loudness and src is not None and duration > 0:
        try:
            z = loudness_z(src, cut_start, duration, cues)
        except Exception as exc:  # noqa: BLE001 - audio cue is best-effort
            logger.warning(f"[emoji] loudness skipped: {exc}")
            z = None
        try:
            apply_loudness(tags, z)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[emoji] loudness apply skipped: {exc}")
    selected = select(tags, cues, density=density, enabled=enabled)
    return {
        "source": result["source"],
        "model": result.get("model"),
        "error": result.get("error"),
        "density": density,
        "tags": tags,
        "selected": selected,
        "raw": result.get("raw", ""),
    }


# --------------------------------------------------------------------------- #
# geometry (shared by the renderer and the live preview)
# --------------------------------------------------------------------------- #

def emoji_geometry(
    settings: Any,
    *,
    frame_w: int = 1080,
    frame_h: int = 1920,
    anchor_y: int | None = None,
    size: int = EMOJI_SIZE,
    preset_key: str | None = None,
) -> dict[str, int]:
    """Where the emoji sits: centred, one gap above the caption block.

    ``preset_key`` must be the preset this RENDER chose, not the config default:
    the clip carries its own style (a bold_stack job rendered with the config's
    bold_yellow guess put the emoji 144 px too high).
    """
    from app import caption_style as _caption_style
    from app import subtitle as _subtitle

    # The RESOLVED object, so the emoji sits above the block the render will
    # actually produce (the preview and the renderer agree by construction).
    resolved = _caption_style.resolve_caption_settings(settings)
    key = preset_key or resolved["presetKey"]
    preset = _subtitle.get_preset(key)
    caption_top: int
    if preset.key == "bold_stack":
        metrics = _subtitle.bold_stack_metrics(float(resolved["sizeScale"]))
        line1_y, _line2_y = _subtitle.bold_stack_positions(
            frame_h, metrics["pitch"], anchor_y=anchor_y,
            position_y=float(resolved["positionY"]),
        )
        caption_top = int(line1_y - metrics["cap_h"] - metrics["outline"])
    else:
        # One line: the block's top is the text row above its baseline.
        baseline = int(round(frame_h * float(resolved["positionY"])))
        caption_top = int(baseline - resolved["fontSize"] * 0.72 - resolved["outline"])
    y = max(int(frame_h * 0.06), caption_top - CAPTION_GAP - size)
    x = (frame_w - size) // 2
    return {"x": x, "y": y, "size": size}


def overlays_for_render(
    selected: list[dict[str, Any]],
    settings: Any,
    *,
    frame_w: int = 1080,
    frame_h: int = 1920,
    anchor_y: int | None = None,
    preset_key: str | None = None,
) -> list[dict[str, Any]]:
    """Selection -> the overlay list encode.render_clip consumes."""
    whitelist = load_whitelist()
    geo = emoji_geometry(
        settings, frame_w=frame_w, frame_h=frame_h, anchor_y=anchor_y,
        preset_key=preset_key,
    )
    out = []
    for item in selected:
        entry = whitelist.get(str(item.get("emoji") or ""))
        if entry is None or not entry.path.exists():
            continue
        out.append({
            "emoji": entry.emoji,
            "path": str(entry.path),
            "start": float(item.get("start") or 0.0),
            "end": float(item.get("end") or 0.0),
            "x": geo["x"], "y": geo["y"], "size": geo["size"],
            "i": item.get("i"),
        })
    return out
