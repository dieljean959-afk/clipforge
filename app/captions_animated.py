"""Ssemble-style animated captions: word-by-word, high contrast, 9:16 safe.

Why this is its own module
--------------------------
The legacy caption path writes ONE ASS event per *cue* and paints the accent on
the cue's first word. That is a static colour: it cannot follow the speaker,
because a cue is 2-4 words of audio and the accent only ever moves when the cue
changes. The trending look is different in kind — every WORD is its own timed
unit, and the word being spoken is lit and scaled for exactly as long as it is
spoken. That needs a different generator, not another flag on the old one.

How the animation is built (all plain ASS, no external compositor)
------------------------------------------------------------------
* **Word timing** — each word carries the Whisper start/end. A chunk is shown
  as a sequence of events, one per word, each re-drawing the whole chunk with
  *that* word as the active one. So the highlight advances on real speech
  boundaries rather than on cue changes.
* **Active word** — colour swap plus a 100 -> 115 % scale pop over 100 ms
  (``\\t(0,100,\\fscx115\\fscy115)``). Neighbours keep their neutral colour.
* **Chunk entrance** — the first event of a chunk fades in and scales 90 -> 100
  over 110 ms. Everything is inside the ASS event, so nothing has to be
  pre-rendered per frame and a re-render is still one ffmpeg pass.
* **Safe area** — alignment 2 (bottom-centre) with ``MarginV`` = 18 % of frame
  height and 90 px side margins, so TikTok's caption/button stack (bottom ~15 %)
  cannot cover the text.

Chunk rules (2-4 words, <= 24 chars, punctuation and >0.4 s pauses break)
------------------------------------------------------------------------
Whisper word timings are the source of truth. A chunk closes when any of:
4 words / 24 characters / a gap of 0.4 s or more / the previous word ends a
sentence or clause (`. ! ? , ; :`) / 2.6 s of speech. A chunk that ends up with
one word is merged into the next one when the result still fits, so a stray
"and" never gets a caption of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# style data (every look is a value, not a code path)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnimatedStyle:
    key: str
    label: str
    font: str                 # family name as libass resolves it
    font_file: str            # bundled file, for the fonts dir + docs
    size: int                 # px at 1080x1920
    fill: str                 # neutral word colour
    active: str               # spoken word colour
    outline: str
    outline_w: int
    shadow: int
    uppercase: bool
    #: 0..1 from the TOP where the caption block's baseline sits.
    position_y: float
    #: percent scale for the spoken word (100 = no pop).
    active_scale: int = 115
    #: extra ASS tags applied to the spoken word (glow / pad / box look).
    active_tags: str = ""
    #: thick accent pad behind the active word (the "box highlight" look).
    active_pad: bool = False
    #: Pad colour for ``active_pad``. Separate from ``active`` because the box
    #: style keeps the word white and colours only the pad behind it.
    pad: str | None = None
    #: attach a Twemoji PNG above the caption for key words.
    emoji: bool = False
    #: letter spacing in px (ASS Spacing).
    spacing: float = 0.0


STYLES: dict[str, AnimatedStyle] = {
    "bold_yellow": AnimatedStyle(
        key="bold_yellow",
        label="Animated · Bold Yellow",
        font="Anton",
        font_file="Anton-Regular.ttf",
        size=94,
        fill="#FFFFFF",
        active="#FFD400",
        outline="#000000",
        outline_w=9,
        shadow=5,
        uppercase=True,
        position_y=0.82,
        spacing=1.0,
        emoji=True,
    ),
    "clean_white": AnimatedStyle(
        key="clean_white",
        label="Animated · Clean White",
        font="Poppins",
        font_file="Poppins-ExtraBold.ttf",
        size=76,
        fill="#FFFFFF",
        active="#FFFFFF",
        outline="#111111",
        outline_w=2,
        shadow=6,
        uppercase=False,
        position_y=0.82,
        active_scale=108,
        # Minimal look: no colour jump, the spoken word just lifts and grows.
        active_tags=r"\bord0",
        spacing=0.0,
    ),
    "green_karaoke": AnimatedStyle(
        key="green_karaoke",
        label="Animated · Green Karaoke",
        font="Montserrat",
        font_file="Montserrat-ExtraBold.ttf",
        size=84,
        fill="#FFFFFF",
        active="#39FF14",
        outline="#000000",
        outline_w=6,
        shadow=3,
        uppercase=True,
        position_y=0.82,
        active_scale=114,
        spacing=0.5,
    ),
    "neon": AnimatedStyle(
        key="neon",
        label="Animated · Neon Glow",
        font="Bebas Neue",
        font_file="BebasNeue-Regular.ttf",
        size=104,
        fill="#EAF4FF",
        active="#00E5FF",
        outline="#04121F",
        outline_w=5,
        shadow=2,
        uppercase=True,
        position_y=0.82,
        active_scale=116,
        # Halo: accent border plus a blurred copy of the word behind it.
        active_tags=r"\3c&H00FFE500&\blur6\bord7",
        spacing=2.0,
    ),
    "box_highlight": AnimatedStyle(
        key="box_highlight",
        label="Animated · Box Highlight",
        font="Poppins",
        font_file="Poppins-ExtraBold.ttf",
        size=74,
        fill="#FFFFFF",
        active="#FFFFFF",
        outline="#000000",
        outline_w=2,
        shadow=2,
        uppercase=True,
        position_y=0.82,
        active_scale=110,
        active_pad=True,
        pad="#FF2D55",
        spacing=0.0,
    ),
}

DEFAULT_STYLE = "bold_yellow"

#: The chunking contract from the brief.
MAX_WORDS = 4
MIN_WORDS = 2
MAX_CHARS = 24
MAX_LINES = 2
PAUSE_BREAK_SEC = 0.4
MAX_CHUNK_SEC = 2.6

#: How long the spoken word takes to reach its full size.
POP_MS = 100
#: Chunk entrance (fade + scale); the brief asks for 80-120 ms.
ENTRANCE_MS = 110


def get_style(key: str | None) -> AnimatedStyle:
    """Resolve a style key; never raises (a bad key renders the default)."""
    return STYLES.get(str(key or ""), STYLES[DEFAULT_STYLE])


def foreground_bottom(
    src_w: int,
    src_h: int,
    frame_w: int = 1080,
    frame_h: int = 1920,
    inset: int = 24,
) -> int | None:
    """Y of the caption baseline at the bottom edge of the FIT foreground.

    A FIT clip scales the source to fill the frame's WIDTH and centres it
    vertically over a blurred copy of itself. So the sharp video occupies a
    band in the middle of the frame, and `frame_h - band` is all blur. A
    caption placed by bottom margin lands in that blur, BELOW the video — it
    reads as a separate graphic rather than something the speaker is saying.

    Measured from the real numbers, never hardcoded: a 16:9 source gives
    fg_h = 608, fg_top = 656, fg_bottom = 1264, so the baseline sits at 1240.
    """
    frame_w = int(frame_w)
    frame_h = int(frame_h)
    fg_h = int(round(frame_w * float(src_h) / float(src_w) / 2.0) * 2)
    fg_h = min(fg_h, frame_h)
    # A source that already fills the frame (portrait) has no blurred band to
    # avoid: anchoring to its bottom edge would park the text under TikTok's
    # own UI, so that case keeps the safe-area margin instead (None).
    if fg_h > frame_h * 0.85:
        return None
    fg_top = int(round((frame_h - fg_h) / 2.0))
    return fg_top + fg_h - int(inset)


# --------------------------------------------------------------------------- #
# ASS primitives
# --------------------------------------------------------------------------- #


def _ass_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _bgr(hex_color: str) -> str:
    """#RRGGBB -> ASS &H00BBGGRR& (the & terminator is included)."""
    h = str(hex_color).lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}&"


def _safe_word(text: str) -> str:
    """A caption word must never contain ASS control characters."""
    return (
        str(text)
        .replace("\\", "/")
        .replace("{", "(")
        .replace("}", ")")
        .strip()
    )


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #

_BREAK_AFTER = (".", "!", "?", ",", ";", ":", "…")


def _chunk(words: list[dict]) -> list[list[dict]]:
    """Group word timings into 2-4 word chunks under the brief's rules."""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        text = str(w.get("word") or "").strip()
        if not text:
            continue
        if cur:
            gap = float(w.get("start") or 0.0) - float(cur[-1].get("end") or 0.0)
            joined = " ".join([str(x.get("word") or "") for x in cur] + [text])
            span = float(w.get("end") or 0.0) - float(cur[0].get("start") or 0.0)
            prev = str(cur[-1].get("word") or "").rstrip("\"'”’)")
            if (
                len(cur) >= MAX_WORDS
                or len(joined) > MAX_CHARS
                or gap >= PAUSE_BREAK_SEC
                or span > MAX_CHUNK_SEC
                or prev.endswith(_BREAK_AFTER)
            ):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)

    # Never leave a one-word chunk: merge it forward (or backward at the end)
    # when the merged text still fits the character budget.
    merged: list[list[dict]] = []
    for c in chunks:
        if (
            len(c) < MIN_WORDS
            and merged
            and len(" ".join(str(x.get("word") or "") for x in merged[-1] + c)) <= MAX_CHARS
            and len(merged[-1]) + len(c) <= MAX_WORDS
        ):
            merged[-1] = merged[-1] + c
            continue
        merged.append(c)
    # A lone word can also be fixed by BORROWING one from a 3+ word neighbour:
    # merging outright needs 5 words (over the cap), but moving one word keeps
    # every line at 2-4 words and still respects the character budget.
    for i, c in enumerate(merged):
        if len(c) != 1:
            continue
        for j in (i + 1, i - 1):
            if not (0 <= j < len(merged)) or not merged[j]:
                continue
            if len(merged[j]) < 3:
                continue
            take = merged[j][0] if j > i else merged[j][-1]
            cand = c + [take] if j > i else [take] + c
            if len(" ".join(str(x.get("word") or "") for x in cand)) > MAX_CHARS:
                continue
            merged[i] = cand
            merged[j] = merged[j][1:] if j > i else merged[j][:-1]
            break

    out: list[list[dict]] = []
    for i, c in enumerate(merged):
        if len(c) < MIN_WORDS and i + 1 < len(merged) and len(c) + len(merged[i + 1]) <= MAX_WORDS:
            out.append(c + merged[i + 1])
            merged[i + 1] = []
            continue
        if c:
            out.append(c)
    return [c for c in out if c]


def _wrap_index(words: list[str]) -> int | None:
    """Where to break a chunk into two lines, computed on PLAIN text.

    Decided once per chunk and reused for every word event: measuring the
    *rendered* strings would measure ASS tag markup too, so the break would
    move between events and the caption would visibly jump lines while the
    highlight advances.

    Returns the index of the first word on line 2, or None for one line.
    """
    if len(words) <= 1 or len(" ".join(words)) <= MAX_CHARS:
        return None
    best: int | None = None
    best_score: int | None = None
    for i in range(1, len(words)):
        a = " ".join(words[:i])
        b = " ".join(words[i:])
        if len(a) > MAX_CHARS or len(b) > MAX_CHARS:
            continue
        score = abs(len(a) - len(b))
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


def _join(parts: list[str], break_at: int | None) -> str:
    """Join rendered words, inserting the chunk's fixed ASS line break."""
    if break_at is None or break_at >= len(parts):
        return " ".join(parts)
    return " ".join(parts[:break_at]) + "\\N" + " ".join(parts[break_at:])


# --------------------------------------------------------------------------- #
# emoji (Twemoji PNGs shipped with the app, overlaid by ffmpeg)
# --------------------------------------------------------------------------- #

EMOJI_DIR = Path(__file__).resolve().parent / "assets" / "emoji"

#: word -> PN G filename in EMOJI_DIR. Deliberately small and literal: an emoji
#: on the wrong word is worse than no emoji.
EMOJI_WORDS: dict[str, str] = {
    "fire": "1f525.png", "hot": "1f525.png", "insane": "1f525.png",
    "crazy": "1f525.png", "wild": "1f525.png",
    "money": "1f4b0.png", "cash": "1f4b0.png", "dollars": "1f4b0.png",
    "dollar": "1f4b0.png", "rich": "1f4b0.png", "selling": "1f4b0.png",
    "sold": "1f4b0.png", "paid": "1f4b0.png", "ebay": "1f4b0.png",
    "fast": "26a1.png", "quick": "26a1.png", "instantly": "26a1.png",
    "speed": "26a1.png",
    "look": "1f440.png", "watch": "1f440.png", "watchin": "1f440.png",
    "see": "1f440.png", "staring": "1f440.png", "stare": "1f440.png",
    "love": "2764.png", "loved": "2764.png", "favorite": "2764.png",
    "cried": "1f62d.png", "cry": "1f62d.png", "sad": "1f62d.png",
    "tears": "1f62d.png",
    "warning": "1f6a8.png", "danger": "1f6a8.png", "problem": "1f6a8.png",
    "viral": "1f680.png", "grow": "1f680.png", "growth": "1f680.png",
    "perfect": "1f4af.png", "best": "1f4af.png", "literally": "1f4af.png",
    "thanks": "1f44f.png", "thank": "1f44f.png", "congrats": "1f44f.png",
    "idea": "1f4a1.png", "think": "1f4a1.png", "realize": "1f4a1.png",
}


def _emoji_for(word: str) -> Path | None:
    key = "".join(ch for ch in str(word).lower() if ch.isalpha())
    name = EMOJI_WORDS.get(key)
    if not name:
        return None
    p = EMOJI_DIR / name
    return p if p.exists() else None


def emoji_overlays(
    cues: list[dict], style_key: str | None, start_offset: float = 0.0
) -> list[dict]:
    """Timed emoji PNGs for the chunks that name a key word.

    Returns ``[{"path", "start", "end"}]`` in clip-relative seconds. Empty when
    the style does not want emoji or no asset is bundled, which is why the
    render path stays a plain ``-vf`` chain for every other style.
    """
    style = get_style(style_key)
    if not style.emoji:
        return []
    out: list[dict] = []
    for cue in cues:
        words = cue.get("words") or []
        if not words:
            continue
        chosen = None
        for w in words:
            p = _emoji_for(str(w.get("word") or ""))
            if p is not None:
                chosen = p
                break
        if chosen is None:
            continue
        start = float(words[0].get("start") or 0.0) - start_offset
        end = float(words[-1].get("end") or 0.0) - start_offset
        if end - start < 0.15:
            continue
        out.append({"path": str(chosen), "start": max(0.0, start), "end": max(0.0, end)})
    return out


# --------------------------------------------------------------------------- #
# ASS generation
# --------------------------------------------------------------------------- #


def build_header(
    style: AnimatedStyle, width: int, height: int, margin_v: int | None = None
) -> str:
    mv = margin_v if margin_v is not None else max(1, int(round(height * (1.0 - style.position_y))))
    return f"""[Script Info]
Title: Clipmake Animated Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Animated,{style.font},{style.size},{_bgr(style.fill)},{_bgr(style.active)},{_bgr(style.outline)},&H80000000&,0,0,0,0,100,100,{style.spacing},0,1,{style.outline_w},{style.shadow},2,90,90,{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_events(
    cues: list[dict],
    style_key: str | None = None,
    start_offset: float = 0.0,
    anchor_y: int | None = None,
    frame_w: int = 1080,
) -> str:
    """One Dialogue line per spoken WORD, re-drawing its 2-4 word chunk.

    Re-emitting the chunk for every word (instead of one static line with the
    first word coloured) is what makes the accent follow the voice. The pop is
    an ASS transform on the active word only; the chunk's entrance fade/scale
    is emitted on the first word of the chunk.
    """
    style = get_style(style_key)
    fill = _bgr(style.fill)
    active = _bgr(style.active)
    lines: list[str] = []

    # Chunk across the WHOLE clip, not cue by cue. The cue list upstream is
    # deliberately 1-2 words per cue (it drives SRT and the 2-line display
    # wrap), so chunking inside a cue could never produce the 2-4 word lines
    # this style is defined by. Word timings are contiguous and clip-relative,
    # so flattening them keeps every timing exact.
    flat: list[dict] = []
    for cue in cues:
        flat.extend(
            w for w in (cue.get("words") or []) if str(w.get("word") or "").strip()
        )
    if not flat:
        # Cues written before word timings existed: fall back to evenly
        # splitting the cue text so the clip still gets captions.
        for cue in cues:
            toks = [t for t in str(cue.get("text") or "").split() if t]
            if not toks:
                continue
            span = max(0.2, float(cue.get("end") or 0.0) - float(cue.get("start") or 0.0))
            for i, t in enumerate(toks):
                a = float(cue.get("start") or 0.0) + span * i / len(toks)
                b = float(cue.get("start") or 0.0) + span * (i + 1) / len(toks)
                flat.append({"start": a, "end": b, "word": t})
    flat.sort(key=lambda w: float(w.get("start") or 0.0))

    if flat:
        for chunk in _chunk(flat):
            texts = [_safe_word(w.get("word") or "") for w in chunk]
            if style.uppercase:
                texts = [t.upper() for t in texts]
            # Fixed per chunk, on plain text, so the break never moves while
            # the highlight advances word by word.
            break_at = _wrap_index(texts)
            for i, word in enumerate(chunk):
                t0 = float(word.get("start") or 0.0)
                if i + 1 < len(chunk):
                    t1 = float(chunk[i + 1].get("start") or word.get("end") or t0)
                else:
                    t1 = float(word.get("end") or t0)
                if t1 <= t0:
                    t1 = t0 + 0.08
                # Keep the last word readable for a beat instead of vanishing
                # on its own stop timestamp.
                if i == len(chunk) - 1:
                    t1 = max(t1, t0 + 0.20)

                rendered: list[str] = []
                for j, text in enumerate(texts):
                    if j == i:
                        tags = (
                            rf"{{\1c{active}"
                            rf"\fscx100\fscy100\t(0,{POP_MS},\fscx{style.active_scale}\fscy{style.active_scale})"
                        )
                        if style.active_tags:
                            tags += style.active_tags
                        if style.active_pad:
                            # A thick accent border reads as a rounded highlight
                            # pad behind the word at caption sizes. The pad has
                            # its own colour: this style keeps the WORD white
                            # and colours only what sits behind it.
                            tags += rf"\3c{_bgr(style.pad or style.active)}\bord16\blur4"
                        tags += "}"
                        rendered.append(tags + text + r"{\rAnimated}")
                    else:
                        rendered.append(rf"{{\1c{fill}}}" + text)

                body = _join(rendered, break_at)
                # Explicit placement instead of a bottom margin: \an2 anchors
                # the block's BOTTOM at (centre, anchor_y), so the text sits
                # against the foreground video's lower edge and never drifts
                # into the blurred band below it.
                if anchor_y is not None:
                    body = rf"{{\an2\pos({frame_w // 2},{int(anchor_y)})}}" + body
                if i == 0:
                    # Chunk entrance: fade + scale in over ENTRANCE_MS.
                    body = (
                        rf"{{\fad({ENTRANCE_MS},60)\fscx90\fscy90"
                        rf"\t(0,{ENTRANCE_MS},\fscx100\fscy100)}}"
                    ) + body
                lines.append(
                    f"Dialogue: 0,{_ass_time(t0 - start_offset)},{_ass_time(t1 - start_offset)},"
                    f"Animated,,0,0,0,,{body}"
                )
    return "\n".join(lines) + "\n"


def build_ass(
    cues: list[dict],
    style_key: str | None,
    width: int,
    height: int,
    start_offset: float = 0.0,
    anchor_y: int | None = None,
) -> str:
    style = get_style(style_key)
    return build_header(style, width, height) + build_events(
        cues, style_key, start_offset, anchor_y=anchor_y, frame_w=width
    )


def write_ass(
    cues: list[dict],
    style_key: str | None,
    path: Path,
    width: int,
    height: int,
    start_offset: float = 0.0,
    anchor_y: int | None = None,
) -> Path:
    path.write_text(
        build_ass(cues, style_key, width, height, start_offset, anchor_y=anchor_y),
        encoding="utf-8",
    )
    return path
