# LOCKED: do not modify — see LOCKED_SPEC.md §1 (ONE caption style).
"""THE caption style — one data file, read by the UI, the preview and the render.

There used to be seven looks here (Bold pop, Vertical 9:16, Karaoke fill, Clean
lower-third, Boxed, Ssemble-style viral and the animated set), each a slightly
different render path. LOCKED_SPEC §1 keeps exactly one: the reference look,
measured at 1080x1920 —

    uppercase, heavy sans, cap height ~38 px (ASS Fontsize 54), black outline 6,
    no box, two stacked lines of 4-5 words (max ~16 chars per line), one line
    white and one coloured, colour cycling yellow -> green -> red per chunk.

Everything downstream (the ASS writer, the live preview, the chunker, the
emoji layer) reads the same resolved object — ``caption_style.resolve_caption_settings``
— so the only thing a user can change is a VALUE (size, highlight colour,
burst length, cap, pause, emoji), never the look.

Fields, and what each one actually does at render time:

    font / font_file   the family libass is pointed at (bundled in app/fonts)
    size               ASS Fontsize in design units (1080x1920 space)
    outline / shadow   ASS Outline / Shadow
    box                opaque box behind the text (BorderStyle=3)
    uppercase          display case
    primary            the white-ish base colour
    highlight          the accent colour used when no fixed colour is picked
    colors             per-chunk colour cycle
    line_layout        1 or 2 stacked lines (2 needs its own Dialogue per line)
    pitch              2-line layouts: distance between the line BASES, px
    max_chars_per_line hard wrap target for a line, characters (0 = no cap)
    words_per_cue      default target burst length
    words_per_cue_max  default hard cap
    pause_sec          default: a silence this long always ends a burst
    position_y         where the block sits (fraction of frame height)
    position_mode      "video_bottom" (anchor to the centred video) or "frame"
    emoji              whether this look opts into the Twemoji layer at all
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CaptionPreset:
    key: str
    label: str
    # --- type ---
    font_size: int
    primary: str
    secondary: str
    outline_color: str
    outline_width: int
    shadow: int
    font_override: str | None = None
    font_file: str | None = None
    fonts_dir_override: str | None = "fonts"
    # --- decoration ---
    box: bool = False
    uppercase: bool = False
    # --- rhythm (preset defaults; the sliders override these) ---
    words_per_cue: int = 3
    words_per_cue_max: int = 4
    pause_sec: float = 0.3
    # --- layout ---
    line_layout: int = 1
    pitch: int = 0
    max_chars_per_line: int = 0
    position_y: float = 0.78
    position_mode: str = "frame"
    # Alias kept for callers written against the older field name.
    max_chars: int = 0
    # --- extras ---
    animation: str = ""
    colors: tuple[str, ...] = ()
    emoji: bool = False
    # Legacy flags some call sites still read; they now mean "animation is pop
    # / karaoke", so there is exactly one source of truth for the motion.
    karaoke: bool = False
    pop: bool = False
    highlight: bool = False
    word_emphasis: bool = False
    animated: str | None = None

    def __post_init__(self) -> None:
        # Keep the legacy aliases consistent with the canonical fields so no
        # two readers can disagree about what this preset does.
        if self.max_chars and not self.max_chars_per_line:
            object.__setattr__(self, "max_chars_per_line", self.max_chars)
        object.__setattr__(self, "max_chars", self.max_chars_per_line)
        object.__setattr__(self, "pop", self.animation == "pop")
        object.__setattr__(self, "karaoke", self.animation == "karaoke")


#: The bold-stack reference numbers, in one place. 1080x1920, measured off the
#: reference at 576x1024 and multiplied by 1.875.
STACK_SIZE = 54            # ASS Fontsize -> ~38 px cap height
STACK_OUTLINE = 6          # thick black outline, no box
STACK_SHADOW = 0
STACK_PITCH = 86           # line base to line base
STACK_MAX_CHARS = 16       # ~55 % of frame width
STACK_COLORS = ("#FFFF00", "#00FF00", "#FF0000")


CAPTION_PRESETS: dict[str, CaptionPreset] = {
    # ------------------------------------------------------------------ #
    # THE look: ALL CAPS, heavy sans, two stacked lines, one of them
    # coloured, colour cycling yellow -> green -> red, no box, thick black
    # outline, static. Size and colour are the only knobs.
    # ------------------------------------------------------------------ #
    "bold_stack": CaptionPreset(
        key="bold_stack",
        label="Bold stack (reference look)",
        font_override="Montserrat ExtraBold",
        font_file="Montserrat-ExtraBold.ttf",
        font_size=STACK_SIZE,
        primary="#FFFFFF",
        secondary=STACK_COLORS[0],
        outline_color="#000000",
        outline_width=STACK_OUTLINE,
        shadow=STACK_SHADOW,
        box=False,
        uppercase=True,
        words_per_cue=4,
        words_per_cue_max=5,
        pause_sec=0.4,
        line_layout=2,
        pitch=STACK_PITCH,
        max_chars_per_line=STACK_MAX_CHARS,
        position_y=0.7205,
        position_mode="video_bottom",
        animation="",
        colors=STACK_COLORS,
        emoji=False,
    ),
}

#: Presets in the order the UI lists them. One entry by design.
PRESET_ORDER: tuple[str, ...] = ("bold_stack",)

#: The one look. Kept as a name because jobs and stored overrides carry a key.
DEFAULT_PRESET = "bold_stack"


def get_preset(key: str | None, highlight_color: str | None = None) -> CaptionPreset:
    """Resolve THE style. Any key — old batch, stale override — resolves here.

    ``highlight_color`` replaces the style's own accent. Empty keeps the
    reference's yellow -> green -> red cycle, which the writer reads from
    ``colors``.
    """
    preset = CAPTION_PRESETS[DEFAULT_PRESET]
    color = (highlight_color or "").strip()
    if not color:
        return preset
    if color.upper() == preset.secondary.upper():
        return preset
    return replace(preset, secondary=color)
