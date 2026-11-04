"""THE caption preset table — one data file, read by the UI, the preview and
the renderer.

Every look the app can burn is described here as VALUES. Nothing downstream is
allowed to special-case a preset by key: the ASS writer, the live preview, the
chunking and the emoji layer all read the same resolved object
(``caption_style.resolve_caption_settings``), which is this table + the user's
slider overrides. That is what makes a slider provably reach the pixels.

Fields, and what each one actually does at render time:

    font / font_file   the family libass is pointed at (bundled in app/fonts)
    size               ASS Fontsize in design units (1080x1920 space)
    outline / shadow   ASS Outline / Shadow
    box                opaque box behind the text (BorderStyle=3)
    uppercase          display case
    primary            the white-ish base colour
    highlight          the accent colour (coloured line / spoken word)
    colors             per-chunk colour cycle (bold stack uses this)
    line_layout        1 or 2 stacked lines (2 needs its own Dialogue per line)
    pitch              2-line presets: distance between the line BASES, px
    max_chars_per_line hard wrap target for a line, characters (0 = no cap)
    words_per_cue      default target burst length
    words_per_cue_max  default hard cap
    pause_sec          default: a silence this long always ends a burst
    position_y         where the block sits (fraction of frame height)
    position_mode      "frame" or "video_bottom" (anchor to the centred video)
    animation          "", "pop" or "karaoke"
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
    # --- motion / extras ---
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
    # 1. The reference look: ALL CAPS, heavy sans, two stacked lines, one
    #    of them coloured, colour cycling yellow -> green -> red, no box,
    #    thick black outline, static (size + colour ARE the style).
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
    # ------------------------------------------------------------------ #
    # 2. Ssemble-style viral: huge ALL CAPS single line, thick outline,
    #    pop-in, high on the frame, yellow accent.
    # ------------------------------------------------------------------ #
    "viral": CaptionPreset(
        key="viral",
        label="Ssemble-style viral",
        font_override="Montserrat ExtraBold",
        font_file="Montserrat-ExtraBold.ttf",
        font_size=96,
        primary="#FFFFFF",
        secondary="#FFE600",
        outline_color="#000000",
        outline_width=8,
        shadow=2,
        box=False,
        uppercase=True,
        words_per_cue=3,
        words_per_cue_max=4,
        pause_sec=0.25,
        line_layout=1,
        position_y=0.62,
        position_mode="frame",
        animation="pop",
        emoji=True,
    ),
    # ------------------------------------------------------------------ #
    # 3. Vertical 9:16 big captions: 95 px, up to two lines, mixed case,
    #    amber accent, sits a little lower than viral.
    # ------------------------------------------------------------------ #
    "vertical_916": CaptionPreset(
        key="vertical_916",
        label="Vertical 9:16 big captions",
        font_override="Montserrat ExtraBold",
        font_file="Montserrat-ExtraBold.ttf",
        font_size=95,
        primary="#FFFFFF",
        secondary="#FFD400",
        outline_color="#000000",
        outline_width=8,
        shadow=3,
        box=False,
        uppercase=False,
        words_per_cue=4,
        words_per_cue_max=5,
        pause_sec=0.35,
        line_layout=2,
        pitch=110,
        max_chars_per_line=18,
        position_y=0.60,
        position_mode="frame",
        animation="pop",
        emoji=True,
    ),
    # ------------------------------------------------------------------ #
    # 4. Bold pop (default): mid-size, neon-green outline, word highlight,
    #    centred, one or two words per line.
    # ------------------------------------------------------------------ #
    "bold_pop": CaptionPreset(
        key="bold_pop",
        label="Bold pop (default)",
        font_override="Montserrat ExtraBold",
        font_file="Montserrat-ExtraBold.ttf",
        font_size=68,
        primary="#FFFFFF",
        secondary="#39FF14",
        outline_color="#39FF14",
        outline_width=5,
        shadow=1,
        box=False,
        uppercase=False,
        words_per_cue=3,
        words_per_cue_max=4,
        pause_sec=0.30,
        line_layout=1,
        position_y=0.52,
        position_mode="frame",
        animation="pop",
        emoji=True,
    ),
    # ------------------------------------------------------------------ #
    # 5. Karaoke fill: every word fills in as it is spoken, mid-size,
    #    sentence case, thin outline, low on the frame.
    # ------------------------------------------------------------------ #
    "karaoke": CaptionPreset(
        key="karaoke",
        label="Karaoke fill",
        font_override="Poppins",
        font_file="Poppins-ExtraBold.ttf",
        font_size=64,
        primary="#FFFFFF",
        secondary="#39FF14",
        outline_color="#101010",
        outline_width=3,
        shadow=1,
        box=False,
        uppercase=False,
        words_per_cue=4,
        words_per_cue_max=5,
        pause_sec=0.35,
        line_layout=1,
        position_y=0.78,
        position_mode="frame",
        animation="karaoke",
    ),
    # ------------------------------------------------------------------ #
    # 6. Clean lower third: small, sentence case, thin outline, one line,
    #    near the bottom, no animation at all.
    # ------------------------------------------------------------------ #
    "lower_third": CaptionPreset(
        key="lower_third",
        label="Clean lower third",
        font_override="Poppins",
        font_file="Poppins-ExtraBold.ttf",
        font_size=48,
        primary="#FFFFFF",
        secondary="#FFFFFF",
        outline_color="#000000",
        outline_width=2,
        shadow=2,
        box=False,
        uppercase=False,
        words_per_cue=6,
        words_per_cue_max=8,
        pause_sec=0.5,
        line_layout=1,
        max_chars_per_line=34,
        position_y=0.86,
        position_mode="frame",
        animation="",
    ),
    # ------------------------------------------------------------------ #
    # 7. Boxed: opaque black plate behind ALL CAPS text, no outline, no
    #    shadow — the opposite end of the visual range from bold stack.
    # ------------------------------------------------------------------ #
    "boxed": CaptionPreset(
        key="boxed",
        label="Boxed",
        font_override="Montserrat",
        font_file="Montserrat-Bold.ttf",
        font_size=60,
        primary="#FFFFFF",
        secondary="#39FF14",
        outline_color="#000000",
        outline_width=0,
        shadow=0,
        box=True,
        uppercase=True,
        words_per_cue=4,
        words_per_cue_max=5,
        pause_sec=0.3,
        line_layout=1,
        max_chars_per_line=22,
        position_y=0.76,
        position_mode="frame",
        animation="",
        emoji=True,
    ),
}

#: Presets in the order the UI lists them.
PRESET_ORDER: tuple[str, ...] = (
    "bold_stack",
    "viral",
    "vertical_916",
    "bold_pop",
    "karaoke",
    "lower_third",
    "boxed",
)

#: The default look when nothing else is chosen.
DEFAULT_PRESET = "bold_stack"


def get_preset(key: str | None, highlight_color: str | None = None) -> CaptionPreset:
    """Resolve a preset by key, falling back to the default (never raises).

    ``highlight_color`` replaces the preset's own accent. Empty keeps the
    preset's accent, so switching presets does not inherit a colour chosen for
    a different one.
    """
    preset = CAPTION_PRESETS.get(key or "", CAPTION_PRESETS[DEFAULT_PRESET])
    color = (highlight_color or "").strip()
    if not color or color.upper() == preset.secondary.upper():
        return preset
    return replace(preset, secondary=color)
