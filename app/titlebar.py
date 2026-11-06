# LOCKED: do not modify — see LOCKED_SPEC.md §2 (white title bar).
"""The white title bar: one PNG per clip, overlaid for the whole clip.

Reference look (measured at 1080x1920): a FULL-WIDTH white bar near the top,
about y=160 to 230, bold black centered text. The bar is drawn by Pillow
rather than libass for one reason: wrapping must be measured, not guessed.
libass cannot tell us a line's pixel width before it renders it, so a long
hook used to run off both edges. Here every candidate line is measured with
the SAME font file the render uses, the font shrinks until the title fits in
two lines inside 90 % of the frame, and the bar grows downward when a second
line is needed. Nothing can overflow an edge: at the minimum font size a
single over-long word is ellipsised.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FRAME_W = 1080
FRAME_H = 1920
BAR_TOP = 160
BAR_MIN_HEIGHT = 70
MAX_WIDTH_FRACTION = 0.90
MAX_LINES = 2
FONT_SIZE = 40
FONT_MIN_SIZE = 22
FONT_STEP = 2
PAD_X = 24
PAD_Y = 12
LINE_FACTOR = 1.18
FONT_FILE = Path(__file__).resolve().parent / "fonts" / "Montserrat-ExtraBold.ttf"
TEXT_COLOR = (10, 10, 10)
BAR_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class TitlePlan:
    lines: tuple[str, ...]
    font_size: int
    height: int
    top: int = BAR_TOP
    width: int = FRAME_W

    @property
    def bottom(self) -> int:
        return self.top + self.height


def _font(size: int):
    from PIL import ImageFont  # noqa: PLC0415 - Pillow ships with the app

    return ImageFont.truetype(str(FONT_FILE), int(size))


def measure(text: str, size: int) -> int:
    font = _font(size)
    box = font.getbbox(text)
    return int(box[2] - box[0])


def wrap(text: str, size: int, limit: int) -> list[str]:
    """Greedy word wrap at a measured pixel width (never splits a word)."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = f"{cur} {word}".strip()
        if not cur or measure(candidate, size) <= limit:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def _ellipsise(word: str, size: int, limit: int) -> str:
    """Clip a word that is wider than the limit even at the minimum size."""
    out = word
    while out and measure(out + "…", size) > limit:
        out = out[:-1]
    return (out + "…") if out else "…"


def plan(title: str, *, frame_w: int = FRAME_W) -> TitlePlan:
    """Fit `title` into at most 2 lines inside 90 % of the frame width."""
    text = " ".join(str(title or "").split())
    limit = int(frame_w * MAX_WIDTH_FRACTION)
    size = FONT_SIZE
    lines: list[str] = [text]
    while True:
        lines = wrap(text, size, limit)
        widest = max((measure(line, size) for line in lines), default=0)
        if (len(lines) <= MAX_LINES and widest <= limit) or size <= FONT_MIN_SIZE:
            break
        size = max(FONT_MIN_SIZE, size - FONT_STEP)
    if len(lines) > MAX_LINES:
        # Keep the first two lines and ellipsise the tail onto the second.
        head, tail = lines[:1], " ".join(lines[1:])
        lines = head + [tail]
    if len(lines) == 2 and measure(lines[1], size) > limit:
        # Last resort at the minimum size: ellipsise the second line.
        words = lines[1].split()
        while len(words) > 1 and measure(" ".join(words) + "…", size) > limit:
            words.pop()
        tail = " ".join(words)
        lines[1] = tail + "…" if measure(tail + "…", size) <= limit else _ellipsise(tail, size, limit)
    if len(lines) == 1 and measure(lines[0], size) > limit:
        lines[0] = _ellipsise(lines[0].split()[0], size, limit)
    line_h = int(round(size * LINE_FACTOR))
    height = max(BAR_MIN_HEIGHT, PAD_Y * 2 + line_h * len(lines))
    return TitlePlan(tuple(lines), size, int(height), BAR_TOP, int(frame_w))


def render_png(plan_in: TitlePlan, path: Path) -> Path:
    """Draw the white bar + bold black title. Returns the PNG path."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (plan_in.width, plan_in.height), BAR_COLOR)
    draw = ImageDraw.Draw(img)
    font = _font(plan_in.font_size)
    line_h = int(round(plan_in.font_size * LINE_FACTOR))
    block_h = line_h * len(plan_in.lines)
    y = (plan_in.height - block_h) // 2
    for line in plan_in.lines:
        box = draw.textbbox((0, 0), line, font=font)
        w = int(box[2] - box[0])
        x = max(PAD_X, int((plan_in.width - w) / 2))
        draw.text((x, y - int(box[1])), line, font=font, fill=TEXT_COLOR)
        y += line_h
    img.save(path, "PNG")
    return path


def first_sentence(cue_texts: list[str]) -> str:
    """Fallback title: the transcript's first sentence (never placeholder text)."""
    joined = " ".join(t.strip() for t in cue_texts if t and t.strip())
    if not joined:
        return ""
    for stop in (".", "!", "?", "…"):
        idx = joined.find(stop)
        if 0 <= idx < 160:
            return joined[: idx + 1].strip()
    words = joined.split()
    return " ".join(words[:10]).strip()
