# LOCKED: do not modify — see LOCKED_SPEC.md §1 (ONE caption style).
"""The one layout helper left from the animated styles: the FIT foreground edge.

The animated caption styles ("Animated · Bold Yellow", karaoke fill, pop,
boxed, viral …) are gone — LOCKED_SPEC §1 keeps exactly one caption style. What
survives here is the geometry they were the only place to compute: the lower
edge of the centred video on a FIT clip, which is where the caption block is
anchored (``captionBottomY``) and where the emoji layer hangs from.
"""

from __future__ import annotations


def foreground_bottom(
    src_w: int,
    src_h: int,
    frame_w: int = 1080,
    frame_h: int = 1920,
    inset: int = 24,
) -> int | None:
    """Y of the caption block's bottom edge at the FIT foreground's lower edge.

    A FIT clip scales the source to fill the frame's WIDTH and centres it
    vertically over a blurred copy of itself. So the sharp video occupies a
    band in the middle of the frame, and ``frame_h - band`` is all blur. A
    caption placed by bottom margin lands in that blur, BELOW the video — it
    reads as a separate graphic rather than something the speaker is saying.

    Measured from the real numbers, never hardcoded: a 16:9 source gives
    fg_h = 608, fg_top = 656, fg_bottom = 1264, so ``captionBottomY`` is 1240.

    Returns None for a source that already fills the frame (portrait): there is
    no blurred band, so the full-frame position applies instead.
    """
    frame_w = int(frame_w)
    frame_h = int(frame_h)
    fg_h = int(round(frame_w * float(src_h) / float(src_w) / 2.0) * 2)
    fg_h = min(fg_h, frame_h)
    if fg_h >= frame_h:
        return None
    fg_top = (frame_h - fg_h) // 2
    return int(fg_top + fg_h - int(inset))
