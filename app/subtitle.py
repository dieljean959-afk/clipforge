# --------------------------------------------------------------------------- #
# Chunking: 3-5 words per burst, split into 2 balanced lines (1-3 words each)
# --------------------------------------------------------------------------- #

_BREAK_AFTER = (".", "!", "?", ",", ";", ":", "…")


def _chunk(words: list[dict]) -> list[list[dict]]:
    """Group word timings into 2-4 word chunks under the brief's rules.

    A chunk closes when any of:
      * the word count reaches STYLE_CHUNK_MAX_WORDS (5),
      * a pause of STYLE_PACE_BREAK_SEC (0.4 s) or more,
      * the previous word ends a sentence or clause (``. ! ? , ; :``),
      * the run of speech exceeds STYLE_CHUNK_SEC (2.6 s).
    Single-word end chunks are merged into a neighbour so a stray word never
    gets a caption of its own (and never leaves a lone word on screen).
    """
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        text = str(w.get("word") or "").strip()
        if not text:
            continue
        if cur:
            gap = float(w.get("start") or 0.0) - float(cur[-1].get("end") or 0.0)
            joined = " ".join(str(x.get("word") or "") for x in cur + [w])
            prev = str(cur[-1].get("word") or "").rstrip("\"'”’")
            if (
                len(cur) >= STYLE_CHUNK_MAX_WORDS
                or gap >= STYLE_PACE_BREAK_SEC
                or prev.rstrip().endswith(_BREAK_AFTER)
                or len(joined) > STYLE_MAX_CHARS_PER_LINE * 2
            ):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)

    # Never leave a one-word chunk: merge it into a neighbour when that keeps
    # every line inside the 3-5 word / character budget.
    merged: list[list[dict]] = []
    for c in chunks:
        if (
            len(c) < STYLE_CHUNK_MIN_WORDS
            and merged
            and len(" ".join(str(x.get("word") or "") for x in merged[-1] + c)) <= STYLE_MAX_CHARS_PER_LINE * 2
            and len(merged[-1]) + len(c) <= STYLE_CHUNK_MAX_WORDS
        ):
            merged[-1] = merged[-1] + c
            continue
        merged.append(c)

    for i, c in enumerate(merged):
        if len(c) != 1:
            continue
        for j in (i + 1, i - 1):
            if not (0 <= j < len(merged)) or not merged[j]:
                continue
            if len(merged[j]) < STYLE_CHUNK_MIN_WORDS:
                continue
            take = merged[j][0] if j > i else merged[j][-1]
            cand = c + [take] if j > i else [take] + c
            if len(" ".join(str(x.get("word") or "") for x in cand)) > STYLE_MAX_CHARS_PER_LINE * 2:
                continue
            merged[i] = cand
            merged[j] = merged[j][1:] if j > i else merged[j][:-1]
            break

    out: list[list[dict]] = []
    for i, c in enumerate(merged):
        if len(c) < STYLE_CHUNK_MIN_WORDS and i + 1 < len(merged) and len(c) + len(merged[i + 1]) <= STYLE_CHUNK_MAX_WORDS:
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
    if len(words) <= 1 or len(" ".join(words)) <= STYLE_MAX_CHARS_PER_LINE:
        return None
    best: int | None = None
    best_score: int | None = None
    for i in range(1, len(words)):
        a = " ".join(words[:i])
        b = " ".join(words[i:])
        if len(a) > STYLE_MAX_CHARS_PER_LINE or len(b) > STYLE_MAX_CHARS_PER_LINE:
            continue
        score = abs(len(a) - len(b))
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


def _line_words(words: list[str], break_at: int | None) -> tuple[str, str]:
    """Splits a chunk's words into two lines.

    Given the plain words of a burst and the precomputed break index, returns
    (line1, line2). If the break index is None or the words do not fit the
    per-line cap on a split, returns (whole, "") — the caller owns how a
    single line renders.
    """
    if break_at is None or break_at >= len(words):
        return " ".join(words), ""
    return " ".join(words[:break_at]), " ".join(words[break_at:])


# --------------------------------------------------------------------------- #
# ASS event writer for the reference look
# --------------------------------------------------------------------------- #

def _line_height(scale: float) -> float:
    """Cap height in pixels for the current size scale (60 % of the 86 px pitch)."""
    return round(STYLE_FONT_SIZE * scale * 0.60)


def _pitch(scale: float) -> float:
    """Line pitch in pixels: cap height + gap = 86 * scale, as required."""
    return round(STYLE_LINE_PITCH * scale)


def _y_positions(cap_h: float, pitch: float, frame_h: int, scale: float) -> tuple[float, float]:
    """Return the y of the BOTTOM EDGE of line 1 and line 2.

    Reference layout: block spans y=1283..1404, centre ~70 % of the frame. We
    choose the bottom edge of line 2 so the centre of the block lands at about
    70 % of the frame height: the two adjacent ASS Dialogue lines are the only
    thing in the block, so the visual centre is (cap_h + pitch) / 2 from the
    line-bottoms. With cap_h = 0.60*86*scale and pitch = 86*scale.
    """
    centre = frame_h * 0.70
    # line2_bottom + cap_h + gap = centre   (gap = pitch - cap_h)
    # => line2_bottom = centre - cap_h - (pitch - cap_h)
    gap = pitch - cap_h
    line2_bottom = centre - cap_h - gap
    line1_bottom = line2_bottom - pitch
    return line1_bottom, line2_bottom


def _size_for_preset(preset: CaptionPreset, scale: float) -> int:
    """Apply the size-scale knob: font size, outline width and pitch all scale together."""
    s = max(SIZE_SCALE_MIN, min(SIZE_SCALE_MAX, scale))
    return max(1, int(round(preset.font_size * s)))


def _build_bold_stack_events(
    cues: list[dict],
    settings: Settings,
    start_offset: float = 0.0,
    anchor_y: int | None = None,
) -> str:
    """One Dialogue line per stacked line, with \an2\pos(540, y).

    Each cue is a 3-5 word burst split into two balanced lines. The ASS events
    file therefore carries TWO Dialogue lines per burst: line 1 at
    \pos(540, line1_bottom), line 2 at \pos(540, line2_bottom), Pitch 86 px.

    Colour cycling: the COLOURS are cycled per chunk, and within a chunk the
    FIRST line carries the colour while the SECOND is white (the reference
    shows the colour on alternating lines from chunk to chunk).
    """
    colours = list(STYLE_COLORS)
    parts: list[str] = []
    for c in cues:
        text = str(c.get("text") or "").upper()
        words = [w for w in text.split() if w]
        if not words:
            continue
        break_at = _wrap_index(words)
        line1, line2 = _line_words(words, break_at)

        scale = float(getattr(settings.subtitle, "size_scale", SIZE_SCALE_DEFAULT))
        sizes = {
            "l1": _size_for_preset(_get_preset(settings, None), scale),
            "l2": _size_for_preset(_get_preset(settings, None), scale),
        }

        cap_h = _line_height(scale)
        pitch = _pitch(scale)
        l1_bottom, l2_bottom = _y_positions(cap_h, pitch, int(settings.video.height), scale)

        # Colour the first line, white the second.
        c1 = colours.pop(0) if colours else colours[0]
        colours.append(c1)

        def _tags(text: str, color: str) -> str:
            return rf"{{\\1c{_hex_to_ass(color)}&}}{text}{{\\1c&H00FFFFFF&}}"

        l1txt = _tags(line1, c1) if line1 else ""
        l2txt = _tags(line2, _hex_to_ass("#FFFFFF")) if line2 else ""

        # Build the ASS event strings.
        start1 = _ass_time(c["start"] - start_offset)
        end1 = _ass_time(c["end"] - start_offset)
        parts.append(
            f"Dialogue: 0,{start1},{end1},{_get_preset(settings, None).key},,0,0,0,,"
            rf"{{\\an2\\pos({int(settings.video.width) // 2},{int(l1_bottom)})}}{l1txt}"
        )
        if line2:
            start2 = _ass_time(c.get("start", c["start"]) - start_offset)
            end2 = _ass_time(c.get("end", c["end"]) - start_offset)
            parts.append(
                f"Dialogue: 0,{start2},{end2},{_get_preset(settings, None).key},,0,0,0,,"
                rf"{{\\an2\\pos({int(settings.video.width) // 2},{int(l2_bottom)})}}{l2txt}"
            )
    return "\n".join(parts) + "\n"


def build_ass_events(
    cues: list[dict], settings: Settings, start_offset: float = 0.0,
    preset_key: str | None = None, anchor_y: int | None = None,
    frame_w: int = 1080,
) -> str:
    p = get_preset(preset_key, settings.subtitle.highlight_color)
    if preset_key == "bold_stack":
        return _build_bold_stack_events(
            cues, settings, start_offset, anchor_y=anchor_y
        )
    # The highlight colour lives on the caption track: the ASS `subtitles`
    # filter does NOT read [Events] colour tags, so we carry the preset's
    # own accent as the highlight colour for this clip's cues — that's the
    # colour the word being spoken gets painted.
    highlight_color = p.secondary if p.secondary else settings.subtitle.highlight_color
    lines: list[str] = []
    for c in cues:
        start = _ass_time(c["start"] - start_offset)
        end = _ass_time(c["end"] - start_offset)
        text = str(c.get("text") or "").replace("{", "(").replace("}", ")")
        if p.pop:
            text = (
                r"{\fad(60,40)\fscx84\fscy84\t(0,90,\fscx105\fscy105)"
                r"\t(90,140,\fscx100\fscy100)}" + text
            )
        if c.get("words") and not c.get("uncertain"):
            text = _build_viral_cue(c, highlight_color)
        if anchor_y is not None:
            text = rf"{{\an2\pos({frame_w // 2},{int(anchor_y)})}}" + text
        lines.append(
            f"Dialogue: 0,{start},{end},{p.key},,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"
