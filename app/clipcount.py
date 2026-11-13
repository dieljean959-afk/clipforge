"""How many clips a source is worth.

A flat "10 clips for everything" is wrong in both directions. A four-minute
clip has maybe two moments worth keeping and rendering ten would pad the batch
with the leftovers; a one-hour interview has an hour of material and stops at
ten only because the number was typed into a config file in 2024.

The count is a CEILING, not a quota. `detect_clips` still drops anything under
the score floor, so a long source that is genuinely ninety minutes of ad-break
monologue returns three clips, not twenty-five — and the job reports both
numbers so the difference between "the cap" and "what was actually good" is
visible rather than something the user has to guess at.
"""

from __future__ import annotations

#: (min_seconds, max_seconds, clips_at_min, clips_at_max, label)
#: Read as "a source in this band is worth between N and M clips". The value is
#: interpolated by position in the band, so a 12-minute video does not jump
#: straight to the top of its range.
BANDS: tuple[tuple[float, float, int, int, str], ...] = (
    (0, 600, 5, 8, "under 10 min"),
    (600, 1500, 10, 15, "10-25 min"),
    (1500, 3600, 15, 25, "25-60 min"),
    (3600, 7200, 25, 40, "60+ min"),
)

#: The last band tops out here. Raising it means every prompt, every slider
#: and every cost estimate has to be re-checked, so it is a deliberate ceiling
#: rather than an accident of arithmetic.
HARD_CEILING = 40


def band_for(duration_sec: float) -> tuple[int, int, int, int, str]:
    """The (lo, hi, clips_lo, clips_hi, label) band a duration falls into."""
    d = max(0.0, float(duration_sec or 0.0))
    for lo, hi, c_lo, c_hi, label in BANDS:
        if d < hi or hi >= BANDS[-1][1]:
            return lo, hi, c_lo, c_hi, label
    return BANDS[-1]


def _fmt_minutes(sec: float) -> str:
    m = sec / 60
    # "10 min source (under 10 min)" reads like a bug. Below the hour, keep the
    # decimal that the band boundary is actually tested against.
    return f"{m:.0f}" if m >= 10 else f"{m:.1f}"


def suggest(duration_sec: float, ceiling: int = HARD_CEILING) -> tuple[int, str]:
    """(clip ceiling for this source, one-line reason for the job log).

    `ceiling` is the configured maximum, and it always wins when it is lower:
    a user who asks for 8 clips on a 90-minute video gets 8, because the
    override is the point of having one.
    """
    limit = max(1, min(int(ceiling or HARD_CEILING), HARD_CEILING))
    d = max(0.0, float(duration_sec or 0.0))
    lo, hi, c_lo, c_hi, label = band_for(d)
    if d <= 0:
        # Duration unknown (a failed probe). Assume the middle of the range
        # rather than guessing high — a wrong guess costs an hour of encoding.
        return min(limit, 10), "source length unknown, assuming a ~15 min video"
    if hi <= lo:
        count = c_lo
    else:
        position = min(1.0, max(0.0, (d - lo) / (hi - lo)))
        count = c_lo + round((c_hi - c_lo) * position)
    count = max(1, min(limit, count))
    mins = _fmt_minutes(d)
    if count >= limit:
        return count, f"{mins} min source ({label}) — held at your max of {limit}"
    return count, f"{mins} min source ({label}) — up to {count} clips"


def candidate_target(requested: int, *, exact: bool) -> int:
    """How many candidates to ask detection for when N is a quota.

    N + 40% (at least N + 3) gives render failures a real replacement moment
    instead of silently returning a short batch. Auto counts are ceilings, not
    quotas, so they keep asking for just the ceiling.
    """
    n = max(1, int(requested or 1))
    if not exact:
        return n
    return n + max(3, (n * 2 + 4) // 5)


def reasonable_clips(
    candidates: list[dict],
    *,
    max_clips: int | None = None,
    fallback: int | None = None,
) -> list[dict]:
    """Trim a candidate list to a job-safe batch.

    `gate` is the scoring gate (score floor + cap). This is the defensive
    wrapper the render stage wants: take what detection produced, hard-cap it,
    and never return nothing when the callers have no other clip source —
    detection already fell back to heuristics upstream, so an empty list here
    would only mean detection genuinely produced zero moments.
    """
    if not candidates:
        return []
    limit = max_clips if max_clips and max_clips > 0 else fallback
    if limit and limit > 0:
        return candidates[: int(limit)]
    return list(candidates)


def gate(candidates: list[dict], limit: int, floor: float) -> tuple[list[dict], int, int]:
    """Keep the strong candidates, up to `limit` — never pad to reach it.

    Returns (selected, cleared_the_floor, below_the_floor).

    The floor is what makes this a gate rather than a quota. `cleared_the_floor`
    is the number that says "detection found more good moments than the cap
    allowed", which is the one case where the cap is what cost the user
    something — worth knowing, and worth showing.

    When NOTHING clears the floor the batch is kept anyway (the best few), so a
    rubric disagreeing with the whole video cannot fail a job outright. The
    counts still say so honestly: `cleared` stays 0 and every candidate is
    reported as below the floor, which is the signal to go and retune
    `min_score` rather than to conclude the batch was good.
    """
    cleared = [c for c in candidates if float(c.get("score", 0)) >= floor]
    chosen = cleared
    if not chosen and candidates:
        strongest = max(float(c.get("score", 0)) for c in candidates)
        chosen = [c for c in candidates if float(c.get("score", 0)) >= strongest]
    chosen.sort(key=lambda c: float(c.get("score", 0)), reverse=True)
    return chosen[:limit], len(cleared), len(candidates) - len(cleared)
