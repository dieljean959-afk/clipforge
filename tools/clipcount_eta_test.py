"""Clip count scales with the source; the time plan tells the truth per stage.

Two claims that are easy to get subtly wrong and hard to notice:

  1. A longer source gets more clips, AND a short one gets fewer than the old
     flat 10 — otherwise "dynamic" is just a bigger constant.
  2. The count is a CEILING and the score floor is a GATE. If a long source
     with nothing good in it still returns the full quota, the feature is
     padding, and the batch is worse than the flat count it replaced.

Plus the ETA plan, which has to be per-stage and has to be labelled with where
each number came from — a prior shown without its provenance is a guess
presented as a measurement.

Run:  .venv/bin/python tools/clipcount_eta_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import clipcount, eta  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


def section_scaling() -> None:
    print("\n[1] the count follows the source length")
    at = lambda m, ceiling=40: clipcount.suggest(m * 60, ceiling)[0]  # noqa: E731

    check("a 2-minute video is worth fewer than 10 clips", at(2) < 10, str(at(2)))
    check("a 5-minute video is fewer than a 9-minute one", at(5) < at(9), f"{at(5)} vs {at(9)}")
    check("10 min is the first band boundary and steps up", at(10) > at(9.9), f"{at(10)} vs {at(9.9)}")
    # The bands MEET at their shared endpoint (25 min is the top of one band and
    # the bottom of the next, both 15), so the count is continuous there. What
    # must not happen is a dip at a boundary.
    check("25 min is continuous, not a dip", at(25) >= at(24.9) >= at(24), f"{at(24.9)} {at(25)}")
    check("60 min is continuous, not a dip", at(60) >= at(59.9) >= at(59), f"{at(59.9)} {at(60)}")
    check("the band label changes with the source, and names the band",
          "10-25 min" in clipcount.suggest(20 * 60, 40)[1]
          and "25-60 min" in clipcount.suggest(40 * 60, 40)[1]
          and "60+ min" in clipcount.suggest(80 * 60, 40)[1],
          f"{clipcount.suggest(20*60, 40)[1]} | {clipcount.suggest(40*60, 40)[1]}")

    check("the count is monotonic across the whole 2-120 min range",
          all(at(a) <= at(b) for a, b in zip(range(2, 120), range(3, 121))),
          str([at(m) for m in (2, 10, 25, 60, 90, 120)]))
    check("a 5 min video lands in the 5-8 band", 5 <= at(5) <= 8, str(at(5)))
    check("a 17 min video lands in the 10-15 band", 10 <= at(17) <= 15, str(at(17)))
    check("a 45 min video lands in the 15-25 band", 15 <= at(45) <= 25, str(at(45)))
    check("a 90 min video lands in the 25-40 band", 25 <= at(90) <= 40, str(at(90)))
    check("the 120 min maximum reaches the ceiling", at(120) == 40, str(at(120)))
    check("nothing exceeds the hard ceiling", max(at(m) for m in range(2, 121)) <= 40)

    check("an unknown duration does not guess high",
          clipcount.suggest(0)[0] <= 10, str(clipcount.suggest(0)))
    check("a zero or negative duration is still a positive count",
          clipcount.suggest(-5)[0] >= 1)


def section_override() -> None:
    print("\n[2] a manual cap always wins")
    check("a ceiling of 8 holds a 90-minute video at 8",
          clipcount.suggest(90 * 60, 8)[0] == 8, str(clipcount.suggest(90 * 60, 8)))
    check("a ceiling below the band still wins", clipcount.suggest(3 * 60, 3)[0] == 3)
    check("the reason says the cap was held, not that it was scaled",
          "max" in clipcount.suggest(90 * 60, 8)[1], clipcount.suggest(90 * 60, 8)[1])
    check("without a cap the reason explains the scaling",
          "min source" in clipcount.suggest(45 * 60, 40)[1], clipcount.suggest(45 * 60, 40)[1])
    check("a ceiling above the hard ceiling is clamped",
          clipcount.suggest(90 * 60, 500)[0] <= clipcount.HARD_CEILING)


def section_gate() -> None:
    print("\n[3] the count is a ceiling and the floor is a gate")
    cands = [{"score": s} for s in (95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 40, 20)]

    top, above, below = clipcount.gate(cands, limit=40, floor=60.0)
    check("everything at or above the floor is kept", above == 8, str(above))
    check("everything below the floor is dropped", below == 4, str(below))
    check("all eight are returned when the cap is not binding", len(top) == 8, str(len(top)))
    check("the survivors are sorted by score", [c["score"] for c in top] == sorted(
        (c["score"] for c in top), reverse=True))

    top, above, below = clipcount.gate(cands, limit=3, floor=60.0)
    check("the cap trims the survivors to 3", len(top) == 3, str(len(top)))
    check("the cap keeps the BEST three", [c["score"] for c in top] == [95, 90, 85], str(top))
    check("the count before the cap is still reported", above == 8, str(above))
    check("dropping to the cap is not counted as 'below the floor'", below == 4, str(below))

    weak = [{"score": 41}, {"score": 38}, {"score": 30}]
    top, above, below = clipcount.gate(weak, limit=10, floor=60.0)
    check("a batch where nothing clears the floor still returns its best moment",
          len(top) == 1 and top[0]["score"] == 41, str(top))
    check("but it is still reported as clearing nothing", above == 0, str(above))
    check("so every candidate is reported below the floor", below == 3, str(below))

    check("no candidates at all returns nothing rather than raising", clipcount.gate([], 5, 60.0)[0] == [])
    one = [{"score": 100}]
    check("a single strong candidate survives", clipcount.gate(one, 40, 60.0)[0] == one)
    check("the floor boundary itself counts as clearing it",
          clipcount.gate([{"score": 60.0}], 5, 60.0)[1] == 1)
    check("a hair under the floor does not",
          clipcount.gate([{"score": 59.9}], 5, 60.0)[1] == 0)


def section_plan() -> None:
    print("\n[4] the time plan is per stage, and says where each number came from")
    kw = dict(duration_sec=3300.0, clip_count=23, clip_seconds=60.0, rtf=0.3)

    p = eta.build_plan(current_stage="transcribing", **kw)
    check("a transcribing job lists transcribe first",
          p[0]["stage"] == "transcribe" and p[0]["current"], str(p[0]))
    check("finished stages are not listed", "fetch" not in [x["stage"] for x in p])
    check("the render stage is still listed ahead", p[-1]["stage"] == "render", str(p[-1]))
    check("every entry carries a label, a number and a provenance",
          all(x.get("label") and isinstance(x.get("sec"), (int, float)) and x.get("source")
              for x in p), str(p))
    check("nothing is claimed as measured before it has run",
          all(x["source"] in {"estimate", "live"} for x in p), str([x["source"] for x in p]))

    p = eta.build_plan(current_stage="transcribing", stage_eta_sec=600.0, **kw)
    check("the current stage's row is the live number", p[0]["sec"] == 600.0 and p[0]["source"] == "live")

    p = eta.build_plan(current_stage="transcribing",
                       measured={"fetch": 12.0}, **kw)
    check("a stage that has already run is not repeated in the plan",
          "fetch" not in [x["stage"] for x in p])

    p = eta.build_plan(current_stage="rendering", stage_eta_sec=480.0,
                       measured={"transcribe": 900.0, "detect": 140.0}, **kw)
    check("a rendering job only lists rendering", [x["stage"] for x in p] == ["render"], str(p))
    check("and the render row is the live estimate", p[0]["source"] == "live")

    trans = eta.stage_prior_sec("transcribe", **kw)
    det = eta.stage_prior_sec("detect", **kw)
    ref = eta.stage_prior_sec("reframe", **kw)
    ren = eta.stage_prior_sec("render", **kw)
    check("transcription scales with audio length",
          trans == 3300.0 * 0.3, str(trans))
    check("detection scales with the number of 2-minute chunks",
          20 < det < 200, str(det))
    check("reframing is cheaper than detection", ref < det, f"{ref} vs {det}")
    check("rendering scales with the clip count",
          eta.stage_prior_sec("render", **{**kw, "clip_count": 40}) > ren)
    check("a longer source means a longer transcription prior",
          eta.stage_prior_sec("transcribe", **{**kw, "duration_sec": 6600.0}) > trans)
    check("a faster host means a shorter transcription prior",
          eta.stage_prior_sec("transcribe", **{**kw, "rtf": 0.05}) < trans)
    check("the plan total is the sum of its rows",
          abs(eta.plan_total_sec(eta.build_plan(current_stage="transcribing", **kw))
              - sum(r["sec"] for r in eta.build_plan(current_stage="transcribing", **kw))) < 0.5)

    check("every pipeline stage maps onto a plan row",
          all(eta.plan_stage_for(s) in eta.STAGE_ORDER
              for s in ("queued", "grabbing", "transcribing", "finding", "reframing",
                        "rendering", "exporting", "complete", "failed")))

    # Recorded timings are rewritten on every stage change, so an already-mapped
    # name is fed back through the mapper. Without the identity case a whole job's
    # history collapses into one "fetch" row (12 rows, 269 s — the first E2E run).
    check("mapping a plan stage again does not move it",
          all(eta.plan_stage_for(eta.plan_stage_for(s)) == eta.plan_stage_for(s)
              for s in ("queued", "transcribing", "finding", "reframing", "rendering",
                        *eta.STAGE_ORDER)))
    check("a plan stage is returned unchanged, not defaulted to fetch",
          [eta.plan_stage_for(s) for s in eta.STAGE_ORDER] == list(eta.STAGE_ORDER),
          str([eta.plan_stage_for(s) for s in eta.STAGE_ORDER]))
    check("a real stage name survives a round trip through the plan",
          eta.plan_stage_for("transcribing") == "transcribe"
          and eta.plan_stage_for(eta.plan_stage_for("transcribing")) == "transcribe")


def section_units() -> None:
    print("\n[5] per-unit extrapolation for the clip loop")
    check("no estimate before the first unit finishes", eta.eta_from_units(10, 0, 20) is None)
    check("3 of 20 clips in 300 s is ~17 min left", abs(eta.eta_from_units(300, 3, 20) - 1700) < 1)
    check("the last clip leaves nothing", eta.eta_from_units(300, 20, 20) == 0.0)
    check("one clip of one leaves nothing", eta.eta_from_units(300, 1, 1) == 0.0)
    check("a slower start extrapolates slower",
          eta.eta_from_units(600, 3, 20) > eta.eta_from_units(300, 3, 20))
    check("missing units give no answer rather than a wrong one",
          eta.eta_from_units(300, None, 20) is None and eta.eta_from_units(300, 3, None) is None)


def main() -> int:
    print("CLIP COUNT SCALING + PER-STAGE TIME PLAN")
    section_scaling()
    section_override()
    section_gate()
    section_plan()
    section_units()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
