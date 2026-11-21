"""The job's own stage names must land on the right plan row.

A missing alias does not fail loudly: `plan_stage_for` falls back to "fetch",
so the dashboard claims the job is still downloading and the ETA is computed
from the wrong anchor. That is what happened while the 81-minute batch was in
its captions stage after a host restart.
"""

from __future__ import annotations

from app import eta


def test_captions_and_export_are_part_of_the_render_row():
    assert eta.plan_stage_for("captions") == "render"
    assert eta.plan_stage_for("export") == "render"
    assert eta.plan_stage_for("exporting") == "render"


def test_plan_starts_at_the_running_stage_not_at_download():
    plan = eta.build_plan(
        current_stage="captions",
        duration_sec=4869.0,
        clip_count=28,
        clip_seconds=45.0,
        stage_eta_sec=600.0,
        measured={"fetch": 1090.8, "transcribe": 491.7, "detect": 226.9,
                  "reframe": 80.0},
    )
    assert [row["stage"] for row in plan] == ["render"]
    assert plan[0]["current"] is True
    assert plan[0]["sec"] == 600.0
