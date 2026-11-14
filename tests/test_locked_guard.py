"""Regression tests for the lock guard and the stage/framing contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from app import reframe


def _guard_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "check_locked.py"
    spec = importlib.util.spec_from_file_location("check_locked_guard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_guard_fails_on_a_changed_locked_file(tmp_path, monkeypatch):
    """The real check is exercised without touching a locked source file."""
    guard = _guard_module()
    target = tmp_path / "locked.txt"
    target.write_text("the exact look\n", encoding="utf-8")
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "MANIFEST", tmp_path / "locked.sha256")
    monkeypatch.setattr(guard, "FILES", ("locked.txt",))
    guard.MANIFEST.write_text(
        f"{guard.digest(target)}\tlocked.txt\n", encoding="utf-8"
    )
    assert guard.verify() == 0
    target.write_text("the exact look plus one byte\n", encoding="utf-8")
    assert guard.verify() == 1


def test_reframing_is_stage_four_not_stage_one():
    from app.jobstatus import describe

    job = {
        "stage": "reframing",
        "status": "running",
        "unitsDone": 3,
        "unitsTotal": 21,
        "progress": 14,
        "lastProgressAt": 1,
    }
    out = describe(job)
    assert out["activeStage"] == "rendering"
    assert out["stageNumber"] == 4
    assert "3/21" in str(out["stageLabel"])


def test_export_never_passes_an_unsupported_job_id_to_update_clip():
    """The real crash: after the first clip, update_clip(job_id=…) killed export."""
    import ast
    import inspect
    import textwrap
    from app.task import Pipeline

    tree = ast.parse(textwrap.dedent(inspect.getsource(Pipeline._stage_export)))
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "update_clip":
            if any(kw.arg == "job_id" for kw in node.keywords):
                bad.append(ast.unparse(node)[:160])
    assert not bad, bad


def test_missing_mediapipe_uses_the_locked_fit_fallback(monkeypatch):
    monkeypatch.setattr(reframe, "_load_mediapipe", lambda: None)
    stats: list[dict] = []
    windows, used = reframe.face_crop_windows(
        "tests/golden/fixtures/look_source.mp4", 0.0, 5.0, 1280, 720,
        shot_stats=stats,
    )
    assert used is False
    assert len(windows) == 1
    assert stats and stats[0]["mode"] == "fit"
    assert "MediaPipe unavailable" in stats[0]["reason"]
