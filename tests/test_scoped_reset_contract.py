"""Regression tests for the scoped reset, source duration and caption cleanup."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from app import source_stream
from app.encode import cfr_args
from app.subtitle import clean_caption_text, clean_cues
from app.video import has_faststart


def test_real_transcript_markers_are_removed_but_speech_is_kept():
    # Copied from the real clip's transcript (Burpees, 185.6-204.5 s).
    assert (
        clean_caption_text("PUEDES. PUEDE SER... [INAUDIBLE]")
        == "PUEDES. PUEDE SER..."
    )
    assert clean_caption_text("[INAUDIBLE]") == ""
    assert clean_caption_text("(laughs) ESO ES TODO [Music]") == "ESO ES TODO"
    assert clean_caption_text("(inaudible)") == ""


def test_marker_only_cues_disappear_but_real_cues_keep_their_timing():
    cues = [
        {"start": 0.0, "end": 1.0, "text": "HOLA [INAUDIBLE]"},
        {"start": 1.0, "end": 2.0, "text": "[INAUDIBLE]"},
        {"start": 2.0, "end": 3.0, "text": "[MUTTERING]", "uncertain": True},
    ]
    out = clean_cues(cues)
    assert [c["text"] for c in out] == ["HOLA"]
    assert out[0]["start"] == 0.0 and out[0]["end"] == 1.0


def _box(name: bytes, payload: bytes) -> bytes:
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + name + payload


def test_faststart_reads_top_level_boxes(tmp_path: Path):
    good = tmp_path / "good.mp4"
    good.write_bytes(_box(b"ftyp", b"isom") + _box(b"moov", b"x") + _box(b"mdat", b"x"))
    assert has_faststart(good) is True
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(_box(b"ftyp", b"isom") + _box(b"mdat", b"x") + _box(b"moov", b"x"))
    assert has_faststart(bad) is False


def test_planned_section_end_is_not_the_source_duration():
    """A 20:37 source must not become 120:00 from 48 planned sections."""
    sections = [
        {"start": 0.0, "end": 150.08},
        {"start": 1050.0, "end": 1200.0},
        {"start": 1200.0, "end": 1237.0},
    ]
    assert source_stream.source_duration(probed=150.08, meta_sec=7200, sections=sections) == 1237.0


def test_cfr_args_use_a_flag_this_ffmpeg_understands():
    args = cfr_args(30)
    assert args[:2] == ["-r", "30"]
    assert "cfr" in args
    assert "-fps_mode" in args or "-vsync" in args


def test_bucket_plan_exposes_the_keys_it_planned_for_apply(monkeypatch):
    from types import SimpleNamespace

    from app import bucketreset, objectstore

    class Store:
        def list_jobs(self):
            return []

    class Cfg:
        enabled = True
        bucket = "b"
        prefix = "p"

    monkeypatch.setattr(objectstore, "config_from_env", lambda: Cfg())
    monkeypatch.setattr(
        objectstore, "list_prefix",
        lambda **_: [{
            "key": "p/uploads/orphan/video.mp4", "size": 10,
            "lastModified": datetime.now(timezone.utc) - timedelta(hours=2),
        }],
    )
    monkeypatch.setattr(objectstore, "client", lambda cfg: None)
    plan = bucketreset.plan(Store(), SimpleNamespace(hosted=SimpleNamespace(retention_hours=96)))
    assert plan["plannedBytes"] == 10
    assert plan["_keys"] == ["p/uploads/orphan/video.mp4"]


def test_share_token_is_stable_until_it_expires_or_is_revoked():
    from app import sharelink

    clip = {}
    token = sharelink.ensure_token(clip)
    assert len(token) >= 16
    assert all(ch.isalnum() or ch in "-_" for ch in token)
    assert sharelink.ensure_token(clip) == token
    assert sharelink.token_state(clip)["state"] == "ready"
    clip["shareExpiresAt"] = 1
    fresh = sharelink.ensure_token(clip)
    assert fresh != token
    assert sharelink.token_state(clip)["state"] == "ready"
    clip["shareRevoked"] = True
    assert sharelink.token_state(clip)["state"] == "revoked"
    assert sharelink.token_state(clip)["detail"] == "this share link was revoked"


def test_bucket_plan_protects_the_source_of_a_failed_clip(monkeypatch):
    from types import SimpleNamespace

    from app import bucketreset, objectstore

    class Store:
        def list_jobs(self):
            return [{"_id": "done", "stage": "complete", "status": "complete"}]

        def list_clips(self, job_id):
            return [{"_id": "c", "renderState": "failed"}]

    class Cfg:
        enabled = True
        bucket = "b"
        prefix = "p"

    monkeypatch.setattr(objectstore, "config_from_env", lambda: Cfg())
    monkeypatch.setattr(
        objectstore, "list_prefix",
        lambda **_: [{
            "key": "p/uploads/done/source.mp4", "size": 10,
            "lastModified": datetime.now(timezone.utc) - timedelta(days=2),
        }],
    )
    monkeypatch.setattr(objectstore, "client", lambda cfg: None)
    plan = bucketreset.plan(Store(), SimpleNamespace(hosted=SimpleNamespace(retention_hours=96)))
    assert plan["plannedBytes"] == 0
    assert "failed clip" in plan["kept"][0]["reason"]


def test_bucket_plan_protects_a_live_batch(monkeypatch):
    from types import SimpleNamespace

    from app import bucketreset, objectstore

    class Store:
        def list_jobs(self):
            return [{"_id": "live", "stage": "rendering", "status": "running"}]

        def list_clips(self, job_id):
            return []

    class Cfg:
        enabled = True
        bucket = "b"
        prefix = "p"

    monkeypatch.setattr(objectstore, "config_from_env", lambda: Cfg())
    monkeypatch.setattr(
        objectstore, "list_prefix",
        lambda **_: [{"key": "p/uploads/live/video.mp4", "size": 10, "lastModified": 1}],
    )
    monkeypatch.setattr(objectstore, "client", lambda cfg: None)
    plan = bucketreset.plan(Store(), SimpleNamespace(hosted=SimpleNamespace(retention_hours=96)))
    assert plan["plannedBytes"] == 0
    assert plan["kept"][0]["reason"] == "live batch reference"
