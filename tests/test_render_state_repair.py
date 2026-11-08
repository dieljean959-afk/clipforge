"""Clip documents must agree with the MP4s that exist on disk.

The bug these tests pin: a worker killed mid-render leaves the finished clip
files behind but never runs the final `write_clip`, so the document still says
`renderState: pending` (or `running`) and carries no `fps`. The dashboard then
shows “queued · no fps”, refuses to play the clip, and blocks download and
share — a clip that works, reported as broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import video


class _FakeStore:
    def __init__(self, clips: list[dict], stage: str = "complete"):
        self._clips = clips
        self.stage = stage
        self.written: dict[str, dict] = {}

    def list_all_clips(self, limit: int = 512):
        return list(self._clips)[:limit]

    def get_job(self, job_id: str):
        return {"_id": job_id, "stage": self.stage}

    def write_clip(self, clip_id, fields=None, *, job_id=None):
        self.written[str(clip_id)] = dict(fields or {})
        return self.written[str(clip_id)]


@pytest.fixture()
def mp4(tmp_path: Path) -> Path:
    p = tmp_path / "clip_004.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096)
    return p


@pytest.fixture(autouse=True)
def _no_ffprobe(monkeypatch):
    monkeypatch.setattr(
        video, "probe",
        lambda path: {"duration": 23.517, "width": 1080, "height": 1920, "fps": 30.0},
    )


def test_a_pending_clip_with_a_finished_mp4_is_promoted(mp4: Path):
    store = _FakeStore([
        {"_id": "c4", "jobId": "j1", "renderState": "pending", "videoPath": str(mp4)},
    ])

    stats = video.repair_render_states(store)

    assert stats["repaired"] == 1
    patch = store.written["c4"]
    assert patch["renderState"] == "done"
    assert patch["fps"] == 30.0
    assert patch["mediaDurationSec"] == 23.517
    assert patch["mediaBytes"] == mp4.stat().st_size
    assert patch["videoPath"] == str(mp4)


def test_a_running_clip_is_promoted_too_but_a_live_job_is_left_alone(mp4: Path):
    stale = _FakeStore([
        {"_id": "c2", "jobId": "j1", "renderState": "running", "videoPath": str(mp4)},
    ])
    assert video.repair_render_states(stale)["repaired"] == 1

    live = _FakeStore(
        [{"_id": "c2", "jobId": "j2", "renderState": "running", "videoPath": str(mp4)}],
        stage="rendering",
    )
    stats = video.repair_render_states(live)
    assert stats["repaired"] == 0
    assert live.written == {}, "a live render owns its document"


def test_a_done_clip_missing_only_fps_gets_it(mp4: Path):
    store = _FakeStore([
        {"_id": "c0", "jobId": "j1", "renderState": "done", "videoPath": str(mp4), "fps": None},
    ])

    assert video.repair_render_states(store)["repaired"] == 1
    assert store.written["c0"]["fps"] == 30.0
    assert store.written["c0"]["renderState"] == "done"


def test_a_clip_whose_file_is_gone_is_never_marked_failed(tmp_path: Path):
    """The bytes may live in the durable bucket and be restored on demand."""
    store = _FakeStore([
        {"_id": "c9", "jobId": "j1", "renderState": "pending",
         "videoPath": str(tmp_path / "missing.mp4")},
    ])

    stats = video.repair_render_states(store)

    assert stats["repaired"] == 0
    assert stats["skipped"] == 1
    assert store.written == {}


def test_a_clip_that_already_agrees_is_not_rewritten(mp4: Path):
    store = _FakeStore([
        {"_id": "c0", "jobId": "j1", "renderState": "done", "fps": 30.0,
         "videoPath": str(mp4)},
    ])

    assert video.repair_render_states(store)["repaired"] == 0
    assert store.written == {}
