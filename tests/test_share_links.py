"""Public share links must actually serve, not 500.

The bug these tests pin: `JobShareLocker` cached its per-master lock in the
same dict that holds the finished share copies, so the FIRST `/s/<id>.mp4`
request for a clip returned a `threading.Lock` where a `Path` belonged and the
endpoint died with:

    AttributeError: '_thread.lock' object has no attribute 'exists'

Every share link in the dashboard showed "Ready" (the copy is built during the
render) and every one of them answered 500 when opened.
"""

from __future__ import annotations

from pathlib import Path

from app import share_video


def _fake_build(monkeypatch, calls: list[Path]):
    def build(master: Path, *, force: bool = False) -> Path:
        calls.append(master)
        out = share_video.share_path_for(master)
        out.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 24)
        return out

    monkeypatch.setattr(share_video, "build_share_version", build)


def test_get_returns_the_share_copy_not_the_lock(tmp_path, monkeypatch):
    master = tmp_path / "clip_000.mp4"
    master.write_bytes(b"\x00" * 64)
    calls: list[Path] = []
    _fake_build(monkeypatch, calls)

    got = share_video.JobShareLocker().get(master)

    assert isinstance(got, Path), f"expected a Path, got {type(got).__name__}"
    assert got == share_video.share_path_for(master)
    assert got.exists() and got.stat().st_size > 0
    assert calls == [master]


def test_a_second_request_reuses_the_copy_without_re_encoding(tmp_path, monkeypatch):
    master = tmp_path / "clip_001.mp4"
    master.write_bytes(b"\x00" * 64)
    calls: list[Path] = []
    _fake_build(monkeypatch, calls)
    locker = share_video.JobShareLocker()

    first = locker.get(master)
    second = locker.get(master)

    assert first == second == share_video.share_path_for(master)
    assert len(calls) == 1, "the derived copy must be built once per master"


def test_each_master_gets_its_own_copy(tmp_path, monkeypatch):
    calls: list[Path] = []
    _fake_build(monkeypatch, calls)
    locker = share_video.JobShareLocker()
    a, b = tmp_path / "clip_000.mp4", tmp_path / "clip_001.mp4"
    for p in (a, b):
        p.write_bytes(b"\x00" * 64)

    assert locker.get(a) == share_video.share_path_for(a)
    assert locker.get(b) == share_video.share_path_for(b)
    assert share_video.share_path_for(a) != share_video.share_path_for(b)
