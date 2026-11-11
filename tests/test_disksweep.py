"""The cleanup routine must free scratch and refuse everything else.

The list it must refuse is the one that costs the user something real: the
credentials, the job documents (the database), the DB backup, the code, the
runtime, the fonts/emoji assets, and any file a running batch holds.
"""

from __future__ import annotations

import time
from pathlib import Path

from app import disksweep


class _Paths:
    def __init__(self, root: Path) -> None:
        self.data_dir = root / "data"
        self.root = root
        self.outputs_dir = root / "outputs"


class _Store:
    def __init__(self, jobs: list[dict] | None = None) -> None:
        self._jobs = jobs or []

    def list_jobs(self) -> list[dict]:
        return self._jobs


def _app_root(tmp_path: Path) -> _Paths:
    paths = _Paths(tmp_path)
    (paths.data_dir / "jobs").mkdir(parents=True)
    (paths.data_dir / "backups").mkdir(parents=True)
    (paths.data_dir / "logs").mkdir(parents=True)
    (paths.data_dir / "work").mkdir(parents=True)
    (paths.root / "app" / "assets" / "emoji").mkdir(parents=True)
    (paths.root / "app" / "fonts").mkdir(parents=True)
    (paths.root / "app" / "__pycache__").mkdir(parents=True)
    (paths.root / ".venv" / "lib").mkdir(parents=True)
    (paths.root / "node_modules" / "pkg").mkdir(parents=True)
    return paths


def test_protected_paths_are_never_deleted(tmp_path):
    paths = _app_root(tmp_path)
    protected = [
        paths.data_dir / "cookies.txt",
        paths.data_dir / ".env",
        paths.data_dir / "jobs" / "job1.json",
        paths.data_dir / "backups" / "db.tgz",
        paths.root / ".venv" / "lib" / "x.py",
        paths.root / "node_modules" / "pkg" / "index.js",
        paths.root / "app" / "assets" / "emoji" / "1f355.png",
        paths.root / "app" / "fonts" / "Anton-Regular.ttf",
    ]
    for p in protected:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("keep me", encoding="utf-8")

    freed: list[float] = []
    for p in protected:
        disksweep._rm(p, dry_run=False, freed=freed)

    assert all(p.exists() for p in protected), "a protected file was deleted"
    assert freed == []


def test_scratch_is_reported_and_removed(tmp_path):
    paths = _app_root(tmp_path)
    cache = paths.root / "app" / "__pycache__"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "x.pyc").write_bytes(b"0" * (2 * 1024 * 1024))

    out = disksweep.sweep(_Store(), paths, include_models=False)

    assert out["ok"] is True
    bytecode = next(c for c in out["categories"] if c["name"] == "python bytecode")
    assert bytecode["mb"] >= 2.0
    assert not cache.exists()


def test_a_running_job_keeps_its_scratch(tmp_path):
    paths = _app_root(tmp_path)
    work = paths.data_dir / "work" / "running1"
    work.mkdir(parents=True)
    (work / "audio.wav").write_bytes(b"0" * 2048)
    stale = paths.data_dir / "work" / "gone1"
    stale.mkdir(parents=True)
    (stale / "words.json").write_text("{}", encoding="utf-8")

    out = disksweep.sweep(
        _Store([{"_id": "running1", "status": "running"}]), paths, include_models=False
    )

    assert work.exists(), "a live batch lost its scratch folder"
    assert not stale.exists(), "orphaned scratch was kept"
    assert out["skipped"] == ["running1"]


def test_dry_run_deletes_nothing(tmp_path):
    paths = _app_root(tmp_path)
    cache = paths.root / "app" / "__pycache__"
    (cache / "x.pyc").write_bytes(b"0" * 4096)

    out = disksweep.sweep(_Store(), paths, dry_run=True, include_models=False)

    assert cache.exists()
    assert out["dryRun"] is True
    assert out["freedMb"] >= 0


def test_nightly_runs_once_per_period(tmp_path):
    paths = _app_root(tmp_path)
    assert disksweep.nightly(_Store(), paths)["ok"] is True
    second = disksweep.nightly(_Store(), paths)
    assert second.get("skipped") == "not due"
