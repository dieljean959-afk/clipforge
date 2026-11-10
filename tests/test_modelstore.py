"""Models are bucket-resident: the disk copy is transient, never the only copy."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import modelstore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A fake bucket plus a fake HF cache, both empty."""
    bucket: dict[str, bytes] = {}

    monkeypatch.setattr(modelstore, "_root", lambda: tmp_path / "models")
    monkeypatch.setattr(modelstore, "_hf_cache_root", lambda: tmp_path / "hub")
    monkeypatch.setattr(
        modelstore.objectstore, "configured", lambda cfg=None: True, raising=False
    )

    def list_prefix(prefix=None, cfg=None):
        return [
            {"key": k, "size": len(v)} for k, v in bucket.items() if k.startswith(prefix or "")
        ]

    def upload_file(path, key, cfg=None, content_type=None):
        bucket[key] = Path(path).read_bytes()
        return True

    def download_to(key, dest, cfg=None):
        data = bucket.get(key)
        if data is None:
            return False
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True

    monkeypatch.setattr(modelstore.objectstore, "list_prefix", list_prefix)
    monkeypatch.setattr(modelstore.objectstore, "upload_file", upload_file)
    monkeypatch.setattr(modelstore.objectstore, "download_to", download_to)
    monkeypatch.setattr(modelstore.objectstore, "config_from_env",
                        lambda: type("C", (), {"prefix": "clipforge"})())
    return bucket


def _snapshot(tmp_path: Path, name: str = "small") -> Path:
    snap = tmp_path / "hub" / f"models--Systran--faster-whisper-{name}" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"w" * 4096)
    (snap / "config.json").write_bytes(b"{}")
    return snap


def test_nothing_local_and_nothing_in_the_bucket_falls_back_to_the_name(store, tmp_path):
    assert modelstore.resolve("small") == "small"


def test_adopt_puts_the_model_in_the_bucket(store, tmp_path):
    _snapshot(tmp_path)
    out = modelstore.adopt("small", wait_s=0.1)
    assert out["ok"] is True
    assert sorted(k.rsplit("/", 1)[-1] for k in store) == ["config.json", "model.bin"]


def test_restore_brings_it_back_for_the_stage(store, tmp_path):
    _snapshot(tmp_path)
    modelstore.adopt("small", wait_s=0.1)
    modelstore.release("small")
    assert modelstore.ready("small") is False
    path = modelstore.resolve("small")
    assert path == str(modelstore.local_dir("small"))
    assert modelstore.ready("small") is True


def test_release_keeps_the_copy_the_bucket_lacks(store, tmp_path):
    _snapshot(tmp_path)
    out = modelstore.release("small")
    assert out["ok"] is False and out["freed"] == 0
    assert modelstore._snapshot_dir("small") is not None


def test_release_frees_disk_and_keeps_the_bucket_copy(store, tmp_path):
    _snapshot(tmp_path)
    modelstore.adopt("small", wait_s=0.1)
    out = modelstore.release("small")
    assert out["ok"] is True
    assert out["freed"] > 0
    assert modelstore._snapshot_dir("small") is None
    assert modelstore._bucket_files("small")


def test_release_all_clears_the_transient_root(store, tmp_path):
    _snapshot(tmp_path)
    modelstore.adopt("small", wait_s=0.1)
    modelstore.resolve("small")
    modelstore.release_all()
    assert not modelstore._root().exists()
