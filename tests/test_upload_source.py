"""Uploads: the source has to still be there when the pipeline reads it.

Two bugs, both ending in "the uploaded file for this batch is no longer on the
server" on a job whose video the user had just attached:

* ``job_source_dir_for_write`` returned ``uploads/<job>`` without creating it,
  so streaming the request body died with ``FileNotFoundError`` and left a job
  behind with no file at all (the "pending-upload" card);
* with durable storage configured, the upload is pushed to the bucket and its
  local bytes are dropped — and ``find_source_file`` only ever looked at local
  directories, so the fetch stage could not find the file it had just stored.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app import objectstore
from app.config import load_config
from app.store import LocalStore


@pytest.fixture()
def store(tmp_path):
    base = load_config(quiet=True)
    paths = replace(
        base.app.paths,
        data_dir=tmp_path,
        downloads_dir=tmp_path / "downloads",
        uploads_dir=tmp_path / "uploads",
    )
    return LocalStore(paths)


def test_source_dir_for_write_exists(store):
    dest = store.job_source_dir_for_write("job_upload")

    assert dest.is_dir()
    # the route streams straight into this path, so it must be usable as-is
    (dest / "upload.bin").write_bytes(b"x")


def test_offloaded_source_is_pulled_back(store, monkeypatch):
    job = {"_id": "job_offloaded", "sourceType": "upload", "source": "source.mp4"}
    store.write_job(job)

    monkeypatch.setattr(
        objectstore,
        "config_from_env",
        lambda: objectstore.ObjectStoreConfig(
            endpoint="https://example.invalid",
            bucket="bucket",
            access_key="a",
            secret_key="s",
            region="auto",
            prefix="clipforge",
            delete_local=True,
        ),
    )

    def fake_materialize(key, dest, cfg=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"pulled from the bucket")
        return True

    monkeypatch.setattr(objectstore, "materialize", fake_materialize)

    found = store.find_source_file("job_offloaded")

    assert found is not None
    assert found.read_bytes() == b"pulled from the bucket"


def test_missing_source_without_durable_storage_is_still_none(store, monkeypatch):
    store.write_job({"_id": "job_local", "sourceType": "upload", "source": "source.mp4"})
    monkeypatch.setattr(
        objectstore,
        "config_from_env",
        lambda: objectstore.ObjectStoreConfig(
            endpoint="", bucket="", access_key="", secret_key="",
            region="auto", prefix="clipforge", delete_local=True,
        ),
    )

    assert store.find_source_file("job_local") is None
