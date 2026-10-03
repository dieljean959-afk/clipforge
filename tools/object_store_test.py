"""Durable media offload, against a fake S3.

The premise of this feature is that the 4 GB virtiofs holding `outputs/` and
`data/` is part of the sandbox and does not come back when the sandbox is
rebuilt — the codebase returns, everything else is empty. So media is pushed
to an S3-compatible bucket and pulled back on demand.

What is asserted here, and why each one matters:

  1. Unconfigured means UNCHANGED. With no bucket, every call is a no-op and no
     file is ever deleted. This is the property that makes the feature safe to
     ship to someone whose bucket has not been created yet.
  2. An upload is pushed before its local copy is released, and the order is
     enforced — a lost upload is the one unrecoverable loss in this app.
  3. A source that is gone from disk is restored through `find_source_file`,
     so a job interrupted by a rebuild resumes instead of dying at Stage 3.
  4. A finished clip is restored before it is served, which is what keeps
     Range requests, ETags and the frontend working untouched.
  5. A failed push keeps the local file. Durability is best-effort; a storage
     outage must never cost a rendered clip.
  6. Deleting a job purges its objects, so "delete" still means delete.

    python tools/object_store_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import objectstore  # noqa: E402
from app.config import Paths  # noqa: E402
from app.store import LocalStore  # noqa: E402

PASS = 0
FAIL = 0

ENV_KEYS = (
    "CLIPFORGE_S3_ENDPOINT",
    "CLIPFORGE_S3_BUCKET",
    "CLIPFORGE_S3_ACCESS_KEY",
    "CLIPFORGE_S3_SECRET_KEY",
    "CLIPFORGE_S3_REGION",
    "CLIPFORGE_S3_PREFIX",
    "CLIPFORGE_OFFLOAD_DELETE_LOCAL",
)


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


def section(title: str) -> None:
    print(f"\n[{title}]")


class FakeS3:
    """Just enough S3 for the offload path, including one switch to fail."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.fail_uploads = False
        self.fail_downloads = False

    def upload_file(self, path: str, bucket: str, key: str, ExtraArgs=None):  # noqa: N803
        self.calls.append(f"put {key}")
        if self.fail_uploads:
            raise RuntimeError("bucket is on fire")
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, bucket: str, key: str, path: str):
        self.calls.append(f"get {key}")
        if self.fail_downloads:
            raise RuntimeError("network unreachable")
        Path(path).write_bytes(self.objects[key])

    def copy_object(self, **kw):
        src = kw["CopySource"]["Key"]
        self.objects[kw["Key"]] = self.objects[src]

    def delete_object(self, **kw):
        self.calls.append(f"del {kw['Key']}")
        self.objects.pop(kw["Key"], None)

    def delete_objects(self, **kw):
        for obj in kw["Delete"]["Objects"]:
            self.calls.append(f"del {obj['Key']}")
            self.objects.pop(obj["Key"], None)

    def head_object(self, **kw):
        if kw["Key"] not in self.objects:
            raise KeyError(kw["Key"])
        return {"ContentLength": len(self.objects[kw["Key"]])}

    def get_paginator(self, name: str):
        def paginate(Bucket: str, Prefix: str):  # noqa: N803
            yield {
                "Contents": [
                    {"Key": k} for k in self.objects if k.startswith(Prefix)
                ]
            }

        return type("P", (), {"paginate": staticmethod(paginate)})()


class env:
    """Set the bucket env vars for the duration of a block."""

    def __init__(self, **kw: str) -> None:
        self.values = kw
        self.saved: dict[str, str | None] = {}

    def __enter__(self):
        for k in ENV_KEYS:
            self.saved[k] = os.environ.get(k)
            os.environ.pop(k, None)
        for k, v in self.values.items():
            os.environ[k] = v
        objectstore._client_cache = None
        return self

    def __exit__(self, *exc):
        for k in ENV_KEYS:
            if self.saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = self.saved[k]
        objectstore._client_override = None
        objectstore._client_cache = None
        return False


BUCKET_ENV = {
    "CLIPFORGE_S3_ENDPOINT": "https://example.r2.cloudflarestorage.com",
    "CLIPFORGE_S3_BUCKET": "clipforge-media",
    "CLIPFORGE_S3_ACCESS_KEY": "AKIA-test",
    "CLIPFORGE_S3_SECRET_KEY": "secret",
}


def _paths() -> Paths:
    base = Path(tempfile.mkdtemp(prefix="clipforge-objstore-"))
    p = Paths(
        data_dir=base / "data",
        outputs_dir=base / "outputs",
        downloads_dir=base / "downloads",
        static_dir=base / "static",
    )
    for d in (p.data_dir, p.outputs_dir, p.downloads_dir):
        d.mkdir(parents=True, exist_ok=True)
    (p.uploads_dir or (p.data_dir / "uploads")).mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- 1
def test_unconfigured_is_a_no_op() -> None:
    section("1. with no bucket configured, nothing changes and nothing is deleted")
    with env():
        paths = _paths()
        store = LocalStore(paths)
        job = store.create_job(source_type="upload", source="source.mp4")
        jid = job["_id"]
        src_dir = store.job_source_dir_for_write(jid)
        src = src_dir / "source.mp4"
        src.write_bytes(b"real bytes" * 100)

        cfg = objectstore.config_from_env()
        check("the store reports itself disabled", cfg.enabled is False)
        check("an upload is 'pushed' to nowhere, successfully-declined",
              objectstore.upload_file(src, "any/key", cfg) is False)
        check("the local file is untouched", src.exists())
        check("nothing is deleted from local disk",
              objectstore.drop_local_after_push(True, src, cfg) is False)
        check("and it stays that way even if a caller insists the push worked",
              objectstore.drop_local_after_push(True, src, objectstore.config_from_env()) is False
              and src.exists())
        check("and the source is still found", store.find_source_file(jid) == src)
        check("status() reports OFF", objectstore.status()["enabled"] is False)


# ---------------------------------------------------------------- 2 + 3
def test_upload_offload_and_restore() -> None:
    section("2 + 3. an upload is offloaded, and restored when the sandbox rebuilds")
    paths = _paths()
    fake = FakeS3()
    with env(**BUCKET_ENV):
        objectstore._CLIENT_OVERRIDE = fake
        store = LocalStore(paths)
        job = store.create_job(source_type="upload", source="source.mp4")
        jid = job["_id"]
        store.write_job({**job, "source": "source.mp4"})
        src_dir = store.job_source_dir_for_write(jid)
        src = src_dir / "source.mp4"
        src.write_bytes(b"a real uploaded video" * 500)

        cfg = objectstore.config_from_env()
        check("the store reports itself enabled", cfg.enabled is True)
        key = objectstore.upload_key(jid, src.name)
        ok = objectstore.upload_file(src, key, cfg)
        check("the upload is pushed", ok and key in fake.objects)
        check("the object has the same bytes",
              fake.objects[key] == src.read_bytes())
        check("no partial object is left behind",
              not any(k.endswith(".uploading") for k in fake.objects))

        dropped = objectstore.drop_local_after_push(ok, src, cfg)
        check("the local copy is released to reclaim disk", dropped is True)
        check("and it really is gone", not src.exists())

        # This is the rebuild: the job document survives, the bytes do not.
        found = store.find_source_file(jid)
        check("find_source_file pulls the source back from the bucket",
              found is not None and found.exists(), str(found))
        check("restored into the canonical uploads location",
              found is not None and found.parent == src_dir, str(found))
        check("with byte-identical contents",
              found is not None and found.read_bytes() == b"a real uploaded video" * 500)
        gets_before = sum(1 for c in fake.calls if c.startswith("get "))
        store.find_source_file(jid)
        gets_after = sum(1 for c in fake.calls if c.startswith("get "))
        check("a second lookup does not re-download",
              gets_after == gets_before, f"{gets_before} -> {gets_after}")

        # ---- a failed push must never cost the local file ---------------
        src2_dir = store.job_source_dir_for_write(jid + "b")
        src2 = src2_dir / "source.mp4"
        src2.write_bytes(b"do not lose me")
        fake.fail_uploads = True
        ok2 = objectstore.upload_file(src2, "k2", cfg)
        check("a failed push reports failure", ok2 is False)
        check("and the local file survives it", src2.exists())
        fake.fail_uploads = False

        # ---- keep-local mode ---------------------------------------------
        src3 = store.job_source_dir_for_write(jid + "c") / "source.mp4"
        src3.write_bytes(b"keep me")
        os.environ["CLIPFORGE_OFFLOAD_DELETE_LOCAL"] = "0"
        cfg3 = objectstore.config_from_env()
        ok3 = objectstore.upload_file(src3, "k3", cfg3)
        objectstore.drop_local_after_push(ok3, src3, cfg3)
        check("keep-local mode keeps the file", src3.exists())
        check("while still pushing", "k3" in fake.objects)


# ---------------------------------------------------------------- 4
def test_clip_offload_and_restore() -> None:
    section("4. a finished clip is offloaded and restored before it is served")
    paths = _paths()
    fake = FakeS3()
    with env(**BUCKET_ENV):
        objectstore._CLIENT_OVERRIDE = fake
        from app.task import Pipeline  # noqa: PLC0415 - needs the override in place

        pipeline = Pipeline.__new__(Pipeline)
        pipeline.settings = type("S", (), {})()
        pipeline._log = lambda *a, **k: None  # type: ignore[method-assign]

        job_dir = paths.outputs_dir / "job42"
        job_dir.mkdir(parents=True, exist_ok=True)
        clip = job_dir / "clip_01.mp4"
        clip.write_bytes(b"rendered mp4 bytes")
        thumb = job_dir / "clip_01.jpg"
        thumb.write_bytes(b"jpeg bytes")

        pushed = pipeline._offload_clip("job42", 1, clip, thumb)
        check("the clip is offloaded", pushed and "clipforge/outputs/job42/clip_01.mp4" in fake.objects)
        check("the poster is offloaded too", "clipforge/outputs/job42/clip_01.jpg" in fake.objects)
        check("the local clip is released", not clip.exists())
        check("the local poster is released", not thumb.exists())

        # What the serving routes do.
        key = objectstore.key_for_path(clip, paths.outputs_dir)
        check("the key is derivable from the path the clip doc recorded",
              key == "clipforge/outputs/job42/clip_01.mp4", str(key))
        restored = objectstore.materialize(key, clip)
        check("materialize brings it back", restored and clip.exists())
        check("with identical bytes", clip.read_bytes() == b"rendered mp4 bytes")
        check("a path outside outputs/ is not treated as ours",
              objectstore.key_for_path(Path("/tmp/elsewhere/x.mp4"), paths.outputs_dir) is None)

        # A pipeline restart must still skip the restored clip, not re-encode.
        check("the restored clip satisfies the resume check",
              clip.exists() and clip.stat().st_size > 0)


# ---------------------------------------------------------------- 5 + 6
def test_delete_purges_remote() -> None:
    section("5 + 6. deleting a job removes its objects too")
    fake = FakeS3()
    with env(**BUCKET_ENV):
        objectstore._CLIENT_OVERRIDE = fake
        import app.main as main_mod  # noqa: PLC0415

        fake.objects["clipforge/uploads/job9/source.mp4"] = b"x"
        fake.objects["clipforge/outputs/job9/clip_01.mp4"] = b"y"
        fake.objects["clipforge/outputs/job9/clip_01.jpg"] = b"z"
        fake.objects["clipforge/outputs/OTHERJOB/clip_01.mp4"] = b"keep me"

        main_mod._purge_remote_job("job9")
        check("the job's source is gone from the bucket",
              "clipforge/uploads/job9/source.mp4" not in fake.objects)
        check("its clips are gone",
              "clipforge/outputs/job9/clip_01.mp4" not in fake.objects)
        check("its posters are gone",
              "clipforge/outputs/job9/clip_01.jpg" not in fake.objects)
        check("another job's objects are untouched",
              fake.objects.get("clipforge/outputs/OTHERJOB/clip_01.mp4") == b"keep me")

        # A failing bucket must not turn "delete" into a 500.
        class Exploding(FakeS3):
            def get_paginator(self, name: str):
                raise RuntimeError("bucket unreachable")

        objectstore._CLIENT_OVERRIDE = Exploding()
        try:
            main_mod._purge_remote_job("job9")
            check("a storage outage does not break job deletion", True)
        except Exception as exc:  # noqa: BLE001
            check("a storage outage does not break job deletion", False, str(exc))
        finally:
            objectstore._CLIENT_OVERRIDE = fake


def main() -> int:
    print("ClipForge durable-media (object store) suite")
    test_unconfigured_is_a_no_op()
    test_upload_offload_and_restore()
    test_clip_offload_and_restore()
    test_delete_purges_remote()
    total = PASS + FAIL
    print(f"\nobjectstore: {PASS}/{total} passed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
