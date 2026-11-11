"""Tests for `app/reclaim.py` — moving finished batches into the bucket.

The point of every test here is the same: a local file may only disappear once
the bucket really has it, and only for a job that has settled. A reclaim that
drops a file before the push, or that touches a running batch, is worse than no
reclaim at all — it destroys the only copy of a render.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import objectstore, reclaim


class FakeS3:
    """Minimal S3 stand-in: uploads land in `objects`, and every failure mode
    the real client has (a rejected push) can be forced with `fail_uploads`."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_uploads = False
        self.calls: list[str] = []

    def upload_file(self, path: str, bucket: str, key: str, ExtraArgs=None):  # noqa: N803
        if self.fail_uploads:
            raise RuntimeError("bucket rejected the upload")
        self.calls.append(f"upload {key}")
        self.objects[key] = Path(path).read_bytes()

    def put_object(self, **kw):
        """The real upload path: ONE PUT, file streamed from disk."""
        if self.fail_uploads:
            raise RuntimeError("bucket rejected the upload")
        body = kw.get("Body")
        data = body.read() if hasattr(body, "read") else bytes(body or b"")
        self.calls.append(f"put {kw['Key']}")
        self.objects[kw["Key"]] = data
        return {}

    def copy_object(self, **kw):
        self.calls.append(f"copy {kw['Key']}")
        self.objects[kw["Key"]] = self.objects.pop(kw["CopySource"]["Key"])

    def delete_object(self, **kw):
        self.objects.pop(kw["Key"], None)

    def head_object(self, **kw):
        if kw["Key"] not in self.objects:
            raise KeyError(kw["Key"])
        return {"ContentLength": len(self.objects[kw["Key"]])}

    def get_object(self, **kw):
        body = self.objects[kw["Key"]]
        header = kw.get("Range")
        if header:
            span = header.split("=")[1]
            first, _, last = span.partition("-")
            start = int(first)
            end = int(last) if last else len(body) - 1
            chunk = body[start : end + 1]
            return {
                "Body": chunk,
                "ContentLength": len(chunk),
                "ContentRange": f"bytes {start}-{end}/{len(body)}",
                "ContentType": "video/mp4",
            }
        return {"Body": body, "ContentLength": len(body), "ContentType": "video/mp4"}


@pytest.fixture()
def bucket(monkeypatch, tmp_path):
    """A configured store whose client is the fake, and a sandbox tree."""
    fake = FakeS3()
    monkeypatch.setattr(objectstore, "_CLIENT_OVERRIDE", fake)
    monkeypatch.setenv("CLIPFORGE_S3_ENDPOINT", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("CLIPFORGE_S3_BUCKET", "clipforge-clips")
    monkeypatch.setenv("CLIPFORGE_S3_ACCESS_KEY", "AKIA-test")
    monkeypatch.setenv("CLIPFORGE_S3_SECRET_KEY", "secret")
    monkeypatch.setenv("CLIPFORGE_S3_PREFIX", "clipforge")
    monkeypatch.setenv("CLIPFORGE_OFFLOAD_DELETE_LOCAL", "1")

    paths = SimpleNamespace(
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs",
        uploads_dir=tmp_path / "data" / "uploads",
        downloads_dir=tmp_path / "downloads",
    )
    for d in (paths.data_dir, paths.outputs_dir, paths.uploads_dir, paths.downloads_dir):
        d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(fake=fake, paths=paths)


def _job(job_id: str, stage: str = "complete", **extra) -> dict:
    return {"_id": job_id, "stage": stage, **extra}


def _store(*jobs: dict) -> SimpleNamespace:
    return SimpleNamespace(list_jobs=lambda: list(jobs))


def _write(p: Path, data: bytes = b"rendered mp4 bytes") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_finished_batch_is_pushed_and_the_local_copy_is_freed(bucket):
    job_id = "job42"
    clip = _write(bucket.paths.outputs_dir / job_id / "clip_000.mp4")
    source = _write(bucket.paths.uploads_dir / job_id / "source.mp4")

    out = reclaim.reclaim(_store(_job(job_id)), bucket.paths)

    assert out["ok"] is True
    assert out["uploaded"] == 2
    assert out["jobs"] == 1
    assert out["freedBytes"] > 0
    assert not clip.exists() and not source.exists()
    assert "clipforge/outputs/job42/clip_000.mp4" in bucket.fake.objects
    assert "clipforge/uploads/job42/source.mp4" in bucket.fake.objects


def test_a_failed_push_never_deletes_the_local_file(bucket):
    job_id = "job42"
    clip = _write(bucket.paths.outputs_dir / job_id / "clip_000.mp4")
    bucket.fake.fail_uploads = True

    out = reclaim.reclaim(_store(_job(job_id)), bucket.paths)

    assert out["uploaded"] == 0
    assert out["freedBytes"] == 0
    assert clip.exists(), "the only copy of a render must survive a storage outage"


def test_a_running_batch_is_never_touched(bucket):
    running = _write(bucket.paths.outputs_dir / "live" / "clip_000.mp4")
    deleting = _write(bucket.paths.outputs_dir / "gone" / "clip_000.mp4")
    paused = _write(bucket.paths.outputs_dir / "paused" / "clip_000.mp4")

    out = reclaim.reclaim(
        _store(
            _job("live", stage="render"),
            _job("gone", stage="deleting"),
            _job("paused", stage="paused"),
        ),
        bucket.paths,
    )

    assert out["uploaded"] == 0
    assert running.exists() and deleting.exists() and paused.exists()
    assert bucket.fake.objects == {}


def test_keep_source_leaves_the_original_upload_alone(bucket):
    job_id = "job42"
    source = _write(bucket.paths.uploads_dir / job_id / "source.mp4")
    clip = _write(bucket.paths.outputs_dir / job_id / "clip_000.mp4")

    out = reclaim.reclaim(_store(_job(job_id, keepSource=True)), bucket.paths)

    assert out["uploaded"] == 1
    assert source.exists(), "keepSource means the source stays on disk"
    assert not clip.exists()


def test_offload_off_reports_why_and_does_nothing(bucket, monkeypatch):
    clip = _write(bucket.paths.outputs_dir / "job42" / "clip_000.mp4")
    monkeypatch.delenv("CLIPFORGE_S3_BUCKET")

    out = reclaim.reclaim(_store(_job("job42")), bucket.paths)

    assert out["ok"] is False
    assert out["reason"]
    assert clip.exists()
    assert bucket.fake.objects == {}


def test_keep_local_mode_reports_why_and_does_nothing(bucket, monkeypatch):
    clip = _write(bucket.paths.outputs_dir / "job42" / "clip_000.mp4")
    monkeypatch.setenv("CLIPFORGE_OFFLOAD_DELETE_LOCAL", "0")

    out = reclaim.reclaim(_store(_job("job42")), bucket.paths)

    assert out["ok"] is False
    assert "DELETE_LOCAL" in out["reason"]
    assert clip.exists()


def test_partial_transfers_and_non_media_are_skipped(bucket):
    job_id = "job42"
    tmp = _write(bucket.paths.outputs_dir / job_id / "clip_000.mp4.uploading")
    partial = _write(bucket.paths.outputs_dir / job_id / "clip_001.mp4.part")
    lock = _write(bucket.paths.outputs_dir / job_id / "job.lock")

    out = reclaim.reclaim(_store(_job(job_id)), bucket.paths)

    assert out["uploaded"] == 0
    assert tmp.exists() and partial.exists() and lock.exists()


def test_reclaiming_nothing_says_so(bucket):
    out = reclaim.reclaim(_store(), bucket.paths)

    assert out["ok"] is True
    assert out["uploaded"] == 0
    assert out["reason"]


def test_streaming_reads_the_bucket_without_writing_to_disk(bucket, tmp_path):
    """What the serving routes do for a reclaimed clip: bytes go bucket -> the
    response, so playback does not refill the disk the reclaim just freed."""
    key = "clipforge/outputs/job42/clip_000.mp4"
    bucket.fake.objects[key] = b"0123456789" * 100

    full = objectstore.open_stream(key)
    assert full is not None
    status, headers, body = full
    assert status == 200
    assert headers["Content-Length"] == str(1000)
    assert body == b"0123456789" * 100

    ranged = objectstore.open_stream(key, "bytes=100-199")
    assert ranged is not None
    status, headers, body = ranged
    assert status == 206
    assert headers["Content-Range"] == "bytes 100-199/1000"
    assert body == b"0123456789" * 10

    assert list(tmp_path.rglob("*.fetching")) == []


def test_streaming_misses_return_none_so_the_404_path_is_kept(bucket):
    assert objectstore.open_stream("clipforge/outputs/nope/clip_000.mp4") is None


def test_watchdog_does_nothing_while_there_is_room(bucket, monkeypatch):
    clip = _write(bucket.paths.outputs_dir / "job42" / "clip_000.mp4")
    monkeypatch.setattr(reclaim, "_free_bytes", lambda paths: 900 * 1024 * 1024)

    assert reclaim.tick(_store(_job("job42")), bucket.paths) is None
    assert clip.exists()
    assert bucket.fake.objects == {}


def test_watchdog_reclaims_once_the_disk_is_tight(bucket, monkeypatch):
    clip = _write(bucket.paths.outputs_dir / "job42" / "clip_000.mp4")
    # Two tight readings first — the watchdog's own check, then reclaim's
    # initial measurement — then comfortable ones, which are what stop the walk.
    tight = 300 * 1024 * 1024
    readings = iter([tight, tight] + [2000 * 1024 * 1024] * 10)
    monkeypatch.setattr(reclaim, "_free_bytes", lambda paths: next(readings))

    out = reclaim.tick(_store(_job("job42")), bucket.paths)

    assert out is not None
    assert out["uploaded"] == 1
    assert "clipforge/outputs/job42/clip_000.mp4" in bucket.fake.objects
    assert not clip.exists()


def test_finishing_offloads_immediately_when_the_disk_is_tight(bucket, monkeypatch):
    """A batch that just finished must not sit on the disk waiting for the
    watchdog: it is the reason the next run fails for lack of space."""
    clip = _write(bucket.paths.outputs_dir / "job42" / "clip_000.mp4")
    source = _write(bucket.paths.uploads_dir / "job42" / "source.mp4")
    monkeypatch.setattr(reclaim, "_free_bytes", lambda paths: 200 * 1024 * 1024)

    reclaim.offload_job(_store(_job("job42")), bucket.paths, "job42")

    # Inline, so the files are already gone when the call returns.
    assert not clip.exists() and not source.exists()
    assert "clipforge/outputs/job42/clip_000.mp4" in bucket.fake.objects


def test_finishing_offloads_a_batch_with_room_to_spare_too(bucket, monkeypatch):
    clip = _write(bucket.paths.outputs_dir / "job42" / "clip_000.mp4")
    monkeypatch.setattr(reclaim, "_free_bytes", lambda paths: 3000 * 1024 * 1024)

    reclaim.offload_job(_store(_job("job42")), bucket.paths, "job42")

    for t in __import__("threading").enumerate():
        if t.name == "offload-finished":
            t.join(timeout=5.0)
    assert not clip.exists()
    assert "clipforge/outputs/job42/clip_000.mp4" in bucket.fake.objects


def test_destination_for_a_clip_matches_what_the_route_serves(bucket):
    """reclaim's key must be the one `clip_video` asks for, or a reclaimed clip
    is pushed to a name nothing ever reads and the file vanishes for real."""
    clip = _write(bucket.paths.outputs_dir / "job42" / "clip_003.mp4")
    pushed_key = reclaim.key_for(bucket.paths, "job42", clip)
    route_key = objectstore.key_for_path(clip, bucket.paths.outputs_dir)
    assert pushed_key == route_key == "clipforge/outputs/job42/clip_003.mp4"


def test_pushing_a_clip_verifies_it_and_then_releases_the_disk(bucket):
    """The core rule: stored means HEAD-verified, and only then local goes."""
    clip = _write(bucket.paths.outputs_dir / "j1" / "clip_000.mp4")
    thumb = _write(bucket.paths.outputs_dir / "j1" / "clip_000.thumb.jpg")

    res = reclaim.push_clip_and_thumb("j1", clip, thumb)

    assert res["ok"] is True
    assert res["dropped"] is True
    assert "clipforge/outputs/j1/clip_000.mp4" in bucket.fake.objects
    assert not clip.exists()
    assert not thumb.exists()


def test_a_failed_upload_keeps_the_clip_on_disk(bucket, monkeypatch):
    """A finished clip is never lost to a flaky link — it stays and is retried."""
    from app import objectstore

    clip = _write(bucket.paths.outputs_dir / "j1" / "clip_000.mp4")
    monkeypatch.setattr(objectstore, "upload_file", lambda *a, **k: False)

    res = reclaim.push_clip_and_thumb("j1", clip, None)

    assert res["ok"] is False
    assert res["dropped"] is False
    assert clip.exists() and clip.stat().st_size > 0


def test_completed_batches_expire_and_running_ones_never_do(bucket, monkeypatch):
    import time as _time

    now_ms = int(_time.time() * 1000)
    jobs = [
        {"_id": "old", "status": "complete", "completedAt": now_ms - 60 * 3600 * 1000},
        {"_id": "running", "status": "running", "completedAt": now_ms - 60 * 3600 * 1000},
    ]
    calls: list[str] = []

    class _S:
        def list_jobs(self, limit=1000):
            return jobs

        def delete_job(self, job_id):
            calls.append(job_id)

    monkeypatch.setattr(reclaim, "delete_job_objects", lambda job_id: calls.append(f"obj:{job_id}") or 0)

    out = reclaim.sweep_expired(_S(), bucket.paths, 48)

    assert out["expired"] == ["old"]
    assert "running" not in out["expired"]


def test_housekeeping_runs_retention_and_budget_when_due(bucket, monkeypatch):
    """The 80/90% guard and retention are only real if something calls them.

    They used to exist as functions with no caller: the bucket could pass 90%
    and nothing ever trimmed it, and an expired batch stayed until a request
    happened to land. The watchdog now runs this pass.
    """
    seen: list[str] = []
    monkeypatch.setattr(reclaim, "_last_housekeeping", 0.0)
    monkeypatch.setattr(objectstore, "configured", lambda *a, **k: True)
    monkeypatch.setattr(objectstore, "reachable", lambda *a, **k: True)
    monkeypatch.setattr(
        reclaim, "sweep_expired",
        lambda store, paths, hours: seen.append(f"sweep:{hours}") or
        {"expired": ["old"], "objectsRemoved": 2, "jobsRemoved": 1},
    )
    monkeypatch.setattr(
        reclaim, "enforce_budget",
        lambda store, paths, **k: seen.append(f"budget:{k.get('retention_hours')}") or
        {"usedPercent": 41.0, "warned": False, "pruned": [], "swept": None},
    )

    out = reclaim.housekeeping(_store(), bucket.paths, retention_hours=48, force=True)

    assert seen == ["sweep:48", "budget:48"]
    assert out["jobsRemoved"] == 1 and out["usedPercent"] == 41.0


def test_housekeeping_throttles_itself_between_passes(bucket, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(reclaim, "_last_housekeeping", 0.0)
    monkeypatch.setattr(objectstore, "configured", lambda *a, **k: True)
    monkeypatch.setattr(objectstore, "reachable", lambda *a, **k: True)
    monkeypatch.setattr(reclaim, "sweep_expired", lambda *a, **k: calls.append("sweep") or
                        {"expired": [], "objectsRemoved": 0, "jobsRemoved": 0})
    monkeypatch.setattr(reclaim, "enforce_budget", lambda *a, **k: calls.append("budget") or
                        {"usedPercent": 1.0, "warned": False, "pruned": [], "swept": None})

    reclaim.housekeeping(_store(), bucket.paths, retention_hours=48, force=True)
    out = reclaim.housekeeping(_store(), bucket.paths, retention_hours=48)

    assert out["skipped"] == "not due"
    assert calls == ["sweep", "budget"]


def test_an_unreachable_bucket_blocks_retention_deletes(bucket, monkeypatch):
    """A bucket that cannot answer a listing must not turn "expired" into "gone"."""
    calls: list[str] = []
    monkeypatch.setattr(reclaim, "_last_housekeeping", 0.0)
    monkeypatch.setattr(objectstore, "configured", lambda *a, **k: True)
    monkeypatch.setattr(objectstore, "reachable", lambda *a, **k: False)
    monkeypatch.setattr(reclaim, "sweep_expired", lambda *a, **k: calls.append("sweep") or {})
    monkeypatch.setattr(reclaim, "enforce_budget", lambda *a, **k: calls.append("budget") or {})

    out = reclaim.housekeeping(_store(), bucket.paths, retention_hours=12, force=True)

    assert out["skipped"] == "bucket unreachable"
    assert calls == []


def test_reachable_reports_false_instead_of_raising_when_the_bucket_is_down(bucket):
    """The liveness probe must fail closed — it is what authorises deletes.

    This branch used to log through a name that does not exist in
    `objectstore`, so an unreachable bucket raised NameError inside the very
    guard meant to protect retention deletes.
    """
    assert objectstore.reachable() is False

    bucket.fake.list_objects_v2 = lambda **kw: {"Contents": []}
    assert objectstore.reachable() is True


# --- the boot sweep must never delete a RUNNING job's scratch --------------
#
# `sweep_startup` used to call `store.list_jobs(limit=1000)`. LocalStore's
# `list_jobs()` takes no arguments, so the call raised TypeError, the handler
# swallowed it, and the job map became {} — at which point EVERY work folder
# looks orphaned. Every API restart therefore deleted the scratch of jobs that
# were still running: a 20-minute transcribe died with "cannot probe duration —
# source file not found: .../transcribe/audio.m4a", because the boot sweep took
# its audio while the decoder was reading it.


def _local_style_store(jobs: list[dict]):
    """A store whose list_jobs takes NO arguments, exactly like LocalStore."""
    return SimpleNamespace(list_jobs=lambda: list(jobs))


def test_boot_sweep_keeps_the_scratch_of_a_running_job(bucket):
    live_id, gone_id = "job-live", "job-gone"
    live = _write(bucket.paths.data_dir / "work" / live_id / "transcribe" / "audio.m4a",
                  b"audio being decoded")
    orphan = _write(bucket.paths.data_dir / "work" / gone_id / "transcribe" / "audio.m4a",
                    b"nobody owns this")

    out = reclaim.sweep_startup(
        _local_style_store([{"_id": live_id, "status": "running", "stage": "transcribing"}]),
        bucket.paths,
    )

    assert live.exists(), "the boot sweep deleted a running job's audio"
    assert not orphan.exists(), "an orphaned folder was left behind"
    assert out["dirsRemoved"] == 1


def test_boot_sweep_keeps_a_queued_job_too(bucket):
    queued = _write(bucket.paths.data_dir / "work" / "job-q" / "transcribe" / "audio.m4a", b"x")
    reclaim.sweep_startup(_local_style_store([{"_id": "job-q", "status": "queued"}]), bucket.paths)
    assert queued.exists(), "a queued job lost its cached work"


def test_boot_sweep_deletes_nothing_when_the_job_list_is_unreadable(bucket):
    """An unreadable store means \"do not delete\", never \"delete everything\"."""
    def boom():
        raise RuntimeError("store is not readable yet")

    keep = _write(bucket.paths.data_dir / "work" / "job-x" / "transcribe" / "audio.m4a", b"x")
    out = reclaim.sweep_startup(SimpleNamespace(list_jobs=boom), bucket.paths)

    assert keep.exists(), "an unreadable job list turned into a full scratch wipe"
    assert out["dirsRemoved"] == 0 and out["filesRemoved"] == 0


def test_all_jobs_handles_both_store_signatures():
    jobs = [{"_id": "a"}]
    assert reclaim._all_jobs(_local_style_store(jobs)) == jobs
    limited = SimpleNamespace(list_jobs=lambda limit=10: jobs[:limit])
    assert reclaim._all_jobs(limited) == jobs


def test_boot_sweep_with_the_real_local_store_keeps_a_running_job(bucket):
    """The production pair: LocalStore's no-argument list_jobs + boot sweep.

    This is the combination that killed the 20:37 job: the real store raised
    TypeError on `limit=`, the sweep treated the empty job list as "every
    folder is an orphan", and the audio file of a transcribing job went with
    it. Uses the real LocalStore rather than a stand-in so the signature itself
    is part of what is being tested.
    """
    from app.store import LocalStore

    job_id = "live-job-from-a-crash"
    jobs_dir = bucket.paths.data_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / f"{job_id}.json").write_text(
        json.dumps({"_id": job_id, "status": "running", "stage": "transcribing"}),
        encoding="utf-8",
    )
    audio = _write(bucket.paths.data_dir / "work" / job_id / "transcribe" / "audio.m4a",
                   b"the 19 MB audio track being decoded")

    out = reclaim.sweep_startup(LocalStore(bucket.paths), bucket.paths)

    assert audio.exists(), "the boot sweep deleted the audio of a running transcribe"
    assert out["dirsRemoved"] == 0
