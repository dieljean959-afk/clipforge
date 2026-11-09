"""Deleted jobs must be recoverable, and the retention sweep must be disableable.

Regression cover for a real data-loss incident: `RetentionCleaner` (24h window
by default) called `store.delete_job()` on every batch older than the window,
and `delete_job()` hard-unlinked the job doc, the clip docs and every rendered
MP4. An entire library vanished overnight with no backup and no undo.

Two guarantees are pinned here:
  1. retention_hours = 0 means "never expire" (the sweeper no-ops)
  2. a delete moves files to data/trash/ and restore_job() puts them back,
     including rendered media at its ORIGINAL absolute path
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import HostedConfig, Paths  # noqa: E402
from app.hosting import RetentionCleaner  # noqa: E402
from app.store import LocalStore  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


def make_store(tmp: Path) -> LocalStore:
    paths = Paths(
        downloads_dir=tmp / "downloads",
        outputs_dir=tmp / "outputs",
        data_dir=tmp / "data",
        static_dir=tmp / "static",
    )
    return LocalStore(paths)


def main() -> int:
    # ---- 1. retention can be turned off ------------------------------
    print("== 1. retention switch")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = make_store(tmp)
        job = store.create_job(source_type="url", source="https://example.com/old")
        job["stage"] = "complete"
        # 10 days ago, long finished
        old_ms = 10 * 24 * 3600 * 1000
        job["createdAt"] -= old_ms
        job["completedAt"] = job["createdAt"] + 1000
        store.write_job(job)
        check("an expired job still exists before any sweep", store.get_job(job["_id"]) is not None)

        cleaner = RetentionCleaner.__new__(RetentionCleaner)

        def _settings(hours: int):
            return type(
                "S",
                (),
                {
                    "hosted": HostedConfig(retention_hours=hours),
                    "app": type("A", (), {"paths": store._paths})(),
                },
            )()

        cleaner.settings = _settings(0)
        cleaner.store = store
        cleaner.interval_min = 30
        removed = cleaner.sweep()
        check("retention_hours=0 removes nothing", removed == 0, f"removed={removed}")
        check(
            "the expired job survives a disabled sweep",
            store.get_job(job["_id"]) is not None,
        )

        cleaner.settings = _settings(1)
        removed = cleaner.sweep()
        check("retention_hours=1 does expire it", removed == 1, f"removed={removed}")

    # ---- 2. delete is reversible -------------------------------------
    print("== 2. delete then restore")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = make_store(tmp)
        job = store.create_job(source_type="upload", source="clip.mp4", title="My batch")
        clips = store.create_clips(job["_id"], [{"start": 0.0, "end": 4.0, "title": "c1"}], "src")

        # A rendered clip on disk, at the path the app would serve it from.
        media = tmp / "outputs" / job["_id"] / "clip_0001.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"fake-mp4-bytes")
        clips[0]["videoPath"] = str(media)
        clips[0]["renderState"] = "done"
        store.write_clip(clips[0])

        job_id = job["_id"]
        check("delete_job reports success", store.delete_job(job_id) is True)
        check("the job is gone from the app", store.get_job(job_id) is None)
        check("the job is tombstoned", store.is_deleted(job_id) is True)
        check("the rendered file is no longer at its path", not media.exists())

        trash_dirs = list((tmp / "data" / "trash").glob(f"{job_id}-*"))
        check("a trash dir was created", len(trash_dirs) == 1, f"found={trash_dirs}")
        if trash_dirs:
            trashed_media = trash_dirs[0] / "media" / media.name
            check("the mp4 bytes were kept, not unlinked", trashed_media.exists())
            check("the job doc was kept in trash", (trash_dirs[0] / f"{job_id}.json").exists())

        check("restore_job reports success", store.restore_job(job_id) is True)
        restored = store.get_job(job_id)
        check("the job doc is back", restored is not None)
        check("the tombstone is cleared", store.is_deleted(job_id) is False)
        check("its title survived", bool(restored) and restored.get("title") == "My batch")
        check("the clip doc is back", len(store.list_clips(job_id)) == 1)
        check(
            "the rendered mp4 is back at its ORIGINAL path",
            media.exists() and media.read_bytes() == b"fake-mp4-bytes",
        )
        check("the job is listed again", any(j["_id"] == job_id for j in store.list_jobs()))

        # Writing must work again after a restore, not be blocked by a tombstone.
        if restored:
            restored["stage"] = "complete"
            store.write_job(restored)
            check(
                "writes are accepted again after restore",
                (store.get_job(job_id) or {}).get("stage") == "complete",
            )

    # ---- 3. tombstone without trash is not a crash -------------------
    print("== 3. edge cases")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = make_store(tmp)
        job = store.create_job(source_type="url", source="https://example.com/x")
        store.delete_job(job["_id"])
        # simulate trash having been cleared out by hand
        for d in (tmp / "data" / "trash").glob("*"):
            for f in d.rglob("*"):
                if f.is_file():
                    f.unlink()
            d.rmdir()
        check("restore with an empty trash reports False", store.restore_job(job["_id"]) is False)
        check("restoring a never-deleted job reports False", store.restore_job("nope") is False)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
