"""Deleting a job must actually stop it.

`DELETE /api/jobs/{id}` used to remove the record and the workspace while
leaving the work untouched. Two real consequences:

1. A deleted job that was still running kept downloading, transcribing and
   spending LLM calls for a record nobody could see any more.
2. Worse, it came BACK. The child process cannot be interrupted mid-render,
   so it finished its work and called `write_job`/`write_clip` — recreating
   exactly the files the delete had just removed. The user deleted a job and
   it reappeared moments later marked "complete", pointing at MP4s that had
   been unlinked underneath it, with zombie clips attached to no job.

The fix has two halves that must both hold:
- `worker.cancel()` drops the id from the queue or SIGKILLs its process group.
- `store.delete_job()` leaves a tombstone, and `write_job()` refuses to write
  a tombstoned id, so a late write cannot resurrect it.

Run:  .venv/bin/python tools/delete_cancels_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Paths  # noqa: E402
from app.store import LocalStore  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(ok: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILS.append(label)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = Paths(
            data_dir=root / "data",
            outputs_dir=root / "outputs",
            downloads_dir=root / "downloads",
            static_dir=root / "static",
        )
        store = LocalStore(paths)

        # ---- 1. a job can be created, found, deleted -------------------
        store.write_job({"_id": "j1", "source": "s", "title": "t", "stage": "rendering"})
        check(store.get_job("j1") is not None, "a job is readable before deletion")
        check(store.delete_job("j1") is True, "delete_job reports success")
        check(store.get_job("j1") is None, "the job record is gone")
        check(store.delete_job("j1") is False, "deleting twice is a no-op, not a crash")

        # ---- 2. THE BUG: a late write resurrects the job ---------------
        store.write_job({
            "_id": "j1", "source": "s", "title": "t",
            "stage": "complete", "clipCount": 10,
        })
        check(store.get_job("j1") is None, "a late write_job does NOT resurrect a deleted job")
        check(store.is_deleted("j1") is True, "the tombstone records the deletion")

        # ---- 3. the tombstone covers every field a runner writes ------
        for stage, extra in (
            ("grabbing", {}),
            ("transcribing", {"progress": 40}),
            ("finding", {"error": "boom"}),
            ("rendering", {"progress": 99}),
            ("complete", {"clipCount": 10, "totalClipsSec": 300.0}),
        ):
            store.write_job({"_id": "j1", "source": "s", "title": "t", "stage": stage, **extra})
            check(store.get_job("j1") is None, f"late '{stage}' write is refused")

        # ---- 4. it unlinks the clip docs, so no zombie clips -----------
        store.write_job({"_id": "j2", "source": "s", "title": "t", "stage": "rendering"})
        clips = store.create_clips("j2", [{"start": 1.0, "end": 9.0, "score": 90, "title": "c"}], "groq")
        clip_id = clips[0]["_id"]
        check(len(store.list_clips("j2")) == 1, "the clip exists before deletion")
        store.delete_job("j2")
        check(store.list_clips("j2") == [], "clip docs are removed with the job")
        check(
            not (paths.data_dir / "clips" / f"{clip_id}.json").exists(),
            "the clip document file is unlinked from disk",
        )

        # ---- 5. deleting does not poison unrelated ids ----------------
        store.write_job({"_id": "j3", "source": "s", "title": "t", "stage": "queued"})
        check(store.is_deleted("j3") is False, "a fresh job is not tombstoned")
        check(store.get_job("j3") is not None, "an unrelated job is unaffected by another deletion")
        store.write_job({"_id": "j3", "source": "s", "title": "t", "stage": "complete"})
        check(store.get_job("j3")["stage"] == "complete", "an unrelated job still accepts writes")

        # ---- 6. the endpoint stops the work before deleting ----------
        src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        body = src.split("def delete_job(", 1)[-1].split("\n@app.", 1)[0]
        cancel_at = body.find("worker.cancel(job_id)")
        delete_at = body.find("store.delete_job(job_id)")
        check(cancel_at != -1, "the delete endpoint cancels the worker")
        check(
            0 <= cancel_at < delete_at,
            "cancel happens BEFORE the record is removed (so no work is orphaned)",
        )
        check('"cancelled"' in body, "the response reports what was cancelled")

        # ---- 7. the queue drops ids that no longer exist --------------
        worker_src = (Path(__file__).resolve().parents[1] / "app" / "worker.py").read_text(encoding="utf-8")
        run_body = worker_src.split("def _run(", 1)[-1].split("\n    def ", 1)[0]
        check(
            "self.store.get_job(job_id) is None" in run_body,
            "the worker skips a dequeued job that no longer exists",
        )
        check("def cancel(" in worker_src, "the worker exposes cancel()")

        # ---- 8. create_clips survives a vanished job ------------------
        # The render loop calls this with the job id after the user may have
        # deleted the job. It used to raise KeyError('_id') on the empty dict
        # it substituted for the missing job, turning a delete into a crash.
        store.create_clips("j1", [{"start": 5.0, "end": 9.0, "score": 90, "title": "zombie"}], "groq")
        check(True, "create_clips on a deleted job does not raise")
        check(
            store.get_job("j1") is None,
            "create_clips on a deleted job leaves the job deleted",
        )

        # ---- 9. the delete endpoint reclaims DISK --------------------
        # A manual delete removed the JSON record and the stage workspace but
        # left downloads/<id> (the source video, 300+ MB) and outputs/<id> (the
        # rendered MP4s) on disk. On a 4 GB host that filled the disk while the
        # dashboard showed nothing to delete, and the disk preflight then
        # rejected every NEW job with a 507. The retention sweeper already
        # removed both dirs; the manual path has to do the same.
        main_src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(
            encoding="utf-8"
        )
        delete_fn = main_src.split("def delete_job(")[1].split("\n@app.")[0]
        check(
            "downloads_dir / job_id" in delete_fn and "outputs_dir / job_id" in delete_fn,
            "deleting a job also removes its downloads/ and outputs/ folders",
        )
        check(
            "shutil.rmtree" in delete_fn,
            "the delete path actually frees the bytes (shutil.rmtree)",
        )

    print()
    if FAILS:
        print(f"{CHECKS - len(FAILS)} passed, {len(FAILS)} FAILED")
        return 1
    print(f"{CHECKS} passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
