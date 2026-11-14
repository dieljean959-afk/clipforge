#!/usr/bin/env python3
"""Re-queue one or more batches for re-transcription.

Same effect as POST /api/jobs/<id>/retranscribe, but driven through the app's
own store and stage-cache APIs and finished with an HTTP call to /retry, which
is the only way to hand work to the SERVER's worker queue (enqueue() from this
process would only ever touch this process's own worker).

Why this exists at all: the endpoint is the supported path and is what the
dashboard button calls. This script is for the window where the running server
predates the endpoint — the container is long-lived, the file is not.

What it invalidates, and what it keeps:
  cleared  transcribe, detect, reframe, captions, export
  kept     fetch — the downloaded source is reused, nothing is re-downloaded
  reset    every clip's renderState, so the re-encoded pixels are not skipped

    .venv/bin/python tools/queue_retranscribe.py f3b81a1dd7e6 fd961148baa5
    .venv/bin/python tools/queue_retranscribe.py --dry-run <job ids>
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config  # noqa: E402
from app.stages import StageWorkspace  # noqa: E402
from app.store import LocalStore  # noqa: E402

BASE = "http://localhost:5173"
RUNNING = {"queued", "grabbing", "transcribing", "finding", "rendering"}


def retry(job_id: str) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{BASE}/api/jobs/{job_id}/retry", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()[:120]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:160]
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(quiet=True)
    store = LocalStore(cfg.app.paths)
    failures = 0

    for job_id in args.jobs:
        job = store.get_job(job_id)
        if job is None:
            print(f"{job_id}  SKIP: no such job")
            failures += 1
            continue
        if job.get("stage") in RUNNING:
            print(f"{job_id}  SKIP: still running at {job.get('stage')}")
            failures += 1
            continue
        if store.find_source_file(job_id) is None:
            print(f"{job_id}  SKIP: source file is gone, cannot re-transcribe")
            failures += 1
            continue
        clips = store.list_clips(job_id)
        if not clips:
            print(f"{job_id}  SKIP: no clips")
            failures += 1
            continue

        done = [c.get("renderState") for c in clips]
        if args.dry_run:
            print(f"{job_id}  would re-transcribe  clips={len(clips)} "
                  f"(renderState: {done.count('done')} done, "
                  f"{done.count('pending')} pending)  source={store.find_source_file(job_id).name}")
            continue

        ws = StageWorkspace(cfg.app.paths.data_dir, job_id)
        had = [s for s in ("transcribe", "detect", "reframe", "captions", "export")
               if ws.is_done(s)]
        ws.clear_from("transcribe")
        for clip in clips:
            clip["renderState"] = "pending"
            store.write_clip(clip)
        store.append_job_log(
            job_id,
            f"re-transcribe queued (tools/queue_retranscribe.py) — cleared "
            f"{', '.join(had) or 'no cached stages'}; the fetched source is reused",
        )
        status, body = retry(job_id)
        print(f"{job_id}  cleared={len(had)} stages  clips_reset={len(clips)}  "
              f"retry HTTP {status} {body[:80]}")
        if status not in (200, 202):
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
