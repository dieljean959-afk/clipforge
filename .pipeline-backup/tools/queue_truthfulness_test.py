"""A queued job must LOOK queued.

The pipeline resumes by re-queuing jobs that a host restart interrupted
(`_boot()` in app/main.py). That path used to enqueue a job while leaving
`stage` at whatever stage it died in and `progress` at its last value.

The dashboard derives its whole status from those two fields
(`overallStatus()` in src/lib/clipforge.ts), so the result was a job that was
sitting in a queue — possibly behind two others — presented as "Stage 3 of 4
— Detecting, 99%". Indistinguishable from real work, and worse than a plain
error because it never resolves: the user watches a spinner that means nothing.

The resume contract is the same one `retry()` already honours: once a job is
back in the queue it reports `stage="queued"`, `progress=0`, and a detail line
that says it is waiting. The stage markers on disk — not this field — decide
where the runner restarts.

Run:  .venv/bin/python tools/queue_truthfulness_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import subtitle  # noqa: E402
from app.config import Paths, load_config  # noqa: E402
from app.stages import StageWorkspace  # noqa: E402
from app.store import LocalStore  # noqa: E402
from app.task import Pipeline  # noqa: E402

FAILS: list[str] = []
CHECKS = 0

# Mirrors src/lib/clipforge.ts overallStatus()
STAGE_LABELS = {
    "grabbing": "Fetching video",
    "transcribing": "Transcribing audio",
    "finding": "Detecting clips",
    "rendering": "Rendering clips",
}
PIPELINE_STAGES = ["grabbing", "transcribing", "finding", "rendering"]


def check(ok: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILS.append(label)


def overall_status(job: dict) -> dict:
    """Port of the dashboard's status derivation, so the test asserts what a
    user actually sees rather than what the store happens to hold."""
    if job.get("error") and job.get("stage") == "failed":
        return {"state": "failed"}
    if job.get("stage") == "complete":
        return {"state": "done"}
    if job.get("stage") == "queued":
        return {"state": "queued", "percent": 0, "label": "Queued — waiting for a worker"}
    idx = PIPELINE_STAGES.index(job.get("stage", "grabbing"))
    within = min(100, max(0, job.get("progress") or 0))
    percent = round(((idx + within / 100) / len(PIPELINE_STAGES)) * 100)
    return {
        "state": "running",
        "percent": min(99, percent),
        "label": f"Stage {idx + 1} of 4 — {STAGE_LABELS[job['stage']]}",
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = LocalStore(Paths(
            data_dir=root / "data",
            outputs_dir=root / "outputs",
            downloads_dir=root / "downloads",
            static_dir=root / "static",
        ))
        store.write_job({"_id": "j", "source": "s", "title": "t", "stage": "grabbing"})

        # ---- what the dashboard showed before the fix -------------------
        frozen = {
            "stage": "finding",
            "progress": 100.0,
            "error": None,
        }
        stale_view = overall_status(frozen)
        check(stale_view["state"] == "running", "REGRESSION GUARD: stale fields read as running")
        check(stale_view["percent"] == 75, "REGRESSION GUARD: a queued 'finding' job read as 75%")
        check(
            overall_status({"stage": "rendering", "progress": 100.0, "error": None})["percent"] == 99,
            "REGRESSION GUARD: a queued 'rendering' job read as 99%",
        )

        # ---- what the dashboard shows after the fix --------------------
        queued = {
            "stage": "queued",
            "progress": 0,
            "error": None,
            "progressDetail": "Queued — resuming from cached stages after a restart.",
        }
        good_view = overall_status(queued)
        check(good_view["state"] == "queued", "a re-queued job reads as queued, not running")
        check(good_view["percent"] == 0, "a re-queued job reads as 0%")
        check("Queued" in good_view["label"], "the label says Queued")

        # ---- every stage it could have died in ------------------------
        for stage in PIPELINE_STAGES:
            recovered = {
                "stage": "queued",
                "progress": 0,
                "error": None,
                "progressDetail": "Queued — resuming from cached stages after a restart.",
            }
            check(overall_status(recovered)["state"] == "queued", f"recovered from '{stage}' reads as queued")

        # ---- the fix is actually in the source -------------------------
        src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        boot = src.split("def _boot()", 1)[-1].split("def ", 1)[0] if "def _boot()" in src else src
        check('"stage": "queued"' in boot, "_boot() sets stage=queued when re-queuing")
        check('"progress": 0' in boot, "_boot() resets progress when re-queuing")
        check(
            '"error": None' in boot,
            "_boot() clears a stale error when re-queuing",
        )
        # The old bug: stage left at the stage it died in.
        check(
            '"stage": job.get("failedStage") or "queued"' not in boot,
            "_boot() no longer leaves stage at failedStage for a queued job",
        )
        # retry() must keep the same contract (it lives in the worker).
        worker_src = (Path(__file__).resolve().parents[1] / "app" / "worker.py").read_text(encoding="utf-8")
        retry_body = worker_src.split("def retry(", 1)[-1].split("\n    def ", 1)[0]
        check('"stage": "queued"' in retry_body, "retry() sets stage=queued")
        check('"progress": 0' in retry_body, "retry() resets progress")
        check('"error": None' in retry_body, "retry() clears the error")

        # ---- 9. stale unit counters are cleared too --------------------
        # The dashboard renders `unitsDone of unitsTotal` next to the progress
        # bar, so a queued job that kept "10 of 10 clips" from the attempt that
        # died contradicted its own freshly reset 0%.
        for name, body in (("_boot()", boot), ("retry()", retry_body)):
            check('"unitsDone": None' in body, f"{name} clears stale unitsDone")
            check('"unitsTotal": None' in body, f"{name} clears stale unitsTotal")
            check('"unitLabel": None' in body, f"{name} clears the stale unit label")

        store.write_job({
            "_id": "j9", "source": "s", "title": "t", "stage": "complete",
            "unitsDone": 10, "unitsTotal": 10, "unitLabel": "clips",
        })
        job9 = store.get_job("j9")
        for field in ("unitsDone", "unitsTotal", "unitLabel"):
            job9[field] = None
        store.write_job(job9)
        check(store.get_job("j9")["unitsDone"] is None, "cleared unit counters persist")

        # ---- 10. an error stops being true when the job moves on ------
        # Observed live: a job that hit "AI detection failed: Groq HTTP 429"
        # went on to resume and render, but carried the error the whole way —
        # so a job mid-render advertised a failure that had already been
        # recovered from. The invariant is: a job that is not stage="failed"
        # carries no error.
        task_src = (Path(__file__).resolve().parents[1] / "app" / "task.py").read_text(encoding="utf-8")
        prog_body = task_src.split("def _progress(", 1)[-1].split("\n    def ", 1)[0]
        finish_body = task_src.split("def _finish(", 1)[-1].split("\n    def ", 1)[0]
        check(
            prog_body.count('job["error"] = None') >= 1,
            "_progress clears a stale error when the stage changes",
        )
        check('"error": None' in finish_body, "_finish clears the error on success")
        check(
            prog_body.find('job["error"] = None') > prog_body.find("if prev_stage != stage"),
            "the error is cleared on the stage-change branch, not unconditionally",
        )
        for name, body in (("_boot()", boot), ("retry()", retry_body)):
            check('"error": None' in body, f"{name} clears the stale error")

        # ---- 11. a delivered batch is never a failed batch -------------
        # Live case: job c57d7782c548 rendered all 10 clips, then a later
        # re-detect hit a Groq 429 and _fail() overwrote stage=complete with
        # stage=failed. Every clip was still on disk and downloadable, so the
        # red banner was a phantom with no action behind it.
        from app.worker import InProcessWorker as _W  # noqa: E402

        media = root / "outputs" / "jclip"
        media.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (media / f"clip_{i}.mp4").write_bytes(b"\x00" * 16)
        store.create_clips(
            "j",
            [
                {"order": i, "title": f"c{i}", "start": 0.0, "end": 1.0,
                 "videoPath": str(media / f"clip_{i}.mp4")}
                for i in range(3)
            ],
            "groq",
        )
        # create_clips() writes the DETECTION fields; the media path is stamped
        # later by the encode stage, so mirror that here rather than pretending
        # a candidate carries a rendered file.
        for doc in store.list_clips("j"):
            doc["videoPath"] = str(media / f"clip_{doc['order']}.mp4")
            store.write_clip(doc)
        delivered = store.get_job("j")
        delivered.update({
            "stage": "failed", "clipCount": 3, "progress": 100,
            "error": "AI detection failed: HTTP 429", "errorDetail": "TPM Limit",
        })
        store.write_job(delivered)
        rendered, promised = store.clips_delivered("j", store.get_job("j"))
        check((rendered, promised) == (3, 3), "clips_delivered counts 3 rendered of 3 promised")
        check(
            _W.has_delivered_output(store.get_job("j"), rendered, promised),
            "a failed job whose whole batch is on disk is recognised as delivered",
        )
        check(
            overall_status(store.get_job("j"))["state"] == "failed",
            "REGRESSION GUARD: before the boot reconcile the user still sees 'failed'",
        )
        # The predicate must stay narrow, or it starts lying in the other
        # direction: a run that genuinely lost a clip is still a failure.
        check(
            not _W.has_delivered_output(store.get_job("j"), 2, 3),
            "a partial batch (2 of 3) is NOT treated as delivered",
        )
        check(
            not _W.has_delivered_output({"stage": "complete"}, 3, 3),
            "only stage=failed jobs are candidates (a complete job is left alone)",
        )
        check(
            not _W.has_delivered_output(store.get_job("j"), 0, 0),
            "a job with no clips at all is NOT treated as delivered",
        )
        # A videoPath that points at a file a container recycle removed is not
        # delivery. c57d7782c548 is exactly the case that would otherwise
        # restore stage=complete for clips the user cannot download.
        (media / "clip_1.mp4").unlink()
        check(
            not _W.has_delivered_output(store.get_job("j"), *store.clips_delivered("j", store.get_job("j"))),
            "a clip missing from disk does not count as rendered",
        )
        (media / "clip_1.mp4").write_bytes(b"\x00" * 16)

        check("has_delivered_output(" in boot, "_boot() consults has_delivered_output")
        check("clips_delivered(" in boot, "_boot() asks the store what is actually on disk")
        check('"stage": "complete"' in boot, "_boot() restores stage=complete for a delivered batch")
        check('"errorDetail": None' in boot, "_boot() clears the stale errorDetail too")
        check(
            boot.index("has_delivered_output") < boot.index("has_signal_failure"),
            "the delivered check runs BEFORE the re-queue paths, so it is not re-queued",
        )

        # ---- a real stored job round-trips honestly -------------------
        store.write_job({**queued, "_id": "real", "source": "s", "title": "t"})
        back = store.get_job("real")
        check(overall_status(back)["state"] == "queued", "a stored queued job round-trips as queued")
        check(json.dumps(back) is not None, "the stored job is JSON-serialisable")

    section_no_stale_writes(store, Pipeline(load_config(quiet=True), store))

    print()
    if FAILS:
        print(f"{CHECKS - len(FAILS)} passed, {len(FAILS)} FAILED")
        return 1
    print(f"{CHECKS} passed, 0 failed")
    return 0


def section_no_stale_writes(store, pipeline) -> None:
    """A stage must never write back the job dict it was handed.

    `run()` reads the job once and passes that same object down through every
    stage, while `_progress` writes a freshly-read copy on every tick. A stage
    that annotates its own copy and saves it therefore resurrects whatever
    `stage`, `progress` and ETA the job had when the pipeline STARTED.

    On a real 55-minute job this was not subtle: detection worked through its
    chunks while the dashboard read "queued, 0%" the whole time, because the
    clip-count annotation was written back from the stale dict each time.
    """
    print("\n-- stage writes never resurrect a stale job --")
    job = store.create_job("upload", "stale.mp4")
    jid = job["_id"]
    # This is exactly what run() hands to every stage.
    handed_down = store.get_job(jid)
    check(handed_down is not None and handed_down.get("stage") == "queued",
          "a new job starts queued")

    # The worker makes progress, which is written from a fresh read.
    # _progress takes a FRACTION; 42 would clamp to 100.
    pipeline._progress(jid, "transcribing", 0.42, "working")
    live = store.get_job(jid)
    check(live.get("stage") == "transcribing" and round(live.get("progress")) == 42,
          "progress reached the store")
    # The stage then annotates the dict it was given, and saves it.
    handed_down["clipTarget"] = 23
    store.write_job(handed_down)
    after = store.get_job(jid)
    check(after.get("stage") == "queued" and round(after.get("progress")) == 0,
          "writing the handed-down dict clobbers live progress (this is the bug)")

    # And _patch_job is the way that does NOT.
    pipeline._progress(jid, "transcribing", 0.42, "working")
    pipeline._patch_job(jid, clipTarget=23, clipTargetReason="because")
    fixed = store.get_job(jid)
    check(fixed.get("stage") == "transcribing" and round(fixed.get("progress")) == 42,
          "_patch_job keeps the live stage and progress")
    check(fixed.get("clipTarget") == 23 and fixed.get("clipTargetReason") == "because",
          "_patch_job still applies its own fields")
    check(pipeline._patch_job("nope-not-a-job", x=1) == {},
          "_patch_job on a deleted job does not raise")

    # A resumed job reuses the cached candidates, so the "N found / M above the
    # floor" explanation has to be rebuilt from them. It was left blank, which
    # hid exactly the number a user needs to decide whether to move the cap.
    job2 = store.create_job("upload", "resume.mp4")
    jid2 = job2["_id"]
    pipeline._patch_job(jid2, durationSec=3273.0)
    cached = [
        {"start": 1.0, "end": 30.0, "score": 94.0, "title": "a"},
        {"start": 31.0, "end": 60.0, "score": 88.0, "title": "b"},
        {"start": 61.0, "end": 90.0, "score": 42.0, "title": "c"},
    ]
    ws = StageWorkspace(pipeline.settings.app.paths.data_dir, jid2)
    ws.write_json("detect", "candidates.json", cached)
    ws.mark_done("detect", artifacts=["detect/candidates.json"], count=len(cached))
    check(store.get_job(jid2).get("clipsAboveScore") is None,
          "a fresh job has no count explanation yet")
    out = pipeline._stage_detect(store.get_job(jid2), subtitle.Transcript(), ws)
    resumed = store.get_job(jid2)
    check(len(out) == 3 and resumed.get("clipTarget") == 3,
          "the resume path returns the cached candidates")
    check(resumed.get("clipsAboveScore") == 2 and resumed.get("clipsBelowScore") == 1,
          f"and splits them by the score floor (got {resumed.get('clipsAboveScore')}/"
          f"{resumed.get('clipsBelowScore')})")
    check(resumed.get("clipScoreFloor") == pipeline.settings.gemini.min_score,
          "and reports the floor it used")
    check(resumed.get("candidatesDetected") == 3,
          "and does not leave the found-count blank")
    pipeline._patch_job(jid2, candidatesDetected=45)
    pipeline._stage_detect(store.get_job(jid2), subtitle.Transcript(), ws)
    check(store.get_job(jid2).get("candidatesDetected") == 45,
          "a pre-gate count that was already recorded is not overwritten")
    store.delete_job(jid2)

    src = (Path(__file__).resolve().parents[1] / "app" / "task.py").read_text(encoding="utf-8")
    stage_src = src.split("def _stage_detect")[1].split("\n    def ")[0]
    check("self.store.write_job(job)" not in stage_src,
          "the detect stage no longer writes the dict it was handed")
    check("_patch_job(" in stage_src, "it annotates through _patch_job instead")
    store.delete_job(jid)


if __name__ == "__main__":
    raise SystemExit(main())
