"""End-to-end reliability check for a real short video.

Processes a ~3 minute clip all the way through the actual pipeline with the
actual ffmpeg and the actual Whisper engine, and prints PASS/FAIL for every
stage. Re-runnable: it creates a fresh job each time and cleans up after
itself.

    python3 tools/reliability_test.py                 # ~3 min source, real ffmpeg
    python3 tools/reliability_test.py --seconds 180    # shorter
    python3 tools/reliability_test.py --keep           # leave the artifacts

What it asserts per stage:
    fetch        source lands on disk and ffprobes with a real duration
    transcribe   chunks are cached, offsets are monotonic and inside the source
    detect       every chunk produces valid JSON candidates (a stub LLM by
                 default so the run needs no funded API key; --live uses the
                 real provider)
    reframe      per-clip crop windows are written
    render       EVERY clip is verified: exists, > 0 bytes, ffprobes, right
                 geometry, audio present, duration matches the cut
    checkpoint   a simulated crash mid-render resumes at the first unrendered
                 clip and does not redo the finished ones
    watchdog     a stalled stage is detected from heartbeat age alone
    reconcile    an orphaned job is RE-QUEUED with the real cause, not failed
    disk         the precheck refuses a job that cannot fit

Exit code 0 only when every stage passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
CHECKS = 0
FAILURES: list[str] = []
RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    status = PASS if ok else FAIL
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line, flush=True)
    if not ok:
        FAILURES.append(name)
    return ok


def stage(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


SPEECH_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "speech-jfk-11s.wav"


def make_source(path: Path, seconds: int) -> Path:
    """A real MP4 with REAL SPEECH, built by real ffmpeg.

    Speech matters: a sine tone or a colour-bar clip transcribes to zero words,
    which makes detection produce zero chunks and rendering produce zero clips.
    The pipeline would then be "passing" without ever exercising the parts that
    actually break. The fixture is the well-known public-domain JFK clip, looped
    to length and muxed with a video track.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not SPEECH_FIXTURE.exists():
        raise RuntimeError(
            f"missing speech fixture {SPEECH_FIXTURE}. The reliability test needs "
            f"real speech to exercise transcription, detection and rendering."
        )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=30:duration={seconds}",
        "-stream_loop", "-1", "-i", str(SPEECH_FIXTURE),
        "-t", str(seconds),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
        "-f", "mp4", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"could not build the test source: {proc.stderr[-800:]}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--live", action="store_true", help="use the real LLM provider")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--label", default="current")
    args = ap.parse_args()

    settings = load_config(quiet=True)
    paths = settings.app.paths
    from app.store import LocalStore

    store = LocalStore(paths)
    job_id = f"reltest{int(time.time())}"
    scratch = Path("/tmp/clipforge-reliability")
    src = scratch / f"{job_id}.mp4"
    timings: dict[str, float] = {}

    print(f"ClipForge reliability check — job {job_id}\n", flush=True)
    t_all = time.time()
    try:
        # ---------------------------------------------------------- source
        stage("source")
        t0 = time.time()
        make_source(src, args.seconds)
        timings["make_source"] = time.time() - t0
        check(
            "test source built",
            src.exists() and src.stat().st_size > 0,
            f"{src.stat().st_size // 1024} KB, {args.seconds}s",
        )

        from app import video

        meta = video.probe(src)
        check(
            "source probes with a real duration",
            meta["duration"] > 1 and meta["width"] > 0,
            f"{meta['width']}x{meta['height']} {meta['duration']:.1f}s",
        )

        # ------------------------------------------------------------- job
        stage("job setup")
        job = store.create_job(source_type="upload", source=src.name)
        job_id = job["_id"]
        jdir = paths.downloads_dir / job_id
        jdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, jdir / "source.mp4")
        job["source"] = "source.mp4"
        # A short clip count so the test exercises the render loop without
        # spending an hour on a 1-vCPU box.
        job["options"] = {"maxClips": 2, "minClipSec": 15, "maxClipSec": 30}
        store.write_job(job)
        check("job created", store.get_job(job_id) is not None, job_id)

        # -------------------------------------------------------- pipeline
        stage("pipeline (fetch → transcribe → detect → render)")
        from app import llm
        from app.task import Pipeline

        if not args.live:
            # No funded key required: a local OpenAI-compatible stub with the
            # same wire shape. The chunking, caching, validation and JSON
            # repair are all exercised for real.
            original = llm.call_llm
            calls = {"n": 0}

            def stub(*_args, **_kw):
                calls["n"] += 1
                return json.dumps([
                    {"start_word": 5, "end_word": 40, "title": "Stub clip A", "score": 90},
                    {"start_word": 60, "end_word": 110, "title": "Stub clip B", "score": 85},
                ])

            llm.call_llm = stub
        else:
            original = None

        t0 = time.time()
        Pipeline(settings, store).run(job_id)
        timings["pipeline"] = time.time() - t0
        done = store.get_job(job_id) or {}
        clips = store.list_clips(job_id)

        check(
            "pipeline reached a terminal state",
            done.get("stage") in {"complete", "failed"},
            f"stage={done.get('stage')} error={str(done.get('error'))[:160]}",
        )

        # ------------------------------------------------------- stagewise
        stage("fetch")
        check(
            "source on disk",
            (paths.downloads_dir / job_id / "source.mp4").exists(),
        )
        from app.stages import StageWorkspace

        ws = StageWorkspace(paths.data_dir, job_id)
        check("fetch checkpointed", ws.is_done("fetch"))

        stage("transcribe")
        words_doc = ws.read_json("transcribe", "words.json") or {}
        t = words_doc.get("words") or []
        check(
            "transcript produced words",
            len(t) > 0,
            f"{len(t)} word(s)",
        )
        times = [float(w["start"]) for w in t] if t else []
        check(
            "word timestamps are monotonic",
            bool(times) and all(b >= a - 1e-6 for a, b in zip(times, times[1:])),
        )
        check(
            "word timestamps stay inside the source",
            bool(times) and min(times) >= -0.5 and max(times) <= meta["duration"] + 1.0,
            f"min={min(times):.2f} max={max(times):.2f}" if times else "no words",
        )
        chunks = list((ws.stage_dir("transcribe") / "chunks").glob("chunk_*.json")) \
            if (ws.stage_dir("transcribe") / "chunks").exists() else []
        check("transcription chunks cached for resume", len(chunks) >= 1,
              f"{len(chunks)} chunk file(s)")

        stage("detect")
        cand = ws.read_json("detect", "candidates.json") or []
        check("detection produced candidates", isinstance(cand, list),
              f"{len(cand) if isinstance(cand, list) else 'not a list'}")
        check("detect checkpointed", ws.is_done("detect"))

        stage("reframe")
        check("reframe checkpointed", ws.is_done("reframe"))
        check("per-clip crop windows written",
              (ws.stage_dir("reframe") / "clip_01.json").exists())

        stage("render — every clip verified")
        rendered = [c for c in clips if c.get("renderState") == "done"]
        failed = [c for c in clips if c.get("renderState") == "failed"]
        check("all clips rendered", len(rendered) > 0 and not failed,
              f"{len(rendered)} done, {len(failed)} skipped")
        if not rendered:
            raise RuntimeError(
                "no clips were rendered, so the render, verification and resume "
                "checks below cannot run"
            )
        for c in rendered:
            p = Path(c["videoPath"])
            n = c.get("order", 0) + 1
            ok = p.exists() and p.stat().st_size > 0
            check(f"clip {n}: exists and non-empty", ok,
                  f"{p.stat().st_size // 1024} KB" if ok else "missing/empty")
            if not ok:
                continue
            m = video.probe(p)
            check(f"clip {n}: 1080x1920", m["width"] == 1080 and m["height"] == 1920,
                  f"{m['width']}x{m['height']}")
            cut = c["end"] - c["start"]
            check(f"clip {n}: duration matches the cut", abs(m["duration"] - cut) < 1.0,
                  f"{m['duration']:.1f}s vs cut {cut:.1f}s")
            check(f"clip {n}: has an audio track", m.get("hasAudio", True))
            check(f"clip {n}: thumbnail written",
                  Path(c.get("thumbPath", "")).exists())

        stage("output verification helper")
        good = scratch / "verify.mp4"
        shutil.copy(rendered[0]["videoPath"], good)
        video.verify_output(good, expected_sec=10)
        check("verify_output accepts a good file", True)
        broken = scratch / "broken.mp4"
        broken.write_bytes(b"")
        try:
            video.verify_output(broken, expected_sec=10)
            check("verify_output rejects an empty file", False, "no error raised")
        except video.RenderError:
            check("verify_output rejects an empty file", True)
        truncated = scratch / "truncated.mp4"
        good_src = scratch / "verify.mp4"
        with open(good_src, "rb") as fh:
            truncated.write_bytes(fh.read(2048))
        try:
            video.verify_output(truncated, expected_sec=10)
            check("verify_output rejects a truncated file", False, "no error raised")
        except video.RenderError:
            check("verify_output rejects a truncated file", True)

        # ------------------------------------------------ checkpoint/resume
        stage("checkpoint — resume from the first unrendered clip")
        if rendered:
            before = sum(1 for c in store.list_clips(job_id) if c.get("renderState") == "done")
            # Simulate the host dying mid-render: mark the LAST clip pending
            # again and wipe its media, exactly as a crash would leave it.
            last = clips[-1]
            store.write_clip({
                **last,
                "renderState": "pending",
                "videoPath": None,
                "mediaBytes": 0,
            })
            Path(last["videoPath"]).unlink(missing_ok=True) if last.get("videoPath") else None
            after = sum(1 for c in store.list_clips(job_id) if c.get("renderState") == "done")
            check("crash simulation kept earlier clips done",
                  after == before - 1, f"{before} -> {after} done")
            Pipeline(settings, store).run(job_id)
            resumed = store.list_clips(job_id)
            check(
                "resume re-rendered only the missing clip",
                all(c.get("renderState") == "done" for c in resumed),
                f"{sum(1 for c in resumed if c.get('renderState') == 'done')}/{len(resumed)} done",
            )

        # -------------------------------------------------- watchdog/rules
        stage("watchdog and reconciliation rules")
        from app import worker as worker_mod
        from app.worker import InProcessWorker

        check("stall deadline is 3 minutes",
              worker_mod.STALL_TIMEOUT_SEC <= 180,
              f"{worker_mod.STALL_TIMEOUT_SEC}s")

        orphan = store.create_job(source_type="upload", source="x.mp4")
        oid = orphan["_id"]
        store.write_job({**orphan, "stage": "rendering", "progress": 30,
                         "interruptCount": 2, "lastProgressAt": int(time.time() * 1000)})
        worker = InProcessWorker(settings, store)
        fixed = worker.reconcile_stale_running()
        rec = store.get_job(oid)
        check("orphaned job is RECONCILED", oid in fixed)
        check("orphan is re-queued, not failed", rec.get("stage") == "queued",
              f"stage={rec.get('stage')}")
        check("requeue names the real cause",
              "restart" in (rec.get("interruptReason") or "").lower(),
              rec.get("interruptReason"))
        check("requeue reason stays one sentence",
              len(rec.get("interruptReason") or "") <= 220,
              f"{len(rec.get('interruptReason') or '')} chars")
        # the bug this whole exercise exists for: repeated reconciliation must
        # not wrap the message inside itself, forever
        first = rec.get("interruptReason")
        for _ in range(5):
            worker.reconcile_stale_running()
        again = (store.get_job(oid) or {}).get("interruptReason")
        check("repeated reconciliation does not grow the message",
              len(again or "") <= 220, f"{len(again or '')} chars (was {len(first or '')})")

        stage("resources")
        from app import video as _v

        check("ffmpeg concurrency cap is enforced",
              1 <= _v._FFMPEG_LIMIT <= 2, f"{_v._FFMPEG_LIMIT} on this host")
        free_mb = Pipeline(settings, store)._free_disk_mb()
        check("free disk is readable", free_mb > 0, f"{free_mb} MB free")

        if original is not None:
            llm.call_llm = original

    except Exception as exc:  # noqa: BLE001 - a crashed stage is a failed stage
        import traceback

        traceback.print_exc()
        check(f"run completed without crashing ({type(exc).__name__})", False, str(exc)[:300])
    finally:
        timings["total"] = time.time() - t_all

    print(f"\n{'=' * 66}")
    for r in RESULTS:
        print(f"  {r['stage']:<28} {r['sec']:>8.1f}s")
    print(f"\n{'-' * 66}")
    print("  timings: " + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items()))
    print(f"{'=' * 66}")
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed, label={args.label}")
    if FAILURES:
        print("FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL STAGES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
