"""End-to-end pipeline test on a long source, run in resumable slices.

Why slices: this host is capped at ONE cpu and 2 GiB, so transcribing a
55-minute interview with Whisper `small` takes the better part of an hour and
rendering twenty clips takes longer still. A terminal command cannot run that
long. So the real `Pipeline` is driven in time-boxed child processes, and
every slice resumes from the pipeline's OWN stage/chunk cache — the same
mechanism a crash, a pause or a container recycle uses. Nothing is stubbed and
nothing is simulated: this is `Pipeline.run(job_id)` on a real 55-minute
video, and the slices only decide when to stop looking at it.

  python tools/e2e_long_test.py submit      # create the job (as the API does)
  python tools/e2e_long_test.py slice 140   # run up to 140s, then report
  python tools/e2e_long_test.py status      # stage, progress, time plan
  python tools/e2e_long_test.py verify      # check every rendered clip
  python tools/e2e_long_test.py logs        # the job log

Source: "How evolution works, in 54 minutes" — a real long-form interview, so
there is genuine speech to transcribe and real moments to find.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE_URL = "https://www.youtube.com/watch?v=h18ohlI-qQc"
STATE = ROOT / "data" / "e2e_long_job.json"


def job_id() -> str | None:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))["jobId"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def load():
    from app.config import load_config
    from app.store import LocalStore

    settings = load_config(quiet=True)
    return settings, LocalStore(settings.app.paths)


def cmd_submit() -> int:
    settings, store = load()
    jid = job_id()
    if jid and store.get_job(jid):
        print(f"already submitted: {jid} ({store.get_job(jid)['stage']})")
        return 0
    job = store.create_job(source_type="url", source=SOURCE_URL)
    jid = job["_id"]
    store.append_job_log(jid, f"e2e test submitted: {SOURCE_URL}")
    STATE.write_text(json.dumps({"jobId": jid, "source": SOURCE_URL}), encoding="utf-8")
    print(f"job {jid}")
    return 0


def cmd_slice(budget: int, provider: str = "") -> int:
    """Run the real pipeline for up to `budget` seconds, then stop looking.

    `provider` pins the detection backend for this slice only, by env var, so
    the test can choose a backend without editing the deployment's config.
    """
    jid = job_id()
    if not jid:
        print("no job — run 'submit' first")
        return 1
    started = time.time()
    env = dict(os.environ)
    if provider:
        env["CLIPFORGE_LLM_PROVIDER"] = provider
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_slice", jid, str(budget)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=env,
    )
    out: list[str] = []
    deadline = time.time() + budget + 12
    while proc.poll() is None and time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            out.append(line.rstrip())
    if proc.poll() is None:
        proc.kill()
        proc.wait()
        print(f"--- slice stopped at the {budget}s mark (resumable) ---")
    else:
        for line in out[-25:]:
            print(line)
    print(f"--- slice wall time {time.time() - started:.0f}s ---")
    return cmd_status()


def cmd_status() -> int:
    jid = job_id()
    if not jid:
        print("no job — run 'submit' first")
        return 1
    settings, store = load()
    job = store.get_job(jid)
    if job is None:
        print("job missing")
        return 1
    print(f"job {jid}  stage={job['stage']}  progress={round(job.get('progress') or 0)}%")
    if job.get("progressDetail"):
        print(f"  detail: {job['progressDetail']}")
    if job.get("error"):
        print(f"  ERROR: {job['error']}")
        if job.get("errorDetail"):
            print("  " + "\n  ".join(str(job['errorDetail']).strip().splitlines()[-12:]))
    if job.get("durationSec"):
        print(f"  source: {job['durationSec'] / 60:.1f} min")
    if job.get("clipTarget"):
        print(f"  clip target: {job['clipTarget']} — {job.get('clipTargetReason')}")
    if job.get("candidatesDetected") is not None:
        print(f"  detection: {job['candidatesDetected']} found, "
              f"{job.get('clipsAboveScore')} above {job.get('clipScoreFloor')}, "
              f"{len(store.list_clips(jid))} rendered")
    plan = job.get("stagePlan") or []
    if plan:
        print("  time plan:")
        for row in plan:
            mark = "->" if row.get("current") else "  "
            print(f"    {mark} {row['label']:<18} ~{row['sec'] / 60:5.1f} min  ({row['source']})")
    print(f"  stageEtaSec={job.get('stageEtaSec')} totalEtaSec={job.get('totalEtaSec')}")
    return 0


def cmd_retry() -> int:
    """Put a failed job back in the queue.

    The stage cache is untouched, so fetch and transcribe are reused and only
    the stage that failed is re-run — which chunks of detection were already
    billed is preserved, not re-spent.
    """
    jid = job_id()
    if not jid:
        return 1
    settings, store = load()
    job = store.get_job(jid)
    if not job:
        print("job missing")
        return 1
    job.update({"stage": "queued", "error": None, "errorDetail": None,
                "pauseRequested": False, "progress": 0})
    store.write_job(job)
    store.append_job_log(jid, "e2e test: retry requested")
    print(f"job {jid} reset to queued")
    return 0


def cmd_seed_candidates() -> int:
    """Assemble candidates from whatever detection chunks are already cached.

    Both free AI tiers ran out mid-detection (Groq's 200k tokens/day, Gemini
    free's 20 requests/day), so the last three chunks cannot be billed today.
    Everything downstream of detection is unaffected by that, and it is the
    half this test has not exercised yet, so the cached chunks are merged with
    the SAME gate and cap the real detect stage would have applied, written as
    the stage's normal output, and the pipeline continues into reframe+render.
    """
    from app import clipcount

    jid = job_id()
    if not jid:
        return 1
    settings, store = load()
    from app.stages import StageWorkspace

    ws = StageWorkspace(settings.app.paths.data_dir, jid)
    chunk_dir = ws.stage_dir("detect") / "chunks"
    files = sorted(chunk_dir.glob("*.json")) if chunk_dir.exists() else []
    if not files:
        print("no cached chunks to seed from")
        return 1
    candidates: list[dict] = []
    for f in files:
        candidates.extend(json.loads(f.read_text(encoding="utf-8")))
    job = store.get_job(jid) or {}
    target = int(job.get("clipTarget") or 23)
    floor = float(settings.gemini.min_score)
    top, above, below = clipcount.gate(candidates, target, floor)
    ws.write_json("detect", "candidates.json", top)
    ws.mark_done("detect", artifacts=["detect/candidates.json"], count=len(top))
    print(f"seeded {len(top)} candidate(s) from {len(files)} cached chunk(s): "
          f"{len(candidates)} found, {above} above the {floor:.0f} floor, {below} below, "
          f"cap {target}")
    return 0


def cmd_logs() -> int:
    jid = job_id()
    if not jid:
        return 1
    settings, store = load()
    for line in (store.get_job(jid) or {}).get("logs", [])[-60:]:
        print(f"  {line}")
    return 0


def cmd_verify() -> int:
    """Check the OUTPUTS, not just that the run finished."""
    from app import video

    jid = job_id()
    if not jid:
        return 1
    settings, store = load()
    job = store.get_job(jid) or {}
    clips = store.list_clips(jid)
    out_dir = settings.app.paths.outputs_dir / jid
    print(f"verifying {len(clips)} clip(s) from {jid}")
    if not clips:
        print("  NO CLIPS — nothing to verify")
        return 1
    bad = 0
    for i, clip in enumerate(clips, 1):
        path = Path(str(clip.get("videoPath") or out_dir / f"clip_{i:02d}.mp4"))
        if not path.exists():
            print(f"  clip {i}: MISSING {path}")
            bad += 1
            continue
        size = path.stat().st_size
        try:
            meta = video.probe(path)
        except Exception as exc:
            print(f"  clip {i}: UNREADABLE — {exc}")
            bad += 1
            continue
        want = round(float(clip["end"]) - float(clip["start"]), 1)
        got = round(float(meta.get("duration") or 0), 1)
        cues = clip.get("captionCues") or []
        words = sum(len(c.get("words") or []) for c in cues)
        over = [c for c in cues if c.get("uncertain")]
        drift = abs(got - want)
        ok = size > 100_000 and meta.get("width") == settings.video.width \
            and meta.get("height") == settings.video.height and drift <= 2.0 and cues
        bad += 0 if ok else 1
        print(
            f"  clip {i:2d}: {size / 1e6:6.2f} MB  {meta.get('width')}x{meta.get('height')}  "
            f"{got:5.1f}s (cut {want:5.1f}s, drift {drift:+.1f})  "
            f"{len(cues):3d} captions, {words:4d} word tags, {len(over)} flagged  "
            f"{'OK' if ok else '<-- PROBLEM'}"
        )
    print(f"  {len(clips) - bad}/{len(clips)} clips verified")
    if job.get("clipTarget"):
        print(f"  target was {job['clipTarget']}, rendered {len(clips)}")
    return 1 if bad else 0


def cmd_slice_inner(jid: str, budget: int) -> int:
    """The actual work, in its own process so a kill cannot corrupt state."""
    from app.task import Pipeline

    settings, store = load()
    p = Pipeline(settings, store)
    started = time.time()

    def ticker() -> None:
        # Print stage transitions as they happen so a killed slice still shows
        # how far it got.
        last = None
        while time.time() - started < budget:
            job = store.get_job(jid) or {}
            key = (job.get("stage"), round(job.get("progress") or 0))
            if key != last:
                print(f"  [{time.time() - started:6.0f}s] {key[0]} {key[1]}% "
                      f"{(job.get('progressDetail') or '')[:70]}")
                last = key
            time.sleep(2)

    import threading

    threading.Thread(target=ticker, daemon=True).start()
    p.run(jid)
    print(f"  pipeline returned after {time.time() - started:.0f}s")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    cmd = args[0]
    if cmd == "submit":
        return cmd_submit()
    if cmd == "slice":
        return cmd_slice(
            int(args[1]) if len(args) > 1 else 140,
            args[2] if len(args) > 2 else "",
        )
    if cmd == "_slice":
        return cmd_slice_inner(args[1], int(args[2]))
    if cmd == "status":
        return cmd_status()
    if cmd == "retry":
        return cmd_retry()
    if cmd == "seed":
        return cmd_seed_candidates()
    if cmd == "logs":
        return cmd_logs()
    if cmd == "verify":
        return cmd_verify()
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
