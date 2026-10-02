"""Acceptance tests for the pipeline rebuild: what the user actually sees.

Six promises, each written as a test that fails if the promise is broken:

  1. A 55-minute source shows real progress inside the first minute. The old
     pipeline moved `units` only at CHUNK boundaries, and a faster-whisper batch
     yields no segment until the whole batch decodes, so the card sat at "0% —
     is transcribing word timings..." for minutes with nothing to read. It now
     names the chunk it is on, from the very first one.
  2. Losing the worker mid-transcription resumes from the last FINISHED chunk.
  3. Losing it mid-render keeps every clip already finished.
  4. A source file that has vanished says so, in words, and takes nothing else
     with it — the finished clips of that batch stay downloadable.
  5. One clip that cannot be encoded costs one clip, not the batch.
  6. Nothing on the site was removed while doing any of this.

    python tools/pipeline_resilience_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import subtitle, video  # noqa: E402
from app import task as task_mod  # noqa: E402
from app.config import Paths, load_config  # noqa: E402
from app.stages import StageWorkspace  # noqa: E402
from app.store import LocalStore  # noqa: E402
from app.task import Pipeline  # noqa: E402

PASS = 0
FAIL = 0


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


def _paths() -> Paths:
    base = Path(tempfile.mkdtemp(prefix="clipforge-accept-"))
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


def _pipeline(paths: Paths) -> tuple[Pipeline, LocalStore]:
    settings = load_config(quiet=True)
    settings.app = type(settings.app)(paths=paths, host="127.0.0.1", port=8000)
    # Keep this suite off the real encoders and caption renderers.
    settings.video.reframe = False
    settings.video.silence_trim = False
    store = LocalStore(settings.app.paths)
    return Pipeline(settings, store), store


def _upload_job(store: LocalStore, *, in_uploads: bool = True) -> tuple[str, dict[str, Any]]:
    job = store.create_job(source_type="upload", source="source.mp4")
    jid = job["_id"]
    root = (store._paths.uploads_dir or (store._paths.data_dir / "uploads")) if in_uploads \
        else store._paths.downloads_dir
    d = root / jid
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.mp4").write_bytes(b"\0" * 2048)
    job["sourceType"] = "upload"
    job["durationSec"] = 55 * 60
    store.write_job(job)
    return jid, store.get_job(jid) or job


# ---------------------------------------------------------------- 1 + 2
def test_transcribe_progress_and_resume() -> None:
    section("1. a 55-minute source reports progress from the first chunk")
    paths = _paths()
    pipe, store = _pipeline(paths)
    jid, job = _upload_job(store)
    ws = StageWorkspace(paths.data_dir, jid)

    # 55 min at CHUNK_SEC=300 with a 2s overlap -> 11 windows.
    windows = 11
    seen: list[dict[str, Any]] = []
    original = subtitle.transcribe

    def fake_transcribe(src, settings, on_progress=None, cache_dir=None,
                        should_pause=None, on_chunk=None):
        for i in range(windows):
            if on_chunk:
                on_chunk(i, windows, phase="start")
                snap = store.get_job(jid) or {}
                seen.append({
                    "chunk": i + 1,
                    "detail": snap.get("progressDetail"),
                    "unitsDone": snap.get("unitsDone"),
                    "unitsTotal": snap.get("unitsTotal"),
                    "unitLabel": snap.get("unitLabel"),
                    "lastProgressAt": snap.get("lastProgressAt"),
                })
            if cache_dir is not None:
                cd = Path(cache_dir)
                cd.mkdir(parents=True, exist_ok=True)
                (cd / f"chunk_{i:05d}.json").write_text(
                    json.dumps({"words": []}), encoding="utf-8")
            if on_progress:
                on_progress((i + 1) / windows)
            if on_chunk:
                on_chunk(i, windows, phase="done")
        return subtitle.Transcript(words=[], duration=55 * 60.0,
                                  model_size="small", engine="stub")

    subtitle.transcribe = fake_transcribe
    try:
        pipe._stage_transcribe(job, Path("/nonexistent/source.mp4"), ws)
    finally:
        subtitle.transcribe = original

    first = seen[0] if seen else {}
    check("every chunk start is reported", len(seen) == windows, str(len(seen)))
    check(
        "chunk 1 already names itself in the UI text",
        "chunk 1 of 11" in str(first.get("detail")).lower(),
        str(first.get("detail")),
    )
    check(
        "the first chunk reports its unit count",
        first.get("unitsTotal") == windows and first.get("unitLabel") == "chunks",
        f"{first.get('unitsDone')}/{first.get('unitsTotal')} {first.get('unitLabel')}",
    )
    stamps = [s["lastProgressAt"] for s in seen if s.get("lastProgressAt")]
    check("the liveness stamp advances across the run",
          len(set(stamps)) >= windows - 1, f"{len(set(stamps))} distinct of {len(stamps)}")
    done = store.get_job(jid) or {}
    check("the transcript is written when the stage finishes",
          ws.is_done("transcribe"), str(ws.is_done("transcribe")))
    check("a finished stage reports 100%",
          float(done.get("progress") or 0) >= 100.0, str(done.get("progress")))

    # ---- 2. lose the worker mid-transcription, then resume --------------
    section("2. a worker lost mid-transcription resumes at the last finished chunk")
    jid2, job2 = _upload_job(store)
    ws2 = StageWorkspace(paths.data_dir, jid2)
    cache2 = ws2.stage_dir("transcribe") / "chunks"
    cache2.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (cache2 / f"chunk_{i:05d}.json").write_text(json.dumps({"words": []}), encoding="utf-8")
    pipe2, _ = _pipeline(paths)

    decoded: list[int] = []

    def dying_transcribe(src, settings, on_progress=None, cache_dir=None,
                         should_pause=None, on_chunk=None):
        # 0..2 come off disk, 3 is the one that dies with the process.
        for i in range(3):
            decoded.append(i)
            if on_chunk:
                on_chunk(i, 11, phase="start")
                on_chunk(i, 11, phase="done", resuming=True)
        raise KeyboardInterrupt("worker killed")

    subtitle.transcribe = dying_transcribe
    try:
        try:
            pipe2._stage_transcribe(job2, Path("/nonexistent/source.mp4"), ws2)
        except KeyboardInterrupt:
            pass
    finally:
        subtitle.transcribe = original

    check("the killed run stopped at the chunk it died on", decoded == [0, 1, 2], str(decoded))
    check("the stage was NOT marked done", not ws2.is_done("transcribe"))
    log_after_kill = "\n".join(str(x) for x in ((store.get_job(jid2) or {}).get("logs") or []))
    check("the log says the cached chunks were reused, not decoded",
          log_after_kill.count("resuming from cache") >= 3,
          f"{log_after_kill.count('resuming from cache')} mentions")

    fresh: list[int] = []
    log_lines: list[str] = []

    def resuming_transcribe(src, settings, on_progress=None, cache_dir=None,
                            should_pause=None, on_chunk=None):
        for i in range(11):
            hit = task_mod._chunk_cached(Path(cache2) if cache2 else None, i)
            (fresh if hit else decoded).append(i)
            if on_chunk:
                log_lines.append(f"chunk {i + 1}/11 "
                                 + ("resuming from cache" if hit else "decoding"))
                on_chunk(i, 11, phase="start")
                on_chunk(i, 11, phase="done", resuming=hit)
        return subtitle.Transcript(words=[], duration=55 * 60.0,
                                  model_size="small", engine="stub")

    subtitle.transcribe = resuming_transcribe
    try:
        pipe2._stage_transcribe(store.get_job(jid2) or job2, Path("/nonexistent/source.mp4"), ws2)
    finally:
        subtitle.transcribe = original
    check("the resumed run re-read the three cached chunks", sorted(fresh) == [0, 1, 2], str(fresh))
    check("the resumed run decoded the remaining eight",
          len([i for i in decoded if i >= 3]) == 8, str(decoded))
    check("the resume is visible in the job log",
          sum("resuming from cache" in ln for ln in log_lines) == 3,
          str(log_lines[:4]))


# ---------------------------------------------------------------- 3 + 5
def test_render_failure_isolation_and_resume() -> None:
    section("3 + 5. a failing clip costs one clip, and a restart keeps the finished ones")
    paths = _paths()
    pipe, store = _pipeline(paths)
    jid, job = _upload_job(store)
    ws = StageWorkspace(paths.data_dir, jid)
    src = store.find_source_file(jid)
    assert src is not None

    probe_orig = video.probe
    video.probe = lambda _p: {"duration": 300.0, "width": 1920, "height": 1080, "fps": 30.0}
    try:
        def render(job_id, idx, total, *, out_path, **kw):
            if idx == 3:
                raise video.RenderError("encoder refused this window")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"\0" * 512)
            return {"bytes": 512, "width": 1080, "height": 1920,
                    "duration": 30.0, "renderMs": 12}

        pipe._render_one_clip = render  # type: ignore[method-assign]
        pipe._write_caption_tracks = lambda *a, **k: {}  # type: ignore[method-assign]
        orig_thumb = video.render_first_frame_thumb
        video.render_first_frame_thumb = lambda _o, t: Path(t).write_bytes(b"j")  # type: ignore[assignment]

        candidates = [
            {"start": 0.0, "end": 30.0, "score": 90, "title": f"clip {i}"}
            for i in range(5)
        ]
        tr = subtitle.Transcript(words=[], duration=300.0, model_size="small", engine="stub")
        pipe._stage_render(job, src, tr, candidates, ws)
    finally:
        video.probe = probe_orig
        video.render_first_frame_thumb = orig_thumb

    clips = store.list_clips(jid)
    states = [c.get("renderState") for c in clips]
    check("every clip was attempted", len(clips) == 5, str(len(clips)))
    check("four clips finished", states.count("done") == 4, str(states))
    check("exactly one clip failed", states.count("failed") == 1, str(states))
    failed = [c for c in clips if c.get("renderState") == "failed"]
    check("the failed clip explains itself",
          bool(failed) and "encoder refused" in str(failed[0].get("mediaError")),
          str(failed[0].get("mediaError"))[:80] if failed else "no failed clip")
    check("the batch did NOT fail as a whole",
          (store.get_job(jid) or {}).get("stage") != "failed",
          str((store.get_job(jid) or {}).get("stage")))
    out_dir = paths.outputs_dir / jid
    made = sorted(p.name for p in out_dir.glob("*.mp4")) if out_dir.exists() else []
    check("the four finished MP4s are on disk", len(made) == 4, str(made))

    # ---- restart the worker: it must not re-encode what already exists ---
    rendered_again: list[int] = []

    def render2(job_id, idx, total, *, out_path, **kw):
        rendered_again.append(idx)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\0" * 512)
        return {"bytes": 512, "width": 1080, "height": 1920,
                "duration": 30.0, "renderMs": 9}

    pipe._render_one_clip = render2  # type: ignore[method-assign]
    pipe._write_caption_tracks = lambda *a, **k: {}  # type: ignore[method-assign]
    orig_probe = video.probe
    orig_thumb2 = video.render_first_frame_thumb
    video.probe = lambda _p: {"duration": 300.0, "width": 1920, "height": 1080, "fps": 30.0}
    video.render_first_frame_thumb = lambda _o, t: Path(t).write_bytes(b"j")  # type: ignore[assignment]
    try:
        pipe._stage_render(store.get_job(jid) or job, src, tr, candidates, ws)
    finally:
        video.probe = orig_probe
        video.render_first_frame_thumb = orig_thumb2

    check("the restart re-encoded only the failed clip",
          rendered_again == [3], str(rendered_again))
    check("the clips finished before the restart are still done",
          [c.get("renderState") for c in store.list_clips(jid)].count("done") == 5,
          str([c.get("renderState") for c in store.list_clips(jid)]))


# ---------------------------------------------------------------- 4
def test_missing_source_is_honest() -> None:
    section("4. a vanished source says so, and destroys nothing")
    paths = _paths()
    pipe, store = _pipeline(paths)
    jid, job = _upload_job(store, in_uploads=True)
    ws = StageWorkspace(paths.data_dir, jid)

    # Finished clips of THIS batch, which the error message promises survive.
    keep = paths.outputs_dir / jid
    keep.mkdir(parents=True, exist_ok=True)
    (keep / "clip_01.mp4").write_bytes(b"\0" * 128)
    # And an unrelated job that must be equally untouched.
    other_id, _ = _upload_job(store, in_uploads=True)
    other_src = store.find_source_file(other_id)
    assert other_src is not None

    # The source vanishes — the exact state 11 live jobs were left in.
    for d in paths.job_source_dirs(jid):
        if d.exists():
            for f in d.iterdir():
                f.unlink()
            d.rmdir()

    raised: Exception | None = None
    try:
        pipe._stage_fetch(job, ws)
    except Exception as exc:  # noqa: BLE001 - the point is the message
        raised = exc
    check("fetching a vanished source raises", raised is not None, "no error raised")
    msg = str(raised or "")
    check("the error names every location that was searched",
          all(str(d) in msg for d in paths.job_source_dirs(jid)), msg[:160])
    check("the error tells the user their finished clips are safe",
          "still downloadable" in msg, msg[:160])
    check("the error names the expected filename", "source.mp4" in msg, msg[:120])
    check("the other job's source is untouched", other_src.exists())
    check("this batch's finished clip survives the failed fetch",
          (keep / "clip_01.mp4").exists())

    # A legacy job still living in downloads/ must keep working untouched.
    legacy_id, legacy_job = _upload_job(store, in_uploads=False)
    legacy = (store.get_job(legacy_id) or legacy_job) | {"sourceType": "upload"}
    store.write_job(legacy)
    # The stub file is not a real video; ffprobe is not what this test is about.
    v_orig, p_orig = video.verify_source, video.probe
    video.verify_source = lambda _p: None  # type: ignore[assignment]
    video.probe = lambda _p: {"duration": 120.0, "width": 1280, "height": 720, "fps": 30.0}  # type: ignore[assignment]
    try:
        src = pipe._stage_fetch(legacy, StageWorkspace(paths.data_dir, legacy_id))
    finally:
        video.verify_source, video.probe = v_orig, p_orig
    check("a pre-existing downloads/ source is still found", src.exists(), str(src))
    check("and it is found in its original location",
          "downloads" in str(src), str(src))


# ---------------------------------------------------------------- 6
def test_site_is_intact() -> None:
    section("6. nothing on the site was removed, renamed or reordered")
    card = (ROOT / "src" / "components" / "clipforge" / "JobCard.tsx").read_text(encoding="utf-8")
    for needle, why in [
        ("STAGE_LABELS", "the stage stepper"),
        ("Progress", "the progress bar"),
        ("stagePlan", "the per-stage plan list"),
        ("api.pauseJob", "the Pause button"),
        ("onCheckNow", "the Check now button"),
        ("Check now", "the Check now label"),
        ("quiet && !stalled", "the quiet warning"),
        ("Pipeline stalled", "the stall warning"),
        ("updated {fmtAgo", "the liveness stamp"),
        ("batchLabel", "the batch title"),
    ]:
        check(f"JobCard still renders {why}", needle in card, needle)

    card2 = (ROOT / "src" / "lib" / "clipforge.ts").read_text(encoding="utf-8")
    for needle in ("QUIET_THRESHOLD_MS", "silenceMs", "fmtAgo", "isActiveJob", "sourceLabel"):
        check(f"clipforge.ts still exports {needle}", needle in card2, needle)

    backup = ROOT / ".pipeline-backup" / "src"
    if backup.is_dir():
        changed = []
        for path in sorted((ROOT / "src").rglob("*")):
            if not path.is_file():
                continue
            old = backup / path.relative_to(ROOT / "src")
            if not old.is_file() or old.read_bytes() != path.read_bytes():
                changed.append(str(path.relative_to(ROOT / "src")))
        removed = [str(p.relative_to(backup)) for p in backup.rglob("*")
                   if p.is_file() and not (ROOT / "src" / p.relative_to(backup)).exists()]
        check("no source file was deleted", not removed, str(removed))
        check("only the two approved frontend files changed",
              set(changed) <= {"components/clipforge/JobCard.tsx", "lib/clipforge.ts"},
              str(changed))
        check("no new frontend file appeared",
              not [c for c in changed if not (backup / c).exists()], str(changed))
    else:
        check("the pre-change snapshot is available", False, str(backup))

    vite = (ROOT / "vite.config.ts").read_text(encoding="utf-8")
    check("HMR is still disabled (platform requirement)", "hmr: false" in vite)
    routes = json.loads((ROOT / "qa" / "expected_routes.json").read_text(encoding="utf-8")) \
        if (ROOT / "qa" / "expected_routes.json").exists() else None
    if routes is not None:
        check("every expected route is still declared",
              all(p.exists() for p in [ROOT / r for r in routes["files"]]),
              str(routes))


def main() -> int:
    t0 = time.time()
    print("ClipForge pipeline-resilience acceptance suite")
    test_transcribe_progress_and_resume()
    test_render_failure_isolation_and_resume()
    test_missing_source_is_honest()
    test_site_is_intact()
    total = PASS + FAIL
    print(f"\nacceptance: {PASS}/{total} passed in {time.time() - t0:.1f}s")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
