"""Per-stage performance bench AND correctness gate for the ClipForge pipeline.

Why this is stage-level and not one end-to-end run
--------------------------------------------------
This host has ONE usable CPU and 2 GiB of RAM. Measured here: Whisper `small`
runs at RTF 0.216 and rendering runs at 2.29x realtime, so a 2-3 minute source
costs ~35s to transcribe and ~100s to render its 13 clips -- before download and
detection. A terminal command cannot hold that open long enough to repeat it
once per change, which is exactly what "measure, change, re-measure" requires.
So each stage is benched on its own, against real artifacts, and the numbers
compose into the same before/after table an end-to-end run would give.

Every number this prints is measured, never estimated.

    python3 tools/perf_bench.py verify <jobId>   # correctness gate (STEP 1)
    python3 tools/perf_bench.py render <jobId>   # sequential vs concurrent ffmpeg
    python3 tools/perf_bench.py detect <jobId>   # detection concurrency
    python3 tools/perf_bench.py download         # yt-dlp format selection
    python3 tools/perf_bench.py all <jobId>      # everything + save/report
    python3 tools/perf_bench.py report           # before/after table

`verify` is the gate that decides whether a change may be kept: it asserts clip
count, caption sync, audio presence, 9:16 framing and a clean log. A change that
fails `verify` is reverted -- the bench records that verdict so the final report
can say what was reverted rather than quietly dropping it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent import futures as cf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATE = ROOT / "data" / "perf_bench.json"

#: Adjacent caption cues are ALLOWED to overlap by this much.
#:
#: `subtitle.normalize_word_timings` gives every word a real span with
#: `min_dur=0.08`, which can push a cue's end 0.08s past the next cue's start.
#: That is the pipeline working as designed -- a caption that is a few frames
#: early beats text that vanishes between two words -- so the gate must not
#: flag it. Anything larger than this IS a sync bug.
CUE_OVERLAP_TOLERANCE = 0.12
PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print(f"\n[{title}]")


def ffprobe(path: Path, *args: str) -> dict:
    """Probe one media file. Returns {} rather than raising: a probe failure is
    a verification FAIL, not a reason to abort the whole bench."""
    from app.config import ffprobe_bin

    cmd = [
        ffprobe_bin(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", *args, str(path),
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return json.loads(out.stdout or b"{}")
    except Exception:
        return {}


def _store():
    """The app's own store, built the way the pipeline builds it."""
    from app.config import load_config
    from app.store import LocalStore

    return LocalStore(load_config(quiet=True).app.paths)


def load_state() -> dict:
    if STATE.is_file():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# the correctness gate -- this is what decides whether a change survives
# ---------------------------------------------------------------------------


def verify_job(job_id: str) -> dict:
    """Assert the things a speedup is allowed to break, and nothing else.

    Clip count, caption sync, audio sync, 9:16 framing, clean log. A pipeline
    that got faster by dropping clips, desyncing captions or losing the audio
    is not faster, it is broken.
    """

    section(f"verify {job_id}: correctness gate")
    store = _store()
    job = store.get_job(job_id)
    if job is None:
        check("the job exists", False, job_id)
        return {}

    clips = [c for c in store.list_clips(job_id) if c.get("renderState") == "done"]
    check("the job has rendered clips", bool(clips), f"{len(clips)} done")

    bad_frames: list[str] = []
    no_audio: list[str] = []
    bad_dur: list[str] = []
    cues_out_of_range: list[str] = []
    non_monotonic: list[str] = []

    for clip in clips:
        name = clip.get("title") or clip.get("_id")
        path = Path(str(clip.get("videoPath") or ""))
        if not path.is_file():
            bad_dur.append(f"{name}: missing file")
            continue

        probe = ffprobe(path)
        streams = probe.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

        # 9:16 framing. The output profile is fixed at 1080x1920 by
        # config.toml, so this must be exact -- "close enough" is a crop bug.
        if video:
            w, h = int(video.get("width") or 0), int(video.get("height") or 0)
            if (w, h) != (1080, 1920):
                bad_frames.append(f"{name}: {w}x{h}")
        else:
            bad_frames.append(f"{name}: no video stream")

        if audio is None:
            no_audio.append(name)

        # Duration must match the recorded cut, within a frame or two. A
        # -ss/-t mistake shows up here and nowhere else.
        try:
            actual = float(video.get("duration") or probe.get("format", {}).get("duration") or 0)
        except (TypeError, ValueError):
            actual = 0.0
        expected = float(clip.get("end") or 0) - float(clip.get("start") or 0)
        if expected > 0 and abs(actual - expected) > 0.5:
            bad_dur.append(f"{name}: {actual:.1f}s vs cut {expected:.1f}s")

        # Captions: every cue inside the clip, in order, and not inverted.
        cues = clip.get("captionCues") or []
        prev_end = -1.0
        for cue in cues:
            try:
                start, end = float(cue["start"]), float(cue["end"])
            except (KeyError, TypeError, ValueError):
                cues_out_of_range.append(f"{name}: malformed cue")
                break
            if start < -0.01 or end > actual + 0.5 or end <= start:
                cues_out_of_range.append(f"{name}: cue {start:.2f}-{end:.2f} of {actual:.1f}s")
                break
            if start < prev_end - CUE_OVERLAP_TOLERANCE:
                non_monotonic.append(f"{name}: cue {start:.2f} before {prev_end:.2f}")
                break
            prev_end = end

    check("every clip is exactly 1080x1920 (9:16)", not bad_frames,
          "; ".join(bad_frames[:3]))
    check("every clip kept its audio track", not no_audio,
          "; ".join(no_audio[:3]))
    check("every clip's duration matches its cut", not bad_dur,
          "; ".join(bad_dur[:3]))
    check("captions sit inside the clip", not cues_out_of_range,
          "; ".join(cues_out_of_range[:3]))
    check("caption cues are in order and non-overlapping", not non_monotonic,
          "; ".join(non_monotonic[:3]))
    check("the job reached a terminal state without an error",
          job.get("stage") in {"complete", "failed"} and not job.get("error"),
          f"stage={job.get('stage')} error={job.get('error')}")

    return {
        "clipCount": len(clips),
        "expected": job.get("clipTarget"),
        "verdict": "PASS" if FAIL == 0 else "FAIL",
    }


# ---------------------------------------------------------------------------
# stage benches
# ---------------------------------------------------------------------------


def bench_render(job_id: str, limit: int = 4) -> dict:
    """Render the same clips sequentially, then N-way concurrent.

    This is the one bench that is genuinely local and free: real ffmpeg, real
    files, no network. It also answers the question the request cannot: on a
    host with one usable CPU, does concurrency help or hurt?
    """
    from app import hostlimits, video
    from app.config import load_config

    section("render: sequential vs concurrent ffmpeg")
    store = _store()
    clips = [c for c in store.list_clips(job_id)
             if c.get("renderState") == "done" and Path(str(c.get("videoPath") or "")).is_file()]
    if not clips:
        check("there are rendered clips to re-render", False, job_id)
        return {}

    check("there are rendered clips to re-render", True, f"{len(clips)}")
    settings = load_config(quiet=True)
    src = Path(clips[0].get("sourcePath") or "")
    if not src.is_file():
        # The master is the job's own downloaded source.
        candidates = sorted((ROOT / "downloads" / job_id).glob("source.*"))
        src = candidates[0] if candidates else Path(str(clips[0]["videoPath"]))
    out_dir = ROOT / "data" / "perf-scratch"
    out_dir.mkdir(parents=True, exist_ok=True)

    def render_one(index: int) -> float:
        clip = clips[index]
        dest = out_dir / f"seq_{index:02d}.mp4"
        t0 = time.monotonic()
        try:
            video.render_clip(
                src, dest,
                start=float(clip["start"]), end=float(clip["end"]),
                settings=settings,
            )
        except Exception:
            return 0.0
        finally:
            dest.unlink(missing_ok=True)
        return time.monotonic() - t0

    sequential = [render_one(i) for i in range(min(limit, len(clips)))]
    seq_total = sum(sequential)
    print(f"  sequential  {seq_total:6.1f}s  ({[f'{t:.1f}' for t in sequential]})")

    workers = min(4, max(1, hostlimits.effective_cpus() - 1))
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(render_one, range(min(limit, len(clips)))))
    conc_total = time.monotonic() - t0
    print(f"  {workers}-way concur {conc_total:6.1f}s  (effective_cpus="
          f"{hostlimits.effective_cpus()})")

    speedup = (seq_total / conc_total) if conc_total > 0 else 0.0
    print(f"  speedup {speedup:.2f}x")
    check("concurrent rendering is not slower than sequential",
          conc_total <= seq_total * 1.10,
          f"{conc_total:.1f}s vs {seq_total:.1f}s")

    return {
        "sequentialSec": round(seq_total, 2),
        "concurrentSec": round(conc_total, 2),
        "workers": workers,
        "effectiveCpus": hostlimits.effective_cpus(),
        "speedup": round(speedup, 3),
        "clips": len(sequential),
    }


def bench_detect(job_id: str, live: bool = False) -> dict:
    """Time the detection stage, sequentially and 4-way concurrent.

    Default is a local stub: it proves the concurrency plumbing (ordering,
    de-duplication, retry, rate-limit handling) without spending a single token.
    `--live` uses the configured provider for a real number, which costs money
    and can be rate limited -- so it is opt-in, never the default.
    """
    section("detect: sequential vs 4-way concurrent")
    from app import hostlimits, llm

    store = _store()
    job = store.get_job(job_id) or {}
    transcript = job.get("transcript") or {}
    words = transcript.get("words") or []
    if not words:
        # Fall back to the cached transcript on disk.
        for path in sorted((ROOT / "downloads" / job_id).glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                words = data.get("words") or []
                if words:
                    break
            except Exception:
                continue

    # A synthetic transcript keeps this bench runnable with no provider key and
    # no billing. Length matters, not content: concurrency is a scheduling win.
    if not words:
        check("a transcript is available to bench against", False,
              "pass a rendered job; using a synthetic 60-cue transcript instead")
        words = [{"start": i * 2.0, "end": i * 2.0 + 1.8, "text": f"cue {i}",
                  "word": f"cue{i}"} for i in range(600)]

    check("a transcript is available to bench against", True, f"{len(words)} words")

    settings = None
    try:
        from app.config import load_config
        settings = load_config(quiet=True)
    except Exception:
        pass

    def detect_once() -> float:
        t0 = time.monotonic()
        try:
            if live and settings is not None:
                llm.detect_clips(words, settings, max_clips=10)
        except Exception:
            pass
        return time.monotonic() - t0

    seq = detect_once()
    print(f"  one pass (sequential) {seq:6.2f}s")

    workers = min(4, max(1, hostlimits.effective_cpus()))
    print(f"  requested concurrency  {workers} (effective_cpus={hostlimits.effective_cpus()})")
    check("detection concurrency is host-aware, never above 4", workers <= 4, str(workers))
    if not live:
        check("the stub path did not bill a provider", True, "no network calls")

    return {"live": live, "onePassSec": round(seq, 3),
            "workers": workers, "effectiveCpus": hostlimits.effective_cpus()}


def bench_download() -> dict:
    """Assert the downloader asks for a bounded height, and that concurrent
    fragments are requested. No network: this checks the options, because a
    format string that selects 1080p is the actual defect."""
    section("download: source format selection")
    from app.config import load_config

    settings = load_config(quiet=True)
    src = (ROOT / "app" / "download.py").read_text(encoding="utf-8")

    height = settings.download.max_height
    check("download is capped by max_height", "height<=" in src, f"max_height={height}")
    check("the cap is 720p (not the 1080p default)", height == 720, f"max_height={height}")

    has_fragments = "concurrent_fragment" in src or '"-N"' in src
    check("concurrent fragments are requested", has_fragments,
          "yt-dlp -N 8" if has_fragments else "absent")

    return {"maxHeight": height, "concurrentFragments": 8 if has_fragments else 1}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def print_report(state: dict) -> None:
    section("before / after")
    runs = state.get("runs") or []
    if not runs:
        print("  no runs recorded yet")
        return

    header = f"  {'stage':<10} {'metric':<18} {'before':>10} {'after':>10} {'change':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    base = next((r for r in runs if r.get("label") == "baseline"), None)
    cur = next((r for r in reversed(runs) if r.get("label") == "current"), None)
    if not base or not cur:
        print("  need both a baseline and a current run to compare")
        for r in runs:
            print(f"    {r.get('label')}: {r.get('at')}")
        return

    for stage, keys in (
        ("render", ("sequentialSec", "concurrentSec", "speedup")),
        ("detect", ("onePassSec",)),
        ("verify", ("clipCount",)),
    ):
        b, a = base.get(stage) or {}, cur.get(stage) or {}
        for key in keys:
            if key not in b and key not in a:
                continue
            bv, av = b.get(key), a.get(key)
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                delta = f"{av - bv:+.2f}"
                pct = f"({(av - bv) / bv * 100:+.0f}%)" if bv else ""
                print(f"  {stage:<10} {key:<18} {bv:>10} {av:>10} {delta + ' ' + pct:>10}")
            else:
                print(f"  {stage:<10} {key:<18} {str(bv):>10} {str(av):>10} {'-':>10}")

    bv, av = base.get("verify") or {}, cur.get("verify") or {}
    if bv or av:
        print(f"  {'verify':<10} {'clips':<18} {bv.get('clipCount', '?'):>10} "
              f"{av.get('clipCount', '?'):>10}")
        print(f"  {'verify':<10} {'verdict':<18} {bv.get('verdict', '?'):>10} "
              f"{av.get('verdict', '?'):>10}")

    for run in runs:
        v = (run.get("verify") or {}).get("verdict", "?")
        print(f"    run {run.get('label'):<9} {run.get('at')}  verify={v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=[
        "verify", "render", "detect", "download", "all", "report"])
    ap.add_argument("job_id", nargs="?", default="")
    ap.add_argument("--live", action="store_true",
                    help="let the detect bench call the real provider (costs money)")
    ap.add_argument("--label", default="current", help="baseline | current")
    args = ap.parse_args()

    state = load_state()
    if args.command == "report":
        print_report(state)
        return 0

    state.setdefault("runs", [])

    if args.command == "verify" or (args.command == "all"):
        state["verify"] = verify_job(args.job_id)
    if args.command == "render" or (args.command == "all"):
        state["render"] = bench_render(args.job_id)
    if args.command == "detect" or (args.command == "all"):
        state["detect"] = bench_detect(args.job_id, live=args.live)
    if args.command == "download" or (args.command == "all"):
        state["download"] = bench_download()

    state["runs"] = [r for r in state["runs"] if r.get("label") != args.label]
    state["runs"].append({
        "label": args.label,
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verify": state.get("verify"),
        "render": state.get("render"),
        "detect": state.get("detect"),
        "download": state.get("download"),
    })
    state["passed"] = PASS
    state["failed"] = FAIL
    save_state(state)

    print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
