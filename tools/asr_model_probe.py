#!/usr/bin/env python3
"""Measure what each Whisper model actually costs on THIS host.

The model must be chosen by measurement, not by a table: `app.membudget` still
carries a conservative estimate (`medium` = 1855 MB at 30 s) that the ladder
uses to SKIP models which would in fact fit. This probe loads each model in an
isolated, RLIMIT_AS-capped child (:mod:`app.asr_iso`), decodes one real window
of real speech, and prints the child's own `VmHWM` peak plus the wall time.

    python tools/asr_model_probe.py --wav data/work/<job>/transcribe/chunks/source.wav \
        --start 30 --seconds 30 --models large-v3-turbo medium small

Never prints a key and never touches the network: the model comes from the
local Hugging Face cache.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import asr_iso, membudget  # noqa: E402
from app.subtitle import ffmpeg_bin  # noqa: E402


def slice_wav(src: Path, dest: Path, start: float, seconds: float) -> Path:
    subprocess.run(
        [
            ffmpeg_bin(), "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(dest),
        ],
        check=True, capture_output=True, timeout=600,
    )
    with wave.open(str(dest), "rb") as fh:
        if fh.getframerate() != 16000 or fh.getnchannels() != 1:
            raise SystemExit(f"{dest} is not 16 kHz mono")
    return dest


def cgroup_now() -> str:
    b = membudget.current_budget()
    return b.describe()


def _rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


def probe(model: str, wav: Path, limit_mb: float | None, rss_cap_mb: float,
          threads: int = 1) -> dict:
    """Load + decode once; report the child's own VmHWM and a sampled peak.

    `limit_mb` sets RLIMIT_AS (what :mod:`app.asr_iso` uses). RLIMIT_AS counts
    VIRTUAL address space, which for CTranslate2 is far larger than the RSS the
    OOM killer looks at, so a model can be refused by that gate and still be
    safe to run. `rss_cap_mb` is the real guard here: a watchdog kills the
    child if its RSS crosses the cap.
    """
    started = time.time()
    killed = False
    sampled = 0.0
    try:
        proc = asr_iso.AsrProcess(
            model, compute="int8", threads=threads, limit_mb=limit_mb,
            log_path=ROOT / "data" / "logs" / "asr_probe.log",
        )
    except Exception as exc:  # noqa: BLE001 - a model that cannot load is a result
        return {"model": model, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "load_sec": round(time.time() - started, 1), "peak_rss_mb": 0.0,
                "rss_cap_mb": rss_cap_mb}
    load_sec = time.time() - started
    pid = proc.proc.pid

    def watch() -> None:
        nonlocal killed, sampled
        while proc.proc.poll() is None:
            mb = _rss_mb(pid)
            sampled = max(sampled, mb)
            if mb > rss_cap_mb:
                killed = True
                proc.kill()
                return
            time.sleep(0.05)

    import threading

    threading.Thread(target=watch, daemon=True).start()
    try:
        reply = proc.decode(wav, language=None, beam=5, vad=True)
    except Exception as exc:  # noqa: BLE001
        reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        peak = max(sampled, _rss_mb(pid))
        proc.kill()
    wall = time.time() - started
    if killed:
        return {"model": model, "ok": False,
                "error": f"killed by the RSS watchdog at {rss_cap_mb:.0f} MB",
                "load_sec": round(load_sec, 1), "peak_rss_mb": round(peak, 1),
                "rss_cap_mb": rss_cap_mb}
    if not reply.get("ok"):
        return {"model": model, "ok": False, "error": str(reply.get("error"))[:300],
                "load_sec": round(load_sec, 1),
                "peak_rss_mb": round(float(reply.get("peak_rss_mb") or peak), 1),
                "rss_cap_mb": rss_cap_mb}
    words = reply.get("words") or []
    text = " ".join(str(w.get("word", "")).strip() for w in words)
    return {
        "model": model,
        "ok": True,
        "load_sec": round(load_sec, 1),
        "wall_sec": round(wall, 1),
        "peak_rss_mb": round(max(float(reply.get("peak_rss_mb") or 0.0), peak), 1),
        "language": reply.get("language"),
        "words": len(words),
        "text": text[:300],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, type=Path)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--models", nargs="+",
                    default=["large-v3-turbo", "medium", "small"])
    ap.add_argument("--limit-mb", type=float, default=0.0,
                    help="RLIMIT_AS for the child; default = no RLIMIT (RSS watchdog only)")
    ap.add_argument("--rss-cap-mb", type=float, default=1500.0,
                    help="kill the child if its RSS crosses this (default 1500 MB)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    limit_mb = (membudget.cgroup_limit_bytes() or 0) / (1024 * 1024)
    limit = args.limit_mb or None
    tmp = ROOT / "data" / "logs"
    tmp.mkdir(parents=True, exist_ok=True)
    piece = tmp / f"probe_{int(args.start)}_{int(args.seconds)}.wav"
    slice_wav(args.wav, piece, args.start, args.seconds)

    print(f"host: cgroup limit {limit_mb:.0f} MB, RLIMIT_AS {limit or 'none'}, "
          f"RSS watchdog {args.rss_cap_mb:.0f} MB, threads {args.threads}, "
          f"window {args.seconds:.0f}s")
    print(f"now:  {cgroup_now()}")
    results = []
    for model in args.models:
        print(f"\n--- {model} ---", flush=True)
        res = probe(model, piece, limit, args.rss_cap_mb, args.threads)
        results.append(res)
        if res["ok"]:
            print(f"  peak RSS {res['peak_rss_mb']:.0f} MB | load {res['load_sec']}s | "
                  f"total {res['wall_sec']}s | {res['words']} words | lang {res['language']}")
            print(f"  text: {res['text'][:160]}")
        else:
            print(f"  FAILED after {res.get('load_sec', '?')}s: {res['error'][:200]}")
        print(f"  after: {cgroup_now()}")
        asr_iso.trim_parent()

    payload = {"limit_mb": limit, "rss_cap_mb": args.rss_cap_mb,
               "window_sec": args.seconds, "results": results}
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    piece.unlink(missing_ok=True)
    return 0 if any(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
