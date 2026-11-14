#!/usr/bin/env python3
"""List every model ClipMake can load, with real files, sizes and hashes.

    .venv/bin/python tools/model_audit.py
    .venv/bin/python tools/model_audit.py --smoke --media tests/golden/fixtures/look_source.mp4

Never prints a key. `--smoke` loads each local Whisper model through the same
isolated child the worker uses and runs the real MediaPipe detector on a real
frame; no generated media is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import modelaudit  # noqa: E402
from app.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="load models and run real media")
    ap.add_argument("--media", type=Path, default=ROOT / "tests/golden/fixtures/look_source.mp4")
    args = ap.parse_args(argv)

    settings = load_config(quiet=True)
    rows = modelaudit.audit(settings)
    smoke: dict[str, object] = {}
    if args.smoke:
        from tools.asr_model_probe import probe, slice_wav  # noqa: PLC0415

        if not args.media.is_file():
            print(f"BLOCKED: real media not found: {args.media}", file=sys.stderr)
            return 2
        piece = ROOT / "data" / "logs" / "model_audit_18s.wav"
        piece.parent.mkdir(parents=True, exist_ok=True)
        slice_wav(args.media, piece, 2.0, 18.0)
        for row in rows:
            if row.get("role") != "transcription" or not row.get("loadable"):
                continue
            name = str(row["name"])
            started = time.time()
            result = probe(name, piece, None, 1500.0, threads=1)
            smoke[f"whisper:{name}"] = {
                "ok": result.get("ok"),
                "load_sec": result.get("load_sec"),
                "wall_sec": result.get("wall_sec"),
                "peak_rss_mb": result.get("peak_rss_mb"),
                "words": result.get("words"),
                "error": result.get("error"),
                "wall": round(time.time() - started, 1),
            }
        from app import reframe  # noqa: PLC0415

        t0 = time.time()
        try:
            windows, used = reframe.face_crop_windows(
                str(args.media), 0.0, min(14.0, 20.0), 1280, 720,
                sample_hz=2.0, shot_stats=[],
            )
            smoke["mediapipe:face"] = {
                "ok": True, "used": used, "windows": len(windows),
                "wall_sec": round(time.time() - t0, 2),
            }
        except Exception as exc:  # noqa: BLE001 - a model failure is a result
            smoke["mediapipe:face"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    payload = {"models": rows, "smoke": smoke}
    required_bad = [
        r for r in rows
        if r.get("state") in {"missing", "failed"}
        and "uploaded subtitles" not in str(r.get("role") or "")
    ]
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if not required_bad else 1

    print(f"{'model':42} {'role':24} {'state':12} {'size':>9}  sha256")
    for row in rows:
        size = f"{float(row.get('sizeMb') or 0):.1f} MB"
        sha = (row.get("sha256") or "—")[:16]
        print(f"{row['id'][:42]:42} {row['role'][:24]:24} {row['state'][:12]:12} {size:>9}  {sha}")
        print(f"  {row.get('path') or '—'}")
        print(f"  {row.get('reason')}")
    if smoke:
        print("\nsmoke:")
        for key, value in smoke.items():
            print(f"  {key}: {json.dumps(value, sort_keys=True)}")
    return 0 if not required_bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
