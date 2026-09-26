"""Re-verify a finished batch: every clip, and the detection stage on real data.

    python tools/batch_recheck.py c57d7782c548            # verify clips
    python tools/batch_recheck.py c57d7782c548 --detect   # + rerun detection

The `--detect` path is the end-to-end proof for an LLM provider swap: it
feeds the batch's REAL faster-whisper transcript (the same 974-word
`words.json` the original run used) through the configured provider and
checks that word indices still resolve to real timestamps.

If `GROQ_API_KEY` is not configured, detection runs against a local
OpenAI-compatible stub instead and says so loudly — the contract, the
chunking and the timing resolution are all still verified, but the real
api.groq.com round trip is NOT, and this tool will not pretend otherwise.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

PASS = 0
FAIL = 0
WARN = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


def warn(name: str) -> None:
    global WARN
    WARN += 1
    print(f"  warn {name}")


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return (json.loads(out).get("streams") or [{}])[0]


# --------------------------------------------------------------------------- #
# clip verification
# --------------------------------------------------------------------------- #

def verify_clips(job_id: str) -> None:
    job = json.loads((ROOT / "data" / "jobs" / f"{job_id}.json").read_text())
    out_dir = ROOT / "outputs" / job_id

    print(f"\n== job {job_id} ==")
    print(f"   stage={job.get('stage')} clipCount={job.get('clipCount')} "
          f"error={job.get('error')!r}")
    check("job completed", job.get("stage") == "complete" and not job.get("error"),
          str(job.get("error")))
    check("error field is null", job.get("error") is None)
    check("not flagged interrupted", not job.get("interrupted"))

    clips = [json.loads(p.read_text()) for p in sorted((ROOT / "data" / "clips").glob("*.json"))
             if json.loads(p.read_text()).get("jobId") == job_id]
    check("all clip docs present", len(clips) == job.get("clipCount"),
          f"{len(clips)} docs vs clipCount={job.get('clipCount')}")

    total = 0.0
    for c in sorted(clips, key=lambda x: x.get("order", 0)):
        name = Path(c["videoPath"]).name
        vid = out_dir / name
        if not vid.exists():
            check(f"{name} exists", False, str(vid))
            continue
        want_dur = float(c["end"]) - float(c["start"])
        st = ffprobe(vid)
        got_w = int(st.get("width") or 0)
        got_h = int(st.get("height") or 0)
        # frame count is the strict check: duration alone rounds too much
        fps = eval(st.get("avg_frame_rate", "0/1"))  # noqa: S307 - ffprobe "num/den"
        nb = int(st.get("nb_frames") or 0)
        total += want_dur
        check(f"{name}: 1080x1920", (got_w, got_h) == (1080, 1920), f"{got_w}x{got_h}")
        check(f"{name}: frame count matches the cut ({want_dur:.2f}s @ {fps:.0f}fps)",
              abs(nb / fps - want_dur) < 0.10 if fps else False,
              f"{nb} frames = {nb / fps if fps else 0:.2f}s, want {want_dur:.2f}s")
        thumb = out_dir / (vid.stem + ".jpg")
        check(f"{name}: thumbnail exists", thumb.exists() and thumb.stat().st_size > 5_000)

    check(f"total clip seconds match the job ({total:.1f}s)",
          abs(total - float(job.get("totalClipsSec") or 0)) < 1.0,
          f"summed {total:.1f}s vs job {job.get('totalClipsSec')}")


# --------------------------------------------------------------------------- #
# detection on the real transcript
# --------------------------------------------------------------------------- #

def _start_stub() -> tuple[str, threading.Thread, object]:
    """A local OpenAI-compatible server that returns a plausible JSON reply."""
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802
            import re as _re

            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            # The prompt numbers words as "[123] 12.30s word" — pick a
            # sub-range from the middle of THIS chunk, the way a real model
            # would, so the returned indices are always in range.
            prompt = (body.get("messages") or [{}])[-1].get("content", "")
            idx = [int(m) for m in _re.findall(r"\[(\d+)\]", prompt)]
            if idx:
                lo = idx[0]
                hi = idx[min(len(idx) - 1, lo - idx[0] + 120)]
            else:
                lo, hi = 0, 120
            reply = {"candidates": [{
                "start_word": lo, "end_word": hi, "score": 88,
                "title": f"stub clip from word {lo}", "hook_text": "watch this",
            }]}
            data = json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/openai/v1", t, httpd


def rerun_detection(job_id: str) -> None:
    from app import llm as llm_mod
    from app.config import load_config

    s = load_config(quiet=True)
    words_path = ROOT / "data" / "work" / job_id / "transcribe" / "words.json"
    if not words_path.exists():
        check("transcript words.json exists", False, str(words_path))
        return
    blob = json.loads(words_path.read_text())
    words = blob["words"]

    print(f"\n== detection re-run on the REAL transcript ({len(words)} words) ==")
    print(f"   engine={blob.get('engine')} model={blob.get('model')} "
          f"duration={blob.get('duration')}s")
    check("transcript is non-trivial", len(words) > 500, str(len(words)))

    active = llm_mod.provider(s)
    print(f"   provider={active} label={llm_mod.provider_label(s)}")
    key = s.groq_api_key if active == "groq" else s.gemini_api_key
    real = bool(key)

    if active == "groq" and not real:
        warn("no GROQ_API_KEY configured — running the Groq client against a LOCAL")
        warn("OpenAI-compatible stub. This verifies the contract, chunking and timing,")
        warn("but NOT a real api.groq.com call. Paste the key in the Keys tab and re-run.")
        url, _t, _httpd = _start_stub()
        s.groq.base_url = url
        s.groq.min_interval_sec = 0.0  # don't pace a local stub
        # detect_clips() re-checks the key itself, so the stub needs a
        # plausible non-empty value or it raises LlmNotConfigured first.
        key = "gsk_local_stub_key_not_a_real_credential"

    orig = type(s).groq_api_key
    type(s).groq_api_key = property(lambda self: key)  # type: ignore[assignment]
    try:
        found = llm_mod.detect_clips(words, s)
    except Exception as exc:  # noqa: BLE001 - report, don't explode
        check("detect_clips completed", False, f"{type(exc).__name__}: {exc}")
        return
    finally:
        type(s).groq_api_key = orig  # type: ignore[assignment]

    check("detect_clips returned clips", len(found) > 0, str(len(found)))
    for i, c in enumerate(found[:3]):
        # The word-index -> real-time resolution is the contract that matters.
        lo = int(round(c["start"] / 0.25)) if words else 0
        check(f"clip {i + 1} starts on a real word boundary "
              f"({c['start']:.2f}s)", any(abs(w["start"] - c["start"]) < 0.01 for w in words),
              str(c["start"]))
        check(f"clip {i + 1} ends after it starts ({c['end']:.2f}s > {c['start']:.2f}s)",
              c["end"] > c["start"])
        check(f"clip {i + 1} is within the media ({c['end']:.2f}s <= {blob['duration']:.1f}s)",
              c["end"] <= blob["duration"] + 0.5, str(c["end"]))
        print(f"      clip {i + 1}: {c['start']:.2f}-{c['end']:.2f}s "
              f"score={c['score']} title={c['title']!r}")
        del lo
    if real:
        print("   ✓ this was a REAL api.groq.com round trip")
    else:
        print("   ✗ NOT a real Groq call — add GROQ_API_KEY to complete this check")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    job_id = sys.argv[1]
    verify_clips(job_id)
    if "--detect" in sys.argv:
        rerun_detection(job_id)
    print(f"\n{PASS} passed, {FAIL} failed, {WARN} warning(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
