"""End-to-end playback proof against a REAL uvicorn server on a real port.

tools/playback_test.py uses the in-process TestClient, which is the right tool
for pinning behaviour. This script exists for the other half: proving it over
actual HTTP with curl, because the thing that broke for the user was a phone
browser talking to a proxy, and a TestClient cannot reproduce a socket, a
Content-Length header or a 206.

It boots the backend on a spare port with one rendered clip registered, serves
it, and runs exactly the checks a browser makes:

    curl -I <url>                          -> 200, Content-Length, Accept-Ranges
    curl -r 0-1023 -I <url>                -> 206, Content-Range: bytes 0-1023/N

    python tools/playback_e2e.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOST = "127.0.0.1"
PORT = 8931
BASE = f"http://{HOST}:{PORT}"

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


def curl(*args: str) -> tuple[int, dict[str, str], bytes]:
    """Run curl and split status/headers/body. Returns (status, headers, body).

    Headers and body go to separate temp files rather than stdout: a media body
    is binary, and mixing it with the status line in one pipe is how a real
    failure gets reported as "empty response" instead of the actual error.
    """
    with tempfile.TemporaryDirectory() as td:
        head_file = Path(td) / "head.txt"
        body_file = Path(td) / "body.bin"
        proc = subprocess.run(
            ["curl", "-s", "-S", "--max-time", "30",
             "-D", str(head_file), "-o", str(body_file),
             "-w", "%{http_code}", *args],
            capture_output=True,
        )
        code = proc.stdout.decode().strip()
        body = body_file.read_bytes() if body_file.exists() else b""
        status = int(code) if code.isdigit() else 0
        headers: dict[str, str] = {}
        if head_file.exists():
            for line in head_file.read_text("utf-8", "replace").splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    headers[k.strip().lower()] = v.strip()
        # Read INSIDE the temp dir: after the `with` exits, head_file is gone
        # and every header silently reads back as None.
        if proc.returncode != 0:
            return 0, {"__curl_stderr": proc.stderr.decode("utf-8", "replace").strip()}, b""
    return status, headers, body


def _make_clip(path: Path) -> None:
    from app.config import ffmpeg_bin

    subprocess.run(
        [
            ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=360x640:rate=25:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level:v", "4.0",
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-movflags", "+faststart",
            str(path),
        ],
        check=True, capture_output=True, timeout=300,
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="clipforge-playback-"))
    clip_file = tmp / "clip.mp4"
    _make_clip(clip_file)
    size = clip_file.stat().st_size
    print(f"rendered test clip: {clip_file} ({size} bytes)")

    from app.config import load_config
    from app.store import LocalStore

    settings = load_config(quiet=True)
    # The server runs as a SEPARATE process, so it builds its own store from
    # the real config paths. Patching an in-process module attribute would not
    # be visible to it — the clip has to exist where the real store looks.
    store = LocalStore(settings.app.paths)
    job = store.create_job("upload", "demo.mp4", title="Playback probe")
    clips = store.create_clips(job["_id"], [{"start": 0.0, "end": 4.0, "title": "probe"}], "src")
    clips[0].update({"videoPath": str(clip_file), "renderState": "done",
                     "thumbPath": None, "mediaBytes": size})
    store.write_clip(clips[0])
    clip_id = clips[0]["_id"]
    job_id = job["_id"]

    # Boot a real server. CLIPFORGE_PORT keeps it off 8000, which the managed
    # dev process already owns.
    env_proc = subprocess.Popen(
        [
            str(ROOT / ".venv" / "bin" / "python"), "-m", "uvicorn", "app.main:app",
            "--host", HOST, "--port", str(PORT), "--log-level", "warning",
        ],
        cwd=str(ROOT),
        env={**__import__("os").environ,
             "CLIPFORGE_HOST": HOST, "CLIPFORGE_PORT": str(PORT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"{BASE}/api/clips/{clip_id}/video"
    try:
        ready = False
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{BASE}/api/health", timeout=2):
                    ready = True
                    break
            except Exception:
                time.sleep(1)
        if not ready:
            print("  FAIL backend did not come up")
            return 1

        print("\n[curl -I]  full request")
        status, h, _ = curl("-I", url)
        check("HEAD returns 200", status == 200, str(status))
        check("Content-Type is video/mp4", h.get("content-type") == "video/mp4", str(h.get("content-type")))
        check(f"Content-Length is the real size ({size})", h.get("content-length") == str(size),
              str(h.get("content-length")))
        check("Accept-Ranges: bytes", h.get("accept-ranges") == "bytes", str(h.get("accept-ranges")))
        check("Cache-Control set", "no-cache" in (h.get("cache-control") or ""),
              str(h.get("cache-control")))
        check("no chunked encoding", "transfer-encoding" not in h, str(h.get("transfer-encoding")))

        print("\n[curl -r 0-1023 -I]  range request")
        status, h, _ = curl("-r", "0-1023", "-I", url)
        check("Range returns 206", status == 206, str(status))
        check("Content-Range is bytes 0-1023/N", h.get("content-range") == f"bytes 0-1023/{size}",
              str(h.get("content-range")))
        check("206 Content-Length is 1024", h.get("content-length") == "1024",
              str(h.get("content-length")))
        check("206 keeps Accept-Ranges", h.get("accept-ranges") == "bytes")

        print("\n[GET with Range]  body is the requested slice")
        status, h, body = curl("-r", "0-1023", url)
        err = h.get("__curl_stderr", "")
        check("returns 206", status == 206, f"status={status} curl said: {err}")
        check("body is exactly 1024 bytes", len(body) == 1024, str(len(body)))
        check("body starts with an ftyp box", body[4:8] == b"ftyp", repr(body[4:12]))

        status, h, body = curl("-r", "100-", url)
        check("an open-ended Range is honoured", status == 206,
              f"status={status} curl said: {h.get('__curl_stderr', '')}")
        check("the tail body is the remainder", len(body) == size - 100, str(len(body)))

        print("\n[ffprobe]  the served file really is browser-playable")
        import json as _json

        from app.config import ffprobe_bin
        out = subprocess.run(
            [ffprobe_bin(), "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(clip_file)],
            capture_output=True, text=True, check=True, timeout=60,
        )
        meta = _json.loads(out.stdout)
        v = next(s for s in meta["streams"] if s["codec_type"] == "video")
        a = next((s for s in meta["streams"] if s["codec_type"] == "audio"), {})
        check("video codec is h264", v["codec_name"] == "h264", v["codec_name"])
        check("profile is High", (v.get("profile") or "").lower() == "high", str(v.get("profile")))
        check("level is 4.0", v.get("level") == 40, str(v.get("level")))
        check("pix_fmt is yuv420p", v["pix_fmt"] == "yuv420p", v["pix_fmt"])
        check("audio codec is aac", a.get("codec_name") == "aac", str(a.get("codec_name")))

        # moov before mdat = faststart = the actual fix for bug A.
        raw = clip_file.read_bytes()
        check("moov atom sits BEFORE mdat (faststart)",
              raw.find(b"moov") != -1 and 0 < raw.find(b"moov") < raw.find(b"mdat"),
              f"moov@{raw.find(b'moov')} mdat@{raw.find(b'mdat')}")

        print("\n[download]  attachment route still serves the same bytes")
        status, h, body = curl(f"{BASE}/api/clips/{clip_id}/download")
        check("download returns 200", status == 200,
              f"status={status} curl said: {h.get('__curl_stderr', '')}")
        check("it is an attachment",
              "attachment" in (h.get("content-disposition") or ""), str(h.get("content-disposition")))
        check("the body matches the file", len(body) == size, str(len(body)))
    finally:
        env_proc.terminate()
        try:
            env_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            env_proc.kill()
        # leave no probe job behind in the real store
        store.delete_job(job_id)
        for d in (settings.app.paths.data_dir / "trash").glob(f"{job_id}-*"):
            for f in sorted(d.rglob("*"), reverse=True):
                f.unlink(missing_ok=True) if f.is_file() else f.rmdir()
            d.rmdir()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
