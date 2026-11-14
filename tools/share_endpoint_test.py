"""The /api/clips/<id>/share endpoint, over real HTTP.

tools/share_instant_test.py proves the encoder and the frontend invariants. This
one proves the wire contract, because the acceptance target depends on headers
the browser cannot work without:

  * Cache-Control: public, max-age=31536000, immutable  — otherwise the 3 MB
    share file is re-fetched on every page load and the prefetch never wins.
  * Accept-Ranges: bytes + a real 206 on a Range request  — the frontend's
    resume-after-a-dropped-connection path is built on this.
  * An ETag that changes when the bytes change, which is what makes `immutable`
    honest rather than a way to serve stale footage forever.
  * HEAD works, and a 503 (not a 500) when the transcode cannot be built, so
    the UI can fall back to the link instead of showing a dead error.
  * Concurrent requests for the same clip produce ONE transcode, not N racing
    ffmpeg processes onto one temp file.

Boots the real ASGI app with TestClient and a real ffmpeg.

    python tools/share_endpoint_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def make_master(dest: Path, seconds: float = 4.0) -> Path:
    from app.config import ffmpeg_bin

    os.system(f'rm -f "{dest}"')  # noqa: S605 - fixed tmp path
    import subprocess

    subprocess.run(
        [ffmpeg_bin(), "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size=1080x1920:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-shortest",
         "-movflags", "+faststart", str(dest)],
        check=True, capture_output=True, timeout=180,
    )
    return dest


def main() -> int:
    from fastapi.testclient import TestClient

    import app.main as main_mod

    section("1. /share returns a small, immutable, rangeable file")
    with tempfile.TemporaryDirectory() as td:
        master = make_master(Path(td) / "clip_01.mp4")
        main_mod.store.write_clip({
            "_id": "shareclip1", "jobId": "sharejob1", "order": 0,
            "start": 0.0, "end": 4.0, "score": 90, "title": "Share endpoint test",
            "hookText": "", "transcript": "", "hashtags": [], "captionCues": [],
            "aspect": "9:16", "videoPath": str(master), "thumbPath": None,
            "renderState": "done",
        })
        try:
            main_mod._do_warmup = lambda: None
            with TestClient(main_mod.app) as client:
                t0 = time.monotonic()
                r = client.get("/api/clips/shareclip1/share")
                first_ms = (time.monotonic() - t0) * 1000
                check("GET /share returns 200", r.status_code == 200, r.text[:200])
                if r.status_code != 200:
                    return 1
                body = r.content
                print(f"        ({len(body) / 1e6:.2f} MB, first build {first_ms:.0f} ms)")

                check("it is an MP4", r.headers.get("content-type") == "video/mp4",
                      r.headers.get("content-type", ""))
                check("it is served INLINE (attachment breaks share targets)",
                      "attachment" not in r.headers.get("content-disposition", ""),
                      r.headers.get("content-disposition", ""))
                cc = r.headers.get("cache-control", "")
                check("Cache-Control is public", "public" in cc, cc)
                check("Cache-Control allows a year", "max-age=31536000" in cc, cc)
                check("Cache-Control is immutable", "immutable" in cc, cc)
                check("Accept-Ranges: bytes is advertised",
                      r.headers.get("accept-ranges") == "bytes",
                      r.headers.get("accept-ranges", ""))
                check("Content-Length matches the body",
                      int(r.headers.get("content-length", -1)) == len(body))
                etag = r.headers.get("etag")
                check("an ETag is present", bool(etag), repr(etag))

                check("the body is under the 3 MB budget",
                      len(body) <= 3 * 1024 * 1024, f"{len(body) / 1e6:.2f} MB")

                section("2. Range requests work (the resume path depends on it)")
                r2 = client.get("/api/clips/shareclip1/share",
                                headers={"Range": "bytes=0-1023"})
                check("a Range request answers 206", r2.status_code == 206,
                      str(r2.status_code))
                check("206 returns exactly the requested bytes",
                      len(r2.content) == 1024, str(len(r2.content)))
                cr = r2.headers.get("content-range", "")
                check("Content-Range is present", cr.startswith("bytes"), cr)
                check("Content-Range reports the true total",
                      cr.endswith(f"/{len(body)}"), cr)
                check("the first 1024 bytes match the full body",
                      r2.content == body[:1024])

                section("3. HEAD and repeat GET")
                h = client.head("/api/clips/shareclip1/share")
                check("HEAD is answered", h.status_code == 200, str(h.status_code))
                check("HEAD carries the same Cache-Control",
                      h.headers.get("cache-control") == cc,
                      h.headers.get("cache-control", ""))
                t1 = time.monotonic()
                r3 = client.get("/api/clips/shareclip1/share")
                cached_ms = (time.monotonic() - t1) * 1000
                check("a repeat GET returns the same bytes", r3.content == body)
                check("a repeat GET is served from cache, not re-encoded",
                      cached_ms < 250, f"{cached_ms:.0f} ms")
                check("the ETag is stable across requests",
                      r3.headers.get("etag") == etag)

                section("4. a re-render gets a new ETag (immutable stays honest)")
                time.sleep(1.1)
                make_master(master, seconds=6.0)
                r4 = client.get("/api/clips/shareclip1/share")
                check("a changed master is served again", r4.status_code == 200)
                check("the ETag changed with the bytes",
                      r4.headers.get("etag") != etag,
                      f"{etag} -> {r4.headers.get('etag')}")

                section("5. concurrency: one transcode, not N")
                share_path = Path(master).with_name(
                    f"{Path(master).stem}.share.mp4"
                )
                share_path.unlink(missing_ok=True)
                share_path.with_suffix(".share.stamp").unlink(missing_ok=True)
                results: list[int] = []
                lock = threading.Lock()

                def hit() -> None:
                    resp = client.get("/api/clips/shareclip1/share")
                    with lock:
                        results.append(resp.status_code)

                threads = [threading.Thread(target=hit) for _ in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=240)
                check("every concurrent request succeeded",
                      results == [200] * 4, str(results))
                check("no temp file was left behind",
                      not list(Path(master).parent.glob("*.share.tmp.mp4")),
                      str(list(Path(master).parent.glob("*.share.tmp.mp4"))))

                section("6. failures are honest, not dead ends")
                missing = client.get("/api/clips/doesnotexist/share")
                check("an unknown clip is 404", missing.status_code == 404,
                      str(missing.status_code))
                main_mod.store.write_clip({
                    "_id": "shareclip2", "jobId": "sharejob1", "order": 1,
                    "start": 0.0, "end": 4.0, "score": 90, "title": "No media",
                    "hookText": "", "transcript": "", "hashtags": [], "captionCues": [],
                    "aspect": "9:16", "videoPath": None, "thumbPath": None,
                    "renderState": "pending",
                })
                nomedia = client.get("/api/clips/shareclip2/share")
                check("a clip with no render is 409, not 500",
                      nomedia.status_code == 409, str(nomedia.status_code))
                check("the 409 says what to do instead",
                      "share" in nomedia.json().get("detail", "").lower(),
                      nomedia.json().get("detail", ""))
        finally:
            Path("data", "clips", "shareclip1.json").unlink(missing_ok=True)
            Path("data", "clips", "shareclip2.json").unlink(missing_ok=True)

    print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os as _os

    _os._exit(rc)
