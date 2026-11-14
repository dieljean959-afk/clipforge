#!/usr/bin/env python3
"""Public share links: /s/<id>.mp4 and /s/<id>.

The requirement is a link that survives everything else — the batch it came
from, a host restart, a cleanup sweep, and the browser that first opened the
dashboard — so that is what these tests attack. A share link is the one thing
in this app a user can send to someone who has no account here.

    python tools/share_public_test.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
FAILURES: list[str] = []
VERBOSE = False


def check(name: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
        if VERBOSE:
            print(f"  ok   {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{title}")


# --------------------------------------------------------------------------- #


def test_share_links() -> None:
    section("Share links: minted, served, and public")

    import warnings

    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app import share_video, sharelink
    from app.config import Paths, load_config
    from app.store import LocalStore

    settings = load_config(quiet=True)
    base = Path(tempfile.mkdtemp(prefix="clipforge-share-"))
    paths = Paths(
        data_dir=base / "data",
        outputs_dir=base / "outputs",
        downloads_dir=base / "downloads",
        static_dir=base / "static",
    )
    for d in (paths.data_dir, paths.outputs_dir, paths.downloads_dir):
        d.mkdir(parents=True, exist_ok=True)
    (paths.data_dir / "clips").mkdir(parents=True, exist_ok=True)
    settings.app.paths = paths

    store = LocalStore(paths)
    # A REAL mp4 on disk: the share copy is produced by ffmpeg, and a fake file
    # would make every assertion below meaningless.
    master = paths.outputs_dir / "jobshare" / "clip_01.mp4"
    master.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=360x640:rate=24:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(master)],
        check=True, capture_output=True, timeout=120,
    )
    check("a real master mp4 was produced for the test", master.exists() and
          master.stat().st_size > 0, f"{master}")

    job = store.create_job(source_type="url", source="https://example.com/v")
    job_id = job["_id"]
    store.write_job({**job, "stage": "complete", "progress": 100})
    clips = store.create_clips(job_id, [{"start": 0.0, "end": 2.0, "score": 80,
                                          "title": "T", "hook_text": "H"}],
                               detected_source="test")
    clip = clips[0]
    # A real poster, because every rendered clip has one and the OG page reads
    # it: testing og:image against a clip with no thumbnail proves nothing.
    thumb = master.parent / "clip_01.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(master),
         "-frames:v", "1", "-vf", "scale=320:-2", str(thumb)],
        check=True, capture_output=True, timeout=60,
    )
    store.write_clip({
        **clip,
        "videoPath": str(master),
        "thumbPath": str(thumb),
        "renderState": "done",
        "mediaWidth": 1080,
        "mediaHeight": 1920,
    })

    # ---- 1. every clip has a short, permanent, random id ----------------
    sid = clip.get("shareId")
    check("a new clip is minted with a share id at creation",
          sharelink.valid_id(sid), str(sid))
    check("the share id is short", isinstance(sid, str) and len(sid) == 10, str(sid))
    check("the share id has no ambiguous characters (0/O/1/I)",
          not set(sid) & set("01OILB8"), str(sid))
    before = clip.get("shareId")
    again = store.create_clips(job_id, [{"start": 0.0, "end": 2.0, "score": 80,
                                         "title": "T", "hook_text": "H"}],
                               detected_source="test")
    check("the id survives a re-detection (same clip, same link)",
          again[0].get("shareId") == before, f"{again[0].get('shareId')} vs {before}")
    ids = {sharelink.new_id() for _ in range(200)}
    check("ids are random, not sequential", len(ids) > 190, f"{len(ids)}/200 unique")
    check("a malformed id is rejected", not sharelink.valid_id("../../etc/passwd")
          and not sharelink.valid_id("short") and not sharelink.valid_id(None))

    # ---- 2. the state is a real probe, not a promise ---------------------
    check("no share copy yet is 'preparing', never 'ready'",
          sharelink.share_state(clip, None)["shareState"] == "preparing")
    zero = base / "zero.mp4"
    zero.write_bytes(b"")
    check("a zero-byte share copy is FAILED, not ready",
          sharelink.share_state(clip, zero)["shareState"] == "failed")
    junk = base / "junk.mp4"
    junk.write_bytes(b"not an mp4 at all")
    check("a file that is not an MP4 is FAILED, not ready",
          sharelink.share_state(clip, junk)["shareState"] == "failed")
    good = share_video.build_share_version(master)
    st = sharelink.share_state(clip, good)
    check("a real share copy is READY", st["shareState"] == "ready", st["shareDetail"])
    check("ready implies there is an id AND bytes",
          st["shareReady"] and st["shareId"] == sid and st["shareUrl"].endswith(".mp4"))

    # ---- 3. the public endpoints -----------------------------------------
    main_mod.settings = settings
    main_mod.store = store
    client = TestClient(main_mod.app)

    r = client.get(f"/s/{sid}.mp4")
    check("GET /s/<id>.mp4 is 200", r.status_code == 200, f"{r.status_code}")
    check("it is video/mp4", r.headers.get("content-type") == "video/mp4",
          r.headers.get("content-type", ""))
    check("it advertises Accept-Ranges", r.headers.get("accept-ranges") == "bytes")
    check("it is CORS-open for embeds",
          r.headers.get("access-control-allow-origin") == "*")
    check("it exposes the range headers a player needs",
          "Content-Range" in (r.headers.get("access-control-expose-headers") or ""),
          r.headers.get("access-control-expose-headers", ""))
    check("the bytes are the real video", len(r.content) > 1000, f"{len(r.content)}B")
    check("the bytes really are an mp4", b"ftyp" in r.content[:32])

    part = client.get(f"/s/{sid}.mp4", headers={"Range": "bytes=0-1023"})
    check("a Range request answers 206", part.status_code == 206, f"{part.status_code}")
    check("206 comes back with Content-Range",
          part.headers.get("content-range", "").startswith("bytes 0-1023/"),
          part.headers.get("content-range", ""))
    check("206 is exactly the requested range", len(part.content) == 1024,
          f"{len(part.content)}B")

    head = client.head(f"/s/{sid}.mp4")
    check("HEAD works (that is what the readiness check uses)",
          head.status_code == 200 and head.headers.get("content-length", "0") != "0",
          f"{head.status_code} len={head.headers.get('content-length')}")

    check("an unknown share id is a 404, not a 500",
          client.get("/s/ZZZZZZZZZZ.mp4").status_code == 404)
    check("a junk path is a 404 too",
          client.get("/s/..%2F..%2Fetc%2Fpasswd.mp4").status_code == 404)

    # ---- 4. the OG page ---------------------------------------------------
    page = client.get(f"/s/{sid}")
    check("GET /s/<id> is 200", page.status_code == 200, f"{page.status_code}")
    html = page.text
    check("it is HTML", page.headers.get("content-type", "").startswith("text/html"))
    check("it carries og:title", 'property="og:title"' in html)
    check("it carries og:video", 'property="og:video"' in html)
    check("og:video points at the mp4", f"/s/{sid}.mp4" in html)
    check("og:video:type is mp4", 'og:video:type" content="video/mp4"' in html)
    check("it carries og:video:width/height (1080x1920)",
          'og:video:width" content="1080"' in html
          and 'og:video:height" content="1920"' in html)
    check("it carries og:image", 'property="og:image"' in html)
    check("it carries twitter:card=player",
          'twitter:card" content="player"' in html)
    check("it carries a twitter:player tag", 'name="twitter:player"' in html)

    # ---- 5. the state travels with the batch list -------------------------
    listed = client.get(f"/api/clips?jobId={job_id}").json()
    check("the batch list carries the share state", bool(listed)
          and listed[0].get("shareId") == sid, str(listed[0].get("shareId")) if listed else "none")
    check("the batch list reports ready from the probe",
          listed[0].get("shareReady") is True, str(listed[0].get("shareDetail")))
    check("the batch list carries the page url for X / Share...",
          listed[0].get("sharePageUrl") == f"/s/{sid}")

    # ---- 6. it survives what would kill anything else ---------------------
    from app.hosting import RetentionCleaner

    cleaner = RetentionCleaner(settings, store)
    cleaner.sweep()
    cleaner.sweep_orphans()
    check("cleanup does not delete the rendered clip", master.exists())
    check("cleanup does not delete the share copy", good.exists())
    check("the link still serves after a cleanup sweep",
          client.get(f"/s/{sid}.mp4").status_code == 200)

    # "restart": a brand new store reading the same files, which is what a
    # fresh process sees. Nothing is cached in memory that the link needs.
    reborn = LocalStore(paths)
    main_mod.store = reborn
    client2 = TestClient(main_mod.app)
    check("the link survives a restart (new store, same files)",
          client2.get(f"/s/{sid}.mp4").status_code == 200)
    check("and the id is still the same one",
          client2.get(f"/s/{sid}").status_code == 200)

    # ---- 7. backfill ------------------------------------------------------
    old = reborn.create_clips("oldjob", [{"start": 0.0, "end": 1.0, "score": 70,
                                          "title": "t", "hook_text": "h"}],
                              detected_source="test")
    legacy = dict(old[0])
    legacy.pop("shareId", None)
    legacy.pop("shareState", None)
    (paths.data_dir / "clips" / f"{legacy['_id']}.json").write_text(
        __import__("json").dumps(legacy), encoding="utf-8"
    )
    stats = sharelink.backfill(reborn)
    check("backfill mints ids for pre-existing clips",
          stats["minted"] >= 1, str(stats))
    after = reborn.get_clip(legacy["_id"])
    check("the backfilled clip has a valid id",
          sharelink.valid_id(after.get("shareId")), str(after.get("shareId")))
    check("backfill is idempotent (a second pass mints nothing)",
          sharelink.backfill(reborn)["minted"] == 0)

    # ---- 8. the source the UI reads ---------------------------------------
    api_ts = (ROOT / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    check("the client can build the public mp4 url", "/s/${" in api_ts
          and ".mp4" in api_ts)
    check("the client can build the public page url", "/s/${" in api_ts)
    share_btn = (ROOT / "src" / "components" / "clipforge"
                 / "ShareButton.tsx").read_text(encoding="utf-8")
    check("the Share button renders a Ready state from the server",
          "shareReady" in share_btn or "Ready" in share_btn)
    check("the Share button can render a Preparing state",
          "Preparing" in share_btn or "preparing" in share_btn)
    check("copying a link confirms it", "Copied" in share_btn)

    main_src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    check("the public mp4 route sets CORS headers",
          "/s/{share_id}.mp4" in main_src
          and "Access-Control-Allow-Origin" in main_src)
    check("the orphan sweep never targets /s", "shareId" not in
          (ROOT / "app" / "hosting.py").read_text(encoding="utf-8"))

    shutil.rmtree(base, ignore_errors=True)


def main() -> int:
    global VERBOSE
    VERBOSE = "--verbose" in sys.argv
    print("ClipForge public share-link suite")
    print(f"  repo: {ROOT}")
    test_share_links()
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("failed:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
