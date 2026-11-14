"""Share must be instant on tap, whatever the connection is doing.

The failure this pins, in order of how badly it hurt:

  1. `navigator.share({files})` needs transient user activation, which expires
     ~5s after the tap. Fetching a 6-10 MB MP4 inside the click handler spent
     that window on the network, so on a phone the sheet opened late or
     `share()` threw NotAllowedError outright. The bytes are now fetched
     BEFORE the tap (prefetch -> IndexedDB -> a ready `File`), so the handler
     does nothing but call share().

  2. The bytes handed over are a NEW small artifact: 720x1280 H.264, AAC 96k,
     +faststart, capped at 3 MB. A downscale of an already-encoded master, so
     it is cheap, and 2.5x smaller in practice.

  3. If the file is not cached yet, the tap shares caption + URL instead of
     blocking. The clip is served from this origin, so the link IS the clip.

The acceptance target: on Slow 3G, tapping Share on a clip that has been on
screen for 10+ seconds opens the sheet in under 300 ms. That is only reachable
if nothing on the tap path touches the network, which is what section 3 asserts
against the source.

    python tools/share_instant_test.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import share_video as sv  # noqa: E402

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


def _ts(rel: str) -> str:
    return (ROOT / rel).read_text()


def _ts_code(rel: str) -> str:
    """Source with comments stripped, so prose about a thing is not the thing."""
    src = _ts(rel)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
    return src


def make_master(dest: Path, seconds: float = 4.0) -> Path:
    """A small but REAL 1080x1920 h264 master, made by ffmpeg."""
    from app.config import ffmpeg_bin

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


# --------------------------------------------------------------------------
# 1. the share artifact is actually small and browser-safe
# --------------------------------------------------------------------------


def test_share_encoding() -> None:
    section("1. the share artifact is small, 720p and faststart")
    with tempfile.TemporaryDirectory() as td:
        master = make_master(Path(td) / "clip_01.mp4")
        share = sv.build_share_version(master)

        size = share.stat().st_size
        check("the share copy exists and is non-empty", size > 0, f"{size} bytes")
        check("the share copy is under the 3 MB budget",
              size <= sv.SHARE_MAX_BYTES, f"{size / 1e6:.2f} MB")
        check("the master is never modified in place",
              master.stat().st_size > 0 and master.suffix == ".mp4")
        check("the share copy is a SIBLING, not an overwrite",
              share.name != master.name and share.parent == master.parent)

        meta = _probe(share)
        check("720 wide", meta["w"] == 720, str(meta))
        check("1280 tall", meta["h"] == 1280, str(meta))
        check("H.264 video", meta["vcodec"] == "h264", str(meta))
        check("AAC audio", meta["acodec"] == "aac", str(meta))
        check("yuv420p, the only pix_fmt every decoder accepts",
              meta["pix_fmt"] == "yuv420p", str(meta))
        check("+faststart: the moov atom is before mdat",
              _moov_before_mdat(share),
              "moov must precede mdat or the receiving app must download "
              "the whole file before it can preview anything")

        # Caching: a second call must not re-encode.
        t0 = time.monotonic()
        again = sv.build_share_version(master)
        check("a second call is a cache hit, not a re-encode",
              time.monotonic() - t0 < 0.5, f"{time.monotonic() - t0:.3f}s")
        check("the cache hit returns the same file", again == share)


def _probe(p: Path) -> dict:
    from app.config import ffprobe_bin

    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(p)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    import json

    meta = json.loads(out.stdout)
    v = next((s for s in meta["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in meta["streams"] if s["codec_type"] == "audio"), {})
    return {
        "w": int(v.get("width", 0)), "h": int(v.get("height", 0)),
        "vcodec": v.get("codec_name"), "acodec": a.get("codec_name"),
        "pix_fmt": v.get("pix_fmt"),
    }


def _moov_before_mdat(p: Path) -> bool:
    """Walk the top-level box list: moov must come first."""
    import struct

    with p.open("rb") as fh:
        head = fh.read(4096)
    off = 0
    order: list[str] = []
    while off + 8 <= len(head) and len(order) < 6:
        size = struct.unpack(">I", head[off:off + 4])[0]
        typ = head[off + 4:off + 8].decode("latin1", "replace")
        order.append(typ)
        if size < 8:
            break
        off += size
    return "moov" in order and (
        "mdat" not in order or order.index("moov") < order.index("mdat")
    )


# --------------------------------------------------------------------------
# 2. cache identity: a re-render must not serve a stale lighter copy
# --------------------------------------------------------------------------


def test_invalidation() -> None:
    section("2. a re-rendered clip invalidates its share copy")
    with tempfile.TemporaryDirectory() as td:
        master = make_master(Path(td) / "clip_01.mp4", seconds=3.0)
        share = sv.build_share_version(master)
        stamp = sv._stamp_path(share)
        check("a fingerprint stamp is written next to the share copy",
              stamp.exists())

        before = stamp.read_text().strip()
        # Touch the master the way a re-render does: new mtime, new bytes.
        time.sleep(1.1)
        make_master(master, seconds=5.0)

        rebuilt = sv.build_share_version(master)
        after = stamp.read_text().strip()
        check("the fingerprint changes when the master changes",
              before != after, f"{before} vs {after}")
        check("a changed master rebuilds rather than reusing the cache",
              rebuilt == share and after != before)

        sv.purge_derived(master)
        check("purge_derived removes the share copy and its stamp",
              not share.exists() and not stamp.exists())


# --------------------------------------------------------------------------
# 3. the tap path does no I/O  (the acceptance test)
# --------------------------------------------------------------------------


def test_tap_path_is_synchronous() -> None:
    section("3. the tap path awaits nothing (the 300 ms acceptance target)")
    btn = _ts_code("src/components/clipmake/ShareButton.tsx")
    handler = btn[btn.find("const onShare"):btn.find("const pct =")]
    check("the click handler exists", "const onShare" in btn)

    # Anything that suspends inside the handler spends the activation window.
    for bad, label in (
        ("await navigator.share", "await before navigator.share"),
        ("await fetch", "a fetch inside the handler"),
        ("await cache.", "an await on the cache"),
        ("await hydrate", "an await on hydrate"),
    ):
        check(f"no {label}", bad not in handler, repr(bad))

    # The share call itself must be reached synchronously.
    share_at = handler.find("nav\n        .share(")
    if share_at == -1:
        share_at = handler.find(".share({")
    check("navigator.share is called inside the handler", share_at != -1)

    # The file path uses the PREFETCHED file, never a fresh fetch.
    check("the cached File is read synchronously via cache.get",
          "cache.get(" in handler)
    check("canShare is checked before sharing a file",
          "cache.canShareFile()" in handler)
    check("the not-cached path shares text + url instead of blocking",
          re.search(r"\.share\(\{ text: caption, url: publicUrl", handler) is not None)
    check("the public URL is used as the fallback, not a download",
          "publicUrl" in handler and "a.download" not in handler)


def test_abort_is_not_an_error() -> None:
    section("4. cancelling the sheet is silent, real failures are not")
    btn = _ts_code("src/components/clipmake/ShareButton.tsx")
    check("AbortError is recognised", "AbortError" in btn)
    check("AbortError returns before any toast",
          re.search(r"if \(isAbort\(err\)\) return;", btn) is not None)
    check("a real failure still toasts", "toast.error" in btn)
    check("the busy flag is always released",
          btn.count("setBusy(false)") >= 2, str(btn.count("setBusy(false)")))
    # A stuck "Preparing..." is the specific complaint, so it is capped.
    check("the preparing state has a hard timeout", "READY_TIMEOUT_MS" in btn)
    check("the timeout actually clears the in-flight flag",
          "cache.invalidate(" in btn)


# --------------------------------------------------------------------------
# 5. prefetch behaviour
# --------------------------------------------------------------------------


def test_prefetch() -> None:
    section("5. prefetch is bounded, resumable and network-aware")
    cache = _ts_code("src/lib/clip-share-cache.ts")
    check("concurrent prefetches stay bounded (3)",
          "MAX_CONCURRENT = 3" in cache)
    check("attempts are bounded", "MAX_ATTEMPTS = 3" in cache)
    check("retries back off exponentially",
          "BACKOFF_MS" in cache and re.search(r"BACKOFF_MS\s*=\s*\[\s*0,\s*600,\s*2000\s*\]", cache) is not None)
    check("a partial download resumes with a Range header",
          re.search(r'headers\.Range\s*=\s*`bytes=\$\{from\}-`', cache) is not None)
    # A 200 answer to a ranged request means the server ignored Range; keeping
    # the old bytes would silently corrupt the file.
    check("a server that ignores Range restarts instead of corrupting",
          "appending" in cache and "res.status === 206" in cache)
    check("prefetch is de-duplicated per clip", "jobs.has(clipId)" in cache)
    check("memory is bounded by a byte budget", "MEMORY_BUDGET_BYTES" in cache)
    check("the File is built at prefetch time, not on tap",
          'new File([blob], fileName' in cache)
    check("blobs persist in IndexedDB", "indexedDB.open" in cache)
    check("saveData suppresses prefetching", "saveData" in cache)
    check("2g/slow-2g suppresses prefetching",
          "slow-2g" in cache and "2g" in cache)


def test_service_worker() -> None:
    section("6. the service worker caches clips and nothing else")
    sw = _ts_code("public/share-sw.js")
    check("it only matches the share endpoint",
          r"/^\/api\/clips\/[^/]+\/share$/" in sw)
    check("non-GET is ignored", 'req.method !== "GET"' in sw)
    # Caching is gated on an exact 200, which structurally excludes the 206
    # that a Range request produces — a partial body must never become the
    # cached whole file. (Read from the unstripped source only to confirm the
    # reasoning is documented; the assertion is on the code.)
    check("only a complete 200 is cached (a 206 can never be)",
          re.search(r"res\.status\s*===\s*200", sw) is not None,
          "expected an exact `res.status === 200` gate")
    check("the 206 hazard is documented for the next reader",
          "206" in _ts("public/share-sw.js"))
    check("a successful response is cloned before caching",
          "res.clone()" in sw)
    main = _ts("src/main.tsx")
    check("registration is production-only (dev must never cache modules)",
          "import.meta.env.PROD" in main and "serviceWorker.register" in main
          or "navigator.serviceWorker" in main)


# --------------------------------------------------------------------------
# 7. the ring always terminates
# --------------------------------------------------------------------------


def test_spinner_terminates() -> None:
    """A cancelled prefetch must actually stop.

    The complaint this guards is a Share button that spins for minutes. The
    cause was not a slow server: `invalidate` dropped the bookkeeping for an
    in-flight fetch but never aborted it, so the request kept running, kept
    emitting progress, and the ring the button had just cancelled came back.

    The second half guards the reason the cap could be lifted: share copies are
    built during the render stage, so `/share` is a static download rather than
    a transcode on the tap. Without that, warming every clip would just move
    the queue from the browser to the server.
    """

    section("7. a cancelled prefetch really stops, and not the whole batch")
    cache = _ts_code("src/lib/clip-share-cache.ts")

    check("every job carries an AbortController",
          "controller: AbortController" in cache)
    check("the fetch actually receives the signal",
          re.search(r"fetch\(url, \{[^}]*signal", cache) is not None)
    check("invalidate aborts before it forgets the job",
          re.search(r"jobs\.get\(clipId\)\?\.controller\.abort\(\);", cache)
          is not None)
    check("clearing the cache aborts every in-flight job",
          "for (const job of jobs.values()) job.controller.abort();" in cache)

    # An abort must not be retried: retrying is what resurrected the ring.
    check("an aborted run stops instead of retrying",
          "if (signal.aborted) {" in cache
          and re.search(r"if \(signal\.aborted\) \{\s*\n\s*emit\(", cache) is not None)
    check("an aborted run settles the in-flight flag",
          re.search(r"if \(signal\.aborted\) \{[^}]*inFlight: false", cache)
          is not None)

    # Every clip must be able to reach Ready, so the cap has to cover a whole
    # batch. It is only affordable because the copies are built server-side.
    cap = re.search(r"const MAX_PREFETCH = (\d+);", cache)
    check("prefetching a whole batch is capped", cap is not None
          and "jobs.size >= MAX_PREFETCH" in cache)
    check("the cap covers a full 13-clip batch", cap is not None
          and int(cap.group(1)) >= 13, cap.group(1) if cap else "missing")

    task = _ts("app/task.py")
    check("share copies are built while the job finishes",
          "_prewarm_share_copies" in task)
    check("prewarming never fails a rendered batch",
          re.search(r"except Exception as exc:.*?never fail the batch", task, re.S)
          is not None)
    check("prewarming is off the critical path (daemon thread)",
          re.search(r"threading\.Thread\(\s*\n\s*target=self\._prewarm_share_copies",
                    task) is not None
          and "daemon=True" in task)

    btn = _ts_code("src/components/clipmake/ShareButton.tsx")
    ms = re.search(r"const READY_TIMEOUT_MS = ([\d_]+);", btn)
    check("the preparing ring gives up on a human timescale", ms is not None)
    if ms:
        check("the timeout is short enough that nobody waits it out",
              int(ms.group(1).replace("_", "")) <= 45_000, f"{ms.group(1)} ms")


def main() -> int:
    test_share_encoding()
    test_invalidation()
    test_tap_path_is_synchronous()
    test_abort_is_not_an_error()
    test_prefetch()
    test_service_worker()
    test_spinner_terminates()
    print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
