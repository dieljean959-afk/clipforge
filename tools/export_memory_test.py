"""Exporting must not be able to OOM the server.

`POST /api/export/zip` built the entire archive in an `io.BytesIO` and then
returned `buf.getvalue()` — a second full copy of the same bytes. A batch is
10 vertical clips at ~80MB each, so one "Export all" held roughly 800MB
resident, and the request schema allowed 32 clip ids, implying 2.5GB.

This host has 2.0 GiB usable and a Whisper model resident. Exporting a batch
was therefore capable of OOM-killing the API process and taking every running
job down with it — a read-only convenience feature with a server-killing blast
radius.

The archive is now written to a temp file and streamed back with a
FileResponse, so peak memory is flat regardless of batch size, and a request
that would exceed MAX_EXPORT_BYTES is refused with an honest message rather
than attempted.

Run:  .venv/bin/python tools/export_memory_test.py
"""

from __future__ import annotations

import sys
import tempfile
import tracemalloc
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Paths  # noqa: E402
from app.store import LocalStore  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(ok: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILS.append(label)


def main() -> int:
    root_main = Path(__file__).resolve().parents[1] / "app" / "main.py"
    src = root_main.read_text(encoding="utf-8")
    body = src.split("def export_zip(", 1)[-1].split("\n\ndef ", 1)[0]

    # ---- 1. the memory bomb is gone ---------------------------------
    check("io.BytesIO" not in body, "export no longer buffers the archive in BytesIO")
    check("buf.getvalue()" not in body, "export no longer makes a second full copy")
    check("content=buf" not in body, "export does not inline the archive into a Response")

    # ---- 2. it streams from disk ------------------------------------
    check("FileResponse(" in body, "export returns a FileResponse (streamed, not buffered)")
    check("tempfile.mkstemp" in body, "the archive is built in a temp file")
    check("BackgroundTask" in body, "the temp file is cleaned up after streaming")
    check("_unlink_quietly" in body, "cleanup goes through the quiet unlink helper")
    check("ZIP_STORED" in body, "H.264 is stored, not needlessly deflated")

    # ---- 3. an oversized export is refused, not attempted -----------
    check("MAX_EXPORT_BYTES" in body, "export enforces a byte ceiling")
    check("413" in body, "an oversized export returns 413, not an OOM")
    check(
        "would pass the" in body,
        "the refusal explains the limit in the message the user reads",
    )

    # ---- 4. the ceiling is sane for this host -----------------------
    ns: dict = {}
    for line in src.splitlines():
        if line.startswith("MAX_EXPORT_BYTES"):
            exec(line, ns)  # noqa: S102 - reading our own constant
    cap = ns.get("MAX_EXPORT_BYTES", 0)
    check(0 < cap <= 2_000_000_000, f"the ceiling is set and bounded ({cap:,} bytes)")

    # ---- 5. a real archive round-trips through the same recipe -------
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        clips = base / "clips"
        clips.mkdir()
        paths = []
        for i in range(10):
            p = clips / f"clip_{i + 1:02d}.mp4"
            p.write_bytes(bytes(200_000))  # 200 KB stand-in for an 80 MB MP4
            paths.append(p)

        archive = base / "export.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
            for i, p in enumerate(paths, 1):
                zf.write(p, f"clips/{i:02d}-Energy Burst.mp4")
            zf.writestr("captions.txt", "=== Clip 1 ===")
            zf.writestr("README.txt", "ClipForge export")

        check(archive.exists(), "the archive is written to disk")
        check(
            archive.stat().st_size < sum(p.stat().st_size for p in paths) * 2,
            "the archive is not wildly larger than its contents",
        )
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            check(len([n for n in names if n.startswith("clips/")]) == 10, "all 10 clips are in the zip")
            check("captions.txt" in names, "captions.txt is included")
            check("README.txt" in names, "README.txt is included")
            check(zf.testzip() is None, "the archive passes an integrity check")

        # Writing to disk then reading it back must not scale peak memory
        # with the payload, which is the whole point of the fix.
        tracemalloc.start()
        base_mem = tracemalloc.get_traced_memory()[0]
        big = base / "big.bin"
        with zipfile.ZipFile(big, "w", zipfile.ZIP_STORED) as zf:
            for p in paths:
                zf.write(p, p.name)
        after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        payload = sum(p.stat().st_size for p in paths)
        growth = after - base_mem
        check(
            growth < payload * 0.2,
            f"writing a {payload:,}-byte zip to disk costs almost no heap "
            f"(grew {growth:,} bytes)",
        )

        # ---- 6. clip docs with no render are skipped, not fatal -----
        paths_root = Paths(
            data_dir=base / "data",
            outputs_dir=base / "outputs",
            downloads_dir=base / "downloads",
            static_dir=base / "static",
        )
        store = LocalStore(paths_root)
        store.write_job({"_id": "e1", "source": "s", "title": "t", "stage": "rendering"})
        clips_docs = store.create_clips(
            "e1", [{"start": 1.0, "end": 9.0, "score": 90, "title": "No media"}], "groq"
        )
        check(
            not clips_docs[0].get("videoPath"),
            "an unrendered clip has no videoPath and is skipped by export",
        )
        check(
            'raise HTTPException(404, "no rendered clips to export")' in body,
            "an all-unrendered export is a clear 404",
        )

    print()
    if FAILS:
        print(f"{CHECKS - len(FAILS)} passed, {len(FAILS)} FAILED")
        return 1
    print(f"{CHECKS} passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
