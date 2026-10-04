#!/usr/bin/env python3
"""Step 2 — local storage hygiene.

Four things were broken, and every one of them made the disk fill faster than
the user could tell:

1. `retention_hours` shipped as 0, which switches the sweeper OFF entirely.
   Deleted batches piled up in `data/trash/` forever — 353 MB at the time of
   writing, more than the clips themselves. Four days is the default now.

2. `CLIPFORGE_RETENTION_HOURS` was DEAD. The line read
   `h.get("retention_hours", os.getenv(...))`, and `dict.get`'s default is only
   evaluated when the key is absent. config.toml declares the key, so the env
   var was never read and setting it changed nothing.

3. The completion cleanup deleted two named WAVs and nothing else, while a
   finished job leaves a ~101 MB workspace and a full source download behind.
   The log said "cleanup: removed 2 temp files" and reclaimed almost nothing.

4. Uploads must NOT be deleted at completion. An uploaded file is the one
   artefact that cannot be re-created — a URL can always be re-downloaded.

    python tools/storage_hygiene_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import task as task_mod  # noqa: E402
from app.config import load_config  # noqa: E402

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


# --------------------------------------------------------------------------- #


def test_retention_default() -> None:
    print("\n[1] retention defaults to four days, not never")
    old = os.environ.pop("CLIPFORGE_RETENTION_HOURS", None)
    try:
        settings = load_config(quiet=True)
        check("config.toml ships a 4-day (96h) retention",
              settings.hosted.retention_hours == 96,
              f"got {settings.hosted.retention_hours}h")
        check("retention is ON, so the sweeper is not disabled",
              settings.hosted.retention_hours > 0)
    finally:
        if old is not None:
            os.environ["CLIPFORGE_RETENTION_HOURS"] = old


def test_retention_env_wins() -> None:
    print("\n[2] CLIPFORGE_RETENTION_HOURS actually overrides config.toml")
    # This is the regression: with `h.get(key, env)` the file always won,
    # because the key is present in it.
    for value, expected in (("6", 6), ("168", 168), ("0", 0), ("12", 12)):
        os.environ["CLIPFORGE_RETENTION_HOURS"] = value
        try:
            got = load_config(quiet=True).hosted.retention_hours
            check(f"env {value!r} -> {expected}h (file says 96)",
                  got == expected, f"got {got}h")
        finally:
            os.environ.pop("CLIPFORGE_RETENTION_HOURS", None)

    # A nonsense value must not crash boot; 0 ("never") is the safe fallback.
    os.environ["CLIPFORGE_RETENTION_HOURS"] = "not-a-number"
    try:
        try:
            got = load_config(quiet=True).hosted.retention_hours
            check("a malformed env value does not crash config load", True, f"-> {got}h")
        except Exception as exc:  # noqa: BLE001
            check("a malformed env value does not crash config load", False, repr(exc))
    finally:
        os.environ.pop("CLIPFORGE_RETENTION_HOURS", None)


def test_tree_size() -> None:
    print("\n[3] _tree_size measures bytes, not the directory inode")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        payload = 3 * 1024 * 1024
        (root / "a.bin").write_bytes(b"\0" * payload)
        nested = root / "deep" / "deeper"
        nested.mkdir(parents=True)
        (nested / "b.bin").write_bytes(b"\0" * payload)
        got = task_mod._tree_size(root)
        check("a nested tree reports every byte under it",
              got == payload * 2, f"got {got}, want {payload * 2}")
        check("it is not the directory's own inode size",
              got > 1_000_000, f"got {got}")

        missing = root / "nope"
        check("a missing path measures 0, it does not raise",
              task_mod._tree_size(missing) == 0)

        single = root / "a.bin"
        check("a single file measures its own length",
              task_mod._tree_size(single) == payload)


class _StubPaths:
    """Just the four path attributes _cleanup_job_temps touches."""

    def __init__(self, base: Path) -> None:
        self.data_dir = base / "data"
        self.downloads_dir = base / "downloads"
        self.uploads_dir = base / "uploads"
        self.outputs_dir = base / "outputs"
        for d in (self.data_dir, self.downloads_dir, self.uploads_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)


class _StubApp:
    def __init__(self, paths: _StubPaths) -> None:
        self.paths = paths


class _StubSettings:
    def __init__(self, paths: _StubPaths) -> None:
        self.app = _StubApp(paths)


class _StubSelf:
    def __init__(self, settings: _StubSettings) -> None:
        self.settings = settings


def test_completion_cleanup() -> None:
    print("\n[4] completion cleanup: scratch + download go, uploads and clips stay")
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        paths = _StubPaths(base)
        job = "job-abc"

        # What a finished job leaves behind.
        workspace = paths.data_dir / "work" / job
        (workspace / "transcribe").mkdir(parents=True)
        (workspace / "transcribe" / "source.wav").write_bytes(b"\0" * 2_000_000)
        (workspace / "render" / "chunk_0.json").parent.mkdir(parents=True)
        (workspace / "render" / "chunk_0.json").write_bytes(b"{}")
        download = paths.downloads_dir / job
        download.mkdir(parents=True)
        (download / "source.mp4").write_bytes(b"\0" * 5_000_000)
        upload = paths.uploads_dir / job
        upload.mkdir(parents=True)
        (upload / "uploaded.mp4").write_bytes(b"\0" * 5_000_000)
        outputs = paths.outputs_dir / job
        outputs.mkdir(parents=True)
        (outputs / "clip_01.mp4").write_bytes(b"\0" * 3_000_000)

        worker = _StubSelf(_StubSettings(paths))
        from app.stages import StageWorkspace

        task_mod.Pipeline._cleanup_job_temps(worker, job, StageWorkspace(paths.data_dir, job))

        check("the whole scratch workspace is gone, not just two WAVs",
              not workspace.exists(), f"still there: {workspace}")
        check("the re-downloadable source download is gone",
              not download.exists(), f"still there: {download}")
        check("the UPLOADED source survives — it cannot be re-created",
              (upload / "uploaded.mp4").exists())
        check("the rendered clips survive — retention owns their lifetime",
              (outputs / "clip_01.mp4").exists())

        # Running it twice must be a no-op, not a crash: the job runner calls
        # it after _finish, and a resumed job can reach here more than once.
        try:
            task_mod.Pipeline._cleanup_job_temps(
                worker, job, StageWorkspace(paths.data_dir, job))
            check("a second cleanup pass is harmless", True)
        except Exception as exc:  # noqa: BLE001
            check("a second cleanup pass is harmless", False, repr(exc))


def test_cleanup_reports_truthfully() -> None:
    """A failed delete must not be logged as a successful one.

    `shutil.rmtree` returns None UNCONDITIONALLY, and `ignore_errors=True`
    swallows every OSError. Counting the removal off that return value meant
    "cleanup: removed 2 scratch dirs, freed 101 MB" was printed even when a
    permission error had left every byte on disk — the log contradicting the
    filesystem, which is the one thing a disk report must never do.
    """
    print("\n[7] the cleanup log cannot claim space it did not free")
    from loguru import logger

    from app.stages import StageWorkspace

    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(m), level="WARNING")

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        paths = _StubPaths(base)
        job = "job-fail"
        workspace = paths.data_dir / "work" / job
        workspace.mkdir(parents=True)
        (workspace / "big.bin").write_bytes(b"\0" * 4_000_000)

        # Make the delete silently do nothing, exactly as a permissions error
        # would with ignore_errors=True.
        original = task_mod.shutil.rmtree

        def _noop(path, **kwargs):  # noqa: ANN001, ANN202
            return None

        task_mod.shutil.rmtree = _noop
        try:
            worker = _StubSelf(_StubSettings(paths))
            task_mod.Pipeline._cleanup_job_temps(
                worker, job, StageWorkspace(paths.data_dir, job))
        finally:
            task_mod.shutil.rmtree = original

        check("the bytes are still on disk (the failure is real)",
              (workspace / "big.bin").exists())
        warned = [r for r in records if "still on disk" in r]
        check("a failed removal is reported, not silently counted",
              bool(warned), f"warnings seen: {records}")
        check("the report names the path that could not be removed",
              any(job in r for r in warned), f"warnings seen: {warned}")
        check("it does NOT claim the space was freed",
              not any("cleanup: removed" in r for r in records),
              f"records seen: {records}")

    logger.remove(sink_id)


def test_disk_measurement() -> None:
    print("\n[5] the disk number is measured, not taken on faith")
    from app.hosting import disk_stats

    settings = load_config(quiet=True)
    stats = disk_stats(settings.app.paths.data_dir, settings=settings)

    check("total is reported", stats.get("totalBytes") is not None)
    check("the real size of our own tree is reported",
          isinstance(stats.get("treeBytes"), int) and stats["treeBytes"] > 0,
          f"treeBytes={stats.get('treeBytes')}")
    check("the raw statvfs figure is kept alongside, not hidden",
          stats.get("statvfsFreeBytes") is not None)

    free, statvfs = stats.get("freeBytes"), stats.get("statvfsFreeBytes")
    if isinstance(free, int) and isinstance(statvfs, int):
        check("reported free space is never MORE optimistic than the filesystem's claim",
              free <= statvfs, f"free={free} statvfs={statvfs}")
        total, tree = stats["totalBytes"], stats["treeBytes"]
        check("it is the conservative of the two real measurements",
              free == min(statvfs, max(0, total - tree)),
              f"free={free} statvfs={statvfs} total-tree={total - tree}")
        print(f"       statvfs says {statvfs / 1024**2:.0f} MB free; "
              f"measured tree is {tree / 1024**2:.0f} MB; "
              f"reporting {free / 1024**2:.0f} MB free")

    # Unknown must stay unknown — never a confident zero, which would refuse
    # every job at submit time.
    bogus = disk_stats(Path("/definitely/not/here/at/all"), settings=settings)
    check("an unmounted path walks up to a real filesystem instead of reading 0",
          bogus.get("freeBytes") is None or bogus["freeBytes"] > 0,
          str(bogus))


def test_startup_sweep() -> None:
    print("\n[6] a sweep runs at boot, not 30 minutes later")
    body = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    startup = body.split("def _start_worker")[1][:2000]
    check("startup calls retention.sweep()",
          "retention.sweep()" in startup)
    check("a sweep failure cannot block startup",
          "except Exception" in startup and "startup sweep failed" in startup)


def main() -> int:
    test_retention_default()
    test_retention_env_wins()
    test_tree_size()
    test_completion_cleanup()
    test_disk_measurement()
    test_startup_sweep()
    test_cleanup_reports_truthfully()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
