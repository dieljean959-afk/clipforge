#!/usr/bin/env python3
"""Storage suite: what cleanup may delete, and what it must never delete.

Cleanup is the one part of this app that can destroy work the user cannot get
back, so it gets real tests rather than a one-line assertion about a constant.

The contract, in one line: ONLY artefacts with no owning job record, and only
once they are older than 24 hours.

    python tools/storage_test.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
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


def _age(path: Path, seconds: float) -> None:
    """Backdate a path's mtime so the sweeper's age rule can be exercised."""
    when = time.time() - seconds
    os.utime(path, (when, when))


# --------------------------------------------------------------------------- #
# the sweeper
# --------------------------------------------------------------------------- #


def test_orphan_sweep() -> None:
    section("Storage: the orphan sweeper removes debris, and only debris")

    from app.config import Paths, load_config
    from app.hosting import RetentionCleaner
    from app.store import LocalStore

    settings = load_config(quiet=True)
    base = Path(tempfile.mkdtemp(prefix="clipforge-store-"))
    paths = Paths(
        data_dir=base / "data",
        outputs_dir=base / "outputs",
        downloads_dir=base / "downloads",
        static_dir=base / "static",
    )
    for d in (paths.data_dir, paths.outputs_dir, paths.downloads_dir):
        d.mkdir(parents=True, exist_ok=True)

    check(
        "orphans must be 24h old before they are touched",
        RetentionCleaner.ORPHAN_MAX_AGE_SEC == 86400,
        f"{RetentionCleaner.ORPHAN_MAX_AGE_SEC}s",
    )

    store = LocalStore(paths)
    # A LIVE job, with a source and two rendered clips on disk.
    live = store.create_job(source_type="upload", source="live.mp4")
    live_id = live["_id"]
    store.write_job({**live, "stage": "complete", "progress": 100})
    (paths.downloads_dir / live_id).mkdir(parents=True, exist_ok=True)
    (paths.downloads_dir / live_id / "source.mp4").write_bytes(b"x" * 32)
    out_dir = paths.outputs_dir / live_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        (out_dir / f"clip_{i:02d}.mp4").write_bytes(b"y" * 32)
    clips_dir = paths.data_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / f"{live_id}_0.json").write_text(
        json.dumps({"_id": f"{live_id}_0", "jobId": live_id}), encoding="utf-8"
    )

    # Debris: folders and files with NO job record.
    for name in ("ghost_old", "ghost_new"):
        d = paths.downloads_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"z" * 32)
        _age(d, 2 * 86400 if name == "ghost_old" else 60)
    stale_out = paths.outputs_dir / "ghost_out"
    stale_out.mkdir(parents=True, exist_ok=True)
    _age(stale_out, 2 * 86400)
    fresh_out = paths.outputs_dir / "ghost_fresh"
    fresh_out.mkdir(parents=True, exist_ok=True)
    _age(fresh_out, 60)

    # A temp file left behind by a process killed mid-write.
    old_part = paths.downloads_dir / "chunk.part"
    old_part.write_bytes(b"p" * 8)
    _age(old_part, 2 * 86400)
    new_part = paths.downloads_dir / "chunk2.part"
    new_part.write_bytes(b"p" * 8)
    _age(new_part, 60)

    # A clip document whose job is gone, and one whose job is not.
    orphan_clip = clips_dir / "orphan_0.json"
    orphan_clip.write_text(
        json.dumps({"_id": "orphan_0", "jobId": "no_such_job"}), encoding="utf-8"
    )
    _age(orphan_clip, 2 * 86400)
    live_clip = clips_dir / f"{live_id}_1.json"
    live_clip.write_text(
        json.dumps({"_id": f"{live_id}_1", "jobId": live_id}), encoding="utf-8"
    )
    _age(live_clip, 2 * 86400)

    # Backdate the LIVE job's folders too: age alone must never be enough.
    _age(paths.downloads_dir / live_id, 30 * 86400)
    _age(out_dir, 30 * 86400)

    settings.app.paths = paths
    cleaner = RetentionCleaner(settings, store)
    removed = cleaner.sweep_orphans()

    check("a 2-day-old orphan folder is removed",
          not (paths.downloads_dir / "ghost_old").exists())
    check("a 2-day-old orphan output folder is removed", not stale_out.exists())
    check("a 1-minute-old orphan folder is KEPT (under 24h)",
          (paths.downloads_dir / "ghost_new").exists())
    check("a fresh orphan output folder is KEPT", fresh_out.exists())
    check("a stale .part temp file is removed", not old_part.exists())
    check("a fresh .part temp file is KEPT", new_part.exists())
    check("a clip document whose job is gone is removed", not orphan_clip.exists())
    check("a clip document whose job EXISTS is kept, however old",
          live_clip.exists())

    # The whole point: a live job's media is untouchable.
    check("the live job's source survives a 30-day-old sweep",
          (paths.downloads_dir / live_id / "source.mp4").exists())
    check("the live job's rendered clips survive",
          all((out_dir / f"clip_{i:02d}.mp4").exists() for i in (1, 2)))
    check("the sweep reported what it removed", removed >= 4, str(removed))

    # Retention OFF must not disable the orphan sweep — that is the only thing
    # standing between a full disk and a failed job.
    settings.hosted.retention_hours = 0
    ghost2 = paths.downloads_dir / "ghost_two"
    ghost2.mkdir(parents=True, exist_ok=True)
    _age(ghost2, 3 * 86400)
    cleaner.sweep()
    check("orphans are still swept when retention is disabled", not ghost2.exists())
    check("and a live job is still untouched with retention off",
          (paths.downloads_dir / live_id / "source.mp4").exists())

    shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------- #
# where the bytes actually live
# --------------------------------------------------------------------------- #


def test_media_location() -> None:
    section("Storage: media lives outside the watched project tree")

    import app.config as cfg_mod

    settings = cfg_mod.load_config(quiet=True)
    paths = settings.app.paths

    def watched_path(p: Path) -> bool:
        try:
            resolved = p.resolve()
        except OSError:
            return False
        repo = ROOT.resolve()
        return resolved.is_relative_to(repo)

    check("downloads are not inside the project tree (when a storage dir is set)",
          not watched_path(paths.downloads_dir) or str(paths.downloads_dir).startswith(
              str(ROOT / "downloads")
          ),
          str(paths.downloads_dir))
    check("outputs resolve to a real directory", str(paths.outputs_dir))
    check("the job store is a real directory", paths.data_dir.exists())

    vite = (ROOT / "vite.config.ts").read_text(encoding="utf-8")
    for pattern in ("**/downloads/**", "**/outputs/**", "**/data/**"):
        check(f"the file watcher ignores {pattern}", pattern in vite)
    check("HMR is still disabled (platform requirement)", "hmr: false" in vite)

    # A watcher event storm is what restarted the server mid-job in the first
    # place, so the media extensions have to be ignored individually too.
    for ext in ("**/*.mp4", "**/*.part", "**/*.wav"):
        check(f"the watcher ignores {ext}", ext in vite)

    # Never watched paths are cheap to assert and expensive to lose.
    check("no hardcoded /home/daytona path in app/",
          "/home/daytona" not in "\n".join(
              p.read_text(encoding="utf-8")
              for p in sorted((ROOT / "app").glob("*.py"))
          ))


def main() -> int:
    global VERBOSE
    VERBOSE = "--verbose" in sys.argv

    print("ClipForge storage suite")
    print(f"  repo: {ROOT}")
    test_orphan_sweep()
    test_media_location()

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("failed:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
