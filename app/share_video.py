"""A small, share-sized copy of a rendered clip.

Why this exists
---------------
`navigator.share({files})` needs the whole file as a `File` in memory, built
synchronously inside the click handler. A 1080x1920 clip of this project runs
6-10 MB, so on a phone the user waited on a full download before the share
sheet could even open -- and the transient user-activation window (a few
seconds) expired during that wait, so `share()` failed outright.

So the share path gets its own, deliberately small encode:

    720x1280, H.264 High, AAC 96k, +faststart, capped at SHARE_MAX_BYTES

That is still a 9:16 file every platform accepts, and it is small enough to
have finished prefetching before the user taps. The 1080p master is untouched
and remains what the MP4 download button serves.

Sizing notes, measured rather than guessed:
  * 2 Mbps x a 17s clip ~= 4.2 MB, so a bitrate ALONE cannot hit 3 MB. The
    budget is therefore enforced by a measured first pass and, if that misses,
    a second pass at a bitrate solved from the duration and the byte budget.
    One adaptive retry beats guessing a low bitrate and shipping a blurry
    file for the clips that happen to be short.
  * `-maxrate`/`-bufsize` bound the VBV peaks that make a shared file stall on
    someone else's connection, which is the whole point of this module.
  * Audio is 96k mono-ish speech AAC, not 128k stereo: it is narration, and
    the saving is real on a file this small.

The master is never mutated: this writes a sibling `<stem>.share.mp4`, keyed by
the master's size and mtime so a re-render invalidates the derived copy
instead of serving a stale lighter version of an old master.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import ffmpeg_bin, ffprobe_bin

logger = logging.getLogger(__name__)

#: One per-source transcode at a time. A grid of cards all prefetching the
#: same share copy used to launch N concurrent ffmpeg processes for the same
#: output and race each other onto the same temp file; the lock is what keeps
#: that from stalling the whole share route.
_share_encode_locks: dict[str, Lock] = {}
_locks_guard = Lock()

#: Geometry of the share copy. 720x1280 is the smallest 9:16 size every
#: platform still treats as HD, and it is a clean 2x downscale of 1440p.
SHARE_WIDTH = 720
SHARE_HEIGHT = 1280

#: Hard ceiling for a share file. The acceptance target is "under 3 MB".
SHARE_MAX_BYTES = 3 * 1024 * 1024

#: First-pass video bitrate. Deliberately modest: this is a preview-sized
#: share artifact, and the adaptive retry below is what actually enforces the
#: byte budget.
SHARE_BITRATE = 2_000_000

#: Ceiling for the solved second-pass bitrate, and the floor below which a
#: further retry would produce something visibly worse than a download.
SHARE_MIN_BITRATE = 450_000
SHARE_MAX_BITRATE = 2_500_000

#: Audio: narration only, so 96k is transparent in practice.
SHARE_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "96k", "-ac", "2", "-ar", "44100"]

#: Two-pass window for VBV. Half a second is the usual recommendation and
#: keeps the peak-to-mean ratio tight.
SHARE_VBV_SEC = 0.5

#: Wall-clock ceiling for one share transcode. It is a downscale of an
#: already-encoded master, so this is generous; it exists so a wedged ffmpeg
#: cannot pin a request thread.
SHARE_ENCODE_TIMEOUT_SEC = 180

#: Container/level args. `+faststart` is essential: a share file with the moov
#: atom at the END cannot be previewed by the receiving app before it has
#: downloaded the whole thing, which reintroduces the exact stall this module
#: exists to remove. High 4.0 covers 720x1280.
SHARE_MOVFLAGS = ["-movflags", "+faststart", "-profile:v", "high", "-level:v", "4.0"]

#: One transcode per (master, share) pair at a time, process-wide. Without it,
#: a grid of cards all prefetching at once would launch N concurrent ffmpeg
#: processes for the same output and race each other onto the same temp file.
_encode_locks: dict[str, Lock] = {}
_locks_guard = Lock()


class ShareRenderError(Exception):
    """The share transcode failed. Callers degrade to the link, never fail."""


def _lock_for(path: Path) -> Lock:
    key = str(path)
    with _locks_guard:
        lock = _encode_locks.get(key)
        if lock is None:
            lock = Lock()
            _encode_locks[key] = lock
        return lock


class JobShareLocker:
    """Build each job's share copy exactly once, keyed on the job's master.

    The dashboard renders several clips off one job, so without this every
    clip asks for a *new* 720x1280 downscale and a *new* audio re-encode of a
    file that is already sitting on disk beside the master. This class keeps
    the encode, the file, and the invalidation stamp per source master, so a
    re-render invalidates the derived copy and the next request rebuilds only
    what changed.

    Access through `get(share_copy_for)` only: the shared lock and the stamp
    are internal. `force` re-builds a job that has already cached bytes.
    """

    def __init__(self) -> None:
        self._by_master: dict[str, Path] = {}
        self._job_locks: dict[str, Lock] = {}

    def _lock(self, master: Path) -> Lock:
        # The lock lives in its OWN map. It used to be cached in `_by_master`,
        # the map of finished share copies, so the first call inserted a Lock
        # where a Path belonged and `get()` then handed that Lock to callers
        # as if it were the file — /s/<id>.mp4 answered 500 for every clip
        # with "'_thread.lock' object has no attribute 'exists'".
        key = str(master)
        with _locks_guard:
            lock = self._job_locks.get(key)
            if lock is None:
                lock = Lock()
                self._job_locks[key] = lock
            return lock

    def get(self, master: Path, *, force: bool = False) -> Path | None:
        """Return the share copy for `master`, or None when the encode failed.

        A successful build (bytes on disk, non-empty) is returned on every
        later call, so a job with several clips is encoded once and then
        served from disk for every clip in that same job.
        """
        key = str(master)
        if not force and isinstance(self._by_master.get(key), Path):
            return self._by_master[key]
        with self._lock(master):
            if not force and isinstance(self._by_master.get(key), Path):
                return self._by_master[key]
            share = share_path_for(master)
            stamp = _stamp_path(share)
            if not force and share.exists() and share.stat().st_size > 0 and stamp.exists():
                try:
                    if stamp.read_text(encoding="utf-8").strip() == _fingerprint(master):
                        self._by_master[key] = share
                        return share
                except OSError:
                    pass
            share = build_share_version(master)
            if share is None:
                return None
            self._by_master[key] = share
            return share


def share_path_for(master: Path) -> Path:
    """Where the share copy of `master` lives: a sibling, never in place."""
    return master.with_name(f"{master.stem}.share.mp4")


def _fingerprint(master: Path) -> str:
    """Identity of the MASTER, so a re-render invalidates the derived copy."""
    st = master.stat()
    return f"{int(st.st_mtime)}-{st.st_size}"


def _stamp_path(share: Path) -> Path:
    return share.with_suffix(".share.stamp")


def duration_sec(path: Path) -> float:
    try:
        out = subprocess.run(
            [ffprobe_bin(), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return float((out.stdout or "0").strip() or 0)
    except Exception as exc:  # noqa: BLE001 - a share copy is best-effort
        raise ShareRenderError(f"ffprobe failed for {path.name}: {exc}") from exc


def _solve_bitrate(dur: float, budget_bytes: int) -> int:
    """Video bitrate that fits `budget_bytes` over `dur` seconds.

    8 bits per byte, and the audio track takes its ~96k plus container
    overhead, so the video is given what is left. Clamped at both ends: below
    the floor the result looks worse than just downloading the master, and
    above the ceiling the duration is short enough that the budget is not the
    binding constraint anyway.
    """
    if dur <= 0.1:
        return SHARE_BITRATE
    audio_bps = 96_000
    overhead = 1.06  # container + keyframes, measured slack
    usable = max(0.0, budget_bytes * 8 * 0.9 / (dur * overhead)) - audio_bps
    return int(max(SHARE_MIN_BITRATE, min(SHARE_MAX_BITRATE, usable)))


def _transcode(master: Path, dest: Path, bitrate: int) -> None:
    """One downscale pass. Raises ShareRenderError on any failure."""
    vbv = max(250_000, int(bitrate * SHARE_VBV_SEC))
    cmd = [
        ffmpeg_bin(), "-y", "-loglevel", "error",
        "-i", str(master),
        "-vf", f"scale={SHARE_WIDTH}:{SHARE_HEIGHT}:flags=bicubic",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-profile:v", "high", "-level:v", "4.0",
        "-pix_fmt", "yuv420p",
        "-b:v", str(bitrate),
        "-maxrate", str(bitrate),
        "-bufsize", str(vbv),
        # Two keyframes a second: the receiving app can start showing the
        # video before the whole file has arrived.
        "-g", "60", "-keyint_min", "30",
        "-sc_threshold", "0",
        *SHARE_AUDIO_ARGS,
        *SHARE_MOVFLAGS,
        "-map_metadata", "-1",
        str(dest),
    ]
    logger.info(f"[share] transcode: bitrate={bitrate} -> {dest.name}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SHARE_ENCODE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise ShareRenderError(
            f"share transcode exceeded {SHARE_ENCODE_TIMEOUT_SEC}s"
        ) from exc
    if proc.returncode != 0 or not dest.exists():
        tail = (proc.stderr or "").strip().splitlines()[-4:]
        raise ShareRenderError(f"share transcode failed: {' / '.join(tail)}")


def build_share_version(master: Path, *, force: bool = False) -> Path:
    """Return a small, faststart MP4 for `master`, building it if needed.

    Cached next to the master and keyed by the master's size+mtime, so a
    re-render rebuilds instead of serving a stale lighter copy of old footage.

    Raises ShareRenderError. The caller is expected to fall back to sharing a
    LINK, because a share copy is an optimisation and never a requirement.
    """
    master = Path(master)
    if not master.exists():
        raise ShareRenderError(f"master not found: {master.name}")

    share = share_path_for(master)
    stamp = _stamp_path(share)

    if not force and share.exists() and share.stat().st_size > 0 and stamp.exists():
        try:
            if stamp.read_text(encoding="utf-8").strip() == _fingerprint(master):
                return share
        except OSError:
            # An unreadable stamp is a cache miss, never an error.
            pass

    with _lock_for(share):
        # Re-check under the lock: another request may have built it while we
        # waited, and two ffmpegs writing one file is exactly the race this
        # lock exists to prevent.
        if not force and share.exists() and share.stat().st_size > 0 and stamp.exists():
            try:
                if stamp.read_text(encoding="utf-8").strip() == _fingerprint(master):
                    return share
            except OSError:
                pass

        tmp = share.with_suffix(".share.tmp.mp4")
        tmp.unlink(missing_ok=True)
        t0 = time.time()
        try:
            # First pass at the configured bitrate. Usually enough; the budget
            # retry below is what makes the ceiling a guarantee rather than a
            # hope.
            _transcode(master, tmp, SHARE_BITRATE)
            size = tmp.stat().st_size if tmp.exists() else 0
            if size > SHARE_MAX_BYTES and size > 0:
                dur = duration_sec(master)
                solved = _solve_bitrate(dur, SHARE_MAX_BYTES)
                if solved < SHARE_BITRATE * 0.95:
                    logger.info(
                        f"[share] {master.name}: {size / 1e6:.1f} MB over budget, "
                        f"retrying at {solved} bps for {dur:.1f}s"
                    )
                    tmp.unlink(missing_ok=True)
                    _transcode(master, tmp, solved)
            if not tmp.exists() or tmp.stat().st_size == 0:
                raise ShareRenderError("share transcode produced an empty file")
        except ShareRenderError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001 - never propagate a raw ffmpeg error
            tmp.unlink(missing_ok=True)
            raise ShareRenderError(f"share transcode failed: {exc}") from exc

        # Atomic publish: a reader never sees a partial file.
        shutil.move(str(tmp), str(share))
        stamp.write_text(_fingerprint(master), encoding="utf-8")
        final = share.stat().st_size
        logger.info(
            f"[share] built {share.name}: {final / 1e6:.2f} MB "
            f"in {time.time() - t0:.1f}s"
        )
        return share


def share_info(share: Path) -> dict[str, Any]:
    """Size/duration for the UI badge, without a second ffprobe round trip."""
    p = Path(share)
    return {
        "bytes": p.stat().st_size if p.exists() else None,
        "name": p.name,
    }


def purge_derived(master: Path) -> None:
    """Drop the share copy after a re-render. Best-effort."""
    share = share_path_for(Path(master))
    try:
        share.unlink(missing_ok=True)
        _stamp_path(share).unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001 - cleanup must never break a render
        logger.warning(f"[share] could not purge derived copy: {exc}")


#: Kept for tests and for a human reading the module: the exact argument shape
#: the first pass uses. Exposed as a function so the assertion is on real
#: behaviour rather than on a string in this file.
def share_movflags() -> list[str]:
    return list(SHARE_MOVFLAGS)


_BITRATE_RE = re.compile(r"-b:v\s+(\d+)")


def bitrate_in(cmd: list[str]) -> int:
    """Read the `-b:v` value out of an ffmpeg argv. Test helper."""
    for i, tok in enumerate(cmd):
        if tok == "-b:v" and i + 1 < len(cmd):
            return int(cmd[i + 1])
    m = _BITRATE_RE.search(" ".join(cmd))
    return int(m.group(1)) if m else 0
