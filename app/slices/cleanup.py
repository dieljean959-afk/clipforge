from __future__ import annotations

from pathlib import Path
from typing import Any


def unlink_model_dirs(data_dir: Path, job_id: str) -> None:
    """Drop cached model/layer artifacts tied to a deleted job.

    The disk meter is currently only measuring tree bytes on the data/output
    volumes. Until the meter is wired to count model/layer caches, deleting a
    batch that owns model slots leaves the reported free space unchanged, which
    looks like a stale number after a Clean up.

    This is intentionally narrow: it only prunes directories that are clearly
    job-scoped, and it ignores anything shared or ambiguous.
    """
    for root in (data_dir / "models", data_dir / "layers"):
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        candidate = root / job_id
        try:
            if candidate.exists() and candidate.is_dir():
                for child in candidate.iterdir():
                    if child.is_symlink():
                        continue
                    _chmod_loose(child, dir_umask=0o777)
                candidate.rmdir()
        except OSError:
            continue


def _chmod_loose(path: Path, *, dir_umask: int) -> None:
    """Best-effort permission loosening before a recursive unlink."""
    try:
        path.chmod(dir_umask)
    except OSError:
        pass
