"""Bound the log files and drop model copies no job needs.

The free-space self-check that failed a finished batch was measuring this, not
the render: ``data/logs/backend.log`` is appended to for the life of the web
process (247,665 lines / 27 MB here), ``api.log`` rotates at 10 MB and keeps 5
generations, and ``data/models`` held 464 MB of weight copies after the batches
that used them had finished.

Every file here is *reproducible*: logs are diagnostics, and model weights come
back from the bucket or HuggingFace. Nothing a user can lose lives in them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from loguru import logger

#: Supervisor capture: keep it useful, never let it grow without limit.
BACKEND_LOG_MAX = 4 * 1024 * 1024
BACKEND_LOG_KEEP = 2 * 1024 * 1024
#: Rotated api logs older than this are diagnostics nobody reads.
API_LOG_KEEP_SEC = 3 * 24 * 3600


def _trim(path: Path, max_bytes: int, keep_bytes: int) -> int:
    """Truncate ``path`` to its last ``keep_bytes`` bytes. Returns bytes freed."""
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if size <= max_bytes:
        return 0
    try:
        with path.open("rb") as fh:
            fh.seek(-keep_bytes, 2)
            tail = fh.read()
        path.write_bytes(b"[... trimmed by log hygiene ...]\n" + tail)
        return size - len(tail)
    except OSError as exc:
        logger.warning(f"[hygiene] could not trim {path}: {exc}")
        return 0


def sweep(settings: Any) -> dict[str, Any]:
    """Trim logs + release idle model weights. Returns what was freed."""
    data_dir: Path = settings.app.paths.data_dir
    logs = data_dir / "logs"
    freed = 0
    trimmed: list[str] = []

    backend = logs / "backend.log"
    got = _trim(backend, BACKEND_LOG_MAX, BACKEND_LOG_KEEP)
    if got:
        freed += got
        trimmed.append(f"backend.log {got // 1048576} MB")

    # Rotated api logs beyond the window. The live api.log is loguru's and is
    # handled by its own rotation/retention, so it is only trimmed if it is
    # somehow oversized.
    cutoff = time.time() - API_LOG_KEEP_SEC
    for p in logs.glob("api.*.log"):
        try:
            if p.stat().st_mtime < cutoff:
                freed += p.stat().st_size
                p.unlink(missing_ok=True)
                trimmed.append(p.name)
        except OSError:
            continue
    got = _trim(logs / "api.log", 20 * 1024 * 1024, 10 * 1024 * 1024)
    if got:
        freed += got
        trimmed.append(f"api.log {got // 1048576} MB")

    released = 0
    try:
        from app import modelstore

        out = modelstore.release_all(lambda m: logger.info(f"[hygiene] {m}"))
        released = int(out.get("freed") or 0)
    except Exception as exc:  # noqa: BLE001 - weights are an optimisation
        logger.debug(f"[hygiene] model release skipped: {exc}")

    total = freed + released
    detail = (
        f"{round(total / 1048576, 1)} MB freed"
        + (f" ({', '.join(trimmed[:3])})" if trimmed else "")
        + (f", model copies {round(released / 1048576, 1)} MB" if released else "")
    )
    if total:
        logger.info(f"[hygiene] {detail}")
    return {"detail": detail, "freedBytes": total, "trimmed": trimmed, "released": released}
