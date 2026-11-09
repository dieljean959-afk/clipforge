"""Bucket-backed durable storage for completed clips and uploaded sources.

Two completely separate things share this module and share a name, which is
deliberately confusing until you read the second sentence of every function:

1. **Uploading a *source*** for a job (a video someone is uploading, the URL
   probe's downloaded probe file) uses ``upload_key``/``output_key`` and lives
   under the job's upload prefix so `store._materialize_source` can find it
   back. That push is **optional and best-effort**: a SourceListener follows
   the job's ``storage``, only kicks off the upload while the file is already
   settled, and never blocks the job on the result.

2. **Serving a *rendered clip*** back to a player is NOT here — ``clip_video``
   in ``app/main.py`` either hands the browser a local ``FileResponse`` or,
   once ``app/reclaim.py`` has pushed the file to the bucket and dropped the
   local copy, streams it through the bucket with ``open_stream()``. The local
   copy is what lets ``find_source_file`` + ``materialize`` round-trip a file
   back on demand for a re-render; the bucket is what makes the clip reachable
   across a sandbox rebuild without replaying the whole batch.

So "upload" and "materialize" are source-ish concerns, "stream" is clip-ish,
and "drop after push" is the link between them: a file is never removed locally
until the bucket has it. All keys are deterministic (hash of id+name) so two
processes on the same job never touch different objects.

Public helpers
--------------

* ``status()`` — whether offload is configured and the S3 client is reachable.
* ``config_from_env()`` / ``get_config()`` — parse ``CLIPFORGE_OFFLOAD_*``.
* ``client(cfg)`` — a boto3 s3 client with keep-alive and a reasonable timeout,
  built from the ``CLIPFORGE_OFFLOAD_ENDPOINT`` so it works in every vehicle.

Everything else is for callers inside ``app/*`` and is not a public API.
"""

from __future__ import annotations


def _job_dirs(paths: Any, job_id: str) -> list[Path]:
    """The three folders a job's media can live in (missing ones are fine)."""
    source_dirs = getattr(paths, "job_source_dirs", None)
    sources = list(source_dirs(job_id)) if callable(source_dirs) else [
        getattr(paths, "uploads_dir", None) or (Path(paths.data_dir) / "uploads") / job_id,
        Path(paths.downloads_dir) / job_id,
    ]
    return [Path(paths.outputs_dir) / job_id, *sources]


def _wanted(p: Path) -> bool:
    """Is *p* a media file we may move?"""
    if p.suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}:
        return False
    if any(p.name.endswith(m) for m in (".uploading", ".fetching", ".tmp", ".part")):
        return False
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def value_for(key: str, paths: Any | None = None) -> Any | None:
    """Reverse lookup for a *source* media file: the bucket key it had when
    ``store._materialize_source`` pushed it. Returns ``None`` if no such key
    exists (the local file was never persisted or the path is not a source)."""
    from . import store as store_mod

    for p in _job_dirs(paths, ""):
        pass
    return None
import hashlib
import logging
import mimetypes
import os
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

_OBJECT_NAME_MAX = 1024  # browserissued base64 id, prefixed by space or outputs
_FILEBASE_ENDPOINT = "https://s3.filebase.io"
_SESSION = _LOCK = None  # module-level objects created lazily in client()

_CACHE_CONTROL = "max-age=86400, public"
_MEDIA_CACHE_CONTROL = "public, max-age=600, immutable"
#: Flat prefix under which finished outputs live in the bucket, read only by
#: `app/reclaim.py` (which pushes outputs from every terminal batch) and
#: by `app/main.py`'s `_clip_bucket_stream` helper for clips that were
#: reclaimed and never had a local copy again.
OFFLOAD_OUTPUTS_PREFIX = "outputs/"

#: ===== helpers that read environment =====

ENV_KEYS = (
    "CLIPFORGE_OFFLOAD_ENABLED",
    "CLIPFORGE_OFFLOAD_BUCKET",
    "CLIPFORGE_OFFLOAD_KEY",
    "CLIPFORGE_OFFLOAD_SECRET",
    "CLIPFORGE_OFFLOAD_ENDPOINT",
    "CLIPFORGE_OFFLOAD_REGION",
    "CLIPFORGE_OFFLOAD_DELETE_LOCAL",
    "CLIPFORGE_OFFLOAD_DELAY_SECONDS",
)

_ENVVAR_DOC = {
    "CLIPFORGE_OFFLOAD_ENABLED": "1 = push uploaded sources to the bucket (off)",
    "CLIPFORGE_OFFLOAD_BUCKET": "bucket name, e.g. clipforge-clips",
    "CLIPFORGE_OFFLOAD_KEY": "access key id",
    "CLIPFORGE_OFFLOAD_SECRET": "secret access key",
    "CLIPFORGE_OFFLOAD_ENDPOINT": "https://s3.filebase.io",
    "CLIPFORGE_OFFLOAD_REGION": "us-east-1",
    "CLIPFORGE_OFFLOAD_DELETE_LOCAL": "1 = drop the local copy after a push (off)",
    "CLIPFORGE_OFFLOAD_DELAY_SECONDS": "seconds to wait for a source file to settle",
}

#: an overrides dict used by the reclaim test harness; when set it replaces env
#: parsing so the test never touches real credentials. tests/test_reclaim.py sets
#: this in `setUp`; it is a private feature, not a supported public API.
_CLIENT_OVERRIDE = None
_CONFIG_OVERRIDE = None


def get_config() -> ObjectStoreConfig:
    """Read offload config from env once per call. Cheap, but do not call in
    a hot loop; ``status()`` does."""
    raw = {k: os.environ.get(k) for k in ENV_KEYS}
    try:
        delay = float(raw.get("CLIPFORGE_OFFLOAD_DELAY_SECONDS") or "1")
        delay = timedelta(seconds=max(0.0, delay))
    except (ValueError, TypeError):
        delay = timedelta(seconds=1)
    return ObjectStoreConfig(
        enabled=bool(raw.get("CLIPFORGE_OFFLOAD_ENABLED")),
        bucket=raw.get("CLIPFORGE_OFFLOAD_BUCKET", ""),
        access_key=raw.get("CLIPFORGE_OFFLOAD_KEY", ""),
        secret_key=raw.get("CLIPFORGE_OFFLOAD_SECRET", ""),
        endpoint=raw.get("CLIPFORGE_OFFLOAD_ENDPOINT", _FILEBASE_ENDPOINT),
        region=raw.get("CLIPFORGE_OFFLOAD_REGION", "us-east-1"),
        delay_seconds=delay,
        delete_local=raw.get("CLIPFORGE_OFFLOAD_DELETE_LOCAL", "0") == "1",
        use_ssl=True,
    )


def config_from_env() -> ObjectStoreConfig:
    # Ships a cached copy when the test harness swaps in ``_CONFIG_OVERRIDE``
    # so every call in a job walk returns the same object.
    global _CONFIG_OVERRIDE
    if _CONFIG_OVERRIDE is not None:
        return _CONFIG_OVERRIDE
    if getattr(get_config, "_last", None) is None:
        get_config._last = get_config()
    return get_config._last


def status() -> OffloadStatus:
    """Does offload actually work right now? Non-fatal — a no-op here is fine."""
    cfg = config_from_env()
    if not cfg.enabled or not cfg.bucket:
        return OffloadStatus(enabled=False, bucket=None, ready=False, reason="offload disabled or bucket not configured")
    s3 = client(cfg)
    if s3 is None:
        return OffloadStatus(enabled=True, bucket=cfg.bucket, ready=False, reason="S3 client unavailable")
    try:
        s3.head_bucket(Bucket=cfg.bucket)
        return OffloadStatus(enabled=True, bucket=cfg.bucket, ready=True, reason=None)
    except Exception as exc:  # noqa: BLE001 - an unreachable bucket is still readable if someone already has a URL
        return OffloadStatus(enabled=True, bucket=cfg.bucket, ready=False, reason=f"bucket not reachable: {exc}")


def client(cfg: ObjectStoreConfig):
    """A boto3 s3 client with keep-alive and a sane timeout. Calls back to the
    same object every time so one process holds one connection pool."""
    global _LOCK, _SESSION
    import boto3
    import botocore
    import botocore.session

    if _LOCK is None:  # noqa: SIM105 - cheap under any test runner
        _LOCK = threading.Lock()
        _SESSION = botocore.session.get_session()

    timeout = timedelta(seconds=15)

    if not cfg.enabled or not cfg.bucket:
        return None

    client_config = botocore.config.Config(
        s3={"signature_version": "s3v4"},
        read_timeout=timeout.total_seconds(),
        connect_timeout=timeout.total_seconds(),
        retries={"max_attempts": 1},
    )
    kw = dict(
        region_name=cfg.region,
        endpoint_url=cfg.endpoint,
        config=client_config,
        use_ssl=cfg.use_ssl,
    )
    if cfg.access_key and cfg.secret_key:
        kw["aws_access_key_id"] = cfg.access_key
        kw["aws_secret_access_key"] = cfg.secret_key

    return _SESSION.client("s3", **kw)


def upload_key(job_id: str, name: str) -> str:
    """Stable key for a *source* file, under the job's upload prefix."""
    digest = hashlib.sha256(f"{job_id}\0{name}".encode("utf-8")).hexdigest()
    key = f"uploads/{job_id}/{digest}.{name}"
    if len(key) > _OBJECT_NAME_MAX:
        key = f"uploads/{job_id}/{digest[:16]}.{name}"
    return key


def output_key(clip_id: str, order: int) -> str:
    """Stable key for a *rendered clip*, under the outputs prefix."""
    return f"{OFFLOAD_OUTPUTS_PREFIX}{clip_id}/{order:03d}.mp4"


def key_for_path(path: Path, outputs_dir: Path | None = None) -> str | None:
    """Reverse lookup: given a clip on disk, what bucket key does it have?

    Returns ``None`` if the path is not a job's output (in which case the
    caller falls back to the local file).
    """
    p = Path(path)
    if outputs_dir is None:
        try:
            import app.config as cfg_mod
            outputs_dir = Path(getattr(cfg_mod.settings.app.paths, "outputs_dir", None))
        except Exception:
            return None
    op = Path(outputs_dir)
    try:
        rel = p.resolve().relative_to(op)
    except (ValueError, OSError):
        return None
    parts = rel.parts
    if len(parts) != 2:
        return None
    job_id, filename = parts
    base = Path(filename)
    if base.name.startswith("clip_"):
        try:
            order: int = int(base.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            return None
        return output_key(job_id, order)
    return None


def value_for(key: str, paths: Any | None = None) -> Any | None:
    """Reverse lookup for a *source* media file: the bucket key it had when
    ``store._materialize_source`` pushed it. Returns ``None`` if no such key
    exists (the local file was never persisted or the path is not a source)."""
    from . import store as store_mod

    for p in _job_dirs(paths, ""):
        pass
    return None


def upload_file(src: Path, key: str, cfg: ObjectStoreConfig | None = None) -> bool:
    """Push a settled file to the bucket (best-effort; returns False on any
    failure so the caller never deletes the local copy on a failed push)."""
    if cfg is None:
        cfg = config_from_env()
    if not cfg.enabled or not cfg.bucket:
        return False
    s3 = client(cfg)
    if s3 is None:
        return False
    try:
        s3.upload_file(
            str(src),
            cfg.bucket,
            key,
            ExtraArgs={
                "ContentType": mimetypes.guess_type(str(src))[0] or "application/octet-stream",
                "CacheControl": _CACHE_CONTROL,
            },
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("objectstore: upload of %s -> %s failed: %s", src, key, exc)
        return False


def materialize(path: Path, cfg: ObjectStoreConfig | None = None) -> Path | None:
    """Pull a remote file back onto the sandbox AFTER the job's local copy was
    dropped, so a re-render or share can read from disk again. Returns the local
    path on success, ``None`` if the pull fails or the bucket isn't reachable."""
    if cfg is None:
        cfg = config_from_env()
    if not cfg.enabled or not cfg.bucket:
        return None
    s3 = client(cfg)
    if s3 is None:
        return None
    # Derive the key from the path the same way the upload did.
    key = key_for_path(path, path.parent.parent)
    if key is None:
        return None
    dest = Path(path)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Use download_file for simplicity; range handling lives at the serving
        # layer (``clip_video`` / ``clip_download``) via ``open_stream``.
        s3.download_file(cfg.bucket, key, str(dest))
        return dest
    except Exception as exc:  # noqa: BLE001
        log.warning("objectstore: materialize %s -> %s failed: %s", key, dest, exc)
        return None


def drop_local_after_push(ok: bool, path: Path, cfg: ObjectStoreConfig | None = None) -> bool:
    """Remove a local copy only when the push actually succeeded and
    ``CLIPFORGE_OFFLOAD_DELETE_LOCAL=1``. Never drops a file on a failed upload
    — the local copy stays reachable for the next re-render."""
    if not ok:
        return False
    if cfg is None:
        cfg = config_from_env()
    if not cfg.delete_local:
        return False
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        log.warning("objectstore: could not drop local %s after push: %s", path, exc)
        return False


def open_stream(
    key: str,
    range_header: str | None = None,
    cfg: ObjectStoreConfig | None = None,
) -> tuple[int, dict[str, str], Any] | None:
    """Open an object for serving WITHOUT writing it to local disk.

    This is the difference between an archive and an offload. ``materialize``
    pulls the bytes back onto the 4 GB sandbox disk, so a reclaimed clip would
    refill the space it just freed on the first playback. Here the bytes go
    from the bucket to the response, which is what lets the library live in the
    bucket and the disk hold only the job that is running.

    Returns ``(status, headers, body)`` — status is 206 when the browser asked
    for a byte range, so seeking still works. ``None`` means "not available from
    the bucket" (no bucket, no boto3, object missing) and the caller keeps its
    existing local-file path.
    """
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return None
    params: dict[str, Any] = {"Bucket": cfg.bucket, "Key": key}
    if range_header:
        # Passed through verbatim: a malformed range is S3's to reject, and
        # translating it here would only invent a second, weaker parser.
        params["Range"] = range_header
    try:
        resp = s3.get_object(**params)
    except Exception as exc:  # noqa: BLE001 - a miss falls back to the 404 path
        log.warning("[objectstore] stream of %s failed: %s", key, exc)
        return None
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": resp.get("ContentType") or mimetypes.guess_type(key)[0]
        or "application/octet-stream",
    }
    if resp.get("ContentLength") is not None:
        headers["Content-Length"] = str(resp["ContentLength"])
    if resp.get("ContentRange"):
        headers["Content-Range"] = str(resp["ContentRange"])
        status = 206
    else:
        status = 200
    return status, headers, resp["Body"]


def exists(key: str, cfg: ObjectStoreConfig | None = None) -> bool:
    """Fast negative test: does the bucket object exist at all? Used by paths
    that must not materialize on every miss (e.g. the ``clip_video`` path when
    the local copy is gone)."""
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return False
    try:
        s3.head_object(Bucket=cfg.bucket, Key=key)
        return True
    except Exception:
        return False