"""Durable offload for media, on any S3-compatible object store.

WHY THIS EXISTS
---------------
The only writable filesystem in this sandbox is a 4 GB virtiofs that is part of
the sandbox itself. When the sandbox is rebuilt, the codebase comes back and the
sibling data directory does not — which is exactly how 11 of 18 live jobs ended
up pointing at a `downloads/<id>/` folder that no longer existed. No local path
can fix that; only something that outlives the sandbox can.

So: media is PUSHED to object storage when it is finished, and PULLED back on
demand. Nothing about the pipeline changes — ffmpeg, Whisper and every serving
route still work on a local file, because a pull-through materialises the
object before anything touches it.

DESIGN RULES (all of them are load-bearing)
--------------------------------------------
1. **Inert unless configured.** With no bucket/keys set, every function here is
   a no-op returning False, and the app behaves exactly as it did before this
   module existed. That is what makes it safe to ship.
2. **Never raises.** A storage outage must not fail a render that has already
   succeeded, and must not turn a playable clip into a 500. Every call is
   wrapped; failures are logged and reported as `False`.
3. **Write-through, then optionally drop the local copy.** The local disk is
   the scarce resource, so after a successful push the local file is removed
   unless `keep_local` is set. It comes straight back on the next read.
4. **Provider-agnostic.** Any S3-compatible endpoint works — Cloudflare R2,
   Backblaze B2, AWS S3, Wasabi, Storj, Tigris.

ENVIRONMENT
-----------
    CLIPFORGE_S3_ENDPOINT    e.g. https://<account>.r2.cloudflarestorage.com
    CLIPFORGE_S3_BUCKET      bucket name
    CLIPFORGE_S3_ACCESS_KEY  access key id
    CLIPFORGE_S3_SECRET_KEY  secret access key
    CLIPFORGE_S3_REGION      optional, default "auto" (R2/B2 convention)
    CLIPFORGE_S3_PREFIX      optional key prefix, default "clipforge"
    CLIPFORGE_OFFLOAD_DELETE_LOCAL   1 (default) to drop the local copy after a
                            successful push; 0 to keep both copies
"""

from __future__ import annotations

import mimetypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

# Test seam: tools/object_store_test.py injects a fake S3 client here so the
# whole offload path is exercised without a network or a real bucket.
_CLIENT_OVERRIDE: Any = None

_client_lock = threading.Lock()
_client_cache: Any = None
_warned: set[str] = set()


@dataclass(frozen=True)
class ObjectStoreConfig:
    """Resolved from the environment. `enabled` means 'all four fields set'."""

    endpoint: str = ""
    bucket: str = ""
    access_key: str = ""
    secret_key: str = ""
    region: str = "auto"
    prefix: str = "clipforge"
    delete_local: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint and self.bucket and self.access_key and self.secret_key)


def config_from_env() -> ObjectStoreConfig:
    """Read the store's settings from the environment.

    Called per operation rather than snapshotted at boot, for the same reason
    the LLM keys are re-read live: the platform can deliver credentials after
    the process is already running, and a restart is not always available.
    """
    def flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    return ObjectStoreConfig(
        endpoint=os.getenv("CLIPFORGE_S3_ENDPOINT", "").strip(),
        bucket=os.getenv("CLIPFORGE_S3_BUCKET", "").strip(),
        access_key=os.getenv("CLIPFORGE_S3_ACCESS_KEY", "").strip(),
        secret_key=os.getenv("CLIPFORGE_S3_SECRET_KEY", "").strip(),
        region=os.getenv("CLIPFORGE_S3_REGION", "auto").strip() or "auto",
        prefix=(os.getenv("CLIPFORGE_S3_PREFIX", "clipforge").strip() or "clipforge"),
        delete_local=flag("CLIPFORGE_OFFLOAD_DELETE_LOCAL", True),
    )


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        logger.warning(f"[objectstore] {msg}")


def client(cfg: ObjectStoreConfig | None = None) -> Any:
    """A cached S3 client, or None when unavailable.

    Cached because building one costs a credential chain walk; rebuilt when the
    configuration changes so newly-delivered keys take effect without a restart.
    """
    global _client_cache
    if _CLIENT_OVERRIDE is not None:
        return _CLIENT_OVERRIDE
    cfg = cfg or config_from_env()
    if not cfg.enabled:
        return None
    with _client_lock:
        cached = _client_cache
        if cached is not None and getattr(cached, "_clipforge_cfg", None) == cfg:
            return cached
        try:
            import boto3  # imported lazily: absent boto3 must not break startup
            from botocore.config import Config
        except ImportError:
            _warn_once(
                "boto3 is not installed — media offload is OFF. "
                "Install it with: .venv/bin/python -m pip install boto3"
            )
            return None
        try:
            made = boto3.client(
                "s3",
                endpoint_url=cfg.endpoint,
                aws_access_key_id=cfg.access_key,
                aws_secret_access_key=cfg.secret_key,
                region_name=cfg.region,
                config=Config(
                    # A single clip can be 350 MB and this host is behind a
                    # sandbox proxy; multipart is what keeps that from failing
                    # on one oversized PUT.
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
            made._clipforge_cfg = cfg
            _client_cache = made
            return made
        except Exception as exc:  # noqa: BLE001 - never break startup
            _warn_once(f"could not build an S3 client ({exc}); offload is OFF")
            return None


# ---------------------------------------------------------------- keys


def upload_key(job_id: str, filename: str) -> str:
    """Where a job's SOURCE file lives in the bucket."""
    return f"{config_from_env().prefix}/uploads/{job_id}/{Path(filename).name}"


def output_key(job_id: str, filename: str) -> str:
    """Where a job's RENDERED clips and thumbnails live in the bucket."""
    return f"{config_from_env().prefix}/outputs/{job_id}/{Path(filename).name}"


def key_for_path(path: Path, outputs_dir: Path | None = None) -> str | None:
    """Derive the bucket key for a local media path, or None if it isn't ours.

    Every served artefact lives at `<outputs_dir>/<job_id>/<name>`, so the key
    falls out of the path. Anything outside that shape returns None and is
    simply served from disk exactly as before.
    """
    cfg = config_from_env()
    if not cfg.enabled:
        return None
    root = Path(outputs_dir) if outputs_dir is not None else None
    if root is None:
        return None
    try:
        rel = Path(path).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    parts = rel.parts
    if len(parts) != 2:
        return None
    return f"{cfg.prefix}/outputs/{parts[0]}/{parts[1]}"


# ---------------------------------------------------------------- operations


def upload_file(
    path: Path,
    key: str,
    cfg: ObjectStoreConfig | None = None,
    content_type: str | None = None,
) -> bool:
    """Push a local file. True only if the object is durably in the bucket.

    Uploads to a temporary name and renames, so an interrupted transfer can
    never leave a half-written object that a later pull-through would happily
    serve as a corrupt clip.
    """
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return False
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    ctype = content_type or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    tmp_key = f"{key}.uploading"
    try:
        s3.upload_file(
            str(p), cfg.bucket, tmp_key,
            ExtraArgs={"ContentType": ctype},
        )
        # Promote the finished temp object to its real key. The copy is what
        # makes the final name appear atomically — a reader that sees `key` can
        # rely on the whole file being there.
        s3.copy_object(
            Bucket=cfg.bucket, Key=key,
            CopySource={"Bucket": cfg.bucket, "Key": tmp_key},
            MetadataDirective="REPLACE",
            ContentType=ctype,
        )
        s3.delete_object(Bucket=cfg.bucket, Key=tmp_key)
        return True
    except Exception as exc:  # noqa: BLE001 - a push failure is never fatal
        logger.warning(f"[objectstore] upload of {p.name} failed: {exc}")
        try:
            s3.delete_object(Bucket=cfg.bucket, Key=tmp_key)
        except Exception:  # noqa: BLE001,S110 - best effort cleanup
            pass
        return False


def download_to(key: str, dest: Path, cfg: ObjectStoreConfig | None = None) -> bool:
    """Pull an object back to `dest`. True if the file is there afterwards."""
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return False
    dest = Path(dest)
    tmp: Path | None = None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Download to a sibling tmp and rename, so a concurrent reader never
        # sees a partially written file at the real path.
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.fetching")
        s3.download_file(cfg.bucket, key, str(tmp))
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001 - a miss is a normal outcome
        logger.warning(f"[objectstore] download of {key} failed: {exc}")
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort cleanup
                pass
        return False


def materialize(key: str, dest: Path, cfg: ObjectStoreConfig | None = None) -> bool:
    """Ensure `dest` exists, pulling it from the bucket if it does not.

    The pull-through that keeps every existing code path working: callers keep
    asking for a local file and never learn whether it came from disk.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    if not key:
        return False
    return download_to(key, dest, cfg)


def open_stream(
    key: str,
    range_header: str | None = None,
    cfg: ObjectStoreConfig | None = None,
) -> tuple[int, dict[str, str], Any] | None:
    """Open an object for serving WITHOUT writing it to local disk.

    This is the difference between an archive and an offload. `materialize`
    pulls the bytes back onto the 4 GB sandbox disk, so a reclaimed clip would
    refill the space it just freed on the first playback. Here the bytes go
    from the bucket to the response, which is what lets the library live in the
    bucket and the disk hold only the job that is running.

    Returns `(status, headers, body)` — status is 206 when the browser asked
    for a byte range, so seeking still works. `None` means "not available from
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
        logger.warning(f"[objectstore] stream of {key} failed: {exc}")
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
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return False
    try:
        s3.head_object(Bucket=cfg.bucket, Key=key)
        return True
    except Exception:  # noqa: BLE001 - absence is an answer, not an error
        return False


def delete(key: str, cfg: ObjectStoreConfig | None = None) -> bool:
    """Remove an object. Used when the user deletes a job."""
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return False
    try:
        s3.delete_object(Bucket=cfg.bucket, Key=key)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[objectstore] delete of {key} failed: {exc}")
        return False


def drop_local_after_push(ok: bool, path: Path, cfg: ObjectStoreConfig | None = None) -> bool:
    """Delete the local copy iff the push succeeded and deletion is enabled.

    Order matters: the local bytes are only released once the durable copy is
    confirmed. If this returns False the caller keeps the local file, which is
    always the safe direction.

    The `enabled` re-check is deliberate belt-and-braces: with no bucket
    configured there is no durable copy, so deleting the local one would
    destroy the only bytes that exist — whatever the caller passed as `ok`.
    """
    if not ok:
        return False
    cfg = cfg or config_from_env()
    if not cfg.enabled or not cfg.delete_local:
        return False
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning(f"[objectstore] could not drop the local copy of {path}: {exc}")
        return False


def status() -> dict[str, Any]:
    """A small dict for the boot banner and /api/health."""
    cfg = config_from_env()
    return {
        "enabled": cfg.enabled,
        "bucket": cfg.bucket or None,
        "endpoint": cfg.endpoint or None,
        "prefix": cfg.prefix,
        "deleteLocal": cfg.delete_local,
        "clientReady": client(cfg) is not None if cfg.enabled else False,
    }
