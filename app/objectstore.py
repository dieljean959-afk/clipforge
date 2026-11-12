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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

#: Tests install a fake here so no credential is needed. Checked first, before
#: any env parsing, in `client()`.
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
    #: "path" (works on Filebase/R2/MinIO), "virtual" (bucket.endpoint) or
    #: "auto" (let botocore decide). Filebase accepts path-style, and a
    #: virtual-host style request against a bucket whose DNS is not set up is
    #: the classic silent 400 — so path stays the default and is overridable.
    addressing: str = "path"
    #: The plan the user is paying for, used by the budget guard.
    budget_bytes: int = 5 * 1024**3

    @property
    def enabled(self) -> bool:
        return bool(
            self.endpoint and self.bucket and self.access_key and self.secret_key
        )


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
        return raw.strip().lower() in {"1", "true", "on", "yes"}

    addressing = os.getenv("CLIPFORGE_S3_ADDRESSING", "path").strip().lower()
    budget_gb = os.getenv("CLIPFORGE_BUCKET_BUDGET_GB", "5").strip()
    try:
        budget_bytes = max(1, int(float(budget_gb) * 1024**3))
    except ValueError:
        budget_bytes = 5 * 1024**3
    return ObjectStoreConfig(
        endpoint=os.getenv("CLIPFORGE_S3_ENDPOINT", "").strip(),
        bucket=os.getenv("CLIPFORGE_S3_BUCKET", "").strip(),
        access_key=os.getenv("CLIPFORGE_S3_ACCESS_KEY", "").strip(),
        secret_key=os.getenv("CLIPFORGE_S3_SECRET_KEY", "").strip(),
        region=os.getenv("CLIPFORGE_S3_REGION", "auto").strip() or "auto",
        prefix=os.getenv("CLIPFORGE_S3_PREFIX", "clipforge").strip() or "clipforge",
        delete_local=flag("CLIPFORGE_OFFLOAD_DELETE_LOCAL", True),
        addressing=addressing if addressing in {"path", "virtual", "auto"} else "path",
        budget_bytes=budget_bytes,
    )


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        logger.warning(f"[objectstore] {msg}")


def client(cfg: ObjectStoreConfig | None = None):
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
            import boto3
            from botocore.config import Config
        except ImportError:
            _warn_once(
                "boto3 is not installed — media offload is OFF. Install it with: "
                ".venv/bin/python -m pip install boto3"
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
                    signature_version="s3v4",
                    s3={"addressing_style": cfg.addressing},
                    retries={"max_attempts": 3, "mode": "standard"},
                    # A hung endpoint must surface as "bucket unavailable" in
                    # seconds, not hold a render for a minute per call.
                    connect_timeout=10,
                    read_timeout=60,
                ),
            )
            # Stashed on the client so a config change is detected by identity,
            # not by re-deriving every credential on each call.
            made._clipforge_cfg = cfg
            _client_cache = made
            return made
        except Exception as exc:  # noqa: BLE001 - a bad endpoint must not crash
            _warn_once(f"could not build an S3 client ({exc}); offload is OFF")
            return None


def upload_key(job_id: str, filename: str) -> str:
    """Where a job's SOURCE file lives in the bucket."""
    return f"{config_from_env().prefix}/uploads/{job_id}/{Path(filename).name}"


def output_key(job_id: str, filename: str) -> str:
    """Where a job's RENDERED clips and thumbnails live in the bucket."""
    return f"{config_from_env().prefix}/outputs/{job_id}/{Path(filename).name}"


def key_for_path(path: str | Path, outputs_dir: str | Path | None) -> str | None:
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


def upload_file(
    path: str | Path,
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
    # ONE PUT, streaming the file from disk. Not `upload_file`, and not a
    # tmp-key+copy either: Filebase's free plan rejects any MULTIPART upload of
    # video above 25 MB ("Free accounts are limited to 25MB video uploads"),
    # while the same bytes go through a single PUT unchanged. `upload_file`
    # switches to multipart at 8 MB, so every real clip — and every source
    # section — was hitting that limit. A single PUT is atomic on S3: the key
    # only exists once the whole body has been stored, so the old tmp+copy
    # dance bought nothing anyway.
    try:
        with p.open("rb") as fh:
            s3.put_object(
                Bucket=cfg.bucket, Key=key, Body=fh, ContentType=ctype,
                ContentLength=p.stat().st_size,
            )
        invalidate_usage()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[objectstore] upload of {p.name} failed: {exc}")
        return False


def download_to(key: str, dest: str | Path, cfg: ObjectStoreConfig | None = None) -> bool:
    """Pull an object back to `dest`. True if the file is there afterwards."""
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return False
    dest = Path(dest)
    tmp = None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Per-process temp name: two workers pulling the same object must not
        # write each other's half-download into the same file.
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.fetching")
        s3.download_file(cfg.bucket, key, str(tmp))
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[objectstore] download of {key} failed: {exc}")
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def materialize(key: str, dest: str | Path, cfg: ObjectStoreConfig | None = None) -> bool:
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


def exists(key: str, cfg: ObjectStoreConfig | None = None) -> bool:
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return False
    try:
        s3.head_object(Bucket=cfg.bucket, Key=key)
        return True
    except Exception:  # noqa: BLE001 - a miss is not an error
        return False


def delete(key: str, cfg: ObjectStoreConfig | None = None) -> bool:
    """Remove an object. Used when the user deletes a job."""
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return False
    try:
        s3.delete_object(Bucket=cfg.bucket, Key=key)
        invalidate_usage()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[objectstore] delete of {key} failed: {exc}")
        return False


def open_stream(
    key: str,
    range_header: str | None = None,
    cfg: ObjectStoreConfig | None = None,
) -> tuple[int, dict[str, str], Any] | None:
    """Open an object for serving WITHOUT writing it to local disk.

    This is the difference between an archive and an offload. `materialize`
    pulls the bytes back onto the 4 GB disk, so a reclaimed clip would refill
    the space it just freed on the first playback. Here the bytes go from the
    bucket to the HTTP response, which is what lets the library live in the
    bucket while the disk holds only the batch that is running.

    Returns `(status, headers, body)` — status is 206 when the browser asked for
    a byte range, so seeking still works. `None` means "not available from the
    bucket" and the caller keeps its existing local-file path.
    """
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return None
    params: dict[str, Any] = {"Bucket": cfg.bucket, "Key": key}
    if range_header:
        # Passed through verbatim: a malformed range is the store's to reject,
        # and re-parsing it here would invent a second, weaker implementation.
        params["Range"] = range_header
    try:
        resp = s3.get_object(**params)
    except Exception as exc:  # noqa: BLE001 - a miss falls back to the 404 path
        logger.warning(f"[objectstore] stream of {key} failed: {exc}")
        return None
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": resp.get("ContentType")
        or mimetypes.guess_type(key)[0]
        or "application/octet-stream",
    }
    if resp.get("ContentLength") is not None:
        headers["Content-Length"] = str(resp["ContentLength"])
    status = 200
    if resp.get("ContentRange"):
        headers["Content-Range"] = str(resp["ContentRange"])
        status = 206
    return status, headers, resp["Body"]


def drop_local_after_push(ok: bool, path: str | Path, cfg: ObjectStoreConfig | None = None) -> bool:
    if not ok:
        return False
    cfg = cfg or config_from_env()
    if not (cfg.enabled and cfg.delete_local):
        return False
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning(f"[objectstore] could not drop the local copy of {path}: {exc}")
        return False


# --------------------------------------------------------------------------- #
# verification, listing, budget, presigning, and the real "Test bucket"
# --------------------------------------------------------------------------- #

#: Usage is a full listing, so it is cached for a minute — the dashboard polls
#: health every few seconds and a listing per poll would be silly.
_USAGE_TTL_SEC = 60.0
_usage_cache: tuple[float, int, int] | None = None  # (monotonic, bytes, objects)
_usage_lock = threading.Lock()


def invalidate_usage() -> None:
    """Forget the cached usage figure (after an upload or a delete)."""
    global _usage_cache
    _usage_cache = None


def head(key: str, cfg: ObjectStoreConfig | None = None) -> dict[str, Any] | None:
    """Object metadata, or None when it is not there.

    The difference between "uploaded" and "verified": `upload_file` returning
    True says the PUT was accepted, this says the object is really in the
    bucket with the size we expect.
    """
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return None
    try:
        resp = s3.head_object(Bucket=cfg.bucket, Key=key)
        return {
            "key": key,
            "size": int(resp.get("ContentLength") or 0),
            "contentType": resp.get("ContentType"),
            "etag": str(resp.get("ETag") or "").strip('"'),
            "lastModified": resp.get("LastModified"),
        }
    except Exception:  # noqa: BLE001 - a miss is a legitimate answer
        return None


def verify(key: str, expect_bytes: int, cfg: ObjectStoreConfig | None = None) -> bool:
    """HEAD the object and require the exact byte count. Never a guess."""
    meta = head(key, cfg)
    return bool(meta is not None and int(meta.get("size") or -1) == int(expect_bytes))


def list_prefix(
    prefix: str | None = None, cfg: ObjectStoreConfig | None = None,
    *, strict: bool = False,
) -> list[dict[str, Any]]:
    """Every object under a prefix, with sizes. Paginated: 1000 per call.

    `strict=True` re-raises a listing failure instead of returning the partial
    list. Callers that report a NUMBER use it: a transient S3 error used to
    come back as "0 objects, 0 bytes" and the dashboard rendered that as
    "Bucket 0.00 GB", which reads exactly like the bucket being wiped. A
    failure is not an empty bucket.
    """
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return []
    prefix = f"{cfg.prefix}/" if prefix is None else prefix
    out: list[dict[str, Any]] = []
    token: str | None = None
    try:
        while True:
            kwargs: dict[str, Any] = {"Bucket": cfg.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []) or []:
                out.append({
                    "key": obj.get("Key"),
                    "size": int(obj.get("Size") or 0),
                    "lastModified": obj.get("LastModified"),
                })
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
            if not token:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[objectstore] listing {prefix} failed: {exc}")
        if strict:
            raise
        return out
    return out


def usage(cfg: ObjectStoreConfig | None = None, *, refresh: bool = False) -> dict[str, Any]:
    """Bytes and objects in the bucket, cached for 60 s.

    Summing a listing is the only portable way to know usage: Filebase and R2
    both expose it, but with different endpoints and units, and a listing is
    also what the reconcile/orphan sweep needs.
    """
    global _usage_cache
    cfg = cfg or config_from_env()
    if not cfg.enabled:
        return {"bytes": 0, "objects": 0, "budgetBytes": cfg.budget_bytes, "cached": False}
    with _usage_lock:
        now = time.monotonic()
        if not refresh and _usage_cache is not None and now - _usage_cache[0] < _USAGE_TTL_SEC:
            return {
                "bytes": _usage_cache[1],
                "objects": _usage_cache[2],
                "budgetBytes": cfg.budget_bytes,
                "cached": True,
            }
        try:
            objs = list_prefix(cfg=cfg, strict=True)
        except Exception as exc:  # noqa: BLE001 - a failed listing is not "empty"
            logger.warning(f"[objectstore] usage listing failed: {exc}")
            if _usage_cache is not None:
                # Keep the last real reading rather than flashing a false zero.
                return {
                    "bytes": _usage_cache[1],
                    "objects": _usage_cache[2],
                    "budgetBytes": cfg.budget_bytes,
                    "cached": True,
                    "stale": True,
                }
            return {
                "bytes": 0,
                "objects": 0,
                "budgetBytes": cfg.budget_bytes,
                "cached": False,
                "unknown": True,
            }
        total = sum(o["size"] for o in objs)
        _usage_cache = (now, total, len(objs))
    return {"bytes": total, "objects": len(objs), "budgetBytes": cfg.budget_bytes, "cached": False}


def delete_prefix(prefix: str, cfg: ObjectStoreConfig | None = None) -> int:
    """Delete every object under a prefix. Returns how many were removed."""
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return 0
    objs = list_prefix(prefix, cfg)
    removed = 0
    for i in range(0, len(objs), 1000):
        batch = [{"Key": o["key"]} for o in objs[i : i + 1000]]
        try:
            s3.delete_objects(Bucket=cfg.bucket, Delete={"Objects": batch, "Quiet": True})
            removed += len(batch)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[objectstore] batch delete under {prefix} failed: {exc}")
            for obj in batch:
                if delete(obj["Key"], cfg):
                    removed += 1
    if removed:
        invalidate_usage()
    return removed


def presign(
    key: str,
    *,
    expires: int = 3600,
    download_name: str | None = None,
    cfg: ObjectStoreConfig | None = None,
) -> str | None:
    """A presigned GET URL.

    Presigned rather than proxied so the browser talks to the bucket directly:
    HTTP Range requests then work natively (mobile seeking), the 4 GB box never
    touches the bytes, and a slow viewer cannot occupy a worker slot.
    """
    cfg = cfg or config_from_env()
    s3 = client(cfg)
    if s3 is None or not cfg.enabled:
        return None
    params: dict[str, Any] = {"Bucket": cfg.bucket, "Key": key}
    if download_name:
        safe = download_name.replace('"', "").replace("\\", "").strip() or "clip.mp4"
        params["ResponseContentDisposition"] = f'attachment; filename="{safe}"'
    try:
        return s3.generate_presigned_url("get_object", Params=params, ExpiresIn=expires)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[objectstore] could not presign {key}: {exc}")
        return None


#: Turning a botocore exception into the sentence that actually fixes it. The
#: whole point of "Test bucket" is that a 403 never leaves the user guessing
#: between a bad secret, a wrong region and a bucket that does not exist.
_ERROR_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("SignatureDoesNotMatch",), "403 signature mismatch — the secret access key does not match this access key id"),
    (("InvalidAccessKeyId",), "invalid access key id — it does not exist on this provider"),
    (("NoSuchBucket", "NotFound"), "the bucket does not exist (check the exact name and the endpoint/account)"),
    (("PermanentRedirect", "AuthorizationHeaderMalformed", "IllegalLocationConstraintException"),
     "wrong region — this bucket lives in another region (us-east-1 is the usual answer for Filebase)"),
    (("AccessDenied", "Forbidden"), "403 access denied — this key cannot put/get/delete in this bucket"),
    (("InvalidBucketName",), "the bucket name is not valid for S3"),
    (("EndpointConnectionError", "ConnectTimeout", "ConnectionError"),
     "could not connect to the endpoint (timeout or DNS) — check the endpoint URL"),
    (("ReadTimeout", "RequestTimeout"), "the endpoint timed out — the bucket is reachable but very slow"),
    (("SSLError",), "TLS error talking to the endpoint — check the scheme (https://) and the host"),
    (("InvalidRegion", "InvalidEndpoint"), "the endpoint or region is not one this provider accepts"),
)


def explain_error(exc: BaseException) -> dict[str, str]:
    """(code, message) for a storage failure, in words that name the cause."""
    code = ""
    detail = str(exc)
    resp = getattr(exc, "response", None) or {}
    err = (resp.get("Error") or {}) if isinstance(resp, dict) else {}
    code = str(err.get("Code") or "") or type(exc).__name__
    status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode") if isinstance(resp, dict) else None
    message = str(err.get("Message") or detail)[:300]
    for codes, hint in _ERROR_HINTS:
        if code in codes:
            return {"code": code, "status": str(status or ""), "message": message, "hint": hint}
    return {"code": code, "status": str(status or ""), "message": message, "hint": ""}


def test_roundtrip(cfg: ObjectStoreConfig | None = None) -> dict[str, Any]:
    """A REAL put → head → ranged get → delete against the configured bucket.

    Configuration that is merely complete is not configuration that works. The
    access key id field rendered empty while the secret said "saved", so
    nothing in the UI could tell a correct pair from a half-saved one. This does
    what the pipeline does, on one tiny object, and reports the first step that
    failed with the provider's own error code.
    """
    cfg = cfg or config_from_env()
    started = time.monotonic()
    out: dict[str, Any] = {
        "ok": False,
        "bucket": cfg.bucket or None,
        "endpoint": cfg.endpoint or None,
        "region": cfg.region,
        "addressing": cfg.addressing,
        "steps": [],
    }
    missing = [
        label
        for label, value in (
            ("endpoint", cfg.endpoint),
            ("bucket", cfg.bucket),
            ("access key id", cfg.access_key),
            ("secret access key", cfg.secret_key),
        )
        if not value
    ]
    if missing:
        out["error"] = {"code": "IncompleteConfig", "message": f"missing: {', '.join(missing)}"}
        out["elapsedMs"] = int((time.monotonic() - started) * 1000)
        return out
    s3 = client(cfg)
    if s3 is None:
        out["error"] = {"code": "NoClient", "message": "boto3 could not build a client — check the endpoint URL"}
        out["elapsedMs"] = int((time.monotonic() - started) * 1000)
        return out

    payload = b"clipforge bucket test\n"
    key = f"{cfg.prefix}/_healthcheck/{int(time.time())}-{os.getpid()}.txt"

    def step(name: str, fn: Any) -> bool:
        t0 = time.monotonic()
        try:
            fn()
            out["steps"].append({"step": name, "ok": True, "ms": int((time.monotonic() - t0) * 1000)})
            return True
        except BaseException as exc:  # noqa: BLE001 - the whole point is reporting it
            info = explain_error(exc)
            out["steps"].append({"step": name, "ok": False, **info, "ms": int((time.monotonic() - t0) * 1000)})
            out["error"] = info
            return False

    if not step("put", lambda: s3.put_object(Bucket=cfg.bucket, Key=key, Body=payload, ContentType="text/plain")):
        out["elapsedMs"] = int((time.monotonic() - started) * 1000)
        return out
    try:
        meta = s3.head_object(Bucket=cfg.bucket, Key=key)
        size_ok = int(meta.get("ContentLength") or -1) == len(payload)
        out["steps"].append({"step": "head", "ok": size_ok, "ms": 0, "bytes": int(meta.get("ContentLength") or 0)})
        if not size_ok:
            out["error"] = {"code": "SizeMismatch", "message": "the object came back with a different size"}
            s3.delete_object(Bucket=cfg.bucket, Key=key)
            out["elapsedMs"] = int((time.monotonic() - started) * 1000)
            return out
    except BaseException as exc:  # noqa: BLE001
        info = explain_error(exc)
        out["steps"].append({"step": "head", "ok": False, **info, "ms": 0})
        out["error"] = info
        out["elapsedMs"] = int((time.monotonic() - started) * 1000)
        return out

    def ranged_get() -> None:
        resp = s3.get_object(Bucket=cfg.bucket, Key=key, Range="bytes=0-3")
        body = resp["Body"].read()
        if body != payload[:4]:
            raise RuntimeError(f"ranged GET returned {body!r}, expected {payload[:4]!r}")

    if not step("ranged get (HTTP Range)", ranged_get):
        s3.delete_object(Bucket=cfg.bucket, Key=key)
        out["elapsedMs"] = int((time.monotonic() - started) * 1000)
        return out
    if not step("delete", lambda: s3.delete_object(Bucket=cfg.bucket, Key=key)):
        out["elapsedMs"] = int((time.monotonic() - started) * 1000)
        return out
    out["ok"] = True
    out["elapsedMs"] = int((time.monotonic() - started) * 1000)
    invalidate_usage()
    return out


def configured(cfg: ObjectStoreConfig | None = None) -> bool:
    """True when an install actually has a bucket wired (not just defaults)."""
    cfg = cfg or config_from_env()
    return bool(cfg.enabled and cfg.bucket)


def reachable(cfg: ObjectStoreConfig | None = None) -> bool:
    """Cheap liveness probe: can we list one key right now?

    Retention and budget pruning DELETE things, so the automatic sweep asks
    this first — a bucket that is down must never turn "expired" into "gone".
    """
    cfg = cfg or config_from_env()
    if not configured(cfg):
        return False
    s3 = client(cfg)
    if s3 is None:
        return False
    try:
        s3.list_objects_v2(Bucket=cfg.bucket, Prefix=f"{cfg.prefix}/", MaxKeys=1)
        return True
    except Exception as exc:  # noqa: BLE001
        from loguru import logger as _logger

        try:
            why = explain_error(exc)
        except Exception:  # noqa: BLE001 - explaining must never mask the probe
            why = f"{type(exc).__name__}: {exc}"
        _logger.warning(f"[objectstore] not reachable: {why}")
        return False


def status() -> dict[str, Any]:
    """A small dict for the boot banner, /api/health and the bucket dialog.

    `hasAccessKey`/`hasSecret` are separate on purpose: the dialog showed
    "saved" for the secret while the id field looked empty, which is
    indistinguishable from a half-saved credential pair.
    """
    cfg = config_from_env()
    use = usage(cfg)
    return {
        "enabled": cfg.enabled,
        "bucket": cfg.bucket or None,
        "endpoint": cfg.endpoint or None,
        "prefix": cfg.prefix,
        "region": cfg.region,
        "addressing": cfg.addressing,
        "deleteLocal": cfg.delete_local,
        "clientReady": cfg.enabled and client(cfg) is not None,
        "hasEndpoint": bool(cfg.endpoint),
        "hasBucket": bool(cfg.bucket),
        "hasAccessKey": bool(cfg.access_key),
        "accessKeyFirst4": cfg.access_key[:4],
        "accessKeyLength": len(cfg.access_key),
        "hasSecret": bool(cfg.secret_key),
        "secretLength": len(cfg.secret_key),
        "usageBytes": use["bytes"],
        "usageObjects": use["objects"],
        "usageUnknown": bool(use.get("unknown")),
        "usageStale": bool(use.get("stale")),
        "budgetBytes": cfg.budget_bytes,
    }
