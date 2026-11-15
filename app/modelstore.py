"""Whisper weights live in the bucket, not on the local disk.

A decoder needs its weights as real files for as long as it decodes, so the
model is streamed in from the bucket for that stage and deleted again once the
batch is over. The bucket keeps the permanent copy; the disk only ever holds the
one model actually in use, and holds nothing between batches.

    models/<name>/model.bin|config.json|tokenizer.json|vocabulary.txt|...

`resolve()` is the single entry point: give it a faster-whisper name and it
returns a local directory to load from (restoring it from the bucket first when
needed) or the bare name when no copy is reachable, which makes faster-whisper
fall back to its normal HuggingFace download. `adopt()` then pushes whatever
that download produced into the bucket so the next batch does not pay for it
again, and `release()` clears both copies from disk.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app import objectstore

#: Files faster-whisper reads out of a local model directory. `model.bin` and
#: `config.json` are the two that make a directory loadable; the others only
#: matter for tokenisation and are uploaded when the snapshot has them.
REQUIRED = ("model.bin", "config.json")

PREFIX = "models"

#: A prepared model is marked for reuse; the bucket remains the durable copy.
#: The marker keeps existing explicit `release()` callers (and their tests)
#: honest: without a prepare, a release still frees the transient copy.
PERSIST_MARKER = ".prepared"
DEFAULT_KEEP = ()

#: A sidecar written after the first verified restore.  It lets every later
#: job prove the cached model.bin is the same bytes instead of trusting size.
#: No repository-specific hash is hardcoded here: the model source may publish
#: a new revision, and a fabricated constant would reject a good download.
VERIFY_SIDECAR = "model.bin.sha256"

_lock = threading.Lock()


def _root() -> Path:
    from app.config import load_config  # noqa: PLC0415 - keeps import cost off module load

    return load_config(quiet=True).app.paths.data_dir / "models"


def local_dir(name: str) -> Path:
    return _root() / str(name).replace("/", "_")


def _hf_cache_root() -> Path:
    explicit = os.getenv("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def ready(name: str) -> bool:
    """True when this name is loadable straight off the local disk."""
    path = local_dir(name)
    return all((path / f).is_file() and (path / f).stat().st_size > 0 for f in REQUIRED)


def _sha256(path: Path) -> str | None:
    import hashlib

    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def verify(name: str) -> tuple[bool, str]:
    """Verify a local copy once after restore; cheap for a cached model."""
    path = local_dir(name) / "model.bin"
    if not path.is_file():
        return False, "model.bin missing"
    sidecar = path.with_name(VERIFY_SIDECAR)
    actual = _sha256(path)
    if actual is None:
        return False, "model.bin unreadable"
    try:
        expected = sidecar.read_text(encoding="utf-8").split()[0].strip()
    except (OSError, IndexError):
        expected = ""
    if expected and actual != expected:
        return False, "sha256 mismatch — discarded instead of loaded"
    if not expected:
        try:
            sidecar.write_text(actual + "  model.bin\n", encoding="utf-8")
        except OSError:
            pass
    return True, "sha256 " + ("verified" if expected else "recorded")


def _key(name: str, filename: str) -> str:

    prefix = objectstore.config_from_env().prefix
    return f"{prefix}/{PREFIX}/{name}/{filename}" if prefix else f"{PREFIX}/{name}/{filename}"


def _bucket_files(name: str) -> list[dict[str, Any]]:

    prefix = objectstore.config_from_env().prefix
    base = f"{prefix}/{PREFIX}/{name}/" if prefix else f"{PREFIX}/{name}/"
    return objectstore.list_prefix(base)


def _restore(name: str, log: Callable[[str], None] | None) -> bool:
    """Pull `name` from the bucket into its local directory. True if loadable."""
    if not objectstore.configured():
        return False
    files = _bucket_files(name)
    if not any(str(f.get("key") or "").endswith("model.bin") for f in files):
        return False
    dest = local_dir(name)
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for item in files:
        key = str(item.get("key") or "")
        filename = key.rsplit("/", 1)[-1]
        if not filename:
            continue
        target = dest / filename
        size = int(item.get("size") or 0)
        if target.is_file() and target.stat().st_size == size:
            total += size
            continue
        if not objectstore.download_to(key, target):
            logger.warning(f"[models] could not restore {key} from the bucket")
            return False
        total += size
    ok = ready(name)
    if ok:
        verified, detail = verify(name)
        if verified:
            if log is not None:
                log(
                    f"[models] {name} restored from the bucket "
                    f"({total / 1048576:.0f} MB, {detail}) — kept for reuse"
                )
        else:
            logger.warning(f"[models] {name} failed verification: {detail}")
            shutil.rmtree(dest, ignore_errors=True)
            ok = False
    return ok


def resolve(name: str, log: Callable[[str], None] | None = None) -> str:
    """A local model path for `name`, else `name` itself (HuggingFace fallback)."""
    name = str(name or "small").strip() or "small"
    if ready(name):
        return str(local_dir(name))
    try:
        if _restore(name, log):
            # Keep the bucket copy authoritative: whatever the download produced
            # is worth having there too, if the restore was partial.
            _adopt_async(name, log)
            return str(local_dir(name))
        snap = _snapshot_dir(name)
        if snap is not None:
            return str(snap)
    except Exception as exc:  # noqa: BLE001 - a model must still be obtainable
        logger.warning(f"[models] bucket restore of {name} failed: {exc}")
    _adopt_async(name, log)
    return name


def _snapshot_dir(name: str) -> Path | None:
    """The HuggingFace snapshot for this faster-whisper model, if one exists."""
    root = _hf_cache_root()
    if not root.is_dir():
        return None
    newest: tuple[float, Path] | None = None
    # `small` lives in models--Systran--faster-whisper-small, while
    # `distil-medium.en` lives in models--Systran--faster-distil-whisper-medium.en;
    # a single fixed '*faster-whisper-<name>' glob missed the distil family.
    repos = list(root.glob(f"models--*faster-whisper-{name}"))
    repos += list(root.glob(f"models--*faster-distil-whisper-{name}"))
    for repo in repos:
        for snap in repo.glob("snapshots/*"):
            if not (snap / "model.bin").is_file():
                continue
            try:
                stamp = (snap / "model.bin").stat().st_mtime
            except OSError:
                continue
            if newest is None or stamp > newest[0]:
                newest = (stamp, snap)
    return newest[1] if newest else None


def _wait_snapshot(name: str, wait_s: float) -> Path | None:
    """The finished snapshot for `name`, waiting for one still downloading.

    `resolve()` starts adoption as soon as the decode begins, and faster-whisper
    writes its files while it loads — so the model.bin size has to hold still
    across two looks before the copy can be trusted as complete.
    """
    deadline = time.monotonic() + max(0.0, wait_s)
    snap: Path | None = None
    last = -1
    while True:
        found = _snapshot_dir(name)
        if found is not None:
            try:
                size = (found / "model.bin").stat().st_size
            except OSError:
                size = -1
            if found == snap and size == last and size > 0:
                return found
            snap, last = found, size
        elif snap is not None:  # disappeared mid-wait: start over
            snap, last = None, -1
        if time.monotonic() >= deadline:
            return snap
        time.sleep(3.0)


def adopt(
    name: str, log: Callable[[str], None] | None = None, *, wait_s: float = 0.0,
) -> dict[str, Any]:
    """Copy a locally downloaded model into the bucket. Nothing local is removed."""
    name = str(name or "").strip()
    snap = _wait_snapshot(name, wait_s) if name else None
    if snap is None:
        return {"ok": False, "reason": "no local snapshot"}
    if not objectstore.configured():
        return {"ok": False, "reason": "bucket not configured"}
    existing = {str(f.get("key")): int(f.get("size") or 0) for f in _bucket_files(name)}
    uploaded = 0
    total = 0
    for src in sorted(snap.iterdir()):
        if not src.is_file() or src.name.startswith("."):
            continue
        try:
            size = src.stat().st_size
        except OSError:
            continue
        key = _key(name, src.name)
        total += size
        if existing.get(key) == size:
            continue
        if objectstore.upload_file(src, key):
            uploaded += 1
    if log is not None and uploaded:
        log(f"[models] {name} stored in the bucket ({total / 1048576:.0f} MB, {uploaded} files)")
    return {"ok": True, "uploaded": uploaded, "bytes": total, "path": str(snap)}


_ADOPTS: dict[str, threading.Thread] = {}


def _adopt_async(name: str, log: Callable[[str], None] | None) -> None:
    """Push a fresh download to the bucket without blocking the decode."""

    def _run() -> None:
        try:
            adopt(name, log, wait_s=3600.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[models] could not store {name} in the bucket: {exc}")

    thread = threading.Thread(target=_run, name=f"models-adopt-{name}", daemon=True)
    _ADOPTS[name] = thread
    thread.start()


def _late_release(name: str, log: Callable[[str], None] | None) -> None:
    """Remove a model as soon as its bucket upload lands, off the job thread.

    `release` refuses to delete without a verified bucket copy, so a model whose
    462 MB upload is still in flight would otherwise sit on the disk until the
    next batch. This waits for that upload and then drops it.
    """

    def _run() -> None:
        thread = _ADOPTS.get(name)
        if thread is not None:
            thread.join(timeout=1800.0)
        try:
            release(name, log)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[models] late release of {name} failed: {exc}")

    threading.Thread(target=_run, name=f"models-release-{name}", daemon=True).start()


def release(name: str, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Delete this model from the disk once the bucket holds a full copy."""
    name = str(name or "").strip()
    if not name:
        return {"ok": False, "reason": "no name"}
    keep = {
        item.strip()
        for item in (os.getenv("CLIPFORGE_KEEP_MODELS") or "").split(",")
        if item.strip()
    }
    try:
        marker = _root() / PERSIST_MARKER
        if marker.is_file():
            keep.update(x.strip() for x in marker.read_text(encoding="utf-8").split() if x.strip())
    except OSError:
        pass
    if name in keep:
        return {"ok": False, "reason": "kept for reuse (prepared model)"}
    # THE ACTIVE MODEL IS NOT TRANSIENT. A copy restored from the bucket for a
    # run (resolve()) never wrote the persist marker, so the next job start's
    # release_all() deleted it — a silent 464 MB re-download on every batch and
    # a direct contradiction of "Keep active model local". The setting is the
    # authority here, not the marker.
    try:
        active, _note = active_model_name()
        from app.config import load_config  # noqa: PLC0415

        if name == active and bool(load_config(quiet=True).whisper.keep_local):
            return {"ok": False, "reason": "active model kept local"}
    except Exception as exc:  # noqa: BLE001 - never block a release decision
        logger.debug(f"[models] keep-local check skipped for {name}: {exc}")
    frozen = {"ok": False, "reason": "bucket has no full copy"}
    with _lock:
        try:
            sizes = {str(f.get("key")).rsplit("/", 1)[-1]: int(f.get("size") or 0)
                     for f in _bucket_files(name)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"bucket listing failed: {exc}"}
        # Verify against whatever local copy is about to be deleted — the HF
        # snapshot before adoption, the transient directory after it — so a
        # half-uploaded bucket copy can never cost us the only good one.
        need: dict[str, int] = {}
        here = [local_dir(name), _snapshot_dir(name)]
        for src_dir in here:
            if src_dir is None or not src_dir.is_dir():
                continue
            for f in REQUIRED:
                if f not in need and (src_dir / f).is_file():
                    need[f] = (src_dir / f).stat().st_size
        if not need or any(sizes.get(f) != size for f, size in need.items()):
            frozen = {"ok": False, "reason": "bucket copy incomplete", "have": sizes,
                      "need": need}
        else:
            frozen = {"ok": True, "reason": "bucket copy verified", "need": need}

    freed = 0
    if frozen["ok"]:
        for path in (local_dir(name), *[p for p in _hf_repos(name)]):
            freed += _tree_bytes(path)
            shutil.rmtree(path, ignore_errors=True)
        freed += _prune_orphan_blobs()
        if log is not None:
            log(f"[models] {name} removed from disk ({freed / 1048576:.0f} MB) — kept in the bucket")
    return {**frozen, "freed": freed}


def _prune_orphan_blobs() -> int:
    """Drop cached weights whose repository directory is gone.

    Newer huggingface_hub keeps one shared `blobs/` tree with a `.refs` sidecar
    per file, so deleting a repo directory leaves the bytes behind: 462 MB of a
    released model once survived its own release that way.
    """
    freed = 0
    root = _hf_cache_root()
    blobs = root / "blobs"
    if not blobs.is_dir():
        return 0
    for path in sorted(blobs.glob("*/*")):
        if not path.is_file() or path.suffix in {".refs", ".lock"}:
            continue
        refs = path.with_name(path.name + ".refs")
        keep = False
        try:
            for line in refs.read_text(encoding="utf-8").splitlines():
                # A ref line names the blob it belongs to, e.g.
                # `models--repo/blobs/<sha>`.  Checking only that the repo
                # DIRECTORY exists kept a 753 MB orphan after a failed model
                # download: the repo survived, the referenced blob did not.
                target = root / line.strip()
                if line.strip() and target.is_file():
                    keep = True
                    break
        except OSError:
            keep = False
        if keep:
            continue
        # A download in flight has no refs yet; never race it.
        try:
            if not refs.exists() and time.time() - path.stat().st_mtime < 600:
                continue
        except OSError:
            continue
        try:
            freed += path.stat().st_size
            path.unlink()
            refs.unlink(missing_ok=True)
            path.with_name(path.name + ".lock").unlink(missing_ok=True)
        except OSError:
            pass
    return freed


def _hf_repos(name: str) -> list[Path]:
    root = _hf_cache_root()
    if not root.is_dir():
        return []
    repos = list(root.glob(f"models--*faster-whisper-{name}"))
    repos += list(root.glob(f"models--*faster-distil-whisper-{name}"))
    return [p for p in repos if p.is_dir()]


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                continue
    return total


def _unusable(name: str) -> bool:
    """True when this host can never load the model, so holding it is waste.

    The ladder refuses models that do not fit the memory cap (`fits`), which is
    how a 756 MB medium once sat on a 4 GB disk it could never be decoded from.
    """
    try:
        from app import asr_quality  # noqa: PLC0415

        return not asr_quality.fits(name)
    except Exception:  # noqa: BLE001 - unknown means keep
        return False


def active_model_name() -> tuple[str, str]:
    """The model the CURRENT configuration will load, as (name, note).

    Reads the same resolver the pipeline uses, so the Reset dialog protects the
    model a real job would actually need — not a name hard-coded here. Falls
    back to the configured value when the resolver cannot answer (a missing
    optional dependency is not a reason to delete the wrong weights).
    """
    try:
        from app.config import load_config
        from app.hostlimits import resolved_model

        settings = load_config(quiet=True)
        name, note = resolved_model(settings.whisper.model)
        return str(name or settings.whisper.model), str(note or "")
    except Exception:  # noqa: BLE001 - accounting must never crash the dialog
        try:
            from app.config import load_config

            return str(load_config(quiet=True).whisper.model), ""
        except Exception:
            return "", ""


def active_local_bytes(name: str) -> int:
    """Bytes on disk for one model, counting BOTH places it can live."""
    return _tree_bytes(local_dir(name)) + sum(_tree_bytes(p) for p in _hf_repos(name))


def releasable(active: str | None = None, *, keep_active_local: bool = True) -> dict[str, Any]:
    """What a release pass can ACTUALLY free, and why the rest is kept.

    The Reset dialog used to promise `sum(status().localBytes)` — every byte under
    data/models — while the apply path only releases what `release_all()` dares
    to. On this host that promised 2263 MB and delivered 2.1 MB (the leftover
    metadata), because the big item was an ORPHAN HuggingFace blob that belongs
    to no model directory and was therefore invisible to per-model accounting.

    So both numbers come from this one function: the dialog's preview and the
    apply's reason list. A byte is counted as freeable only when it would really
    go — unused model (re-downloadable, no bucket check needed), a disabled
    "keep active model local" with a verified bucket copy, or an orphan blob.
    """
    active = (active or "").strip()
    freeable = 0
    items: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    cache = _hf_cache_root()

    names: set[str] = set()
    root = _root()
    if root.is_dir():
        names.update(p.name for p in root.iterdir() if p.is_dir())
    if cache.is_dir():
        for repo in cache.glob("models--*faster-whisper-*"):
            names.add(repo.name.split("faster-whisper-", 1)[-1])

    for name in sorted(names):
        # local_dir + every HF repo for the same model. Counted ONCE: adding the
        # local dir twice was a 2x over-report in the preview, which is the same
        # class of lie as the 2263 MB promise.
        size = active_local_bytes(name)
        if size <= 0:
            continue
        if name == active:
            if keep_active_local:
                kept.append({
                    "what": f"model {name}",
                    "mb": round(size / 1048576, 1),
                    "why": "active model kept local (setting: Keep active model local)",
                })
                continue
            ok, why = verify(name)
            if not ok:
                kept.append({
                    "what": f"model {name}",
                    "mb": round(size / 1048576, 1),
                    "why": f"active model has no verified bucket copy ({why})",
                })
                continue
            freeable += size
            items.append({"what": f"model {name}", "mb": round(size / 1048576, 1),
                          "why": "verified in the bucket"})
            continue
        if _unusable(name):
            freeable += size
            items.append({"what": f"model {name}", "mb": round(size / 1048576, 1),
                          "why": "cannot load on this host; re-downloadable"})
            continue
        # Unused model: re-downloadable, so no bucket verification is required.
        freeable += size
        items.append({"what": f"model {name}", "mb": round(size / 1048576, 1),
                      "why": "unused by the current configuration; re-downloadable"})

    # Orphan blobs (bytes with no repo pointing at them). These are pure waste
    # and are what the old accounting missed entirely.
    try:
        orphans = _orphan_bytes()
    except Exception:  # noqa: BLE001 - accounting must not fail the dialog
        orphans = 0
    if orphans:
        freeable += orphans
        items.append({"what": "orphan model blobs", "mb": round(orphans / 1048576, 1),
                      "why": "no model directory references them; re-downloadable"})

    return {
        "bytes": freeable,
        "mb": round(freeable / 1048576, 1),
        "items": items,
        "kept": kept,
    }


def _orphan_bytes() -> int:
    """Bytes of HF blobs that no surviving repo/snapshot references."""
    cache = _hf_cache_root()
    if not cache.is_dir():
        return 0
    total = 0
    for blob_root in cache.rglob("blobs"):
        if not blob_root.is_dir():
            continue
        referenced: set[str] = set()
        for ref in blob_root.parent.rglob("*"):
            if ref.is_symlink():
                try:
                    referenced.add(str(ref.resolve()))
                except OSError:
                    continue
        for blob in blob_root.rglob("*"):
            if blob.is_file() and not blob.is_symlink() and str(blob) not in referenced:
                try:
                    total += blob.stat().st_size
                except OSError:
                    continue
    return total


def release_all(log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Drop transient models; the configured small copy stays on disk."""
    names: set[str] = set()
    root = _root()
    if root.is_dir():
        names.update(p.name for p in root.iterdir() if p.is_dir())
    cache = _hf_cache_root()
    if cache.is_dir():
        for repo in cache.glob("models--*faster-whisper-*"):
            names.add(repo.name.split("faster-whisper-", 1)[-1])
    # A model that cannot fit in this host's memory will never decode here, so
    # it is dropped even without a bucket copy: HuggingFace remains its source.
    wasted = 0
    for name in sorted(names):
        if not _unusable(name):
            continue
        for path in _hf_repos(name):
            wasted += _tree_bytes(path)
            shutil.rmtree(path, ignore_errors=True)
        local = local_dir(name)
        if local.is_dir():
            wasted += _tree_bytes(local)
            shutil.rmtree(local, ignore_errors=True)
        names.discard(name)
    if wasted and log is not None:
        log(f"[models] dropped {wasted / 1048576:.0f} MB of weights this host cannot load")
    wasted += _prune_orphan_blobs()
    # A model downloaded by the HuggingFace fallback (the memory ladder can
    # choose distil-medium.en even when config says small) used to stay in the
    # cache forever because `release` refuses to delete without a bucket copy.
    # Adopt that snapshot first, then release the local bytes.  If the bucket
    # is unavailable, release still refuses and the copy is kept.
    for name in sorted(names):
        try:
            if _snapshot_dir(name) is None:
                continue
            have = {
                str(f.get("key")).rsplit("/", 1)[-1]
                for f in _bucket_files(name)
            }
            if "model.bin" not in have:
                adopt(name, log, wait_s=0.2)
        except Exception as exc:  # noqa: BLE001 - cleanup must not fail a job
            logger.debug(f"[models] bucket adoption before release failed for {name}: {exc}")
    out = {name: release(name, log) for name in sorted(names)}
    if root.is_dir():
        # Never remove a kept copy here: `release()` above already decided
        # which directories are transient, and this sweep used to delete the
        # persistent model right after the keep check allowed it.
        try:
            root.rmdir()  # succeeds only when it is genuinely empty
        except OSError:
            pass
    # Uploads still in flight: hand them to a waiter instead of leaving the
    # weights on the disk for the rest of the batch.
    pending = [
        n for n, v in out.items()
        if not v.get("ok") and (_ADOPTS.get(n) is not None and _ADOPTS[n].is_alive())
    ]
    for n in pending:
        _late_release(n, log)
    if log is not None and pending:
        log(f"[models] {len(pending)} model(s) still uploading — removed from the disk when it lands")
    freed = wasted + sum(int(v.get("freed") or 0) for v in out.values())
    if log is not None and freed:
        log(f"[models] {freed / 1048576:.0f} MB of model weight released from the disk")
    return {"released": out, "freed": freed, "unusable": wasted, "pending": pending}


def prepare(
    name: str | None = None, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Idle-time model preparation with a disk preflight.

    This is the only deliberate download path.  It never runs inside a chunk
    or a decode: the caller invokes it from the dashboard/tool while idle.  If
    there is not enough free space for the weights plus headroom, it reports
    that instead of filling the disk and failing midway through a job.
    """
    import shutil as _shutil

    from app.config import load_config  # noqa: PLC0415

    settings = load_config(quiet=True)
    wanted = str(name or settings.whisper.model or "small").strip() or "small"
    data_dir = Path(settings.app.paths.data_dir)
    free = _shutil.disk_usage(data_dir).free
    # The measured small weights are ~464 MB; keep 300 MB for ffmpeg/scratch.
    # Also preserve the owner's 2000 MB floor: downloading while it would leave
    # less than that is explicitly reported as blocked rather than filling the
    # disk and failing halfway through a later stage.
    need = 500 * 1024 * 1024
    floor = 2000 * 1024 * 1024
    if free < need + floor:
        return {
            "ok": False,
            "model": wanted,
            "state": "blocked",
            "reason": (
                f"only {free / 1048576:.0f} MB free; downloading {wanted} would "
                "fall below the 2000 MB disk floor, so it stays bucket-only"
            ),
        }
    started = time.time()
    path = resolve(wanted, log)
    ok = ready(wanted)
    verified, detail = verify(wanted) if ok else (False, "not loadable")
    if ok and verified:
        try:
            marker = data_dir / "models" / PERSIST_MARKER
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(wanted + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning(f"[models] could not write reuse marker: {exc}")
    result = {
        "ok": bool(ok and verified),
        "model": wanted,
        "state": "ready" if ok and verified else "failed",
        "path": path,
        "reason": detail,
        "wallSec": round(time.time() - started, 1),
        "freeMb": round(free / 1048576, 1),
    }
    if log is not None:
        log(f"[models] prepare {wanted}: {result['state']} ({detail})")
    return result


def status() -> dict[str, Any]:
    """What each known model costs locally, and whether the bucket has it."""
    out: dict[str, Any] = {"root": str(_root()), "models": {}}
    names = set()
    if _root().is_dir():
        names.update(p.name for p in _root().iterdir() if p.is_dir())
    for name in sorted(names):
        out["models"][name] = {
            "localBytes": _tree_bytes(local_dir(name)),
            "loadable": ready(name),
            "huggingfaceBytes": sum(_tree_bytes(p) for p in _hf_repos(name)),
        }
    return out


__all__ = ["adopt", "local_dir", "ready", "release", "release_all", "resolve", "status"]
