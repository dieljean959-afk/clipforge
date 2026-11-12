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
    if log is not None:
        log(
            f"[models] {name} restored from the bucket ({total / 1048576:.0f} MB on disk "
            "for this stage only)"
        )
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
    for repo in root.glob(f"models--*faster-whisper-{name}"):
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
    if os.getenv("CLIPFORGE_KEEP_MODELS", "").strip() not in ("", "0", "false"):
        return {"ok": False, "reason": "CLIPFORGE_KEEP_MODELS set"}
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
                repo = line.strip().split("/", 1)[0]
                if repo and (root / repo).is_dir():
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
    return [p for p in root.glob(f"models--*faster-whisper-{name}") if p.is_dir()]


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


def release_all(log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Drop every model this project downloaded. Bucket copies stay."""
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
    out = {name: release(name, log) for name in sorted(names)}
    if root.is_dir():
        for leftover in root.iterdir():
            if leftover.is_dir():
                shutil.rmtree(leftover, ignore_errors=True)
    try:
        root.rmdir()
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
