"""One honest list of every model the app can load.

This is deliberately read-only and cheap enough for the health card: it walks
the model directories and the installed MediaPipe package, but never imports
MediaPipe into the API process and never loads Whisper. The Doctor tool calls
:func:`audit` for hashes and the smoke tests live in ``tools/model_audit.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any


def _hash_file(path: Path, *, read: int = 1 << 20) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(read)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _mediapipe_asset(name: str) -> Path | None:
    """Locate a tflite graph without importing the 150 MB MediaPipe package."""
    try:
        spec = importlib.util.find_spec("mediapipe")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    root = Path(spec.origin).parent
    candidate = root / "modules" / "face_detection" / name
    return candidate if candidate.is_file() else None


def whisper_models(settings: Any, *, hash_files: bool = False) -> list[dict[str, Any]]:
    from app import asr, modelstore  # noqa: PLC0415 - keep the API import light

    info = asr.model_status(settings)
    found: list[dict[str, Any]] = []
    configured = str(info.get("model") or "")
    for name in ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"):
        local = modelstore.local_dir(name)
        files = sorted(p for p in local.rglob("*") if p.is_file()) if local.is_dir() else []
        if not files and name != configured:
            continue
        size = sum(_size(p) for p in files)
        model_bin = local / "model.bin"
        vocabulary = local / "vocabulary.txt"
        loadable = model_bin.is_file() and (local / "config.json").is_file()
        if loadable and not vocabulary.is_file():
            state, reason = "failed", "missing vocabulary.txt"
        elif loadable:
            state, reason = "ready", "loads from local disk"
        elif info.get("bucketPresent") and name == configured:
            state, reason = "bucket-only", "restored from the bucket per batch"
            # The local hygiene sweep releases idle weights; report the real
            # bucket size instead of a misleading 0.0 MB.
            size = int(round(float(info.get("sizeMb") or 0.0) * 1048576))
        else:
            state, reason = "missing", "no local weights and no bucket copy"
        entry: dict[str, Any] = {
            "id": f"faster-whisper/{name}",
            "role": "transcription",
            "name": name,
            "state": state,
            "reason": reason,
            "path": str(local),
            "sizeBytes": size,
            "sizeMb": round(size / 1048576, 1) if size else 0.0,
            "loadable": bool(loadable),
            "configured": name == configured,
            "threads": info.get("threads"),
            "compute": info.get("computeType"),
        }
        if hash_files and model_bin.is_file():
            entry["sha256"] = _hash_file(model_bin)
        found.append(entry)
    return found


def face_models(*, hash_files: bool = False) -> list[dict[str, Any]]:
    tflite = _mediapipe_asset("face_detection_full_range_sparse.tflite")
    graph = _mediapipe_asset("face_detection_full_range_cpu.binarypb")
    if tflite is None and graph is None:
        return [{
            "id": "mediapipe/face-detection-full-range",
            "role": "reframe",
            "name": "face_detection_full_range_sparse.tflite",
            "state": "missing",
            "reason": "MediaPipe package not installed; reframe will use centred FIT fallback",
            "path": None,
            "sizeBytes": 0,
            "sizeMb": 0.0,
            "threads": 1,
        }]
    entry: dict[str, Any] = {
        "id": "mediapipe/face-detection-full-range",
        "role": "reframe",
        "name": tflite.name if tflite else "face_detection_full_range_cpu.binarypb",
        "state": "ready" if tflite and graph else "failed",
        "reason": "graph + tflite present" if tflite and graph else "graph or tflite missing",
        "path": str(tflite or graph),
        "sizeBytes": _size(tflite) if tflite else _size(graph) if graph else 0,
        "sizeMb": round((_size(tflite) if tflite else _size(graph) if graph else 0) / 1048576, 2),
        "threads": int(os.getenv("ORT_NUM_THREADS", "1") or 1),
    }
    if hash_files and tflite:
        entry["sha256"] = _hash_file(tflite)
    return [entry]


def optional_models(settings: Any, *, hash_files: bool = False) -> list[dict[str, Any]]:
    """Models that are not used by the default clip path but can be loaded."""
    from app import ctc_align  # noqa: PLC0415

    root = Path(settings.app.paths.data_dir) / "models"
    onnx = ctc_align.model_dir(root) / "model_quantized.onnx"
    entry: dict[str, Any] = {
        "id": "wav2vec2-base-960h/onnx",
        "role": "forced-alignment (uploaded subtitles only)",
        "name": "model_quantized.onnx",
        # Optional by design: forced alignment is only used for uploaded
        # subtitles, and the normal Whisper path never loads this graph.  A
        # missing optional file is not a failed model and must never block a
        # job or make the status card look broken.
        "state": "ready" if onnx.is_file() else "optional",
        "reason": "present" if onnx.is_file() else "optional — skipped for the normal Whisper path",
        "path": str(onnx),
        "sizeBytes": _size(onnx),
        "sizeMb": round(_size(onnx) / 1048576, 1) if onnx.is_file() else 0.0,
        "threads": 1,
    }
    if hash_files and onnx.is_file():
        entry["sha256"] = _hash_file(onnx)
    return [entry]


def provider_model(settings: Any) -> dict[str, Any]:
    provider = str(getattr(getattr(settings, "llm", None), "provider", "") or "")
    cfg = getattr(settings, provider, None)
    model = str(getattr(cfg, "model", "") or "")
    return {
        "id": f"{provider}/{model}" if provider else "detection/unconfigured",
        "role": "highlight detection",
        "name": model or provider,
        "state": "remote" if provider and model else "missing",
        "reason": (
            f"configured {provider}; key is resolved server-side"
            if provider and model
            else "no detection provider configured"
        ),
        "path": None,
        "sizeBytes": 0,
        "sizeMb": 0.0,
        "threads": None,
    }


def quick(settings: Any) -> list[dict[str, Any]]:
    """Cheap state for /api/health — no hashes, no model loads.

    Carries the policy label so the status card can show which models are
    required, optional or unused, and the size of each one.
    """
    entries = (
        whisper_models(settings)
        + face_models()
        + optional_models(settings)
        + [provider_model(settings)]
    )
    for entry in entries:
        entry.setdefault("policy", classify(entry))
    return entries


#: Policy labels. "required" must exist for the default clip path, "optional"
#: is loaded only for a specific feature, "unused" is left behind by an older
#: configuration and is safe to release (it is re-downloadable).
POLICY_REQUIRED = "required"
POLICY_OPTIONAL = "optional"
POLICY_UNUSED = "unused"


def classify(entry: dict[str, Any]) -> str:
    """The policy label for one audit entry.

    Derived from the entry itself (role/state/reason) so the registry cannot
    drift from the audit: anything the default Whisper or reframe path needs is
    required, forced-alignment weights are optional, and anything the host has
    declared unloadable is unused and may be swept.
    """
    state = str(entry.get("state") or "")
    role = str(entry.get("role") or "")
    reason = str(entry.get("reason") or "").lower()
    if "cannot load" in reason or "too large" in reason or "unloadable" in reason:
        return POLICY_UNUSED
    if role in {"transcribe", "transcription", "reframe"}:
        # The ACTIVE transcription weights are required even while they live in
        # the bucket rather than on this disk (state "bucket-only"): the batch
        # restores them, so sweeping them must not be described as "unused".
        return POLICY_REQUIRED
    if role.startswith("forced-alignment") or state == "optional":
        return POLICY_OPTIONAL
    return POLICY_REQUIRED if state == "ready" else POLICY_OPTIONAL


def registry(settings: Any) -> list[dict[str, Any]]:
    """Every model with its policy label — what the status card and sweeper read.

    The idle sweeper only ever removes POLICY_UNUSED entries, and the download
    path only ever fetches entries marked required for the current config: no
    extra model is downloaded silently.
    """
    out: list[dict[str, Any]] = []
    for entry in (whisper_models(settings) + face_models()
                  + optional_models(settings) + [provider_model(settings)]):
        item = dict(entry)
        item["policy"] = classify(entry)
        out.append(item)
    return out


def audit(settings: Any) -> list[dict[str, Any]]:
    """Doctor/CLI view with real SHA-256 hashes and policy labels."""
    entries = (
        whisper_models(settings, hash_files=True)
        + face_models(hash_files=True)
        + optional_models(settings, hash_files=True)
        + [provider_model(settings)]
    )
    for entry in entries:
        entry.setdefault("policy", classify(entry))
    return entries
