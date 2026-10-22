"""Resumable stage workspace: work/{job_id}/{stage}/ artifacts + completion markers.

The pipeline writes each stage's output to disk BEFORE advancing, so a retry
resumes from the first incomplete stage instead of re-running everything (the
long-video failure mode: a 90-minute transcription that dies at 85% must not
restart from zero). Long stages also write incrementally *within* the stage
(transcription caches per-window results) so a resume loses at most one
decode window, not the whole stage.

Layout:

    data/work/{job_id}/
      fetch/       source video pointer + meta.json + DONE
      transcribe/  windows/w_00000.json ... (incremental) + words.json + DONE
      detect/      candidates.json + DONE
      reframe/     cuts.json + per-clip crop windows + DONE
      captions/    per-clip .ass files + DONE
      export/      marker for the final zip-able state

A stage is complete when its DONE marker exists AND its artifacts are still on
disk. Markers are written atomically (tmp+rename) exactly like the job store.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from loguru import logger

# Canonical stage order. Used for resume decisions and for the UI.
STAGES = ["fetch", "transcribe", "detect", "reframe", "captions", "export"]


def workspace_root(data_dir: Path, job_id: str) -> Path:
    return Path(data_dir) / "work" / job_id


def remove_workspace(data_dir: Path, job_id: str) -> None:
    """Delete a job's stage artifacts (called when the job itself is deleted).

    Without this the per-stage mirrors — including the extracted audio WAV and
    the per-chunk transcript caches — accumulate forever on a hosted disk.
    """
    root = workspace_root(data_dir, job_id)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
        logger.info(f"[stages] removed workspace {root}")


class StageWorkspace:
    """Per-job artifact store rooted at data/work/{job_id}."""

    def __init__(self, data_dir: Path, job_id: str) -> None:
        self.root = workspace_root(data_dir, job_id)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # paths
    # ------------------------------------------------------------------ #

    def stage_dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _marker(self, stage: str) -> Path:
        return self.root / stage / "DONE"

    # ------------------------------------------------------------------ #
    # completion markers
    # ------------------------------------------------------------------ #

    def is_done(self, stage: str) -> bool:
        m = self._marker(stage)
        if not m.exists():
            return False
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            return False
        # artifacts declared in the marker must still exist (user may have
        # deleted downloads/ while a job sits in the queue)
        for rel in meta.get("artifacts", []):
            if not (self.root / rel).exists():
                logger.info(f"[stages] {stage} marker stale (missing {rel}) — re-running stage")
                return False
        return True

    def mark_done(self, stage: str, artifacts: list[str] | None = None, **meta: Any) -> None:
        d = self.stage_dir(stage)
        payload = {
            "stage": stage,
            "at": int(time.time() * 1000),
            "artifacts": artifacts or [],
            **meta,
        }
        tmp = d / "DONE.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._marker(stage))

    def write_clips(self, stage: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Persist detection candidates as per-clip stage artifacts + a merged list.

        Detection now writes ONE file per candidate under
        ``<stage>/clips/<i:03d>.json``, plus a merged ``candidates.json`` that
        the downstream stages already read. Writing per-clip makes the render
        stage resumable per clip: if the host dies mid-export, a retry can read
        the clips that were already persisted and only re-render the missing
        ones, instead of re-relying on a single point-in-time ``candidates.json``
        that might have been overwritten by a later (failed) re-detect.

        Each candidate is written with its resolved ``start``/``end`` in seconds
        so downstream stages never re-derive them from word indices.
        """
        clips_dir = self.stage_dir(stage) / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        written: list[dict[str, Any]] = []
        for i, c in enumerate(candidates):
            p = clips_dir / f"{i:03d}.json"
            p.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
            written.append(c)
        return written

    def done_meta(self, stage: str) -> dict[str, Any]:
        try:
            return json.loads(self._marker(stage).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def clear(self, stage: str, *, leave_scratch_audio: bool = False) -> None:
        """Drop a stage and everything after it.

        `leave_scratch_audio` keeps the extracted 16kHz mono WAV cache that
        transcribe and the render stages both read. A repair that clears from
        transcribe must keep it: on a single shared 1-CPU / 2-GB slot the
        silence budget is measured from cached transcription, and deleting the
        scratch before the re-queued attempt can finish turns a stalled attempt
        into an infinite re-queue loop on the health dashboard.
        """
        d = self.root / stage
        if d.exists():
            if leave_scratch_audio and stage == "transcribe":
                # Keep the one extracted-audio cache both the transcribe and
                # render stages read. This is the job's scratch output directly
                # after its own re-transcribe, so it is deleted by the normal
                # upstream cleanup once the new clips exist.
                pass
            else:
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"[stages] cleared stage {stage}")

    def clear_from(self, stage: str, *, leave_scratch_audio: bool = False) -> None:
        """Clear `stage` and all downstream stages.

        `leave_scratch_audio` keeps the extracted 16kHz mono WAV cache that
        transcribe and the render stages both read. A repair that clears from
        transcribe must keep it: the host is a single shared 1-CPU / 2-GB slot,
        and deleting the scratch before the re-queued attempt can finish is what
        made the dashboard show "Transcribing... 0%" forever.
        """
        if stage not in STAGES:
            return
        for s in STAGES[STAGES.index(stage):]:
            self.clear(s, leave_scratch_audio=leave_scratch_audio)

    def first_incomplete(self) -> str | None:
        for s in STAGES:
            if not self.is_done(s):
                return s
        return None

    # ------------------------------------------------------------------ #
    # json helpers (atomic)
    # ------------------------------------------------------------------ #

    def write_json(self, stage: str, name: str, data: Any) -> Path:
        path = self.stage_dir(stage) / name
        # Nested artifact names ("chunks/chunk_000.json") are used by the
        # per-chunk caches. stage_dir() only creates the stage directory, so
        # without this the write failed with FileNotFoundError on the very
        # first cached chunk.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path

    # ------------------------------------------------------------------ #
    # typed artifact validation
    # ------------------------------------------------------------------ #

    def read_json(
        self, stage: str, name: str, *, expected: type | None = None,
        min_items: int | None = None, max_items: int | None = None,
    ) -> Any | None:
        """Read a stage artifact and validate its shape.

        Replaces the old untyped read so a corrupt/missing/wrong-shape
        artifact invalidates only that stage instead of crashing the pipeline
        the moment it is accessed. ``expected`` is the Python type we expect
        (``list`` or ``dict``); ``min_items``/``max_items`` apply to
        sequences.

        Returns None when the file is missing, unreadable, or the wrong shape
        — the caller is then expected to re-run the stage (or, for
        per-clip artifacts, skip it) rather than crash.
        """
        path = self.root / stage / name
        if not path.exists():
            logger.info(f"[stages] {stage}/{name}: missing — stage needs re-run")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"[stages] {stage}/{name}: unreadable json: {exc}")
            return None
        if expected is not None and not isinstance(data, expected):
            logger.warning(
                f"[stages] {stage}/{name}: wrong shape "
                f"(expected {expected.__name__}, got {type(data).__name__}) — "
                f"stage needs re-run"
            )
            return None
        if isinstance(data, list):
            n = len(data)
            if min_items is not None and n < min_items:
                logger.warning(
                    f"[stages] {stage}/{name}: too few items "
                    f"({n} < {min_items}) — stage needs re-run"
                )
                return None
            if max_items is not None and n > max_items:
                logger.warning(
                    f"[stages] {stage}/{name}: too many items "
                    f"({n} > {max_items}) — stage needs re-run"
                )
                return None
        return data
