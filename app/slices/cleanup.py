from __future__ import annotations

import shutil
from pathlib import Path


def unlink_model_dirs(data_dir: Path, job_id: str) -> None:
    """Drop job-scoped model/layer caches when a batch is deleted.

    The disk meter sums the data/output tree, so a delete only shows up as
    freed space once these per-job caches are gone too — otherwise the
    dashboard keeps reporting the same "free" number after a Clean up.

    Deliberately narrow: only directories named exactly after the job id are
    touched, so shared caches are never removed.
    """
    for name in ("models", "layers"):
        shutil.rmtree(data_dir / name / job_id, ignore_errors=True)
