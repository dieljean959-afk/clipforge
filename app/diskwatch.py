"""Per-job disk ledger: what this batch costs, stage by stage, and the cap.

The 4 GB box has ~1.6 GB of code (venv + node_modules) and ~680 MB of Whisper
models, so a job that copies the source video to disk leaves nothing for the
OS. That is what made "Transcribing 0%" fail with "only 234 MB of disk free".

Two jobs here:

* **Tell the truth about the cost.** `JobLedger.stage()` records free bytes at
  the start and end of every stage and logs ``disk delta: -12 MB``; the sampler
  thread records the high-water mark every `SAMPLE_SEC`, which is what proves
  "peak local use was about one clip" instead of asserting it.
* **Stop a runaway before it fills the disk.** If the job's own scratch crosses
  `SCRATCH_CAP_BYTES` the stage is stopped and the responsible file is named,
  because "disk full" with no filename is unactionable.

Only the job's own directories are measured (`downloads/<job>`, `outputs/<job>`,
`data/work/<job>`, `data/uploads/<job>`), never `.venv` or `node_modules`, so
"disk used by this job" means what it says. Free space is read from `df`'s
equivalent (`shutil.disk_usage`), which is the number the OS enforces.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
#: 300 MB: large enough for one 1080x1920 clip plus its caption/title artefacts,
#: small enough that a second copy of the source cannot hide in it.
SCRATCH_CAP_BYTES = 300 * _MB
#: How often the sampler reads free space. The brief asks for 5 s samples.
SAMPLE_SEC = 5.0
#: A finished batch must give the disk back to within this much of where it
#: started. Logs, the job JSON and normal filesystem noise live inside it.
RECOVERY_TOLERANCE_BYTES = 50 * _MB


def free_bytes(path: str | Path) -> int | None:
    try:
        return int(shutil.disk_usage(str(path)).free)
    except Exception:  # noqa: BLE001 - a missing dir is "unknown", not "zero"
        return None


def mb(value: int | float | None) -> float | None:
    """Bytes in, MB out — the ONE conversion used by every caller."""
    if value is None:
        return None
    return round(float(value) / _MB, 1)


def tree_bytes(path: str | Path) -> int:
    """Bytes under one directory, or 0 when it does not exist."""
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def biggest_file(paths: list[Path]) -> tuple[str, int] | None:
    """The largest file in these trees — the one to name when the cap trips."""
    best: tuple[str, int] | None = None
    for root in paths:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                if not p.is_file() or p.is_symlink():
                    continue
                size = p.stat().st_size
            except OSError:
                continue
            if best is None or size > best[1]:
                best = (str(p), size)
    return best


class ScratchCapExceeded(RuntimeError):
    """Scratch crossed the cap; carries the file that is responsible."""

    def __init__(self, used_bytes: int, path: str, size: int, stage: str) -> None:
        self.used_bytes = used_bytes
        self.path = path
        self.size = size
        self.stage = stage
        super().__init__(
            f"this stage needed {mb(used_bytes)} MB of local scratch, over the "
            f"{mb(SCRATCH_CAP_BYTES)} MB cap — the largest file was "
            f"{path} ({mb(size)} MB) during {stage}. Nothing was deleted from "
            f"the bucket; free disk or lower the clip count and retry."
        )


class JobLedger:
    """Free-space bookkeeping for ONE job, sampled in the background."""

    def __init__(
        self, job_id: str, roots: list[Path], *, data_dir: Path,
        on_log: Callable[[str], None] | None = None,
        on_sample: Callable[[dict[str, Any]], None] | None = None,
        cap_bytes: int = SCRATCH_CAP_BYTES,
        sample_sec: float = SAMPLE_SEC,
    ) -> None:
        self.job_id = job_id
        self.roots = [Path(r) for r in roots]
        self.data_dir = Path(data_dir)
        self.on_log = on_log
        self.on_sample = on_sample
        self.cap_bytes = cap_bytes
        self.sample_sec = sample_sec

        self.start_free = free_bytes(self.data_dir)
        self.peak_job_bytes = 0
        self.peak_free = self.start_free
        self.samples: list[dict[str, Any]] = []
        self.stages: list[dict[str, Any]] = []
        self.breach: ScratchCapExceeded | None = None
        self._stage = "start"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- sampling -----------------------------------------------------
    def job_bytes(self) -> int:
        return sum(tree_bytes(r) for r in self.roots)

    def sample(self, *, stage: str | None = None) -> dict[str, Any]:
        now = free_bytes(self.data_dir)
        used = self.job_bytes()
        point = {
            "t": round(time.time(), 1),
            "stage": stage or self._stage,
            "freeMb": mb(now),
            "jobMb": mb(used),
        }
        self.samples.append(point)
        # A 5 s sampler over a 40 min job is 480 points; keep the ledger small
        # by dropping the oldest detail once the shape is established.
        if len(self.samples) > 720:
            self.samples = self.samples[::2]
        if now is not None:
            self.peak_free = min(v for v in (self.peak_free, now) if v is not None)
        self.peak_job_bytes = max(self.peak_job_bytes, used)
        if used > self.cap_bytes and self.breach is None:
            worst = biggest_file(self.roots)
            self.breach = ScratchCapExceeded(
                used,
                worst[0] if worst else "unknown",
                worst[1] if worst else 0,
                point["stage"],
            )
            logger.warning(f"[job {self.job_id}] {self.breach}")
            if self.on_log:
                self.on_log(str(self.breach))
        return point

    def _run(self) -> None:
        while not self._stop.wait(self.sample_sec):
            try:
                point = self.sample()
                if self.on_sample:
                    self.on_sample(self.report())
            except Exception as exc:  # noqa: BLE001 - never kill a job over stats
                logger.warning(f"[job {self.job_id}] disk sample failed: {exc}")

    def start(self) -> None:
        self.sample(stage="start")
        self._thread = threading.Thread(
            target=self._run, name=f"disk-{self.job_id[:8]}", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---- the stage table ----------------------------------------------
    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Record free space before/after ONE stage and log the delta."""
        self._stage = name
        before = free_bytes(self.data_dir)
        job_before = self.job_bytes()
        started = time.time()
        row: dict[str, Any] = {"stage": name, "freeBeforeMb": mb(before)}
        try:
            yield row
        finally:
            after = free_bytes(self.data_dir)
            row.update({
                "freeAfterMb": mb(after),
                "deltaMb": (
                    None if before is None or after is None
                    else round((after - before) / _MB, 1)
                ),
                "jobMb": mb(self.job_bytes()),
                "seconds": round(time.time() - started, 1),
                "peakJobMb": mb(self.peak_job_bytes),
            })
            self.stages.append(row)
            if before is not None and after is not None:
                delta = (after - before) / _MB
                if self.on_log:
                    self.on_log(
                        f"[disk] {name}: {mb(after)} MB free "
                        f"(disk delta: {delta:+.0f} MB, this job holds "
                        f"{mb(self.job_bytes())} MB)"
                    )
            self._stage = "between stages"
            # `job_before` is recorded so a stage that leaves its own scratch
            # behind is visible in the table even when df barely moves.
            row["jobBeforeMb"] = mb(job_before)

    # ---- the hard guard ----------------------------------------------
    def check(self, where: str) -> None:
        """Raise `ScratchCapExceeded` if the cap has already been crossed."""
        point = self.sample(stage=where)
        if self.breach is not None:
            raise self.breach
        if (point.get("jobMb") or 0) > mb(self.cap_bytes):
            worst = biggest_file(self.roots)
            self.breach = ScratchCapExceeded(
                int((point.get("jobMb") or 0) * _MB),
                worst[0] if worst else "unknown",
                worst[1] if worst else 0,
                where,
            )
            raise self.breach

    # ---- reporting ----------------------------------------------------
    def report(self) -> dict[str, Any]:
        now = free_bytes(self.data_dir)
        return {
            "startFreeMb": mb(self.start_free),
            "nowFreeMb": mb(now),
            "peakJobMb": mb(self.peak_job_bytes),
            "peakFreeMb": mb(self.peak_free),
            "jobMb": mb(self.job_bytes()),
            "capMb": mb(self.cap_bytes),
            "stages": self.stages,
            "samples": len(self.samples),
            "sampleSec": self.sample_sec,
        }

    def recovered(self) -> tuple[bool, str]:
        """Did the batch give the disk back? (self-check before "complete")"""
        now = free_bytes(self.data_dir)
        if self.start_free is None or now is None:
            return True, "free space unreadable — self-check skipped"
        short = self.start_free - now
        if short <= RECOVERY_TOLERANCE_BYTES:
            return True, (
                f"{mb(now)} MB free, started at {mb(self.start_free)} MB "
                f"(within {mb(RECOVERY_TOLERANCE_BYTES)} MB)"
            )
        worst = biggest_file(self.roots)
        return False, (
            f"this batch left {mb(short)} MB on the disk: {mb(now)} MB free, "
            f"started at {mb(self.start_free)} MB, tolerance "
            f"{mb(RECOVERY_TOLERANCE_BYTES)} MB"
            + (f" — largest leftover {worst[0]} ({mb(worst[1])} MB)" if worst else "")
        )


def cleanup(paths: list[Path]) -> int:
    """Delete the job's scratch trees; returns bytes freed."""
    freed = 0
    for root in paths:
        try:
            if root.exists():
                freed += tree_bytes(root)
                shutil.rmtree(root, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[disk] could not remove {root}: {exc}")
    return freed
