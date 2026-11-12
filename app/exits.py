"""One vocabulary for every way a job process can stop.

The dashboard used to show whatever number the OS happened to hand back —
"code 143", "code -15", "code 137" — and the same event arrived under two
different numbers depending on whether Python's ``Popen`` reported the signal
(negative) or a shell-style runner exited with ``128 + signal``. Users read that
as three unrelated crashes and could not tell a host restart from a bug.

Everything funnels through :func:`classify`, which returns a stable class, a
plain-language sentence, an actionable hint and the automatic response. Both
conventions are normalised first, so ``-15`` and ``143`` are one verdict.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass

# --- classes --------------------------------------------------------------- #

EXTERNAL_INTERRUPTION = "external_interruption"
RESOURCE = "resource"
EXTERNAL_KILL = "external_kill"
NATIVE_CRASH = "native_crash"
INTERRUPTED = "interrupted"
INCOMPLETE_EXIT = "incomplete_exit"
APP_ERROR = "app_error"
STAGE_TIMEOUT = "stage_timeout"
MISSING_BINARY = "missing_binary"
DISK_FULL = "disk_full"
UNKNOWN = "unknown"

# --- automatic responses --------------------------------------------------- #

AUTO_RESUME = "auto_resume"
RETRY_AFTER_CLEANUP = "retry_after_cleanup"
RETRY_LIGHTER = "retry_lighter"
RETRY_STAGE = "retry_stage"
FAIL_FAST = "fail_fast"
CAPTURE_AND_RESUME = "capture_and_resume"


@dataclass(frozen=True)
class Verdict:
    """How to read one process exit, and what to do about it."""

    kind: str
    message: str
    hint: str
    action: str

    @property
    def resumable(self) -> bool:
        return self.action in {
            AUTO_RESUME, RETRY_LIGHTER, RETRY_STAGE, CAPTURE_AND_RESUME,
            RETRY_AFTER_CLEANUP,
        }

    @property
    def retryable(self) -> bool:
        """A disk-full is retryable after cleanup; it is never fail-fast."""
        return self.action != FAIL_FAST


# 128 + N is how a shell reports "died from signal N"; Popen reports -N.
_SIGNAL_MAX = 64


def signum(code: int | None) -> int | None:
    """The signal number behind an exit code, whichever convention it uses."""
    if code is None:
        return None
    code = int(code)
    if -_SIGNAL_MAX < code < 0:
        return -code
    if 128 < code <= 128 + _SIGNAL_MAX:
        return code - 128
    return None


def code_label(code: int | None) -> str:
    """Stable human label for an exit code, e.g. ``-15 (SIGTERM)``."""
    sig = signum(code)
    if sig is not None:
        try:
            return f"{code} ({signal.Signals(sig).name})"
        except ValueError:
            return str(code)
    return str(code)


def _stage(stage: str) -> str:
    return f" during {stage}" if stage else ""


def classify(
    code: int | None,
    *,
    stage: str = "",
    oom: bool = False,
    traceback_seen: bool = False,
    timeout: bool = False,
    missing: str = "",
    disk_full: bool = False,
) -> Verdict:
    """Map one exit to a class, a sentence, a hint and an automatic response.

    ``disk_full`` (ENOSPC, or free space below the stage's requirement) is its
    own class: the correct response is cleanup, wait, retry — not fail-fast.
    Recording it as fail-fast is what turned a full disk into a permanently
    dead batch that could not be resumed even after space was freed.
    """
    if disk_full:
        return Verdict(
            DISK_FULL,
            f"The disk ran out of space{_stage(stage)} (ENOSPC).",
            "Finished clips are kept. Scratch and reclaimable media are cleared, "
            "the batch waits for free space, then retries the stage — it is not "
            "marked failed by the disk being full.",
            RETRY_AFTER_CLEANUP,
        )
    if timeout:
        return Verdict(
            STAGE_TIMEOUT,
            f"The {stage or 'current'} stage exceeded its time budget and was stopped.",
            "Its stack was written to the job log. The stage is retried, and "
            "finished work is kept.",
            RETRY_STAGE,
        )
    if missing:
        return Verdict(
            MISSING_BINARY,
            f"Cannot run the {stage or 'current'} stage: {missing} is not installed.",
            f"Install {missing} (or point the config at it) and press Resume — "
            f"nothing is retried until it exists.",
            FAIL_FAST,
        )

    sig = signum(code)
    if sig == int(signal.SIGTERM):
        return Verdict(
            EXTERNAL_INTERRUPTION,
            f"Interrupted by SIGTERM (code {code}){_stage(stage)} — the host or a "
            f"deploy stopped the worker. This was not a crash.",
            "Finished stages and chunks are kept on disk; the job resumes at the "
            "first unfinished unit.",
            AUTO_RESUME,
        )
    if sig == int(signal.SIGKILL):
        if oom:
            return Verdict(
                RESOURCE,
                f"The kernel out-of-memory killer stopped the worker (code {code})"
                f"{_stage(stage)}.",
                "The job resumes with lighter settings; if it keeps happening, "
                "lower the model or the concurrency.",
                RETRY_LIGHTER,
            )
        return Verdict(
            EXTERNAL_KILL,
            f"The worker was force-killed (code {code}, SIGKILL){_stage(stage)} — "
            f"nothing in the app can trap SIGKILL, and no OOM record was found.",
            "The host or a deploy removed the process. Checkpoints on disk let it "
            "resume; if this repeats, the host is recycling the sandbox.",
            AUTO_RESUME,
        )
    if sig in {int(signal.SIGSEGV), int(signal.SIGABRT), int(signal.SIGBUS)}:
        name = signal.Signals(sig).name
        return Verdict(
            NATIVE_CRASH,
            f"Native code crashed with {name} (code {code}){_stage(stage)} — this is "
            f"a fault in a compiled library (Whisper/ctranslate2, ffmpeg), not a bug "
            f"in the job definition.",
            "The stage is retried with lighter settings and the stack is captured.",
            RETRY_LIGHTER,
        )
    if sig == int(signal.SIGINT):
        return Verdict(
            INTERRUPTED,
            f"Interrupted by SIGINT (code {code}){_stage(stage)}.",
            "Treated as an interruption: the job resumes from its checkpoints.",
            AUTO_RESUME,
        )
    if code == 0:
        return Verdict(
            INCOMPLETE_EXIT,
            f"The worker exited cleanly (code 0){_stage(stage)} before the job was "
            f"done.",
            "Recorded and resumed; a 0 exit is not a success on its own.",
            AUTO_RESUME,
        )
    if sig is not None:
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"signal {sig}"
        return Verdict(
            UNKNOWN,
            f"The worker was stopped by {name} (code {code}){_stage(stage)}.",
            "Everything known was captured; the job resumes from its checkpoints.",
            CAPTURE_AND_RESUME,
        )
    if code == 124:
        return Verdict(
            STAGE_TIMEOUT,
            f"The {stage or 'current'} stage timed out (code 124).",
            "Its stack was written to the job log and the stage is retried.",
            RETRY_STAGE,
        )
    if code in {126, 127}:
        return Verdict(
            MISSING_BINARY,
            f"A required tool could not be executed (code {code}){_stage(stage)}.",
            "Check that ffmpeg/ffprobe/yt-dlp exist and are executable, then press "
            "Resume.",
            FAIL_FAST,
        )
    if code == 1 and traceback_seen:
        return Verdict(
            APP_ERROR,
            f"The worker raised an error (code 1){_stage(stage)}.",
            "The traceback is in the job details; the stage is retried up to two "
            "times before the job is marked failed.",
            RETRY_STAGE,
        )
    return Verdict(
        UNKNOWN,
        f"The worker exited with code {code_label(code)}{_stage(stage)}.",
        "The full output was captured; the job resumes from its checkpoints.",
        CAPTURE_AND_RESUME,
    )


#: Substrings that mean "the filesystem said no", across ffmpeg/Python/OS error
#: shapes. Kept as one table so the classifier and the job-status layer agree.
_DISK_FULL_MARKERS = (
    "no space left on device",
    "enospc",
    "errno 28",
    "disk quota exceeded",
    "not enough space",
)


def is_disk_full(exc_or_text: object) -> bool:
    """Does this exception/message describe a full disk (ENOSPC)?"""
    if exc_or_text is None:
        return False
    if isinstance(exc_or_text, OSError) and getattr(exc_or_text, "errno", None) == 28:
        return True
    text = str(exc_or_text).lower()
    return any(marker in text for marker in _DISK_FULL_MARKERS)


def classify_error(exc: BaseException, *, stage: str = "") -> Verdict:
    """Classify a raised exception (as opposed to a process exit code)."""
    if is_disk_full(exc):
        return classify(None, stage=stage, disk_full=True)
    return classify(1, stage=stage, traceback_seen=True)
