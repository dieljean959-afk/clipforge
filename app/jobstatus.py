"""One authoritative answer to "what is this job doing?".

The inconsistency this module exists to kill
------------------------------------------
The same job was reported as "Failed during AI clip detection" in one view and
"Stage 3 of 4, 61% overall, Detection: chunk 3/7 (cached), Starting on the
server worker…" in another. Both readings came from the SAME job document, and
neither was a lie about the bytes on disk — they were two different questions
answered by two different pieces of code:

  * the failure banner read `stage` + `error`;
  * the progress block read `progress`, `progressDetail`, `stagePlan`,
    `unitsDone/unitsTotal` and `queuePosition` — all of which are left BEHIND
    by a failed job, because the pipeline stops writing them the moment it
    fails and nothing ever clears them.

A failed job therefore carried a full set of running-looking fields, and any
view that keyed off the wrong pair of fields showed a spinner for a job that
had already died. The fix is structural: derive status ONCE, in one place,
from the same inputs, and hand the result to every route — so no two views can
disagree by construction.

What the derivation guarantees
------------------------------
`describe()` returns a single record with:
  * `status`      — queued | running | paused | failed | complete | deleting
  * `isTerminal`  — true once no further progress is possible
  * `active`      — the stage a RUNNING job is on, else None
  * `progress`    — 0-100, and forced to 100 for a terminal job so a dead job
                    can never render as "61% overall" with a live spinner
  * every running-only field (stagePlan, units, ETAs, queuePosition,
    progressDetail) is CLEARED for a terminal job, so a view that forgets to
    check `isTerminal` shows nothing rather than a lie

The UI still renders whatever it likes, but it can only ever read consistent
values, and the values that make a job look alive are removed the moment it
stops being alive.
"""

from __future__ import annotations

from typing import Any

#: Stages that mean "a child process is (or should be) working on this".
ACTIVE_STAGES: frozenset[str] = frozenset({
    "grabbing", "fetch", "fetching",
    "transcribing", "transcribe",
    "finding", "detecting",
    "rendering", "reframe", "captions", "export",
})

#: Terminal stages. Nothing further can happen to a job in one of these.
#:
#: `deleting` belongs here for a hard reason: the stale-heartbeat reconciler
#: re-queues any live-looking stage with no worker behind it, and re-queuing a
#: batch that is being purged would run the pipeline on files the purge is
#: removing. A batch in `deleting` is settled — `app/purge.py` is finishing it,
#: in the background, with no worker involved.
TERMINAL_STAGES: frozenset[str] = frozenset({"complete", "failed", "paused", "deleting"})

#: The pipeline stages a user is shown, in order. Used to name the stage a
#: failed job died in when the raw stage name is an internal alias.
_PLAN_ORDER: tuple[str, ...] = ("grabbing", "transcribing", "finding", "rendering")
_PLAN_ALIASES: dict[str, str] = {
    "fetch": "grabbing", "fetching": "grabbing",
    "transcribe": "transcribing",
    "detecting": "finding", "detect": "finding",
    "reframe": "rendering", "render": "rendering",
    "captions": "rendering", "export": "rendering",
}

_PLAN_LABELS: dict[str, str] = {
    "queued": "Queued",
    "grabbing": "Fetching video",
    "transcribing": "Transcribing",
    "finding": "AI clip detection",
    "rendering": "Rendering 9:16",
}

#: Fields that only make sense WHILE a job is running. Left in place on a dead
#: job, each one is a chance for a view to render a spinner, a progress bar or
#: a countdown for work that is not happening.
_RUNNING_ONLY_FIELDS: tuple[str, ...] = (
    "stagePlan",
    "unitsDone",
    "unitsTotal",
    "unitLabel",
    "stageEtaSec",
    "totalEtaSec",
    "queueEtaSec",
    "queuePosition",
    "pausedStage",
    "pausedAt",
)


def plan_stage(stage: str | None) -> str | None:
    """Map a raw stage name onto the stage the user thinks in."""
    if not stage:
        return None
    return _PLAN_ALIASES.get(stage, stage)


def status_of(job: dict[str, Any]) -> str:
    """The one status word for a job. Never guesses from progress."""
    stage = str(job.get("stage") or "queued")
    if stage in TERMINAL_STAGES:
        return stage
    # An explicit "queued" is the worker saying "this job is waiting for a
    # slot". The stage field still holds the step it will resume at, so trusting
    # the stage first made a waiting job look like it was already running it.
    # `queuedAt` vs `lastProgressAt` is what makes this safe for documents
    # written by an older build: the moment the pipeline ticks again, progress
    # is newer than the enqueue and the job is running again.
    if _waiting_for_slot(job):
        return "queued"
    if stage in ACTIVE_STAGES:
        # A signalled worker leaves `interrupted` set while the resume runs;
        # it is still running work, just resumed, so it stays "running".
        return "running"
    if not stage or stage in {"queued", "waiting-upload"}:
        # A job whose upload never finished is not running: it waits, and the
        # upload endpoint is what queues it. Its card says "Waiting for upload".
        return "queued"
    # An unknown stage is a real possibility after a bad deploy. Treat it as
    # running rather than inventing a new status the UI cannot render.
    return "running"


def _waiting_for_slot(job: dict[str, Any]) -> bool:
    """Is this job parked in the queue, rather than running the stage it keeps?"""
    if str(job.get("status") or "") != "queued":
        return False
    try:
        queued_at = int(job.get("queuedAt") or 0)
    except (TypeError, ValueError):
        queued_at = 0
    try:
        last = int(job.get("lastProgressAt") or 0)
    except (TypeError, ValueError):
        last = 0
    # No enqueue timestamp: this never started, so it is genuinely waiting.
    return queued_at >= last


def is_terminal(job: dict[str, Any]) -> bool:
    return status_of(job) in TERMINAL_STAGES


def active_stage(job: dict[str, Any]) -> str | None:
    """The stage a RUNNING job is on, else None."""
    if status_of(job) != "running":
        return None
    return plan_stage(job.get("stage"))


def stage_label(job: dict[str, Any]) -> str | None:
    """Human name of the stage this job is on, or died in."""
    raw = job.get("stage")
    if not raw:
        return None
    if status_of(job) == "running":
        s = plan_stage(raw)
    elif status_of(job) == "failed":
        # `_fail` overwrites `stage` with "failed" but keeps the stage it died
        # in as `failedStage`; that is the only place the real one survives.
        s = plan_stage(job.get("failedStage") or raw)
    elif status_of(job) == "paused":
        s = plan_stage(job.get("pausedStage") or job.get("failedStage") or raw)
    else:
        s = plan_stage(raw)
    if not s or s in TERMINAL_STAGES:
        return None
    # Never hand the card a raw snake_case identifier ("queued" was rendering
    # next to capitalised labels from the same map).
    return _PLAN_LABELS.get(s, s.replace("_", " ").capitalize())


def describe(job: dict[str, Any]) -> dict[str, Any]:
    """The authoritative status record for one job.

    Returns the job with a normalised, mutually-consistent set of status
    fields. Terminal jobs have every running-only field removed, so a caller
    that reads `progress` or `stagePlan` without checking `isTerminal` gets
    `None` rather than a stale "chunk 3/7 (cached)".
    """
    out = dict(job)
    status = status_of(job)
    terminal = status in TERMINAL_STAGES

    out["status"] = status
    out["isTerminal"] = terminal
    out["activeStage"] = active_stage(job)
    out["stageLabel"] = stage_label(job)

    if terminal:
        # Progress is meaningless once the job has stopped, and a partial
        # value is worse than none: "61% overall" beside "Failed" is exactly
        # the contradiction being fixed.
        out["progress"] = 100 if status == "complete" else 0
        for field in _RUNNING_ONLY_FIELDS:
            out.pop(field, None)
        # A terminal job's progressDetail is ALWAYS overwritten, never merely
        # filled in when empty. "Detection: chunk 3/7 (cached)" is the exact
        # line that made a dead job look live, and it is still sitting in the
        # document because the pipeline stops writing the moment it fails.
        out["progressDetail"] = _TERMINAL_DETAIL.get(status, "")
    else:
        try:
            pct = float(out.get("progress") or 0.0)
        except (TypeError, ValueError):
            pct = 0.0
        # A running job is never "done" — clamp so a job at 100% mid-render
        # cannot read as complete, which is the mirror image of the bug above.
        out["progress"] = max(0.0, min(99.0, pct))
        # A running job whose progress was reset to 0 (e.g. re-queued after a
        # host restart) still carries stale running-only fields from the prior
        # attempt. Those fields make a freshly-queued job read as "5 of 15
        # clips" with a filled-in stage plan, which is the exact contradiction
        # this module exists to prevent. Clear them the same way we do for
        # terminal jobs. Preserve progressDetail only if it's a valid resume
        # message (set by the worker when re-queuing), otherwise clear it too.
        if out["progress"] == 0.0:
            for field in _RUNNING_ONLY_FIELDS:
                out.pop(field, None)
            # Only preserve progressDetail if it's a resume message set by the
            # worker when re-queuing. Stale progress details from a prior run
            # (e.g. "Rendering clip 5 of 15") must be cleared.
            if not out.get("progressDetail", "").lower().startswith("re-queued"):
                out.pop("progressDetail", None)
    return out


_TERMINAL_DETAIL: dict[str, str] = {
    "failed": "Stopped — this job is not running.",
    "paused": "Paused — resume to continue from the last checkpoint.",
    "complete": "Done — the clips are ready.",
    "deleting": "Deleting — removing this batch's files in the background.",
}


def describe_all(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [describe(j) for j in jobs]
