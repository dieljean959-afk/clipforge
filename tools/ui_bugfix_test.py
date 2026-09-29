"""Tests for the four dashboard bugs, checked against the real source.

  BUG 2  one job reading "Failed" and "stage 3 of 4, running" at the same time
  BUG 3  "932.8100000000011s of clips"
  BUG 4  a raw Python traceback overflowing a phone screen
  BUG 5  "youtube.com/watch · 23 clips" for every batch

The formatters are executed for real (extracted from the TS so the arithmetic is
checked, not just grepped for). The component behaviour is asserted against the
source, which is how the rest of this repo's UI suites work.

    python tools/ui_bugfix_test.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


def src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# BUG 3 — duration formatting. Run the real function via node.
# --------------------------------------------------------------------------- #

def _fmt_duration_long_js() -> str:
    """Extract fmtDurationLong from clipforge.ts and run it under node."""
    body = src("src/lib/clipforge.ts")
    m = re.search(
        r"export function fmtDurationLong\([^)]*\)[^{]*\{.*?\n\}", body, re.DOTALL
    )
    if not m:
        return ""
    fn = m.group(0).replace("export function", "function")
    return fn


def test_duration_formatting() -> None:
    print("\n[BUG 3] durations render as mm:ss / h:mm:ss, rounded")
    fn = _fmt_duration_long_js()
    check("fmtDurationLong exists in clipforge.ts", bool(fn))
    if not fn:
        return
    cases = [
        # (input, expected) — the float-sum and rounding cases are the point
        (932.8100000000011, "15:33"),
        (271.46000000000004, "4:31"),
        (0, "0:00"),
        (59.4, "0:59"),
        (59.6, "1:00"),
        (60, "1:00"),
        (61, "1:01"),
        (600, "10:00"),
        (3599, "59:59"),
        (3600, "1:00:00"),
        (3661, "1:01:01"),
        (7325, "2:02:05"),
        (-5, "0:00"),
    ]
    runner = f"""
{fn}
const cases = {json.dumps(cases)};
const out = cases.map(([i, e]) => [i, fmtDurationLong(i), e]);
console.log(JSON.stringify(out));
"""
    r = subprocess.run(
        ["node", "-e", runner], capture_output=True, text=True, timeout=30, cwd=ROOT
    )
    if r.returncode != 0:
        check("node ran the formatter", False, r.stderr[:200])
        return
    for got_in, got, want in json.loads(r.stdout):
        check(f"fmtDurationLong({got_in}) == {want!r}", got == want, f"got {got!r}")

    # The exact string from the bug report must not survive anywhere.
    for rel in ("src/pages/Dashboard.tsx", "src/components/clipforge/JobCard.tsx"):
        body = src(rel)
        check(f"{rel} no longer interpolates a raw float",
              not re.search(r"\{[^}]*totalClipsSec[^}]*\}s", body),
              "found `{...totalClipsSec}s`")
        check(f"{rel} uses fmtDurationLong", "fmtDurationLong" in body)


# --------------------------------------------------------------------------- #
# BUG 5 — batch labels
# --------------------------------------------------------------------------- #

def test_batch_label() -> None:
    print("\n[BUG 5] batches are identified by title or video ID, not bare host")
    body = src("src/lib/clipforge.ts")
    check("batchLabel exists", "export function batchLabel" in body)
    check("sourceId exists", "export function sourceId" in body)

    # youtu.be/<id> and /watch?v=<id> are the two shapes seen in the wild.
    m = re.search(r"export function sourceId\([^)]*\)[^{]*\{.*?\n\}", body, re.DOTALL)
    if m:
        fn = m.group(0).replace("export function", "function")
        runner = f"""
{fn}
const cases = [
  ["https://youtu.be/LsNjFCfQKis", false],
  ["https://www.youtube.com/watch?v=zg50i-UmRwA", false],
  ["https://www.youtube.com/watch?v=O0yDqRwmMSE&t=30", false],
  ["https://www.youtube.com/shorts/LsNjFCfQKis", false],
  ["https://www.youtube.com/", false],
  ["https://vimeo.com/12345", false],
];
console.log(JSON.stringify(cases.map(([s,u]) => [s, sourceId({{source: s, isUpload: u}})])));
"""
        r = subprocess.run(
            ["node", "-e", runner], capture_output=True, text=True, timeout=30, cwd=ROOT
        )
        if r.returncode == 0:
            for url, got in json.loads(r.stdout):
                want = {
                    "https://youtu.be/LsNjFCfQKis": "LsNjFCfQKis",
                    "https://www.youtube.com/watch?v=zg50i-UmRwA": "zg50i-UmRwA",
                    "https://www.youtube.com/watch?v=O0yDqRwmMSE&t=30": "O0yDqRwmMSE",
                    "https://www.youtube.com/shorts/LsNjFCfQKis": "LsNjFCfQKis",
                    "https://www.youtube.com/": "",
                    "https://vimeo.com/12345": "",
                }[url]
                check(f"sourceId({url}) == {want!r}", got == want, f"got {got!r}")
        else:
            check("node ran sourceId", False, r.stderr[:200])

    dash = src("src/pages/Dashboard.tsx")
    check("Dashboard pill uses batchLabel", "batchLabel(job)" in dash)
    check("Dashboard pill shows the date", "fmtDate(job.createdAt)" in dash)
    check("Dashboard no longer slices sourceLabel into the pill",
          "sourceLabel(job).slice(" not in dash)
    card = src("src/components/clipforge/JobCard.tsx")
    check("JobCard header uses batchLabel", "batchLabel(job)" in card)


# --------------------------------------------------------------------------- #
# BUG 4 — friendly error, no raw traceback
# --------------------------------------------------------------------------- #

def test_friendly_error() -> None:
    print("\n[BUG 4] a short message up front, traceback behind a disclosure")
    card = src("src/components/clipforge/JobCard.tsx")
    check("FailureNotice component exists", "function FailureNotice" in card)
    check("the failed block uses it", "<FailureNotice job={job} />" in card)
    check("the traceback is NOT rendered inline",
          not re.search(r"\{job\.errorDetail\}\s*</p>", card),
          "errorDetail still rendered as plain text")
    check("details are behind a disclosure",
          "Show details" in card and "aria-expanded" in card)
    check("details are scrollable, not page-widening",
          "overflow-auto" in card and "whitespace-pre-wrap" in card)
    # The specific overflow bug was long unbreakable paths.
    check("wrapping is enforced on the message",
          "[overflow-wrap:anywhere]" in card)
    check("a friendly message is derived", "function friendlyError" in card)
    check("chunk-level detection failure is named in words",
          "AI detection failed on chunk" in card)

    # The real error from the bug report must map to something short.
    m = re.search(r"function friendlyError\([^)]*\)[^{]*\{.*?\n\}", card, re.DOTALL)
    if not m:
        m = re.search(
            r"function friendlyError\([^)]*\): string \{.*?\n\}\n", card, re.DOTALL
        )
    check("friendlyError found in source", m is not None)
    if m:
        fn = m.group(0)
        # Re-indent the nested template literal into valid JS.
        fn = fn.replace(": string {", " {")
        runner = f"""
{fn}
const e1 = "AI detection failed: all 1/7 chunk(s) failed after retries and fallback models. The transcript is fine; the detection backend is not responding with usable JSON right now.";
const e2 = "AI detection failed on chunk 3/7 — skipping it and keeping the rest";
const e3 = "Groq rate limit needs 92 minutes before it will accept another call - this is a daily quota, not a burst one.";
console.log(JSON.stringify([e1, e2, e3].map((e) => friendlyError(e, null))));
"""
        r = subprocess.run(
            ["node", "-e", runner], capture_output=True, text=True, timeout=30, cwd=ROOT
        )
        if r.returncode == 0:
            outs = json.loads(r.stdout)
            for got in outs:
                check(f"friendly error is short ({got[:40]!r})", len(got) < 220, f"{len(got)} chars")
                check(f"friendly error has no traceback ({got[:30]!r})",
                      "Traceback" not in got and "File \"" not in got)
            check("names the failed chunk count", "1 of 7 chunks" in outs[0], outs[0])
            check("quota error becomes an action", "quota" in outs[2].lower(), outs[2])
        else:
            check("node ran friendlyError", False, r.stderr[:300])


# --------------------------------------------------------------------------- #
# BUG 2 — one source of truth
# --------------------------------------------------------------------------- #

def test_single_source_of_truth() -> None:
    print("\n[BUG 2] status comes from the backend, in one place")
    jobstatus = src("app/jobstatus.py")
    check("jobstatus module exists", "def describe(" in jobstatus)
    check("it clears running-only fields on a terminal job",
          "_RUNNING_ONLY_FIELDS" in jobstatus)
    check("it never mutates the stored document", "out = dict(job)" in jobstatus)

    main = src("app/main.py")
    check("list_jobs uses the shared derivation", "jobstatus.describe_all" in main)
    check("get_job uses the same derivation", "return jobstatus.describe(job)" in main)
    check("both reconcile stale running state", main.count("reconcile_stale_running") >= 2)

    worker = src("app/worker.py")
    check("worker has reconcile_stale_running", "def reconcile_stale_running" in worker)
    check("it only touches non-terminal jobs", "jobstatus.is_terminal" in worker)
    check("it skips jobs it is actually running", "if job.get(\"_id\") in live" in worker)

    dash = src("src/pages/Dashboard.tsx")
    check("Dashboard filters on the backend status",
          "j.isTerminal" in dash and 'j.status === "running"' in dash)
    check("Dashboard no longer re-derives from stage",
          'j.stage !== "complete" && j.stage !== "failed"' not in dash)

    card = src("src/components/clipforge/JobCard.tsx")
    check("JobCard keys off status", 'job.status === "failed"' in card)
    check("JobCard uses activeStage for the stepper", "job.activeStage" in card)
    check("live progress gated on terminal state",
          "const watching = !job.isTerminal" in card)
    # The exact contradiction: a live block rendered for a failed job.
    check("live progress block is gated", "{watching || paused} &&" in card)

    api = src("src/lib/api.ts")
    check("JobDto carries status", "status:" in api and '"queued" | "running"' in api)
    check("JobDto carries isTerminal", "isTerminal: boolean" in api)
    check("Check now reports the real status",
          "fresh.isTerminal" in dash and "has failed" in dash)


def main() -> int:
    test_duration_formatting()
    test_batch_label()
    test_friendly_error()
    test_single_source_of_truth()
    print(f"\nui_bugfix: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
