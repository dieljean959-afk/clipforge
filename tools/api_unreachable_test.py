"""A 4xx must never be reported as "API unreachable".

The dashboard showed a hard red "API unreachable: HTTP 400" over a page that
was otherwise rendering live job cards, and the copy blamed `/api/health` when
the failing call was `/api/jobs`. Three separate defects:

  1. `useJobs` — the hook that renders that banner — never got the 4xx triage
     that `useHealth` had. The earlier fix hardened only the health poller, so
     the one still unguarded was the one that actually fires. That is why this
     kept coming back.
  2. the banner copy named the wrong route, so the message pointed the reader
     at the wrong thing during exactly the moment they needed it right.
  3. a single transient failure was enough to raise the alarm, with no
     distinction between "never had data" and "have data, last poll failed".

    python tools/api_unreachable_test.py
"""

from __future__ import annotations

import json
import re
import subprocess
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


def test_triage_logic() -> None:
    print("\n[1] a 4xx is fatal only when it is not a 4xx")
    poll = src("src/hooks/use-poll.ts")
    m = re.search(r"function triage\([^)]*\)[^{]*\{.*?\n\}", poll, re.DOTALL)
    check("triage helper exists", m is not None)
    if not m:
        return
    fn = m.group(0)
    runner = f"""
{fn}
const cases = [
  "HTTP 400", "HTTP 404", "HTTP 422", "HTTP 401", "HTTP 403",
  "HTTP 500", "HTTP 502", "HTTP 503",
  "Failed to fetch", "NetworkError when attempting to fetch resource.",
  "API unreachable",
];
console.log(JSON.stringify(cases.map((c) => [c, triage(c).fatal])));
"""
    r = subprocess.run(["node", "-e", runner], capture_output=True, text=True,
                       timeout=30, cwd=ROOT)
    if r.returncode != 0:
        check("node ran triage", False, r.stderr[:300])
        return
    got = dict((c, f) for c, f in json.loads(r.stdout))
    for c in ("HTTP 400", "HTTP 404", "HTTP 422", "HTTP 401", "HTTP 403"):
        check(f"{c} is NOT fatal", got[c] is False, str(got.get(c)))
    for c in ("HTTP 500", "HTTP 502", "HTTP 503", "Failed to fetch",
              "NetworkError when attempting to fetch resource."):
        check(f"{c} IS fatal", got[c] is True, str(got.get(c)))


def test_usejobs_guards() -> None:
    print("\n[2] useJobs has the guards useHealth has")
    poll = src("src/hooks/use-poll.ts")
    jobs = poll.split("export function useJobs")[1].split("export function useClips")[0]
    check("routes failures through triage", "triage(" in jobs)
    check("has a stale channel separate from error", "const [stale" in jobs)
    check("tracks whether it ever had data", "haveData" in jobs)
    check("requires consecutive failures", "FAILURES_BEFORE_ALARM" in jobs)
    check("counts failures", "failures.current" in jobs)
    check("resets the count on success", "failures.current = 0" in jobs)
    check("a 4xx never reaches setError",
          re.search(r"if \(!fatal\) \{[^}]*setError", jobs) is None,
          "a non-fatal path calls setError")
    check("a soft blip does not blank the job list", "setJobs(data)" in jobs)
    check("'Check now' is triaged too", jobs.count("applyFailure(") >= 2)
    check("returns the stale channel", "stale, refresh, checkOne" in jobs)
    # The threshold must be > 1, or one blip still alarms.
    n = re.search(r"FAILURES_BEFORE_ALARM = (\d+)", poll)
    check("threshold is more than one failure", n is not None and int(n.group(1)) >= 3,
          n.group(1) if n else "not found")


def test_banner_copy() -> None:
    print("\n[3] the banner names the route that actually failed")
    dash = src("src/pages/Dashboard.tsx")
    check("copy names /api/jobs, not /api/health",
          "/api/jobs" in dash and "on /api/health" not in dash,
          "still blames /api/health")
    check("reads the new stale channel", "stale: apiStale" in dash)
    check("renders a soft amber note for stale", "apiStale && !apiError" in dash)
    check("the amber note is not the red banner", "border-amber-500/40" in dash)
    check("the red banner stays for a real outage", "apiError &&" in dash)
    check("red banner still says 'API unreachable'", "API unreachable:" in dash)
    check("soft note says it is reachable",
          "server is reachable" in dash, "soft copy does not say it is up")


def test_health_unchanged() -> None:
    print("\n[4] the existing useHealth triage is intact")
    poll = src("src/hooks/use-poll.ts")
    health = poll.split("export function useHealth")[1]
    check("useHealth still refuses to alarm on 4xx",
          "status.startsWith(\"5\")" in health or "fatal = status === undefined" in health)
    check("useHealth still has its own stale channel", "const [stale" in health)


def main() -> int:
    test_triage_logic()
    test_usejobs_guards()
    test_banner_copy()
    test_health_unchanged()
    print(f"\napi_unreachable: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
