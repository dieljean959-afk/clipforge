"""The backend supervisor must not latch into a permanent dead state.

`vite-plugin-clipforge-backend.ts` respawns the FastAPI child when a proxied
/api request arrives and the child is gone. It used to give up for good once
`attempts` hit MAX_ATTEMPTS: "giving up on respawn. Reload the app to retry."
The web process is managed, so the user cannot do that — a backend that was
down for two minutes stayed down indefinitely. That is the "API unreachable"
that kept coming back.

The fix is a cooldown instead of a latch: still at most one spawn per
COOLDOWN_MS, but forever, so the failure heals on its own.

`backoffFor` is pure and is exercised here directly.

    python tools/backend_supervisor_test.py
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


def test_backoff_is_bounded() -> None:
    print("\n[1] the retry interval is always bounded (fork-bomb protection)")
    plugin = src("vite-plugin-clipforge-backend.ts")
    m = re.search(
        r"const BACKOFF_MS = \[([^\]]+)\];.*?const MAX_ATTEMPTS = (\d+);"
        r".*?const COOLDOWN_MS = ([\d_]+);",
        plugin, re.DOTALL,
    )
    check("constants present", m is not None)
    if not m:
        return
    backoff_src = m.group(1)
    max_attempts = int(m.group(2))
    cooldown = int(m.group(3).replace("_", ""))
    fn = (
        f"const BACKOFF_MS = [{backoff_src}];\n"
        f"const MAX_ATTEMPTS = {max_attempts};\n"
        f"const COOLDOWN_MS = {cooldown};\n"
    )
    fn += re.search(
        r"function backoffFor\(attempts: number\): number \{.*?\n\}",
        plugin, re.DOTALL,
    ).group(0)
    runner = fn + """
const out = [];
for (let a = 0; a <= 40; a++) out.push([a, backoffFor(a)]);
console.log(JSON.stringify(out));
"""
    r = subprocess.run(["node", "-e", runner], capture_output=True, text=True,
                       timeout=30, cwd=ROOT)
    if r.returncode != 0:
        check("node ran backoffFor", False, r.stderr[:300])
        return
    vals = dict((a, w) for a, w in json.loads(r.stdout))
    check("fast retries at first", vals[0] <= 1_000, str(vals[0]))
    check("backs off progressively", vals[2] > vals[0], str(vals))
    check("never exceeds the cooldown cap",
          all(v <= cooldown for v in vals.values()),
          f"max {max(vals.values())} vs cooldown {cooldown}")
    check("beyond the budget it settles at the cooldown",
          all(vals[a] == cooldown for a in range(max_attempts, 41)),
          "post-budget attempts are not the cooldown")
    # The important one: the interval stays FINITE far past the old ceiling.
    check("still retrying far past MAX_ATTEMPTS",
          vals[max_attempts + 5] == cooldown and vals[40] == cooldown,
          f"attempt {max_attempts+5} -> {vals[max_attempts+5]}")
    check("cooldown is long enough to be safe", cooldown >= 30_000, str(cooldown))


def test_no_permanent_giveup() -> None:
    print("\n[2] there is no permanent give-up path")
    plugin = src("vite-plugin-clipforge-backend.ts")
    # Strip comments: the fix's own explanation names the strings it removed,
    # and a naive substring check then fails on the explanation of the fix.
    code = re.sub(r"//.*", "", plugin)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    check("no 'giving up on respawn' in executable code",
          "giving up on respawn" not in code)
    check("no 'Reload the app to retry' in executable code",
          "Reload the app to retry" not in code)
    # The respawn branch must no longer be gated behind a give-up.
    respawn = code.split("backend down — respawning on demand")[0][-800:]
    check("respawn is not inside an else-if that can be skipped",
          "sinceLast >= wait" in respawn, "respawn still conditional on a give-up")
    check("exhausted budget is now only a log note", "crash-loop budget spent" in code)


def test_existing_guards_intact() -> None:
    print("\n[3] crash-loop protection is still there")
    plugin = src("vite-plugin-clipforge-backend.ts")
    check("MAX_ATTEMPTS still defined", "const MAX_ATTEMPTS" in plugin)
    check("respawn is still rate-limited by elapsed time",
          "Date.now() - state.lastAttemptAt" in plugin)
    check("concurrent respawns are still collapsed",
          "state.starting = state.starting ?? startBackend()" in plugin)
    check("only one child is tracked", "state.proc === null" in plugin)
    check("health probe still used", "/api/health" in plugin)
    check("teardown still kills the child", "state.proc.kill(\"SIGTERM\")" in plugin)


def test_frontend_triage_untouched() -> None:
    print("\n[4] the frontend triage still names this correctly")
    dash = src("src/pages/Dashboard.tsx")
    check("banner still says API unreachable for a real outage",
          "API unreachable:" in dash)
    check("banner names /api/jobs", "/api/jobs" in dash)
    check("soft stale note exists", "apiStale" in dash)


def main() -> int:
    test_backoff_is_bounded()
    test_no_permanent_giveup()
    test_existing_guards_intact()
    test_frontend_triage_untouched()
    print(f"\nbackend_supervisor: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
