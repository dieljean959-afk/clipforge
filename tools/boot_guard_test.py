"""The boot guard must never turn a slow load into a dead page.

Regression cover for a self-inflicted outage. The boot fallback in index.html
used to declare failure at 15s and then reload 30 times at 5s intervals. In dev
the browser fetches hundreds of separate modules, and first paint of this app
is roughly 3 MB across a dozen round trips (1 MB react-dom, 1 MB integrations,
460 KB react-router, 186 KB CSS). On a phone that legitimately exceeds 15s — so
every reload discarded the modules already downloaded and restarted the same
transfer. The page could then NEVER finish loading, and the user was told the
dev server was down when it was serving 200s perfectly.

The rules pinned here:
  - a slow load is progress, not failure
  - a reload is only ever issued for a genuine module-load failure
  - that reload is bounded (8), not endless
  - a mount timeout never auto-reloads at all
  - progress is visible, so "slow" is distinguishable from "broken"
  - the dev-only toolbar is not on the critical path

    python tools/boot_guard_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


def src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_no_reload_storm() -> None:
    print("\n[1] no reload storm")
    html = src("index.html")

    check("there is no 15s hard mount timeout", "}, 15000)" not in html,
          "a setTimeout(..., 15000) mount deadline still exists")
    check("auto-reloads are bounded to 8", "MAX_RELOADS = 8" in html)
    check("the old 30-attempt loop is gone", "MAX_ATTEMPTS" not in html)
    check("the old 30x5s cadence is gone", "MAX_ATTEMPTS" not in html and "attempt " not in html)


def test_slow_is_not_failure() -> None:
    print("\n[2] a slow load is reported as progress")
    html = src("index.html")

    check("subresource loads are counted", 'observe({ type: "resource"' in html)
    check("progress extends the deadline", "lastProgressAt = Date.now()" in html)
    check("a stalled window means 'not progressing', not 'broken'",
          "idleFor < 4000" in html)
    check("the user is told it is a slow connection", "slow connection" in html)
    check("the module count is surfaced", "modules" in html)
    check("a real progress bar is rendered", 'id="boot-meter"' in html)

    # The load-bearing one: the give-up path must NOT reload.
    ceiling = html.split("elapsed > HARD_CEILING_MS && idleFor > 15000")[1]
    ceiling = ceiling.split("}, 1000);")[0]
    check("the give-up path never reloads",
          "hardReload" not in ceiling and "scheduleReload" not in ceiling,
          "the timeout branch schedules a reload")
    check("the give-up path explains why reloading will not help",
          "will not help" in ceiling)
    check("reloads only come from the module-failure path",
          html.count("scheduleReload()") == 2,
          f'found {html.count("scheduleReload()")} call sites (expect def + module-failure only)')


def test_real_failures_still_recover() -> None:
    print("\n[3] a genuine module failure still self-heals")
    html = src("index.html")
    check("a failed module script is caught",
          'e.target.tagName === "SCRIPT"' in html)
    check("a module URL is reported", "Failed to load " in html)
    check("that path schedules a reload", "failAndRetry" in html)
    check("the reload is cache-busted",
          "?reload=" in html and "Date.now()" in html)
    check("unhandled rejections are still caught", "unhandledrejection" in html)
    check("the manual Reload button survives", 'id="boot-reload"' in html)
    check("auto-retry can be switched off", 'id="boot-stop"' in html)
    check("main.tsx still signals a successful boot",
          "__clipforgeBooted" in src("src/main.tsx"))
    check("index.html still watches for that signal",
          "window.__clipforgeBooted" in html)


def test_payload() -> None:
    print("\n[4] the phone is not made to fetch dev-only chrome")
    main = src("src/main.tsx")
    check("the dev toolbar is lazy-loaded",
          'lazy(() => import("../vly-toolbar-readonly.tsx"))' in main)
    check("the toolbar is NOT eagerly imported",
          'import { VlyToolbar } from "../vly-toolbar-readonly.tsx"' not in main)
    check("the lazy toolbar is still wrapped in Suspense",
          "<Suspense fallback={null}>" in main)
    check("routes stay lazy", 'lazy(() => import("./pages/Dashboard.tsx"))' in main)
    check("the toolbar error boundary is kept", "ToolbarErrorBoundary" in main)


def main() -> int:
    test_no_reload_storm()
    test_slow_is_not_failure()
    test_real_failures_still_recover()
    test_payload()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
