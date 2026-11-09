"""Static mobile-overflow lint for ClipForge's phone problem areas.

A browser cannot be opened from this environment, so this does the next best
thing: it parses the compiled class strings out of the components that were
overflowing on a ~380px screen and asserts the layout constraints that make
overflow impossible are actually present. It caught two real omissions while
this fix was being written (a missing `min-w-0` on the sheet's row items, and
the PostMenu sheet itself missing `box-border`), which is the bar for keeping it.

    python tools/mobile_overflow_test.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

FAILS: list[str] = []
CHECKS = 0


def check(ok: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  ok   {label}")
    else:
        FAILS.append(label)
        print(f"  FAIL {label}")


def read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def classes_of(rel: str) -> str:
    """All double-quoted className literals in one file, joined."""
    return " ".join(re.findall(r'className="([^"]*)"', read(rel)))


def has_all(haystack: str, needles: list[str], label: str) -> None:
    missing = [n for n in needles if n not in haystack]
    check(not missing, f"{label}" + (f" (missing {missing})" if missing else ""))


# --------------------------------------------------------------------------- #
# The bottom sheet is the element the user literally could not read: wider than
# a 360px viewport and cut off on the right.
# --------------------------------------------------------------------------- #
print("[1] PostMenu share sheet stays inside the viewport")
sheet = classes_of("components/clipmake/PostMenu.tsx")
has_all(
    sheet,
    ["w-full", "max-w-[100vw]", "box-border", "overflow-x-hidden", "p-4"],
    "sheet is full-width, capped at the viewport, border-box, and clips x",
)
# Captions and titles must wrap rather than push: `whitespace-normal` where a
# long caption renders, and `[overflow-wrap:anywhere]` for the unbroken case.
has_all(
    sheet,
    ["whitespace-normal", "[overflow-wrap:anywhere]", "break-words"],
    "caption + title wrap (no nowrap, arbitrary break allowed)",
)
postmenu_all = read("components/clipmake/PostMenu.tsx")
check("whitespace-nowrap" not in postmenu_all, "no nowrap anywhere in PostMenu")

# Every action row inside the sheet needs min-w-0 on the flex rows so a long
# label shrinks instead of pushing the row wider. (The trigger button and the
# SheetContent itself are width-capped parents, not content rows — excluded.)
rows = [
    c
    for c in re.findall(r'className="([^"]*)"', postmenu_all)
    if "w-full" in c and "rounded-lg border" in c
]
check(rows, "found the action rows inside the sheet")
for i, row in enumerate(rows):
    check(
        "min-w-0" in row,
        f"sheet row {i + 1} can shrink (min-w-0): {row[:58]}...",
    )

# --------------------------------------------------------------------------- #
# The platform cards: label squeezed to one word per line beside the buttons.
# --------------------------------------------------------------------------- #
print("\n[2] SocialAccounts cards fill the row on mobile")
sa = read("components/clipmake/SocialAccounts.tsx")
check("flex-col" in sa and "sm:flex-row" in sa,
      "cards stack vertically on mobile, row from sm")
check("min-w-0" in sa, "card content is allowed to fill the row (min-w-0)")
check("border-l" not in sa, "no vertical timeline border next to the cards")

# --------------------------------------------------------------------------- #
# Page level: nothing may be able to widen the document.
# --------------------------------------------------------------------------- #
print("\n[3] Page shell cannot scroll horizontally")
dash = read("pages/Dashboard.tsx")
check("overflow-x-hidden" in dash, "status area clips horizontal overflow")
# The 380px sidebar column must be allowed to shrink on narrow viewports, or
# the grid's fixed track widens the page.
check(
    "lg:grid-cols-[minmax(0,1fr)_380px]" in dash,
    "sidebar grid track can shrink (minmax(0,1fr))",
)
check(dash.count("min-w-0") >= 3, "grid children opt out of intrinsic width (min-w-0)")

lwc = read("components/clipmake/LocalWorkerCard.tsx")
check("min-w-0" in lwc, "status card can shrink below its content width")

# --------------------------------------------------------------------------- #
# The banner is for real outages only.
# --------------------------------------------------------------------------- #
print("\n[4] Health banner only on network failure / 5xx, after retries")
poll = read("hooks/use-poll.ts")
check("failures.current" in poll, "failure counter for the retry rule")
check('failures.current < 3' in poll or "FAILURES_BEFORE_ALARM" in poll,
      "two silent retries before any banner")
check('status.startsWith("5")' in poll, "5xx is banner-worthy")
check("HTTP (\\d{3})" in poll.replace("\\\\", "\\"), "status parsed out of the message")
# The historical false alarm: a 400 on one poll must never set the banner.
# Scoped to the health segment on purpose — the file grew a shared `triage()`
# helper above it, and the old positional slice from the first
# `const status = ` landed in that helper and reported a false failure.
health = poll.split("export function useHealth")[1]
check('startsWith("4")' not in health, "4xx never becomes a banner in useHealth")
# The same rule now has to hold for the job poller, which is the hook that
# actually renders the "API unreachable" banner.
jobs = poll.split("export function useJobs")[1].split("export function useClips")[0]
check("triage(" in jobs, "useJobs triages failures")
check("setStale(" in jobs, "a 4xx becomes a soft note, not a banner")
check("FAILURES_BEFORE_ALARM" in jobs, "useJobs needs repeated failures to alarm")

print(f"\n{CHECKS} passed, {len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
