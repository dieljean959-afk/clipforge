"""Measure how long the preview port is unreachable, per outage episode.

Reads the platform toolbox log and, for every run of `502 bad gateway` on
/proxy/5173, reports the wall-clock length of the run and the gap until the
first successful proxied request afterwards. That gap is exactly the window the
preview UI renders as "booting up preview" (97%).

Usage: .venv/bin/python tools/preview_boot_census.py [/tmp/daytona-daemon.log]
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta

LINE = re.compile(r"^time=(?P<ts>\S+)")

# A proxied request that reached 5173 successfully.
OK = re.compile(r'msg="Incoming request".*request\.path=/proxy/5173/')
# A proxied request the proxy could not reach (bad gateway / refused / reset).
BAD = re.compile(r'msg="HTTP error: 502 bad gateway".*request\.path=/proxy/5173/')

# Longest gap between two consecutive log lines before we call the log idle.
IDLE = timedelta(minutes=20)


def parse(path: str) -> list[tuple[datetime, bool]]:
    events: list[tuple[datetime, bool]] = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            head = LINE.match(line)
            if not head:
                continue
            try:
                ts = datetime.fromisoformat(head.group("ts").replace("Z", "+00:00"))
            except ValueError:
                continue
            if BAD.search(line):
                events.append((ts, False))
            elif OK.search(line):
                events.append((ts, True))
    return events


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/daytona-daemon.log"
    events = parse(path)
    print(f"proxied /proxy/5173 events: {len(events)}")

    episodes: list[tuple[datetime, datetime, timedelta | None]] = []
    idx = 0
    while idx < len(events):
        if events[idx][1]:
            idx += 1
            continue
        start = events[idx][0]
        end = start
        while idx < len(events) and not events[idx][1]:
            end = events[idx][0]
            idx += 1
        recovered: datetime | None = events[idx][0] if idx < len(events) else None
        # Only attribute a recovery to this episode if the log stayed busy;
        # otherwise the next OK belongs to a much later boot.
        if recovered is not None and recovered - end > IDLE:
            recovered = None
        gap = (recovered - end) if recovered is not None else None
        episodes.append((start, end, gap))
        if idx < len(events) and events[idx][0] - end > IDLE:
            continue

    print(f"outage episodes: {len(episodes)}\n")
    print(f"{'start':<21}{'last 502':<21}{'outage':>9}  {'to recovery':>11}")
    for start, end, gap in episodes[-25:]:
        span = f"{int((end - start).total_seconds())}s"
        tail = "never" if gap is None else f"{gap.total_seconds():.1f}s"
        print(f"{start:%m-%d %H:%M:%S}      {end:%H:%M:%S}      {span:>9}  {tail:>11}")

    gaps = [g.total_seconds() for _, _, g in episodes if g is not None]
    if gaps:
        gaps.sort()
        print(
            f"\nrecovery gaps: n={len(gaps)} "
            f"min={gaps[0]:.1f}s median={gaps[len(gaps) // 2]:.1f}s max={gaps[-1]:.1f}s"
        )
        over_60 = sum(1 for g in gaps if g > 60)
        print(f"episodes whose recovery took >60s: {over_60}/{len(gaps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
