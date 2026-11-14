"""Measure the two windows behind "the preview is stuck at 97%".

Window A — port down: from the first 502 on /proxy/5173 to the first success.
Window B — port up but page still loading: from that first success to the first
  /proxy/5173/api/health, which the app only issues once React has mounted.
  This is how long our own boot card is on screen, and it is the window any
  progress display is judged against.

Usage: .venv/bin/python tools/preview_first_paint.py [/tmp/daytona-daemon.log]
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta

TS = re.compile(r"^time=(?P<ts>\S+)")
REQ = re.compile(r"request\.path=(?P<path>/proxy/5173\S*)")
CODE = re.compile(r"response\.status=(?P<status>\d+)")

IDLE = timedelta(minutes=5)


def parse(path: str) -> list[tuple[datetime, str, int]]:
    rows: list[tuple[datetime, str, int]] = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            head = TS.match(line)
            req = REQ.search(line)
            if not head or not req:
                continue
            try:
                ts = datetime.fromisoformat(head.group("ts").replace("Z", "+00:00"))
            except ValueError:
                continue
            code = CODE.search(line)
            rows.append((ts, req.group("path"), int(code.group("status")) if code else 0))
    return rows


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round((p / 100) * (len(values) - 1))))
    return values[idx]


def report(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label}: no samples")
        return
    s = sorted(values)
    print(
        f"{label}: n={len(s)} min={s[0]:.1f}s p50={pct(s, 50):.1f}s "
        f"p90={pct(s, 90):.1f}s max={s[-1]:.1f}s"
    )


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/daytona-daemon.log"
    rows = parse(path)
    print(f"proxied responses parsed: {len(rows)}")

    down: list[float] = []
    up_to_mount: list[float] = []
    boots = 0
    idx = 0
    while idx < len(rows):
        ts, req_path, status = rows[idx]
        if status != 502 or not req_path.startswith("/proxy/5173"):
            idx += 1
            continue
        # A boot = a run of failures, then the first success after it.
        last_bad = ts
        j = idx
        while j < len(rows) and rows[j][2] == 502:
            last_bad = rows[j][0]
            j += 1
        if j >= len(rows) or rows[j][0] - last_bad > IDLE:
            idx = j
            continue
        boots += 1
        up_at = rows[j][0]
        down.append((up_at - ts).total_seconds())
        # First /api/health at or after the port came up == React mounted.
        mount = None
        k = j
        while k < len(rows) and rows[k][0] - up_at <= IDLE:
            if rows[k][1].startswith("/proxy/5173/api/health"):
                mount = rows[k][0]
                break
            k += 1
        if mount is not None:
            up_to_mount.append((mount - up_at).total_seconds())
        idx = j

    report("A. port down  (502 -> first 200)", down)
    report("B. up -> mount (200 -> first /api/health)", up_to_mount)
    if up_to_mount:
        s = sorted(up_to_mount)
        print(
            f"   boot card visible for >10s in {sum(1 for v in s if v > 10)}/{len(s)} boots,"
            f" >20s in {sum(1 for v in s if v > 20)}/{len(s)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
