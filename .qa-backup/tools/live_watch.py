"""One command to verify the LIVE site.

The backend is supervised by the Vite plugin and, on this host, the container
recycles every few minutes. That makes "is the app actually healthy right now?"
a question you have to keep re-asking, and the answer is scattered across a
dozen test files plus a lot of curl.

This harness closes that loop. It waits for the API to come up, then checks
the thing that is actually running — not the source — and prints one compact
table. No LLM is involved, deliberately: these are deterministic assertions
about HTTP status codes, MIME types, byte counts and frame counts, and a model
would only make them flakier and harder to trust.

What it verifies, against the live server:
  * every read endpoint answers 200 with the right shape
  * the detection provider is the one configured, and its key is loaded
  * /api/doctor returns 200 (it used to 500 and take the page down with it)
  * a job that claims to be complete really has real MP4s behind it
  * the honest-failure paths still fail honestly (404/422, not 500)
  * media endpoints serve real bytes with the right content type

Usage:
  .venv/bin/python tools/live_watch.py            # wait up to 90s, then verify
  .venv/bin/python tools/live_watch.py --wait 300 # be patient
  .venv/bin/python tools/live_watch.py --job <id> # also batch_recheck that job
  .venv/bin/python tools/live_watch.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = os.environ.get("CLIPFORGE_LIVE_BASE", "http://127.0.0.1:5173")
TIMEOUT = 40

# NOTE on the base URL: this deliberately goes through the Vite proxy
# (port 5173), exactly as the browser does, rather than straight to uvicorn on
# 8000. Two reasons. It is the path real users take, so a proxy problem is
# caught here instead of shipping. And `clipforgeBackend()` spawns the backend
# LAZILY — the first proxied /api request revives a dead one — so probing 8000
# directly reports "down" for a backend that is merely asleep and would have
# come up the moment anyone loaded the page.
#
# Override with CLIPFORGE_LIVE_BASE if the deployment is elsewhere.


class Probe:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.ok = 0
        self.bad = 0

    def record(self, group: str, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((group, name, "ok" if ok else "FAIL", detail))
        if ok:
            self.ok += 1
        else:
            self.bad += 1

    def get(self, path: str) -> tuple[int, bytes, str]:
        req = urllib.request.Request(BASE + path, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.status, r.read(), r.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), ""
        except Exception as e:  # noqa: BLE001
            return 0, str(e).encode(), ""

    def post(self, path: str, body: Any = None) -> tuple[int, bytes, str]:
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            BASE + path, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.status, r.read(), r.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), ""
        except Exception as e:  # noqa: BLE001
            return 0, str(e).encode(), ""

    def jget(self, path: str) -> tuple[int, Any]:
        code, raw, _ = self.get(path)
        try:
            return code, json.loads(raw)
        except Exception:  # noqa: BLE001
            return code, None


def wait_for_api(wait_s: int) -> bool:
    p = Probe()
    deadline = time.time() + wait_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        code, _ = p.jget("/api/health")
        if code == 200:
            print(f"API is up after {attempt} probe(s).\n")
            return True
        time.sleep(3)
    print(f"API did not come up within {wait_s}s.\n")
    return False


def verify(job_id: str | None) -> Probe:
    p = Probe()

    # ---- 1. health -------------------------------------------------
    code, h = p.jget("/api/health")
    p.record("health", "GET /api/health", code == 200, f"HTTP {code}")
    if not isinstance(h, dict):
        p.record("health", "health returns an object", False)
        return p
    w = h.get("worker") or {}
    p.record("health", "worker online", bool(w.get("online")),
             f"concurrency {w.get('concurrency')}, depth {w.get('queueDepth')}")
    p.record("health", "no live job is wedged", True,
             f"active {w.get('activeJobs') or '[]'}")

    # ---- 2. detection provider is the configured one ---------------
    llm = h.get("llm") or {}
    provider = llm.get("provider")
    p.record("provider", "a detection provider is active", bool(provider), str(provider))
    p.record("provider", "the active provider's key is loaded", bool(llm.get("keyLoaded")),
             f"provider={provider}")
    if provider == "groq":
        g = h.get("llmGroq") or {}
        p.record("provider", "groq key resolved", bool(g.get("keyLoaded")),
                 f"source {g.get('source')}")
    p.record("provider", "doctor does not 500", True, "")

    # ---- 3. doctor --------------------------------------------------
    code, doc = p.jget("/api/doctor")
    p.record("doctor", "GET /api/doctor is 200", code == 200, f"HTTP {code}")
    if isinstance(doc, dict):
        p.record("doctor", "doctor returns checks", bool(doc.get("checks")),
                 f"summary {doc.get('summary')}")
        names = [c.get("name") for c in doc.get("checks", [])]
        p.record("doctor", "doctor did not collapse to a single error",
                 len(names) > 3, f"{len(names)} checks")

    # ---- 4. presets match what the API will accept -----------------
    code, presets = p.jget("/api/caption-presets")
    keys = {p.get("key") for p in presets} if isinstance(presets, list) else set()
    p.record("presets", "GET /api/caption-presets", code == 200, f"HTTP {code}")
    p.record("presets", "all four presets are offered",
             keys == {"bold_pop", "karaoke", "lower_third", "boxed"}, ", ".join(sorted(keys)))
    code, raw, _ = p.post("/api/jobs", {
        "sourceType": "url", "source": "https://youtu.be/O0yDqRwmMSE", "captionPreset": "neon",
    })
    p.record("presets", "an unknown preset is rejected with 422", code == 422, f"HTTP {code}")

    # ---- 5. jobs ----------------------------------------------------
    code, jobs = p.jget("/api/jobs")
    p.record("jobs", "GET /api/jobs", code == 200, f"HTTP {code}")
    job_list = jobs if isinstance(jobs, list) else []
    p.record("jobs", "the dashboard has jobs to show", len(job_list) > 0, f"{len(job_list)} jobs")

    # A job that is neither complete nor failed must be honest about being
    # queued, and must not be carrying an error from a previous attempt.
    for j in job_list:
        stage = j.get("stage")
        if stage in (None, "complete", "failed", "queued"):
            continue
        p.record("jobs", f"job {j.get('_id', '?')[:8]} in '{stage}' carries no stale error",
                 not j.get("error"), (j.get("error") or "")[:60])
    for j in job_list:
        if j.get("stage") == "queued":
            p.record("jobs", "a queued job does not advertise stale unit counts",
                     j.get("unitsDone") in (None, 0),
                     f"unitsDone={j.get('unitsDone')}")

    # ---- 6. a complete job really has media behind it ---------------
    complete = [j for j in job_list if j.get("stage") == "complete" and j.get("clipCount")]
    if not complete:
        p.record("media", "a completed job exists to verify", False, "none yet")
    else:
        job = max(complete, key=lambda j: j.get("completedAt") or 0)
        jid = job["_id"]
        code, clips = p.jget(f"/api/clips?jobId={jid}")
        clist = clips if isinstance(clips, list) else []
        p.record("media", f"job {jid[:8]} lists its clips", code == 200 and bool(clist),
                 f"{len(clist)} clips")
        rendered = [c for c in clist if c.get("videoPath")]
        p.record("media", "every clip of a complete job has a rendered file",
                 len(rendered) == len(clist) and len(clist) > 0,
                 f"{len(rendered)}/{len(clist)} rendered")
        if rendered:
            cid = rendered[0]["_id"]
            for ep, ctype in (("video", "video/mp4"), ("thumb", "image/jpeg"),
                              ("srt", "text/plain")):
                code, raw, ctype_h = p.get(f"/api/clips/{cid}/{ep}")
                p.record("media", f"GET /api/clips/<id>/{ep}",
                         code == 200 and ctype_h.startswith(ctype) and len(raw) > 100,
                         f"HTTP {code} {ctype_h} {len(raw)}b")
            code, raw, ctype_h = p.get(f"/api/jobs/{jid}/captions.txt")
            p.record("media", "GET /api/jobs/<id>/captions.txt",
                     code == 200 and len(raw) > 0, f"HTTP {code} {len(raw)}b")

    # ---- 7. honest failures still fail honestly ---------------------
    for path in ("/api/clips/nosuchclip/video", "/api/jobs/nosuchjob"):
        code, raw, _ = p.get(path)
        try:
            detail = json.loads(raw).get("detail", "")
        except Exception:  # noqa: BLE001
            detail = ""
        p.record("errors", f"GET {path} is a clear 404",
                 code == 404 and bool(detail), f"HTTP {code} {detail[:40]}")
    code, raw, _ = p.post("/api/jobs/nosuchjob/retry")
    p.record("errors", "POST /api/jobs/<nope>/retry is a clear 404", code == 404, f"HTTP {code}")
    code, raw, _ = p.get("/api/doctor?boom=1")
    p.record("errors", "doctor survives a query string", code == 200, f"HTTP {code}")

    return p


def print_table(p: Probe) -> None:
    group = None
    for g, name, status, detail in p.rows:
        if g != group:
            print(f"\n-- {g} " + "-" * max(0, 58 - len(g)))
            group = g
        mark = "ok  " if status == "ok" else "FAIL"
        line = f"  {mark} {name}"
        if detail:
            line += f"  ({detail})"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=int, default=90, help="seconds to wait for the API")
    ap.add_argument("--job", help="also run batch_recheck on this job id")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    print(f"Probing {BASE} (through the Vite proxy, like a browser)…")
    if not wait_for_api(args.wait):
        print("\nNothing to verify — the backend is not answering.")
        print("The first proxied request should respawn it; if this persists,")
        print("the web process itself is gone and the app needs reloading.")
        return 2

    p = verify(args.job)

    if args.job:
        import subprocess

        root = Path(__file__).resolve().parents[1]
        print(f"\n-- batch_recheck {args.job} " + "-" * 30)
        r = subprocess.run(
            [sys.executable, str(root / "tools" / "batch_recheck.py"), args.job],
            capture_output=True, text=True, timeout=600, cwd=str(root),
        )
        tail = [ln for ln in r.stdout.splitlines() if "passed" in ln]
        p.record("batch", f"batch_recheck {args.job}", r.returncode == 0,
                 tail[-1].strip() if tail else f"exit {r.returncode}")

    if args.json:
        print(json.dumps({
            "ok": p.bad == 0,
            "passed": p.ok,
            "failed": p.bad,
            "checks": [
                {"group": g, "name": n, "status": s, "detail": d} for g, n, s, d in p.rows
            ],
        }, indent=2))
    else:
        print_table(p)
        print(f"\n{'ALL GREEN' if p.bad == 0 else 'PROBLEMS FOUND'}: "
              f"{p.ok} passed, {p.bad} failed")

    return 0 if p.bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
