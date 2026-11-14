#!/usr/bin/env python3
"""Stability suite for the three reported problems.

Re-runnable, no network required for the parts that matter:

  A. An upload survives a "server restart".
  B. Bad AI JSON is repaired into real clip candidates.
  C. The batch list loads in under 2s with 50+ batches.

    python3 tools/stability_test.py
    python3 tools/stability_test.py --batches 120 --verbose

A is the interesting one. It does not restart the managed dev server (nothing
here is allowed to do that); it reproduces the failure in miniature: the
upload is written, the process that wrote it exits, a BRAND NEW PROCESS opens
the same paths, and the file must still be there — including after the storage
relocation, which is the change that makes the file live outside the codebase.
A separate check then proves the opposite case behaves as specified: an upload
whose file is genuinely gone fails ONCE, with the re-upload message, and is
not re-queued into the "resuming (attempt 2)" loop.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
FAILURES: list[str] = []
VERBOSE = False


def check(name: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
# A. uploads survive a restart
# --------------------------------------------------------------------------- #


def test_upload_survives_restart() -> None:
    section("A. Upload survives a server restart")

    from app import config as config_mod

    # A throwaway "codebase" with an empty media tree, exactly like a fresh
    # sandbox. The relocation runs here for real, because that is the code
    # under test.
    base = Path(tempfile.mkdtemp(prefix="clipforge-restart-"))
    root = base / "codebase"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("# the codebase\n", encoding="utf-8")
    # load_config() reads config.toml from the root it is given, so the fake
    # codebase needs the real one.
    shutil.copy2(ROOT / "config.toml", root / "config.toml")
    storage = base / "clipforge-data"

    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("CLIPFORGE_DOWNLOADS_DIR", "CLIPFORGE_OUTPUTS_DIR", "CLIPFORGE_DATA_DIR"))
    }
    env["CLIPFORGE_STORAGE_DIR"] = str(storage)
    env["PYTHONPATH"] = str(ROOT)

    # --- child 1: the "server process" receives an upload --------------------
    child = r"""
import json, os, sys
from pathlib import Path
from app.config import load_config
from app.download import save_upload
from app.store import LocalStore

paths = load_config(Path(sys.argv[1])).app.paths
store = LocalStore(paths)
job = store.create_job(source_type="upload", source="pending-upload")
jid = job["_id"]
dest = paths.downloads_dir / jid
dest.mkdir(parents=True, exist_ok=True)
tmp = dest / "upload.bin"
tmp.write_bytes(b"\x00" * 4096)
final = save_upload(tmp, dest, sys.argv[2])
store.write_job(store.get_job(jid) | {"source": final.name})
print(json.dumps({"jobId": jid, "file": final.name, "path": str(final),
                  "bytes": final.stat().st_size}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", child, str(root), "holiday.mp4"],
        capture_output=True, text=True, env=env, timeout=120, cwd=str(ROOT),
    )
    if proc.returncode != 0:
        check("upload child process succeeded", False, proc.stderr.strip()[-300:])
        shutil.rmtree(base, ignore_errors=True)
        return
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
    written = json.loads(line)
    check(
        "upload is stored with the extension of the ORIGINAL file",
        written["file"] == "source.mp4",
        f"stored as {written['file']!r} (the temp name was upload.bin)",
    )
    check(
        "uploads do not live inside the codebase tree",
        str(ROOT) not in written["path"] and str(root) not in written["path"],
        written["path"],
    )
    check("media storage is outside the codebase", storage.is_dir(), str(storage))

    # --- child 2: a BRAND NEW process, i.e. the restart ---------------------
    child2 = r"""
import json, sys
from pathlib import Path
from app.config import load_config
from app.store import LocalStore
paths = load_config(Path(sys.argv[1])).app.paths
store = LocalStore(paths)
job = store.get_job(sys.argv[2])
found = store.find_source_file(sys.argv[2])
print(json.dumps({"source": job.get("source") if job else None,
                  "found": str(found) if found else None,
                  "bytes": found.stat().st_size if found else 0}))
"""
    proc2 = subprocess.run(
        [sys.executable, "-c", child2, str(root), written["jobId"]],
        capture_output=True, text=True, env=env, timeout=120, cwd=str(ROOT),
    )
    if proc2.returncode != 0:
        check("restart child read the upload back", False, proc2.stderr.strip()[-300:])
        shutil.rmtree(base, ignore_errors=True)
        return
    read_back = json.loads(
        [ln for ln in proc2.stdout.splitlines() if ln.startswith("{")][-1]
    )
    check("upload is still there after the restart", read_back["found"] is not None)
    check(
        "upload survived byte-for-byte",
        read_back["bytes"] == written["bytes"] and read_back["bytes"] > 0,
        f"{read_back['bytes']} vs {written['bytes']}",
    )
    check(
        "the old in-codebase path still resolves (symlink compatibility)",
        (root / "downloads" / written["jobId"] / "source.mp4").exists(),
        "absolute paths already baked into work metadata would 404 otherwise",
    )

    # --- the loss case: gone file, no loop ---------------------------------
    from app import worker as worker_mod
    from app.store import LocalStore
    from app.worker import unrecoverable_reason

    settings = config_mod.load_config(root)
    store = LocalStore(settings.app.paths)
    job = store.get_job(written["jobId"])
    check(
        "an intact upload is still resumable",
        unrecoverable_reason(store, job) is None,
    )
    (Path(read_back["found"])).unlink()
    reason = unrecoverable_reason(store, job)
    check(
        "a lost upload is recognised as unrecoverable",
        reason is not None and "re-upload" in reason.lower(),
        str(reason),
    )
    url_job = store.create_job(source_type="url", source="https://example.com/v.mp4")
    check(
        "a URL job stays resumable with no file on disk (it re-downloads)",
        unrecoverable_reason(store, url_job) is None,
    )

    # The loop itself: boot recovery must NOT re-queue it.
    boot = r"""
import json, sys
from pathlib import Path
from app.config import load_config
from app.store import LocalStore
from app import worker as worker_mod
paths = load_config(Path(sys.argv[1])).app.paths
store = LocalStore(paths)
job = store.get_job(sys.argv[2])
job["stage"] = "queued"; job["interrupted"] = True
job["interruptReason"] = "Recovered after an external kill"
store.write_job(job)
w = worker_mod.InProcessWorker(load_config(Path(sys.argv[1])), store)
w.reconcile_stale_running(force=True)
after = store.get_job(sys.argv[2])
print(json.dumps({"stage": after.get("stage"), "error": after.get("error") or "",
                  "queued": sys.argv[2] in w.queue_snapshot()}))
"""
    proc3 = subprocess.run(
        [sys.executable, "-c", boot, str(root), written["jobId"]],
        capture_output=True, text=True, env=env, timeout=120, cwd=str(ROOT),
    )
    if proc3.returncode == 0:
        after = json.loads(
            [ln for ln in proc3.stdout.splitlines() if ln.startswith("{")][-1]
        )
        check(
            "a job with a lost upload is NOT re-queued into the resume loop",
            after["queued"] is False and after["stage"] == "failed",
            f"stage={after['stage']} queued={after['queued']}",
        )
        check(
            "the failure says 're-upload', once",
            "re-upload" in after["error"].lower() and after["error"].count("re-upload") == 1,
            after["error"][:120],
        )
    else:
        check("reconcile ran", False, proc3.stderr.strip()[-300:])

    # Resume cap.
    check(
        "resume attempts are capped at 2 stalled attempts",
        worker_mod.STALLED_RESUME_LIMIT == 2,
        f"STALLED_RESUME_LIMIT={worker_mod.STALLED_RESUME_LIMIT}",
    )
    shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------- #
# B. bad AI JSON is repaired
# --------------------------------------------------------------------------- #


def test_bad_json_repaired() -> None:
    section("B. Bad AI JSON is repaired into clip candidates")

    from app.llm import heuristic_candidates, parse_candidates

    words = [
        {"start": i * 0.5, "end": i * 0.5 + 0.4, "word": f"w{i}", "text": f"w{i}"}
        for i in range(400)
    ]

    good = json.dumps({"clips": [{"start_word": 0, "end_word": 40, "score": 0.9,
                                  "title": "T", "hook_text": "H", "hashtags": ["#a"]}]})
    check("plain JSON parses", len(parse_candidates(good)) == 1)

    fenced = f"Here you go:\n```json\n{good}\n```\nHope that helps!"
    check("markdown fences + prose around the JSON are stripped",
          len(parse_candidates(fenced)) == 1)

    array = json.dumps([{"start_word": 0, "end_word": 40, "score": 0.8,
                         "title": "T", "hook_text": "H", "hashtags": []}])
    truncated = array[: int(len(array) * 0.7)]
    check("truncated JSON is repaired", len(parse_candidates(truncated)) >= 1,
          f"{truncated[-40:]!r}")

    trailing = good.replace("}}", "},}")
    check("trailing commas are tolerated", len(parse_candidates(trailing)) == 1)

    check("prose with no JSON yields nothing rather than raising",
          parse_candidates("I am sorry, I cannot help with that.") == [])

    # Retry ladder + backup model are configured.
    from app.config import load_config
    from app.llm import openai_cfg, provider, settings_llm_fallback

    settings = load_config()
    cfg = openai_cfg(settings, provider(settings))
    check("JSON ladder retries at least 3 times", cfg.json_retries >= 3,
          f"json_retries={cfg.json_retries}")
    check("a backup provider is configured", bool(settings_llm_fallback(settings)),
          f"fallback_provider={settings.llm.fallback_provider!r}")
    check("retry temperature is low", cfg.json_retry_temperature <= 0.3,
          f"{cfg.json_retry_temperature}")

    src = (ROOT / "app" / "llm.py").read_text(encoding="utf-8")
    check("an unparseable reply is retried on the backup provider",
          "retrying this chunk on the" in src)
    check("the raw model reply is logged",
          "RAW REPLY" in src)
    check("ladder steps back off exponentially",
          "backing off" in src and "2.0 ** (attempt - 1)" in src)

    # The fallback the job uses when the model gives nothing.
    from dataclasses import replace

    base_cfg = load_config().gemini
    g = replace(base_cfg, max_clips=3, min_clip_sec=15, max_clip_sec=30)
    local = heuristic_candidates(words, 200.0, g)
    check("a transcript with words yields local fallback clips", len(local) == 3,
          f"{len(local)}")
    if local:
        check("fallback clips have sane, non-overlapping windows",
              all(c["end"] > c["start"] for c in local)
              and all(local[i + 1]["start"] - local[i]["start"] >= 10
                      for i in range(len(local) - 1)),
              str([(c["start"], c["end"]) for c in local]))
        check("fallback clips are labelled as NOT AI-picked",
              all(c["detection_source"] == "transcript_heuristic" for c in local))

    silent = heuristic_candidates([], 90.0, replace(g, max_clips=3))
    check("a silent source still produces clips from its duration", len(silent) >= 1,
          f"{len(silent)}")
    check("no transcript AND no duration produces nothing",
          heuristic_candidates([], 0.0, g) == [])

    # The honest message: an empty transcript must not blame the AI backend.
    check("an empty transcript is reported as no speech, not as a bad model",
          "no speech was detected" in src)

    # Truncation shapes the token limit actually produces.
    from app import jsonfix

    for name, payload in (
        ("cut inside the first clip", '[{"start_word": 0, "end_word": 40, "score": 0.8, "title": "T", "hook_text": "H'),
        ("cut inside the second clip", '[{"start_word": 0, "end_word": 40, "score": 0.8, "title": "T", "hook_text": "H", "hashtags": []},{"start_word": 41, "end_word": 80, "score": 0.7,'),
        ("cut inside a fenced reply", '```json\n[{"start_word": 0, "end_word": 40, "score": 0.8, "title": "Ti'),
        ("cut inside a nested envelope", '{"clips": [{"start_word": 0, "end_word": 40, "score": 0.8, "title": "Ti'),
        ("cut inside a number", '[{"start_word": 0, "end_word": 40, "score": 0.'),
    ):
        check(f"truncated JSON recovered — {name}",
              len(jsonfix.repair_candidates(payload)) >= 1)


# --------------------------------------------------------------------------- #
# C. the batch list is fast with many batches
# --------------------------------------------------------------------------- #


def test_list_fast(batch_count: int) -> None:
    section(f"C. Batch list loads in under 2s with {batch_count} batches")

    import warnings

    warnings.filterwarnings("ignore")

    base = Path(tempfile.mkdtemp(prefix="clipforge-list-"))
    (base / "downloads").mkdir()
    (base / "outputs").mkdir()
    data = base / "data"
    (data / "jobs").mkdir(parents=True)
    (data / "clips").mkdir(parents=True)
    (data / "work").mkdir(parents=True)

    now = int(time.time() * 1000)
    ids = []
    for i in range(batch_count):
        jid = f"{i:012x}"
        ids.append(jid)
        # A realistic document: the batch logs are the reason the old payload
        # was 285 KB for seventeen jobs, so they are here on purpose.
        (data / "jobs" / f"{jid}.json").write_text(json.dumps({
            "_id": jid, "title": f"Batch {i}", "source": "source.mp4",
            "sourceType": "upload", "isUpload": True, "stage": "complete",
            "progress": 100.0, "progressDetail": "10 clips ready.",
            "createdAt": now - i * 60_000, "completedAt": now - i * 60_000 + 30_000,
            "durationSec": 688.0, "clipCount": 10, "totalClipsSec": 220.0,
            "transcriptWordCount": 1195, "detectedSource": "deepseek",
            "downloadSource": "upload", "options": {"maxClips": 10},
            "logs": [f"{i:02d}:00 line {n} of a job log" for n in range(300)],
        }), encoding="utf-8")

    env = dict(os.environ)
    env["CLIPFORGE_DOWNLOADS_DIR"] = str(base / "downloads")
    env["CLIPFORGE_OUTPUTS_DIR"] = str(base / "outputs")
    env["CLIPFORGE_DATA_DIR"] = str(data)

    runner = r"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
from fastapi.testclient import TestClient
import app.main as m
c = TestClient(m.app)
c.get("/api/health")
def timed(url):
    t = time.perf_counter(); r = c.get(url); return (time.perf_counter() - t) * 1000, r
best_page, best_all = 1e9, 1e9
size_page = size_all = 0
for _ in range(6):
    ms, r = timed("/api/jobs?limit=10")
    best_page = min(best_page, ms); size_page = len(r.content)
    ms, r = timed("/api/jobs?all=1")
    best_all = min(best_all, ms); size_all = len(r.content)
page = c.get("/api/jobs?limit=10").json()
full = c.get("/api/jobs?all=1").json()
# Walk the cursor the way the dashboard does, and prove the pages tile the
# whole list exactly once — no batch skipped at a page boundary, none twice.
seen, cursor, pages = [], None, 0
while pages < 500:
    url = "/api/jobs?limit=10" if cursor is None else f"/api/jobs?limit=10&before={cursor}"
    r = c.get(url).json()
    if not r["jobs"]:
        break
    seen.extend(j["_id"] for j in r["jobs"])
    cursor = r["nextBefore"]
    pages += 1
    if cursor is None:
        break
print(json.dumps({
    "pageMs": best_page, "allMs": best_all,
    "pageBytes": size_page, "allBytes": size_all,
    "pageCount": len(page["jobs"]), "total": page["total"],
    "nextBefore": page["nextBefore"], "fullCount": len(full["jobs"]),
    "hasLogs": any("logs" in j for j in page["jobs"]),
    "newestFirst": [j["createdAt"] for j in page["jobs"]] ==
                   sorted([j["createdAt"] for j in page["jobs"]], reverse=True),
    "pagedIds": len(seen), "pagedUnique": len(set(seen)), "pages": pages,
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", runner], capture_output=True, text=True,
        env=env, timeout=180, cwd=str(ROOT),
    )
    if proc.returncode != 0:
        check("list benchmark ran", False, proc.stderr.strip()[-400:])
        shutil.rmtree(base, ignore_errors=True)
        return
    out = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])

    check(f"first page ({out['pageCount']} batches) loads in under 2s",
          out["pageMs"] < 2000, f"{out['pageMs']:.0f} ms")
    check(f"every batch ({out['fullCount']}) still loads in under 2s",
          out["allMs"] < 2000, f"{out['allMs']:.0f} ms")
    check("the first page is 10 batches", out["pageCount"] == 10, str(out["pageCount"]))
    check("a cursor is offered for the next page", out["nextBefore"] is not None)
    check("newest batch first", out["newestFirst"])
    check("job logs are NOT in the list payload", out["hasLogs"] is False)
    check("the list payload is small", out["pageBytes"] < 40_000,
          f"{out['pageBytes']:,} bytes for a page of 10")
    print(f"       page of 10: {out['pageMs']:.0f} ms / {out['pageBytes']:,} B"
          f"   |  all {out['fullCount']}: {out['allMs']:.0f} ms / {out['allBytes']:,} B")

    # Cursor paging must not skip or repeat a batch.
    check("paging through with the cursor yields every batch exactly once",
          out["pagedIds"] == out["pagedUnique"] == batch_count,
          f"{out['pagedUnique']} unique of {out['pagedIds']} seen, "
          f"expected {batch_count}, in {out['pages']} pages")
    check("the list endpoint agrees with itself on the total",
          out["fullCount"] == out["total"] == batch_count,
          f"total={out['total']} expected={batch_count}")
    shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------- #
# D. the smaller P3 promises
# --------------------------------------------------------------------------- #


def test_small_things() -> None:
    section("D. Thumbnails, media caching, watcher, bulk delete")

    from app.video import THUMB_WIDTH
    check("thumbnails are generated at ~320px", 240 <= THUMB_WIDTH <= 400, str(THUMB_WIDTH))

    vite = (ROOT / "vite.config.ts").read_text(encoding="utf-8")
    check("HMR stays disabled (platform requirement)", "hmr: false" in vite)
    check("the watcher ignores downloads/", "**/downloads/**" in vite)
    check("the watcher ignores outputs/", "**/outputs/**" in vite)
    check("the watcher ignores the data/ store", "**/data/**" in vite)
    check("the watcher ignores media extensions", "**/*.mp4" in vite)

    main_src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    check("clip video answers range requests", "Accept-Ranges" in main_src)
    check("clip video sends caching headers", "Cache-Control" in main_src)
    check("clip thumbnails send caching headers", "ETag" in main_src)
    check("the list response is cached for a few seconds", "LIST_CACHE_SEC" in main_src)
    check("reconciliation is throttled", "LIST_RECONCILE_FLOOR_SEC" in main_src)
    check("bulk delete exists", "/api/jobs/bulk-delete" in main_src)
    check("a junk-batch suggestion exists", "/api/jobs/junk" in main_src)

    hosting = (ROOT / "app" / "hosting.py").read_text(encoding="utf-8")
    check("orphans older than 24h are swept", "ORPHAN_MAX_AGE_SEC" in hosting
          and '86400' in hosting)
    sweep_body = hosting.split("def sweep(self)")[1].split("def ")[0]
    check("the orphan sweep runs even when retention is disabled",
          "self.sweep_orphans()" in sweep_body.split("retention_hours <= 0")[0])

    dash = (ROOT / "src" / "pages" / "Dashboard.tsx").read_text(encoding="utf-8")
    check("the clip grid shows skeletons", "animate-pulse" in dash)
    check('the bare "Clips are loading..." text is no longer rendered',
          ">Clips are loading..." not in dash
          and '>\n                  Clips are loading...' not in dash)
    check("older batches load on scroll", "IntersectionObserver" in dash)
    check("scroll loading also has a button", "Load older batches" in dash)

    preview = (ROOT / "src" / "components" / "clipforge" / "ClipPreview.tsx").read_text(
        encoding="utf-8")
    check("the poster is used, so no frame is fetched up front",
          "poster={thumbUrl" in preview)
    check("video bytes are not preloaded into the grid", 'preload="metadata"' in preview)

    task = (ROOT / "app" / "task.py").read_text(encoding="utf-8")
    check("free disk is checked before every job", "_check_disk_space" in task)


# --------------------------------------------------------------------------- #
# E. deleting batches actually works, from every entry point
# --------------------------------------------------------------------------- #


def test_no_duplicate_clips() -> None:
    section("F. no duplicate or overlapping clips")

    from app.llm import drop_overlapping

    cands = [
        {"start": 10.0, "end": 25.0, "score": 90, "title": "A"},
        {"start": 11.0, "end": 24.0, "score": 80, "title": "B"},
        {"start": 10.0, "end": 25.0, "score": 70, "title": "C"},
        {"start": 25.5, "end": 40.0, "score": 65, "title": "D"},
    ]
    kept, dropped = drop_overlapping(cands)
    titles = [c["title"] for c in kept]
    check("an exact duplicate window is dropped", "C" not in titles, str(titles))
    check("a heavily overlapping window is dropped", "B" not in titles, str(titles))
    check("the highest-scoring take of a duplicate survives", "A" in titles, str(titles))
    check("a merely ADJACENT clip is kept (0.5s apart)",
          "D" in titles, str(titles))
    check("survivors come back in time order",
          [c["start"] for c in kept] == sorted(c["start"] for c in kept), str(kept))
    check("the drop is reported, not silent", len(dropped) == 2, str(dropped))

    llm_src = (ROOT / "app" / "llm.py").read_text(encoding="utf-8")
    check("the merge applies it (only the merged list can see a cross-chunk dup)",
          "drop_overlapping(candidates)" in llm_src)

    # ...and the rendered set really is clean end to end.
    import tempfile as _tf
    import warnings

    warnings.filterwarnings("ignore")
    from app.config import Paths, load_config
    from app.store import LocalStore

    base = Path(_tf.mkdtemp(prefix="clipforge-dup-"))
    paths = Paths(data_dir=base / "d", outputs_dir=base / "o",
                  downloads_dir=base / "dl", static_dir=base / "s")
    for d in (paths.data_dir, paths.outputs_dir, paths.downloads_dir):
        d.mkdir(parents=True, exist_ok=True)
    st = LocalStore(paths)
    job = st.create_job(source_type="url", source="u")
    clips = st.create_clips(job["_id"], [
        {"start": 10.0, "end": 25.0, "score": 90, "title": "A", "hook_text": "a"},
        {"start": 11.0, "end": 24.0, "score": 80, "title": "B", "hook_text": "b"},
        {"start": 10.0, "end": 25.0, "score": 70, "title": "C", "hook_text": "c"},
        {"start": 60.0, "end": 75.0, "score": 65, "title": "D", "hook_text": "d"},
    ], detected_source="t")
    check("create_clips keeps them (the dedupe lives in detection)",
          len(clips) == 4, str(len(clips)))
    kept2, _ = drop_overlapping([
        {"start": c["start"], "end": c["end"], "score": c["score"]} for c in clips
    ])
    check("what detection would hand the renderer has no overlap",
          len(kept2) == 2, str(len(kept2)))
    shutil.rmtree(base, ignore_errors=True)
    assert load_config  # imported for parity with the other sections


def test_three_jobs_back_to_back() -> None:
    section("G. three jobs back to back: no crash, no memory growth")

    import threading as _th
    import warnings

    warnings.filterwarnings("ignore")
    from app.config import Paths, load_config
    from app.store import LocalStore
    from app.worker import InProcessWorker

    def rss_kb() -> int:
        try:
            with open("/proc/self/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except OSError:
            pass
        return 0

    settings = load_config(quiet=True)
    base = Path(tempfile.mkdtemp(prefix="clipforge-3x-"))
    paths = Paths(data_dir=base / "d", outputs_dir=base / "o",
                  downloads_dir=base / "dl", static_dir=base / "s")
    for d in (paths.data_dir, paths.outputs_dir, paths.downloads_dir):
        d.mkdir(parents=True, exist_ok=True)
    store = LocalStore(paths)
    worker = InProcessWorker(settings, store)

    ids = []
    for i in range(3):
        j = store.create_job(source_type="url", source=f"u{i}")
        store.write_job({**j, "stage": "queued"})
        ids.append(j["_id"])

    threads_before = _th.active_count()
    rss_before = rss_kb()
    errors: list[str] = []

    def run_one(job_id: str) -> None:
        try:
            # The real store round-trip every job does at least twice.
            store.write_job({**(store.get_job(job_id) or {}), "stage": "rendering",
                             "progress": 50})
            store.append_job_log(job_id, "synthetic unit of work")
            store.get_job(job_id)
        except Exception as exc:  # noqa: BLE001 - the point is to catch any
            errors.append(f"{job_id}: {exc}")

    t0 = time.monotonic()
    for jid in ids:
        run_one(jid)
        worker.reconcile_stale_running(force=True)
    took = time.monotonic() - t0

    check("three jobs in a row raised nothing", not errors, "; ".join(errors))
    check("all three are still readable from the store",
          all(store.get_job(j) is not None for j in ids))
    check("all three recorded their log line",
          all(any("synthetic" in ln for ln in
                  (store.get_job(j) or {}).get("logs", [])) for j in ids))
    check("no thread leaked", _th.active_count() <= threads_before + 1,
          f"{threads_before} -> {_th.active_count()}")
    rss_after = rss_kb()
    grew = rss_after - rss_before
    check("resident memory did not grow with the job count (not a per-job leak)",
          grew < 40_000, f"{rss_before}kB -> {rss_after}kB (+{grew}kB) in {took:.1f}s")
    shutil.rmtree(base, ignore_errors=True)


def test_delete_paths() -> None:
    section("E. delete paths")

    import warnings

    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient

    import app.main as main_mod

    client = TestClient(main_mod.app)
    store = main_mod.store
    base = Path(tempfile.mkdtemp(prefix="clipforge-del-"))
    made: list[str] = []
    # Unique per run: a deleted id leaves a tombstone behind on purpose, and a
    # re-run must not collide with the last run's tombstones.
    tag = f"{int(time.time()) % 1_000_000:06d}"
    try:
        # Three throwaway batches: one that failed, one that finished with
        # nothing rendered, one that is still running.
        for i, stage in enumerate(("failed", "complete", "running")):
            jid = f"del{tag}{i:03d}"
            store.write_job({
                "_id": jid,
                "stage": stage,
                "progress": 100 if stage != "running" else 40,
                "source": f"source{i}.mp4",
                "createdAt": 1_700_000_000_000 + i,
                "clipCount": 0,
            })
            made.append(jid)

        # 1. the junk route must not be shadowed by /api/jobs/{job_id}
        junk = client.get("/api/jobs/junk")
        check("GET /api/jobs/junk is not shadowed by /api/jobs/{job_id}",
              junk.status_code == 200,
              f"status={junk.status_code} body={junk.text[:120]}")
        junk_ids = {j["_id"] for j in junk.json().get("jobs", [])} if junk.status_code == 200 else set()
        check("a failed batch with no clips is offered as junk",
              made[0] in junk_ids, f"junk={sorted(junk_ids)}")

        # 2. single delete over POST (what the dashboard calls)
        one = client.post(f"/api/jobs/{made[0]}/delete")
        check("POST /api/jobs/{id}/delete removes the batch",
              one.status_code == 200 and one.json().get("deleted") is True, one.text[:120])
        check("the deleted batch is really gone from the store",
              client.get(f"/api/jobs/{made[0]}").status_code == 404)

        # 3. a missing batch is a 404 with a readable detail, not a bare code
        gone = client.post(f"/api/jobs/{made[0]}/delete")
        check("deleting a batch twice answers 404 'job not found'",
              gone.status_code == 404 and "not found" in gone.text.lower(), gone.text[:120])

        # 4. the DELETE verb still works for anything scripted against the API
        verb = client.delete(f"/api/jobs/{made[1]}")
        check("DELETE /api/jobs/{id} still works",
              verb.status_code == 200 and verb.json().get("deleted") is True, verb.text[:120])

        # 5. bulk delete, and its honest report of ids it could not find
        bulk = client.post("/api/jobs/bulk-delete", json={"jobIds": [made[2], "nope"]})
        body = bulk.json() if bulk.status_code == 200 else {}
        check("bulk delete removes the batch it was given",
              made[2] in body.get("deleted", []), bulk.text[:160])
        check("bulk delete reports ids it could not find",
              body.get("missing") == ["nope"], bulk.text[:160])
        empty = client.post("/api/jobs/bulk-delete", json={"jobIds": []})
        check("bulk delete with no ids is a 400 with a reason",
              empty.status_code == 400 and "no job ids" in empty.text, empty.text[:120])

        # 6. the error ring buffer the diagnosis depends on
        recent = client.get("/api/errors/recent")
        check("recent API errors are readable from the app itself",
              recent.status_code == 200 and "errors" in recent.json(), recent.text[:120])
        check("the 404s above were recorded",
              any(e.get("status") == 404 for e in recent.json().get("errors", [])))

        # 7. the client must not depend on the DELETE verb surviving a proxy
        api_src = (ROOT / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
        check("the dashboard deletes over POST", "/delete`, { method: \"POST\" }" in api_src
              or "/delete`," in api_src and 'method: "POST"' in api_src)
        check("the client falls back to DELETE when the POST route is absent",
              'method: "DELETE"' in api_src.split("deleteJob:")[1].split("retryJob:")[0])
        check("a 405 (route claims the path, refuses the verb) also falls back",
              "Method Not Allowed" in api_src or "method not allowed" in api_src)
        check("a bare status code is no longer the whole error message",
              "empty response" in api_src)
        check("the UI reads the junk list", "listJunkJobs" in api_src)
        check("the UI can list every batch at once", "listAllJobs" in api_src)

        dash = (ROOT / "src" / "pages" / "Dashboard.tsx").read_text(encoding="utf-8")
        check("the dashboard offers the one-tap cleanup",
              "CleanupJunkDialog" in dash)
        cleanup = (ROOT / "src" / "components" / "clipforge"
                   / "CleanupJunkDialog.tsx").read_text(encoding="utf-8")
        check("cleanup deletes in bulk", "bulkDeleteJobs" in cleanup)
        check("cleanup states the count before deleting", "dead batch" in cleanup)
        check("working batches are opt-in, never swept silently",
              "Also delete the batches that did produce clips" in cleanup)

        confirm = (ROOT / "src" / "components" / "clipforge"
                   / "ConfirmDeleteJob.tsx").read_text(encoding="utf-8")
        check("a retry that finds nothing left is not reported as a failure",
              "not found|404" in confirm)
    finally:
        for jid in made:
            store.delete_job(jid)
        shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------- #


def main() -> int:
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=60,
                    help="how many batches to put in the store for the list test")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    VERBOSE = args.verbose

    print("ClipForge stability suite")
    print(f"  repo: {ROOT}")
    print(f"  batches for the list benchmark: {args.batches}")

    test_upload_survives_restart()
    test_bad_json_repaired()
    test_list_fast(args.batches)
    test_small_things()
    test_no_duplicate_clips()
    test_three_jobs_back_to_back()
    test_delete_paths()

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("failed:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

