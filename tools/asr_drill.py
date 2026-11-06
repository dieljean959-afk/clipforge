"""Failure drills for the memory-proof ASR pipeline.

    .venv/bin/python tools/asr_drill.py b   # kill -9 the ASR child mid-chunk
    .venv/bin/python tools/asr_drill.py c   # kill the whole job, then resume
    .venv/bin/python tools/asr_drill.py d   # orphan ASR process + cleanup

Every drill prints what it did and ends with a finished transcript or a label.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import subtitle, worker  # noqa: E402
from app.config import load_config  # noqa: E402

MEDIA = ROOT / "outputs" / "yt-audio" / "trailer.webm"
DRILLS = ROOT / "outputs" / "asr-drills"


def _settings(cache: Path):
    settings = load_config(quiet=True)
    paths = replace(settings.app.paths, data_dir=cache.parent / "data")
    return replace(settings, app=replace(settings.app, paths=paths))


def _children(name: str) -> list[int]:
    """PIDs whose cmdline mentions `name` and whose parent is us."""
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
            stat = (entry / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if name in cmdline and ppid == os.getpid():
            out.append(int(entry.name))
    return out


def _rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def drill_b() -> int:
    cache = DRILLS / "b" / "chunks"
    cache.mkdir(parents=True, exist_ok=True)
    for stale in cache.glob("chunk_*.json"):
        stale.unlink()
    settings = _settings(cache)
    box: dict = {}
    killed: dict = {}

    def run():
        try:
            box["t"] = subtitle.transcribe(
                MEDIA, settings, cache_dir=cache,
                on_detail=lambda text: print(f"DETAIL {text}", flush=True),
            )
        except Exception as exc:  # noqa: BLE001 - the drill reports it
            box["e"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while time.time() < deadline and not killed:
        pids = _children("app.asr_child")
        if pids:
            pid = pids[0]
            time.sleep(1.2)  # let the model finish loading and start decoding
            if Path(f"/proc/{pid}").exists():
                killed = {"pid": pid, "rss_mb": round(_rss_mb(pid), 1)}
                os.kill(pid, signal.SIGKILL)
                print(f"KILL -9 asr child pid={pid} rss={killed['rss_mb']} MB", flush=True)
                break
        time.sleep(0.1)
    thread.join(timeout=160)

    print(f"kill: {killed}", flush=True)
    if "e" in box:
        print(f"FAILED {type(box['e']).__name__}: {box['e']}", flush=True)
        return 1
    t = box["t"]
    died = [
        a for a in t.attempts
        if "died while decoding" in str(a.get("error"))
        or "ASR process exited with code" in str(a.get("error"))
    ]
    if died:
        print("child death recorded:", died[0].get("error"), flush=True)
    first = sorted(t.attempts, key=lambda a: a.get("start") or 0)[:3]
    print("first attempts after the kill:", json.dumps(first, default=str)[:600], flush=True)
    print(f"child-death attempts recorded: {len(died)}", flush=True)
    print(
        f"RESULT words={len(t.words)} label={t.label!r} models={sorted({a['model'] for a in t.attempts if a.get('ok')})} "
        f"chunks_on_disk={len(list(cache.glob('chunk_*.json')))}",
        flush=True,
    )
    return 0 if died and t.words else 1


def drill_c() -> int:
    cache = DRILLS / "c" / "chunks"
    cache.mkdir(parents=True, exist_ok=True)
    for stale in cache.glob("chunk_*.json"):
        stale.unlink()
    settings = _settings(cache)
    script = ROOT / "tools" / "asr_job_once.py"
    child = subprocess.Popen(
        [sys.executable, str(script), str(MEDIA), str(cache)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        if len(list(cache.glob("chunk_*.json"))) >= 1:
            break
        time.sleep(0.2)
    saved = len(list(cache.glob("chunk_*.json")))
    child.send_signal(signal.SIGKILL)
    child.wait(timeout=20)
    print(f"RESTART killed the job after {saved} checkpointed chunk(s)", flush=True)
    t = subtitle.transcribe(MEDIA, settings, cache_dir=cache)
    resumed = [a for a in t.attempts if a.get("resumed")]
    print(
        f"RESULT words={len(t.words)} resumed_chunks={len(resumed)} "
        f"label={t.label!r} chunks_on_disk={len(list(cache.glob('chunk_*.json')))}",
        flush=True,
    )
    return 0 if resumed and t.words else 1


def drill_d() -> int:
    data = DRILLS / "d" / "data"
    (data / "asr-tmp").mkdir(parents=True, exist_ok=True)
    stale = data / "asr-tmp" / "stale-span.wav"
    stale.write_bytes(b"RIFF")
    old = time.time() - 2 * 24 * 3600
    os.utime(stale, (old, old))
    fifo = data / "orphan.pipe"
    if fifo.exists():
        fifo.unlink()
    os.mkfifo(fifo)
    # The orphan-to-be must open the FIFO first (a writer blocks until a
    # reader appears), so start it in the background, then hold the write end
    # open: the child blocks on readline forever with no parent left.
    cfg = json.dumps({"model": "tiny", "limit_mb": 1200, "threads": 1})
    subprocess.run(
        ["sh", "-c",
         f"cd {ROOT} && {sys.executable} -m app.asr_child '{cfg}' < {fifo} >/dev/null 2>&1 &"],
        check=False,
    )
    writer = open(fifo, "w")
    time.sleep(6.0)  # let the orphan load its model and settle on readline
    orphans = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
            stat = (entry / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if "app.asr_child" in cmdline and ppid == 1:
            orphans.append(int(entry.name))
    print(
        "orphan asr child ppid=1: "
        + str([(p, round(_rss_mb(p), 1)) for p in orphans]),
        flush=True,
    )
    killed = worker.cleanup_orphans(data)
    writer.close()
    # A killed orphan reparented to init can linger as a zombie there; that is
    # dead for every purpose except /proc existing.
    def _dead(pid: int) -> bool:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            return True
        return state == "Z"

    still = [p for p in orphans if not _dead(p)]
    print(f"cleanup_orphans killed: {killed}", flush=True)
    print(f"orphans still alive: {still}; stale wav exists: {stale.exists()}", flush=True)
    return 0 if orphans and not still and not stale.exists() else 1


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "b"
    return {"b": drill_b, "c": drill_c, "d": drill_d}[which]()


if __name__ == "__main__":
    raise SystemExit(main())
