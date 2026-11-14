#!/usr/bin/env python3
"""ClipForge worker-agent — run this on your home machine (over Tailscale).

The hosted dashboard dispatches the heavy pipeline (whisper transcription,
Gemini detection, ffmpeg rendering) to this agent instead of running it on the
platform. Pull-model: the agent registers itself, then polls for jobs, so no
inbound ports are needed on your side beyond the tailnet.

Setup:
  1. Install Tailscale on the home machine, note its 100.x.x.x IP.
  2. On the same machine: python3.11+, ffmpeg, and `pip install -e .` from
     this repo (the agent reuses app/ services).
  3. Run:
       WORKER_TOKEN=<same-as-server> python3 tools/worker_agent.py \
           --server http://<clipforge-host> --listen 0.0.0.0:8765
     (or use --register to push your tailnet address to the server's
      WORKER_URL once, instead of configuring the platform env by hand)
  4. Set WORKER_URL=http://<your-tailnet-ip>:8765 and WORKER_TOKEN on the
     ClipForge deployment (Keys/API keys), restart, and flip to dispatch mode.

Endpoints (bound to the tailnet by tailscale serve, or your LAN):
  GET  /agent/ping        → liveness + queue depth
  POST /agent/jobs        → multipart upload of source + job spec
  GET  /agent/output/{id}/{n}.mp4 → pull a rendered clip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from app.config import load_config  # noqa: E402
from app.store import LocalStore  # noqa: E402
from app.worker import InProcessWorker  # noqa: E402

AGENT_VERSION = "1.0.0"


class AgentState:
    def __init__(self, token: str, store: LocalStore, worker: InProcessWorker) -> None:
        self.token = token
        self.store = store
        self.worker = worker
        self.uploads: dict[str, Path] = {}
        self.lock = threading.Lock()
        self.started = time.time()

    def check_auth(self, headers: dict[str, str]) -> bool:
        if not self.token:
            return True  # token optional on a tailnet-only agent
        got = headers.get("authorization", "") or headers.get("Authorization", "")
        return got == f"Bearer {self.token}"


def build_handler(state: AgentState):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        server_version = f"ClipForgeAgent/{AGENT_VERSION}"

        # -- helpers ------------------------------------------------------
        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            return state.check_auth(dict(self.headers))

        # -- routes -------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/agent/ping":
                if not self._authed():
                    self._json(401, {"detail": "invalid worker token"})
                    return
                self._json(
                    200,
                    {
                        "version": AGENT_VERSION,
                        "uptimeSec": int(time.time() - state.started),
                        "queueDepth": state.worker.status()["queueDepth"],
                        "currentJob": state.worker.status()["currentJob"],
                    },
                )
                return

            if self.path.startswith("/agent/output/"):
                if not self._authed():
                    self._json(401, {"detail": "invalid worker token"})
                    return
                _, _, _, job_id, name = self.path.split("/", 4)
                clip_dir = load_settings().app.paths.outputs_dir / job_id
                f = clip_dir / name
                if not f.exists():
                    self._json(404, {"detail": f"not rendered yet: {name}"})
                    return
                data = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self._json(404, {"detail": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/agent/jobs":
                self._json(404, {"detail": "not found"})
                return
            if not self._authed():
                self._json(401, {"detail": "invalid worker token"})
                return

            ctype = self.headers.get("Content-Type", "")
            boundary = None
            if "multipart/form-data" in ctype and "boundary=" in ctype:
                boundary = ctype.split("boundary=", 1)[1].strip().encode()

            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""

            fields: dict[str, str] = {}
            file_bytes: bytes | None = None
            file_name = "source.mp4"
            if boundary:
                for part in body.split(b"--" + boundary):
                    if b"Content-Disposition" not in part:
                        continue
                    head, _, payload = part.partition(b"\r\n\r\n")
                    payload = payload.rsplit(b"\r\n", 1)[0]
                    if b'filename="' in head:
                        file_name = head.split(b'filename="')[1].split(b'"')[0].decode()
                        file_bytes = payload
                    else:
                        name = head.split(b'name="')[1].split(b'"')[0].decode()
                        fields[name] = payload.decode("utf-8", "replace")
            else:
                try:
                    fields = json.loads(body.decode("utf-8") or "{}")
                except Exception:
                    self._json(400, {"detail": "bad request body"})
                    return

            job_id = fields.get("jobId") or state.store.new_id()
            dest_dir = load_settings().app.paths.downloads_dir / job_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            src = dest_dir / f"source{Path(file_name).suffix.lower() or '.mp4'}"
            if file_bytes is not None:
                src.write_bytes(file_bytes)
            elif not src.exists():
                self._json(400, {"detail": "no file in upload and no local copy"})
                return

            state.store.create_job(
                source_type=fields.get("sourceType", "upload"),
                source=fields.get("source") or src.name,
                title=None,
                min_clip_sec=int(fields["minClipSec"]) if fields.get("minClipSec") else None,
                max_clip_sec=int(fields["maxClipSec"]) if fields.get("maxClipSec") else None,
                max_clips=int(fields["maxClips"]) if fields.get("maxClips") else None,
            )
            # the agent keeps its own doc-id mapping server→agent
            state.uploads[job_id] = src
            state.worker.enqueue(job_id)
            logger.info(f"[agent] accepted job {job_id} ({src.stat().st_size} bytes)")
            self._json(
                200,
                {"accepted": True, "version": AGENT_VERSION, "jobId": job_id},
            )

        def log_message(self, fmt: str, *args) -> None:  # silence default noise
            logger.debug(f"[agent-http] {fmt % args}")

    return Handler, ThreadingHTTPServer


_settings = None


def load_settings():
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings


def report_status_loop(server_base: str, token: str, store: LocalStore, stop: threading.Event) -> None:
    """Push agent progress for dispatched jobs back to the hosted server."""

    def post(path: str, payload: dict) -> None:
        req = urllib.request.Request(
            f"{server_base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as exc:
            logger.debug(f"[agent] status post failed: {exc}")

    last_stage: dict[str, str] = {}
    while not stop.is_set():
        for job in store.list_jobs():
            jid = job["_id"]
            stage = job.get("stage")
            if stage in {"complete", "failed"} and last_stage.get(jid) == stage:
                continue
            if stage in {"transcribing", "finding", "rendering", "complete", "failed"}:
                last_stage[jid] = stage
                if stage == "failed":
                    post(
                        f"/api/internal/jobs/{jid}/failed",
                        {"error": job.get("error") or "worker error", "detail": job.get("errorDetail")},
                    )
                elif stage == "complete":
                    post(
                        f"/api/internal/jobs/{jid}/complete",
                        {
                            "clipCount": job.get("clipCount") or 0,
                            "totalClipsSec": job.get("totalClipsSec") or 0,
                            "captionsTxt": job.get("captionsTxt"),
                        },
                    )
                else:
                    post(
                        f"/api/internal/jobs/{jid}/progress",
                        {
                            "stage": stage,
                            "progress": job.get("progress") or 0,
                            "progressDetail": job.get("progressDetail"),
                            "unitLabel": job.get("unitLabel"),
                            "unitsDone": job.get("unitsDone"),
                            "unitsTotal": job.get("unitsTotal"),
                        },
                    )
        stop.wait(5.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="ClipForge home worker agent")
    ap.add_argument("--server", required=True, help="hosted ClipForge base URL")
    ap.add_argument("--listen", default="0.0.0.0:8765", help="bind for the agent HTTP API")
    ap.add_argument("--register", action="store_true", help="tell the server our tailnet URL once")
    ap.add_argument("--public-url", default=None, help="URL the server should use to reach us (tailnet IP)")
    args = ap.parse_args()

    token = os.getenv("WORKER_TOKEN", "").strip()
    settings = load_settings()
    store = LocalStore(settings.app.paths)
    worker = InProcessWorker(settings, store)
    worker.start()

    logger.info(f"[agent] ClipForge worker-agent {AGENT_VERSION}")
    logger.info(f"[agent] local pipeline online (model: {settings.whisper.model})")

    if args.register and args.public_url:
        req = urllib.request.Request(
            f"{args.server}/api/internal/register-worker",
            data=json.dumps({"workerUrl": args.public_url, "token": token}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10).read()
            logger.info(f"[agent] registered {args.public_url} with {args.server}")
        except Exception as exc:
            logger.error(f"[agent] registration failed: {exc} — set WORKER_URL on the platform instead")

    stop = threading.Event()
    threading.Thread(
        target=report_status_loop, args=(args.server, token, store, stop), daemon=True
    ).start()

    Handler, Server = build_handler(AgentState(token, store, worker))
    host, _, port = args.listen.rpartition(":")
    httpd = Server((host or "0.0.0.0", int(port or 8765)), Handler)
    logger.info(f"[agent] listening on {args.listen} — waiting for dispatched jobs")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
