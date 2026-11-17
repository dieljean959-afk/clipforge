# ClipForge relocation prompt

Copy this file into the destination project/agent, and attach or copy the
`clipforge-source.tar.gz` bundle next to it.

---

You are relocating an existing, working ClipForge application. Do **not**
redesign it, rename it, rebuild it from a screenshot, or convert it to another
backend. Copy the supplied source, make it run on the destination host, and
prove it with the checks below.

## 0. Read this first

This is a **Python/FastAPI + separate worker + React/Vite** application, not a
Convex app. The destination must be a Linux host (VM or container) with:

- Python 3.11+
- `ffmpeg` and `ffprobe`
- 2 vCPU, 3 GB RAM, 20 GB free disk minimum
- outbound HTTPS (YouTube, LLM providers, object storage)

A frontend-only or Convex-only workspace **cannot** run this pipeline. Convex
can store job metadata, but ffmpeg/Whisper must run on a real worker host. If
the destination has no Python/ffmpeg runtime, stop and report that; do not
fake the pipeline in the browser.

No credentials are included. Ask the user to enter them through the
destination's key manager. Never put keys in source, logs, or chat.

## 1. Source and destination shape

The bundle contains `app/` (API, pipeline, worker, store), `src/` (dashboard),
`convex/` (generated client only, not the backend), `config.toml`,
`LOCKED_SPEC.md`, `deploy/RUNBOOK.md`, tests and tools.

Choose one destination shape and state it before changing anything:

- **VM/systemd (recommended):** follow `deploy/RUNBOOK.md`; install at
  `/opt/clipforge`, data at `/var/lib/clipforge`, use
  `deploy/clipforge.service`.
- **Container:** use `Dockerfile` and `docker-compose.yml`; mount persistent
  storage for `CLIPFORGE_DATA_DIR` and outputs.
- **Platform workspace:** only if it can run a long-lived Python process and
  ffmpeg. Otherwise use the VM for the backend and keep the workspace as the
  UI.

Do not expose the app publicly yet: this checkout has **no authentication**
(see `RELOCATION_BRIEF.md` §12). Internal/private access only until auth is
added.

If this is a cloud-workspace preview and every route returns
`401 ClipForge API is tailnet-only`, the preview proxy connects from a
non-loopback address. For preview only, set `TAILSCALE_ONLY=false`, restart
through the managed process, and do not press Publish. Re-enable it before any
shared deployment. See `migration/DESTINATION_AGENT_PROMPT.md` for the exact
one-paste finish steps.

## 2. Install and boot

On the destination:

```bash
# Python backend
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
sudo apt-get install -y ffmpeg

# Frontend (if building locally)
bun install
bun run build

# API + worker
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`start.sh` is the local one-command bootstrap. For an always-on host use the
systemd unit or compose file instead; do not rely on a dev preview.

Copy the existing data directory only if the user explicitly wants old jobs.
Otherwise start empty: old `data/` may contain cookies, jobs and personal media
and was deliberately excluded from the bundle.

## 3. Required configuration (names only)

Read `README.md` and `config.toml` for the authoritative list. At minimum:

- LLM: `DEEPSEEK_API_KEY`, `GROQ_API_KEY`
- Storage (optional but recommended): `CLIPFORGE_S3_ENDPOINT`,
  `CLIPFORGE_S3_BUCKET`, `CLIPFORGE_S3_ACCESS_KEY`,
  `CLIPFORGE_S3_SECRET_KEY`, `CLIPFORGE_S3_REGION`, `CLIPFORGE_S3_PREFIX`
- Fetch: `YT_DLP_COOKIES_FILE` (or `YT_DLP_COOKIES_FROM_BROWSER`)
- Whisper: `CLIPFORGE_WHISPER_MODEL`, `CLIPFORGE_WHISPER_COMPUTE`
- Worker split (optional): `WORKER_URL`, `WORKER_TOKEN`
- Storage location: `CLIPFORGE_DATA_DIR`, `CLIPFORGE_STORAGE_DIR`
- Public URL, only after auth: `CLIPFORGE_PUBLIC_URL`

Validate with:

```bash
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/status
.venv/bin/python tools/doctor.py
```

## 4. Preserve these recent reliability fixes

Do not revert or “simplify” these while relocating:

1. **Joined-section audio gate:** two-section spans get a 100 ms tolerance in
   `app/audiosync.py`; normal clips keep the strict 40 ms gate. `app/task.py`
   passes the wider tolerance only for joined spans.
2. **Per-clip resume:** caption/emoji files whose `clipId` matches are reused;
   existing real titles are reused instead of paying for another LLM call.
3. **Explicit quota:** an explicit 5/10/15/20/25/30 request must never finish
   `complete` with fewer clips. `_finish` writes `failed` with an honest
   “X of Y clips ready” message, and `app/invariants.py` must not repair that
   back to `complete`.
4. **Clip projection:** `create_clips`/`sync_clip_docs` rewrite the job’s clip
   list and preserve `requestedClipTarget`; do not resurrect deleted docs or
   overwrite the requested count with the candidate count.
5. **Stale errors:** a successful render clears `mediaError`/`failedAt`; do not
   leave a red error on a ready clip.
6. **Unchanged look:** caption style/rendering are locked. Do not modify:
   `app/caption_presets.py`, `app/caption_style.py`, `app/subtitle.py`,
   `app/titlebar.py`, `app/memguard.py`,
   `src/components/clipmake/CaptionStyleDialog.tsx`,
   `src/components/clipmake/NewJobForm.tsx`.

## 5. Verification (required)

Run on the destination and report real output:

```bash
.venv/bin/python -m pytest -q
bun tsc -b --noEmit
.venv/bin/ruff check app/task.py app/invariants.py app/audiosync.py app/store.py tests/test_regressions_boot_disk_state.py
```

Expected current baseline: **418 passed**; TypeScript OK; focused ruff clean.
The repository-wide `ruff check app tools tests` has pre-existing lint noise in
standalone drill scripts; do not treat that as a migration failure or use it to
justify unrelated rewrites.

Then prove the app, not just the tests:

1. Create a short URL job with an explicit count (for example 5).
2. Wait until it is terminal; check `GET /api/jobs/{id}` and
   `GET /api/clips?jobId={id}`.
3. Confirm exactly the requested number of `done` clips, no `mediaError`, and
   each `/api/clips/{clipId}/video` returns playable `video/mp4`.
4. Interrupt/retry a job and confirm it reuses source, transcript and finished
   clips rather than re-downloading/re-rendering them.
5. Check `/api/health` and the worker is online.

## 6. Rules

- Copy working code; do not rewrite the pipeline.
- No duplicate job/clip documents, no silent count changes, no fake progress.
- Do not delete completed media or old jobs unless the user asks.
- Do not commit or expose `.env*`, cookies, API keys, `data/`, `downloads/`,
  `outputs/`, or model caches.
- If a step needs a credential or host action you do not have, stop and give
  the exact command/name the user must provide. Do not claim deployment until
  the public URL and `/api/health` are verified.
