# ClipForge — full site handoff

This archive is the complete runnable source for the ClipForge site in this
workspace. It includes the FastAPI backend, the video pipeline, the React/Vite
dashboard, configuration, deployment files, tests, tools, and the migration
prompt.

It deliberately excludes secrets, job data, media, model caches,
`node_modules`, and build output. Re-enter keys on the destination.

## Read in this order

1. `MIGRATION_PROMPT.md` — paste this into the destination coding agent.
2. `DESTINATION_HANDOFF.md` — exact env vars, safe no-login access, boot and
   health checks.
3. `README.md` — setup and feature documentation.
4. `RELOCATION_BRIEF.md` — hosting/security requirements.
5. `deploy/RUNBOOK.md` and `deploy/clipforge.service` — always-on VM setup.
6. `LOCKED_SPEC.md` — caption/layout behavior that must not change.

## What this actually is

- Backend: Python 3.11, FastAPI, uvicorn.
- Worker: in-process by default; optional remote worker via `WORKER_URL` and
  `WORKER_TOKEN`.
- Pipeline: yt-dlp source fetch, ffmpeg, faster-whisper/WhisperX transcription,
  LLM clip selection, caption burn-in, vertical reframe, MP4 render.
- Frontend: React 19, Vite, TypeScript, Tailwind, shadcn/ui.
- Store: JSON job/clip documents on disk (`app/store.py`); media can offload to
  any S3-compatible bucket.
- `convex/` contains only generated client files. It is **not** the backend.

## Requirements

- Linux host, Python 3.11+, ffmpeg/ffprobe
- 2 vCPU, 3 GB RAM, 20 GB free disk
- Outbound HTTPS for YouTube, LLM APIs and object storage

## Boot

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
sudo apt-get install -y ffmpeg

bun install
bun run build

.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Local one-command bootstrap: `./start.sh`. Always-on host: use
`deploy/clipforge.service` or `docker-compose.yml`.

## Verification

```bash
.venv/bin/python -m pytest -q
bun tsc -b --noEmit
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/status
.venv/bin/python tools/doctor.py
```

Baseline in this workspace: **418 tests passed**, TypeScript clean.

Then run one real URL job with an explicit count; confirm the requested number
of `done` clips, playable `/api/clips/{id}/video`, and no `mediaError`.

## Configuration names

Read `README.md` and `config.toml` for all options. Core names:

- `DEEPSEEK_API_KEY`, `GROQ_API_KEY`
- `CLIPFORGE_DATA_DIR`, `CLIPFORGE_STORAGE_DIR`
- `CLIPFORGE_S3_ENDPOINT`, `CLIPFORGE_S3_BUCKET`,
  `CLIPFORGE_S3_ACCESS_KEY`, `CLIPFORGE_S3_SECRET_KEY`,
  `CLIPFORGE_S3_REGION`, `CLIPFORGE_S3_PREFIX`
- `YT_DLP_COOKIES_FILE`
- `CLIPFORGE_WHISPER_MODEL`, `CLIPFORGE_WHISPER_COMPUTE`
- `WORKER_URL`, `WORKER_TOKEN`
- `CLIPFORGE_PUBLIC_URL`

No authentication is built in. Do not expose this publicly until auth is added
(`RELOCATION_BRIEF.md` §12).

## Important files

```text
app/                 FastAPI app, pipeline, worker, store, object storage
src/                 React dashboard and components
config.toml          runtime defaults
deploy/              systemd unit and host runbook
docs/                architecture and operational notes
tests/               pytest suite (418 passing baseline)
tools/               doctor, QA and migration/worker utilities
public/              static assets
convex/              generated client only, not a backend
```

## Recent reliability fixes to preserve

- Joined two-section spans use a 100 ms audio-sync tolerance; normal clips stay
  at 40 ms (`app/audiosync.py`, `app/task.py`).
- Retries reuse source, transcript, candidates, captions, emoji tracks and
  completed media instead of redoing work.
- An explicit clip quota can never finish `complete` short; it stays `failed`
  and retryable, with ready clips preserved (`_finish` in `app/task.py`,
  `app/invariants.py`).
- Clip projection preserves the requested count and does not resurrect deleted
  clip documents (`app/store.py`).
- Successful renders clear stale `mediaError`/`failedAt`.

## Locked files — do not edit

`app/caption_presets.py`, `app/caption_style.py`, `app/subtitle.py`,
`app/titlebar.py`, `app/memguard.py`,
`src/components/clipmake/CaptionStyleDialog.tsx`,
`src/components/clipmake/NewJobForm.tsx`.
