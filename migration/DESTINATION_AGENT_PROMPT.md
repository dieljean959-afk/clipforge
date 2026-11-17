# ClipForge — one-paste finish prompt

Paste the block below into the destination coding agent. It assumes the source
is already copied and the Python tests have passed. Do not paste API keys into
the agent chat.

---

You are finishing the existing ClipForge relocation in this workspace. Do not
redesign it, rebuild it from a screenshot, convert it to Convex, or edit the
locked caption/layout files. Make only the changes below and report real
command output.

Current state: the source is already present, the Python suite reports
`411 passed, 1 skipped`, and no AI key is configured. The preview currently
shows:

`{"detail":"ClipForge API is tailnet-only (TAILSCALE_ONLY=true) — connect over Tailscale."}`

## 1. Unlock the preview

The Freebuff preview proxy connects from a non-loopback address, so
`TAILSCALE_ONLY=true` rejects it. For preview access only:

1. Set `TAILSCALE_ONLY=false` in this workspace's Keys/API keys panel, or in
   the runtime env file if this host uses one.
2. Restart the app through the platform's managed process. Do not start a
   second server.
3. Keep the existing host/port binding. Do not press **Publish**.
4. Verify:

```bash
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/status
```

`/api/health` returning HTTP 200 is not enough. Require `degraded` to be empty,
`worker.online` to be true, and `/api/status` to have `ok: true`.

This app has no real login. With `TAILSCALE_ONLY=false`, anyone with the
preview link can control every job. Keep it unlisted, do not Publish, and set
`TAILSCALE_ONLY=true` again before any tailnet/private deployment.

## 2. AI key

The user must add `DEEPSEEK_API_KEY` (preferred) and optionally
`GROQ_API_KEY` in the workspace's Keys/API keys panel. Never ask for the key in
chat, print it, log it, or write it into source. If a key is already present, do
not ask again; verify presence only through:

```bash
curl -fsS http://127.0.0.1:8000/api/status
```

Check `detection.ready` and the `deepseek`/`groq` key status. A key is not
"working" just because the variable exists; report the status object.

## 3. Build the dashboard

The API serves the built UI from `webui/`, not `dist/`. Without this step the
page is blank.

```bash
bun install
bun run build
rm -rf webui
mkdir -p webui
cp -r dist/* webui/
```

Restart the app after the copy and confirm `/` returns the dashboard HTML, not
a JSON error.

## 4. Apply safe limits for this disk

Only ~4.7 GB is free. Use short sources and keep one worker:

```bash
TAILSCALE_ONLY=false
CLIPFORGE_DATA_DIR=/var/lib/clipforge
CLIPFORGE_WHISPER_MODEL=small
CLIPFORGE_WHISPER_ENGINE=auto
CLIPFORGE_MAX_SOURCE_MIN=30
CLIPFORGE_MAX_SOURCE_MB=800
CLIPFORGE_SCRATCH_CAP_MB=1200
CLIPFORGE_RETENTION_HOURS=72
CLIPFORGE_WORKER_CONCURRENCY=1
CLIPFORGE_PUBLIC_URL=
```

Do not run a 120-minute source on this volume. The first job also downloads the
Whisper model. Clear old jobs if free space drops below 2 GB.

## 5. Run the real acceptance test

After the key is present and the UI loads:

1. Create one short URL job with an explicit count of 5.
2. Wait for a terminal state.
3. Fetch the job and its clips:
   ```bash
   curl -fsS http://127.0.0.1:8000/api/jobs/<job_id>
   curl -fsS "http://127.0.0.1:8000/api/clips?jobId=<job_id>"
   ```
4. Confirm the job is `complete`, exactly 5 clips are `done`, no clip has
   `mediaError`, and `/api/clips/<clip_id>/video` returns a playable MP4
   (`video/mp4`, non-zero bytes).
5. If it fails, report the exact job state, clip states, and server error. Do
   not mark the relocation complete or claim the site is live.

## 6. Report back

Include the actual output of:

```bash
uname -a
python3 --version
ffmpeg -version | head -1
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/status
```

State separately: preview reachable yes/no, UI loads yes/no, AI key present
yes/no, worker online yes/no, 5-clip job result. Do not include any key value.
