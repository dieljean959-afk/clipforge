# Steps 5–7: env, safe access, boot and health

Use this on the destination host after the source is copied. Do not paste real
keys into source, chat, or logs.

## Step 5 — environment variables

Create the runtime env file on the host:

```bash
sudo install -d -m 700 /etc/clipforge
sudo install -m 600 /dev/null /etc/clipforge/env
sudo nano /etc/clipforge/env
```

Start with this template and fill only the values you actually have:

```bash
# --- required: at least one LLM provider -------------------------------
DEEPSEEK_API_KEY=
GROQ_API_KEY=

# Optional extra fallback provider
GEMINI_API_KEY=

# --- persistent storage ------------------------------------------------
CLIPFORGE_DATA_DIR=/var/lib/clipforge

# --- Whisper -----------------------------------------------------------
CLIPFORGE_WHISPER_MODEL=small
CLIPFORGE_WHISPER_ENGINE=auto

# --- host safety / limits ---------------------------------------------
TAILSCALE_ONLY=true
CLIPFORGE_WORKER_CONCURRENCY=1
CLIPFORGE_MAX_SOURCE_MIN=120
CLIPFORGE_RETENTION_HOURS=96

# --- optional: durable media offload (all six together) ---------------
CLIPFORGE_S3_ENDPOINT=
CLIPFORGE_S3_BUCKET=
CLIPFORGE_S3_ACCESS_KEY=
CLIPFORGE_S3_SECRET_KEY=
CLIPFORGE_S3_REGION=auto
CLIPFORGE_S3_PREFIX=clipforge

# --- optional: split worker (leave blank on a single VM) --------------
WORKER_URL=
WORKER_TOKEN=

# --- optional: YouTube cookies if downloads get bot-checked -----------
YT_DLP_COOKIES_FILE=

# --- optional: only if ffmpeg/ffprobe are not on PATH -----------------
FFMPEG_PATH=
FFPROBE_PATH=

# --- leave blank until you have a real domain + access control --------
CLIPFORGE_PUBLIC_URL=
```

Notes:

- `DEEPSEEK_API_KEY` and `GROQ_API_KEY` are the actual detection providers.
  At least one must be valid; both is better because one can rate-limit.
- S3 is optional. Without it, media stays under `/var/lib/clipforge`; with it,
  jobs survive a host rebuild. If you set any of the four core S3 values, set
  all four (`ENDPOINT`, `BUCKET`, `ACCESS_KEY`, `SECRET_KEY`).
- Keep `CLIPFORGE_WORKER_CONCURRENCY=1` on 2 vCPU / 3 GB. Each job runs Whisper
  and ffmpeg; more workers will swap or OOM.
- Do not set `CLIPFORGE_PUBLIC_URL` until access control is in place.

## Step 6 — security: there is no real login yet

`src/lib/auth.ts` is a UI stub: it says the project has no `/api/auth/*`
backend. Do **not** tell the user a login screen protects this app. It does not.
All jobs and clips share one data directory; anyone who can reach the API can
use it.

Use one of these:

**A. Tailscale only (recommended, built in):**

```bash
# on the host
sudo tailscale up
sudo tailscale serve --bg 8000
```

Keep `TAILSCALE_ONLY=true` and bind uvicorn to `127.0.0.1` (see Step 7).
Tailnet devices reach the app; the public internet gets 401.

**B. SSH tunnel only:**

```bash
ssh -L 8000:127.0.0.1:8000 user@your-host
# open http://127.0.0.1:8000 on your laptop
```

**C. Public domain with an access layer:** put Cloudflare Access, Caddy basic
auth, or an equivalent in front of `127.0.0.1:8000`. Do not expose port 8000
directly. If multiple people will use it, add real app authentication and
per-user data isolation first.

Never run the API on `0.0.0.0` with no firewall and no access layer.

## Step 7 — build, start, verify

The FastAPI app serves the built dashboard from `webui/`. Building into `dist/`
alone leaves a blank page, so copy it explicitly:

```bash
cd /opt/clipforge
bun install
bun run build
rm -rf webui
mkdir -p webui
cp -r dist/* webui/
```

Install and start:

```bash
sudo -u clipforge .venv/bin/pip install -e /opt/clipforge
sudo cp deploy/clipforge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable clipforge
```

Add the env file and private bind with a drop-in:

```bash
sudo systemctl edit clipforge
```

```ini
[Service]
EnvironmentFile=/etc/clipforge/env
ExecStart=
ExecStart=/opt/clipforge/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then:

```bash
sudo systemctl restart clipforge
systemctl status clipforge --no-pager
curl -fsS http://127.0.0.1:8000/api/health | jq '{ok,degraded,worker,warmup,disk}'
curl -fsS http://127.0.0.1:8000/api/status | jq '{ok,dependencies}'
```

`/api/health` always returns HTTP 200; that alone is not success. Require:

- `degraded` is empty
- `worker.online` is true
- `warmup.stage` is not `failed`
- disk has free space
- `/api/status` has `ok: true`

Then create one real URL job with an explicit count (for example 5). Confirm:

- the job reaches `complete`
- exactly 5 clips are `done`
- no clip has `mediaError`
- `/api/clips/{clip_id}/video` returns a playable MP4

Report back with:

```bash
uname -a
python3 --version
ffmpeg -version | head -1
systemctl is-active clipforge
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/doctor
```

If a check fails, paste the exact JSON/error and the command that produced it.
Do not claim the site is live until the public/tailnet URL and these checks are
verified.
