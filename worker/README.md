# Clipmake Download Worker

A tiny standalone HTTP service your phone (or home server) runs so the Clipmake
app can download YouTube (and other supported) videos through your own browser
session instead of through the hosted server.

It uses yt-dlp to fetch the video on your device, merges it to mp4 with ffmpeg,
and streams the file back to the app over HTTPS + a Bearer token.

**Real downloads only.** It never fakes a file, never logs your token, and never
stores anything between jobs.

## What you need on the device

- Node.js (18+)
- yt-dlp
- ffmpeg (for the mp4 merge)

Example on Termux:

```sh
pkg update -y && pkg upgrade -y
pkg install -y nodejs ffmpeg curl
pip install -U "yt-dlp[default]"
```

## Start it on your phone (Termux)

One command runs everything — the worker, the tunnel, and the link registration:

```sh
APP_URL="https://your-clipmake-app" ./start.sh          # cloudflared, auto-detected
APP_URL="https://your-clipmake-app" ./start.sh --tunnel ngrok
./start.sh --no-tunnel                                  # worker alone, for local tests
```

It keeps all three alive: if the worker exits it is restarted, if cloudflared
exits it is restarted with a fresh link and `register.sh` re-registers it. Logs
land in `worker.log`, `cloudflared.log` (or `ngrok.log`) and `register.log`.
Ctrl-C stops everything. The token is read from `token.txt` and never printed.

```sh
mkdir -p ~/clipmake-worker
cd ~/clipmake-worker

# copy start.js and this README into the folder, then:
node start.js "paste-your-token-here"
```

Pick any token you like and use the same one in the app. A good one:

```sh
node -e "console.log(require('crypto').randomBytes(24).toString('hex'))"
```

The worker only listens on `127.0.0.1:3001` by default. It does not expose a
public URL by itself — that part is done by a tunnel (below).

Keep it running while you forge clips. If you close Termux, the worker stops.

## Expose it to Clipmake

The app runs on the Cloud preview, so the worker must be reachable from
`https://...trycloudflare.com` or whatever origin the preview uses.

Two common options:

- **Cloudflare tunnel (recommended for quick tests)**

```sh
cloudflared tunnel --url http://localhost:3001 --no-autoupdate
```

It prints an `https://...trycloudflare.com` URL. Copy that into the app's
Settings > Download worker > Worker URL.

The tunnel link changes every time you restart `cloudflared`. Let `register.sh`
tell the app about it instead of pasting the link — it reads the current link
from the tunnel log, registers it, and does it again whenever the link changes:

```sh
cloudflared tunnel --url http://localhost:3001 --no-autoupdate 2>&1 | tee cloudflared.log &
APP_URL="https://your-clipmake-app" sh register.sh
```

Keep it running next to the worker (one line in your Termux `start.sh`). It never
prints the token — only its last 4 characters — and it reports the real HTTP
status of every attempt: `401` means `token.txt` and the app's `WORKER_TOKEN`
differ, `400` means the link is not a trycloudflare/ngrok URL, `503` means the
app has no token saved yet.

- **ngrok**

```sh
ngrok http 3001
```

Use the HTTPS URL it prints, same as above.

## Health endpoint

`GET /health` with header `Authorization: Bearer <token>` returns:

```json
{
  "ok": true,
  "ytdlp": "2024.12.01",
  "node": "v20.18.0",
  "js_runtime": "node",
  "egjs_ejs": true,
  "busy": false
}
```

`egjs_ejs` is `true` when yt-dlp reports that the EJS JS solver is available.

## Download endpoint

`POST /download` with the same Bearer header and body:

```json
{
  "url": "https://youtube.com/watch?v=...",
  "maxHeight": 720,
  "proxy": "http://host:port"
}
```

`maxHeight` is optional. If omitted, it asks for the best merged mp4 yt-dlp
finds under the current client settings.

On success it streams a `video/mp4` response. On failure it returns JSON with
`error` and `detail` so the app can show the exact yt-dlp message.

Errors you may see:

- `401 unauthorized` — token does not match
- `429 worker busy, try again soon` — another job is running
- `502 { error: "download failed", detail: "..." }` — yt-dlp failed

## How the app uses it

1. In Settings, paste the tunnel URL and the token. Save.
2. Hit **Test connection**. The app calls your worker's `/health` through the
   tunnel and reports yt-dlp version, Node version, round-trip time and busy/idle.
3. When a Forge job starts and the worker is online, the app sends the download
   job to the worker first and falls back to other methods if the worker is
   offline or fails.

## Token handling

- The token is compared with a timing-safe hash on the worker — the worker never
  logs the raw token.
- The app only ever sends the token to your own worker over HTTPS. It is not
  stored in client code or sent to the hosted server for the worker check.

## Notes

- One job at a time. If two clips try to download simultaneously, the second
  gets `429`.
- Temp files are deleted after each job, whether it succeeded or failed.
- If `cloudflared` or ngrok restarts, the public URL changes. `register.sh`
  re-registers the new link within a few seconds; without it, update the URL in
  the app's Settings.
