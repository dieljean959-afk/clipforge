# ClipForge — always-on host runbook

The Freebuff dev preview is **not** a public site: the web process (and the
uvicorn child it supervises) only run while the project is open. For a permanent
URL, deploy the backend to a host you control. Everything in this file is
**server-side and therefore BLOCKED from this sandbox** — run it on the target
host.

## 1. Host shape

| resource | minimum | why |
|---|---|---|
| CPU | 2 vCPU | 1 inference thread + 1 ffmpeg |
| RAM | 3 GB | whisper `small` int8 = ~1.1 GiB + ffmpeg + API |
| Disk | 20 GB | source + sections + clips + model cache |
| Network | outbound HTTPS | yt-dlp, DeepSeek/Gemini, S3-compatible bucket |

## 2. Install (BLOCKED: needs the host)

```bash
sudo useradd -r -s /usr/sbin/nologin clipforge
sudo mkdir -p /opt/clipforge /var/lib/clipforge
sudo chown -R clipforge: /opt/clipforge /var/lib/clipforge
# put a checkout of this project at /opt/clipforge (any transport you use)
cd /opt/clipforge
sudo -u clipforge python3 -m venv .venv
sudo -u clipforge .venv/bin/pip install -e .
sudo apt-get install -y ffmpeg      # ffprobe comes with it
```

Secrets go in `/etc/clipforge/env` (`chmod 600`, owner root), **never** in the
repo: `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, and the S3 credentials
(`CLIPFORGE_S3_*`). The API only ever reports "key present" — it never logs a
value.

## 3. Supervise

```bash
sudo cp deploy/clipforge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now clipforge
systemctl status clipforge --no-pager
```

The unit starts **one** uvicorn worker with no `--reload`, `Restart=on-failure`,
`KillMode=mixed`, `TimeoutStopSec=60`. The API spawns `python -m app.workerd` in
its own session so a deploy does not kill a mid-render job; the worker traps
SIGTERM, checkpoints, marks the job `interrupted` and exits.

Two API processes cannot share a data dir: `data/api.lock` holds the owner pid
and a second server exits with

```
[boot] FATAL instance-lock: another API instance (pid N) already owns this data dir
```

A stale lock (dead pid) is reclaimed automatically on boot.

## 4. Health + observability

```bash
curl -fsS localhost:8000/api/health    # liveness; 200 in ~2 ms, memory only
curl -fsS localhost:8000/api/status    # per-dependency: state, bucket, whisper,
                                       # deepseek, yt-dlp, ffmpeg, disk, worker
curl -fsS localhost:8000/api/doctor    # the same report the Doctor button shows
.venv/bin/python tools/doctor.py       # CLI; exit 1 on FAIL
```

Boot order is deliberate: the port opens first (~0.6 s), then recovery, sweeps,
model preload and update checks run as timed background phases. A failing phase
is listed in `/api/boot` and `/api/status` and never blocks the API. A real boot
failure (bad config, lock held, import error) writes `data/boot_failure.json`
and prints one line:

```
[boot] FATAL <where>: <cause> → FIX: <what to do>
```

Point an external health check at `/api/health` with a 5 s timeout, 30 s
interval, and alert after 3 failures. Alert separately on
`/api/status.ok == false`.

## 5. Reverse proxy (Caddy example)

```
clips.example.com {
    encode zstd gzip
    request_body {
        max_size 2GB
    }
    reverse_proxy 127.0.0.1:8000 {
        transport http {
            read_timeout 30m   # renders stream for minutes
        }
    }
}
```

`/api/export/zip` and `/api/clips/*/download` stream; a 30 s proxy timeout
breaks long downloads while the job itself is fine.

## 6. Deploy / rollback

```bash
# update the checkout at /opt/clipforge, then:
sudo -u clipforge /opt/clipforge/.venv/bin/pip install -e /opt/clipforge
sudo systemctl restart clipforge          # jobs resume from checkpoints
curl -fsS localhost:8000/api/status | jq '.ok, .boot.failedPhases'
```

Rollback is "restore the previous checkout, restart". A restart never re-runs a
finished stage: `stageStatus`, the transcript chunks and the clip documents are
the checkpoints, and `/api/status`'s `job_recovery` phase reports what was
recovered (`N resumed, M restored from clips`).

## 7. What is NOT done here (BLOCKED from the sandbox)

- DNS/TLS, firewall, object-store bucket policy.
- Installing the unit and secrets on the real host.
- A load test with real traffic.
