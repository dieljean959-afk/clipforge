# ClipForge Worker VM (Oracle Cloud Always Free)

The render worker is a real Linux VM that runs **yt-dlp via its Python API**
(primary fetcher) and can also run ffmpeg/faster-whisper. The dashboard has a
glowing **ON/OFF switch** that starts and stops the instance through the OCI
API — the VM costs ~$0/mo while under Always Free limits and costs nothing at
all while stopped.

## Shape

| Preference | Shape | Notes |
|---|---|---|
| 1st choice | `VM.Standard.A1.Flex` (4 OCPU / 24 GB, ARM) | free, fast ffmpeg |
| fallback   | `VM.Standard.E2.1.Micro` (1 GB, AMD) | free, almost always available |

Ubuntu 22.04, any region with capacity.

## Deploy (fresh VM -> working worker in <15 min)

```bash
ssh ubuntu@<vm>
sudo apt-get update && sudo apt-get install -y git
git clone <this repo> && cd <repo>/vm
CLIPFORGE_BASE_URL=https://<your-app>.convex.site \
CLIPFORGE_WORKER_TOKEN=<same value as the deployment env> \
  bash deploy.sh
```

`deploy.sh` installs python3.11, ffmpeg, redis-server (AOF persistence **ON**
so queued jobs survive reboots), yt-dlp + faster-whisper, writes the env file,
and installs the `clipforge-worker.service` + `clipforge-redis.service` systemd
units with `Restart=always`.

## Connectivity — stable hostname, never an IP

Public IPs change on every stop/start. Run **Tailscale** (preferred) or a
**Cloudflare Tunnel** on the VM so `REDIS_URL` and any callbacks use a stable
hostname:

```bash
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
# then use the 100.x.y.z / MagicDNS name — NOT the public IP
```

## Env vars (app side — Keys/API keys tab)

| Var | Purpose |
|---|---|
| `OCI_TENANCY_OCID` | tenancy OCID for the control plane |
| `OCI_USER_OCID` | IAM user the app signs as |
| `OCI_FINGERPRINT` | API key fingerprint |
| `OCI_PRIVATE_KEY` | PEM contents of the signing key |
| `OCI_INSTANCE_OCID` | the worker instance to START/STOP |
| `OCI_REGION` | e.g. `us-ashburn-1` |
| `CLIPFORGE_WORKER_TOKEN` | bearer token the VM uses for `/api/worker/*` |

Minimal IAM policy (only this one instance):

```
Allow group clipforge-workers to manage instance in compartment <X>
 where request.instance.id = '<OCI_INSTANCE_OCID>'
Allow group clipforge-workers to use virtual-network-family in compartment <X>
 where target.vcn.id = '<vcid>'   # only if the app must reach the VM
```

## Worker API (token-gated, served by the app's Convex HTTP router)

| Route | Method | Purpose |
|---|---|---|
| `/api/worker/heartbeat` | POST | liveness + model state; returns `drainAndStop` when a STOP is pending |
| `/api/worker/claim` | GET | oldest queued URL job |
| `/api/worker/progress` | POST | yt-dlp download % (bytes done/total) |
| `/api/worker/fetch-failed` | POST | exact yt-dlp/stderr error, per attempt |
| `/api/worker/upload-url` | POST | Convex storage upload URL for the fetched MP4 |

## yt-dlp rot policy

YouTube breaks yt-dlp without warning. The worker self-updates it **in
process** (`python -m pip install -U yt-dlp`) on import failure and once
between attempts; `YT_DLP_AUTOUPDATE=1` is the default. Set it to `0` to pin.

## Acceptance test

1. Toggle OFF → ON: OCI shows `STOPPED → STARTING → RUNNING`; within ~75s the
   worker heartbeat turns the dot green ("Worker online — yt-dlp + ffmpeg
   ready").
2. Submit a YouTube URL: the job log shows `yt-dlp claimed on the worker VM`,
   real download %, `download complete (N MB) — ffprobe verified`, then the
   normal transcribe → detect → render pipeline.
3. Toggle OFF while idle: OCI `STOPPING → STOPPED`; the switch returns gray
   after reload. Queued jobs stay in the queue and resume on next wake.
4. cobalt never appears anywhere in the codebase or errors again.
