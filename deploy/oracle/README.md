# ClipForge on Oracle Cloud (Always Free)

This is the always-on deployment path: one Ubuntu VM, the FastAPI app, its
worker, and the prebuilt dashboard. The Freebuff preview sleeps; this VM does
not.

You need an Oracle Cloud account. The VM itself cannot be created from this
project because it requires your Oracle tenancy login and SSH key.

## 1. Create the VM

In the Oracle Cloud console:

1. **Compute → Instances → Create instance**.
2. **Image:** Ubuntu 24.04 (or 22.04 with Python 3.11+).
3. **Shape:** `VM.Standard.A1.Flex` with **4 OCPUs / 24 GB RAM** if Always Free
   capacity is available. The `VM.Standard.E2.1.Micro` (1 GB RAM) is too small
   for Whisper; it will run out of memory.
4. **Boot volume:** 100 GB (or the largest Always Free allowance).
5. **Networking:** create/select a public subnet, assign a **public IPv4**.
6. **SSH keys:** upload or generate a key pair. Download the private key.
7. Create the instance and note its public IP.

Open these ingress rules in the instance's security list / network security
group:

| port | protocol | why |
|---|---|---|
| 22 | TCP | SSH |
| 80 | TCP | HTTP / Let's Encrypt |
| 443 | TCP | HTTPS |
| 8000 | TCP | direct test before Caddy is configured |

## 2. Upload the project

From your computer, where you downloaded `clipforge-full-site-v4.zip`:

```bash
scp -i /path/to/ssh-key clipforge-full-site-v4.zip ubuntu@<PUBLIC_IP>:/tmp/
ssh -i /path/to/ssh-key ubuntu@<PUBLIC_IP>
```

On the VM:

```bash
sudo apt-get update
sudo apt-get install -y unzip
sudo unzip -q /tmp/clipforge-full-site-v4.zip -d /opt
sudo chown -R ubuntu:ubuntu /opt/clipforge
cd /opt/clipforge
```

The archive contains one top-level directory, `clipforge/`.

## 3. Bootstrap the VM

```bash
sudo bash deploy/oracle/setup.sh /opt/clipforge
```

The script installs ffmpeg, Python, Bun, the Python package, builds `webui/`,
creates the `clipforge` service user, installs the systemd unit, and enables it.
It also creates `/etc/clipforge/env` from `deploy/oracle/env.example` if the
file does not exist.

## 4. Add your keys

```bash
sudo nano /etc/clipforge/env
```

At minimum fill in:

```dotenv
DEEPSEEK_API_KEY=...
```

Optional but recommended:

```dotenv
GROQ_API_KEY=...          # backup LLM / faster ASR path
GEMINI_API_KEY=...        # optional alternate provider
CLIPFORGE_PUBLIC_URL=http://<PUBLIC_IP>:8000
```

For clips that survive a VM rebuild, add an S3-compatible bucket:

```dotenv
CLIPFORGE_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
CLIPFORGE_S3_BUCKET=clipforge
CLIPFORGE_S3_ACCESS_KEY=...
CLIPFORGE_S3_SECRET_KEY=...
CLIPFORGE_S3_REGION=auto
CLIPFORGE_S3_PREFIX=clipforge
```

Then:

```bash
sudo chmod 600 /etc/clipforge/env
sudo systemctl restart clipforge
sudo systemctl status clipforge --no-pager
```

You can also paste keys into the app's Keys panel. The service runs as the
`clipforge` user and the project directory is writable by that user, so the
panel can save `.env.local` without root.

## 5. Verify

```bash
curl -fsS localhost:8000/api/health
curl -fsS localhost:8000/api/status
```

Open `http://<PUBLIC_IP>:8000`.

A healthy deployment has:

- `/api/health` HTTP 200;
- empty `degraded`;
- `worker.online: true`;
- `/api/status.ok: true`.

The first job downloads the Whisper model, so allow time and disk for it.

## 6. HTTPS with a domain (recommended)

Point an `A` record at the public IP, then install Caddy:

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
```

`/etc/caddy/Caddyfile`:

```caddyfile
clips.example.com {
    encode zstd gzip
    request_body {
        max_size 2GB
    }
    reverse_proxy 127.0.0.1:8000 {
        transport http {
            read_timeout 30m
        }
    }
}
```

```bash
sudo systemctl reload caddy
```

Then set `CLIPFORGE_PUBLIC_URL=https://clips.example.com` in
`/etc/clipforge/env` and restart the service. Remove the public `8000` rule
once Caddy works.

## 7. Updates and logs

```bash
sudo journalctl -u clipforge -f
sudo systemctl restart clipforge
```

A restart does not lose a finished stage: the job artifacts and clip documents
are the checkpoints. A render that was in progress resumes from the last saved
stage.

## Security note

This app has no built-in login. Do not expose port 8000 or a public domain with
real keys until you put an access layer in front of it (Caddy basic auth, a
VPN/Tailscale, or an identity-aware proxy). `TAILSCALE_ONLY=false` is required
for a public URL; it is not an authentication method.
