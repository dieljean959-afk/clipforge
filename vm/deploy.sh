#!/usr/bin/env bash
# ClipForge worker VM — fresh Oracle Cloud Always Free Ubuntu 22.04 -> working
# render worker in <15 min.
#
# Usage (as the ubuntu user on the VM):
#   CLIPFORGE_BASE_URL=https://<your-app>.convex.site \
#   CLIPFORGE_WORKER_TOKEN=<token> \
#   bash deploy.sh
#
# Shape: VM.Standard.A1.Flex (4 OCPU / 24 GB ARM, free) preferred; if capacity
# is unavailable, VM.Standard.E2.1.Micro (1 GB AMD, free) works — ffmpeg is
# slower but functional. Redis is installed with AOF persistence ON so queued
# jobs survive reboots. Connectivity uses a stable hostname (Tailscale or
# Cloudflare Tunnel) — NEVER a raw public IP, which changes on stop/start.

set -euo pipefail

: "${CLIPFORGE_BASE_URL:?Set CLIPFORGE_BASE_URL (https://<app>.convex.site)}"
: "${CLIPFORGE_WORKER_TOKEN:?Set CLIPFORGE_WORKER_TOKEN (same value as the deployment env)}"

APT_PKGS=(python3.11 python3.11-venv python3-pip ffmpeg redis-server jq)

echo "==> [1/7] apt packages: ${APT_PKGS[*]}"
sudo apt-get update -y
sudo apt-get install -y "${APT_PKGS[@]}"

echo "==> [2/7] redis: AOF persistence ON (queued jobs survive reboots)"
sudo tee /etc/redis/redis-clipforge.conf >/dev/null <<EOF
port 6379
bind 127.0.0.1
appendonly yes
appendfsync everysec
dir /var/lib/redis-clipforge
EOF
sudo mkdir -p /var/lib/redis-clipforge && sudo chown redis:redis /var/lib/redis-clipforge
sudo systemctl enable --now redis-server
sudo systemctl restart redis-server

echo "==> [3/7] worker install dir + venv"
sudo mkdir -p /opt/clipforge-worker/downloads
sudo chown -R "$USER":"$USER" /opt/clipforge-worker
python3.11 -m venv /opt/clipforge-worker/venv
/opt/clipforge-worker/venv/bin/pip install --upgrade pip
/opt/clipforge-worker/venv/bin/pip install yt-dlp faster-whisper requests

echo "==> [4/7] stable hostname (Tailscale preferred, Cloudflare Tunnel fallback)"
# Tailscale: `sudo tailscale up` once, then use the 100.x.dsn hostname below.
if command -v tailscale >/dev/null 2>&1; then
  WORKER_HOSTNAME="$(sudo tailscale ip -4 | head -1)"
  echo "    tailscale IP: ${WORKER_HOSTNAME} (stable across reboots)"
else
  echo "    Tailscale not found. Install with:  curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up"
  echo "    (or set up a Cloudflare Tunnel: cloudflared tunnel create clipforge-worker)"
  echo "    REDIS_URL for the app must use that hostname — never a public IP."
fi

echo "==> [5/7] worker env file"
sudo tee /opt/clipforge-worker/worker.env >/dev/null <<EOF
CLIPFORGE_BASE_URL=${CLIPFORGE_BASE_URL}
CLIPFORGE_WORKER_TOKEN=${CLIPFORGE_WORKER_TOKEN}
YT_DLP_AUTOUPDATE=1
WORKER_HOME=/opt/clipforge-worker
EOF
sudo chmod 600 /opt/clipforge-worker/worker.env

# copy the agent script next to this deploy.sh if not already there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/clipforge_worker.py" ]; then
  cp "${SCRIPT_DIR}/clipforge_worker.py" /opt/clipforge-worker/clipforge_worker.py
fi

echo "==> [6/7] systemd services (Restart=always)"
sudo tee /etc/systemd/system/clipforge-worker.service >/dev/null <<EOF
[Unit]
Description=ClipForge render worker (yt-dlp fetch agent)
After=network-online.target redis-server.service
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=/opt/clipforge-worker
EnvironmentFile=/opt/clipforge-worker/worker.env
ExecStart=/opt/clipforge-worker/venv/bin/python /opt/clipforge-worker/clipforge_worker.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/clipforge-redis.service >/dev/null <<EOF
[Unit]
Description=Redis for ClipForge (AOF persistence ON)
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/redis-server /etc/redis/redis-clipforge.conf
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now clipforge-redis.service clipforge-worker.service

echo "==> [7/7] verify"
sleep 5
systemctl is-active clipforge-worker && echo "worker: ACTIVE"
journalctl -u clipforge-worker -n 5 --no-pager || true
echo
echo "Boot log should show: 'ClipForge worker online — ${CLIPFORGE_BASE_URL} — yt-dlp ready'"
echo "Done. Flip the ON switch in the ClipForge dashboard; the worker's"
echo "heartbeat turns the dot green within ~75s."
