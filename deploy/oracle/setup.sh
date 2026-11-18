#!/usr/bin/env bash
# ClipForge — Oracle Cloud / Ubuntu bootstrap.
#
#   sudo bash deploy/oracle/setup.sh /opt/clipforge
#
# Installs system packages, the Python app, the dashboard build, the service
# user, and the systemd unit. It does NOT start the service: fill in
# /etc/clipforge/env first, then `sudo systemctl start clipforge`.
set -euo pipefail

PROJECT_DIR="${1:-/opt/clipforge}"
SERVICE_USER="clipforge"
ENV_FILE="/etc/clipforge/env"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this with sudo: sudo bash deploy/oracle/setup.sh $PROJECT_DIR" >&2
  exit 1
fi
if [ ! -f "$PROJECT_DIR/pyproject.toml" ] || [ ! -f "$PROJECT_DIR/package.json" ]; then
  echo "Not a ClipForge checkout: $PROJECT_DIR" >&2
  exit 1
fi
if ! command -v systemctl >/dev/null 2>&1; then
  echo "This bootstrap targets a systemd host (Ubuntu/Debian)." >&2
  exit 1
fi

echo "==> Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl ffmpeg python3 python3-pip python3-venv unzip

echo "==> Creating the $SERVICE_USER service user"
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "/home/$SERVICE_USER" \
    --shell /usr/sbin/nologin "$SERVICE_USER"
fi
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" \
  /var/lib/clipforge /var/lib/clipforge/outputs /var/lib/clipforge/downloads
chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR"

echo "==> Building the dashboard"
if [ ! -x "/home/$SERVICE_USER/.bun/bin/bun" ] && ! command -v bun >/dev/null 2>&1; then
  sudo -u "$SERVICE_USER" -H bash -c 'curl -fsSL https://bun.sh/install | bash'
fi
BUN="/home/$SERVICE_USER/.bun/bin/bun"
if [ ! -x "$BUN" ]; then
  BUN="$(command -v bun || true)"
fi
if [ -z "$BUN" ] || [ ! -x "$BUN" ]; then
  echo "Bun install failed; install Bun or Node 20+ and rebuild manually." >&2
  exit 1
fi
sudo -u "$SERVICE_USER" -H bash -c "cd '$PROJECT_DIR' && '$BUN' install --frozen-lockfile"
sudo -u "$SERVICE_USER" -H bash -c "cd '$PROJECT_DIR' && '$BUN' run build"
rm -rf "$PROJECT_DIR/webui"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" "$PROJECT_DIR/webui"
cp -r "$PROJECT_DIR/dist/." "$PROJECT_DIR/webui/"
chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/webui"

echo "==> Installing the Python package"
sudo -u "$SERVICE_USER" -H python3 -m venv "$PROJECT_DIR/.venv"
sudo -u "$SERVICE_USER" -H "$PROJECT_DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "$SERVICE_USER" -H "$PROJECT_DIR/.venv/bin/pip" install --quiet -e "$PROJECT_DIR"

echo "==> Installing the service unit"
install -d -m 700 /etc/clipforge
if [ ! -f "$ENV_FILE" ]; then
  install -m 600 "$PROJECT_DIR/deploy/oracle/env.example" "$ENV_FILE"
  echo "    created $ENV_FILE from the template"
else
  echo "    keeping the existing $ENV_FILE"
fi
install -m 644 "$PROJECT_DIR/deploy/clipforge.service" /etc/systemd/system/clipforge.service
systemctl daemon-reload
systemctl enable clipforge

cat <<EOF

==> Installed. Now edit the keys:

    sudo nano $ENV_FILE

At minimum set DEEPSEEK_API_KEY. For a public URL also set
CLIPFORGE_PUBLIC_URL. Keep TAILSCALE_ONLY=false.

Then start it:

    sudo systemctl start clipforge
    sudo systemctl status clipforge --no-pager
    curl -fsS localhost:8000/api/health

If the direct port is open in the Oracle security list:
    http://<PUBLIC_IP>:8000

Put Caddy or another access layer in front before exposing this to the
internet with real keys. This app has no built-in login.
EOF
