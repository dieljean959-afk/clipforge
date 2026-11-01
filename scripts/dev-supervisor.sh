#!/usr/bin/env bash
# ClipMake dev supervisor — keeps the preview alive.
#   * starts Vite (the platform preview origin) on PORT (default 3000)
#   * restarts it automatically if it exits
#   * allows the exact preview hostname (never allowedHosts: true)
#   * clears a stale backend on 8000 before each restart so the respawn can bind
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-3000}"
HOST="${PREVIEW_HOSTNAME:-}"

# Only the exact preview host (plus localhost for direct curl checks) is allowed.
export __VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS="$HOST"

while true; do
  echo "[supervisor] $(date -Is) starting vite on 0.0.0.0 + [::]:${PORT} (host allowlist: ${HOST:-none})"
  node node_modules/vite/bin/vite.js --host :: --port "$PORT" >> preview-dev.log 2>&1
  code=$?
  echo "[supervisor] $(date -Is) vite exited with code ${code}; restarting in 3s"
  # vite died: its uvicorn child may be orphaned and still holding 8000.
  fuser -k 8000/tcp 2>/dev/null || true
  sleep 3
done
