#!/usr/bin/env bash
# Start (or restart) the ClipMake preview: Vite on $PORT + the FastAPI child.
# Idempotent, pidfile-based — never kills by process-name pattern.
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-3000}"
HOST="${PREVIEW_HOSTNAME:-}"
PIDFILE=".preview-supervisor.pid"

if [ -f "$PIDFILE" ]; then
  OLD="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "${OLD:-}" ] && kill -0 "$OLD" 2>/dev/null; then
    echo "[preview-up] supervisor already running (pid $OLD)"
    exit 0
  fi
fi

PREVIEW_HOSTNAME="$HOST" PORT="$PORT" nohup bash scripts/dev-supervisor.sh \
  >> supervisor.log 2>&1 &
echo $! > "$PIDFILE"
echo "[preview-up] supervisor pid $(cat "$PIDFILE") on port $PORT (host: ${HOST:-none})"
