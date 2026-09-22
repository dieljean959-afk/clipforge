#!/usr/bin/env bash
# ClipForge — one-command local bootstrap.
#   ./start.sh               # create venv, install deps, launch the app
#   ./start.sh --with-extras # also install silero-vad + mediapipe (silence trim + face reframe)
#
# The Whisper model (~500MB-3GB depending on size) is pre-downloaded at app
# boot with a visible progress stage — never silently mid-job.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

echo "==> ClipForge local bootstrap"

if [ ! -x "$PY" ]; then
  echo "==> Creating virtualenv at $VENV"
  python3 --version || { echo "python3 is required (3.11+)"; exit 1; }
  python3 -m venv "$VENV"
fi

echo "==> Installing dependencies (pip)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e .

if [ "${1:-}" = "--with-extras" ]; then
  echo "==> Installing optional extras (silero-vad, mediapipe)"
  "$PY" -m pip install --quiet ".[vad,reframe]"
fi

echo "==> Checking ffmpeg"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found on PATH — install it first:"
  echo "  macOS:   brew install ffmpeg"
  echo "  Ubuntu:  sudo apt install ffmpeg"
  echo "  Windows: winget install Gyan.FFmpeg"
  echo "The app will fail fast at boot until ffmpeg is available."
fi

# Build the dashboard if bun/node is present and the webui has never been built.
if [ ! -f webui/index.html ]; then
  if command -v bun >/dev/null 2>&1 || command -v npm >/dev/null 2>&1; then
    echo "==> Building dashboard (first run)"
    if command -v bun >/dev/null 2>&1; then
      bun install && bun run build
    else
      npm install && npm run build
    fi
    mkdir -p webui && cp -r dist/* webui/
  else
    echo "NOTE: bun/node not found — skipping UI build. The API will run without the dashboard."
  fi
fi

echo "==> Starting ClipForge (API + in-process worker, one process)"
echo "    Dashboard: http://127.0.0.1:8000"
exec "$PY" -m uvicorn app.main:app --host "${CLIPFORGE_HOST:-127.0.0.1}" --port "${CLIPFORGE_PORT:-8000}"
