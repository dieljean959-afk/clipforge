#!/usr/bin/env bash
# ClipForge — one-command local bootstrap.
#   ./start.sh                  # create venv, install deps, launch the app
#   ./start.sh --with-extras    # also install silero-vad + mediapipe
#   ./start.sh --with-whisperx  # install WhisperX (large torch/alignment stack)
#   ./start.sh --all-extras     # install every optional backend/feature
#
# The Whisper model (~500MB-3GB depending on size) is pre-downloaded at app
# boot with a visible progress stage — never silently mid-job. WhisperX remains
# opt-in so constrained hosts do not spend their disk/RAM on it unexpectedly.
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

# venvs created by `uv venv` ship without pip — bootstrap it so installs work.
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "==> Bootstrapping pip (ensurepip)"
  "$PY" -m ensurepip --upgrade >/dev/null
fi

echo "==> Installing dependencies (pip)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e .

WITH_EXTRAS=0
WITH_WHISPERX=0
for arg in "$@"; do
  case "$arg" in
    --with-extras) WITH_EXTRAS=1 ;;
    --with-whisperx) WITH_WHISPERX=1 ;;
    --all-extras) WITH_EXTRAS=1; WITH_WHISPERX=1 ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done

if [ "$WITH_EXTRAS" = "1" ]; then
  echo "==> Installing optional extras (silero-vad, mediapipe)"
  "$PY" -m pip install --quiet ".[vad,reframe]"
fi

if [ "$WITH_WHISPERX" = "1" ]; then
  echo "==> Installing WhisperX (large download; medium + alignment models are larger)"
  "$PY" -m pip install --quiet ".[whisperx]"
  echo "==> Checking WhisperX import and hardware selection"
  "$PY" -m app.doctor || true
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
