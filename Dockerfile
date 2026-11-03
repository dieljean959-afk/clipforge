# ClipMake backend — Render / Docker web service (API only).
#
# Build:   docker build -t clipmake-api .
# Run:     docker run --rm -p 10000:10000 --env-file .env.local \
#             -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" clipmake-api
#
# Render injects $PORT and expects the process to listen on 0.0.0.0:$PORT. The
# CMD below does exactly that, falling back to 10000 when $PORT is unset (local
# docker run).
#
# The base image is pinned to bookworm on purpose: the apt package names below
# were verified against Debian 12, and letting `python:3.11-slim` float to a
# newer Debian release is how a green build silently turns into a failing one.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLIPFORGE_HOST=0.0.0.0 \
    CLIPFORGE_PORT=10000 \
    CLIPFORGE_DATA_DIR=/app/data \
    CLIPFORGE_DOWNLOADS_DIR=/app/downloads \
    CLIPFORGE_OUTPUTS_DIR=/app/outputs

# ffmpeg is the one mandatory media dependency. There is NO "ffprobe" package in
# Debian: the ffmpeg 5.1.x package ships BOTH /usr/bin/ffmpeg and
# /usr/bin/ffprobe, so listing "ffprobe" here makes apt-get abort the whole build
# with "E: Unable to locate package ffprobe".
# libgomp1 is the OpenMP runtime that ctranslate2 (faster-whisper) dlopen()s at
# runtime; python:3.11-slim does not ship it, and without it the first
# transcription fails with "libgomp.so.1: cannot open shared object file".
# curl backs the HEALTHCHECK at the bottom of this file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Point the app at the binaries explicitly. app.config.ffmpeg_bin()/ffprobe_bin()
# read these first and only then fall back to shutil.which(), so this removes the
# PATH guesswork the boot probe would otherwise depend on. The assert keeps a
# broken base layer from reaching Render: the build fails here instead of the
# deploy booting with a red "ffmpeg probe failed" line and ffmpeg_ready=false.
ENV FFMPEG_PATH=/usr/bin/ffmpeg \
    FFPROBE_PATH=/usr/bin/ffprobe
RUN ffmpeg -version >/dev/null && ffprobe -version >/dev/null

WORKDIR /app

# Install the Python package from pyproject.toml (hatchling backend; there is no
# requirements.txt, and plain pip cannot read uv.lock).
#
# This is deliberately an EDITABLE install (`-e .`), not a wheel install:
# app.config derives ROOT from the parent of the app/ package, so editable keeps
# ROOT=/app and load_config() finds the config.toml copied below. A non-editable
# install would place app/ under site-packages, ROOT would point there, the
# container would boot with no config file and the app would start degraded.
COPY pyproject.toml README.md ./
COPY app ./app
RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

# Runtime config the app reads at startup. Must land after the editable install
# so it sits next to the package the app resolves ROOT from.
COPY config.toml ./config.toml

# Durable, writable state. Mount these as volumes in production so they persist
# across restarts and redeploys.
RUN mkdir -p /app/data /app/downloads /app/outputs
VOLUME ["/app/data", "/app/downloads", "/app/outputs"]

EXPOSE 10000

# Docker HEALTHCHECK: local-only, for `docker run`. Render runs its own probes.
# /api/health is served without touching models, the store, the worker or
# yt-dlp, so this is a real liveness signal and not a slow readiness check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD /bin/sh -c "exec curl -fsS http://127.0.0.1:${PORT:-10000}/api/health || exit 1"

# Start the single public server, bound to 0.0.0.0 so Render's port probe reaches
# it immediately. `exec` hands PID 1 to uvicorn: it receives SIGTERM directly
# (graceful shutdown on redeploy) and reaps its own ffmpeg/ffprobe children
# without a shell sitting in between.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --log-level info"]
