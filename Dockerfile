# ClipMake backend — Render Docker web service.
#
# Build:   docker build -t clipmake-api .
# Run:     docker run --rm -p 10000:10000 --env-file .env.local \
#             -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" clipmake-api
#
# This image is the API only. Render injects $PORT and expects the process to
# listen on 0.0.0.0:$PORT. The CMD below does exactly that, falling back to
# 10000 when $PORT is unset (local docker run).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLIPFORGE_HOST=0.0.0.0 \
    CLIPFORGE_PORT=10000 \
    CLIPFORGE_DATA_DIR=/app/data \
    CLIPFORGE_DOWNLOADS_DIR=/app/downloads \
    CLIPFORGE_OUTPUTS_DIR=/app/outputs

# ffmpeg/ffprobe are mandatory for rendering and probing source video.
# curl is used by the Docker HEALTHCHECK (Render also runs its own probes).
# nodejs is required by yt-dlp's JS post-processing for heavily nested sites.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ffprobe curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the Python package + its dependencies from pyproject.toml.
# The repo uses hatchling as the build backend; there is no requirements.txt.
# uv.lock exists but plain pip does not read it, so we install from pyproject.toml.
COPY pyproject.toml README.md ./
COPY app ./app
RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

# Runtime config the app reads at startup.
COPY config.toml ./config.toml

# Durable, writable state. Mount these as volumes in production so they persist
# across restarts and redeploys.
RUN mkdir -p /app/data /app/downloads /app/outputs
VOLUME ["/app/data", "/app/downloads", "/app/outputs"]

EXPOSE 10000

# Docker HEALTHCHECK: local-only. Render runs its own TCP/HTTP probes, so this
# line is only useful for `docker run` healthchecks. It tracks the same port the
# app actually listens on (${PORT:-10000}).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD /bin/sh -c "exec curl -fsS http://127.0.0.1:${PORT:-10000}/api/health || exit 1"

# Start the server. $PORT is set by Render (default 10000). The CMD below binds
# 0.0.0.0:$PORT so Render's port probe can reach the API immediately.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --log-level info"]
