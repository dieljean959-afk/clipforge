# ClipMake backend — single-image deployment: FastAPI + ffmpeg.
#
#   docker build -t clipmake-api .
#   docker run --rm -p 10000:10000 --env-file .env.local \
#     -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" clipmake-api
#
# This image is the API only. In production the dashboard is served by the
# Cloudflare Worker on Sites, which proxies /api/* and /media/* here — so there
# is no Node build stage, and the build context needs nothing beyond the files
# in this package.
#
# The API listens on 0.0.0.0:$PORT (default 10000) so Render, Fly, Railway and
# `docker run -p` all work unchanged.

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLIPFORGE_HOST=0.0.0.0 \
    CLIPFORGE_PORT=10000 \
    CLIPFORGE_DATA_DIR=/app/data \
    CLIPFORGE_DOWNLOADS_DIR=/app/downloads \
    CLIPFORGE_OUTPUTS_DIR=/app/outputs

# ffmpeg/ffprobe are mandatory (render + probe). curl is for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Node.js runtime for yt-dlp's JS post-processing (heavily nested YouTube
# player hacks). yt-dlp itself is installed from PyPI below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the Python package first so dependency layers cache independently of
# source edits.
COPY pyproject.toml README.md ./
COPY app ./app
RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

COPY config.toml ./config.toml

# Durable, writable state. Mount these as volumes in production.
RUN mkdir -p /app/data /app/downloads /app/outputs
VOLUME ["/app/data", "/app/downloads", "/app/outputs"]

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD /bin/sh -c "exec curl -fsS http://127.0.0.1:${PORT:-10000}/api/health || exit 1"

CMD ["clipforge"]
