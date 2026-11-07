# syntax=docker/dockerfile:1
#
# Clipmake in one container.
#
# The hosted app is a single Python process: `uvicorn app.main:app` serves the
# API *and* the built SPA (app/main.py mounts ./webui at "/"), and spawns
# ffmpeg / yt-dlp / the decoder as child processes per job. The Vite dev server
# the platform runs in the sandbox (vite-plugin-clipforge-backend.ts) is a dev
# harness, not part of the runtime — here the frontend is built once and served
# as static files by FastAPI.
#
#   docker build -t clipmake .
#   docker run --rm -p 8000:8000 -v clipmake-data:/data clipmake
#
# Everything writable — job artifacts, renders, the Whisper weights, the
# wav2vec2 aligner — lives under /data, so the container itself stays read-only
# after the first boot and a restart does not re-download ~1.4 GB of models.

# --------------------------------------------------------------------------- #
# 1. frontend: build src/ -> webui/
# --------------------------------------------------------------------------- #
# `webui/` is what app/main.py mounts and it is gitignored, so it must be built
# here. The repo's own `bun run build` is `tsc -b && vite build`; the extra
# --outDir mirrors what the deployed tree already contains.
FROM oven/bun:1-debian AS web
WORKDIR /web

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY tsconfig.json tsconfig.app.json tsconfig.node.json ./
COPY vite.config.ts vite-plugin-clipforge-backend.ts postcss.config.cjs components.json ./
COPY index.html sst-env.d.ts vly-toolbar-readonly.tsx ./
COPY src ./src
COPY public ./public

# Optional: the SPA reads VITE_* values at build time. Leave empty for the
# local/sandbox build; pass --build-arg VITE_CONVEX_URL=... to bake a deployment.
ARG VITE_CONVEX_URL=""
ENV VITE_CONVEX_URL=${VITE_CONVEX_URL}
RUN bun run build -- --outDir webui --emptyOutDir

# --------------------------------------------------------------------------- #
# 2. runtime: Python API + worker + ffmpeg
# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg/ffprobe do every cut, reframe, caption overlay and loudness pass;
# fonts-dejavu-core is the libass fallback behind the bundled app/fonts/*.ttf.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
      fonts-dejavu-core \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so a code change does not reinstall CTranslate2/numpy.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install . \
 && pip install "onnxruntime>=1.17"

# Optional smart reframe (face-tracked crop). Off by default: mediapipe +
# opencv add ~250 MB, and app/reframe.py falls back to a centred crop when they
# are missing. Enable with --build-arg INSTALL_REFRAME=true.
ARG INSTALL_REFRAME=false
RUN if [ "$INSTALL_REFRAME" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
      && pip install ".[reframe]" && rm -rf /var/lib/apt/lists/*; \
    fi

COPY config.toml ./
COPY --from=web /web/webui ./webui

# Non-root, and every writable path on the volume. The app creates these dirs
# on boot if they are missing; pre-creating them just gets the ownership right.
RUN useradd --create-home --uid 10001 clipmake \
 && mkdir -p /data/data /data/downloads /data/outputs /data/uploads \
            /data/media /data/.cache/huggingface \
 && chown -R clipmake:clipmake /data /app

# Relative paths in config.toml ("data", "downloads", "outputs") would land in
# the image layer; point them at the volume. HF_HOME is what keeps the Whisper
# and aligner weights between restarts.
ENV CLIPFORGE_DATA_DIR=/data/data \
    CLIPFORGE_DOWNLOADS_DIR=/data/downloads \
    CLIPFORGE_OUTPUTS_DIR=/data/outputs \
    CLIPFORGE_UPLOADS_DIR=/data/uploads \
    CLIPFORGE_STORAGE_DIR=/data/media \
    HF_HOME=/data/.cache/huggingface \
    CLIPFORGE_HOST=0.0.0.0 \
    CLIPFORGE_PORT=8000

USER clipmake
WORKDIR /app
EXPOSE 8000
VOLUME ["/data"]

# The decoder loads on first use, not at boot, so a slow start is normal.
# python -c, not curl: the runtime image has no curl and does not need one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

# The Keys UI writes secrets to /app/.env.local at runtime, which is why /app
# is owned by the container user above — mount that file as a secret if you
# would rather inject keys from the environment.
#
# One process: API + in-process worker + static SPA. --proxy-headers and the
# explicit host/port keep the same-origin /api paths working behind any proxy.
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
