# ClipForge

**Hosted video → viral clips.** Paste a link to a video you own (or attach the
file), and ClipForge finds the most viral moments and renders them as vertical
9:16 clips with animated, burned-in captions — processed on ClipForge's
server, with files deleted after you download them.

Forked from [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)
(MIT) and repurposed from "topic → stock-footage video" to "existing video →
viral clips". See [LICENSE](LICENSE).

## How it works

1. **Grabs** it — the server pulls up to 1080p MP4 via yt-dlp (Python API,
   self-updates on failure, borrows your browser cookies for age-gated
   videos). Manual file upload always works as the fallback.
2. **Transcribes** it — WhisperX runs VAD-filtered, batched `medium` inference
   and forced alignment for precise **word-level timestamps**. If the optional
   WhisperX/torch stack is unavailable, ClipForge visibly falls back to
   batched faster-whisper (`medium`, CPU int8) rather than failing silently.
   Both paths use anti-repetition/no-speech thresholds and overlapping
   resumable chunks. The active engine, device, compute type, model and batch
   size are reported by health checks, diagnostics and each job's logs.
3. **Finds** the hooks — the transcript is chunked and scored by an AI
   backend. Top 5–15 windows become clips with titles, hook lines and scores
   0–100. Only transcript **text** is ever sent — never audio or video. No
   key? The UI says exactly that — no fake heuristics.

   Two interchangeable backends, chosen in `config.toml` `[llm] provider`
   (or `CLIPFORGE_LLM_PROVIDER`), switchable from the dashboard:

   | Backend | Speed | Cost / privacy | Needs |
   |---|---|---|---|
   | `gemini` (default) | **Fastest** — the request overlaps local CPU work | Hosted API, free tier | `GEMINI_API_KEY` |
   | `ollama` | **Slower** — it shares the same CPU as Whisper, so it runs sequentially after the transcript exists | **Free and fully private** — the transcript never leaves the machine | `ollama serve` + the model pulled, no API key |

   This choice affects **detection only**. Transcription is always
   faster-whisper/WhisperX, and the dashboard says so next to the toggle.

   **Model rot self-heals:** if Google retires the configured Gemini model,
   the server parses the replacement model name out of the API error, retries
   the request, and persists the new model to `config.toml` `[gemini]` —
   no manual edit, no failed jobs.

### Gemini key is detected LIVE

The key is **never snapshotted at boot**. The backend re-reads the env files
whenever they change on disk (and on every health poll), so a key saved in
the platform's Keys/API keys tab — even minutes after the app started — is
picked up automatically. The dashboard status chip shows exactly where the
key was found, or precisely why a present-but-unusable value was rejected
(encrypted blob, placeholder text, malformed paste), and a **Re-check**
button forces an immediate re-read with the result in a toast. Supported
names: `GEMINI_API_KEY` (canonical), plus `GOOGLE_API_KEY`,
`GOOGLE_GENAI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, `GEMINI_KEY`,
`VITE_GEMINI_API_KEY`.
4. **Renders** the clips — FFmpeg cuts each segment frame-accurately,
   MediaPipe face-tracking follows the speaker into the 9:16 crop (when
   installed), Silero VAD trims dead air (when installed), and ASS captions
   (bold white, thick `#39FF14` outline, word-pop animation, 2–3 words at a
   time, lower third) are burned into the MP4.

Thumbnails are real first frames, previews stream the real files, downloads
serve real MP4s, and "Export all" zips the real renders with `captions.txt`.

## Deployment: how the backend is hosted

There is **no local setup** for the hosted app. The platform runs one managed
web process — the Vite dev server — and the `clipforgeBackend()` Vite plugin
(`vite-plugin-clipforge-backend.ts`) spawns and supervises FastAPI **as a
child of that web process**:

```
platform web process (Vite, managed by Freebuff)
  └── vite-plugin-clipforge-backend
        └── uvicorn app.main:app  (127.0.0.1:8000, respawned if it dies)
              └── InProcessWorker thread (API + worker = one process)
                    └── one child process per job
                          └── yt-dlp / WhisperX / ffmpeg run HERE, on the host
```

The browser calls same-origin relative `/api/...` URLs; the Vite proxy
forwards them to the backend. No localhost from the browser, no hardcoded
IPs, no CORS in play. Boot log shows: `[boot] API up on http://127.0.0.1:8000
(worker: local in-process)` and `backend is UP on 127.0.0.1:8000 — /api proxy
is live (same-origin)`.

### Subprocess support

The pipeline uses `subprocess` **inside the backend process** (ffmpeg/ffprobe
invocations and `pip install -U yt-dlp` for rot-recovery). yt-dlp itself runs
via its **Python API** (`import yt_dlp`) — no shell-outs for fetching. On
platforms that forbid subprocesses entirely, the app fails honestly at boot
(`ffmpeg not found on PATH` / clear ConfigError) rather than pretending to
render.

**Caption fallback:** if the host's ffmpeg build lacks libass (no `subtitles`
filter), the render path automatically switches to a pure-Python Pillow
fallback — each caption cue is rendered to a transparent PNG and burned in via
ffmpeg's overlay filter with per-cue time windows. Captions export either way;
they are never silently dropped.

**yt-dlp rot guard:** on download failure the server self-updates yt-dlp
(pip, or uv when the venv was created by uv — `ensurepip` is bootstrapped
automatically) and retries once before surfacing the exact error.

### Persistent disk

`downloads/`, `outputs/` and `data/` live inside the deployment sandbox and
survive between requests. They are **not** guaranteed across platform
redeploys — the retention cleaner also deletes finished job artifacts 24h
after completion (`[hosted] retention_hours`), so treat the hosted storage as
temporary: download your clips.

### Worker queue and live ETA

Jobs run FIFO in dedicated child processes. The pool size is bounded by both
configuration and actual CPU/RAM, so a small host automatically narrows itself
instead of overcommitting. Queued cards show their real FIFO position
(`waiting behind 2 other jobs`), which distinguishes a busy queue from a stuck
worker. A hung process group is killed by the watchdog; Retry kills the old
attempt before starting a fresh process and resumes completed on-disk stages.

During transcription, ClipForge shows remaining time. Before enough progress
exists it uses `audio duration × RTF` from measurements saved on that host;
afterwards it switches to elapsed-time progress. `/api/health` exposes the
current engine, device, compute type, batch size, RTF history and host limits.

### Plan limits

Sources longer than `max_source_min` (default **120 min / 2 hours**) are rejected at
submit time with a readable 413 — never a mid-render failure. The limit comes
from `/api/health` (`limits.maxSourceMin`) and is shown on the status card.
Long transcription work is split into overlapping five-minute resume chunks;
each completed chunk is cached atomically under the job's transcribe stage.

## Tailscale modes

### Mode A — private API (`TAILSCALE_ONLY=true`)

The pipeline API answers only tailnet/loopback clients; the public internet
gets `401 ClipForge API is tailnet-only`. Run `tailscale serve` on the host
(or connect directly from a tailnet device to the container's 100.x.y.z
address). Boot log prints `[boot] tailnet IP: 100.x.x.x` when tailscaled is
reachable, and the dashboard status card shows the tailnet IP.

```bash
# on the host (sidecar or same netns):
tailscale up --authkey $TAILSCALE_AUTHKEY --ephemeral
tailscale serve --bg 8000      # proxies tailnet → 127.0.0.1:8000
```

### Mode B — dashboard here, rendering at home (`WORKER_URL`)

If the hosted sandbox is too weak for long renders, dispatch the heavy work
to your own machine over the tailnet:

1. On your home machine: install this repo, ffmpeg, python3.11+, and run

   ```bash
   WORKER_TOKEN=<shared-secret> python3 tools/worker_agent.py \
       --server https://<your-clipforge-host> \
       --listen 0.0.0.0:8765 \
       --register --public-url http://<your-tailnet-ip>:8765
   ```

   (self-registration needs `CLIPFORGE_ALLOW_SELF_REGISTER=true` on the
   server; otherwise set `WORKER_URL`/`WORKER_TOKEN` in the platform env.
   Required variable names and safe placeholders are listed in `.env.example`;
   never paste secrets into source files.)

2. On the server: `WORKER_URL=http://<your-tailnet-ip>:8765`,
   `WORKER_TOKEN=<shared-secret>`.

The agent polls for dispatched uploads at `POST /agent/jobs`, runs the full
pipeline locally (its own Whisper model), and reports progress back through
`/api/internal/jobs/{id}/progress|logs|failed|complete` (Bearer
`WORKER_TOKEN`). The dashboard shows "home worker online/unreachable" from
`GET /agent/ping` probes. Note: rendered MP4s stay on the home machine in
dispatch mode — pull them from `GET /agent/output/{jobId}/{clip}.mp4`.

## Configuration

| Where | What |
|---|---|
| `config.toml` | Whisper model/precision, Gemini limits, subtitle style (font, size, outline color/width, words per cue, position), output size/quality, reframe & silence trim toggles, yt-dlp settings, `[hosted]` limits + dispatch |
| Platform env (Keys/API keys) | `GEMINI_API_KEY`, `YT_DLP_COOKIES_FROM_BROWSER`, `WORKER_URL`, `WORKER_TOKEN`, `TAILSCALE_ONLY`, `TAILSCALE_AUTHKEY`, `CLIPFORGE_MAX_SOURCE_MIN`, `CLIPFORGE_RETENTION_HOURS`, optional `FFMPEG_PATH` / `FFPROBE_PATH` |

## Running it yourself (dev)

Requirements: **Python 3.11+** and **ffmpeg/ffprobe** on PATH. The hosted
Freebuff dashboard is already managed for you; use its provided URL rather than
starting a duplicate local server. For a separate local installation:

```bash
./start.sh --all-extras
# or install only the large WhisperX stack:
./start.sh --with-whisperx
```

`--with-whisperx` is deliberately opt-in: WhisperX can add several gigabytes of
Torch/alignment dependencies and model downloads. On CPU, ClipForge uses
`medium` with int8 and a host-sized batch; on CUDA it uses float16. Run
`python -m app.doctor` after installation to verify ffmpeg capabilities, GPU,
engine availability, effective device/compute type, model fit and disk space.
`GET /api/doctor` returns the same checks as JSON.

## Rights & fairness

**Only clip content you own or have permission to use.** Clipping other
creators' videos may violate platform terms of service and copyright law.
The UI asks you to confirm this before every job.

## Verify the pipeline

```bash
python3 tools/compile_check.py   # compile + undefined-name check every backend module
python3 tools/transcribe_unit_test.py  # chunking, repetition, engine logic, .wav.part regression
python3 tools/asr_engine_test.py       # mocked queue/retry + ETA checks
python3 tools/detect_backend_test.py   # Gemini vs Ollama detection, text-only payloads
bun tools/overall_status_test.ts       # job Done/Failed/overall-progress indicator logic
python3 -m app.doctor                  # hardware/engine/ffmpeg diagnostics
curl localhost:8000/api/health         # worker/warmup/LLM/system status
curl "localhost:8000/api/dev/verify-clips?jobId=<id>"   # ffprobe every rendered clip
python3 tools/smoke_test.py      # API + worker + honest failure paths
python3 tools/e2e_render_test.py # real ffmpeg: cut → crop → captions → thumbnail
```

`tools/transcribe_unit_test.py` includes a **live** regression for the
`.wav.part` extraction bug: it runs real ffmpeg to a `source.wav.part` temp
path and asserts the extraction succeeds, lands the atomic rename, and leaves
no `.part` behind. That path previously died with `Invalid argument`, because
ffmpeg infers the output muxer from the file extension and `.part` is not a
known container. Every ffmpeg call in the repo now pins `-f` explicitly.

For a real same-audio decode benchmark (requires enough RAM/disk and, ideally,
CUDA), run `CLIPFORGE_BENCH=1 CLIPFORGE_BENCH_SECONDS=120 python3
tools/asr_engine_test.py`. The optional benchmark reports wall time and RTF;
full WhisperX before/after and 2-hour accuracy comparisons must be run on a
host that can hold the torch/alignment/model stack, not on a constrained
1-CPU/2-GiB deployment.

## Removed in this fork

TTS/voice services, stock-footage (Pexels/Pixabay), MPT's script-generation
flow, and all cloud VM/OCI machinery. Clips keep the original audio.
