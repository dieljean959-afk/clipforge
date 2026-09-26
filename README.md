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

### CPU-only render degradation

A 1080p60 source encoded to 9:16 with libx264 on a single vCPU runs at roughly
**0.1x realtime**, so a 15-second clip takes minutes and long sources never
finish. `app/encode.py` makes that degrade gracefully instead of stalling, and
every step is reported by `app.doctor` and `GET /api/health` under `encode`:

| Lever | Behaviour |
|---|---|
| **Hardware encoder** | NVENC → VAAPI → QSV is auto-detected and used when present. Detection is **functional**: ffmpeg must both list the encoder *and* successfully encode one frame. A `/dev/dri` node without a working driver is rejected — otherwise a merely-slow render becomes a hard "unknown encoder" failure. |
| **x264 preset** | With no hardware encoder, the preset is lowered to `ultrafast` (never below `superfast`) and the downgrade is logged, not silent. Costs compression efficiency, not correctness. |
| **Frame sampling** | Scene-cut detection (`scdet`) and face-tracked reframing sample every Nth frame — 2.0/s with hardware, 0.75/s on CPU-only, with the stride scaled to the source fps. Shot boundaries survive coarse sampling because a cut is a large instantaneous score change. |
| **Render deadline** | Two separate, labelled deadlines: **20 min** on a hardware encoder (fail fast, a broken job should not hold a worker slot) and **90 min** on the CPU-only fallback. The timeout error names which one was hit and what to do about it. |

The throwaway intermediate cut stays `libx264/veryfast` on purpose — it is
deleted at the end of the render, and the final encode is where quality is
decided.

✅ **NVENC detection is confirmed on a GPU host.** The functional probe passes
there — ffmpeg lists `h264_nvenc` *and* a real 64x64 frame encodes through it —
so the acceptance path, not just the rejection path, has been observed:

```
encoder: h264_nvenc | hardwareEncoded: True | preset veryfast (no downgrade)
codec args: -c:v h264_nvenc -cq 20 -b:v 0 -tune p4
deadline: 20 min | analysis sampling: 2.00/s at 60fps
```

What that does and does not establish: the encoder initialises and accepts our
argument set, so the probe is not merely trusting ffmpeg's encoder list. It
does not yet prove sustained 1080p60 throughput, absence of frame drops, or
correct output at clip length — a 64x64 single frame is a liveness check, not a
benchmark.

⚠️ **Still outstanding.** Real-content render timings on a full-length clip, and
WhisperX `large-v3` RTF before/after, have not been measured. A 1-vCPU/2-GiB
sandbox encodes at ~0.1x realtime and is not a valid measurement of either.

### Surviving a host restart

On a host that restarts containers (managed sandboxes, autoscaling platforms),
the runtime sends **SIGTERM** to in-flight processes partway through a render.
That is not a pipeline bug, and it is not OOM — but it used to be fatal:

| Symptom before | Now |
|---|---|
| `Worker process exited unexpectedly (code -15)` with no cause | Named for what it is: an external SIGTERM, explicitly **not** OOM. `-9` and `-15` are distinct. |
| Job frozen at `rendering` with no terminal state | Resumed on next boot from cached stages |
| Job killed once → `stage: failed` **forever** | Auto-resumed, up to `CLIPFORGE_INTERRUPT_MAX_RESUMES` (default 5) |
| Old failures could never be recovered | Boot recovery also rescues jobs whose recorded failure was a signal banner |
| `ffmpeg failed:` with eight progress frames and no error | Exit code + signal name lead; progress noise filtered out |

How it decides, at every boot:

1. **mid-flight** — stage is `grabbing/transcribing/finding/rendering/queued`:
   the process died before writing a terminal state, so re-queue it.
2. **flagged interrupted** — a shutdown hook or a previous SIGTERM marked it.
3. **failed, but from a signal** — `stage: failed` whose message is a signal
   banner. The stage cache is intact, so the false terminal state is cleared and
   the job resumes.

The retry is **bounded** — after `INTERRUPT_MAX_RESUMES` interrupts a job is
left failed rather than looping. A job with a real content error (a 404, a bad
URL) is *not* recovered: only signal-caused failures match, so retries cannot
loop on something that will always fail.

Resume is **stage-level**, so the expensive work is not repeated: a job killed
during rendering restarts at `captions`, reusing the download, transcript and
detected candidates already on disk.

**A per-clip timeout cannot solve this.** A timeout inside a process cannot
fire once that process has been killed; only boot-time reconciliation can.

### Multi-language caption tracks

`[translate]` writes **sidecar** subtitle files next to each clip — separate
from the burned-in ASS captions, which style the on-screen text:

```
outputs/<job>/clip01_en.srt  clip01_fr.srt  clip01_ja.srt  …  (and .vtt)
```

Off by default (each language costs one LLM call per batch of cues). Enable and
pick languages in `config.toml`:

```toml
[translate]
enabled = true
source_lang = "en"
languages = ["en", "fr", "es", "pt", "de", "ru", "it", "ja", "ko", "zh", "ar", "hi"]
glossary = ["ClipForge"]
```

How it behaves, and why:

| | |
|---|---|
| **Timing** | Never touched. Only `cue["text"]` is replaced, so a bad translation can reword a caption but can never desynchronise it. |
| **Source track** | `source_lang` is copied **verbatim** from the transcript, never round-tripped through a translator. |
| **Engine** | Reuses the `[llm]` provider (gemini or ollama) — no new dependency, no extra key, no model download. |
| **Whisper cannot do this** | Its `task="translate"` targets **English only**, so it covers 1 of the 12 languages. Do not fold this into the ASR path. |
| **Line width** | Per-script, not global: 42 chars for Latin/Cyrillic, 40 for Arabic/Devanagari, **18** for full-width CJK. A single global limit overflows badly in Japanese. |
| **Overflow** | An over-long cue is **flagged** and kept in full — never truncated. Losing subtitle text silently is worse than a long line. |
| **Positional integrity** | A reply with the wrong number of lines is a hard error. Padding or truncating would shift every following caption. |
| **Glossary** | Terms are requested verbatim, and a term present in the source but missing from the output is logged. A term the speaker never said cannot raise a false alarm. |
| **Failure isolation** | One language failing yields a missing track and a log line; the other eleven still write, and the render still succeeds. |
| **Resume** | An existing non-empty track is reused, so Retry does not re-pay for the same translations. |

Tracks are downloadable per clip (`GET /api/clips/{id}/tracks/{file}`, served by
matching against recorded paths, never by joining user input) and appear as
language chips on each clip card in the dashboard.

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
python3 tools/encode_profile_test.py   # encoder/preset/deadline degradation logic
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
