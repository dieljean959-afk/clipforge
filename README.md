# ClipForge

**Local video → viral clips.** Paste a link to a video you own (or attach the
file), and ClipForge finds the most viral moments and renders them as vertical
9:16 clips with animated, burned-in captions — 100% on your machine.

Forked from [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)
(MIT) and repurposed from "topic → stock-footage video" to "existing video →
viral clips". See [LICENSE](LICENSE).

## How it works

1. **Grabs** it — yt-dlp (Python API) pulls up to 1080p MP4 into `./downloads/`,
   self-updates on failure, and can borrow your browser cookies for age-gated
   videos. Manual file upload always works as the fallback.
2. **Transcribes** it — faster-whisper produces **word-level timestamps**
   (model size configurable: `small` default / `medium` / `large-v3`,
   pre-downloaded at boot so jobs never wait on a silent download).
3. **Finds** the hooks — the transcript is chunked and scored by Gemini
   (free tier). Top 5–15 windows become clips with titles, hook lines and
   scores 0–100. No key? The UI says exactly that — no fake heuristics.
4. **Renders** the clips — FFmpeg cuts each segment frame-accurately,
   MediaPipe face-tracking follows the speaker into the 9:16 crop (when
   installed), Silero VAD trims dead air (when installed), and ASS captions
   (bold white, thick `#39FF14` outline, word-pop animation, 2–3 words at a
   time, lower third) are burned into the MP4.

Thumbnails are real first frames, previews stream the real files, downloads
serve real MP4s, and "Export all" zips the real renders with `captions.txt`.

## Quick start

Requirements: **Python 3.11+** and **ffmpeg/ffprobe** on PATH
(`brew install ffmpeg` / `sudo apt install ffmpeg` / `winget install Gyan.FFmpeg`).

```bash
# 1. create .env with your Gemini key (free: aistudio.google.com/apikey):
#      GEMINI_API_KEY=your-key-here
#    optional:
#      YT_DLP_COOKIES_FROM_BROWSER=chrome
# 2. one command to run everything:
./start.sh                    # venv → deps → UI build → launch (API + worker, one process)
# optional extras (face tracking + silence trim):
./start.sh --with-extras
```

Open **http://127.0.0.1:8000**. First boot downloads the Whisper model
(~500 MB–3 GB) with a visible progress stage, then prints:

```
[boot] Whisper model loaded: small from ~/.cache/huggingface in 42.0s
[boot] LLM: gemini | key loaded: YES
In-process worker online, consuming queue: clipforge-local
```

The web UI is served by the API itself from `webui/` — `./start.sh` builds it
for you the first time (bun or npm). For UI development, `bun run dev` proxies
`/api` to the local backend.

## Architecture

```
FastAPI (app/main.py)  ── serves REST API + built UI
   └── InProcessWorker (app/worker.py)  ── daemon thread, same process
         └── Pipeline (app/task.py)
               ├── download.py   yt-dlp fetcher (Python API, self-updating)
               ├── subtitle.py   faster-whisper + ASS captions (Pillow fallback)
               ├── llm.py        Gemini clip detection (429 backoff, no heuristics)
               ├── vad.py        Silero VAD silence windows      (optional)
               ├── reframe.py    MediaPipe face-tracked crops    (optional)
               └── video.py      FFmpeg render 1080x1920 libx264/AAC
```

- **API and worker are one process** — "worker offline" is structurally
  impossible. Jobs queue in memory and re-queue on restart.
- State lives in `./data/` (JSON), videos in `./downloads/`, clips in
  `./outputs/` — nothing leaves your machine.
- `config.toml` holds the pipeline settings (MPT heritage, pruned);
  secrets live only in `.env`.

## Configuration

| Where | What |
|---|---|
| `config.toml` | Whisper model/precision, Gemini limits, subtitle style (font, size, outline color/width, words per cue, position), output size/quality, reframe & silence trim toggles, yt-dlp settings |
| `.env` | `GEMINI_API_KEY`, `YT_DLP_COOKIES_FROM_BROWSER` (e.g. `chrome`), optional `FFMPEG_PATH` / `FFPROBE_PATH` / `GEMINI_MODEL` |

## Rights & fairness

**Only clip content you own or have permission to use.** Clipping other
creators' videos may violate platform terms of service and copyright law.
The UI asks you to confirm this before every job.

## Verify the pipeline

```bash
python3 tools/compile_check.py   # compile-check every backend module
curl localhost:8000/api/health   # worker/warmup/LLM status
```

Then submit a video in the dashboard and watch the four stages report real
progress (yt-dlp %, whisper minutes processed, clip N/M rendering). Every
failure path surfaces the exact error — nothing fakes success.

## Removed in this fork

TTS/voice services, stock-footage (Pexels/Pixabay), MPT's script-generation
flow, and all cloud worker/VM machinery. Clips keep the original audio.
