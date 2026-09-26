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

   Interchangeable backends, chosen in `config.toml` `[llm] provider`
   (or `CLIPFORGE_LLM_PROVIDER`), switchable from the dashboard:

   | Backend | Speed | Cost / privacy | Needs |
   |---|---|---|---|
   | `groq` (default) | **Fast** — the request overlaps local CPU work | Hosted API, OpenAI-compatible, ~30 req/min free tier (calls are paced below that) | `GROQ_API_KEY` |
   | `gemini` | **Fast** | Hosted fallback, used when the Groq quota is spent | `GEMINI_API_KEY` |
   | `ollama` | **Slower** — it shares the same CPU as Whisper, so it runs sequentially after the transcript exists | **Free and fully private** — the transcript never leaves the machine | `ollama serve` + the model pulled, no API key |

   This choice affects **detection only**. Transcription is always
   faster-whisper/WhisperX, and the dashboard says so next to the toggle.

   There is deliberately **no automatic failover between providers**. A
   quota error fails the stage loudly rather than silently scoring clips
   with a model the operator did not pick — the same "no silent fallback"
   rule the rest of this module follows.

   **Groq specifics.** Groq speaks the OpenAI chat-completions wire format,
   so the client is a plain `POST {base_url}/chat/completions` with an
   `Authorization: Bearer $GROQ_API_KEY` header — no new Python dependency
   (stdlib `urllib`, same as the Ollama client). Two defences against the
   30 req/min free tier: a process-wide floor between calls
   (`[groq] min_interval_sec`, default 2.1s ≈ 28 rpm) so a 60-chunk
   transcript never bursts into the limit, and 429 handling that honours the
   server's `Retry-After` when present and otherwise backs off exponentially
   with jitter. `GROQ_BASE_URL` repoints the client, which is how
   `tools/llm_provider_test.py` tests it against a local stub.

   **Model rot does NOT self-heal on Groq.** Google names the replacement
   model in its 404, so Gemini can auto-migrate; Groq does not. A retired
   model therefore fails immediately (no retry burn) with the exact API error
   and a pointer to `console.groq.com/docs/models`. Set `[groq] model` and
   re-run. `python -m app.doctor` with `CLIPFORGE_DOCTOR_LIVE=1` lists the
   models your key can actually see and warns if the configured one is not
   among them.

### Detection keys are detected LIVE

The key is **never snapshotted at boot**. The backend re-reads the env files
whenever they change on disk (and on every health poll), so a key saved in
the platform's Keys/API keys tab — even minutes after the app started — is
picked up automatically. The dashboard status chip shows exactly where the
key was found, or precisely why a present-but-unusable value was rejected
(encrypted blob, placeholder text, malformed paste), and a **Re-check**
button forces an immediate re-read with the result in a toast. The chip
follows the **active** detection backend, so it never shows Gemini's key
state while Groq is the one actually making calls.

Supported names: `GROQ_API_KEY` (canonical) or `GROQ_KEY`; `GEMINI_API_KEY`
(canonical), plus `GOOGLE_API_KEY`, `GOOGLE_GENAI_API_KEY`,
`GOOGLE_GENERATIVE_AI_API_KEY`, `GEMINI_KEY`, `VITE_GEMINI_API_KEY`.

Because the platform's Keys tab has no edit for an existing variable, the
app exposes in-dashboard **Set / Replace / Remove** for both
(`POST|DELETE /api/settings/groq-key`, `.../gemini-key`) so a key can be
pasted from a phone with no terminal. Set and delete both mirror into the
process env, because the live resolver prefers OS env and a stale copy from
an older `.env.local` would otherwise shadow the new value.

---

## The working pipeline, exactly

Every command below is the one the code actually runs, extracted from
`app/download.py`, `app/subtitle.py`, `app/llm.py` and `app/video.py` — not
a reconstruction. The end-to-end reference run is the `O0yDqRwmSE` batch
(`data/jobs/c57d7782c548.json`): 974 words, 930.17s of audio, 10 clips,
349.61s of total output, `stage=complete`, `error=None`.

### 1. Download — yt-dlp (Python API, not a shell command)

`app/download.py: download()`. There is no `yt-dlp` CLI invocation; the
options are passed as a dict:

```python
ydl_opts = {
    "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]",
    "merge_output_format": "mp4",
    "outtmpl": str(dest_dir / "source.%(ext)s"),
    "noplaylist": True,
    "quiet": True, "no_warnings": True,
    "retries": 3,
    "progress_hooks": [_hook],
}
# plus "cookiesfrombrowser" when [download] cookies_from_browser is set
```

Equivalent CLI, if you want to reproduce it by hand:

```bash
yt-dlp -f "bv*[height<=1080]+ba/b[height<=1080]" --merge-output-format mp4 \
       -o "source.%(ext)s" --no-playlist --retries 3 "<URL>"
```

Failure policy: on error, `pip install -U yt-dlp` is run, the module is
reloaded, and the fetch is retried **once**. Both failures surface the exact
yt-dlp message (that is what the `not-a-video.mp4` test job shows).

### 2. Transcribe — faster-whisper, word-level timings

Confirmed from the artifact itself:

```
$ python -c "import json;print(json.load(open('data/work/c57d7782c548/transcribe/words.json')))"
{'duration': 930.167, 'model': 'medium', 'engine': 'faster-whisper', 'rtf': 0.1217}
```

**Yes — 974 words, faster-whisper, and yes, those word timings drive clip
boundary detection.** The file holds 974 `{start, end, word}` entries
(first: `0.16→0.30 'Y'`, last: `929.92→930.08 'bro'`).

Note the recorded `model` is `medium` while the host actually ran `small`:
the 2.0 GiB container cannot hold `medium` (it needs ~2.4 GiB and gets
OOM-killed), so `app/hostlimits.py` downgrades at boot. `rtf 0.1217` is the
measured real-time factor. Health and the dashboard report the **resolved**
model, not the configured one.

### 3. Detect — the LLM returns WORD INDICES, not timestamps

This is the load-bearing design decision. The model never does time math:

```python
# app/llm.py — transcript chunked to ~120s, each word numbered
candidates = [{"start_word": int, "end_word": int,
               "score": 0-100, "title": str, "hook_text": str}, ...]
# then, against the SAME faster-whisper words:
start = words[start_word]["start"]
end   = words[end_word]["end"]
```

So the clip boundaries are exactly faster-whisper word boundaries. If
transcription changes, the cuts move with it — there is no second,
independent source of timing. `tools/llm_provider_test.py` asserts this
resolves end-to-end through the real parser.

### 4. Crop + render — ONE encode pass

`app/video.py: render_clip()`. The full command, with `-ss` **before** `-i`
(input-side seek — no intermediate file):

```bash
ffmpeg -y -ss <start> -i source.mp4 -t <dur> \
  -vf "<crop-pan>,scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[,silence-trim],subtitles=clip_NN.ass" \
  <codec_args> -c:a aac -b:a 160k -f mp4 clip_NN.mp4
```

- **crop-pan** — `crop_pan_filter()` emits a time-interpolated `x` so
  MediaPipe face tracking follows the speaker. A window marked `"cut": True`
  makes `x` **snap** instead of interpolating, because the subject genuinely
  jumps at a shot boundary and a smooth pan through it would be a lie.
  With no windows: `crop=<crop_w>:ih:(in_w-crop_w)/2:0` (honest centre crop).
- **codec_args** — from `app/encode.py`; on this 1-vCPU/2-GiB host the
  functional encoder probe finds no usable VA-API/QSV, so it falls back to
  `libx264` with the preset clamped to `ultrafast`.
- **-c:a aac -b:a 160k** — the source audio is *not* pre-extracted any more,
  so it is re-encoded rather than copied.

**One pass, not two.** The original code cut each segment into a throwaway
`.tmp.mp4` and then re-encoded that into the real output. Measured on real
1080p60: 173s of a 246.5s clip (70%) producing a file that was `unlink()`ed
at function end. Removing it took clip 5 from ~204s to **75.6s** (0.54x
realtime), a ~2.7x speedup, and is what let a 10-clip job finish inside a
host-restart window. `tools/e2e_render_test.py` monkeypatches `video._run` and
**fails if a render ever makes more than one video encode pass** or writes a
`.tmp.mp4` again.

### 5. Captions — ASS built per clip, burned with libass

`app/subtitle.py: write_ass()` writes `data/work/<job>/captions/clip_NN.ass`
and ffmpeg burns it with the `subtitles=` filter. There is no `drawtext` path.

Timing is **clip-relative**: `build_ass_events(cues, settings,
start_offset=<clip start>)` subtracts the clip start, so a cue at 10.0s of
the source becomes `0:00:00.00` in a clip that starts at 10.0s.

The real `clip_05.ass` from the reference batch (21 `Dialogue` lines):

```ini
PlayResX: 1080
PlayResY: 1920
Style: ClipForge,Arial Black,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,
       -1,0,0,0,100,100,0,0,1,2,2,2,60,60,268,1
Dialogue: 0,0:00:00.00,0:00:02.23,ClipForge,,0,0,0,,What did
Dialogue: 0,0:00:02.23,0:00:02.45,ClipForge,,0,0,0,,he say?
```

> **Correction worth flagging:** that style line is **not** "Bold pop
> (default)". Font size 48, black outline, `MarginV 268` (= `1920 ×
> (1 − 0.86)`) is the **`lower_third`** preset. The job was submitted with
> `options.captionPreset = "lower_third"`, so the render is correct — the
> render faithfully used the style that was actually selected. But nothing
> asserted that mapping, so a wrong style would have been indistinguishable
> from a right one. `tools/caption_preset_test.py` now checks every preset's
> size/colour/outline/shadow/border against its `Style:` line, that an
> omitted preset resolves to `bold_pop`, and that the clip-relative offset is
> applied to both cue start and end.

**Bold pop (default)** is `font_size 68`, primary `#FFFFFF`, secondary and
outline `#39FF14`, `outline_width 5`, `shadow 1`, baseline at 78% height,
and carries the pop animation on each cue:

```ini
{\fad(60,40)\fscx84\fscy84\t(0,90,\fscx105\fscy105)\t(90,140,\fscx100\fscy100)}
```

If the ffmpeg build has no libass, `render_clip()` falls back to rendering
each cue as a transparent PNG and compositing with `overlay` — captions are
never silently dropped.
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

| **Single encode pass** | A clip is cut, reframed and captioned in **one** ffmpeg pass that seeks the source directly. There used to be a second pass that cut the segment into a throwaway `.tmp.mp4` first, which was then re-encoded into the real output. |

#### Why the intermediate pass was removed

`render_clip` used to make two encode passes per clip: the first cut the segment
into a throwaway `.tmp.mp4`, the second re-encoded that into the final file.
Measured on a real 1080p60 source (44.4s clip, one vCPU):

```
05:05:49  pass 1  -ss 477.140 -i source.mp4 -t 44.340 ... clip_02.tmp.mp4        -> 173s
05:08:42  pass 2  -i clip_02.tmp.mp4 -vf crop,scale,crop,subtitles ... clip_02.mp4 ->  73s
05:09:55  rendered clip_02.mp4 in 246.5s
```

**70% of the clip went into producing a file that is deleted at the end of the
function** — and that pass used `settings.video.preset` raw, so it ignored the
`ultrafast` downgrade the log line directly above it had just announced. On a
1-vCPU host that pushed a 10-clip job beyond the interval at which the host is
restarted, so long jobs appeared to loop forever instead of converging.

The intermediate was only ever used for `probe()` (width/height, also available
from the source) and as the filtergraph input. Seeking the source directly is
frame-accurate (`-ss` before `-i` decodes and discards up to the exact frame) and
keeps the filtergraph on clip-relative timestamps, so crop windows, ASS cue
timings and silence cuts are interpreted identically. `tools/e2e_render_test.py`
asserts both the output duration and that the render makes exactly **one** encode
pass and writes no `.tmp.mp4`.

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
python3 tools/detect_backend_test.py   # provider dispatch, text-only payloads
python3 tools/llm_provider_test.py     # Groq wire contract, 429 backoff, pacing, key handling
python3 tools/caption_preset_test.py   # caption preset -> ASS style line + clip-relative timing
python3 tools/encode_profile_test.py   # encoder/preset/deadline degradation logic
bun tools/overall_status_test.ts       # job Done/Failed/overall-progress indicator logic
python3 -m app.doctor                  # hardware/engine/ffmpeg diagnostics
curl localhost:8000/api/health         # worker/warmup/LLM/system status
curl "localhost:8000/api/dev/verify-clips?jobId=<id>"   # ffprobe every rendered clip
python3 tools/smoke_test.py      # API + worker + honest failure paths
python3 tools/e2e_render_test.py # real ffmpeg: cut → crop → captions → thumbnail
python3 tools/batch_recheck.py <jobId> --detect  # ffprobe every clip + rerun detection
```

`tools/batch_recheck.py <jobId>` re-verifies a finished batch from its own
artifacts: it ffprobes every rendered clip and checks the **frame count**
(not just the rounded duration) against the recorded cut, confirms every
thumbnail exists, and re-sums the total against the job's `totalClipsSec`.
`--detect` additionally re-runs the detection stage on the batch's **real**
faster-whisper transcript through the configured provider, which is the
end-to-end proof for an LLM provider swap: it exercises chunking, the wire
contract and the word-index → real-timestamp resolution on production data.
Without `GROQ_API_KEY` it falls back to a local OpenAI-compatible stub and
prints a loud warning that the real api.groq.com round trip was **not**
tested — add the key and re-run to complete the check.

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
