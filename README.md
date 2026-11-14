# ClipForge

**Hosted video → viral clips.** Paste a link to a video you own (or attach the
file), and ClipForge finds the most viral moments and renders them as vertical
9:16 clips with animated, burned-in captions —processed on ClipForge's
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
   | `deepseek` (default) | **Fast** — the request overlaps local CPU work | Hosted API, OpenAI-compatible (DeepSeek platform). Billed against a **balance**, not a fixed daily token ceiling, so a long source cannot die half way through detection — and the cheapest of the hosted options (~$0.15/1M input, ~$0.60/1M output off-peak) | `DEEPSEEK_API_KEY` |
   | `nvidia` | **Fast** | Hosted alternative (NVIDIA NIM) on the same wire format. Hosts its own `deepseek-*` models, but on an **NVIDIA quota**, not a DeepSeek balance | `NVIDIA_API_KEY` |
   | `groq` (backup) | **Fast** | Hosted API, OpenAI-compatible. ~30 req/min and a **200k tokens-per-DAY** ceiling per model on the free tier; calls are paced below the request limit, but a spent daily quota can only be waited out | `GROQ_API_KEY` |
   | `gemini` | **Fast** | Hosted alternative, selectable without a redeploy | `GEMINI_API_KEY` |
   | `ollama` | **Slower** — it shares the same CPU as Whisper, so it runs sequentially after the transcript exists | **Free and fully private** — the transcript never leaves the machine | `ollama serve` + the model pulled, no API key |

   `DEEPSEEK_API_KEY` is the one to paste in the Keys tab. `config.toml`
   `[llm] fallback_provider = "groq"` makes a chunk that fails on the primary
   (spent quota, endpoint down) retry on Groq before it is skipped; blank
   disables that, and a missing PRIMARY key is always a loud config error
   rather than a silent downgrade. All providers share one code path — the
   JSON-repair ladder, pacing and the model chain are per-provider config, not
   per-provider code — and each provider's rate budget is tracked separately,
   so a DeepSeek call never waits out Groq's 30-rpm floor.

   **DeepSeek specifics.** DeepSeek's thinking mode is **on by default** at
   high effort, and while it is on it *ignores* `temperature` (plus the
   presence/frequency penalties) — the docs say such parameters "will not
   trigger an error but will also have no effect". It also returns the
   chain-of-thought in `reasoning_content`, billed as **output** tokens. So
   `config.toml` `[deepseek] thinking_mode = "disabled"` is what actually makes
   the `temperature = 0.0` determinism and the JSON-retry ladder mean anything;
   leave it on and every retry pays for reasoning that is then thrown away.
   Its documented ceiling is a **concurrency** limit (2500 for
   `deepseek-flash`, 500 for `deepseek-v4-pro`), not a tokens-per-minute one,
   so `tokens_per_min` is only a burst guard. Note that DeepSeek keys are `sk-…`
   and come from `platform.deepseek.com/api_keys` — they are *not* NVIDIA's
   `nvapi-…`, even though both providers happen to host a model called
   "DeepSeek".

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

   **Naming follows the provider, never a hardcoded one.** The switch to Groq
   left a few Gemini-shaped assumptions behind, all of which are now covered by
   `tools/llm_provider_test.py`:

   - `/api/jobs/{id}/redetect` checked `settings.gemini_api_key` and refused
     with *"add GEMINI_API_KEY"*. For anyone who had only ever set a Groq key
     it was simply broken. It now reuses the same `detection_status()`
     readiness the dashboard shows, so the message names the provider actually
     in use.
   - The pipeline's offline failure message overrode the exception's own
     provider-accurate text with a hardcoded Gemini string. It now surfaces
     the exception, which already says `NVIDIA_API_KEY missing…` or
     `ollama server unreachable…`.
   - `detectedSource` was written as the literal `"gemini"` no matter which
     backend ran, so a Groq-detected batch was filed as Gemini work. It now
     records `llm.provider(...)` and the card reads *"AI-detected via groq"*.
     This one was a **landmine**: the dashboard inferred success from
     `detectedSource === "gemini"`, so correcting the provenance without
     fixing the UI would have relabelled every successful job
     *"Detection failed"*.

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

Supported names: `DEEPSEEK_API_KEY` (canonical) or `DEEPSEEK_KEY`;
`NVIDIA_API_KEY` (canonical), plus `NVIDIA_KEY`,
`NVIDIA_NIM_API_KEY`, `NGC_API_KEY`; `GROQ_API_KEY` (canonical) or
`GROQ_KEY`; `GEMINI_API_KEY`
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

### Resume must not ship a stale render

Retrying a job reuses clip documents **by index** so finished renders
survive. That is only sound while the cut is unchanged. A re-detection may
well return a different window at the same index — clip 3 is simply no
longer the same three seconds — and if the old encode is kept, the job
reports success while serving the previous video under the new metadata.

This was not theoretical. The Groq re-run of `c57d7782c548` produced fresh
cuts; `create_clips` kept `renderState="done"`; all ten renders were skipped;
and `tools/batch_recheck.py` then found frame counts up to 2× off (clip 5
wanted 20.35s and measured 40.87s). Ten wrong videos, zero reported errors.

So reuse is now conditional on the cut actually matching:

- `LocalStore._cut_moved()` compares the stored window to the new candidate
  with 20 ms of slack (float noise from word-index rounding is not a move).
  Any real move resets `renderState` to `pending` and clears `videoPath`,
  `thumbPath`, `media*`, `cropBox`, `captionCues` and `silenceCuts`.
- `running` and `failed` are unfinished work, not results: both go back to
  `pending` and the stale `mediaError` is dropped.
- The render-skip in `task.py` additionally requires the recorded
  `captionPreset` to match the requested one — captions are burned into the
  pixels, not stored beside them, so a style change must re-encode.

`tools/resume_invalidation_test.py` (34 checks) locks all of this down,
including that an unchanged cut still resumes instantly and keeps its clip id,
so download links stay stable.

### The dashboard must not narrate work that isn't happening

Two rules, both learned the hard way on a host that recycles mid-job.

**A queued job looks queued.** On boot, `_boot()` re-queues jobs a restart
interrupted. It used to enqueue them while leaving `stage` at the stage they
died in and `progress` at its last value. The dashboard derives its entire
status from those two fields, so a job sitting in a queue rendered as
"Stage 3 of 4 — Detecting clips, 75%" — indistinguishable from real work, and
worse than an error because it never resolved. Stale `unitsDone`/`unitsTotal`
compounded it: "10 of 10 clips" beside a 0% bar. Every re-queue path (`_boot()`
and `retry()`) now sets `stage="queued"`, `progress=0`, a queue-worded detail
line, and clears the unit counters. The stage markers on disk — not this
field — decide where the runner restarts.

**Deleting a job actually stops it.** `DELETE /api/jobs/{id}` removed the
record and workspace but left the work running: the server kept downloading,
transcribing and spending LLM calls on a job nobody could see. Worse, the job
came *back* — the child cannot be interrupted mid-render, so it finished and
its `write_job`/`write_clip` calls recreated exactly the files the delete had
removed, leaving a "complete" job pointing at unlinked MP4s plus zombie clips
attached to no job. Now `worker.cancel()` drops the id from the queue or
SIGKILLs its process group *before* the record goes, and `delete_job()` leaves
a tombstone that `write_job()` honours, so a late write is a logged no-op.
`create_clips()` also stopped raising `KeyError: '_id'` when the job it was
writing metadata for had just been deleted.

> **A recovered job stops saying it failed.** Observed live: a job that hit
> `AI detection failed: Groq failed after retries: HTTP 429` was later resumed
> by the boot reconciler and spent its entire render still carrying that
> error. `_finish()` never cleared it either, so a job could finish as
> `stage: "complete"` while its API payload still reported a detection
> failure — a contradiction that outlives the job in every response. The
> invariant is now: **a job that is not `stage="failed"` carries no error.**
> `_progress()` clears it on each stage change, and `_finish()`, `_boot()` and
> `retry()` all clear it on the way to a fresh attempt.

### Export must not be able to OOM the server

`POST /api/export/zip` built the whole archive in an `io.BytesIO` and returned
`buf.getvalue()` — a second full copy. A batch is 10 vertical clips at ~80 MB,
so one "Export all" held **~800 MB resident**, and the 32-clip request ceiling
implied 2.5 GB. This host has 2.0 GiB usable with a Whisper model loaded, so a
read-only convenience feature could OOM-kill the API process and take every
running job with it.

The archive is now written to a temp file and streamed back via
`FileResponse`, cleaned up by a `BackgroundTask`. Peak memory is flat whatever
the batch size — `tools/export_memory_test.py` measures it, and a 2 MB archive
grows the heap by ~5 KB. `ZIP_STORED` is kept deliberately: H.264 is already
compressed, so deflating would burn CPU for no gain. Requests above
`MAX_EXPORT_BYTES` (1.5 GB) are refused with a 413 explaining the limit and
suggesting a smaller batch, rather than being attempted.

> **The stored transcript names the model that actually ran.** `hostlimits`
> downgrades a model that cannot fit the container, so `config.toml` saying
> `medium` on a 2 GiB host means a `small` model did the work. `transcribe()`
> records `loaded.model_name` — the resolved model — and logs the downgrade
> instead of labelling the transcript with the configured value, because RTF
> is a per-model number and a mislabelled transcript makes it meaningless.
> `python -m app.doctor` reports this as `whisper model fits host`.

### Nothing hangs for 20 minutes, and one failure retries itself

A job sat at "Rendering clip 10 of 10" for 20+ minutes with nothing
happening, and another showed "Queued — waiting for a worker" at 0% behind a
single worker. Four separate causes, all fixed:

**A hung ffmpeg used to hold the only worker for 90 minutes.** The CPU render
deadline was a flat `90 * 60`, sized when a wrong shared GPU deadline was
causing "the render just stops" and nobody re-examined it. It is now
`render_deadline_sec(clip_seconds)` — a 6-minute floor, a per-second-of-output
allowance (6× realtime, measured at ~1× on this host), and a 30-minute
ceiling, so a typical 50s clip gets **11 minutes** and nothing gets an hour.
The stall watchdog also dropped from 20 minutes to 10.

**The real failure was an external kill, and it did not retry.**
`ffmpeg exited -9 (SIGKILL) — this was an EXTERNAL kill` is what this host
actually produces when the container recycles mid-render. That error text was
not in the retryable set, so the one failure most worth retrying silently did
not. Retryable failures now get **one automatic requeue** before the user is
told, and the set deliberately excludes bad URLs, missing keys and detection
errors — those would just spend quota to reach the same error. The retry is
cheap: the pipeline resumes from the last **completed** clip, so the download,
the transcript and the paid detection call are all reused from disk.

**ffmpeg was spawning 64 threads on a 1-CPU container.** ffmpeg sizes its
thread pools from `nproc`, which inside a cgroup reports the *host's* cores —
64 here, while the container is allowed 1. Every render was spawning ~64
frame and ~64 slice threads to share one CPU. The encoder is now told the
container's real allowance via `-threads/-filter_threads`, and on hardware
encoders too. Measured on a real 1080×1920 clip: **10s → 4s (2.5× faster) for
byte-identical output** (SSIM 0.9999). Less time per clip also means less
chance of a render being caught by a container recycle.

**A queued job claimed "~any moment" with a queue in front of it.** A queued
job has no ETA of its own, and `fmtEta` turns a missing number into "any
moment". The worker now computes `queueEtaSec` — the running job's remaining
time plus the jobs ahead — and the card says `starts in ~4 min`, falling back
to "any moment" only when there is genuinely nothing ahead.

Resume from a host restart was already working and is confirmed on real data:
the `zg50i-UmRwA` batch (`fd961148baa5`) was killed mid-render and clips 09
and 10 are on disk, frame-accurate against their recorded cuts.

> Queue depth is 1 worker on purpose. `hostlimits` narrows the pool because
> each concurrent job loads its own ~660 MB Whisper model into 2.0 GiB of RAM;
> a second worker would OOM rather than double throughput. So the fix is an
> honest wait estimate, not more workers.

### `/api/health` always answers, and never depends on a third party

The dashboard showed a red **"API unreachable: HTTP 400"** banner directly
above a fully populated status card — real CPU, RAM, disk, worker and provider
numbers from the previous poll, with one transient failure claiming the API
was gone. Two independent defects:

**One failing sub-component could fail the whole endpoint.** `health()` called
`system_stats()`, `disk_stats()`, the RTF store, whisper status and the key
resolvers inline. Any one raising took down the entire payload. Every block is
now built through `_block()`: a failure degrades that field to
`{"error": ...}`, is listed in a new `degraded` array, sets `ok: false`, and
the endpoint still returns 200 with everything else intact.

**It was coupled to the AI provider.** `health()` embedded
`detection_status()`, which for the ollama backend calls `ollama_status()` — an
HTTP request with a 5s timeout. A third-party hiccup could stall our own
health endpoint. `detection_status(probe=False)` now answers from
configuration and key presence only and makes **no network call**; health uses
it. **Groq is not the cause of your 400** — its readiness was always a local
env lookup, never a vendor round trip, and that is now explicit and tested.

The frontend was also lying by omission: `useHealth` treated any failed poll
as fatal, so one blip blanked a card full of real readings. It now separates
`error` (never responded) from `stale` (a poll failed, last good reading still
shown), and the card renders a quiet amber note instead of a red banner.

On the status code: 400 was never valid for this route — it is a GET with no
parameters to reject. A health check returns **200 whenever the process
answered at all**, with `ok`/`degraded` describing what is wrong. Returning
503 for "the worker is busy" would teach clients to ignore the difference
between "my server is broken" and "a job is running".

### Pause, resume, and telling "stuck" from "just slow"

Three manual controls, because a spinner that never resolves is indistinguishable
from work that is genuinely happening.

**Pause is cooperative, so it can never corrupt a file.** The real work runs in
a per-job child process, and a child cannot be interrupted part-way through an
ffmpeg write without risking exactly the corruption a pause is supposed to
avoid. So `POST /api/jobs/{id}/pause` sets a flag the **pipeline** polls
between units of work — transcript chunks, detection chunks, rendered clips,
stage boundaries — and the job then stops itself and records a clean `paused`
state. Nothing is ever killed mid-write. The API distinguishes `paused` (it has
stopped) from `pausing` (still finishing the current unit), because claiming
"paused" instantly while a 40-second encode is still running would be a lie.

**Resume really does not redo paid work — and detection had a genuine gap
here.** `detect_clips` looped over chunks calling the LLM and kept the results
only in memory, so *any* resume re-ran and re-billed every chunk. A 7-chunk
transcript caught by a host restart at chunk 5 paid for all seven again. Each
chunk's resolved candidates now persist under `detect/chunks/` *before* progress
is reported, so a kill in the next instant still keeps that chunk. Measured on
the real `O0yDqRwmMSE` transcript (974 words, 930s, 7 chunks):

```
PAUSE  -> paused after chunk 3/7 — 1 candidate(s) kept
  Groq calls so far: 3
  chunks cached    : chunk_000.json, chunk_001.json, chunk_002.json
RESUME -> completed
  Groq calls on resume: 4  (not 7)
  saved               : 3 call(s) not re-billed
```

Transcription already had per-chunk caching and now honours pause at the same
boundaries; the render loop already skipped finished clips and now checkpoints
between them. `paused` is a real terminal state — never auto-retried, never
reported as running or failed, and never auto-resumed on boot.

Fixing this also surfaced a latent bug: `StageWorkspace.write_json` did not
create parent directories, so a nested name like `chunks/chunk_000.json` failed
with `FileNotFoundError` on the very first cached chunk.

**Liveness is now visible and self-updating.** Each active card ticks
"updated 12s ago" every second from `lastProgressAt`, so a frozen bar is
obvious without opening the logs. Two *separate* thresholds, deliberately far
apart: **75s** shows a quiet amber "No recent activity" note — explicitly not
an error, because one long encode legitimately emits no progress lines — while
**10 minutes** is where the server-side watchdog has actually given up. The old
UI waited 20 minutes, twice as long as the server would, so a job already
killed and auto-retried still showed a bare spinner.

**"Check now"** forces an immediate poll of that one job (`GET /api/jobs/{id}`)
and reports what it saw, instead of making you reload the page or wait out the
2s interval.

### The bitrate is adaptive, and it was measured

A flat CRF is a *quality* setting, not a size policy. On a CPU-only host the
preset is lowered to `ultrafast`, which drops the lookahead and CABAC work
x264 normally does, so the same CRF produces a much larger file than the
config implies. Measured on a real batch: **crf 20 at 1080×1920/60 came out at
13.2 Mbps — 81 MB for a 50-second clip**, roughly 400 MB per batch on a 3 GB
disk.

`EncoderChoice.for_output(width, height, fps)` now sizes each render to the
frame it is actually producing:

- The VBR ceiling comes from **bits-per-pixel-per-frame**, so the policy is
  geometry-driven rather than a magic constant. 1080×1920@60 → 7.5 Mbps;
  the same frame at 30fps → 3.7 Mbps; 4K60 → 14 Mbps (clamped). A 30fps
  source used to be sized as if it were 60 and could spend double its budget.
- The CRF adapts to the preset that actually ran, because the presets are not
  equally efficient — `ultrafast` gets +3, a capable preset keeps or lowers it.
- The ceiling is carried by **all four** encoder paths (libx264, NVENC, QSV,
  VAAPI), or the saving would silently evaporate on a host with a GPU.

Measured on stress content at the true output geometry:

| config | bitrate | size | SSIM vs source |
|---|---|---|---|
| before: ultrafast crf20, uncapped | 71.8 Mbps | 95.8 MB | 0.906 |
| after: ultrafast crf23 + 7.5M cap | **6.2 Mbps** | **8.2 MB** | 0.895 |

**-91% size for -0.011 SSIM** — and note the old setting was not actually
high quality either. On the real 13.2 Mbps batch the cap projects to roughly
81 MB → 46 MB per clip. Easy content stays well under the ceiling, so this
costs nothing on simple footage. `tools/encode_profile_test.py` asserts the
geometry math, the preset-adaptive CRF, and that every encoder path carries
the cap.

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

The supervisor respawns a dead backend on the first proxied `/api` request,
and the reviving request **waits** for the spawn so the first page load after a
crash succeeds instead of 500ing. Two properties are deliberate:

- **A failed first boot is recoverable.** The guard used to require
  `state.sawBackend` — "only revive a backend this process already saw alive".
  Since that flag is only set once `/api/health` has answered, a backend that
  crashed on boot (a bad import, a missing dependency) never set it, the
  on-demand branch never fired, and every request 500'd until the whole web
  process was restarted. A crash loop and a failed first boot are the same
  failure and now recover the same way.
- **Crash loops are throttled.** Dropping the condition without a backoff
  would have made a backend that dies on boot respawn on *every* request. So
  failed boots back off (1s → 30s) and give up after 12 attempts with a
  message telling the operator to reload, instead of forking.

`bun tools/backend_respawn_test.ts` guards all of it.

> **The backend is spawned lazily and only by the proxy.** Probing
> `127.0.0.1:8000` directly reports "down" for a backend that is merely
> asleep. `tools/live_watch.py` deliberately goes through the Vite proxy on
> :5173, exactly as a browser does, so it both tests the real path and
> triggers the revival.

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
| **Engine** | Reuses the `[llm]` provider (groq, gemini or ollama) — no new dependency, no extra key, no model download. |
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

### Surviving a restarted server

The sandbox periodically restarts the whole web process, which SIGKILLs the
pipeline child mid-job. That is an external event, not a pipeline fault, and it
used to surface as *"The worker stopped while this job was in progress and
never reported back"* — a message that named no cause, burned a resume-budget
slot, and left a job whose work was 99% cached looking permanently dead.

| | |
|---|---|
| An orphaned stage is **re-queued** | Reconciliation resumes the job from its stage markers instead of failing it. |
| The message names the real cause | *"the server restarted during 'Rendering 9:16' — 5 restart(s) so far"*, derived from counts only — never from its own previous output. |
| Evidence survives | The interrupt reason, count and last signal are preserved for the next diagnosis. |
| The budget counts progress, not restarts | A restart that advanced the job costs nothing. Only `STALLED_RESUME_LIMIT` consecutive restarts that made **no** progress stop the loop. |
| Nothing escapes the process | `sys.excepthook`, `threading.excepthook`, `sys.unraisablehook` and a `SIGTERM` handler all write a real failure onto the job first. |
| One clip costs one clip | A failed render retries once with safer settings, is verified (exists, non-empty, probes, right duration), then is **skipped and logged** — the batch still finishes. |
| Nothing stacks | At most 2 ffmpeg children host-wide (1 on a single-vCPU or low-RAM host), reentrant per thread so a probe inside a render never deadlocks. |
| The disk is checked first | A job that cannot fit fails up front with the megabytes it needs and has, not at clip 9 of 10. |
| Scratch is cleaned | The 16 kHz WAVs are deleted when the job completes. |

`STALL_TIMEOUT_SEC` is 3 minutes, and the heartbeat spans the **whole**
pipeline — every stage beats every 5s, so the watchdog distinguishes "slow"
from "stuck" instead of killing working transcodes.

Run `python3 tools/reliability_test.py` to prove all of it end to end on a real
short video, printing PASS/FAIL for every stage.

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

## Post to social (optional, additive)

Every finished clip grows a **Post to...** button *next to* the existing MP4
button. The download, SRT and `captions.txt` paths are **unchanged** — nothing in
this section can block or alter them. Tokens live in `data/accounts/<platform>.json`
at mode `0600`; the browser only ever sees whether an account is connected and
what it is called.

### Which platforms you can actually use today

They are **not** symmetric, and the difference is the whole point of the table.

| Platform | Gate | What you have to do | Result |
|---|---|---|---|
| **YouTube Shorts** | none | Enable YouTube Data API v3 in a Google Cloud project, add an OAuth client, paste `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` | **Works.** No App Review. Default quota is 10,000 units/day and a Short costs 1,600 — about 6/day, raisable self-serve |
| **X** | payment | Create a developer app, enable billing, paste `X_CLIENT_ID` / `X_CLIENT_SECRET` | Works once billing is on. X moved to pay-per-use in 2025; free-tier posting is no longer reliably available. This is a payment step, not a queue |
| **TikTok** | audit | Create the app, paste `TIKTOK_CLIENT_KEY` / `TIKTOK_CLIENT_SECRET` | **Works, but posts land PRIVATE.** Unaudited clients are forced to `SELF_ONLY` and capped at 5 users per rolling 24h. Add `TIKTOK_APP_AUDITED=true` after passing the Direct Post audit |
| **Instagram Reels** | app review | Meta app + `instagram_content_publish` (Advanced Access) | **Not usable yet.** Needs App Review: a screencast plus a business/creator IG account, typically days to weeks. Until then it falls back to download |

The approval reality lives in one readable block in `app/social/__init__.py`
(`APPROVAL_NOTES`) and is served verbatim to the UI, so the settings panel and
the docs cannot drift apart.

### What happens when a platform can't post

It falls back rather than failing: the "Post to..." menu offers **Share sheet /
download instead**, which is the behaviour that existed before posting was added.
The connect endpoint answers **409** for a platform with no credentials and
**401** for a revoked token, so the toast can say *"Failed — reconnect Instagram
in Settings"* instead of a bare status code. A clip with no rendered MP4 is
refused with **409** before any network call is attempted.

### Environment variables

`YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `X_CLIENT_ID`,
`X_CLIENT_SECRET`, `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `META_APP_ID`,
`META_APP_SECRET`, and `CLIPFORGE_PUBLIC_URL` (must match the redirect URI
registered with each provider). Every one is optional — a missing credential only
disables that one platform.

### Post captions

Each clip carries **one line of caption plus a 5–7 tag block**: 2–3 discovery
tags (`#fyp`, `#foryou`, `#viral`) and 2–4 content tags mined from that clip's
own transcript. "Copy caption" copies exactly that — the title and the 300-word
transcript are deliberately gone, because pasting them under a clip is what made
the old copy unusable. The full text is still one SRT download away.

The rules are enforced in **both** `app/caption_format.py` and
`src/lib/clipforge.ts` rather than trusted to the model, so a sloppy reply still
produces a well-formed block, and clips rendered before this change copy in the
new format.

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
python3 tools/deepseek_provider_test.py # DeepSeek wire contract, thinking-mode field, failover
python3 tools/nvidia_provider_test.py  # NIM as a selectable provider, pacing isolation
python3 tools/usage_isolation_test.py  # daily counters survive restarts, per-provider budgets
python3 tools/llm_provider_test.py     # Groq wire contract, 429 backoff, pacing, key handling
python3 tools/caption_preset_test.py   # caption preset -> ASS style line + clip-relative timing
python3 tools/caption_social_test.py   # post-caption rules + social readiness/OAuth/error mapping
python3 tools/encode_profile_test.py   # encoder/preset/deadline + adaptive bitrate logic
python3 tools/live_watch.py            # verify the RUNNING site: waits, then checks every endpoint
python3 tools/health_resilience_test.py # health never fails on a sub-block, never calls a vendor
python3 tools/pause_resume_test.py        # pause stops cleanly; resume re-bills only what is left
bun tools/backend_respawn_test.ts       # the backend supervisor recovers from a failed boot
bun tools/overall_status_test.ts       # job Done/Failed/overall-progress indicator logic
python3 -m app.doctor                  # hardware/engine/ffmpeg diagnostics
curl localhost:8000/api/health         # worker/warmup/LLM/system status
curl "localhost:8000/api/dev/verify-clips?jobId=<id>"   # ffprobe every rendered clip
python3 tools/smoke_test.py      # API + worker + honest failure paths
python3 tools/e2e_render_test.py # real ffmpeg: cut → crop → captions → thumbnail
python3 tools/batch_recheck.py <jobId> --detect  # ffprobe every clip + rerun detection
python3 tools/resume_invalidation_test.py  # a moved cut must not reuse the old render
python3 tools/queue_truthfulness_test.py  # a queued job must look queued
python3 tools/delete_cancels_test.py       # deleting a job must actually stop it
python3 tools/export_memory_test.py        # zip export must not OOM the server
python3 tools/playback_test.py     # rendered MP4s are browser-playable (faststart, Range, player)
python3 tools/preview_layout_test.py  # the preview fills its 9:16 card and actually plays
python3 tools/boot_guard_test.py      # the boot guard never reload-storms a slow load
python3 tools/playback_e2e.py      # the same, over real HTTP with curl -I and curl -r
python3 tools/subtitle_langs_test.py  # multi-language caption tracks: timing, wrapping, SRT/VTT
python3 tools/restore_test.py      # a deleted job is restorable; retention can be disabled
python3 tools/render_loop_test.py  # an empty balance must not look like a hung render
python3 tools/share_intents_test.py   # sharing never silently downloads
python3 tools/share_instant_test.py   # the share copy is small/faststart; the tap path awaits nothing
python3 tools/share_endpoint_test.py  # /share over real HTTP: immutable, ETag, Range, 206, concurrency
python3 tools/heartbeat_test.py      # long steps prove liveness without faking progress
python3 tools/reliability_test.py    # END TO END on a real short video: PASS/FAIL per stage
python3 tools/detect_concurrency_test.py  # concurrent detection == sequential, exactly
python3 tools/perf_bench.py verify <jobId>  # correctness gate: framing, sync, audio, count
python3 tools/perf_bench.py render <jobId>  # sequential vs concurrent ffmpeg, measured
python3 tools/perf_bench.py report           # before/after table from saved runs
python3 tools/remux_faststart.py --apply   # move moov to the front of already-rendered clips
```

`tools/batch_recheck.py <jobId>` re-verifies a finished batch from its own
artifacts: it ffprobes every rendered clip and checks the **frame count**
(not just the rounded duration) against the recorded cut, confirms every
thumbnail exists, and re-sums the total against the job's `totalClipsSec`.
`--detect` additionally re-runs the detection stage on the batch's **real**
faster-whisper transcript through the configured provider, which is the
end-to-end proof for an LLM provider swap: it exercises chunking, the wire
contract and the word-index → real-timestamp resolution on production data.
Without a provider key it falls back to a local OpenAI-compatible stub and
prints a loud warning that the real round trip was **not**
tested — add the key and re-run to complete the check. On the reference
batch it reports `✓ this was a REAL api.groq.com round trip`.

`tools/resume_invalidation_test.py` guards the resume path described in
[Resume must not ship a stale render](#resume-must-not-ship-a-stale-render):
an unchanged cut still resumes and keeps its clip id, while a cut that moved
by more than 20 ms invalidates the render and clears every stale media field.

`tools/queue_truthfulness_test.py` and `tools/delete_cancels_test.py` guard the
two honesty rules below — a queued job must look queued, and deleting a job
must actually stop it. `tools/export_memory_test.py` guards the export path
against buffering a whole batch in RAM.

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
