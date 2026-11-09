# ClipForge → standalone **web app** (relocation brief)

Copy this whole file into the new project. It is written as instructions to whoever/whatever
builds it. Read §0 before anything else — it is the difference between a web app and a local
script with a web page glued on.

---

## §0 Non-negotiables: this must ship as a WEB APP

The target is a **multi-user, hosted web application reachable at an https URL**, not a local
tool, not a desktop app, and not a "run it on your machine" repo. If a requirement below can
only be satisfied by something running on the reader's own computer, it is wrong.

Hard requirements:

1. **Server-side everything.** Download, transcode, ASR, LLM calls, caption burn-in and clip
   rendering all happen on the server. A phone browser is a thin client: it uploads a URL or a
   file, polls job state, plays and downloads the finished MP4s.
2. **No key ever reaches the browser.** ASR/LLM/storage credentials live only in server env.
   The client talks to *your* API; it never calls a vendor directly.
3. **Accounts.** Every job belongs to a user. `/api/jobs` returns only the caller's jobs. An
   unauthenticated request gets 401, an authenticated request for someone else's id gets 404
   (never 403 that confirms existence). This repo currently has **no auth and no owner field —
   that is the single biggest thing to add**, not a later TODO.
4. **Jobs outlive the browser tab.** Closing the phone does not stop a job. Progress is
   re-readable from the server on any device, after any reload.
5. **A real background worker.** The API process must not run ffmpeg or Whisper. A separate
   worker process owns a queue, resumes interrupted work, and survives an API restart.
6. **Deletion and retention are server features.** A user can delete a batch and its disk
   usage goes away; an unclaimed batch is swept after N hours.
7. **Deployable with one command.** Container(s) + reverse proxy + TLS, with `/api/health`
   wired to the platform's health check.

Everything below assumes those hold.

---

## §1 What the product does

Paste a YouTube (or direct) video URL, or upload a file. The server picks the strongest
self-contained moments, re-frames each to 9:16 with burned-in animated captions and emoji,
and hands back N downloadable MP4s (plus `.srt`/`.vtt`). One source typically yields 5–15
clips. A batch of 15 clips from a 21-minute source takes ~45–90 min on a 1 vCPU box.

## §2 Stack

- **API**: Python 3.11, FastAPI + uvicorn, pydantic models, JSON-on-disk job store (swap for
  Postgres when going multi-user; see §12).
- **Worker**: separate Python process, single-flight queue, spawns the pipeline as a child per
  job (`nice -n 15`), talks to the API over a unix socket or a DB queue.
- **Pipeline**: `ffmpeg`/`ffprobe` binaries, `faster-whisper`, `yt-dlp` (Python API, not CLI).
- **Web**: React 19 + Vite + TypeScript + Tailwind + shadcn/ui, same-origin `/api` calls.
- **Storage**: local volume by default; S3/R2 adapter when `CLIPFORGE_S3_*` is set.

Copy from this repo: `app/` (pipeline, worker, store), `config.toml`, `LOCKED_SPEC.md`.
Rewrite: the store (add `ownerId`), the API (auth middleware), the frontend (login screens).

## §3 Data model

`Job`: `_id`, `ownerId`, `sourceType` (`url|upload`), `source`, `url`, `title`, `createdAt`,
`durationSec`, `stage`, `status` (`queued|running|paused|failed|complete`), `progress`,
`progressDetail`, `clipCount`, `error`, `failedStage`, `logs` (bounded!), `captionSettings`,
`options {minClipSec, maxClipSec, maxClips}`, `language`.

`Clip`: `_id`, `jobId`, `order`, `start`, `end`, `score`, `title`, `transcript`,
`captionCues[{start,end,text,uncertain}]`, `emojiTrack[{i,emoji,start,end,byHandle}]`,
`aspect`, `videoPath`, `thumbPath`, `mediaBytes/Width/Height/DurationSec`, `renderState`,
`captionCoverage` (speech-vs-caption audit), `shareId`.

Derive `status` in exactly one place from `stage` + progress + liveness; two places disagree
and the UI shows "Failed" beside "61% overall, running".

## §4 Pipeline stages (each resumable from its own artifacts)

| stage | input | output | notes |
|---|---|---|---|
| `fetch` | URL | `<job>/source.mp4` | yt-dlp; reuse existing file, never re-download |
| `transcribe` | source | `transcribe/words.json`, `speech16k.wav` | 16 kHz mono WAV; cache per (file, model) |
| `detect` | transcript | `detect/clips.json` | LLM picks moments + scores |
| `reframe` | clips | `reframe/track.json` | face/centre crop path per clip |
| `captions` | transcript + clips | `captions/clip_NNN.json` | chunking, language, emoji |
| `render` | everything | `outputs/<job>/clip_NNN.mp4` | ffmpeg, captions burned in |

Resume per **stage**, and ideally per **clip** inside `render` — a restart must not redo 11
finished clips. Reuse `fetch/meta.json` if the source file still exists.

## §5 Models

- **ASR**: `faster-whisper`, `int8`. Ladder by RAM: `medium` ≥ 8 GB, `small` ≥ 3 GB, `base`
  below. Never advertise a model the host refuses to load; show the resolved one.
  `distil-*` models are English-only and useless for non-English sources — exclude them.
- **Highlight detection**: OpenAI-compatible providers behind one interface with failover:
  DeepSeek (cheap default), NVIDIA NIM, Groq (fast, strict TPM), Gemini, Ollama (local, slow
  but private). Out-of-credit must fail over, not fail the job.
- **Emoji per caption chunk**: keyword fallback when the model declines; a hand-picked emoji
  is recorded `byHandle: true`.

## §6 Captions

The caption look, size, colours, position, chunking and line splitting are **locked** — see
`LOCKED_SPEC.md`. Do not "improve" them during relocation. `app/workerd.py` and the caption
renderer carry a do-not-modify header for the same reason.

## §7 yt-dlp and cookies.txt (the fragile part)

Options that matter: `format` preferring `avc1` at ≤720p with progressive fallback,
`merge_output_format=mp4`, `concurrent_fragment_downloads=8`, `noplaylist=True`,
`js_runtimes={node:{}}` + `remote_components={"ejs:github"}` (YouTube's n/sig challenge needs
a JS runtime or you get images instead of video).

Cookie jar, with the plaster this repo now ships:

1. **Health before use.** Parse the jar once per download: present? non-empty? actually a
   Netscape file (not JSON)? any non-session cookies? how many days old? Warn at >**30 days**.
   Expose it on `/api/health` so the dashboard shows `cookies.txt 41d old · refresh it` *before*
   a URL dies.
2. **Retry ladder when the site asks for a sign-in**: alternate player clients (`ios`, `mweb`,
   `web_safari`) **with** the jar → same clients **without** it → only then a message naming
   the jar and its age. A stale jar fails *harder* than no jar, so never treat "a jar exists"
   as "cookies work".
3. **Match every YouTube URL form.** `youtu.be/<id>`, `m.`, `music.`, `youtube-nocookie.com`
   are all YouTube. A substring check for `"youtube"` silently skips the retry for short links
   — the form most people paste.
4. **Auto-update once.** On extraction failure, `pip install -U yt-dlp`, reload, retry once.
   Never loop it.
5. **Never** loop a failed fetch. Classify: auth/bot wall → terminal with an actionable
   message; network → one retry; everything else → fail the job.

## §8 Environment (names only; values come from the host's key manager)

ASR: `CLIPFORGE_WHISPER_MODEL`, `CLIPFORGE_WHISPER_COMPUTE`, `CLIPFORGE_ASR_ENGINE`.
LLM: `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `NVIDIA_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_HOST`.
Fetch: `YT_DLP_COOKIES_FILE`, `YT_DLP_COOKIES_FROM_BROWSER`.
Storage: `CLIPFORGE_S3_ENDPOINT`, `CLIPFORGE_S3_BUCKET`, `CLIPFORGE_S3_ACCESS_KEY`,
`CLIPFORGE_S3_SECRET_KEY`, `CLIPFORGE_S3_REGION`, `CLIPFORGE_S3_PREFIX`, `CLIPFORGE_KEEP_LOCAL`.
Hosting: `CLIPFORGE_STORAGE_DIR`, `CLIPFORGE_MAX_SOURCE_MIN`, `CLIPFORGE_RETENTION_HOURS`,
`CLIPFORGE_CONCURRENCY`, `CLIPFORGE_STALL_TIMEOUT_SEC`.
Web app only (new): `AUTH_SECRET`, `AUTH_PROVIDER_*`, `DATABASE_URL`, `PUBLIC_BASE_URL`.

## §9 API surface (same-origin only)

`POST /api/jobs` (create), `POST /api/jobs/upload`, `GET /api/jobs?limit&before|all`,
`GET /api/jobs/{id}`, `DELETE /api/jobs/{id}` **and** `POST /api/jobs/{id}/delete`,
`POST /api/jobs/bulk-delete`, `POST /api/jobs/{id}/{retry,redetect,rerender,retranscribe,pause,resume}`,
`GET /api/jobs/{id}/logs`, `GET /api/clips?jobId`, `GET /api/clips/{id}/{video,thumb,download,srt,share}`,
`POST /api/clips/{id}/retry-render`, `POST /api/export/zip`, `GET /api/health`, `GET /api/doctor`.

Two notes learned the hard way: keep **both** delete verbs (some proxies dislike DELETE), and
have the client treat a 404 on delete as success — the retry after a lost response gets one.

## §10 Worker rules

- One daemon, one job at a time on a small host; concurrency comes from the host, not hope.
- `enqueue()` refuses a job that already has a live child; boot skips jobs with live pids
  instead of resetting them to `queued`. Violating this requeues a running job forever and the
  card reads "Queued… 0%" while the disk fills.
- On terminal states, clear `queuePosition`, `stagePlan`, ETAs. Never show a running-only
  field on a dead job.
- Cap the job log array (this repo: last 600 lines). It is rewritten on every append.
- **Disk preflight**: refuse to start with < **400 MB** free, terminally, with the number in
  the message. A render needs source + WAV + clips + temps; starting at 100 MB free is a
  guaranteed mid-render ENOSPC.
- Stall detection: no progress for `CLIPFORGE_STALL_TIMEOUT_SEC` → mark failed with the reason.

## §11 Sizing and disk math

Reference host that works: 2 vCPU / 4 GB RAM / 40 GB disk for one concurrent job.
Per 21-minute source: source 170 MB, `speech16k.wav` 41 MB, 15 clips ≈ 60 MB, temps ≈ 100 MB.
Budget ~2.5× the source size in flight, plus the ASR model (small ≈ 480 MB, medium ≈ 1.5 GB).
A 4 GB root with a 1.5 GB model cache is a full disk waiting for a job — put media on a
separate volume (`CLIPFORGE_STORAGE_DIR`).

## §12 Auth & multi-tenancy (new work, not optional)

- Add `ownerId` to every job/clip; every query filters on it.
- Session cookie (httpOnly, SameSite=Lax) or JWT; `PUBLIC_BASE_URL`-scoped.
- Upload limits per user (`CLIPFORGE_MAX_SOURCE_MIN`), plus a per-user daily job cap.
- Share links are the only public read path: unguessable id, `no-store`, separate `/s/<id>`.
- Deleting a user's data must delete their media too, not just rows.

## §13 Deployment

- `docker-compose`: `api`, `worker`, `web` (static build behind nginx) or one image serving
  both; a volume mounted at `CLIPFORGE_STORAGE_DIR`; `/api/health` as the health check.
- Reverse proxy with generous timeouts on media routes (a 60 s default kills a download).
- Restart policy `unless-stopped`; the worker must be able to boot into an existing queue.
- Log to stdout; keep `data/logs/` capped.
- TLS everywhere; `PUBLIC_BASE_URL` must match the cookie domain.

## §14 Ops checklist

`/api/health` reports: API ok, worker online + queue depth, resolved ASR model, provider keys
present, disk free, cookie jar age, ffmpeg/yt-dlp versions. `/api/doctor` turns each of those
into a check with a fix line.

## §15 Pitfalls (all observed, all cheap to avoid)

1. A proxy in front of the stack answers with **HTML error pages**; never render a response
   body directly in the UI — detect HTML and say "the proxy answered 5xx, retry".
2. Retry idempotent mutations (delete) on 5xx/HTML/network, not on 4xx.
3. `youtu.be` is YouTube (§7.3).
4. Requeue-on-boot without a liveness check = infinite duplicate work (§10).
5. Unbounded job logs inflate every poll and every write.
6. A paged list that merges pages must be told about deletions by the client, or deleted
   batches reappear on the next poll and the Delete button "does not work".
7. Whisper on 1 vCPU is real-time-ish at `small int8` (RTF ≈ 0.44); quote ETAs from measured
   RTF, not optimism.
8. Groq's free TPM budget makes 9k-token chunks wait ~30 s; chunk the transcript.
9. Emoji/caption render changes invalidate the render cache — hash the resolved caption style.
10. Share-link transcodes on a weak host can exceed 180 s; make them best-effort, never block
    the clip download.

## §16 Done means

- A new user signs up, pastes a URL on a phone, closes the browser, comes back and downloads
  the clips.
- A second user cannot see or fetch the first user's batch by id.
- Killing the worker mid-job resumes it without re-downloading the source and without
  duplicating queue entries.
- Deleting a batch removes its rows, media and disk usage, and the UI shows it gone.
- An expired `cookies.txt` is visible on the status card and one URL still gets through before
  it is replaced.
- `docker compose up` on a clean host yields a working https deployment with a green health
  check.
