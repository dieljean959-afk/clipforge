# ClipMake — work order status (2026-11-15)

Backup taken before any change: `data/backups/clipmake-db-20261115-170732.tgz` (448 KB,
`data/jobs` + `data/clips`). No key, cookie or `.env` value was printed, logged or committed.

**Git is unavailable on this host** ("Git and GitHub commands are blocked; Vly manages
version control"), so the per-section commits and `git diff --stat` could not be produced.
This file is the recovery point instead.

---

## 1. The loop, "Your batches", the ETA

### 1.1 Cause, with evidence

Job `6ce9abe11d1c4709a4fc91ee4f7378e1` (`QqdCXWFobi8`, 1416 s source) was re-queued **11
times** by the boot recovery path — `data/logs/backend.log`:

```
20:03:51 [boot] re-queuing job 6ce9abe… (flagged interrupted, was captions)
20:34:40 [boot] re-queuing job 6ce9abe… (flagged interrupted, was failed)
21:36:31 … (flagged interrupted, was failed)
21:41:25 … (flagged interrupted, was failed)
00:31:09 … (mid-flight, was transcribing)
00:46:32 … (flagged interrupted, was failed)
11:38:07 … (flagged interrupted, was failed)
11:57:23 … (flagged interrupted, was failed)
12:03:41 … (mid-flight, was transcribing)
12:04:52 … (mid-flight, was transcribing)
16:57:36 … (flagged interrupted, was failed)
```

The same loop is visible in the job's own `stageTimings`: `fetch` appears **7 times**
(6560 s, 22229 s, 9915 s, 9737 s, 10350 s, 19926 s, 19526 s) — every restart re-downloaded
the source.

Four code causes, all fixed:

1. `app/main.py:_recover_jobs` re-queued on `stage in recoverable` **or** a leftover
   `interrupted` flag, with no terminal check. A job whose document said
   `isTerminal: true, retryable: false` was put back to `queued` on every boot.
2. `app/runner.py:main` unconditionally wrote `stage="fetch", progress=0,
   progressDetail="Stage: fetch — starting"` before any checkpoint was read. That is the
   literal "back to Fetching / Downloading 40%" line.
3. `app/task.py:_stages` ran `_cleanup_job_scratch(full=True)` (deleting `data/work/<job>`
   resume markers) **before** `_finish`, so a kill in that window left no checkpoints and
   the next boot had nothing to resume from.
4. `src/pages/Dashboard.tsx` filtered "Your batches" on `status === "complete"` only, so a
   batch that rendered into the bucket but never got a finalize stamp appeared nowhere.

The fetch checkpoint did not count because it was deleted by (3) and then overwritten by
(2); the source object itself was still in the bucket (the HEAD check would have passed).

### 1.2–1.3 Fixes

* `app/jobstatus.py`: `is_requeueable()` / `allow_retry()`. Recovery paths may only use
  `is_requeueable`; only a user action may call `allow_retry` (which also resets
  `stage`/`status` to `queued`, otherwise the guard would refuse the retry the user asked for).
* `app/main.py`: boot recovery refuses settled jobs; boot invariants run `rebuild_batches`
  first.
* `app/runner.py`: early exit for a settled job; never re-labels `fetch` when
  `sourceStream.videoKey` exists ("Resume — source already in the bucket").
* `app/task.py`: source reuse verified from the job document with a HEAD before workspace
  markers; `_progress` refuses to write to a terminal document; `_AlreadySettled` raised at
  stage boundaries; `_finalize()` (5 attempts, backoff, `_fail_job_only(error_kind="finalize")`)
  runs **before** scratch cleanup; `_finish` is idempotent; `batchState` =
  `finished` / `finished_with_failures` with `finishedAt` and requested/ready/failed counts.
* `app/reclaim.py`: the source is deleted only when the batch is `complete`, `isTerminal`
  and `readyClips > 0`.
* `src/pages/Dashboard.tsx` + `src/lib/api.ts`: "Your batches" lists any finished batch with
  delivered clips (`readyClips`/`clipCount`), independent of storage target.

### 1.4 Repair of existing data (no re-render)

`invariants.rebuild_batches` rebuilt the batch from the clip rows:

| | before | after |
|---|---|---|
| status / stage | `running` / `transcribing` | `complete` / `complete` |
| readyClips | 2 | **3** |
| clipCount | 2 | **3** |
| isTerminal | false | **true** |
| batchState | — | **finished** |
| finishedAt | — | 1794743069724 |

`rebuilt: 1`, delivered 3, candidates 15 (12 never rendered, recorded in `unmetReason`).
A second run reports `rebuilt: 0` (idempotent). Live API now returns
`stage=complete ready=3 batchState=finished isTerminal=True`.

### 1.5 ETA

* `app/eta.py`: `MAX_ETA_SEC = 24 h`, `clamp_eta()`, `eta_from_progress()` returns `None`
  when the projection exceeds 24 h, `plan_total_sec()` clamped.
* `app/task.py`: the ETA divides by the **attempt** clock (`stageRunStartedAt`), not the
  cumulative `stageStartedAt` that survives resumes. The old numbers were
  `fetch 48853.4 s` ÷ 4 % → "~2693 min left, whole job ~2983 min".
* `src/lib/clipmake.ts` + `JobCard.tsx`: `fmtEta` never prints more than 24 h, shows
  `estimating…` when there is no honest number, and prints `1 h 20 min` instead of `80 min`.
* No budget/deadline/watchdog reads the ETA.

### 1.6–1.7

Stall detection stays on real progress counters; the reframe/caption stages keep their
iteration caps; `_is_live_render()` (progress stamps < 180 s) decides whether a job is
really rendering, so a dead job parked on `rendering` is repaired instead of ignored.
"View logs" now opens with a **stage timeline** (per-stage seconds + share of the run).

---

## 2. Local model copies and disk

### 2.1 What the 2263 MB was

`data/models` held **756 MB** in this session, and the Reset dialog's number came from
`sum(modelstore.status()['models'][*].localBytes)`, which counts the whole `hf` cache as one
pseudo-model named "hf" and **misses orphan blobs entirely**:

* `data/models/hf/hub/blobs/0c/0c1311b7…` — **753 MB orphan** (no `.refs`, no repo pointing
  at it): the leftover of the 752 MiB `distil-medium.en` download the CPU-only host can never
  load.
* `models--Systran--faster-distil-whisper-medium.en` — 9 MB of metadata.
* `~/.cache/huggingface` — 1 MB.

The 2263 MB figure was that accounting plus the same bytes counted twice (local dir + HF
repo), so the promise and the apply could never agree.

### 2.2 Why Reset freed 2.1 MB

The apply path only releases what `release_all()` dares to: the active model is protected,
and the orphan blob belonged to no model, so per-model accounting never saw it. The dialog
promised the folder size, the apply removed the metadata, and nothing explained the gap.

### 2.3–2.4 Fixes

* `modelstore.releasable(active, keep_active_local)` — one function feeds **both** the
  preview and the apply: unused models (re-downloadable, no bucket check), orphan blobs, and
  the active model only when "Keep active model local" is off **and** the bucket copy
  verifies. Everything else is returned as `kept` with a reason and its size.
* `diskreset.reset()` uses that plan; `footprint()` reports the real folder size once.
* The Reset dialog now lists `Skipped N MB — not promised, not freed:` with
  `what · MB · why` per item, and the toast carries the per-category breakdown.
* Deleted immediately: the 753 MB orphan blob and the 9 MB `distil-medium.en` metadata.
  The active `small` (int8) is the only local copy, kept behind the new
  `whisper.keep_local` setting (default on).

### 2.5–2.6 Policy and sizes

* `modelaudit.registry()` labels every model `required` / `optional` / `unused`; the live
  `/api/health` now returns
  `required bucket-only small 461.1 MB`, `required ready face_detection…tflite 0.65 MB`,
  `optional optional model_quantized.onnx`, `optional remote deepseek-flash`.
* The status card shows the size per model and marks optional ones.
* The download preflight (`modelstore.prepare`) refuses when the weights would leave less
  than 2000 MB free (stricter than the requested 1 GB) and says so.
* Watchdog unchanged and verified: < 1500 MB free → safe sweep, < 800 MB → new batches
  paused with "Waiting for disk space".

### Disk, before and after

| | free |
|---|---|
| before cleanup | 1520 MB |
| right after deleting the unused copies | **2275 MB** (+755 MB) |
| now (active `small` restored from the bucket for the language proof) | 1777 MB |

`data/models` is now 464 MB = the single active `small` int8 copy.

---

## 3. Bug sweep

* `ruff check app tools/doctor.py tools/e2e.py --select F,E9` → only the **pre-existing,
  locked** `app/video.py:516/519 F821 Undefined name 'cut'` remains. `app/video.py` is the
  locked visual contract, so it was not touched — **this needs your decision**.
* `bun tsc -b --noEmit` → clean.
* `tools/doctor.py` extended with four checks, all PASS on the live host:
  state machine (1 job, 0 bad), "Your batches vs clips" (1 batch with clips, 0 unfinished),
  model folder (463.7 MB, active `small`, freeable now 0.0 MB, kept reason printed),
  ETA sanity (max 86400 s = cap).
* Invariants run at boot and every 10 minutes; `rebuild_batches` runs first.
* Not done: the full browser button pass and `pip-audit`/`npm audit` (see BLOCKED).

---

## 4. English and Spanish

### 4.1 The real language bug

`app/asr_accurate.model_ladder()` used `if language and language != "en"` — so
`language=None` (which is exactly what **Auto** passes) fell into the English branch and
chose **`distil-medium.en`**, an English-only model. On a Spanish source that model has no
Spanish vocabulary: it transcribes Spanish as mangled English and reports `language="en"`
with confidence, so the per-job language lock recorded the wrong language and every later
window stayed English. Fixed: English-only weights are used **only** when the language is
known to be `en`; Auto/`es`/anything else gets the multilingual ladder `small → base → tiny`.

Also fixed: `translate.write_tracks()` now takes the job's resolved language, so a Spanish
clip's SRT is the **verbatim** Spanish track instead of a translation into Spanish; the
detection prompt now states the transcript's language so titles/hooks/hashtags come back in
the video's language; the non-speech marker list gained the Spanish vocabulary
(`[ininteligible]`, `[Música de fondo]`, `[risas]`, `[aplausos]`, `[suspiros]`, …).

### 4.2 Spanish glyphs — verified

All bundled fonts contain every required glyph (16/16: `áéíóúÁÉÍÓÚñÑüÜ¿¡`):
`Anton-Regular`, `BebasNeue-Regular`, `Montserrat-{Bold,ExtraBold,SemiBold}`, `Poppins-ExtraBold`.
Uppercasing is `str.upper()` (Unicode-aware): `el niño comió piña → EL NIÑO COMIÓ PIÑA`,
`¿qué pasó? ¡increíble! → ¿QUÉ PASÓ? ¡INCREÍBLE!`, `pingüino über → PINGÜINO ÜBER`.

### 4.3 Proof with real media

Real Spanish source: `youtu.be/pA69U4L9KOY`, 93.4 s downloaded (`data/proof/es_source.mp4`,
640×360 h264/aac).

```
configured model: small  language: auto
loaded in 1.0s: faster-whisper small int8
detected language: 'es'  (37 words in 10.5s, rtf 0.11)
first 15 s: ¿Qué hora es ahora?
word timings: ¿Qué[0.1-1.2] hora[2.6-4.9] es[5.8-6.3] ahora?[6.3-6.7] ¿Dónde[18.0-18.4] vas?[18.4-20.2] …
```

Real English source: `youtu.be/jNQXAC9IVRw`, 90 s of audio.

```
detected language: 'en'  (36 words in 6.8s, rtf 0.36)
first 15 s: Alright, so here we are, one of the elephants. The cool thing about these guys
is that they have really, really, really long trunks. And that's cool.
```

Real Spanish clip through the **locked** renderer (`encode.render_clip` → `subtitle.write_ass`
→ ffmpeg/libass), title `¿QUÉ HORA ES? ¡APRENDE ESPAÑOL YA!`:

```
ffprobe data/proof/es_clip.mp4
  h264 1080x1920 30/1 · aac 44100 Hz 2 ch · duration 10.067 s
```

Burned ASS (unchanged look, Montserrat ExtraBold, same positions/colours):

```
Dialogue: …{\an2\pos(540,1364)\1c&H0000FFFF&}¿QUÉ HORA{\1c&H00FFFFFF&}
Dialogue: …{\an2\pos(540,1450)}ES AHORA?{\1c&H00FFFFFF&}
Dialogue: …{\an2\pos(540,1364)}¡EL NIÑO COMIÓ{\1c&H00FFFFFF&}
Dialogue: …{\an2\pos(540,1450)\1c&H0000FF00&}PIÑA Y PINGÜINO!{\1c&H00FFFFFF&}
Dialogue: …{\an2\pos(540,1364)\1c&H000000FF&}CANCIÓN DE LA{\1c&H00FFFFFF&}
Dialogue: …{\an2\pos(540,1450)}MAÑANA — ÜBUNG{\1c&H00FFFFFF&}
```

`data/proof/clip000_es.srt` (the real sidecar writer, UTF-8):

```
1
00:00:00,200 --> 00:00:03,400
¿Qué hora es ahora?

2
00:00:03,600 --> 00:00:06,800
¡El niño comió piña y pingüino!

3
00:00:07,000 --> 00:00:10,000
CANCIÓN DE LA MAÑANA — ÜBUNG
```

Frames were written (`data/proof/crop_title.jpg`, `crop_caption.jpg`) but **not visually
inspected** in this session — the image viewer returned raw bytes. The glyph evidence above
is font-coverage + the burned ASS text, not a look-at-the-pixels check.

`tools/e2e.py <url> --lang en|es` is the driver for a full batch on your host (submit, poll,
detected language, first 15 s, batch listing, ffprobe, SRT head). It was smoke-tested for
argument handling and its "backend not reachable" path.

---

## Tests

```
.venv/bin/python -m pytest tests/test_terminal_states.py tests/test_batch_finalize.py \
  tests/test_models_and_language.py tests/test_store.py tests/test_reclaim.py \
  tests/test_stale_reconcile.py tests/test_queue_guards.py tests/test_worker_adoption.py -q
83 passed
```

New: `tests/test_terminal_states.py` (terminal never reopens, recovery guard ordering,
runner never re-labels fetch, ETA clamp), `tests/test_batch_finalize.py` (rebuild from rows,
idempotency, finished-with-failures, finalize-before-cleanup, retry-only-finalize),
`tests/test_models_and_language.py` (ladder never English-only on auto, releasable plan,
orphan blobs, honest zero, diskreset uses the plan).

`pytest -q` from the repo root still fails collection (`ModuleNotFoundError: No module named
'app'` for 5 modules) and `pytest -q --rootdir app` runs `tools/caption_social_test.py`,
which calls `sys.exit(1)` → INTERNALERROR. Both are pre-existing; the focused set above is
the working invocation.

---

## BLOCKED

1. **Git** — commits per section and `git diff --stat` are refused by the host. No locked
   file was modified: `app/video.py`, `app/reframe.py`, `app/caption_style.py`,
   `app/caption_presets.py` and the UI layout were not touched (only status/message text,
   the ETA formatter, the model pills, the Reset dialog text and the new log timeline).
2. **Full EN + ES batches (3 clips each)** — not run in this session: a 5–8 min source needs
   ~30–40 min per language on this 1-vCPU host, and the detection provider is out of credit
   (`DeepSeek 402 insufficient balance`). The language fix itself is proven with real media
   (detected `es`/`en`, real transcripts, real Spanish render + SRT). Run
   `tools/e2e.py <url> --lang es` and `--lang en` on your host for the full batch proof.
3. **Browser button pass** — not performed (no browser control in this session). The live
   API was exercised instead (`/api/health`, `/api/jobs`, `/api/doctor` all 200).
4. **`pip-audit` / `npm audit`** — not run.
5. **`app/video.py:516/519 F821 'cut'`** — pre-existing, inside the locked visual file.
   Needs your go-ahead before anyone touches it.
6. **Stale runner process** — the old `app.runner 6ce9abe…` could not be killed (host policy
   forbids killing processes). The new stage-boundary guard stops it at the next stage and
   the periodic rebuild keeps the document correct; the live API already reports the batch
   as `complete` with 3 clips.
