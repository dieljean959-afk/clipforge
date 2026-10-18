# DEBUG — "Forge clips" returns Internal Server Error

## Reproduction

- Date: 2026-10-18
- Input: `POST /api/jobs` with
  `{"sourceType":"url","source":"https://youtu.be/nKbiX5PwTcM?si=9lY094G_vsHE7nKB","captionPreset":"bold_pop","language":"en"}`
- Result: `HTTP/1.1 500 Internal Server Error`, body `Internal Server Error`
  (no JSON detail, no job created).

## Where the exception is raised

`app/main.py:1428` (and the identical call at `app/main.py:1458` in
`/api/jobs/upload`):

```
job = store.create_job(...)
```

Full traceback (captured by calling `create_job` in-process):

```
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
  File "/home/daytona/codebase/app/main.py", line 1428, in create_job
    job = store.create_job(
          ^^^^^^^^^^^^^^^^
AttributeError: 'LocalStore' object has no attribute 'create_job'
```

`app/store.py` (LocalStore) never defines `create_job`; it only has
`save_job` / `write_job` / `get_job` / `new_job_id`. So every "Forge clips"
submission (URL or upload) crashed with a raw 500 before a job was ever
queued — this is the bug the user saw.

## Secondary bugs found on the way (all fixed)

1. **`app/task.py:872` (`_stage_fetch`)** — `job['source'][:80]` crashes with
   `TypeError: 'NoneType' object is not subscriptable` when a legacy/partial
   job record has `source: null` (present in `data/jobs/8ae6de742f1e.json`).
   It also logged that the job was "paused" when it had actually failed.
2. **`CreateJobIn.language` / upload `language` were accepted but never
   stored** in the job document, so the ENG/ESP/Auto caption-language choice
   never reached the pipeline.
3. Resume reconcile wrote `stage paused` StageWorkspace metadata even for a
   dead/paused job it re-queued — cleaned up to avoid fake "paused" state.

## What was verified to work already (not the cause)

- Direct yt-dlp fetch of the test URL (Python API): OK — 1276 s, 251 MB 1080p
  downloaded in seconds (`app/download.py::download_url`).
- `probe_url`: OK — title/duration resolved.
- ffmpeg/ffprobe: installed at /usr/bin, libass available.
- DeepSeek key: loaded from env.
