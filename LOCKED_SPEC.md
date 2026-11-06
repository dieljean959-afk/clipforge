# LOCKED SPEC — do not modify

These values are the contract. A change here is rejected unless it is a
deliberate, reviewed decision; the golden test (`tests/test_locked_render.py`)
fails when the pixels stop matching, and that failure blocks a deploy.

Files carrying the `LOCKED: do not modify` header:
`app/caption_presets.py`, `app/caption_style.py`, `app/subtitle.py`,
`app/titlebar.py`, `app/memguard.py`, `src/components/clipmake/CaptionStyleDialog.tsx`,
`src/components/clipmake/NewJobForm.tsx`.

---

## 1) ONE caption style — the reference video look (measured at 1080x1920)

- Uppercase, heavy sans (Montserrat ExtraBold, bundled `app/fonts/Montserrat-ExtraBold.ttf`).
  ASS `Fontsize 54` ⇒ cap height **38 px**. Black outline **6**. **No box**.
  ASS `PlayResX/Y = 1080x1920`.
- Each chunk = **2 stacked lines**, **4-5 words** total, max **~16 characters per line**.
  Each line is its own Dialogue event: `\an2\pos(540,y)`. Line pitch **86 px**.
  Normal spaces between words.
- One line white `#FFFFFF`, the other coloured. Colour cycles per chunk through
  yellow `#FFFF00`, green `#00FF00`, red `#FF0000` — or the user's chosen
  Highlight colour (then it is that colour for every chunk).
- Position: the bottom edge of the **centred video**, not the blurred band:
  ```
  fgH           = round(1080 * srcH / srcW)
  fgTop         = (1920 - fgH) / 2
  fgBottom      = fgTop + fgH
  captionBottomY = fgBottom - 24
  ```
  Line 2 bottom edge at `captionBottomY`, line 1 at `captionBottomY - 86`; the
  block grows upward. Full-frame 9:16 source: line 1 ≈ **y 1340**, line 2 ≈ **y 1426**.
- **ONE style.** The preset zoo is removed: Bold pop, Vertical 9:16 big captions,
  Karaoke fill, Clean lower-third, Boxed, all "Animated ·" styles and
  Ssemble-style viral no longer exist in the UI or the code paths.
- Allowed customisation (only what changes the render):
  Caption size (80-140 %, scales font + outline + pitch together), Highlight
  colour (Auto cycle or fixed), Burst length, Hard cap, Pause threshold,
  Emoji toggle with Low/Medium/High.
  The **custom** badge shows only when a value differs from the default;
  **Reset** restores the defaults.
- Preview and render read the same resolved settings:
  `app.caption_style.resolve_caption_settings` is the one function.

## 2) WHITE TITLE BAR

- Full-width white bar near the top: **y = 160**, base height **70 px**
  (⇒ 160-230 at 1080x1920), bold black centred text, shown for the **whole clip**.
- Per clip, from the clip's own content: the LLM writes a short hook title from
  the transcript, **in the video's own language**.
- The look is kept. The one allowed change: **the text is never cut off** — wrap
  to at most **2 lines** inside **90 % of the width**, shrink the font down to a
  minimum size to fit, and let the bar grow in height. Never overflow an edge.
- LLM failure: **retry once**, then use the transcript's **first sentence**,
  labelled `fallback: first sentence` in the clip metadata. No placeholder text.

## 3) EMOJI

- The existing Twemoji feature stays: bundled PNGs, **one overlay pass**,
  density Low/Medium/High. Not removed, not redesigned.

## 4) JOBS MUST STOP FAILING

- **Memory admission, one helper, explicit units** (`app/memguard.py`): bytes in,
  MB out. Int8 Whisper `small` needs **~900 MB**. Never refuse when
  `usable >= required`. The old message — *"Host is low on memory (1992 MB
  usable, needs ~1000000000 MB to start)"* — was bytes printed as MB, and
  **1992 MB usable must pass** (unit test).
- **Auto-resume.** On startup, jobs with a stale heartbeat become `interrupted`
  and are re-queued automatically, max **3** attempts, resuming from the last
  cached stage. The Retry button stays. Completed stages show as done in the chips.
- **`source == "pending-upload"` is not runnable** until the upload finished.
  The card says **"Waiting for upload"** and the job expires after **30 minutes**.
- **Preflight before every job**: yt-dlp, ffmpeg, env vars (presence only), free
  disk, free memory (via the helper). On failure the message names the exact
  cause and fix. Never exit 0 as success; the worker sends explicit
  `stage_started` / `stage_done` / `done` / `failed` events, and an exit before
  `done` means failed with the real reason.
- API and worker run as separate processes under a supervisor with auto-restart,
  the worker at low priority (`nice 15`), and no heavy work in the API.
- Stage timeouts, a cache keyed by video id + settings hash, concurrency 1, one
  Whisper model reused per worker.

## 5) LOCK-IN

- This file, plus the `LOCKED: do not modify` headers listed above.
- Golden regression test `tests/test_locked_render.py` — renders a sample clip
  and checks from the frames: cap height 38 px (±4), line pitch 86 px (±6),
  caption bottom at `captionBottomY` (±10), white bar present and not clipped,
  exactly one style in the UI, and the emoji overlay present. Runs in
  `pytest -q` and before every deploy. **If it fails, the change is rejected.**
