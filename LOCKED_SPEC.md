# LOCKED: do not modify without the owner's written approval — LOCKED_SPEC.md

This is the measured, current ClipMake look. It was captured from real 1080x1920
renders of the real 23:36 source (`https://youtu.be/QqdCXWFobi8`, 1416.482 s,
1280x720, 59.94 fps) using the existing renderer. **No synthetic media was
used.** The three measured reference renders are `clip_000`, `clip_001` and
`clip_003` from the real source's current cut list.

> Measurement limitation (BLOCKED part of A1): the configured bucket contained
> **0 output objects** at measurement time (`clipforge/outputs/` = 0); only the
> source and Whisper weights were present. The three references therefore come
> from three real clips of the one real source in the bucket, not three
> different videos. YouTube exposed only storyboard formats for the second
> requested source (`627Fvn2fEjU`: `Requested format is not available`), so a
> three-video measurement needs a host-side source/output. This limitation is
> recorded rather than filled with generated media.

Reference renders: 34.067 s, 1080x1920. Audio/sync gates: `passed=True`, lag
`-20 ms`, correlation `0.9500 / 0.9602 / 0.9757` (speech check used local
`base`, which is non-decisive on this 2 GiB host; placement remains decisive).

---

## 1. Frame and contract

- **1080 x 1920**, H.264 High profile, level 4.0, **yuv420p**, **30 fps CFR**
  (`avg_frame_rate=r_frame_rate=30/1`; 1020 frames per 34.067 s reference).
- `+faststart`; `moov` before `mdat`.
- AAC LC, 44.1 kHz, stereo, nominal 128 kbps (measured 128.4–128.7 kbps).
- Full decode clean; duration drift 0.067 s on a 34.000 s plan.
- No loudness normalization is applied. Measured integrated loudness of the
  three references: **-15.6, -15.5, -16.4 LUFS**; true peak -0.8, -0.8, -1.4
  dBFS. The audio bytes are the source range declared by the per-clip
  `audioPlan`, not a second decode or a different clip's audio.

## 2. Blurred-background FIT layout (measured)

For a 16:9 source in the 9:16 frame:

- The full-frame blurred copy is **1080x1920**, scaled to cover, cropped to
  1080x1920, `gblur=sigma=30`, square pixels.
- The sharp foreground is **1080x608**, scaled to frame width with a bicubic
  upscale, centred horizontally.
- Sharp band: **top y=656, bottom y=1264** (measured at the sharp/blur seam on
  both side strips; all three references agree).
- Blur band above: y=0–655 (656 px); below: y=1265–1919 (655 px).
- **Darken: none. Saturation adjustment: none.** There is no `eq` or colour
  filter in the graph; the background is only scaled, cropped and blurred.
- `setsar=1`; output `fps=30`.

## 3. White title bar (measured)

- Full-width white plate, x=0–1079. **Short title: y=160–230 inclusive
  (71 px tall)**. Measured on all three references.
- Text: bold black (`RGB 10,10,10`), centred, Montserrat ExtraBold,
  **40 px** for the short reference. The long-title plan uses **38 px** and
  grows the plate to **114 px** (y=160–274) rather than clipping.
- Bar width 1080; padding x=24, y=12; line factor 1.18.
- Wrap at **90% of frame width (972 px)**, at most **2 lines**; shrink in 2 px
  steps to a 22 px floor, then ellipsise only an impossible single word.
- Hook title text comes from the clip's own transcript/content in the source
  language. LLM failure: retry once, then the transcript's first sentence,
  labelled `fallback: first sentence`; no placeholder.
- The plate is overlaid for the entire clip at y=160; no title text outside the
  plate.

## 4. Caption block (measured from real frames)

- **One style.** Montserrat ExtraBold, uppercase, heavy. ASS `PlayResX/Y =
  1080x1920`; ASS font size **81** renders a measured visible capital height of
  **37–38 px**. Black outline **6**. No box.
- Two stacked lines, 4–5 words per chunk, at most **16 characters per line**.
  Wrapping is balanced by character count.
- Measured line groups (visible glyph rows): **line 1 y=1304–1340/1347**,
  **line 2 y=1390–1426**. Line pitch measured **86 px** (top-to-top).
- ASS bottom-anchored positions are `\an2\pos(540,1364)` and
  `\an2\pos(540,1450)`. Relative to the sharp video's bottom edge (y=1264), the
  visible line bottoms are **+76 px** and **+162 px**; the block sits in the
  lower blur band of the current look.
- Colors: one line white `#FFFFFF`; the other cycles per chunk through
  **yellow `#FFFF00` → green `#00FF00` → red `#FF0000`**, alternating which
  line is coloured. A user-fixed highlight colour replaces the cycle for every
  chunk. Outline is black.
- Auto-fit max line width: **972 px (90% of 1080)**. The widest line measured in
  the three reference frames is **524 px** (the auto-fit floor still protects
  unusual strings).
- Position is derived from the resolved style, never from the preview alone.
  Both preview and render call `app.caption_style.resolve_caption_settings`.

## 5. Emoji

- Bundled Twemoji PNGs; **140x140 px** (source asset 144 px with transparent
  padding), centred at x=470, y=1132 as resolved by `app.emoji.emoji_geometry`.
- Gap above the caption block: **24 px**.
- Animation: fade in **0.10 s**, fade out **0.12 s**, 12 px upward lift over
  **0.18 s**; one overlay pass; density Low/Medium/High (15/35/60%).
- Current references have one selected emoji (fallback: keywords, confidence
  0.62), so the emoji layer is real and present.

## 6. Audio and sync contract

- Every clip renders video and audio from the same explicit `audioPlan`
  (`sourceKey`, absolute `start`/`end`, `streamStart`, `order`).
- The renderer rejects a missing/mismatched plan, wrong source key, wrong order,
  or a shared seek.
- Sync gate before `ready`: placement within the gate, duplicate check, and the
  local speech check where the model is decisive.
- References: **lag -20 ms**, correlation 0.9500–0.9757, **all passed**.

## 7. Locked files

The visual contract lives in these files. They carry a
`LOCKED: do not modify` header and their SHA-256 is in `locked.sha256`:

- `app/caption_presets.py` — caption style constants
- `app/caption_style.py` — resolved style / preview+runtime source of truth
- `app/captions_animated.py` — FIT foreground geometry
- `app/subtitle.py` — caption builder, ASS generator, transcript output
- `app/titlebar.py` — white title-bar renderer
- `app/emoji.py` — emoji selection/overlay geometry
- `app/encode.py` — layout/filter graph and clip encode
- `app/video.py` — crop/filter contract, probe, clip contract
- `app/memguard.py` — memory admission units
- `src/components/clipmake/CaptionStyleDialog.tsx` — style UI
- `src/components/clipmake/NewJobForm.tsx` — batch form UI

`tools/check_locked.py` fails when any hash differs. Updating hashes requires
the owner to say **UNLOCK** in chat. `tests/test_locked_render.py` and
`tests/golden/` fail on visual drift with real media. If a change is needed in a
locked file, stop and ask the owner first.
