# ClipForge

Turn one long video into ten viral 9:16 clips — automatically.

ClipForge is a fork of [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) (MIT),
repurposed from "topic → stock-footage video" into "existing video → viral clips". Paste a video
you own, and the forge:

1. **Grabs** it — yt-dlp primary (1080p max MP4), cobalt.tools fallback, manual upload last resort,
   with retry built into the UI from day one (yt-dlp rots fast when YouTube changes).
2. **Transcribes** it — faster-whisper word-level timestamps power caption timing and silence cuts.
3. **Finds clips** — chunked transcript windows scored 0–100 for hook strength, pacing and energy;
   top 5–15 become clips with titles, hooks and hashtags (LLM scoring with a deterministic fallback).
4. **Renders** — 9:16 1080×1920 (libx264/AAC), speaker-tracked reframing, dead-air removal,
   and word-by-word pop-in captions: bold white text, `#39FF14` neon green outline.

Built on the Freebuff stack: React + Vite + Tailwind, Convex backend with auth, AI clip detection
via the platform LLM integration with a deterministic heuristic fallback.

## Rights notice

**Only clip content you own or have permission to use.** Clipping other creators' videos may
violate platform terms of service and copyright law. ClipForge shows this notice across the app
and requires an ownership confirmation before any job runs.

## Development

```bash
bun install
bun dev          # frontend
bunx convex dev  # backend (pushes src/convex/*)
```

The pipeline (`src/convex/engine.ts`) ships as a deterministic simulation of the
yt-dlp → faster-whisper → LLM → MoviePy worker so the product runs in this hosted
environment; the data shapes match the real worker, so attaching one only requires
swapping the stage implementations in `src/convex/clips.ts`.

## License

MIT — see [LICENSE](./LICENSE). Original MoneyPrinterTurbo code © Harry (harry0703) and
contributors; ClipForge changes © ClipForge contributors.
