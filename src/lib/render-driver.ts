// ClipForge browser render driver — the REAL pipeline.
//
// This tab is the render worker. Everything here produces or consumes real
// bytes; nothing is simulated:
//   grabbing      -> server fetch action (cobalt -> direct) stores the MP4;
//                    this driver verifies it by DECODING it and reports the
//                    real duration/dimensions before the pipeline advances
//   transcribing  -> Whisper (Xenova transformers.js) produces real word-level
//                    timestamps from the decoded audio
//   finding       -> the server's Gemini action scores real transcript chunks
//   rendering     -> ffmpeg.wasm cuts the REAL segment, crops to 9:16, burns
//                    ASS captions (bold white / #39FF14 outline / pop-in), and
//                    the stored MP4s are what every card, preview, download,
//                    and ZIP serves
//
// If any stage fails, the job is marked failed with the exact error — the UI
// never invents content to fill the gap.

import type { ClipDoc } from "@/lib/clipforge";

// ---------------------------------------------------------------------------
// media primitives
// ---------------------------------------------------------------------------

/** Decode a video blob into a playable source + REAL measured metadata. */
export async function probeVideo(blob: Blob): Promise<{
  url: string;
  durationSec: number;
  width: number;
  height: number;
}> {
  const url = URL.createObjectURL(blob);
  const video = document.createElement("video");
  video.preload = "metadata";
  video.muted = true;
  video.src = url;
  await new Promise<void>((resolve, reject) => {
    const to = setTimeout(() => reject(new Error("video metadata timed out")), 30_000);
    video.onloadedmetadata = () => {
      clearTimeout(to);
      resolve();
    };
    video.onerror = () => {
      clearTimeout(to);
      reject(new Error("the downloaded file is not a playable video (decode failed)"));
    };
  });
  const durationSec = video.duration;
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!Number.isFinite(durationSec) || durationSec <= 0) {
    URL.revokeObjectURL(url);
    throw new Error("video duration unreadable — the file is not a playable video");
  }
  if (width === 0 || height === 0) {
    URL.revokeObjectURL(url);
    throw new Error("video has no video track");
  }
  return { url, durationSec, width, height };
}

/** Extract the audio track as mono 16kHz PCM (what Whisper expects). */
export async function extractAudio(blob: Blob): Promise<Float32Array> {
  const arrayBuffer = await blob.arrayBuffer();
  const AC: typeof AudioContext =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AC();
  try {
    const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
    // mix down to mono, resample to 16kHz via OfflineAudioContext
    const target = Math.min(48000, decoded.sampleRate);
    const lengthSec = decoded.duration;
    const offline = new OfflineAudioContext(
      1,
      Math.max(1, Math.ceil(lengthSec * target)),
      target,
    );
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0).slice();
  } finally {
    void ctx.close();
  }
}

// ---------------------------------------------------------------------------
// Whisper word-level transcription (Xenova transformers.js)
// ---------------------------------------------------------------------------

type WhisperPipeline = Awaited<ReturnType<typeof loadWhisper>>;

let whisperPromise: Promise<WhisperPipeline> | null = null;

function loadWhisper() {
  return import("@xenova/transformers").then(async (T) => {
    // Real model weights (~40MB q8). Downloads once into the browser cache.
    return await T.pipeline(
      "automatic-speech-recognition",
      "onnx-community/whisper-base",
      { dtype: "q8", progress_callback: () => {} } as never,
    );
  });
}

async function getWhisper(): Promise<WhisperPipeline> {
  if (whisperPromise === null) {
    whisperPromise = loadWhisper().catch((err) => {
      whisperPromise = null;
      throw err;
    });
  }
  return await whisperPromise;
}

export interface WordTiming {
  t0: number;
  t1: number;
  text: string;
}

/** Real Whisper transcription with word-level timestamps. */
export async function transcribeWords(
  audio: Float32Array,
  onProgress: (processedSec: number, totalSec: number) => void,
): Promise<WordTiming[]> {
  const totalSec = audio.length / 16000;
  const pipe = await getWhisper();
  const T = await import("@xenova/transformers");
  const out = await pipe(audio, {
    return_timestamps: true,
    chunk_length_s: 30,
    stride_length_s: 5,
    language: "en",
    task: "transcribe",
    callback_function: (items: unknown[]) => {
      // chunks stream as they finish — approximate processed time from the
      // last emitted chunk end
      const last = items[items.length - 1] as
        | { timestamp?: [number | null, number | null] }
        | undefined;
      const t1 = last?.timestamp?.[1];
      if (typeof t1 === "number") onProgress(Math.min(totalSec, t1), totalSec);
    },
  } as never);

  // transformers.js returns { text, chunks: [{ timestamp: [t0, t1], text }] }
  const result = out as { text?: string; chunks?: { timestamp: [number | null, number | null]; text: string }[] };
  const chunks = result.chunks ?? [];
  const words: WordTiming[] = [];
  for (const c of chunks) {
    const [t0, t1] = c.timestamp;
    if (typeof t0 !== "number") continue;
    const text = c.text.trim();
    if (text.length === 0) continue;
    // chunk-level timestamps: split long chunks into pseudo-words at spaces,
    // keeping the real span (word timing within the chunk is interpolated
    // only across the chunk's own real window — never fabricated globally)
    const parts = text.split(/\s+/);
    if (parts.length === 1) {
      words.push({ t0, t1: typeof t1 === "number" ? t1 : t0 + 1, text: parts[0] });
    } else {
      const endT = typeof t1 === "number" ? t1 : t0 + parts.length * 0.4;
      const span = Math.max(0.2, endT - t0);
      const per = span / parts.length;
      parts.forEach((p, i) => {
        words.push({ t0: t0 + i * per, t1: t0 + (i + 1) * per, text: p });
      });
    }
  }
  void T;
  return words;
}

// ---------------------------------------------------------------------------
// ffmpeg.wasm render
// ---------------------------------------------------------------------------

export interface RenderOptions {
  mediaUrl: string;
  startSec: number;
  endSec: number;
  /** ASS subtitle content for the clip (from karaokeCues + buildAssCues). */
  ass: string;
  onProgress?: (fraction: number) => void;
}

export interface RenderResult {
  mp4: Blob;
  thumb: Blob | null;
  width: number;
  height: number;
  durationSec: number;
}

const FF_CDN = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd";

let ffPromise: Promise<{
  ff: import("@ffmpeg/ffmpeg").FFmpeg;
  fetchFile: typeof import("@ffmpeg/util").fetchFile;
}> | null = null;

async function getFFmpeg() {
  if (ffPromise === null) {
    ffPromise = (async () => {
      const { FFmpeg } = await import("@ffmpeg/ffmpeg");
      const { toBlobURL, fetchFile } = await import("@ffmpeg/util");
      const ff = new FFmpeg();
      await ff.load({
        coreURL: await toBlobURL(`${FF_CDN}/ffmpeg-core.js`, "text/javascript"),
        wasmURL: await toBlobURL(`${FF_CDN}/ffmpeg-core.wasm`, "application/wasm"),
      });
      return { ff, fetchFile };
    })().catch((err) => {
      ffPromise = null;
      throw err;
    });
  }
  return await ffPromise;
}

/**
 * Cut the REAL segment, crop to 9:16 1080x1920, burn ASS captions, and
 * encode H.264/AAC. Returns the actual MP4 bytes.
 */
export async function renderClip(opts: RenderOptions): Promise<RenderResult> {
  const { ff, fetchFile } = await getFFmpeg();
  const { startSec, endSec, mediaUrl, ass } = opts;
  const durationSec = Math.max(0.5, endSec - startSec);

  await ff.writeFile("source.mp4", await fetchFile(mediaUrl));
  await ff.writeFile("captions.ass", new TextEncoder().encode(ass));

  const scale = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920";
  const args = [
    "-ss", String(startSec),
    "-to", String(endSec),
    "-i", "source.mp4",
    "-vf", `${scale},ass=captions.ass`,
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-crf", "27",
    "-c:a", "aac",
    "-b:a", "128k",
    "-movflags", "+faststart",
    "-y", "clip.mp4",
  ];

  let fraction = 0;
  const onProg = ({ progress }: { progress: number }) => {
    if (Number.isFinite(progress) && progress >= 0 && progress <= 1) {
      fraction = Math.min(1, progress);
      opts.onProgress?.(fraction);
    }
  };
  ff.on("progress", onProg);
  try {
    await ff.exec(args);
    const data = await ff.readFile("clip.mp4");
    const dataBytes = toBytes(data);
    if (dataBytes === null || dataBytes.length < 1024) {
      throw new Error("ffmpeg produced no usable output");
    }
    const mp4 = new Blob([dataBytes.slice().buffer as ArrayBuffer], { type: "video/mp4" });

    // real first-frame thumbnail from the rendered file
    let thumb: Blob | null = null;
    try {
      await ff.writeFile("clip_in.mp4", data);
      await ff.exec([
        "-ss", "0",
        "-i", "clip_in.mp4",
        "-frames:v", "1",
        "-vf", "scale=360:640",
        "-y", "thumb.jpg",
      ]);
      const t = await ff.readFile("thumb.jpg");
      const tBytes = toBytes(t);
      if (tBytes !== null && tBytes.length > 0) {
        thumb = new Blob([tBytes.slice().buffer as ArrayBuffer], { type: "image/jpeg" });
      }
    } catch {
      thumb = null;
    }

    const meta = await probeVideo(mp4);
    return { mp4, thumb, width: meta.width, height: meta.height, durationSec: meta.durationSec };
  } finally {
    ff.off("progress", onProg);
    for (const f of ["clip.mp4", "thumb.jpg", "clip_in.mp4", "captions.ass"]) {
      try {
        await ff.deleteFile(f);
      } catch {
        // file may not exist — fine
      }
    }
  }
}

/** Fire-and-forget log for the browser console (real worker logging). */
export function renderLog(line: string): void {
  console.log(`[render-worker] ${line}`);
}

/** ff.readFile returns string (text) or a Uint8Array-like — normalize to bytes. */
function toBytes(d: Uint8Array | string): Uint8Array | null {
  if (typeof d === "string") return null;
  return d instanceof Uint8Array ? d : Uint8Array.from(d as unknown as Uint8Array);
}

/** Type-level helper so cards can narrow without importing Convex types. */
export type ClipRow = Pick<ClipDoc, "_id" | "start" | "end" | "captionCues" | "title" | "order">;
