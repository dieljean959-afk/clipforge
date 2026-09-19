// ClipForge engine — pure functions, no Convex imports.
//
// This module is the TypeScript port of the MoneyPrinterTurbo pipeline pieces
// ClipForge reuses (subtitle -> caption -> clip rendering), adapted from
// "topic -> stock-footage video" to "existing video -> viral clips":
//   - word-level transcript (faster-whisper equivalent shape: {t0, t1, word})
//   - chunked clip detection with 0-100 virality scoring (LLM or heuristic)
//   - 9:16 render plan (crop box, silence cuts, word-pop caption timing)
//   - SRT caption export (captions.txt)
//
// Transcript + render execution are simulated in this hosted environment;
// the data shapes and stage ordering match the real yt-dlp / faster-whisper /
// Gemini / MoviePy pipeline so the UI and storage layer do not change when a
// real worker is attached.

export type ClipStage =
  | "queued"
  | "grabbing"
  | "transcribing"
  | "finding"
  | "rendering"
  | "complete"
  | "failed";

export const STAGE_ORDER: ClipStage[] = [
  "queued",
  "grabbing",
  "transcribing",
  "finding",
  "rendering",
  "complete",
];

export interface TranscriptWord {
  /** word start time, seconds from video start */
  t0: number;
  /** word end time, seconds */
  t1: number;
  /** the word itself */
  word: string;
}

export interface TranscriptSegment {
  id: number;
  start: number;
  end: number;
  text: string;
}

export interface ClipPlan {
  start: number;
  end: number;
  score: number;
  title: string;
  hookText: string;
  transcript: string;
  hashtags: string[];
  captionCues: CaptionCue[];
  cropBox: { x: number; y: number };
  silenceCuts: { start: number; end: number }[];
}

export interface CaptionCue {
  start: number;
  end: number;
  text: string;
}

// ---------------------------------------------------------------------------
// Seeded RNG so the same video always produces the same clips
// ---------------------------------------------------------------------------

function xmur3(str: string): () => number {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return (h ^= h >>> 16) >>> 0;
  };
}

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Synthetic word-level transcript (stand-in for faster-whisper output)
// ---------------------------------------------------------------------------

const OPENERS = [
  "Here is the part",
  "Nobody tells you this, but",
  "The biggest mistake",
  "What most people miss",
  "Let me show you exactly how",
  "This changed everything",
  "Stop doing",
  "The truth about",
  "I tested this for thirty days and",
  "You are leaving money on the table with",
];

const MIDS = [
  "your editing workflow",
  "the first three seconds",
  "retention curves",
  "short form pacing",
  "your hook structure",
  "posting cadence",
  "audience targeting",
  "the algorithm shift",
  "caption styling",
  "framing and reframe",
  "sound design choices",
  "topic selection",
];

const PUNCHES = [
  "and it instantly doubled watch time",
  "so the clip lands before the swipe",
  "which is why the last one hit a million views",
  "and suddenly the numbers flipped",
  "that single tweak carried the whole video",
  "because viewers stay for the payoff",
  "and the retention graph went flat vertical",
  "so test it on your next upload",
  "and the comments exploded overnight",
  "which nobody talks about",
];

const FILLER = ["okay", "look", "right", "seriously", "honestly", "now"];

function pick<T>(rand: () => number, arr: T[]): T {
  return arr[Math.floor(rand() * arr.length)];
}

function hashString(str: string): number {
  const seedFn = xmur3(str);
  return seedFn();
}

/** Simulated duration for a source video, in seconds. */
export function simulateDuration(source: string, isUpload: boolean): number {
  const rand = mulberry32(hashString(`dur:${source}`));
  const minutes = isUpload ? 3.5 + rand() * 4 : 8.5 + rand() * 6;
  return Math.round(minutes * 60);
}

/**
 * Build a deterministic word-level transcript for the given source.
 * Mirrors the shape faster-whisper returns so swapping in a real worker
 * requires no changes downstream.
 */
export function buildTranscript(
  source: string,
  durationSec: number,
): TranscriptWord[] {
  const rand = mulberry32(hashString(`words:${source}:${durationSec}`));
  const words: TranscriptWord[] = [];
  let t = 0.6 + rand() * 0.8;

  while (t < durationSec - 2) {
    const sentence: string[] = [];
    sentence.push(pick(rand, OPENERS));
    sentence.push(pick(rand, MIDS));
    if (rand() > 0.35) sentence.push(pick(rand, FILLER));
    sentence.push(pick(rand, PUNCHES));

    for (const chunk of sentence) {
      for (const w of chunk.split(" ")) {
        const dur = 0.24 + rand() * 0.18;
        words.push({ t0: round2(t), t1: round2(t + dur), word: w });
        t += dur + 0.02 + rand() * 0.05;
      }
    }
    // natural pause between sentences; occasionally dead air worth cutting
    t += rand() < 0.22 ? 0.9 + rand() * 1.4 : 0.25 + rand() * 0.5;
  }
  return words;
}

export function segmentTranscript(words: TranscriptWord[]): TranscriptSegment[] {
  const segments: TranscriptSegment[] = [];
  let current: TranscriptWord[] = [];
  const flush = () => {
    if (current.length === 0) return;
    const id = segments.length;
    segments.push({
      id,
      start: current[0].t0,
      end: current[current.length - 1].t1,
      text: current.map((w) => w.word).join(" "),
    });
    current = [];
  };
  for (let i = 0; i < words.length; i++) {
    current.push(words[i]);
    const gapToNext = i + 1 < words.length ? words[i + 1].t0 - words[i].t1 : 0;
    const len = current.length;
    if (gapToNext > 0.6 || len >= 12 || i === words.length - 1) flush();
  }
  return segments;
}

// ---------------------------------------------------------------------------
// Clip detection
// ---------------------------------------------------------------------------

interface DetectOptions {
  /** target word count for the sliding window */
  minWords?: number;
  maxWords?: number;
  maxClips?: number;
  seedSalt?: string;
}

const CURIOSITY = [
  "nobody",
  "mistake",
  "truth",
  "secret",
  "why",
  "how",
  "tested",
  "changed",
  "stop",
  "miss",
];

const ENERGY = [
  "instantly",
  "doubled",
  "exploded",
  "flipped",
  "million",
  "vertical",
  "payoff",
  "landed",
  "hit",
];

function scoreWindow(
  words: TranscriptWord[],
  rand: () => number,
): { score: number; reasons: { hook: number; energy: number } } {
  const text = words.map((w) => w.word).join(" ");
  const lower = text.toLowerCase();
  let hook = 0;
  for (const c of CURIOSITY) if (lower.includes(c)) hook += 1;
  let energy = 0;
  for (const e of ENERGY) if (lower.includes(e)) energy += 1;

  // clip pacing: prefer ~30-45s windows, penalize rambling ones
  const seconds = words[words.length - 1].t1 - words[0].t0;
  const pacing =
    seconds >= 22 && seconds <= 52 ? 1 : seconds < 18 ? 0.5 : 0.65;

  const wordRate = words.length / Math.max(seconds, 1); // speech density
  const density = wordRate > 2.2 ? 1 : wordRate / 2.2;

  const score = Math.round(
    Math.min(
      98,
      38 + hook * 7 + energy * 5 + pacing * 14 + density * 12 + rand() * 9,
    ),
  );
  return { score, reasons: { hook, energy } };
}

function titleFrom(words: TranscriptWord[]): string {
  const stop = new Set([
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "is",
    "are",
    "you",
    "your",
    "for",
    "with",
    "that",
    "this",
    "it",
    "to",
    "of",
    "in",
    "on",
    "so",
  ]);
  const notable = words
    .map((w) => w.word.replace(/[^a-zA-Z0-9' ]/g, ""))
    .filter((w) => w.length > 3 && !stop.has(w.toLowerCase()));
  const base = notable.slice(0, 5).join(" ") || "Viral moment";
  return base.charAt(0).toUpperCase() + base.slice(1);
}

export function buildHashtags(title: string): string[] {
  const base = ["#clips", "#shorts", "#viral"];
  const tags = title
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 3)
    .slice(0, 3)
    .map((t) => `#${t}`);
  return [...new Set([...tags, ...base])].slice(0, 6);
}

/**
 * Heuristic clip detection: slide a sentence-aligned window over the
 * transcript, score each window, dedupe overlaps, keep the top candidates.
 * This is the fallback used when the LLM detection call is unavailable.
 */
export function heuristicClipDetection(
  words: TranscriptWord[],
  opts: DetectOptions = {},
): ClipCandidateOutput[] {
  const minWords = opts.minWords ?? 45;
  const maxWords = opts.maxWords ?? 130;
  const maxClips = opts.maxClips ?? 12;
  const rand = mulberry32(hashString(`detect:${opts.seedSalt ?? "default"}`));

  const candidates: ClipCandidateOutput[] = [];

  // Slide a sentence-aligned window over the word stream.
  let i = 0;
  while (i < words.length) {
    let j = i;
    let count = 0;
    let endT = words[i].t0;
    while (j < words.length && count < maxWords) {
      count++;
      endT = words[j].t1;
      j++;
      if (count >= minWords) {
        const gap = j < words.length ? words[j].t0 - words[j - 1].t1 : Infinity;
        if (count >= maxWords || (gap > 0.5 && rand() > 0.35)) break;
      }
    }

    const start = words[i].t0;
    const windowWords = words.slice(i, j);
    if (count >= minWords * 0.6 && endT - start >= 15) {
      const { score } = scoreWindow(windowWords, rand);

      // dedupe: skip windows overlapping > 40% with an accepted candidate
      const overlapsTooMuch = candidates.some((c) => {
        const overlap = Math.min(c.end, endT) - Math.max(c.start, start);
        return overlap > 0.4 * Math.min(c.end - c.start, endT - start);
      });
      if (!overlapsTooMuch) {
        const title = titleFrom(windowWords);
        candidates.push({
          start: round2(start),
          end: round2(endT),
          score,
          title,
          hookText: windowWords
            .slice(0, 14)
            .map((w) => w.word)
            .join(" "),
          transcript: windowWords.map((w) => w.word).join(" "),
          hashtags: buildHashtags(title),
        });
      }
    }

    if (candidates.length >= maxClips * 3) break;
    // advance by roughly half a window so overlapping takes can be re-scored
    i += Math.max(8, Math.floor(count * 0.5));
  }

  candidates.sort((a, b) => b.score - a.score);
  return candidates.slice(0, maxClips);
}

export interface ClipCandidateOutput {
  start: number;
  end: number;
  score: number;
  title: string;
  hookText: string;
  transcript: string;
  hashtags: string[];
}

// ---------------------------------------------------------------------------
// AI clip detection: prompt + response parsing
// ---------------------------------------------------------------------------

const MAX_PROMPT_CHARS = 14000;

export function buildClipDetectionPrompt(
  segments: TranscriptSegment[],
  durationSec: number,
): string {
  const lines: string[] = [];
  let used = 0;
  for (const seg of segments) {
    const line = `[${seg.start.toFixed(1)}-${seg.end.toFixed(1)}] ${seg.text}`;
    if (used + line.length > MAX_PROMPT_CHARS) break;
    lines.push(line);
    used += line.length;
  }
  return [
    "You are a short-form video editor. Below is a word-level transcript of a video",
    `(${Math.round(durationSec)} seconds total) with segment timestamps.`,
    "",
    "Select the 5-12 most viral-worthy clips for TikTok/Reels/Shorts:",
    "- each clip 20-55 seconds long",
    "- starts on a hook, ends on a payoff",
    "- do not cut mid-sentence",
    "",
    'Return ONLY a JSON array: [{"start": number (seconds), "end": number, "score": 0-100, "title": "3-6 word punchy title", "hook_text": "opening line"}]',
    "",
    "TRANSCRIPT:",
    ...lines,
  ].join("\n");
}

export function parseClipDetectionResponse(
  raw: string,
  words: TranscriptWord[],
  maxClips: number,
): ClipCandidateOutput[] {
  // extract the first JSON array from the response
  const match = raw.match(/\[[\s\S]*\]/);
  if (!match) throw new Error("No JSON array found in model response");
  const parsed: unknown = JSON.parse(match[0]);
  if (!Array.isArray(parsed)) throw new Error("Response is not an array");

  const out: ClipCandidateOutput[] = [];
  for (const item of parsed.slice(0, maxClips)) {
    const rec = item as Record<string, unknown>;
    const start = Number(rec.start);
    const end = Number(rec.end);
    const score = Number(rec.score);
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
    if (!Number.isFinite(score)) continue;
    const startR = Math.max(0, round2(start));
    const endR = Math.max(startR + 8, round2(end));
    const inWindow = words.filter((w) => w.t0 >= startR && w.t1 <= endR);
    const transcript =
      inWindow.length > 0
        ? inWindow.map((w) => w.word).join(" ")
        : String(rec.hook_text ?? "");
    const title = titleFrom(
      inWindow.length > 0
        ? inWindow
        : String(rec.title ?? "Viral moment")
            .split(" ")
            .map((word) => ({ t0: startR, t1: endR, word })),
    );
    out.push({
      start: startR,
      end: endR,
      score: Math.max(0, Math.min(100, Math.round(score))),
      title: title.slice(0, 60),
      hookText: String(rec.hook_text ?? transcript.slice(0, 90)).slice(0, 120),
      transcript: transcript.slice(0, 800),
      hashtags: buildHashtags(title),
    });
  }
  if (out.length === 0) throw new Error("Model returned no usable clips");
  out.sort((a, b) => b.score - a.score);
  return out;
}

// ---------------------------------------------------------------------------
// 9:16 render plan: silence cuts + face-tracking crop + caption cues
// ---------------------------------------------------------------------------

/** Simulated Silero-VAD style dead-air cuts: the sentence gaps marked in the transcript. */
export function detectSilenceCuts(
  words: TranscriptWord[],
  clipStart: number,
  clipEnd: number,
): { start: number; end: number }[] {
  const cuts: { start: number; end: number }[] = [];
  const inClip = words.filter((w) => w.t0 >= clipStart && w.t1 <= clipEnd);
  for (let i = 1; i < inClip.length; i++) {
    const gapStart = inClip[i - 1].t1;
    const gapEnd = inClip[i].t0;
    if (gapEnd - gapStart > 0.55) {
      cuts.push({ start: round2(gapStart - clipStart), end: round2(gapEnd - clipStart) });
    }
  }
  return cuts;
}

/** Simulated MediaPipe face-track crop box: x drifts to follow the speaker. */
export function computeCropBox(
  source: string,
  clipStart: number,
): { x: number; y: number } {
  const rand = mulberry32(hashString(`face:${source}:${Math.floor(clipStart)}`));
  return { x: Math.round(rand() * 480), y: 0 };
}

/**
 * Word-pop caption cues: group 1-3 words per cue (max 1.1s), times relative
 * to clip start — the same timing data the ASS renderer consumes.
 */
export function buildCaptionCues(
  words: TranscriptWord[],
  clipStart: number,
  clipEnd: number,
): CaptionCue[] {
  const inClip = words.filter((w) => w.t0 >= clipStart && w.t1 <= clipEnd);
  const cues: CaptionCue[] = [];
  let current: TranscriptWord[] = [];
  const flush = () => {
    if (current.length === 0) return;
    cues.push({
      start: round2(current[0].t0 - clipStart),
      end: round2(current[current.length - 1].t1 - clipStart),
      text: current.map((w) => w.word).join(" "),
    });
    current = [];
  };
  for (const w of inClip) {
    current.push(w);
    const dur = w.t1 - current[0].t0;
    if (current.length >= 3 || dur >= 1.1) flush();
  }
  flush();
  return cues;
}

// ---------------------------------------------------------------------------
// SRT export
// ---------------------------------------------------------------------------

function srtTime(seconds: number): string {
  const ms = Math.max(0, Math.round(seconds * 1000));
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const milli = ms % 1000;
  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(milli, 3)}`;
}

export function buildSrt(cues: CaptionCue[]): string {
  return cues
    .map((cue, i) => `${i + 1}\n${srtTime(cue.start)} --> ${srtTime(cue.end)}\n${cue.text}\n`)
    .join("\n");
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

/** Full pipeline: transcript -> candidates -> render plans. */
export function forgeClips(
  source: string,
  isUpload: boolean,
  candidates: ClipCandidateOutput[],
  maxClips: number,
): ClipPlan[] {
  const duration = simulateDuration(source, isUpload);
  const words = buildTranscript(source, duration);
  const top = candidates.slice(0, Math.max(5, Math.min(15, maxClips)));
  return top.map((c) => {
    const clipStart = Math.max(0, c.start);
    const clipEnd = Math.min(duration, Math.max(clipStart + 8, c.end));
    return {
      start: clipStart,
      end: clipEnd,
      score: c.score,
      title: c.title,
      hookText: c.hookText,
      transcript: c.transcript,
      hashtags: c.hashtags,
      captionCues: buildCaptionCues(words, clipStart, clipEnd),
      cropBox: computeCropBox(source, clipStart),
      silenceCuts: detectSilenceCuts(words, clipStart, clipEnd),
    };
  });
}
