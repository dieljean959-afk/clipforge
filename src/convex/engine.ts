// ClipForge engine — pure helpers shared by the Convex backend and the
// browser pipeline. There is NO simulation here: every unit, cue, and
// candidate is derived from REAL inputs (actual bytes fetched, actual
// Whisper word timings, actual LLM output).

// ---------------------------------------------------------------------------
// shared types
// ---------------------------------------------------------------------------

export interface TranscriptWord {
  /** seconds from source start */
  t0: number;
  t1: number;
  text: string;
}

export interface CaptionCue {
  start: number;
  end: number;
  text: string;
}

export interface ClipCandidateOutput {
  start: number;
  end: number;
  /** 0-100 */
  score: number;
  title: string;
  hookText: string;
  transcript: string;
  hashtags: string[];
}

export interface StageUnits {
  unitLabel: string;
  unitsTotal: number;
  unitsDone: number;
  /** partial-unit detail shown inside the stage line, e.g. "12.4 / 30.1 MB" */
  detail: string;
}

// ---------------------------------------------------------------------------
// stage work units (all computed from real values)
// ---------------------------------------------------------------------------

/** Fetch stage units: real MB downloaded. Total may be unknown until the
 *  fetcher reports Content-Length — that's shown honestly. */
export function fetchUnits(bytesDone: number, bytesTotal: number | null): StageUnits {
  const doneMb = bytesDone / (1024 * 1024);
  if (bytesTotal === null || bytesTotal <= 0) {
    return {
      unitLabel: "MB",
      unitsTotal: 0,
      unitsDone: Math.round(doneMb * 10) / 10,
      detail: `${doneMb.toFixed(1)} MB downloaded (total size unknown yet)`,
    };
  }
  const totalMb = bytesTotal / (1024 * 1024);
  return {
    unitLabel: "MB",
    unitsTotal: Math.round(totalMb * 10) / 10,
    unitsDone: Math.round(Math.min(doneMb, totalMb) * 10) / 10,
    detail: `${doneMb.toFixed(1)} / ${totalMb.toFixed(1)} MB`,
  };
}

/**
 * Transcribe stage units: seconds of audio processed. Whisper decodes in
 * 30s windows, so total is the ffprobe/measured duration rounded up to
 * windows. All inputs are the real source duration measured in the browser.
 */
export function transcribeUnits(
  durationSec: number,
  progressPct: number,
): StageUnits {
  const total = Math.ceil(durationSec / 30) * 30;
  const done = Math.min(total, Math.round(total * Math.max(0, progressPct) / 100 / 30) * 30);
  return {
    unitLabel: "s audio",
    unitsTotal: total,
    unitsDone: done,
    detail: `${fmtMin(done)} of ${fmtMin(total)} processed`,
  };
}

/** Finding stage units: transcript chunks sent to the LLM. */
export function findUnits(
  durationSec: number,
  progressPct: number,
): StageUnits {
  // ~90s chunks of transcript go to the model, streamed as they're ready
  const total = Math.max(3, Math.ceil(durationSec / 90));
  const done = Math.min(total, Math.ceil(total * Math.max(0, progressPct) / 100));
  return {
    unitLabel: "chunks",
    unitsTotal: total,
    unitsDone: done,
    detail: `${done} / ${total} chunks scored`,
  };
}

/** Render stage units: REAL count of finished clips. */
export function renderUnits(doneCount: number, total: number): StageUnits {
  const t = Math.max(1, total);
  return {
    unitLabel: "clips",
    unitsTotal: t,
    unitsDone: Math.min(t, doneCount),
    detail: `${Math.min(t, doneCount)} / ${t} clips encoded`,
  };
}

/**
 * Hard per-stage timeouts. These exist to convert hangs into readable
 * failures — a job must NEVER sit in a stage past its limit.
 */
export const STAGE_TIMEOUT_MS: Record<string, number> = {
  grabbing: 10 * 60 * 1000, // download 10 min
  transcribing: 15 * 60 * 1000, // generous for long videos on CPU whisper
  finding: 3 * 60 * 1000, // LLM scoring of ~7 chunks
  rendering: 45 * 60 * 1000, // ffmpeg.wasm across all clips
};

export function transcribeTimeoutMs(durationSec: number): number {
  return Math.max(2 * 60 * 1000, 2 * durationSec * 1000);
}

/** Pipeline stage order (used for total-ETA estimates after the current one). */
export const STAGE_ORDER = [
  "grabbing",
  "transcribing",
  "finding",
  "rendering",
] as const;

/** Median of a numeric sample list (null when empty) — learned stage speeds. */
export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function fmtMin(sec: number): string {
  if (sec >= 60) {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return s === 0 ? `${m} min` : `${m}m ${s}s`;
  }
  return `${Math.round(sec)}s`;
}

// ---------------------------------------------------------------------------
// transcript segmentation + AI detection prompt/response
// ---------------------------------------------------------------------------

/** Group words into ~90s chunks for LLM scoring (the real pipeline shape). */
export function segmentTranscript(
  words: TranscriptWord[],
  targetSec = 90,
): { start: number; end: number; text: string }[] {
  const segments: { start: number; end: number; text: string }[] = [];
  let current: TranscriptWord[] = [];
  let start = words[0]?.t0 ?? 0;
  const flush = () => {
    if (current.length === 0) return;
    segments.push({
      start,
      end: current[current.length - 1].t1,
      text: current.map((w) => w.text).join(" ").slice(0, 1200),
    });
    current = [];
  };
  for (const w of words) {
    if (current.length > 0 && w.t0 - start >= targetSec) {
      flush();
      start = w.t0;
    }
    current.push(w);
  }
  flush();
  return segments;
}

/** Build the clip-detection prompt for the LLM. */
export function buildClipDetectionPrompt(
  segments: { start: number; end: number; text: string }[],
  durationSec: number,
): string {
  const lines = segments
    .map((s, i) => `[${s.start.toFixed(1)}s-${s.end.toFixed(1)}s] ${s.text}`)
    .join("\n");
  return [
    "You are a short-form video editor. Below is a word-level transcript of a",
    `${Math.round(durationSec)} second video with timestamps. Pick the 5-12 most`, 
    "viral-worthy moments for TikTok/Shorts/Reels (strong hook, emotion,",
    "payoff, quotable lines).",
    "",
    "Rules:",
    "- Each clip must be 15-75 seconds long.",
    "- start/end are seconds from the beginning of the video, aligned to the",
    "  timestamps you see.",
    "- score is your virality estimate 0-100 (be discriminating).",
    "- title: max 6 words, punchy, no quotes.",
    "- hook_text: the single most gripping line inside the clip.",
    "- hashtags: 3-5 lowercase tags without the # symbol.",
    "",
    "Respond with ONLY a JSON array, no markdown fences, no commentary:",
    '[{"start":12.5,"end":48.0,"score":91,"title":"...","hook_text":"...","hashtags":["x","y"]}]',
    "",
    "TRANSCRIPT:",
    lines,
  ].join("\n");
}

/** Extract the first JSON array from an LLM response (handles fences/prose). */
function extractJsonArray(text: string): unknown[] | null {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenced !== null ? fenced[1] : text;
  const begin = candidate.indexOf("[");
  const end = candidate.lastIndexOf("]");
  if (begin === -1 || end === -1 || end <= begin) return null;
  try {
    const parsed: unknown = JSON.parse(candidate.slice(begin, end + 1));
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function titleFromWords(words: string[]): string {
  const t = words.slice(0, 6).join(" ").replace(/["']/g, "").trim();
  return t.length >= 8 ? t : words.slice(0, 9).join(" ").trim() || "Untitled moment";
}

function hashtagsFrom(text: string): string[] {
  const stop = new Set([
    "this","that","with","from","have","just","like","they","them","then",
    "what","when","your","about","would","could","there","their","going",
    "really","because","actually","think","know","want","things","gonna",
  ]);
  const counts = new Map<string, number>();
  for (const raw of text.toLowerCase().split(/[^a-z0-9']+/)) {
    const w = raw.replace(/'/g, "");
    if (w.length < 4 || stop.has(w)) continue;
    counts.set(w, (counts.get(w) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([w]) => w);
}

/**
 * Parse the LLM's clip list, snapping windows to real word boundaries so
 * captions never start mid-word. Returns [] when the output is unusable —
 * callers must NOT silently fall back to fake scoring.
 */
export function parseClipDetectionResponse(
  text: string,
  words: TranscriptWord[],
  maxClips: number,
): ClipCandidateOutput[] {
  const arr = extractJsonArray(text);
  if (arr === null) return [];

  const out: ClipCandidateOutput[] = [];
  for (const raw of arr) {
    if (raw === null || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    const start = typeof r.start === "number" ? r.start : Number.NaN;
    const end = typeof r.end === "number" ? r.end : Number.NaN;
    const score = typeof r.score === "number" ? r.score : Number.NaN;
    if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(score)) continue;
    if (end - start < 8) continue;

    // snap to word boundaries inside the window
    const inside = words.filter((w) => w.t1 > start && w.t0 < end);
    if (inside.length === 0) continue;
    const wStart = inside[0];
    const wEnd = inside[inside.length - 1];
    const clipped: TranscriptWord[] = inside.filter(
      (w) => w.t1 - wStart.t0 <= 75 && w.t1 <= wEnd.t1,
    );
    const span = clipped.length > 0 ? clipped : inside;
    const transcript = span.map((w) => w.text).join(" ");
    if (transcript.trim().length < 20) continue;

    const hookRaw = typeof r.hook_text === "string" ? r.hook_text : "";
    const titleRaw = typeof r.title === "string" ? r.title : "";
    const tagsRaw = Array.isArray(r.hashtags)
      ? r.hashtags.filter((t): t is string => typeof t === "string")
      : [];

    out.push({
      start: Math.max(0, wStart.t0),
      end: span[span.length - 1].t1,
      score: Math.round(clamp(score, 0, 100)),
      title: titleRaw.trim().slice(0, 60) || titleFromWords(transcript.split(" ")),
      hookText: hookRaw.trim().slice(0, 160) || transcript.slice(0, 140),
      transcript,
      hashtags: (tagsRaw.length >= 2 ? tagsRaw : hashtagsFrom(transcript)).slice(0, 5),
    });
  }

  out.sort((a, b) => b.score - a.score);
  return out.slice(0, maxClips);
}

// ---------------------------------------------------------------------------
// karaoke captions from REAL word timings
// ---------------------------------------------------------------------------

/**
 * Build 2-3-word caption cues from Whisper word timings, relative to clip
 * start (the rendered clip's local timeline). These are the cues the ASS
 * subtitle burn-in and the SRT export both use — one source of truth.
 */
export function karaokeCues(
  words: TranscriptWord[],
  clipStart: number,
  clipEnd: number,
): CaptionCue[] {
  const inClip = words.filter((w) => w.t1 > clipStart && w.t0 < clipEnd);
  const cues: CaptionCue[] = [];
  let group: TranscriptWord[] = [];
  const flush = () => {
    if (group.length === 0) return;
    cues.push({
      start: Math.max(0, group[0].t0 - clipStart),
      end: Math.max(0.3, group[group.length - 1].t1 - clipStart),
      text: group.map((w) => w.text).join(" ").toUpperCase(),
    });
    group = [];
  };
  for (const w of inClip) {
    group.push(w);
    const gapAheadNeedsBreak = w.t1 - clipStart > 0 && group.length >= 3;
    if (gapAheadNeedsBreak) flush();
  }
  flush();
  return cues;
}

/** 00:01:23,500 (SRT timecode) */
export function fmtSrtTime(sec: number): string {
  const ms = Math.max(0, Math.round(sec * 1000));
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const milli = ms % 1000;
  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(milli, 3)}`;
}

/** SRT body for a cue list (used by exports and captions.txt). */
export function buildSrt(cues: CaptionCue[]): string {
  return cues
    .map((c, i) => `${i + 1}\n${fmtSrtTime(c.start)} --> ${fmtSrtTime(c.end)}\n${c.text}\n`)
    .join("\n");
}

/**
 * ASS subtitle content for the ffmpeg burn-in: bold white text, thick
 * #39FF14 green outline, lower-third position, one cue at a time (karaoke
 * style). The pop/scale animation is applied by libass \t transforms.
 */
export function buildAssCues(
  cues: CaptionCue[],
  opts: { playResX: number; playResY: number } = { playResX: 1080, playResY: 1920 },
): string {
  const fmt = (t: number) => {
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60);
    const s = t % 60;
    return `${h}:${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
  };
  const events = cues
    .map((c) => {
      // pop-in: scale 62 -> 100 over the first 120ms of each cue
      const fx = `{\\fscx62\\fscy62\\t(0,120,\\fscx100\\fscy100)}`;
      return `Dialogue: 0,${fmt(c.start)},${fmt(c.end)},Pop,,0,0,0,,${fx}${c.text}`;
    })
    .join("\n");
  return `[Script Info]
ScriptType: v4.00+
PlayResX: ${opts.playResX}
PlayResY: ${opts.playResY}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,Arial Black,78,&H00FFFFFF,&H00FFFFFF,&H0000FF39,&H96000000,-1,0,0,0,100,100,1,0,1,9,3,2,80,80,320,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
${events}
`;
}
