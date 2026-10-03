// Shared ClipForge client library. Formatting helpers and stage metadata.
// NOTE: the old "scene preview" generator (invented labels/gradients) was
// deleted on purpose — previews are real <video> elements of rendered MP4s.

import type { ClipDto, JobDto } from "@/lib/api";

export type JobDoc = JobDto;
export type ClipDoc = ClipDto;

export type ClipStage =
  | "queued"
  | "grabbing"
  | "transcribing"
  | "finding"
  | "rendering"
  | "complete"
  | "failed"
  /** A user-requested stop at a safe boundary. Not a failure: resume picks
      the job up from the same stage, reusing completed work. */
  | "paused";

export const STAGES: ClipStage[] = [
  "queued",
  "grabbing",
  "transcribing",
  "finding",
  "rendering",
  "complete",
];

export const STAGE_LABELS: Record<ClipStage, string> = {
  queued: "Queued",
  grabbing: "Fetching video",
  transcribing: "Transcribing",
  finding: "AI clip detection",
  rendering: "Rendering 9:16",
  complete: "Complete",
  failed: "Failed",
  paused: "Paused",
};

// Generic, provider-neutral fallbacks. These used to name "Gemini" while the
// configured backend was deepseek, which is how a billing problem on one
// provider came to look like an outage on another. The card normally renders
// the backend's own `progressDetail`, which names the real provider; this map
// is only a fallback, so it must not assert a provider it cannot know.
export const STAGE_LOADING: Record<string, string> = {
  grabbing: "Clippy is grabbing the video...",
  transcribing: "Whisper is transcribing word timings...",
  finding: "AI is scoring viral moments...",
  rendering: "Rendering 9:16 crops with captions...",
};

// ---------------------------------------------------------------------------
// whole-pipeline status
// ---------------------------------------------------------------------------

/** The real stages a job walks through, in order (queued is not a stage). */
export const PIPELINE_STAGES: ClipStage[] = [
  "grabbing",
  "transcribing",
  "finding",
  "rendering",
];

export type OverallState = "queued" | "running" | "done" | "failed" | "paused";

export interface OverallStatus {
  state: OverallState;
  /** 1-based index of the active stage, or the total when finished. */
  stageNumber: number;
  total: number;
  /** 0-100 across the WHOLE pipeline, not just the active stage. */
  percent: number;
  /** Plain-language headline, e.g. "Stage 3 of 4 — AI clip detection". */
  label: string;
}

/**
 * One unambiguous answer to "is this job running, done, or broken?".
 *
 * Deliberately separate from the per-stage stepper: that one answers "which
 * step am I on", this answers "should I still be waiting, or can I go use the
 * clips?".
 *
 * The backend derives `status` in ONE place (app/jobstatus.py) and strips
 * every running-only field off a terminal job, so this reads THAT rather than
 * re-deriving from `stage` + `progress`. The stage/progress fallback stays for
 * a job that predates the field. `done` is never inferred from progress, so a
 * job at 99% cannot read as finished, and a failed job cannot read as running
 * just because it still carries a stale progress number.
 */
export function overallStatus(job: JobDoc): OverallStatus {
  const total = PIPELINE_STAGES.length;
  const status = job.status ?? job.stage;
  if (status === "complete") {
    return {
      state: "done",
      stageNumber: total,
      total,
      percent: 100,
      label: "Done — clips ready",
    };
  }
  if (status === "failed") {
    // `stage` is "failed", so the stage that actually broke lives in
    // `failedStage` (recorded server-side before `stage` is overwritten), or
    // in the backend's own `stageLabel`.
    const raw = job.stageLabel ?? job.failedStage ?? null;
    const where = raw
      ? (STAGE_LABELS[raw as ClipStage] ?? raw)
      : null;
    const idx = PIPELINE_STAGES.indexOf(raw as ClipStage);
    return {
      state: "failed",
      stageNumber: idx >= 0 ? idx + 1 : total,
      total,
      percent: 0,
      label: where ? `Failed during ${where}` : "Failed",
    };
  }
  if (status === "paused") {
    // A pause is a deliberate stop, not a failure and not "still running".
    // It must never read as running, or the card shows a live spinner for a
    // job that is deliberately idle until the user resumes it.
    return {
      state: "paused",
      stageNumber: 0,
      total,
      percent: Math.min(99, Math.round(job.progress ?? 0)),
      label: job.stageLabel
        ? `Paused at ${job.stageLabel}`
        : job.pausedStage
          ? `Paused at ${STAGE_LABELS[job.pausedStage as ClipStage] ?? job.pausedStage}`
          : "Paused by user",
    };
  }
  if (status === "queued") {
    return {
      state: "queued",
      stageNumber: 0,
      total,
      percent: 0,
      label: "Queued — waiting for a worker",
    };
  }
  const live = job.activeStage ?? job.stage;
  const idx = PIPELINE_STAGES.indexOf(live as ClipStage);
  const stageNumber = idx === -1 ? 1 : idx + 1;
  const within = Math.min(100, Math.max(0, job.progress ?? 0));
  const percent = Math.round(((stageNumber - 1 + within / 100) / total) * 100);
  return {
    state: "running",
    stageNumber,
    total,
    percent: Math.min(99, percent),
    label: `Stage ${stageNumber} of ${total} — ${STAGE_LABELS[live as ClipStage] ?? live}`,
  };
}

// ---------------------------------------------------------------------------
// formatting
// ---------------------------------------------------------------------------

/** 12.3s */
export function fmtDuration(sec: number): string {
  if (!Number.isFinite(sec)) return "—";
  return `${(Math.round(sec * 10) / 10).toFixed(1)}s`;
}

/** 1:23 */
export function fmtClock(sec: number): string {
  if (!Number.isFinite(sec)) return "0:00";
  const total = Math.floor(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * A DURATION as mm:ss, or h:mm:ss past an hour. Rounded to the nearest second.
 *
 * Exists because `${job.totalClipsSec}s of clips` rendered the raw float:
 * "932.8100000000011s of clips". Durations are sums of clip lengths, so they
 * are never whole — but the sub-second remainder is noise, and float error in
 * the sum is noise on top of that. Rounding first and formatting second is the
 * only order that hides both.
 */
export function fmtDurationLong(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return "0:00";
  const total = Math.max(0, Math.round(sec));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
  return h > 0
    ? `${h}:${mm}:${String(s).padStart(2, "0")}`
    : `${mm}:${String(s).padStart(2, "0")}`;
}

/**
 * Human ETA: "~2 min left" / "~45s left". Pairs with the per-stage unit line
 * ("4 of 11 min processed") in the job card.
 */
export function fmtEta(sec: number): string {
  if (!Number.isFinite(sec) || sec <= 0) return "any moment";
  if (sec < 90) return `${Math.max(1, Math.round(sec))}s`;
  return `${Math.round(sec / 60)} min`;
}

/**
 * Silence thresholds, in ms.
 *
 * DISTINCT on purpose, and deliberately far apart:
 *
 * - QUIET is the early "is anything happening?" signal. It is only a hint, not
 *   a verdict: a single long ffmpeg encode legitimately produces no progress
 *   line for a while, so this must not look like an error.
 * - STALL is where the server-side watchdog has given up and killed the worker.
 *   It mirrors CLIPFORGE_STALL_TIMEOUT_SEC.
 */
export const QUIET_THRESHOLD_MS = 75_000;
export const STALL_THRESHOLD_MS = 90_000;

/** True while a job is doing work we can meaningfully watch. */
export function isActiveJob(job: JobDoc): boolean {
  if (job.isTerminal !== undefined) return !job.isTerminal && job.status !== "queued";
  return job.stage !== "complete" && job.stage !== "failed" && job.stage !== "paused";
}

/** ms since the job last reported progress (falls back through its clocks). */
export function silenceMs(job: JobDoc, now: number): number {
  const last = job.lastProgressAt ?? job.stageStartedAt ?? job.createdAt;
  if (typeof last !== "number") return 0;
  return Math.max(0, now - last);
}

/** "just now" / "12s ago" / "4m ago" — the human-readable liveness stamp. */
export function fmtAgo(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "just now";
  const s = Math.floor(ms / 1000);
  if (s < 3) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
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

/** "Mar 14, 3:42 PM" */
export function fmtDate(ts: number): string {
  return new Date(ts).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Short display label for a job source. */
export function sourceLabel(job: Pick<JobDoc, "source" | "isUpload">): string {
  if (job.isUpload) return job.source;
  try {
    const url = new URL(job.source);
    return url.hostname.replace(/^www\./, "") + (url.pathname !== "/" ? url.pathname : "");
  } catch {
    return job.source;
  }
}

/**
 * The platform/video ID from a source URL, e.g. "LsNjFCfQKis".
 * Returns "" when the URL carries no recognisable ID, so callers can fall back
 * rather than render an empty label.
 */
export function sourceId(job: Pick<JobDoc, "source" | "isUpload">): string {
  if (job.isUpload) return "";
  const id = (() => {
    try {
      const url = new URL(job.source);
      const host = url.hostname.replace(/^www\./, "");
      // youtu.be/<id>, /watch?v=<id>, /embed/<id>, /shorts/<id>
      if (host === "youtu.be") return url.pathname.slice(1).split("/")[0] ?? "";
      const v = url.searchParams.get("v");
      if (v) return v;
      const m = /^\/(?:embed|shorts|v|live)\/([^/?#]+)/.exec(url.pathname);
      return m?.[1] ?? "";
    } catch {
      return "";
    }
  })();
  // Platform IDs are base64-ish and short; anything else is a path, not an ID.
  return /^[A-Za-z0-9_-]{6,20}$/.test(id) ? id : "";
}

/**
 * What a batch pill should say: the VIDEO TITLE when we have one, else the
 * platform + ID.
 *
 * "youtube.com/watch" told the user nothing — three batches from three
 * different videos all read "youtube.com/watch · 23 clips", so they could not
 * be told apart at a glance. The title (or at minimum the video ID) is what
 * identifies a batch; the date (rendered separately) disambiguates re-runs of
 * the same video.
 */
export function batchLabel(job: Pick<JobDoc, "source" | "isUpload" | "title">): string {
  const title = (job.title ?? "").trim();
  if (title) return title.length > 60 ? `${title.slice(0, 59)}…` : title;
  const id = sourceId(job);
  if (id) {
    try {
      return `${new URL(job.source).hostname.replace(/^www\./, "")} · ${id}`;
    } catch {
      return id;
    }
  }
  // No title and no ID: the hostname alone is the least-bad identifier.
  return sourceLabel(job);
}

export function hostOf(source: string): string {
  try {
    return new URL(source).hostname.replace(/^www\./, "");
  } catch {
    return "upload";
  }
}

/** Score band for badge styling. */
export function scoreBand(score: number): "hot" | "warm" | "mild" {
  if (score >= 85) return "hot";
  if (score >= 70) return "warm";
  return "mild";
}

// --------------------------------------------------------------------------
// Post-caption shape. Mirrors app/caption_format.py -- the clipboard text, the
// text shown in the composer preview, and the text a post submits must all be
// identical, and the Python side owns the same rules for clips rendered there.
// --------------------------------------------------------------------------

/** Discovery tags earn reach. Kept in sync with DISCOVERY_TAGS in Python. */
export const DISCOVERY_TAGS = ["#fyp", "#foryou", "#viral"] as const;

const MAX_TAGS = 7;
const MIN_TAGS = 5;
const MAX_DISCOVERY = 3;
const MIN_DISCOVERY = 2;
export const MAX_CAPTION_CHARS = 150;

/** Normalise one tag to a bare lowercase `#word`, or "" if it carries no tag. */
export function cleanTag(raw: unknown): string {
  const text = String(raw ?? "").trim().replace(/^#+/, "").toLowerCase();
  const cleaned = text.replace(/[^0-9a-z]+/g, "");
  return cleaned ? `#${cleaned}` : "";
}

/**
 * Dedupe, enforce the 2-3 discovery / 2-4 content mix, cap at 7.
 *
 * Relevance over volume: a shortfall is topped up from the discovery tags only,
 * never with generic filler like #love or #explore that would suit any video.
 */
export function normalizeHashtags(tags: unknown): string[] {
  if (!Array.isArray(tags)) return [...DISCOVERY_TAGS];

  const seen = new Set<string>();
  const content: string[] = [];
  const discovery: string[] = [];
  for (const raw of tags) {
    const tag = cleanTag(raw);
    if (!tag || seen.has(tag)) continue;
    seen.add(tag);
    (DISCOVERY_TAGS as readonly string[]).includes(tag)
      ? discovery.push(tag)
      : content.push(tag);
  }

  if (discovery.length > MAX_DISCOVERY) {
    content.unshift(...discovery.splice(MAX_DISCOVERY));
  }
  for (const tag of DISCOVERY_TAGS) {
    if (discovery.length >= MIN_DISCOVERY && discovery.length + content.length >= MIN_TAGS) {
      break;
    }
    if (seen.has(tag)) continue;
    seen.add(tag);
    discovery.push(tag);
  }
  return [...discovery, ...content].slice(0, MAX_TAGS);
}

/** Collapse to a single line, trimmed on a word boundary. */
export function shortenCaption(raw: unknown, limit = MAX_CAPTION_CHARS): string {
  const text = String(raw ?? "").replace(/\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return text.slice(0, limit).replace(/\s+\S*$/, "").replace(/[\s,.;:\-–—]+$/, "");
}
