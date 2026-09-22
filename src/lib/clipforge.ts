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
  | "failed";

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
};

export const STAGE_LOADING: Record<string, string> = {
  grabbing: "Clippy is grabbing the video...",
  transcribing: "Whisper is transcribing word timings...",
  finding: "Gemini is scoring viral moments...",
  rendering: "Rendering 9:16 crops with captions...",
};

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
 * Human ETA: "~2 min left" / "~45s left". Pairs with the per-stage unit line
 * ("4 of 11 min processed") in the job card.
 */
export function fmtEta(sec: number): string {
  if (!Number.isFinite(sec) || sec <= 0) return "any moment";
  if (sec < 90) return `${Math.max(1, Math.round(sec))}s`;
  return `${Math.round(sec / 60)} min`;
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
