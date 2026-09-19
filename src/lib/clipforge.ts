// Shared ClipForge client library. Clip data lives in Convex; these helpers
// generate per-clip preview metadata (scene labels, gradient frames, SRT
// timecodes) deterministically from the clip id so previews are stable.

import type { Doc } from "@/convex/_generated/dataModel";

export type JobDoc = Doc<"jobs">;
export type ClipDoc = Doc<"clips">;

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
  grabbing: "Grabbing video",
  transcribing: "Transcribing",
  finding: "Finding clips",
  rendering: "Rendering 9:16",
  complete: "Complete",
  failed: "Failed",
};

export const STAGE_LOADING: Record<string, string> = {
  grabbing: "Clippy is grabbing the video...",
  transcribing: "Whisper is transcribing word timings...",
  finding: "Scoring viral moments...",
  rendering: "Rendering 9:16 crops with captions...",
};

// ---------------------------------------------------------------------------
// deterministic hashing for stable preview metadata
// ---------------------------------------------------------------------------

export function hash(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

const SCENE_LABELS = [
  "Face-forward monologue",
  "Desk setup wide shot",
  "Screen share overlay",
  "Two-shot interview",
  "Camera B close-up",
  "Whiteboard segment",
  "Walk-and-talk",
  "POV hands-on shot",
];

const SCENE_GRADIENTS = [
  "from-[#123318] via-[#0b1d10] to-black",
  "from-[#122b2e] via-[#0a1a1c] to-black",
  "from-[#1c2a12] via-[#101c0a] to-black",
  "from-[#241a2a] via-[#120c16] to-black",
  "from-[#12283a] via-[#0a1622] to-black",
  "from-[#2a2312] via-[#1a150a] to-black",
  "from-[#2a1218] via-[#160a0c] to-black",
  "from-[#132230] via-[#0b141c] to-black",
];

const SCENE_TINTS = [
  "#39FF14",
  "#7CFFB2",
  "#9BFF6B",
  "#54E38A",
  "#C6FF00",
  "#39FF14",
  "#B0FF57",
  "#54E38A",
];

export interface SceneSpec {
  label: string;
  gradient: string;
  tint: string;
}

/** 3-4 deterministic scenes per clip for the simulated 9:16 preview. */
export function clipScenes(clipId: string): SceneSpec[] {
  const count = 3 + (hash(clipId) % 2);
  return Array.from({ length: count }, (_, i) => {
    const h = (hash(clipId + ":" + i) + i) % SCENE_LABELS.length;
    return {
      label: SCENE_LABELS[h],
      gradient: SCENE_GRADIENTS[h],
      tint: SCENE_TINTS[h],
    };
  });
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
