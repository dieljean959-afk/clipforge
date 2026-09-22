// Typed REST client for the ClipForge LOCAL backend (FastAPI, same machine).
// Every call goes to the local API process — no cloud, no broker.

import type { ClipStage } from "@/lib/clipforge";

export interface JobDto {
  _id: string;
  sourceType: "url" | "upload";
  source: string;
  title?: string | null;
  isUpload: boolean;
  stage: ClipStage;
  progress: number;
  progressDetail?: string | null;
  unitLabel?: string | null;
  unitsDone?: number | null;
  unitsTotal?: number | null;
  stageEtaSec?: number | null;
  totalEtaSec?: number | null;
  lastProgressAt?: number;
  slow?: boolean;
  stageStartedAt?: number;
  logs: string[];
  error?: string | null;
  errorDetail?: string | null;
  durationSec?: number | null;
  clipCount?: number | null;
  totalClipsSec?: number | null;
  detectedSource?: string | null;
  downloadSource?: string | null;
  transcriptWordCount?: number | null;
  captionsTxt?: string | null;
  createdAt: number;
  completedAt?: number | null;
}

export interface ClipDto {
  _id: string;
  jobId: string;
  order: number;
  start: number;
  end: number;
  score: number;
  title: string;
  hookText: string;
  transcript: string;
  hashtags: string[];
  captionCues: { start: number; end: number; text: string }[];
  silenceCuts?: { start: number; end: number }[] | null;
  aspect: string;
  videoPath: string | null;
  thumbPath: string | null;
  mediaBytes?: number | null;
  mediaWidth?: number | null;
  mediaHeight?: number | null;
  mediaDurationSec?: number | null;
  mediaError?: string | null;
  renderState: "pending" | "running" | "done" | "failed";
  renderMs?: number | null;
}

export interface HealthDto {
  ok: boolean;
  version: string;
  worker: { online: boolean; currentJob: string | null; queueDepth: number };
  warmup: { stage: string; detail: string; startedAt: number | null };
  llm: { provider: string; keyLoaded: boolean };
  whisper: { model: string; compute: string };
}

export interface VerifyClipResult {
  clipId: string;
  order?: number;
  title?: string;
  valid: boolean;
  reason: string;
  durationSec?: number;
  width?: number;
  height?: number;
  bytes?: number;
}

export interface VerifyClipsDto {
  total: number;
  valid: number;
  invalid: number;
  clips: VerifyClipResult[];
}

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep default
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

/** Request that returns a raw Blob (file downloads). */
async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep default
    }
    throw new Error(detail);
  }
  return await res.blob();
}

export const api = {
  health: () => request<HealthDto>("/api/health"),
  listJobs: () => request<JobDto[]>("/api/jobs"),
  getJob: (jobId: string) => request<JobDto>(`/api/jobs/${jobId}`),
  createJob: (source: string, sourceType: "url" | "upload" = "url") =>
    request<{ jobId: string }>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sourceType, source }),
    }),
  uploadJob: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<{ jobId: string; file: string; bytes: number }>("/api/jobs/upload", {
      method: "POST",
      body: fd,
    });
  },
  deleteJob: (jobId: string) =>
    request<{ deleted: boolean }>(`/api/jobs/${jobId}`, { method: "DELETE" }),
  retryJob: (jobId: string) =>
    request<{ queued: boolean }>(`/api/jobs/${jobId}/retry`, { method: "POST" }),
  redetect: (jobId: string) =>
    request<{ queued: boolean }>(`/api/jobs/${jobId}/redetect`, { method: "POST" }),
  verifyClips: (jobId: string) =>
    request<VerifyClipsDto>(`/api/dev/verify-clips?jobId=${encodeURIComponent(jobId)}`),
  jobLogs: (jobId: string) => request<string[]>(`/api/jobs/${jobId}/logs`),
  listClips: (jobId: string) => request<ClipDto[]>(`/api/clips?jobId=${encodeURIComponent(jobId)}`),
  exportZip: (clipIds: string[]) =>
    requestBlob("/api/export/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clipIds }),
    }),
  clipVideoUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/video`,
  clipThumbUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/thumb`,
  clipDownloadUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/download`,
  clipSrtUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/srt`,
  jobCaptionsUrl: (jobId: string) => `${BASE}/api/jobs/${jobId}/captions.txt`,
};
