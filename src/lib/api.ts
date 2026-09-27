// Typed REST client for the ClipForge backend (FastAPI, same origin).
// Every call uses relative /api URLs — same origin, no localhost, no IPs.

import type { ClipStage } from "@/lib/clipforge";

export type DetectionProvider = "groq" | "gemini" | "ollama";

/** Which backend scores the viral moments (NOT the ASR engine). */
export interface DetectionDto {
  provider: DetectionProvider;
  label: string;
  /** Plain-language speed/privacy tradeoff, shown BEFORE the user picks. */
  tradeoff: string;
  options: { value: DetectionProvider; label: string; tradeoff: string }[];
  models: Record<DetectionProvider, string>;
  /** The ASR engine, unchanged by this choice — surfaced to avoid confusion. */
  transcriptionEngine?: string;
  /** Ollama only: is the local server reachable with the model pulled? */
  ready?: boolean;
  detail?: string;
}

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
  /** How the ETA was derived: "measured" progress or an "rtf-prior". */
  etaSource?: string | null;
  /** 0-based position in the waiting queue; null once a worker picks it up. */
  queuePosition?: number | null;
  /** Server estimate of seconds until this queued job starts. null when it is
      next in line with an idle worker — "any moment" is honest there. */
  queueEtaSec?: number | null;
  /** Set while a user-requested pause is waiting for the next safe boundary. */
  pauseRequested?: boolean;
  /** Where the job stopped when paused, e.g. "finding" or "rendering (after clip 4 of 10)". */
  pausedStage?: string | null;
  pausedAt?: number | null;
  stageTimings?: { stage: string; sec: number }[];
  lastProgressAt?: number;
  slow?: boolean;
  stageStartedAt?: number;
  logs: string[];
  error?: string | null;
  errorDetail?: string | null;
  /** The stage that failed — `stage` is overwritten with "failed", so this
   *  is the only way to tell the user WHICH step broke. */
  failedStage?: string | null;
  durationSec?: number | null;
  clipCount?: number | null;
  totalClipsSec?: number | null;
  detectedSource?: string | null;
  downloadSource?: string | null;
  transcriptWordCount?: number | null;
  captionsTxt?: string | null;
  options?: {
    minClipSec?: number | null;
    maxClipSec?: number | null;
    maxClips?: number | null;
    captionPreset?: string | null;
  } | null;
  createdAt: number;
  completedAt?: number | null;
  /**
   * Set when the host killed the worker (SIGTERM/SIGKILL) mid-job, e.g. a
   * container restart. The job is auto-resumed rather than failed, so this is
   * normally transient and does not mean the job is lost.
   */
  interrupted?: boolean | null;
  interruptReason?: string | null;
  /** How many host restarts this job has survived; bounded server-side. */
  interruptCount?: number | null;
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
  captionCues: {
    start: number;
    end: number;
    text: string;
    /** True when the model was unsure and the text is a [muttering]/[inaudible] marker. */
    uncertain?: boolean;
  }[];
  /** Cues replaced by a placeholder because the model was not confident. */
  uncertainCues?: number | null;
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
  /** Sidecar caption tracks written by [translate], one record per language. */
  translationTracks?: TranslationTrack[] | null;
}

export interface TranslationTrack {
  /** ISO-639-1 code, e.g. "fr". */
  lang: string;
  /** True for the original transcript, which is never machine-translated. */
  source?: boolean;
  /** True when the track was reused from a previous run. */
  cached?: boolean;
  /** Present when this language failed; the other languages still wrote. */
  error?: string;
  /** Format -> filename, e.g. { srt: "clip01_fr.srt", vtt: "clip01_fr.vtt" }. */
  files?: Record<string, string>;
}

export interface SystemStatsDto {
  cpuPercent: number | null;
  cores: number | null;
  memTotalBytes: number | null;
  memUsedBytes: number | null;
  load1: number | null;
  pid: number | null;
}

export interface DiskStatsDto {
  freeBytes: number | null;
  totalBytes: number | null;
}

export interface KeyStatusDto {
  keyLoaded: boolean;
  source: string;
  problem: string | null;
  lastCheckedAt: number;
  aliases: string[];
}

export interface AsrDto {
  engine: "whisperx" | "faster-whisper" | string;
  device: string;
  computeType: string;
  model: string;
  batchSize: number;
  batched: boolean;
  detail: string;
  estimatedRtf: number;
  rtfSource: string;
  rtfHistory?: Record<
    string,
    { rtf: number; lastRtf?: number; samples: number; engine?: string; device?: string; model?: string }
  >;
}

export interface HealthDto {
  ok: boolean;
  version: string;
  worker: {
    online: boolean;
    currentJob: string | null;
    queueDepth: number;
    concurrency?: number;
    activeCount?: number;
    activeJobs?: string[];
    queuedJobs?: string[];
  };
  warmup: { stage: string; detail: string; startedAt: number | null };
  llm: KeyStatusDto;
  /** Live GROQ_API_KEY state — the key for the DEFAULT detection backend. */
  llmGroq?: KeyStatusDto;
  detection?: DetectionDto;
  // Resolved engine/model/compute (already host-limit adjusted). The raw
  // config.toml request, when it differs, is `configuredModel`.
  whisper: {
    model: string;
    compute: string;
    engine?: string;
    configuredModel?: string;
    isWhisperx?: boolean;
  };
  asr?: AsrDto;
  system: SystemStatsDto;
  disk: DiskStatsDto;
  tailnetIp: string | null;
  stallTimeoutSec?: number | null;
  limits: { maxSourceMin: number; retentionHours: number };
  dispatch:
    | { enabled: false }
    | {
        enabled: true;
        workerUrl: string;
        remote: { online: boolean; detail: string; queueDepth?: number } | null;
      };
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

export interface DoctorCheck {
  name: string;
  status: "ok" | "warn" | "fail";
  detail: string;
  fix: string;
}

export interface DoctorReport {
  ok: boolean;
  summary: { fail: number; warn: number; ok: number };
  checks: DoctorCheck[];
}

export interface CaptionPresetDto {
  key: string;
  label: string;
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

export interface SocialPlatformDto {
  slug: SocialSlug;
  label: string;
  /** needs_app | not_connected | connected -- what the UI should offer next. */
  status: "needs_app" | "not_connected" | "connected";
  /** Plain-language reason for `status`, safe to show as-is. */
  reason: string;
  accountName: string;
  connected: boolean;
  /** none | payment | audit | app_review -- how hard approval actually is. */
  gate: "none" | "payment" | "audit" | "app_review";
  headline: string;
  detail: string;
  docs: string;
}

export type SocialSlug = "youtube" | "x" | "tiktok" | "instagram";

export interface PostResultDto {
  platform: SocialSlug;
  id: string;
  url: string;
  privacy: string;
  /** TikTok only: an unaudited app can only publish privately. */
  audited?: boolean;
  note?: string;
  postedAt: number;
  captionLength: number;
}

export const api = {
  health: () => request<HealthDto>("/api/health"),
  doctor: () => request<DoctorReport>("/api/doctor"),
  captionPresets: () => request<CaptionPresetDto[]>("/api/caption-presets"),
  listJobs: () => request<JobDto[]>("/api/jobs"),
  getJob: (jobId: string) => request<JobDto>(`/api/jobs/${jobId}`),
  createJob: (
    source: string,
    sourceType: "url" | "upload" = "url",
    captionPreset?: string,
  ) =>
    request<{ jobId: string }>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sourceType, source, captionPreset }),
    }),
  uploadJob: (file: File, captionPreset?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    if (captionPreset) fd.append("captionPreset", captionPreset);
    return request<{ jobId: string; file: string; bytes: number }>("/api/jobs/upload", {
      method: "POST",
      body: fd,
    });
  },
  deleteJob: (jobId: string) =>
    request<{ deleted: boolean; cancelled: "queued" | "running" | "absent" }>(
      `/api/jobs/${jobId}`,
      { method: "DELETE" },
    ),
  retryJob: (jobId: string) =>
    request<{ queued: boolean }>(`/api/jobs/${jobId}/retry`, { method: "POST" }),
  /** Cooperative stop at the next safe boundary. "pausing" means the current
      unit of work (one chunk / one clip) is still finishing. */
  pauseJob: (jobId: string) =>
    request<{ state: "paused" | "pausing"; stage: string }>(`/api/jobs/${jobId}/pause`, {
      method: "POST",
    }),
  resumeJob: (jobId: string) =>
    request<{ resumed: boolean; queueDepth: number }>(`/api/jobs/${jobId}/resume`, {
      method: "POST",
    }),
  redetect: (jobId: string) =>
    request<{ queued: boolean }>(`/api/jobs/${jobId}/redetect`, { method: "POST" }),
  /**
   * Re-encode with fresh captions, reusing the cached transcript and cuts.
   * No re-transcription and no AI call — minutes of CPU, nothing else.
   */
  rerender: (jobId: string) =>
    request<{ queued: boolean; clipsReset: number }>(`/api/jobs/${jobId}/rerender`, {
      method: "POST",
    }),
  verifyClips: (jobId: string) =>
    request<VerifyClipsDto>(`/api/dev/verify-clips?jobId=${encodeURIComponent(jobId)}`),
  jobLogs: (jobId: string) => request<string[]>(`/api/jobs/${jobId}/logs`),
  /** Force the server to re-read env files and re-resolve the LLM keys. */
  reloadSecrets: () =>
    request<KeyStatusDto & { groq?: KeyStatusDto; activeProvider?: string }>(
      "/api/settings/reload-secrets",
      { method: "POST" },
    ),
  /** Save/replace GROQ_API_KEY in the backend's env file. */
  setGroqKey: (key: string) =>
    request<KeyStatusDto & { storedShape: { first6: string; last4: string; length: number } }>(
      "/api/settings/groq-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      },
    ),
  /** Delete GROQ_API_KEY from the backend's env file. */
  deleteGroqKey: () =>
    request<KeyStatusDto & { linesRemoved: number }>("/api/settings/groq-key", {
      method: "DELETE",
    }),
  /** Save/replace GEMINI_API_KEY in the backend's env file (missing Edit button). */
  setGeminiKey: (key: string) =>
    request<KeyStatusDto & { storedShape: { first6: string; last4: string; length: number } }>(
      "/api/settings/gemini-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      },
    ),
  /** Delete GEMINI_API_KEY from the backend's env file (missing Delete button). */
  deleteGeminiKey: () =>
    request<KeyStatusDto & { linesRemoved: number }>("/api/settings/gemini-key", { method: "DELETE" }),
  /** Switch the AI-detection backend. Transcription engine is untouched. */
  setDetectionBackend: (provider: DetectionProvider) =>
    request<DetectionDto>("/api/settings/detection", {
      method: "POST",
      body: JSON.stringify({ provider }),
    }),
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

  // --- "Post to..." (additive; none of this affects the download paths) ----
  socialAccounts: () =>
    request<{ platforms: SocialPlatformDto[] }>("/api/social/accounts"),
  /** Begin OAuth. Resolves to the provider's consent URL to open in a tab. */
  socialConnect: (slug: SocialSlug) =>
    request<{ authUrl: string; state: string }>(`/api/social/${slug}/connect`, {
      method: "POST",
    }),
  socialDisconnect: (slug: SocialSlug) =>
    request<{ disconnected: boolean; platforms: SocialPlatformDto[] }>(
      `/api/social/${slug}/disconnect`,
      { method: "POST" },
    ),
  postClip: (
    clipId: string,
    body: { platform: SocialSlug; caption?: string; title?: string; privacy?: string },
  ) =>
    request<PostResultDto>(`/api/clips/${clipId}/post`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
