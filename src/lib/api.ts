// Typed REST client for the ClipForge backend (FastAPI, same origin).
// Every call uses relative /api URLs — same origin, no localhost, no IPs.

import type { ClipStage } from "@/lib/clipmake";

// NVIDIA NIM is the primary hosted backend; Groq is the backup and shares the
// same OpenAI-compatible wire format. Order matters — it is the order the
// dashboard renders the backend picker in.
export type DetectionProvider = "deepseek" | "nvidia" | "groq" | "gemini" | "ollama";

/** One row of the per-stage time plan. */
export interface StagePlanEntry {
  stage: "fetch" | "transcribe" | "detect" | "reframe" | "render";
  label: string;
  /** Seconds still expected for this stage. */
  sec: number;
  /** How that number was arrived at, so a guess is never shown as a fact. */
  source: "live" | "measured" | "estimate";
  current: boolean;
}

/** The caption look knobs the dashboard can change. */
export interface CaptionStyleDto {
  /** Target burst length in words; a pause or the cap still ends a burst sooner. */
  words_per_cue: number;
  /** Hard cap on a burst. Always >= words_per_cue. */
  words_per_cue_max: number;
  /** A gap of at least this many seconds before the next word ends the burst. */
  pause_sec: number;
  /** "" means the preset's own accent colour. */
  highlight_color: string;
  /** Caption size as a multiple of the preset's own (0.8-1.4; bold_stack). */
  size_scale?: number;
  /** Which caption look preset this overrides resolve to. */
  presetKey?: string;
  /** Every preset the backend ships, in the order the picker shows them. */
  presetKeys?: string[];
  /** Human label of the resolved look. */
  label?: string;
  /** Twemoji overlay switch + how many chunks get one (low 15% / medium 35% / high 60%). */
  emojiEnabled?: boolean;
  emojiDensity?: "low" | "medium" | "high";
  /** Slider names whose value differs from the preset's own (drives the badge). */
  custom?: string[];
  /** Resolved look, so the preview draws the same text the renderer will. */
  fontSize?: number;
  outline?: number;
  pitch?: number;
  lines?: number;
  uppercase?: boolean;
  box?: boolean;
  colors?: string[];
  /** Resolved settings hash: the render cache key. */
  hash?: string;
  /** Which of these are pinned here rather than inherited from config.toml. */
  overrides: Record<string, number | string>;
  limits: Record<string, { min: number; max: number; step: number }>;
}

/** A partial save: an omitted key stops overriding that one value. */
export type CaptionStyleUpdate = Partial<
  Pick<
    CaptionStyleDto,
    | "words_per_cue"
    | "words_per_cue_max"
    | "pause_sec"
    | "highlight_color"
    | "size_scale"
    | "presetKey"
    | "emojiEnabled"
    | "emojiDensity"
  >
> & {
  /** Drop the slider overrides, keep the chosen preset ("Reset to preset"). */
  reset?: boolean;
};

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
  /** The backend the app falls back to when the primary's quota runs out. */
  fallbackProvider?: string | null;
}

export interface JobDto {
  _id: string;
  sourceType: "url" | "upload";
  source: string;
  title?: string | null;
  isUpload: boolean;
  /** Wall-clock creation time (ms), sent by the backend's job list/detail. */
  createdAt?: number;
  /** Clips actually rendered for this job (0 or absent while running). */
  clipCount?: number;
  /** Seconds of source video, probed server-side. */
  durationSec?: number | null;
  /** Seconds of rendered clip footage across all clips. */
  totalClipsSec?: number | null;
  /** Which provider's model detected the highlights, e.g. "deepseek". */
  detectedSource?: string | null;
  /** Where the source video was fetched from (yt-dlp profile, "direct", ...). */
  downloadSource?: string | null;
  /** Word count of the transcript (server-measured). */
  transcriptWordCount?: number | null;
  /** How it was transcribed when that differs from the default, e.g.
   *  "transcribed with base (low memory)". Null for a normal run. */
  transcriptLabel?: string | null;
  /** Peak RSS of the ASR child during transcription, in MB. */
  asrPeakRssMb?: number | null;
  /** Human-readable failure message. Present on failed jobs. */
  error?: string | null;
  /** Full traceback / detail behind `error`, shown collapsed. */
  errorDetail?: string | null;
  /** The stage the job died in (recorded before `stage` is overwritten). */
  failedStage?: string | null;
  /** True when the job was interrupted by a host/sandbox restart. */
  interrupted?: boolean;
  interruptCount?: number;
  interruptReason?: string | null;
  stage: ClipStage;
  /** Authoritative status from the backend's single derivation. Prefer this
   *  over inferring one from `stage` + `progress` — those two disagreeing is
   *  exactly what produced "Failed" beside "61% overall, running". */
  status: "queued" | "running" | "paused" | "failed" | "complete" | "deleting";
  /** True once no further progress is possible. When true, every running-only
   *  field (stagePlan, units, ETAs, queuePosition) is absent from the payload. */
  isTerminal: boolean;
  /** Present while a delete is being purged in the background. `error` is the
   *  last step that failed, kept so a stuck purge is visible, never silent. */
  deleting?: { error?: string | null; attempts?: number; removed?: number } | null;
  /** The stage a RUNNING job is on, or null. Never set for a dead job. */
  activeStage?: string | null;
  /** Human name of the stage this job is on, or died in. */
  stageLabel?: string | null;
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
  stageTimings?: { stage: string; sec: number; at?: number }[];
  /** Remaining stages, nearest first, each with its own estimate. */
  stagePlan?: StagePlanEntry[];
  /** Clips this source is worth; scaled from its length unless overridden. */
  clipTarget?: number;
  /** Why that number — the sentence the user sees in the job log. */
  clipTargetReason?: string;
  /** Candidates the detector found before the score floor. */
  candidatesDetected?: number;
  /** Candidates that cleared the score floor (before the clip cap). */
  clipsAboveScore?: number;
  /** Candidates dropped as not-a-highlight. */
  clipsBelowScore?: number;
  /** The score a candidate had to clear. */
  clipScoreFloor?: number;
  lastProgressAt?: number;
  slow?: boolean;
  stageStartedAt?: number;
  /** Present on GET /api/jobs/{id}; omitted from the list projection. */
  logs?: string[] | null;
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
  /** The emoji burned above each chunk (empty when none was chosen). */
  emojiTrack?: CaptionEmojiDto[] | null;
  /** "llm" when the model chose them, "fallback: keywords" when it did not. */
  emojiSource?: string | null;
  /** Every chunk's judgement, including the ones that got no emoji. */
  emojiTags?: CaptionEmojiTagDto[] | null;
  silenceCuts?: { start: number; end: number }[] | null;
  aspect: string;
  videoPath: string | null;
  thumbPath: string | null;
  mediaBytes?: number | null;
  mediaWidth?: number | null;
  mediaHeight?: number | null;
  mediaDurationSec?: number | null;
  mediaError?: string | null;
  fps?: number | null;
  /** The resolved caption look this clip was burned with (from the job). */
  captionSettings?: {
    presetKey?: string;
    fontSize?: number;
    outline?: number;
    pitch?: number;
    lines?: number;
    uppercase?: boolean;
    box?: boolean;
    colors?: string[];
    highlightColor?: string;
    positionY?: number;
    emojiEnabled?: boolean;
    emojiDensity?: string;
  } | null;
  renderState: "pending" | "running" | "done" | "failed";
  /** COVERAGE GUARD: true when this clip's captions miss speech. */
  needsReview?: boolean | null;
  /** "needs review: captions missing at 27.1-43.5 s" — never shown as ready. */
  captionWarning?: string | null;
  /** Speech vs caption numbers behind that flag (seconds, clip-relative). */
  captionCoverage?: {
    speechSec?: number;
    captionSec?: number;
    coveredFraction?: number;
    uncoveredSec?: number;
    uncovered?: number[][];
    lastSpeechSec?: number | null;
    lastCaptionSec?: number | null;
    passed?: boolean;
    needsReview?: boolean;
    reason?: string;
  } | null;
  renderMs?: number | null;
  translationTracks?: { lang: string; source?: boolean; cached?: boolean; error?: string; files?: Record<string, string> }[] | null;
  shareId?: string | null;
  shareState?: "missing" | "preparing" | "ready" | "failed";
  shareDetail?: string;
  shareReady?: boolean;
  shareUrl?: string | null;
  sharePageUrl?: string | null;
}

/** One emoji burned onto a clip, with the judgement behind it. */
export interface CaptionEmojiDto {
  i: number;
  emoji: string;
  emotion?: string;
  intensity?: number | null;
  confidence?: number | null;
  reason?: string;
  start: number;
  end: number;
  cueText?: string;
  /** True when a human picked it in the caption editor. */
  byHandle?: boolean;
}

/** The model's per-chunk judgement (emoji may be null for a skipped chunk). */
export interface CaptionEmojiTagDto {
  i: number;
  emotion: string;
  intensity: number;
  emoji: string | null;
  reason: string;
  confidence: number;
  sensitive: boolean;
  loudness_z?: number;
}

export interface EmojiWhitelistDto {
  source: string;
  emojis: { emoji: string; emotions: string[]; keywords: string[] }[];
}

export interface KeyStatusDto {
  provider: string;
  keyLoaded: boolean;
  source: string;
  problem: string | null;
  lastCheckedAt: number;
  aliases: string[];
}

export interface StorageKeysInput {
  endpoint?: string;
  bucket?: string;
  accessKey?: string;
  secretKey?: string;
  region?: string;
  prefix?: string;
}

/** A failed/empty job returned by GET /api/jobs/junk (or derived client-side). */
export interface JunkJobDto {
  _id: string;
  title: string;
  stage: string;
  createdAt?: number;
  reason: string;
}

/** Result of GET /api/dev/verify-clips for one job. */
export interface VerifyClipsDto {
  total: number;
  valid: number;
  invalid: number;
  clips: {
    clipId: string;
    order?: number;
    title?: string;
    valid: boolean;
    reason: string;
    durationSec?: number;
    width?: number;
    height?: number;
    bytes?: number;
  }[];
}

export interface HealthDto {
  ok: boolean;
  /** Names of the health blocks that degraded — present only when ok is false. */
  degraded?: string[];
  version: string;
  worker: {
    online: boolean;
    currentJob: string | null;
    queueDepth: number;
    concurrency?: number;
    activeCount?: number;
    stallTimeoutSec?: number | null;
  };
  /** Model warmup progress (loading_model / failed / skipped / ...). */
  warmup?: { stage: string; detail?: string } | null;
  /** The optional remote home-worker dispatch (Tailscale). */
  dispatch?: {
    enabled: boolean;
    workerUrl?: string | null;
    remote?: { online: boolean; detail?: string } | null;
  };
  /** AI highlight-detection backend state (probe=False: key presence only). */
  detection?: DetectionDto;
  /** The RESOLVED transcription engine, not what config.toml asks for. */
  whisper?: {
    model: string;
    compute: string;
    configuredModel?: string | null;
    engine?: string | null;
    isWhisperx?: boolean;
  };
  /** Real transcription facts: engine, RTF this host measured, host limits. */
  asr?: {
    engine?: string | null;
    model?: string | null;
    device?: string | null;
    detail?: string | null;
    estimatedRtf?: number | null;
    rtfSource?: string | null;
  };
  system?: {
    cpuPercent?: number | null;
    cores?: number | null;
    memUsedBytes?: number | null;
    memTotalBytes?: number | null;
    load1?: number | null;
  } | null;
  disk?: {
    freeBytes?: number | null;
    totalBytes?: number | null;
    path?: string | null;
    scratchMb?: number | null;
    cacheMb?: number | null;
    modelMb?: number | null;
    lastCleanedAt?: number | null;
    lastFreedMb?: number | null;
  } | null;
  /**
   * The real toolchain, read from the binaries on PATH per poll, plus the
   * folder rendered clips land in. Nulls mean "could not run it", which the
   * status card shows instead of pretending a version.
   */
  tools?: {
    ffmpeg?: string | null;
    ffprobe?: string | null;
    ytdlp?: string | null;
    mediaDir?: string | null;
    downloadsDir?: string | null;
  } | null;
  tailnetIp?: string | null;
  /** The yt-dlp cookie jar: exported when, how many cookies, and whether the
   *  server already considers it stale/broken. Absent when no jar is set. */
  cookies?: {
    present: boolean;
    path?: string | null;
    ageDays?: number | null;
    stale?: boolean;
    cookieCount?: number;
    sessionOnly?: boolean;
    problem?: string | null;
  } | null;
  limits?: { maxSourceMin?: number; retentionHours?: number } | null;
  /** Durable media offload (S3/R2) state, present once /api/health answers. */
  storage?: {
    enabled: boolean;
    clientReady: boolean;
    bucket?: string | null;
    endpoint?: string | null;
    prefix?: string;
    deleteLocal?: boolean;
    /** True only after a real put/head/ranged-get/delete passed. */
    verified?: boolean;
    /** verified && clientReady — what "Media offload on" means. */
    ready?: boolean;
    hasAccessKey?: boolean;
    hasSecret?: boolean;
    accessKeyFirst4?: string;
    usageBytes?: number;
    usageGb?: number;
    budgetGb?: number;
    usedPercent?: number;
    /** Why it is not ready, in words the dialog can show. */
    reason?: string;
    lastTestError?: {
      code?: string;
      message?: string;
      hint?: string;
    } | null;
  };
  /** The worker's real silence budget in seconds (drives stall detection). */
  stallTimeoutSec?: number | null;
  /** Live key state per vendor (Gemini / Groq / NVIDIA / DeepSeek). */
  llm?: KeyStatusDto;
  llmGroq?: KeyStatusDto;
  llmNvidia?: KeyStatusDto;
  llmDeepSeek?: KeyStatusDto;
  encode?: { engine?: string | null; [k: string]: unknown };
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

/** Sandbox/backend readiness helper for cold starts and restarts. */
export interface WaitForSandboxOptions {
  initialDelayMs?: number;
  backoffMs?: number[];
  timeoutMs?: number;
  waitForWorker?: boolean;
  onUpdate?: (state: "starting" | "ready" | "failed", message: string) => void;
}

const DEFAULT_BACKOFF_MS = [2000, 5000, 10000, 20000];

export async function waitForSandbox(
  opts: WaitForSandboxOptions = {},
): Promise<{ ok: boolean; health: HealthDto | null; error: string }> {
  const initialDelayMs = opts.initialDelayMs ?? 1500;
  const backoffMs = opts.backoffMs ?? DEFAULT_BACKOFF_MS;
  const timeoutMs = opts.timeoutMs ?? 60000;
  const waitForWorker = opts.waitForWorker ?? false;

  await new Promise((r) => setTimeout(r, initialDelayMs));

  const start = Date.now();
  let lastError = "";

  for (let attempt = 0; ; attempt++) {
    const remaining = timeoutMs - (Date.now() - start);
    if (remaining <= 0) {
      const msg =
        lastError &&
        /HTTP 5\d\d/.test(lastError)
          ? `The backend is up but unhealthy: ${lastError}`
          : lastError ||
            `Sandbox did not become ready within ${timeoutMs / 1000}s — it may still be starting.`;
      opts.onUpdate?.(("failed" as const), msg);
      return { ok: false, health: null, error: msg };
    }

    try {
      const health = await api.health();
      if (waitForWorker && !health.worker?.online) {
        lastError = "worker not yet online";
        opts.onUpdate?.(("starting" as const), `Sandbox is starting… (worker not online yet)`);
      } else {
        opts.onUpdate?.(("ready" as const), "Sandbox is ready.");
        return { ok: true, health, error: "" };
      }
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
      opts.onUpdate?.(("starting" as const), `Sandbox is starting… (${lastError})`);
    }

    const delay =
      attempt < backoffMs.length
        ? backoffMs[attempt]
        : backoffMs[backoffMs.length - 1];
    const waited = Date.now() - start + delay;
    if (waited > timeoutMs) break;
    await new Promise((r) => setTimeout(r, delay));
  }

  const msg =
    lastError &&
    /HTTP 5\d\d/.test(lastError)
      ? `The backend is up but unhealthy: ${lastError}`
      : lastError ||
        `Sandbox did not become ready within ${timeoutMs / 1000}s — it may still be starting.`;
  opts.onUpdate?.(("failed" as const), msg);
  return { ok: false, health: null, error: msg };
}  /** Minimal internal sleep used by the readiness helper. */

/** API base URL, normalised. */
const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

/** Same guarantee for the path half of every URL this module builds. */
function path(p: string): string {
  return (p.startsWith("/") ? p : `/${p}`).replace(/\/{2,}/g, "/");
}

/** Turn a failed response into a message a human can act on. */
async function errorText(res: Response): Promise<string> {
  let raw = "";
  try {
    raw = (await res.text()).trim();
  } catch {
    return `HTTP ${res.status} (no response body)`;
  }
  if (raw) {
    // An HTML body is not this API talking: every route answers JSON. A page
    // here came from a hop IN FRONT of the API — the preview proxy or the
    // CDN edge answering 502/504 while the sandbox was restarting. Pasting
    // 200 characters of `<!DOCTYPE html>` into a toast is how "Delete does
    // not work" was reported: the reader was shown the error page's source
    // instead of the one thing they needed to know.
    if (/^\s*(<!doctype|<html)/i.test(raw)) {
      // Short message the reader can act on, with the real status kept in
      // parentheses. Never the page source: this body came from a hop in
      // FRONT of the API (the proxy/CDN answering 502/504 while the sandbox
      // restarted), not from any route in this app.
      return `Server is restarting, retrying… (HTTP ${res.status}, proxy error page — not the API)`;
    }
    try {
      const body = JSON.parse(raw) as {
        detail?: unknown;
        errorMessage?: unknown;
        error?: unknown;
        request_id?: unknown;
        restarting?: unknown;
      };
      if (typeof body.detail === "string") return body.detail;
      if (Array.isArray(body.detail)) {
        const first = body.detail[0] as { msg?: unknown } | undefined;
        if (first && typeof first.msg === "string") return first.msg;
      }
      if (typeof body.errorMessage === "string") return body.errorMessage;
      // The backend's own envelope: {error, request_id}. Two different things
      // must not read the same in the UI:
      //   * the supervisor is restarting the API (retry, nothing to report), and
      //   * the API answered and crashed on this request (a real bug, and the
      //     request id is the only handle on its traceback).
      if (typeof body.error === "string") {
        if (body.restarting === true) return `API restarting — ${body.error}`;
        const rid = typeof body.request_id === "string" ? body.request_id.slice(0, 8) : null;
        return rid ? `${body.error} (id ${rid})` : body.error;
      }
    } catch {
      // not JSON — fall through and show the text
    }
    return raw.slice(0, 200);
  }
  return `HTTP ${res.status} (empty response — the request did not reach the API)`;
}

async function request<T>(path_: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path(path_)}`, init);
  if (!res.ok) throw new Error(await errorText(res));
  return (await res.json()) as T;
}

async function requestBlob(path_: string, init?: RequestInit): Promise<Blob> {
  const res = await fetch(`${BASE}${path(path_)}`, init);
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
  status: "needs_app" | "not_connected" | "connected";
  reason: string;
  accountName: string;
  connected: boolean;
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
  audited?: boolean;
  note?: string;
  postedAt: number;
  captionLength: number;
}

export interface UsageModelDto {
  model: string;
  tokens: number;
  calls: number;
  active: boolean;
}

export interface UsageDto {
  used: number;
  limit: number;
  remaining: number;
  percentUsed: number;
  source: "provider-report" | "local-sum";
  localTokens: number;
  reportedUsed: number | null;
  refillAt: number | null;
  refillSec: number | null;
  calls: number;
  activeModel: string;
  activeModelTokens: number;
  models: UsageModelDto[];
  rateLimit: Record<string, string>;
  configured: boolean;
  provider?: DetectionProvider;
  budgeted?: boolean;
  providerLabel?: string;
}

export const api = {
  /** Current backend/API process health. */
  health: () => request<HealthDto>("/api/health"),
  /** Restart the backend worker daemon (the sandbox). */
  restartSandbox: () =>
    request<{ ok: boolean; pid?: number; error?: string }>("/api/sandbox/restart", {
      method: "POST",
    }),
  doctor: () => request<DoctorReport>("/api/doctor"),
  /** Today's token budget for the ACTIVE detection provider. Never throws. */
  usage: () => request<UsageDto>("/api/usage"),
  captionPresets: () => request<CaptionPresetDto[]>("/api/caption-presets"),
  /** One page of batches, newest first. */
  listJobs: (limit = 10) =>
    request<{ jobs: JobDto[]; total: number; nextBefore: number | null }>(
      `/api/jobs?limit=${limit}`,
    ),
  /** Every batch, unpaged — used by the cleanup dialog's "delete it all". */
  listAllJobs: () =>
    request<{ jobs: JobDto[]; total: number; nextBefore: number | null }>("/api/jobs?all=1"),
  getJob: (jobId: string) => request<JobDto>(`/api/jobs/${jobId}`),
  /**
   * Create a job, but only after confirming the backend is reachable.
   *
   * The old code called POST /api/jobs directly and then showed the raw
   * platform error when the container was cold or restarting. That produced the
   * "SANDBOX_NOT_RUNNING / failed to resolve container IP" JSON the user saw.
   *
   * This wrapper first checks /api/health; if it does not answer within the
   * default 60s window, it returns a human message ("Sandbox is starting…")
   * instead of the raw failure. If health answers but the worker is not online
   * yet, it keeps waiting (same backoff) because a live process with no worker
   * is the same cold-start situation.
   */
  createJobWithHealthCheck: (
    source: string,
    sourceType: "url" | "upload" = "url",
    captionPreset?: string,
    maxClips?: number,
    opts?: WaitForSandboxOptions,
  ) =>
    waitForSandbox(opts).then((ready) => {
      if (!ready.ok) {
        throw new Error(ready.error);
      }
      return api.createJob(source, sourceType, captionPreset, maxClips);
    }),
  createJob: (
    source: string,
    sourceType: "url" | "upload" = "url",
    captionPreset?: string,
    maxClips?: number,
    emoji?: { enabled: boolean; density: "low" | "medium" | "high" },
  ) =>
    request<{ jobId: string }>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // maxClips is omitted entirely for "auto" — the server scales the count
      // from the source length, and sending null/undefined would be read as an
      // explicit override of 0 by some clients.
      body: JSON.stringify({
        sourceType,
        source,
        captionPreset,
        ...(maxClips ? { maxClips } : {}),
        ...(emoji ? { emojiEnabled: emoji.enabled, emojiDensity: emoji.density } : {}),
      }),
    }),
  uploadJob: (file: File, captionPreset?: string, maxClips?: number) => {
    const fd = new FormData();
    fd.append("file", file);
    if (captionPreset) fd.append("captionPreset", captionPreset);
    if (maxClips) fd.append("maxClips", String(maxClips));
    return request<{ jobId: string; file: string; bytes: number }>("/api/jobs/upload", {
      method: "POST",
      body: fd,
    });
  },
  /** URL-path convenience wrapper (health-gated). Delegates to createJobWithLanguage. */
  createJobWithLanguageFromUrl: (
    url: string,
    captionPreset?: string,
    maxClips?: number,
    language: CaptionLanguage = "auto",
    opts?: WaitForSandboxOptions,
    emoji?: { enabled: boolean; density: "low" | "medium" | "high" },
  ) =>
    waitForSandbox(opts).then((ready) => {
      if (!ready.ok) {
        throw new Error(ready.error);
      }
      return api.createJobWithLanguage(url.trim(), "url", captionPreset, maxClips, language, emoji);
    }),
  /** Language-aware upload job: sends the ENG/ESP choice as a `language` field. */
  uploadJobWithLanguage: (
    file: File,
    captionPreset?: string,
    maxClips?: number,
    language: CaptionLanguage = "auto",
    emoji?: { enabled: boolean; density: "low" | "medium" | "high" },
  ) => {
    const fd = new FormData();
    fd.append("file", file);
    if (captionPreset) fd.append("captionPreset", captionPreset);
    if (maxClips) fd.append("maxClips", String(maxClips));
    fd.append("language", language);
    if (emoji) {
      fd.append("emojiEnabled", String(emoji.enabled));
      fd.append("emojiDensity", emoji.density);
    }
    return request<{ jobId: string; file: string; bytes: number }>("/api/jobs/upload", {
      method: "POST",
      body: fd,
    });
  },
  /**
   * Mark one batch for deletion. The server answers 202 immediately and
   * finishes removing files/objects in the background, so this call must NOT
   * wait for a purge — it waits only for the acknowledgement, and it is safe
   * to repeat (marking an already-`deleting` batch is idempotent).
   *
   * Retry budget is ~30 s (1+2+4+8+15), because that is how long a sandbox
   * restart or a load-balancer failover takes; the old 3 s budget expired
   * inside the outage and reported a failure for every tap. Only 502/503/504,
   * a proxy error page, and a fetch that never reached an HTTP status are
   * retried — a 400/401/403 is a real answer and is surfaced at once.
   */
  deleteJob: async (jobId: string, keepSource = false) => {
    type DeleteResult = {
      deleting: boolean;
      jobId: string;
      status: "deleting";
      keptSource?: boolean;
      deleted?: boolean;
      cancelled?: "queued" | "running" | "absent";
    };
    const id = encodeURIComponent(jobId);
    const routeMissing = (msg: string) =>
      /^not found$/i.test(msg) || /method not allowed/i.test(msg);
    const transient = (err: unknown): boolean => {
      const msg = err instanceof Error ? err.message : "";
      // No status at all: fetch threw (offline, DNS, origin dropped).
      if (!/HTTP \d{3}/.test(msg)) return true;
      if (/API restarting|proxy error page|instead of the API/i.test(msg)) return true;
      const status = /HTTP (\d{3})/.exec(msg)?.[1];
      return status === "502" || status === "503" || status === "504";
    };
    const waits = [1000, 2000, 4000, 8000, 15000];
    let lastError: unknown = null;
    for (let attempt = 0; attempt <= waits.length; attempt++) {
      try {
        return await request<DeleteResult>(`/api/jobs/${id}/delete`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keepSource }),
        });
      } catch (err) {
        lastError = err;
        const msg = err instanceof Error ? err.message : "";
        if (routeMissing(msg)) {
          // This server only has the DELETE verb (an older build): same
          // operation, same idempotence, so the fallback is safe.
          try {
            return await request<DeleteResult>(`/api/jobs/${id}`, { method: "DELETE" });
          } catch (inner) {
            lastError = inner;
            const innerMsg = inner instanceof Error ? inner.message : "";
            if (routeMissing(innerMsg) || !transient(inner)) throw inner;
          }
        } else if (!transient(err)) {
          throw err;
        }
        if (attempt === waits.length) break;
        await new Promise((r) => setTimeout(r, waits[attempt]));
      }
    }
    const msg = lastError instanceof Error ? lastError.message : "Delete failed.";
    throw new Error(
      `${msg} — gave up after about 30 s of retries. The batch is still there; try again.`,
    );
  },
  /** Failed batches with nothing delivered — the deletable ones. */
  listJunkJobs: () => request<{ jobs: JunkJobDto[] }>("/api/jobs/junk"),
  /** Mark many batches for deletion in one call (max 200 ids per request). */
  bulkDeleteJobs: (jobIds: string[]) =>
    request<{ deleting: string[]; deleted: string[]; missing: string[] }>("/api/jobs/bulk-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobIds }),
    }),
  retryJob: (jobId: string) => request<{ queued: boolean }>(`/api/jobs/${jobId}/retry`, { method: "POST" }),
  /** Cooperative stop at the next safe boundary. */
  pauseJob: (jobId: string) =>
    request<{ state: "paused" | "pausing"; stage: string }>(`/api/jobs/${jobId}/pause`, { method: "POST" }),
  resumeJob: (jobId: string) =>
    request<{ resumed: boolean; queueDepth: number }>(`/api/jobs/${jobId}/resume`, { method: "POST" }),
  redetect: (jobId: string) => request<{ queued: boolean }>(`/api/jobs/${jobId}/redetect`, { method: "POST" }),
  /** Re-encode with fresh captions, reusing the cached transcript and cuts. */
  rerender: (jobId: string) =>
    request<{ queued: boolean; clipsReset: number }>(`/api/jobs/${jobId}/rerender`, { method: "POST" }),
  /** Re-decode the audio (and re-score the cuts), then re-encode. */
  retranscribe: (jobId: string) =>
    request<{ queued: boolean }>(`/api/jobs/${jobId}/retranscribe`, { method: "POST" }),
  verifyClips: (jobId: string) =>
    request<VerifyClipsDto>(
      `/api/dev/verify-clips?jobId=${encodeURIComponent(jobId)}`,
    ),
  jobLogs: (jobId: string) => request<string[]>(`/api/jobs/${jobId}/logs`),
  /** Force the server to re-read env files and re-resolve the LLM keys. */
  reloadSecrets: () =>
    request<{ keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[]; groq?: { keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[] }; nvidia?: { keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[] }; deepseek?: { keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[] }; activeProvider?: string }>(
      "/api/settings/reload-secrets",
      { method: "POST" },
    ),
  /** Save/replace DEEPSEEK_API_KEY in the backend's env file. */
  setDeepseekKey: (key: string) =>
    request<{ keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[]; storedShape: { first6: string; last4: string; length: number } }>(
      "/api/settings/deepseek-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      },
    ),
  /** Delete DEEPSEEK_API_KEY from the backend's env file. */
  deleteDeepseekKey: () =>
    request<{ keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[]; linesRemoved: number }>(
      "/api/settings/deepseek-key?confirm=true",
      { method: "DELETE" },
    ),
  /** Save/replace the S3/R2 offload credentials in the backend's env file. */
  setStorageKeys: (keys: StorageKeysInput) =>
    request<{ enabled: boolean; clientReady: boolean; bucket: string | null; endpoint: string | null; prefix: string; deleteLocal: boolean; savedFields: string[]; storedShape: { secretLength: number; accessKeyFirst4: string } }>(
      "/api/settings/s3-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(keys),
      },
    ),
  /**
   * Move every finished batch's media into the bucket and free local disk.
   *
   * Clip playback streams back out of the bucket, so this is what makes the
   * 4 GB disk survivable: the library lives in the 5 GB bucket and only the
   * batch that is running needs local space.
   */
  /**
   * A REAL bucket round-trip: put, HEAD size check, ranged GET, delete.
   *
   * The four fields being filled in is not the same as the keys working, and
   * the dialog could not tell those apart — the access key id rendered empty
   * while the secret said "saved". This returns the step that failed and the
   * provider's own code (SignatureDoesNotMatch, NoSuchBucket, wrong region…).
   */
  testStorage: () =>
    request<{
      ok: boolean;
      bucket: string | null;
      endpoint: string | null;
      region: string;
      addressing: string;
      elapsedMs: number;
      steps: Array<{ step: string; ok: boolean; ms: number; code?: string; hint?: string; message?: string }>;
      error?: { code?: string; message?: string; hint?: string } | null;
      gate?: { ready?: boolean; verified?: boolean; reason?: string };
    }>("/api/storage/test", { method: "POST" }),
  reclaimStorage: () =>
    request<{
      ok: boolean;
      freedBytes: number;
      freedMb: number;
      uploaded: number;
      scanned: number;
      jobs: number;
      freeBytes: number | null;
      reason: string;
    }>("/api/storage/reclaim", { method: "POST" }),
  /** Dry-run by default; `apply` backs up the job DB and performs the reset. */
  diskReset: (apply = false) =>
    request<{
      ok: boolean;
      dryRun: boolean;
      freeBeforeMb: number;
      freeAfterMb: number;
      freedMb: number;
      plannedMb: number;
      categories: Array<{
        name: string;
        mb: number;
        count: number;
        paths: string[];
        note: string;
      }>;
      kept: Array<{ path: string; reason: string }>;
      openDeleted: Array<{ pid: number; cmd: string; path: string; mb: number }>;
      backup: string | null;
      skipped: Array<{ what: string; why: string }>;
    }>("/api/storage/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apply }),
    }),
  /**
   * Bucket media whose batch no longer exists anywhere in the dashboard.
   *
   * These are the batches a reader cannot see (their job documents are gone),
   * yet their source and clips still count against the plan. Listing is a
   * single bucket read; deleting re-checks every id against the live jobs on
   * the server, so an old list can never remove a batch that has restarted.
   */
  unusedStorage: () =>
    request<{
      items: Array<{ jobId: string; bytes: number; objects: number; kinds: string[] }>;
      totalBytes: number;
      objects: number;
      enabled: boolean;
    }>("/api/storage/unused"),
  deleteUnusedStorage: (jobIds?: string[]) =>
    request<{
      deleted: string[];
      skipped: string[];
      removed: number;
      bytesFreed: number;
    }>("/api/storage/unused/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jobIds && jobIds.length > 0 ? { jobIds } : {}),
    }),
  /** Delete the S3/R2 credentials. Region and prefix (non-secret) survive. */
  deleteStorageKeys: () =>
    request<{ enabled: boolean; clientReady: boolean; bucket: string | null; endpoint: string | null; prefix: string; deleteLocal: boolean; linesRemoved: number }>(
      "/api/settings/s3-key?confirm=true",
      { method: "DELETE" },
    ),
  /** Save/replace NVIDIA_API_KEY in the backend's env file. */
  setNvidiaKey: (key: string) =>
    request<{ keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[]; storedShape: { first6: string; last4: string; length: number } }>(
      "/api/settings/nvidia-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      },
    ),
  /** Delete NVIDIA_API_KEY from the backend's env file. */
  deleteNvidiaKey: () =>
    request<{ keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[]; linesRemoved: number }>(
      "/api/settings/nvidia-key?confirm=true",
      { method: "DELETE" },
    ),
  /** Save/replace GROQ_API_KEY in the backend's env file. */
  setGroqKey: (key: string) =>
    request<{ keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[]; storedShape: { first6: string; last4: string; length: number } }>(
      "/api/settings/groq-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      },
    ),
  /** Delete GROQ_API_KEY from the backend's env file. */
  deleteGroqKey: () =>
    request<{ keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[]; linesRemoved: number }>(
      "/api/settings/groq-key?confirm=true",
      { method: "DELETE" },
    ),
  /** Save/replace GEMINI_API_KEY in the backend's env file. */
  setGeminiKey: (key: string) =>
    request<{ keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[]; storedShape: { first6: string; last4: string; length: number } }>(
      "/api/settings/gemini-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      },
    ),
  /** Delete GEMINI_API_KEY from the backend's env file. */
  deleteGeminiKey: () =>
    request<{ keyLoaded: boolean; source: string; problem: string | null; lastCheckedAt: number; aliases: string[]; linesRemoved: number }>(
      "/api/settings/gemini-key?confirm=true",
      { method: "DELETE" },
    ),
  /** Save/replace the yt-dlp live cookies path (cookies.txt) in the server's env. */
  setCookiesKey: (body: { cookies_file?: string; cookies_from_browser?: string }) =>
    request<{ cookies_file: string | null; cookies_from_browser: string | null; source: string | null; problem: string | null }>(
      "/api/settings/cookies",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),

  /** Delete the yt-dlp cookies from the server's env file. */
  deleteCookiesKey: (confirm = false) =>
    request<{ linesRemoved: number; envDropped: string }>(
      `/api/settings/cookies${confirm ? "?confirm=true" : ""}`,
      { method: "DELETE" },
    ),

  /** Read the current yt-dlp cookies values from the server's env. */
  getCookiesKey: () =>
    request<{ cookies_file: string | null; cookies_from_browser: string | null; source: string | null; problem: string | null }>(
      "/api/settings/cookies",
      { method: "GET" },
    ),


  /** Which detection backend is active right now. */
  detectBackend: () => request<DetectionDto>("/api/settings/detection"),
  /** Switch the AI detection backend. Transcription engine is untouched. */
  setDetectionBackend: (provider: DetectionProvider) =>
    request<DetectionDto>("/api/settings/detection", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    }),
  /** Language-aware job creation: sends the ENG/ESP choice as `language`. */
  createJobWithLanguage: (
    source: string,
    sourceType: "url" | "upload" = "url",
    captionPreset?: string,
    maxClips?: number,
    language: CaptionLanguage = "auto",
    emoji?: { enabled: boolean; density: "low" | "medium" | "high" },
  ) =>
    request<{ jobId: string }>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sourceType,
        source,
        captionPreset,
        language,
        ...(maxClips ? { maxClips } : {}),
        ...(emoji ? { emojiEnabled: emoji.enabled, emojiDensity: emoji.density } : {}),
      }),
    }),
  /** The caption knobs, merged over config.toml, plus which ones are pinned. */
  captionStyle: () => request<CaptionStyleDto>("/api/settings/captions"),
  /** Save the caption knobs. Takes effect on the next render, not this one. */
  saveCaptionStyle: (values: CaptionStyleUpdate) =>
    request<CaptionStyleDto & { saved: Record<string, unknown>; styleId: string }>("/api/settings/captions", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    }),
  /** Drop every override and fall back to config.toml. */
  resetCaptionStyle: () =>
    request<CaptionStyleDto>("/api/settings/captions", { method: "DELETE" }),
  /** Re-render a single clip (its export stage) without re-running detection or transcription. */
  retryClipRender: (clipId: string) =>
    request<{ queued: boolean; clipId: string }>(`/api/clips/${clipId}/retry-render`, {
      method: "POST",
    }),
  /**
   * Swap, add or remove ONE chunk's emoji, then re-render that clip.
   *
   * The server keeps the original judgement (emotion, reason, confidence) and
   * marks a hand-picked emoji `byHandle`, so the editor never claims the model
   * chose something a human typed in.
   */
  setClipEmoji: (clipId: string, i: number, emoji: string | null) =>
    request<{ emojiTrack: CaptionEmojiDto[]; emojiSource?: string | null; rerendering: boolean }>(
      `/api/clips/${clipId}/emoji`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ i, emoji }),
      },
    ),
  emojiWhitelist: () => request<EmojiWhitelistDto>("/api/emoji/whitelist"),
  listClips: (jobId: string) =>
    request<ClipDto[]>(`/api/clips?jobId=${encodeURIComponent(jobId)}`),
  exportZip: (clipIds: string[]) =>
    requestBlob("/api/export/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clipIds }),
    }),
  /**
   * The rendered clip as it plays in the dashboard preview.
   *
   * This is the SAME endpoint the clip card's `<video src>` hits, so an
   * embedded preview does not need its own parallel rendering path or its own
   * separate caption source. The burned-in captions are already in the MP4;
   * the extra cues are only enough to render a readable on-card overlay.
   */  clipVideoUrl: (clipId: string, sizeHint?: number) => {
    const url = `${BASE}/api/clips/${clipId}/video`;
    if (sizeHint && sizeHint > 0 && Number.isFinite(sizeHint)) {
      return `${url}?s=${Math.round(sizeHint)}`;
    }
    return url;
  },
  /**
   * A small, share-sized copy of the clip (<=3 MB, 720x1280, faststart).
   *
   * This is NOT what the preview plays. It exists for `navigator.share({files})`
   * and for any downstream consumer that wants a lightweight MP4, not for the
   * card preview.
   */
  clipShareUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/share`,
  /** The PUBLIC mp4 URL, `/s/<id>.mp4`. */
  clipPublicMp4Url: (shareId: string) => `${BASE}/s/${shareId}.mp4`,
  /** The public OG page, `/s/<id>`. */
  clipPublicPageUrl: (shareId: string) => `${BASE}/s/${shareId}`,
  /** Mint/verify this clip's share link and read back its real state. */
  ensureShareLink: (clipId: string) =>
    request<{
      shareId: string | null;
      shareState: "missing" | "preparing" | "ready" | "failed";
      shareDetail: string;
      shareReady: boolean;
      shareUrl: string | null;
      sharePageUrl: string | null;
    }>(`/api/clips/${encodeURIComponent(clipId)}/share-link`, { method: "POST" }),
  clipThumbUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/thumb`,
  clipDownloadUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/download`,
  clipSrtUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/srt`,
  jobCaptionsUrl: (jobId: string) => `${BASE}/api/jobs/${jobId}/captions.txt`,
  /**
   * The clip as it would render inside an embedded player: same video URL and
   * the rendered caption cues attached, so an embed does not need its own
   * per-clip transcript request before it can show the burned-in text.
   */
  clipEmbedData: (clipId: string) =>
    request<ClipDto>(`/api/clips?jobId=${encodeURIComponent(clipId)}`),

  // --- "Post to..." (additive; none of this affects the download paths) ----
  socialAccounts: () => request<{ platforms: SocialPlatformDto[] }>("/api/social/accounts"),
  /** Begin OAuth. Resolves to the provider's consent URL to open in a tab. */
  socialConnect: (slug: SocialSlug) =>
    request<{ authUrl: string; state: string }>(`/api/social/${slug}/connect`, { method: "POST" }),
  socialDisconnect: (slug: SocialSlug) =>
    request<{ disconnected: boolean; platforms: SocialPlatformDto[] }>(`/api/social/${slug}/disconnect`, {
      method: "POST",
    }),
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

// ---------------------------------------------------------------------------
// Caption language extension (ENG / ESP for DeepSeek subtitle generation)
// ---------------------------------------------------------------------------

/** Caption language choices for the DeepSeek subtitle path.
 *
 *  `auto` is the default and lets the backend decide when it can.
 *  `en` and `es` are explicit declarations for English- or Spanish-dominant
 *  source videos.
 */
export type CaptionLanguage = "en" | "es" | "auto";

export const LANGUAGE_LABELS: Readonly<Record<CaptionLanguage, string>> = {
  en: "English",
  es: "Español",
  auto: "Auto (detect from video)",
};

export const LANGUAGE_CODES: Readonly<Record<Exclude<CaptionLanguage, "auto">, string>> = {
  en: "en",
  es: "es",
};

const DEFAULT_LANGUAGE: CaptionLanguage = "auto";

/** Create a job the same way the dashboard does today, but also pass a chosen
 *  caption language into the backend's POST /api/jobs `language` field.
 *
 *  The backend already accepts `language` on the job-in body (pattern
 *  `^(auto|en|es)$`), and the worker reads that field at the translate stage,
 *  so this helper is the client-side hook for the ENG/ESP choice.
 */
export function createJobWithLanguage(
  source: string,
  sourceType: "url" | "upload" = "url",
  captionPreset?: string,
  maxClips?: number,
  language: CaptionLanguage = DEFAULT_LANGUAGE,
  opts?: WaitForSandboxOptions,
): Promise<{ jobId: string }> {
  return waitForSandbox(opts).then((ready) => {
    if (!ready.ok) {
      throw new Error(ready.error);
    }
    return api.createJobWithLanguage(source, sourceType, captionPreset, maxClips, language);
  });
}

/** Direct (no health gate) variant of the language-aware job creation call. */
export function createJobWithLanguageDirect(
  source: string,
  sourceType: "url" | "upload" = "url",
  captionPreset?: string,
  maxClips?: number,
  language: CaptionLanguage = DEFAULT_LANGUAGE,
): Promise<{ jobId: string }> {
  return api.createJobWithLanguage(source, sourceType, captionPreset, maxClips, language);
}

/** Convenience wrapper for the "Video URL" path: pick the language that
 *  matches the user's ENG/ESP choice for an English- or Spanish-dominant
 *  source video.
 */
export function createJobWithLanguageFromUrl(
  url: string,
  captionPreset?: string,
  maxClips?: number,
  language: CaptionLanguage = DEFAULT_LANGUAGE,
  opts?: WaitForSandboxOptions,
): Promise<{ jobId: string }> {
  return createJobWithLanguage(
    url.trim(),
    "url",
    captionPreset,
    maxClips,
    language,
    opts,
  );
}

/** Upload-based variant that also sends the chosen caption language. */
export function uploadJobWithLanguage(
  file: File,
  captionPreset?: string,
  maxClips?: number,
  language: CaptionLanguage = DEFAULT_LANGUAGE,
): Promise<{ jobId: string; file: string; bytes: number }> {
  const fd = new FormData();
  fd.append("file", file);
  if (captionPreset) fd.append("captionPreset", captionPreset);
  if (maxClips) fd.append("maxClips", String(maxClips));
  fd.append("language", language);
  return request<{ jobId: string; file: string; bytes: number }>("/api/jobs/upload", {
    method: "POST",
    body: fd,
  });
}

/** Switch the AI detection backend and then reload secrets so the newly-chosen
 *  provider's key is picked up immediately. Returns the refreshed DetectionDto.
 */
export async function switchBackend(
  provider: "deepseek" | "nvidia" | "groq" | "gemini" | "ollama",
): Promise<DetectionDto> {
  await api.setDetectionBackend(provider);
  await api.reloadSecrets();
  return (await api.detectBackend()) as DetectionDto;
}
