// Typed REST client for the ClipForge backend (FastAPI, same origin).
// Every call uses relative /api URLs — same origin, no localhost, no IPs.

import type { ClipStage } from "@/lib/clipforge";

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
  /** Which of these are pinned here rather than inherited from config.toml. */
  overrides: Record<string, number | string>;
  limits: Record<string, { min: number; max: number; step: number }>;
}

/** A partial save: an omitted key stops overriding that one value. */
export type CaptionStyleUpdate = Partial<
  Pick<
    CaptionStyleDto,
    "words_per_cue" | "words_per_cue_max" | "pause_sec" | "highlight_color"
  >
>;

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
  /** Authoritative status from the backend's single derivation. Prefer this
   *  over inferring one from `stage` + `progress` — those two disagreeing is
   *  exactly what produced "Failed" beside "61% overall, running". */
  status: "queued" | "running" | "paused" | "failed" | "complete";
  /** True once no further progress is possible. When true, every running-only
   *  field (stagePlan, units, ETAs, queuePosition) is absent from the payload. */
  isTerminal: boolean;
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
  logs?: string[];
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
  /** Short permanent public id for this clip, minted at creation. */
  shareId?: string | null;
  /** missing | preparing | ready | failed — from a real probe of the file. */
  shareState?: "missing" | "preparing" | "ready" | "failed";
  /** Why it is in that state, safe to show as-is. */
  shareDetail?: string;
  /** True only when the public URL is really serving bytes. */
  shareReady?: boolean;
  /** Relative public URLs, ready to hand to a social app. */
  shareUrl?: string | null;
  sharePageUrl?: string | null;
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
  /**
   * The filesystem actually measured — the data dir jobs write to, which is
   * NOT the repo root once CLIPFORGE_DATA_DIR / CLIPFORGE_STORAGE_DIR point
   * job artefacts at a larger volume. Null when the server could not resolve it.
   */
  path?: string | null;
}

/**
 * Durable-media offload (S3/R2). Mirrors objectstore.status().
 *
 * `enabled` means all four credential fields are set; `clientReady` means boto3
 * is installed AND a client could actually be built. Those differ on a host
 * without boto3, and telling the operator "offload is on" when nothing can be
 * pushed is the misleading state this split exists to prevent.
 */
export interface StorageDto {
  enabled: boolean;
  clientReady: boolean;
  bucket: string | null;
  endpoint: string | null;
  prefix: string;
  deleteLocal: boolean;
}

/** The save form. Every field optional; blank/absent means "keep saved value". */
export interface StorageKeysInput {
  endpoint?: string;
  bucket?: string;
  accessKey?: string;
  secretKey?: string;
  region?: string;
  prefix?: string;
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
  /** Live GROQ_API_KEY state — the key for the BACKUP detection backend. */
  llmGroq?: KeyStatusDto;
  /** Live NVIDIA_API_KEY state — an alternative detection backend. */
  llmNvidia?: KeyStatusDto;
  /** Live DEEPSEEK_API_KEY state — the key for the DEFAULT detection backend. */
  llmDeepSeek?: KeyStatusDto;
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
  /** Durable-media offload state. Absent only if the server could not read it. */
  storage?: StorageDto;
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

/** A batch the cleanup dialog offers to delete. */
export interface JunkJobDto {
  _id: string;
  title: string;
  stage: string;
  createdAt?: number;
  /** Why it counts as junk: the failure text, or "no clips rendered". */
  reason: string;
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

/**
 * Sandbox/backend readiness helper for cold starts and restarts.
 *
 * The app now treats a missing or not-yet-ready backend as a
 * "sandbox is starting" state instead of dumping raw fetch errors into the UI.
 * This is the shared primitive for that: it polls /api/health, waits for the
 * process to answer, and then optionally waits until the worker is online.
 *
 * Backoff is exponential-ish and capped so a cold start does not become an
 * instant hammering of the proxy/platform before the container has started.
 */
export interface WaitForSandboxOptions {
  /** Milliseconds before the first probe. Defaults to 1500. */
  initialDelayMs?: number;
  /** Backoff per attempt, in ms. Defaults to 2000, 5000, 10000, 20000. */
  backoffMs?: number[];
  /** Max total time including backoff. Defaults to 60000. */
  timeoutMs?: number;
  /** If true, wait until `health.worker.online` is true in addition to 200. */
  waitForWorker?: boolean;
  /** Called on each failed probe with the latest error text. Optional. */
  onUpdate?: (state: "starting" | "ready" | "failed", message: string) => void;
}

const DEFAULT_BACKOFF_MS = [2000, 5000, 10000, 20000];

export async function waitForSandbox(
  opts: WaitForSandboxOptions = {},
): Promise<{ ok: boolean; health: import("./api").HealthDto | null; error: string }> {
  const initialDelayMs = opts.initialDelayMs ?? 1500;
  const backoffMs = opts.backoffMs ?? DEFAULT_BACKOFF_MS;
  const timeoutMs = opts.timeoutMs ?? 60000;
  const waitForWorker = opts.waitForWorker ?? false;

  await sleep(initialDelayMs);

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
      opts.onUpdate?.("failed", msg);
      return { ok: false, health: null, error: msg };
    }

    try {
      const health = await api.health();
      if (waitForWorker && !health.worker?.online) {
        lastError = "worker not yet online";
        opts.onUpdate?.("starting", `Sandbox is starting… (worker not online yet)`);
      } else {
        opts.onUpdate?.("ready", "Sandbox is ready.");
        return { ok: true, health, error: "" };
      }
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
      opts.onUpdate?.("starting", `Sandbox is starting… (${lastError})`);
    }

    const delay =
      attempt < backoffMs.length
        ? backoffMs[attempt]
        : backoffMs[backoffMs.length - 1];
    const waited = Date.now() - start + delay;
    if (waited > timeoutMs) break;
    await sleep(delay);
  }

  const msg =
    lastError &&
    /HTTP 5\d\d/.test(lastError)
      ? `The backend is up but unhealthy: ${lastError}`
      : lastError ||
        `Sandbox did not become ready within ${timeoutMs / 1000}s — it may still be starting.`;
  opts.onUpdate?.("failed", msg);
  return { ok: false, health: null, error: msg };
}

/** Minimal internal sleep used by the readiness helper. */
function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * API base URL, normalised.
 *
 * A trailing slash on VITE_API_BASE produced "/api//health"-style URLs — the
 * Vite dev proxy still matched them, but the FastAPI route did not, and the
 * response came back as a 400/404 that the dashboard reported as "API
 * unreachable". Collapsing duplicates at the one place the two strings meet
 * removes that whole failure class.
 */
const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

/** Same guarantee for the path half of every URL this module builds. */
const path = (p: string) => (p.startsWith("/") ? p : `/${p}`).replace(/\/{2,}/g, "/");

/**
 * Turn a failed response into a message a human can act on.
 *
 * This used to read `body.detail` and fall back to `HTTP 400`, which meant the
 * two worst cases looked identical: FastAPI saying something useful, and a hop
 * in front of it answering with no body at all. Every report of "the deleter
 * does nothing, it says HTTP 400" was the second kind, so the fallback now
 * names what it saw instead of a bare status code.
 */
async function errorText(res: Response): Promise<string> {
  let raw = "";
  try {
    raw = (await res.text()).trim();
  } catch {
    return `HTTP ${res.status} (no response body)`;
  }
  if (raw) {
    try {
      const body = JSON.parse(raw) as { detail?: unknown; errorMessage?: unknown };
      if (typeof body.detail === "string") return body.detail;
      // FastAPI validation errors are an array of objects, not a string.
      if (Array.isArray(body.detail)) {
        const first = body.detail[0] as { msg?: unknown; loc?: unknown } | undefined;
        if (first && typeof first.msg === "string") return first.msg;
      }
      if (typeof body.errorMessage === "string") return body.errorMessage;
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

/** Request that returns a raw Blob (file downloads). */
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

export interface UsageModelDto {
  model: string;
  tokens: number;
  calls: number;
  active: boolean;
}

export interface UsageDto {
  /** Tokens used today — the provider's own figure when it has told us, else our sum. */
  used: number;
  limit: number;
  remaining: number;
  percentUsed: number;
  /**
   * `provider-report` = the provider's own 429 figure (authoritative, Groq
   * only). `local-sum` = our own tally, which is the only source on providers
   * that bill a balance rather than publishing a per-model daily token
   * ceiling. Every number here is scoped to the ACTIVE provider — the daily
   * budgets are not comparable, so mixing them would report one provider's
   * spend against another's limit.
   */
  source: "provider-report" | "local-sum";
  localTokens: number;
  reportedUsed: number | null;
  /** Epoch ms when the daily window next frees capacity, or null if unknown. */
  refillAt: number | null;
  refillSec: number | null;
  calls: number;
  activeModel: string;
  activeModelTokens: number;
  models: UsageModelDto[];
  rateLimit: Record<string, string>;
  configured: boolean;
  /**
   * Which provider `limit`/`used` describe, and the ONLY provider they count.
   * The figures are scoped to it, not a cross-provider total: the budgets are
   * not comparable (Groq states a real per-model tokens-per-day ceiling;
   * DeepSeek and NIM bill a balance or credits), so a combined total would
   * report one provider's spend against another's limit.
   */
  provider?: DetectionProvider;
  /**
   * False when the active backend has no daily token budget to meter at all
   * (Ollama is local and free; Gemini has no published daily ceiling). The
   * numbers are then zeroed placeholders, not a real allowance — the UI says
   * so rather than drawing a meter against an invented denominator.
   */
  budgeted?: boolean;
  /** Display name for the active provider ("NVIDIA", "Groq", ...). */
  providerLabel?: string;
}

export const api = {
  health: () => request<HealthDto>("/api/health"),
  doctor: () => request<DoctorReport>("/api/doctor"),
  /** Today's token budget for the ACTIVE detection provider. Never throws. */
  usage: () => request<UsageDto>("/api/usage"),
  captionPresets: () => request<CaptionPresetDto[]>("/api/caption-presets"),
  /**
   * One page of batches, newest first. The list projection deliberately omits
   * `logs`: the dashboard re-polls this every two seconds and shipping 300 log
   * lines per job on every poll was 285 KB a time for seventeen batches.
   * `nextBefore` is the cursor to pass as `before` for the following page.
   */
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
  ) =>
    request<{ jobId: string }>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // maxClips is omitted entirely for "auto" — the server scales the count
      // from the source length, and sending null/undefined would be read as an
      // explicit override of 0 by some clients.
      body: JSON.stringify(
        maxClips ? { sourceType, source, captionPreset, maxClips } : { sourceType, source, captionPreset },
      ),
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
  /**
   * Delete one batch.
   *
   * POST /api/jobs/{id}/delete, not DELETE /api/jobs/{id}: a verb is the one
   * thing a proxy in front of the app can silently refuse, and the browser's
   * refusal arrived as a bodyless 400 with the batch still sitting in the list
   * and no way to tell why. POST also needs no CORS preflight.
   *
   * `keepSource` keeps the downloaded/uploaded source video on the server and
   * removes only the job record, workspace and rendered clips — the checkbox
   * in the delete dialog. Default false: the usual delete frees everything.
   *
   * Retried once. Deleting a batch is idempotent — the second attempt answers
   * 404 if the first one actually worked — and a single bodyless 4xx from a
   * proxy should not cost the user their cleanup.
   */
  deleteJob: async (jobId: string, keepSource = false) => {
    type DeleteResult = { deleted: boolean; cancelled: "queued" | "running" | "absent"; keptSource?: boolean };
    const id = encodeURIComponent(jobId);
    // A backend that does not know this route answers 404 "Not Found" or 405
    // "Method Not Allowed" (the GET /api/jobs/{id} pattern claims the path and
    // refuses the verb) — never this app's "job not found". Those two mean the
    // route is missing, and the DELETE verb still works there.
    const routeMissing = (msg: string) =>
      /^not found$/i.test(msg) || /method not allowed/i.test(msg);
    // One retry, because deleting a batch is idempotent and a single bodyless
    // 4xx from something in the request path should not cost the user a
    // cleanup they were owed. Never retried when the route itself is missing.
    const withRetry = async (url: string, init: RequestInit): Promise<DeleteResult> => {
      try {
        return await request<DeleteResult>(url, init);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "";
        if (routeMissing(msg) || !/^HTTP 4\d\d/.test(msg)) throw err;
        return await request<DeleteResult>(url, init);
      }
    };
    try {
      return await withRetry(`/api/jobs/${id}/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keepSource }),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "";
      if (routeMissing(msg)) {
        return await withRetry(`/api/jobs/${id}`, { method: "DELETE" });
      }
      throw err;
    }
  },
  /** Failed batches with nothing delivered — the deletable ones. */
  listJunkJobs: () =>
    request<{ jobs: JunkJobDto[] }>("/api/jobs/junk"),
  /** Delete many batches in one call (max 200 ids per request). */
  bulkDeleteJobs: (jobIds: string[]) =>
    request<{ deleted: string[]; missing: string[] }>("/api/jobs/bulk-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobIds }),
    }),
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
  /**
   * Re-decode the audio (and re-score the cuts), then re-encode. The repair
   * path for anything wrong with WHAT WAS HEARD rather than how it looks.
   */
  retranscribe: (jobId: string) =>
    request<{ queued: boolean }>(`/api/jobs/${jobId}/retranscribe`, { method: "POST" }),
  verifyClips: (jobId: string) =>
    request<VerifyClipsDto>(`/api/dev/verify-clips?jobId=${encodeURIComponent(jobId)}`),
  jobLogs: (jobId: string) => request<string[]>(`/api/jobs/${jobId}/logs`),
  /** Force the server to re-read env files and re-resolve the LLM keys. */
  reloadSecrets: () =>
    request<
      KeyStatusDto & {
        groq?: KeyStatusDto;
        nvidia?: KeyStatusDto;
        deepseek?: KeyStatusDto;
        activeProvider?: string;
      }
    >(
      "/api/settings/reload-secrets",
      { method: "POST" },
    ),
  /** Save/replace DEEPSEEK_API_KEY in the backend's env file. */
  setDeepseekKey: (key: string) =>
    request<KeyStatusDto & { storedShape: { first6: string; last4: string; length: number } }>(
      "/api/settings/deepseek-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      },
    ),
  /** Delete DEEPSEEK_API_KEY from the backend's env file. */
  deleteDeepseekKey: () =>
    request<KeyStatusDto & { linesRemoved: number }>("/api/settings/deepseek-key?confirm=true", {
      method: "DELETE",
    }),
  /** Save/replace the S3/R2 offload credentials in the backend's env file.
   *
   *  Every field is optional and blank means "keep what is already saved", so
   *  a rotated secret can be pasted on its own. The server persists what it is
   *  given, re-reads the environment per operation and keys its S3 client on
   *  the resolved config, so this takes effect with no restart. */
  setStorageKeys: (keys: StorageKeysInput) =>
    request<StorageDto & { savedFields: string[]; storedShape: { secretLength: number; accessKeyFirst4: string } }>(
      "/api/settings/s3-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(keys),
      },
    ),
  /** Delete the S3/R2 credentials. Region and prefix (non-secret) survive. */
  deleteStorageKeys: () =>
    request<StorageDto & { linesRemoved: number }>("/api/settings/s3-key?confirm=true", {
      method: "DELETE",
    }),
  /** Save/replace NVIDIA_API_KEY in the backend's env file. */
  setNvidiaKey: (key: string) =>
    request<KeyStatusDto & { storedShape: { first6: string; last4: string; length: number } }>(
      "/api/settings/nvidia-key",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      },
    ),
  /** Delete NVIDIA_API_KEY from the backend's env file. */
  deleteNvidiaKey: () =>
    request<KeyStatusDto & { linesRemoved: number }>("/api/settings/nvidia-key?confirm=true", {
      method: "DELETE",
    }),
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
    request<KeyStatusDto & { linesRemoved: number }>("/api/settings/groq-key?confirm=true", {
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
    request<KeyStatusDto & { linesRemoved: number }>("/api/settings/gemini-key?confirm=true", {
      method: "DELETE",
    }),
  /** Switch the AI-detection backend. Transcription engine is untouched. */
  setDetectionBackend: (provider: DetectionProvider) =>
    request<DetectionDto>("/api/settings/detection", {
      method: "POST",
      body: JSON.stringify({ provider }),
    }),
  /** The caption knobs, merged over config.toml, plus which ones are pinned. */
  captionStyle: () => request<CaptionStyleDto>("/api/settings/captions"),
  /** Save the caption knobs. Takes effect on the next render, not this one. */
  saveCaptionStyle: (values: CaptionStyleUpdate) =>
    request<CaptionStyleDto & { saved: Record<string, unknown>; styleId: string }>(
      "/api/settings/captions",
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      },
    ),
  /** Drop every override and fall back to config.toml. */
  resetCaptionStyle: () =>
    request<CaptionStyleDto>("/api/settings/captions", { method: "DELETE" }),
  /**
   * Re-render a single clip (its export stage) without re-running detection or
   * transcription. Useful when a clip's render state is "failed" and you want
   * to retry just that one.
   */
  retryClipRender: (clipId: string) =>
    request<{ queued: boolean; clipId: string }>(`/api/clips/${clipId}/retry-render`, {
      method: "POST",
    }),
  listClips: (jobId: string) => request<ClipDto[]>(`/api/clips?jobId=${encodeURIComponent(jobId)}`),
  exportZip: (clipIds: string[]) =>
    requestBlob("/api/export/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clipIds }),
    }),
  clipVideoUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/video`,
  /**
   * The small, share-sized copy (<=3 MB, 720x1280, faststart) built on first
   * request by app/share_video.py. This is what the Web Share sheet consumes:
   * the 1080p master is 6-10 MB and would blow the user-activation window
   * before `navigator.share()` could be called.
   */
  clipShareUrl: (clipId: string) => `${BASE}/api/clips/${clipId}/share`,
  /**
   * The PUBLIC mp4 URL, `/s/<id>.mp4`.
   *
   * Public on purpose: no login, no expiry, survives the batch and the host.
   * This is what "Share link" copies and what a social app is pointed at.
   */
  clipPublicMp4Url: (shareId: string) => `${BASE}/s/${shareId}.mp4`,
  /** The public OG page, `/s/<id>` — what X and "Share..." post. */
  clipPublicPageUrl: (shareId: string) => `${BASE}/s/${shareId}`,
  /**
   * Mint/verify this clip's share link and read back its real state.
   * Idempotent: an already-ready clip costs one stat, and the returned id is
   * the same one forever.
   */
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
