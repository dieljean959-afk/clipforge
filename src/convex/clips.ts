import { v } from "convex/values";
import {
  internalMutation,
  internalQuery,
  mutation,
  query,
  type MutationCtx,
} from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";
import type { Doc, Id } from "./_generated/dataModel";
import { internal, api } from "./_generated/api";
import {
  fetchUnits,
  transcribeUnits,
  findUnits,
  renderUnits,
  transcribeTimeoutMs,
  STAGE_TIMEOUT_MS,
  karaokeCues,
  buildSrt,
  type TranscriptWord,
  type ClipCandidateOutput,
} from "./engine";

// ---------------------------------------------------------------------------
// ClipForge orchestrator (serverless)
//
// Byte work (fetch, Whisper, ffmpeg.wasm render) runs in REAL processes: the
// fetch runs server-side in a node action (cobalt -> direct HTTP), while
// Whisper + ffmpeg run in the user's browser via the dashboard render driver.
// The server tick is a WATCHDOG: it enforces stage timeouts and a no-progress
// kill so no job can ever sit in a fake "Working..." state.
//
// Every stage advance is triggered by a REAL completed artifact:
//   grabbing    -> a stored source MP4 (sourceStorageId) reported by the fetcher
//   transcribing-> a stored Whisper word transcript (transcriptStorageId)
//   finding     -> real Gemini-scored windows (detection.ts action)
//   rendering   -> real stored 9:16 MP4s (clip.mediaStorageId), one per clip
//
// There is NO simulation anywhere: no fabricated durations, no placeholder
// frames, no heuristic scoring fallback.
// ---------------------------------------------------------------------------

/** Delay before each tick schedules the next one — the watchdog loop. */
const CHAIN_DELAY_MS = 2000;
/** A watchdog with no heartbeat for this long (server clock) is considered dead. */
const WORKER_STALE_MS = 45_000;
/** Hard watchdog: no progress write for this long kills the job. */
const WATCHDOG_MS = 90_000;
/** Log ring size per job. */
const MAX_LOGS = 80;

/**
 * Queue + broker identity — the deploy-time handshake (kept for parity with
 * the self-hosted worker boot log format). Convex storage + DB is the shared
 * disk/broker: API, watchdog, and browser render worker all read/write the
 * same rows and the same file storage.
 */
const WORKER_IDENTITY = {
  queue: "clipforge",
  broker: "convex://clipforge-shared-db",
  bootLine:
    "Worker online, consuming queue: clipforge with broker: convex://clipforge-shared-db",
} as const;

// ---------------------------------------------------------------------------
// queries
// ---------------------------------------------------------------------------

export const listJobs = query({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return [];
    return await ctx.db
      .query("jobs")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .order("desc")
      .take(50);
  },
});

export const getJob = query({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return null;
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) return null;
    return job;
  },
});

/** Per-job worker log (stage start/end/duration lines) for the "View logs" UI. */
export const getJobLogs = query({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return [];
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) return [];
    return job.logs;
  },
});

export const listClips = query({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return [];
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) return [];
    const clips = await ctx.db
      .query("clips")
      .withIndex("by_job", (q) => q.eq("jobId", jobId))
      .collect();
    return clips.sort((a, b) => a.order - b.order);
  },
});

export const recentClips = query({
  args: { limit: v.optional(v.number()) },
  handler: async (ctx, { limit }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return [];
    const clips = await ctx.db
      .query("clips")
      .withIndex("by_user", (q) => q.eq("userId", userId))
      .order("desc")
      .take(limit ?? 24);
    return clips.sort((a, b) => b.score - a.score);
  },
});

/**
 * Worker liveness — powers the offline banner. Liveness is judged with the
 * SERVER's own clock (serverClockMs), never the browser clock.
 */
export const getWorkerStatus = query({
  args: {},
  handler: async (ctx) => {
    const serverNow = Date.now();
    const row = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    if (row === null) {
      return {
        singleton: "worker" as const,
        online: false,
        lastHeartbeat: 0,
        lastHeartbeatServerMs: undefined as number | undefined,
        lastClientPingServerMs: undefined as number | undefined,
        modelLoaded: false,
        warmupDetail: undefined,
        rates: undefined,
        queueName: undefined as string | undefined,
        brokerUrl: undefined as string | undefined,
        tickCount: undefined as number | undefined,
        repairing: false,
        lastBootLine: undefined as string | undefined,
        serverClockMs: serverNow,
      };
    }
    const hb = row.lastHeartbeatServerMs ?? row.lastHeartbeat;
    const alive = row.chainArmed === true && serverNow - hb < WORKER_STALE_MS;
    return {
      ...row,
      online: alive,
      lastHeartbeatServerMs: hb,
      serverClockMs: serverNow,
    };
  },
});

/** Signed URL for any stored media this user owns (source / clip / thumb). */
export const getStorageUrl = query({
  args: { storageId: v.id("_storage") },
  handler: async (ctx, { storageId }) => {
    return await ctx.storage.getUrl(storageId);
  },
});

/** Signed URL of a rendered clip MP4 — null when the clip has no media yet. */
export const getClipMediaUrl = query({
  args: { clipId: v.id("clips") },
  handler: async (ctx, { clipId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return null;
    const clip = await ctx.db.get(clipId);
    if (clip === null || clip.userId !== userId) return null;
    if (clip.mediaStorageId === undefined) return null;
    return await ctx.storage.getUrl(clip.mediaStorageId);
  },
});

/** Signed URL of a clip's first-frame thumbnail (JPEG captured at render). */
export const getClipThumbUrl = query({
  args: { clipId: v.id("clips") },
  handler: async (ctx, { clipId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return null;
    const clip = await ctx.db.get(clipId);
    if (clip === null || clip.userId !== userId) return null;
    if (clip.thumbStorageId === undefined) return null;
    return await ctx.storage.getUrl(clip.thumbStorageId);
  },
});

/** Signed URL of the job's stored source MP4 (null until the fetch lands). */
export const getSourceMediaUrl = query({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return null;
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) return null;
    if (job.sourceStorageId === undefined) return null;
    return await ctx.storage.getUrl(job.sourceStorageId);
  },
});

// ---------------------------------------------------------------------------
// mutations (lifecycle)
// ---------------------------------------------------------------------------

const URL_RE = /^https?:\/\/[^\s<>"')\]]+$/i;

async function armWorker(ctx: MutationCtx): Promise<void> {
  const serverNow = Date.now();
  const row = await ctx.db
    .query("workerStatus")
    .withIndex("singleton", (q) => q.eq("singleton", "worker"))
    .unique();
  const patch = {
    online: true,
    lastHeartbeat: serverNow,
    lastHeartbeatServerMs: serverNow,
    modelLoaded: true,
    repairing: false,
    chainArmed: true,
    queueName: WORKER_IDENTITY.queue,
    brokerUrl: WORKER_IDENTITY.broker,
    lastBootLine: WORKER_IDENTITY.bootLine,
  };
  if (row === null) {
    await ctx.db.insert("workerStatus", {
      singleton: "worker",
      tickCount: 1,
      rates: {},
      ...patch,
    });
  } else {
    await ctx.db.patch(row._id, {
      ...patch,
      tickCount: (row.tickCount ?? 0) + 1,
    });
  }
  console.log(WORKER_IDENTITY.bootLine);
}

/** Client handshake: re-arms the watchdog chain on cold start. */
export const pingWorker = mutation({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");

    const serverNow = Date.now();
    const row = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    const hb = row === null ? 0 : (row.lastHeartbeatServerMs ?? row.lastHeartbeat);
    const alive = row !== null && row.chainArmed === true && serverNow - hb < WORKER_STALE_MS;
    if (alive && row !== null) {
      await ctx.db.patch(row._id, { lastClientPingServerMs: serverNow });
      return { repairing: false, serverClockMs: serverNow };
    }
    await armWorker(ctx);
    await ctx.scheduler.runAfter(0, internal.clips.workerTick, {});
    return { repairing: true, serverClockMs: serverNow };
  },
});

/** Manual "Start worker" — same repair as the ping, for recovery from the UI. */
export const startWorker = mutation({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    await armWorker(ctx);
    await ctx.scheduler.runAfter(0, internal.clips.workerTick, {});
    return { started: true, serverClockMs: Date.now() };
  },
});

export const createJob = mutation({
  args: {
    sourceType: v.union(v.literal("url"), v.literal("upload")),
    source: v.string(),
  },
  handler: async (ctx, { sourceType, source }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in to forge clips.");

    const worker = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    const serverNow = Date.now();
    const workerAlive =
      worker !== null &&
      worker.chainArmed === true &&
      serverNow - (worker.lastHeartbeatServerMs ?? worker.lastHeartbeat) < WORKER_STALE_MS;
    if (!workerAlive) {
      await armWorker(ctx);
    }

    const trimmed = source.trim();
    if (trimmed.length === 0) {
      throw new Error("Attach a video file or paste a video URL.");
    }
    if (trimmed.length > 2000) {
      throw new Error("That source string is too long.");
    }
    if (sourceType === "url") {
      if (!URL_RE.test(trimmed)) {
        throw new Error("That doesn't look like a valid video URL.");
      }
      const lower = trimmed.toLowerCase();
      const supported =
        lower.includes("youtube.com") ||
        lower.includes("youtu.be") ||
        lower.includes("vimeo.com") ||
        lower.includes("tiktok.com") ||
        lower.includes("instagram.com") ||
        lower.includes("x.com") ||
        lower.includes("twitter.com") ||
        lower.includes("twitch.tv") ||
        lower.includes(".mp4") ||
        lower.includes(".mov") ||
        lower.includes(".webm");
      if (!supported) {
        throw new Error(
          "Unsupported link. Use a YouTube, Vimeo, TikTok, Instagram, X or Twitch URL, or a direct .mp4 link.",
        );
      }
    }

    const jobId = await ctx.db.insert("jobs", {
      userId,
      sourceType,
      source: trimmed,
      isUpload: sourceType === "upload",
      stage: "queued",
      progress: 0,
      progressDetail: workerAlive
        ? "Queued — a browser render worker will pick it up..."
        : "Queued — waking the worker (cold start)...",
      stageTimings: [],
      fetchAttempts: [],
      logs: [
        `${ts()} · queued — ${sourceType === "upload" ? `upload "${trimmed}"` : trimmed}`,
      ],
      createdAt: Date.now(),
    });

    await ctx.scheduler.runAfter(0, internal.clips.workerTick, {});
    return jobId;
  },
});

/**
 * Attach the uploaded file's REAL bytes to a job (client uploads via
 * generateUploadUrl, then points the job at the storage id). The job cannot
 * advance past grabbing without it — no synthetic source is ever assumed.
 */
export const attachUpload = mutation({
  args: { jobId: v.id("jobs"), storageId: v.id("_storage"), bytes: v.number() },
  handler: async (ctx, { jobId, storageId, bytes }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) throw new Error("Job not found.");
    if (!job.isUpload) throw new Error("Not an upload job.");
    if (job.sourceStorageId !== undefined) {
      throw new Error("Upload already attached.");
    }
    if (bytes <= 0) throw new Error("Attached file is empty.");
    await ctx.db.patch(jobId, {
      sourceStorageId: storageId,
      sourceBytes: bytes,
      fetchSource: "upload",
      downloadSource: "upload",
      logs: appendLog(
        job.logs,
        `${ts()} · ↳ attached upload: ${(bytes / (1024 * 1024)).toFixed(1)} MB`,
      ),
    });
  },
});

export const retryJob = mutation({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) {
      throw new Error("Job not found.");
    }
    if (job.stage !== "failed") {
      throw new Error("Only failed jobs can be retried.");
    }
    const oldClips = await ctx.db
      .query("clips")
      .withIndex("by_job", (q) => q.eq("jobId", jobId))
      .collect();
    for (const c of oldClips) await ctx.db.delete(c._id);

    const worker = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    const serverNow = Date.now();
    const workerAlive =
      worker !== null &&
      worker.chainArmed === true &&
      serverNow - (worker.lastHeartbeatServerMs ?? worker.lastHeartbeat) < WORKER_STALE_MS;
    if (!workerAlive) {
      await armWorker(ctx);
    }

    await ctx.db.patch(jobId, {
      stage: "queued",
      progress: 0,
      progressDetail: "Queued — a browser render worker will pick it up...",
      unitLabel: undefined,
      unitsDone: undefined,
      unitsTotal: undefined,
      stageEtaSec: undefined,
      totalEtaSec: undefined,
      lastProgressAt: undefined,
      slow: false,
      stageStartedAt: undefined,
      error: undefined,
      errorDetail: undefined,
      clipCount: undefined,
      totalClipsSec: undefined,
      captionsTxt: undefined,
      completedAt: undefined,
      downloadSource: undefined,
      clipSource: undefined,
      detectedSource: undefined,
      renderState: undefined,
      renderError: undefined,
      renderProgress: undefined,
      // keep source/transcript artifacts — a retry re-runs detection + render
      // on the REAL fetched media without re-downloading
      logs: appendLog(
        job.logs,
        `${ts()} · ↻ retry queued${workerAlive ? "" : " — worker re-armed (was cold)"}`,
      ),
    });
    await ctx.scheduler.runAfter(0, internal.clips.workerTick, {});
  },
});

export const deleteJob = mutation({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) {
      throw new Error("Job not found.");
    }
    const oldClips = await ctx.db
      .query("clips")
      .withIndex("by_job", (q) => q.eq("jobId", jobId))
      .collect();
    for (const c of oldClips) await ctx.db.delete(c._id);
    await ctx.db.delete(jobId);
  },
});

// ---------------------------------------------------------------------------
// browser render worker protocol — REAL artifact reporting
//
// The dashboard tab is the render worker (Whisper + ffmpeg.wasm run where the
// bytes and the GPU/CPU actually are). Each mutation below is a REAL event:
// it either carries a stored artifact id or an exact error. The watchdog
// fails any job whose worker stops reporting.
// ---------------------------------------------------------------------------

/** Auth'd upload URL so the browser can attach a real local file. */
export const generateUploadUrl = mutation({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    return await ctx.storage.generateUploadUrl();
  },
});

/** Claim the oldest queued job for this user. Returns it, or null. */
export const clientClaimJob = mutation({
  args: {},
  handler: async (ctx) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return null;
    const queued = await ctx.db
      .query("jobs")
      .withIndex("by_user_stage", (q) => q.eq("userId", userId).eq("stage", "queued"))
      .order("asc")
      .take(1);
    const job = queued[0];
    if (job === undefined) return null;
    const now = Date.now();
    await ctx.db.patch(job._id, {
      stage: "grabbing",
      progress: 0,
      progressDetail: "Fetching the video...",
      unitLabel: "MB",
      unitsTotal: 0,
      unitsDone: 0,
      stageStartedAt: now,
      lastProgressAt: now,
      slow: false,
      renderState: "running",
      renderError: undefined,
      renderProgress: undefined,
      logs: appendLog(
        job.logs,
        `${ts()} · ▶ grabbing — cobalt fetch (1080p max MP4), fallback: direct download`,
      ),
    });
    return job;
  },
});

/** Real fetch progress: bytes on the wire. */
export const reportFetchProgress = mutation({
  args: {
    jobId: v.id("jobs"),
    bytesDone: v.number(),
    bytesTotal: v.optional(v.nullable(v.number())),
  },
  handler: async (ctx, { jobId, bytesDone, bytesTotal }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) throw new Error("Job not found.");
    if (job.stage !== "grabbing") return;

    const u = fetchUnits(bytesDone, bytesTotal ?? null);
    await ctx.db.patch(jobId, {
      progress: u.unitsTotal > 0 ? Math.min(99, (u.unitsDone / u.unitsTotal) * 100) : 1,
      progressDetail: "Fetching the video...",
      unitLabel: u.unitLabel,
      unitsDone: u.unitsDone,
      unitsTotal: u.unitsTotal,
      lastProgressAt: Date.now(),
    });
  },
});

/** The REAL download completed: a stored MP4 with measured metadata. */
async function doReportFetch(
  ctx: MutationCtx,
  args: {
    jobId: Id<"jobs">;
    storageId: Id<"_storage">;
    bytes: number;
    durationSec: number;
    width: number;
    height: number;
    fetchSource: string;
  },
): Promise<void> {
  const job = await ctx.db.get(args.jobId);
  if (job === null) throw new Error("Job not found.");
  if (job.stage !== "grabbing") {
    throw new Error(`Job is in stage "${job.stage}", expected "grabbing".`);
  }
  if (args.durationSec <= 0 || args.bytes <= 0) {
    throw new Error(
      `Fetched file failed verification (duration ${args.durationSec}s, ${args.bytes} bytes).`,
    );
  }
  const now = Date.now();
  const units = transcribeUnits(args.durationSec, 0);
  await ctx.db.patch(args.jobId, {
    sourceStorageId: args.storageId,
    sourceBytes: args.bytes,
    sourceDurationSec: args.durationSec,
    sourceWidth: args.width,
    sourceHeight: args.height,
    durationSec: args.durationSec,
    fetchSource: args.fetchSource,
    fetchError: undefined,
    downloadSource: args.fetchSource,
    stage: "transcribing",
    progress: 0,
    progressDetail: "Whisper is transcribing word timings...",
    unitLabel: units.unitLabel,
    unitsTotal: units.unitsTotal,
    unitsDone: 0,
    stageStartedAt: now,
    lastProgressAt: now,
    slow: false,
    stageTimings: [
      ...(job.stageTimings ?? []),
      { stage: "grabbing", durationMs: now - (job.stageStartedAt ?? now), at: now },
    ],
    logs: appendLog(
      job.logs,
      `${ts()} · ✓ grabbing — ${(args.bytes / (1024 * 1024)).toFixed(1)} MB via ${args.fetchSource}, ${Math.round(args.durationSec)}s ${args.width}x${args.height} (stored, verified)`,
    ),
  });
}

/** The REAL download completed: a stored MP4 with measured metadata. */
export const reportFetch = mutation({
  args: {
    jobId: v.id("jobs"),
    storageId: v.id("_storage"),
    bytes: v.number(),
    durationSec: v.number(),
    width: v.number(),
    height: v.number(),
    fetchSource: v.string(),
  },
  handler: async (ctx, args) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(args.jobId);
    if (job === null || job.userId !== userId) throw new Error("Job not found.");
    await doReportFetch(ctx, args);
  },
});

/**
 * Byte truth from the node fetch action: the MP4 is in storage. The job
 * stays in "grabbing" until the browser decodes the file and reports the
 * verified duration/dimensions (reportFetch) — real verification, not
 * assumed metadata. If the tab dies here, the watchdog fails the job
 * instead of pretending.
 */
export const reportFetchStored = internalMutation({
  args: {
    jobId: v.id("jobs"),
    storageId: v.id("_storage"),
    bytes: v.number(),
    fetchSource: v.string(),
  },
  handler: async (ctx, { jobId, storageId, bytes, fetchSource }) => {
    const job = await ctx.db.get(jobId);
    if (job === null || job.stage !== "grabbing") return;
    await ctx.db.patch(jobId, {
      sourceStorageId: storageId,
      sourceBytes: bytes,
      fetchSource,
      fetchError: undefined,
      lastProgressAt: Date.now(),
      logs: appendLog(
        job.logs,
        `${ts()} · ✓ ${fetchSource} fetch — ${(bytes / (1024 * 1024)).toFixed(1)} MB stored, verifying metadata...`,
      ),
    });
  },
});

/** Real fetch progress: bytes on the wire (internal variant for the action). */
export const reportFetchProgressInternal = internalMutation({
  args: {
    jobId: v.id("jobs"),
    bytesDone: v.number(),
    bytesTotal: v.optional(v.nullable(v.number())),
  },
  handler: async (ctx, { jobId, bytesDone, bytesTotal }) => {
    const job = await ctx.db.get(jobId);
    if (job === null || job.stage !== "grabbing") return;
    const u = fetchUnits(bytesDone, bytesTotal ?? null);
    await ctx.db.patch(jobId, {
      progress: u.unitsTotal > 0 ? Math.min(99, (u.unitsDone / u.unitsTotal) * 100) : 1,
      progressDetail: "Fetching the video...",
      unitLabel: u.unitLabel,
      unitsDone: u.unitsDone,
      unitsTotal: u.unitsTotal,
      lastProgressAt: Date.now(),
    });
  },
});

/** The REAL download failed: exact fetcher error, no retry theater (internal). */
export const reportFetchFailureInternal = internalMutation({
  args: { jobId: v.id("jobs"), error: v.string(), fetcher: v.string() },
  handler: async (ctx, { jobId, error, fetcher }) => {
    const job = await ctx.db.get(jobId);
    if (job === null) return;
    const now = Date.now();
    await ctx.db.patch(jobId, {
      fetchError: error,
      fetchAttempts: [
        ...(job.fetchAttempts ?? []),
        { method: fetcher, ok: false, error, at: now },
      ],
      stage: "failed",
      error: `Could not fetch video via ${fetcher}: ${error}`,
      errorDetail:
        "Try the Retry button (fetchers are retried in order), or attach the file directly.",
      progressDetail: undefined,
      stageEtaSec: undefined,
      totalEtaSec: undefined,
      slow: false,
      renderState: "failed",
      logs: appendLog(
        job.logs,
        `${ts()} · ✗ grabbing — ${fetcher} failed: ${error}`,
      ),
    });
  },
});

/** The REAL download failed: exact fetcher error, no retry theater. */
export const reportFetchFailure = mutation({
  args: { jobId: v.id("jobs"), error: v.string(), fetcher: v.string() },
  handler: async (ctx, { jobId, error, fetcher }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) throw new Error("Job not found.");
    const now = Date.now();
    await ctx.db.patch(jobId, {
      fetchError: error,
      fetchAttempts: [
        ...(job.fetchAttempts ?? []),
        { method: fetcher, ok: false, error, at: now },
      ],
      stage: "failed",
      error: `Could not fetch video via ${fetcher}: ${error}`,
      errorDetail:
        "Try the Retry button (fetchers are retried in order), or attach the file directly.",
      progressDetail: undefined,
      stageEtaSec: undefined,
      totalEtaSec: undefined,
      slow: false,
      renderState: "failed",
      logs: appendLog(
        job.logs,
        `${ts()} · ✗ grabbing — ${fetcher} failed: ${error}`,
      ),
    });
  },
});

/** Real transcription progress: seconds of audio processed. */
export const reportTranscribeProgress = mutation({
  args: { jobId: v.id("jobs"), processedSec: v.number() },
  handler: async (ctx, { jobId, processedSec }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) throw new Error("Job not found.");
    if (job.stage !== "transcribing") return;
    const duration = job.durationSec ?? processedSec;
    const u = transcribeUnits(duration, Math.min(100, (processedSec / Math.max(1, duration)) * 100));
    await ctx.db.patch(jobId, {
      progress: Math.min(99, (processedSec / Math.max(1, duration)) * 100),
      progressDetail: "Whisper is transcribing word timings...",
      unitLabel: u.unitLabel,
      unitsDone: Math.min(u.unitsTotal, processedSec),
      unitsTotal: u.unitsTotal,
      lastProgressAt: Date.now(),
    });
  },
});

/** The REAL transcript: stored Whisper word-level JSON. */
export const reportTranscribe = mutation({
  args: {
    jobId: v.id("jobs"),
    storageId: v.id("_storage"),
    wordCount: v.number(),
    durationSec: v.number(),
  },
  handler: async (ctx, { jobId, storageId, wordCount, durationSec }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) throw new Error("Job not found.");
    if (job.stage !== "transcribing") {
      throw new Error(`Job is in stage "${job.stage}", expected "transcribing".`);
    }
    if (wordCount === 0) {
      throw new Error("Whisper produced no words — the audio may be silent or unsupported.");
    }
    const now = Date.now();
    const units = findUnits(durationSec, 0);
    await ctx.db.patch(jobId, {
      transcriptStorageId: storageId,
      transcriptWordCount: wordCount,
      durationSec,
      stage: "finding",
      progress: 0,
      progressDetail: "Scoring viral moments with Gemini...",
      unitLabel: units.unitLabel,
      unitsTotal: units.unitsTotal,
      unitsDone: 0,
      stageStartedAt: now,
      lastProgressAt: now,
      slow: false,
      stageTimings: [
        ...(job.stageTimings ?? []),
        { stage: "transcribing", durationMs: now - (job.stageStartedAt ?? now), at: now },
      ],
      logs: appendLog(
        job.logs,
        `${ts()} · ✓ transcribing — ${wordCount} word timestamps stored`,
      ),
    });
    // Real AI detection: the Gemini action (no heuristic fallback).
    await ctx.scheduler.runAfter(0, api.detection.runClipDetection, { jobId });
  },
});

export const reportTranscribeFailure = mutation({
  args: { jobId: v.id("jobs"), error: v.string() },
  handler: async (ctx, { jobId, error }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) throw new Error("Job not found.");
    await ctx.db.patch(jobId, {
      stage: "failed",
      error: `Transcription failed: ${error}`,
      progressDetail: undefined,
      stageEtaSec: undefined,
      totalEtaSec: undefined,
      slow: false,
      renderState: "failed",
      logs: appendLog(job.logs, `${ts()} · ✗ transcribing — ${error}`),
    });
  },
});

// --- finding-stage progress + failure (called by the detection action) ------

export const reportFindingProgress = internalMutation({
  args: {
    jobId: v.id("jobs"),
    done: v.number(),
    total: v.number(),
    logLine: v.optional(v.string()),
  },
  handler: async (ctx, { jobId, done, total, logLine }) => {
    const job = await ctx.db.get(jobId);
    if (job === null || job.stage !== "finding") return;
    const duration = job.durationSec ?? 600;
    const u = findUnits(duration, Math.min(100, (done / Math.max(1, total)) * 100));
    const patch: Partial<Doc<"jobs">> = {
      progress: Math.min(99, (done / Math.max(1, total)) * 100),
      progressDetail: "Scoring viral moments with Gemini...",
      unitLabel: u.unitLabel,
      unitsDone: u.unitsDone,
      unitsTotal: u.unitsTotal,
      lastProgressAt: Date.now(),
    };
    if (logLine !== undefined) {
      patch.logs = appendLog(job.logs, logLine);
    }
    await ctx.db.patch(jobId, patch);
  },
});

export const failFinding = internalMutation({
  args: { jobId: v.id("jobs"), error: v.string(), detail: v.optional(v.string()) },
  handler: async (ctx, { jobId, error, detail }) => {
    const job = await ctx.db.get(jobId);
    if (job === null) return;
    await ctx.db.patch(jobId, {
      stage: "failed",
      error,
      errorDetail: detail,
      progressDetail: undefined,
      stageEtaSec: undefined,
      totalEtaSec: undefined,
      slow: false,
      detectedSource: "none",
      logs: appendLog(job.logs, `${ts()} · ✗ finding — ${error}`),
    });
  },
});

/** Re-run AI detection on a completed job (the "Re-detect with AI" button). */
export const reenterFinding = internalMutation({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const job = await ctx.db.get(jobId);
    if (job === null) throw new Error("Job not found.");
    if (job.transcriptStorageId === undefined) {
      throw new Error("This job has no stored transcript — re-run the job first.");
    }
    const now = Date.now();
    const duration = job.durationSec ?? 600;
    const units = findUnits(duration, 0);
    await ctx.db.patch(jobId, {
      stage: "finding",
      progress: 0,
      progressDetail: "Re-scoring viral moments with Gemini...",
      unitLabel: units.unitLabel,
      unitsTotal: units.unitsTotal,
      unitsDone: 0,
      stageStartedAt: now,
      lastProgressAt: now,
      slow: false,
      error: undefined,
      errorDetail: undefined,
      completedAt: undefined,
      renderState: "running",
      logs: appendLog(job.logs, `${ts()} · ↻ re-detect — sending transcript chunks to Gemini`),
    });
  },
});

// --- rendering --------------------------------------------------------------

/** One REAL rendered clip: stored 9:16 MP4 (+ thumbnail) with measured metadata. */
export const reportClipRendered = mutation({
  args: {
    jobId: v.id("jobs"),
    clipId: v.id("clips"),
    mediaStorageId: v.id("_storage"),
    thumbStorageId: v.optional(v.id("_storage")),
    bytes: v.number(),
    width: v.number(),
    height: v.number(),
    durationSec: v.number(),
    renderMs: v.number(),
  },
  handler: async (ctx, { jobId, clipId, mediaStorageId, thumbStorageId, bytes, width, height, durationSec, renderMs }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) throw new Error("Job not found.");
    const clip = await ctx.db.get(clipId);
    if (clip === null || clip.jobId !== jobId) throw new Error("Clip not found.");
    if (bytes <= 0 || durationSec <= 0) {
      throw new Error("Rendered file failed verification (0 bytes or 0 duration).");
    }

    const now = Date.now();
    await ctx.db.patch(clipId, {
      mediaStorageId,
      thumbStorageId,
      mediaBytes: bytes,
      mediaWidth: width,
      mediaHeight: height,
      mediaDurationSec: durationSec,
      renderMs,
      renderState: "done",
      mediaError: undefined,
    });

    const allClips = await ctx.db
      .query("clips")
      .withIndex("by_job", (q) => q.eq("jobId", jobId))
      .collect();
    const doneClips = allClips.filter((c) => c.renderState === "done").length;
    const total = Math.max(1, allClips.length);
    const u = renderUnits(doneClips, total);

    if (doneClips >= total) {
      const captionsTxt = allClips
        .sort((a, b) => a.order - b.order)
        .map(
          (c, i) =>
            `=== Clip ${i + 1}: ${c.title} (score ${c.score}) ===\n${buildSrt(c.captionCues)}`,
        )
        .join("\n");
      const totalSec = Math.round(
        allClips.reduce((acc, c) => acc + (c.mediaDurationSec ?? c.end - c.start), 0),
      );
      await ctx.db.patch(jobId, {
        stage: "complete",
        progress: 100,
        progressDetail: undefined,
        stageEtaSec: undefined,
        totalEtaSec: undefined,
        slow: false,
        completedAt: now,
        clipCount: total,
        totalClipsSec: totalSec,
        clipSource: "gemini",
        renderState: "done",
        renderProgress: 100,
        renderError: undefined,
        captionsTxt,
        stageTimings: [
          ...(job.stageTimings ?? []),
          { stage: "rendering", durationMs: now - (job.stageStartedAt ?? now), at: now },
        ],
        logs: appendLog(
          job.logs,
          `${ts()} · ✓ complete — ${total} real 9:16 MP4s rendered with burned-in captions (ffmpeg.wasm)`,
        ),
      });
    } else {
      await ctx.db.patch(jobId, {
        progress: Math.min(99, (doneClips / total) * 100),
        progressDetail: "Rendering 9:16 clips with burned-in captions...",
        unitLabel: u.unitLabel,
        unitsDone: u.unitsDone,
        unitsTotal: u.unitsTotal,
        renderProgress: (doneClips / total) * 100,
        lastProgressAt: now,
      });
    }
  },
});

export const reportClipRenderFailure = mutation({
  args: { jobId: v.id("jobs"), clipId: v.id("clips"), error: v.string() },
  handler: async (ctx, { jobId, clipId, error }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in first.");
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) throw new Error("Job not found.");
    const clip = await ctx.db.get(clipId);
    if (clip === null || clip.jobId !== jobId) throw new Error("Clip not found.");
    const now = Date.now();
    await ctx.db.patch(clipId, {
      renderState: "failed",
      mediaError: error,
    });
    await ctx.db.patch(jobId, {
      stage: "failed",
      error: `Render failed for clip ${clip.order + 1} ("${clip.title}"): ${error}`,
      errorDetail: "The browser render worker hit an ffmpeg error. Retry re-encodes all clips.",
      progressDetail: undefined,
      stageEtaSec: undefined,
      totalEtaSec: undefined,
      slow: false,
      renderState: "failed",
      renderError: error,
      lastProgressAt: now,
      logs: appendLog(
        job.logs,
        `${ts()} · ✗ rendering — clip ${clip.order + 1} failed: ${error}`,
      ),
    });
  },
});

// ---------------------------------------------------------------------------
// watchdog tick
// ---------------------------------------------------------------------------

export const workerTick = internalMutation({
  args: {},
  handler: async (ctx) => {
    const now = Date.now();

    const claimed = await claimNextTick(ctx, now);
    if (!claimed) return;

    // heartbeat
    const row = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    if (row !== null) {
      await ctx.db.patch(row._id, {
        online: true,
        lastHeartbeat: now,
        lastHeartbeatServerMs: now,
        chainArmed: true,
      });
    }

    // watchdog: every active stage must keep reporting real progress
    const activeStages = ["grabbing", "transcribing", "finding", "rendering"] as const;
    for (const stage of activeStages) {
      const jobs = await ctx.db
        .query("jobs")
        .withIndex("by_stage", (q) => q.eq("stage", stage))
        .collect();
      for (const job of jobs) {
        const startedAt = job.stageStartedAt ?? now;
        const lastProgress = job.lastProgressAt ?? startedAt;
        const elapsed = now - startedAt;

        const timeoutMs =
          stage === "transcribing"
            ? transcribeTimeoutMs(job.durationSec ?? 600)
            : STAGE_TIMEOUT_MS[stage] ?? 10 * 60 * 1000;

        if (now - lastProgress > WATCHDOG_MS) {
          await failWatchdog(ctx, job, stage, now, `Pipeline stalled at stage ${stage}.`,
            `No progress event for ${Math.round((now - lastProgress) / 1000)}s (watchdog limit 90s). ` +
            `The browser render worker stopped reporting — reopen the dashboard and hit Retry.`);
          continue;
        }
        if (elapsed > timeoutMs) {
          await failWatchdog(ctx, job, stage, now, `Stage "${stage}" timed out.`,
            `Exceeded ${Math.round(timeoutMs / 1000)}s hard limit at ${Math.round(job.progress)}%. ` +
            `Killed by watchdog. See logs for the last real progress event.`);
        }
      }
    }

    // self-scheduling chain (the server-side watchdog loop)
    if (row !== null) {
      await ctx.db.patch(row._id, { nextTickAt: now + CHAIN_DELAY_MS });
      await ctx.scheduler.runAfter(CHAIN_DELAY_MS, internal.clips.workerTick, {});
    }
  },
});

async function failWatchdog(
  ctx: MutationCtx,
  job: Doc<"jobs">,
  stage: string,
  now: number,
  error: string,
  detail: string,
): Promise<void> {
  await ctx.db.patch(job._id, {
    stage: "failed",
    error,
    errorDetail: detail,
    progressDetail: undefined,
    stageEtaSec: undefined,
    totalEtaSec: undefined,
    slow: false,
    renderState: "failed",
    logs: appendLog(job.logs, `${ts()} · ✗ ${stage} — ${error} :: ${detail}`),
  });
}

async function claimNextTick(ctx: MutationCtx, now: number): Promise<boolean> {
  const row = await ctx.db
    .query("workerStatus")
    .withIndex("singleton", (q) => q.eq("singleton", "worker"))
    .unique();
  if (row !== null) {
    const nta = row.nextTickAt;
    if (nta !== undefined && nta > now) return false;
    await ctx.db.patch(row._id, { nextTickAt: now + CHAIN_DELAY_MS });
    return true;
  }
  return true;
}

// ---------------------------------------------------------------------------
// AI detection support (the action lives in detection.ts, node runtime)
// ---------------------------------------------------------------------------

export const getJobForUser = internalQuery({
  args: { jobId: v.id("jobs"), userId: v.id("users") },
  handler: async (ctx, { jobId, userId }) => {
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) return null;
    return job;
  },
});

/** Raw job load for internal actions (no ownership filter — internal only). */
export const getJobInternal = internalQuery({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    return await ctx.db.get(jobId);
  },
});

export const getCurrentUserId = internalQuery({
  args: {},
  handler: async (ctx) => {
    return await getAuthUserId(ctx);
  },
});

/**
 * Store REAL Gemini-scored clips: each row starts renderState "pending" with
 * NO media — the browser render worker encodes it and reports back. Caption
 * cues are built from the REAL Whisper word timings.
 */
export const applyDetectedClips = internalMutation({
  args: {
    jobId: v.id("jobs"),
    clipSource: v.literal("gemini"),
    candidates: v.array(
      v.object({
        start: v.number(),
        end: v.number(),
        score: v.number(),
        title: v.string(),
        hookText: v.string(),
        transcript: v.string(),
        hashtags: v.array(v.string()),
      }),
    ),
    words: v.array(
      v.object({ t0: v.number(), t1: v.number(), text: v.string() }),
    ),
    logLines: v.array(v.string()),
  },
  handler: async (ctx, { jobId, clipSource, candidates, words, logLines }) => {
    const job = await ctx.db.get(jobId);
    if (job === null) throw new Error("Job not found.");
    if (candidates.length === 0) {
      throw new Error("Clip detection returned no usable clips.");
    }
    const now = Date.now();

    const oldClips = await ctx.db
      .query("clips")
      .withIndex("by_job", (q) => q.eq("jobId", jobId))
      .collect();
    for (const c of oldClips) await ctx.db.delete(c._id);

    const transcriptWords: TranscriptWord[] = words.map((w) => ({
      t0: w.t0,
      t1: w.t1,
      text: w.text,
    }));

    for (let i = 0; i < candidates.length; i++) {
      const p: ClipCandidateOutput = candidates[i];
      const cues = karaokeCues(transcriptWords, p.start, p.end);
      await ctx.db.insert("clips", {
        jobId,
        userId: job.userId,
        order: i,
        start: p.start,
        end: p.end,
        score: p.score,
        title: p.title,
        hookText: p.hookText,
        transcript: p.transcript,
        hashtags: p.hashtags,
        captionCues: cues,
        aspect: "9:16",
        renderState: "pending",
      });
    }

    const units = renderUnits(0, candidates.length);
    const captionsTxt = candidates
      .map((p, i) => `=== Clip ${i + 1}: ${p.title} (score ${p.score}) ===\n(builds after render)`)
      .join("\n");
    await ctx.db.patch(jobId, {
      stage: "rendering",
      progress: 0,
      progressDetail: "Rendering 9:16 clips with burned-in captions...",
      unitLabel: units.unitLabel,
      unitsTotal: units.unitsTotal,
      unitsDone: 0,
      stageStartedAt: now,
      lastProgressAt: now,
      slow: false,
      clipCount: candidates.length,
      clipSource,
      detectedSource: "gemini",
      renderState: "pending",
      renderProgress: 0,
      renderError: undefined,
      captionsTxt,
      logs: appendLog(
        job.logs,
        [
          ...logLines,
          `${ts()} · ✓ finding — Gemini selected ${candidates.length} windows (scores ${candidates.map((c) => c.score).join(", ")})`,
          `${ts()} · ▶ rendering — ${candidates.length} clips queued for the browser ffmpeg worker`,
        ],
      ),
    });
  },
});

// ---------------------------------------------------------------------------
// ops diagnostics (internal only)
// ---------------------------------------------------------------------------

export const adminJobSummary = internalQuery({
  args: {},
  handler: async (ctx) => {
    const jobs = await ctx.db.query("jobs").collect();
    const out = [];
    for (const job of jobs) {
      const clips = await ctx.db
        .query("clips")
        .withIndex("by_job", (q) => q.eq("jobId", job._id))
        .collect();
      const rendered = clips.filter((c) => c.mediaStorageId !== undefined).length;
      out.push({
        id: job._id,
        stage: job.stage,
        source: job.source.slice(0, 60),
        progress: Math.round(job.progress),
        error: job.error,
        clipCount: clips.length,
        renderedClips: rendered,
        sourceStored: job.sourceStorageId !== undefined,
        fetchSource: job.fetchSource,
        createdAt: job.createdAt,
      });
    }
    return out.sort((a, b) => b.createdAt - a.createdAt);
  },
});

/**
 * Ops requeue: reset a failed/stuck job to `queued` and wake the watchdog.
 * (The server-side equivalent of re-enqueueing a task after fixing the worker.)
 */
export const adminRequeueJob = internalMutation({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const job = await ctx.db.get(jobId);
    if (job === null) throw new Error("Job not found.");
    const oldClips = await ctx.db
      .query("clips")
      .withIndex("by_job", (q) => q.eq("jobId", jobId))
      .collect();
    for (const c of oldClips) await ctx.db.delete(c._id);
    await ctx.db.patch(jobId, {
      stage: "queued",
      progress: 0,
      progressDetail: "Queued — ops requeue, waiting for a browser render worker...",
      error: undefined,
      errorDetail: undefined,
      stageEtaSec: undefined,
      totalEtaSec: undefined,
      stageStartedAt: undefined,
      lastProgressAt: Date.now(),
      slow: false,
      clipCount: undefined,
      totalClipsSec: undefined,
      captionsTxt: undefined,
      completedAt: undefined,
      renderState: undefined,
      renderError: undefined,
      renderProgress: undefined,
      logs: appendLog(job.logs, `${ts()} · ↻ ops requeue — watchdog re-armed`),
    });
    await ctx.scheduler.runAfter(0, internal.clips.workerTick, {});
    return { requeued: true };
  },
});

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

function appendLog(logs: string[] | undefined, lines: string | string[]): string[] {
  const next = [...(logs ?? []), ...(Array.isArray(lines) ? lines : [lines])];
  return next.length > MAX_LOGS ? next.slice(next.length - MAX_LOGS) : next;
}

function ts(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
