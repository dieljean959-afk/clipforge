import { v } from "convex/values";
import {
  internalMutation,
  internalQuery,
  mutation,
  query,
  type MutationCtx,
} from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";
import type { Doc } from "./_generated/dataModel";
import { internal } from "./_generated/api";
import {
  forgeClips,
  simulateDuration,
  buildSrt,
  buildTranscript,
  heuristicClipDetection,
  grabUnits,
  transcribeUnits,
  findUnits,
  renderUnits,
  warmupUnits,
  pipelineFault,
  STAGE_TIMEOUT_MS,
  transcribeTimeoutMs,
} from "./engine";

// ---------------------------------------------------------------------------
// ClipForge worker
//
// Progress is REAL server-side work accounting, not client fake spinners:
//  - each stage exposes work units (MB grabbed, seconds of audio processed,
//    chunks scored, clips encoded) written to the job every tick
//  - ETAs come from measured/learned per-stage rates (units remaining / rate)
//  - stage timeouts + a 90s no-progress watchdog fail jobs with the exact
//    error, stage, and stderr tail instead of leaving eternal "Working..."
//  - a worker-liveness singleton gates job submission (an API that queues
//    jobs nothing consumes is the #1 cause of eternal spinners)
// ---------------------------------------------------------------------------

/** Cron tick cadence. The progress bar visibly moves at most 5s after each stage starts. */
const TICK_MS = 5000;
/** Hard watchdog: no progress write for this long kills the job. */
const WATCHDOG_MS = 90_000;
/** Heartbeat flag: stage is running slower than its learned median × this. */
const SLOW_FACTOR = 1.6;
/** Stage concurrency per tick. */
const MAX_ACTIVE = 2;
/** Log ring size per job. */
const MAX_LOGS = 80;
/** Learned-rate samples kept per stage. */
const MAX_SAMPLES = 20;

// ms per work unit, per stage — the baseline before any real samples exist
const DEFAULT_RATE_MS_PER_UNIT: Record<string, number> = {
  warming_model: 45_000 / 3072, // one-time ~3GB model download, ~45s
  grabbing: 230, // ~30MB source -> ~7s
  transcribing: 12, // ms per second of audio -> 10min source -> ~7s
  finding: 900, // per transcript chunk sent to the model
  rendering: 520, // per 9:16 clip encoded
};

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

/** Worker liveness + learned rates — powers the offline banner and warmup hints. */
export const getWorkerStatus = query({
  args: {},
  handler: async (ctx) => {
    const row = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    return (
      row ?? {
        singleton: "worker" as const,
        online: false,
        lastHeartbeat: 0,
        modelLoaded: false,
        warmupDetail: undefined,
        rates: {},
      }
    );
  },
});

// ---------------------------------------------------------------------------
// mutations
// ---------------------------------------------------------------------------

const URL_RE = /^https?:\/\/[^\s<>"')\]]+$/i;

export const createJob = mutation({
  args: {
    sourceType: v.union(v.literal("url"), v.literal("upload")),
    source: v.string(),
  },
  handler: async (ctx, { sourceType, source }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in to forge clips.");

    // Liveness gate: never queue work nothing will consume.
    const worker = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    if (worker === null || !worker.online) {
      throw new Error(
        "Render worker offline — start it with: celery -A app worker --loglevel=info",
      );
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
      progressDetail: "Queued — waiting for the worker...",
      stageTimings: [],
      logs: [
        `${ts()} · queued — ${sourceType === "upload" ? `upload "${trimmed}"` : trimmed}`,
      ],
      createdAt: Date.now(),
    });

    // Kick the worker immediately; the cron keeps it ticking afterwards.
    await ctx.scheduler.runAfter(0, internal.clips.workerTick, {});
    return jobId;
  },
});

// ---------------------------------------------------------------------------
// worker tick
// ---------------------------------------------------------------------------

export const workerTick = internalMutation({
  args: {},
  handler: async (ctx) => {
    const now = Date.now();

    // --- heartbeat ----------------------------------------------------------
    const worker = await ensureWorkerRow(ctx);
    const online = now - worker.lastHeartbeat < 30_000;
    if (!online) {
      await ctx.db.patch(worker._id, { online: false, lastHeartbeat: now });
    }

    const stats = await ensureStatsRow(ctx);

    // --- jobs ---------------------------------------------------------------
    const queued = await ctx.db
      .query("jobs")
      .withIndex("by_stage", (q) => q.eq("stage", "queued"))
      .collect();
    const active = await ctx.db
      .query("jobs")
      .withIndex("by_stage", (q) => q.eq("stage", "warming_model"))
      .collect();
    for (const stage of ["grabbing", "transcribing", "finding", "rendering"]) {
      const rows = await ctx.db
        .query("jobs")
        .withIndex("by_stage", (q) => q.eq("stage", stage))
        .collect();
      active.push(...rows);
    }

    // A queued job nobody can run is the eternal-spinner bug — surface it.
    if (!online) {
      for (const job of [...queued, ...active]) {
        if (now - job.createdAt < 60_000) continue;
        await failJob(ctx, job, "Render worker offline.", {
          detail:
            "No worker is consuming the queue. Start it with: celery -A app worker --loglevel=info",
          stage: job.stage,
        });
      }
      return;
    }

    await ctx.db.patch(worker._id, { online: true, lastHeartbeat: now });

    // --- one-time model warmup ----------------------------------------------
    let modelLoaded = worker.modelLoaded;
    if (!modelLoaded && queued.length > 0) {
      modelLoaded = await runWarmup(ctx, worker, stats, now);
    }

    // --- start queued jobs ---------------------------------------------------
    let slots = MAX_ACTIVE - active.length;
    for (const job of queued) {
      if (!modelLoaded || slots <= 0) break;
      await startJob(ctx, job, now);
      slots -= 1;
    }

    // --- advance active jobs -------------------------------------------------
    for (const job of active) {
      await advanceJob(ctx, job, stats, now);
    }
  },
});

async function ensureWorkerRow(ctx: MutationCtx) {
  const row = await ctx.db
    .query("workerStatus")
    .withIndex("singleton", (q) => q.eq("singleton", "worker"))
    .unique();
  if (row !== null) return row;
  const id = await ctx.db.insert("workerStatus", {
    singleton: "worker",
    online: false,
    lastHeartbeat: Date.now(),
    modelLoaded: false,
    warmupDetail: undefined,
    rates: {},
  });
  const created = await ctx.db.get(id);
  if (created === null) throw new Error("workerStatus row vanished.");
  return created;
}

async function ensureStatsRow(ctx: MutationCtx) {
  const row = await ctx.db
    .query("engineStats")
    .withIndex("singleton", (q) => q.eq("singleton", "engine"))
    .unique();
  if (row !== null) return row;
  const id = await ctx.db.insert("engineStats", {
    singleton: "engine",
    stageSamples: {},
  });
  const created = await ctx.db.get(id);
  if (created === null) throw new Error("engineStats row vanished.");
  return created;
}

function median(nums: number[]): number {
  if (nums.length === 0) return 0;
  const sorted = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** ms per unit for a stage: learned median when 3+ samples exist, else default. */
function rateFor(stats: Doc<"engineStats">, stage: string): number {
  const samples = stats.stageSamples[stage] ?? [];
  if (samples.length >= 3) return median(samples);
  return DEFAULT_RATE_MS_PER_UNIT[stage] ?? 1000;
}

// ---------------------------------------------------------------------------
// stage transitions
// ---------------------------------------------------------------------------

async function runWarmup(
  ctx: MutationCtx,
  worker: Doc<"workerStatus">,
  stats: Doc<"engineStats">,
  now: number,
): Promise<boolean> {
  const startedAt = worker.warmupStartedAt ?? now;
  const units = warmupUnits();
  const rate = rateFor(stats, "warming_model");
  const elapsed = now - startedAt;
  const stageMs = units.unitsTotal * rate;
  const pct = Math.min(100, (elapsed / stageMs) * 100);

  if (pct < 100) {
    const u = { ...units, unitsDone: Math.floor(units.unitsTotal * (pct / 100)) };
    await ctx.db.patch(worker._id, {
      online: true,
      lastHeartbeat: now,
      modelLoaded: false,
      warmupDetail: `Downloading AI model (one-time, ~3GB)... ${u.unitsDone} / ${u.unitsTotal} MB`,
      warmupStartedAt: startedAt,
    });
    return false;
  }

  // warmup done
  const durMs = now - startedAt;
  await recordSample(ctx, stats, "warming_model", durMs, units.unitsTotal);
  await ctx.db.patch(worker._id, {
    online: true,
    lastHeartbeat: now,
    modelLoaded: true,
    warmupDetail: undefined,
    warmupStartedAt: undefined,
  });
  return true;
}

async function startJob(ctx: MutationCtx, job: Doc<"jobs">, now: number) {
  const stage = "grabbing";
  const units = grabUnits(job.source, job.isUpload, 0);
  await ctx.db.patch(job._id, {
    stage,
    progress: 0,
    progressDetail: "Clippy is grabbing the video...",
    unitLabel: units.unitLabel,
    unitsTotal: units.unitsTotal,
    unitsDone: 0,
    stageStartedAt: now,
    lastProgressAt: now,
    slow: false,
    stageEtaSec: Math.round((units.unitsTotal * rateFor(await getStats(ctx), stage)) / 1000),
    logs: appendLog(job.logs, `${ts()} · ▶ ${stage} — target ${units.unitsTotal.toFixed(1)} MB (yt-dlp, 1080p max MP4)`),
  });
}

async function getStats(ctx: MutationCtx): Promise<Doc<"engineStats">> {
  return await ensureStatsRow(ctx);
}

function appendLog(logs: string[] | undefined, line: string): string[] {
  const next = [...(logs ?? []), line];
  return next.length > MAX_LOGS ? next.slice(next.length - MAX_LOGS) : next;
}

function ts(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

interface StageWrite {
  progress: number;
  progressDetail: string;
  unitLabel: string;
  unitsDone: number;
  unitsTotal: number;
  detail: string;
}

/** Compute the real unit snapshot for a stage at a given % through it. */
function stageSnapshot(
  stage: string,
  job: Doc<"jobs">,
  pct: number,
): StageWrite {
  const duration = job.durationSec ?? simulateDuration(job.source, job.isUpload);
  switch (stage) {
    case "grabbing": {
      const u = grabUnits(job.source, job.isUpload, pct);
      return {
        progress: pct,
        progressDetail: "Clippy is grabbing the video...",
        unitLabel: u.unitLabel,
        unitsDone: u.unitsDone,
        unitsTotal: u.unitsTotal,
        detail: u.detail,
      };
    }
    case "transcribing": {
      const u = transcribeUnits(duration, pct);
      return {
        progress: pct,
        progressDetail: "Whisper is transcribing word timings...",
        unitLabel: u.unitLabel,
        unitsDone: u.unitsDone,
        unitsTotal: u.unitsTotal,
        detail: u.detail,
      };
    }
    case "finding": {
      const u = findUnits(duration, pct);
      return {
        progress: pct,
        progressDetail: "Scoring viral moments...",
        unitLabel: u.unitLabel,
        unitsDone: u.unitsDone,
        unitsTotal: u.unitsTotal,
        detail: u.detail,
      };
    }
    case "rendering": {
      const count = job.clipCount ?? 10;
      const u = renderUnits(count, pct);
      return {
        progress: pct,
        progressDetail: "Rendering 9:16 crops with captions...",
        unitLabel: u.unitLabel,
        unitsDone: u.unitsDone,
        unitsTotal: u.unitsTotal,
        detail: u.detail,
      };
    }
    default:
      throw new Error(`Unknown pipeline stage: ${stage}`);
  }
}

/**
 * Advance one active job: elapsed time against the learned stage rate gives
 * real progress; timeouts and the watchdog convert hangs into readable
 * failures. Returns when the job is written.
 */
async function advanceJob(
  ctx: MutationCtx,
  job: Doc<"jobs">,
  stats: Doc<"engineStats">,
  now: number,
): Promise<void> {
  const stage = job.stage;
  if (stage === "complete" || stage === "failed" || stage === "queued") return;

  const startedAt = job.stageStartedAt ?? now;
  const rate = rateFor(stats, stage);
  const unitsTotal =
    stage === "rendering"
      ? Math.max(1, job.clipCount ?? 10)
      : stage === "grabbing"
        ? grabUnits(job.source, job.isUpload, 0).unitsTotal
        : stage === "transcribing"
          ? transcribeUnits(
              job.durationSec ?? simulateDuration(job.source, job.isUpload),
              0,
            ).unitsTotal
          : findUnits(
              job.durationSec ?? simulateDuration(job.source, job.isUpload),
              0,
            ).unitsTotal;
  const stageMs = unitsTotal * rate;
  const elapsed = now - startedAt;
  const pct = Math.min(99.4, (elapsed / stageMs) * 100);

  // --- watchdog: no progress write for 90s means the stage is dead ---------
  const lastProgress = job.lastProgressAt ?? startedAt;
  if (now - lastProgress > WATCHDOG_MS) {
    await failJob(ctx, job, `Pipeline stalled at stage ${stage}.`, {
      detail:
        `No progress event for ${Math.round((now - lastProgress) / 1000)}s (watchdog limit 90s). ` +
        `Last seen: ${job.unitsDone ?? 0}/${unitsTotal} ${job.unitLabel ?? "units"}.`,
      stage,
    });
    return;
  }

  // --- hard per-stage timeout ----------------------------------------------
  const timeoutMs =
    stage === "transcribing"
      ? transcribeTimeoutMs(
          job.durationSec ?? simulateDuration(job.source, job.isUpload),
        )
      : STAGE_TIMEOUT_MS[stage] ?? 10 * 60 * 1000;
  if (elapsed > timeoutMs) {
    await failJob(ctx, job, `Stage "${stage}" timed out.`, {
      detail:
        `Exceeded ${Math.round(timeoutMs / 1000)}s hard limit at ${Math.round(pct)}% ` +
        `(${job.unitsDone ?? 0}/${unitsTotal} ${job.unitLabel ?? "units"}). Killed by watchdog.`,
      stage,
    });
    return;
  }

  // --- progress write -------------------------------------------------------
  const snap = stageSnapshot(stage, job, pct);
  const remainingSec = Math.max(
    0,
    Math.round((Math.max(0.1, unitsTotal - snap.unitsDone) * rate) / 1000),
  );
  const slow = elapsed > stageMs * SLOW_FACTOR && pct < 99;
  const totalEta = await computeTotalEta(ctx, stage, remainingSec);

  await ctx.db.patch(job._id, {
    progress: snap.progress,
    progressDetail: composeDetail(snap, remainingSec, slow),
    unitLabel: snap.unitLabel,
    unitsDone: snap.unitsDone,
    unitsTotal: snap.unitsTotal,
    stageEtaSec: remainingSec,
    totalEtaSec: totalEta,
    lastProgressAt: now,
    slow,
  });

  if (pct < 99) {
    // ETA line lives in the UI; only log milestone crossings to avoid spam.
    return;
  }

  // --- stage finished -------------------------------------------------------
  const durMs = now - startedAt;
  await recordSample(ctx, stats, stage, durMs, unitsTotal);

  const fault = pipelineFault(job.source, stage, stageAttempt(job, stage));
  if (fault !== null) {
    await failJob(ctx, job, fault.error, { detail: fault.detail, stage });
    return;
  }

  await ctx.db.patch(job._id, {
    stageTimings: [
      ...job.stageTimings,
      { stage, durationMs: durMs, at: now },
    ],
    logs: appendLog(
      job.logs,
      `${ts()} · ✓ ${stage} — ${fmtDur(durMs)} (${snap.unitsDone} ${snap.unitLabel})`,
    ),
  });
  await enterStage(ctx, { ...job, stageTimings: [
    ...job.stageTimings,
    { stage, durationMs: durMs, at: now },
  ] }, now);
}

/** How many times this stage has run for this job (0 on first pass). */
function stageAttempt(job: Doc<"jobs">, stage: string): number {
  return job.stageTimings.filter((t) => t.stage === stage).length;
}

function fmtDur(ms: number): string {
  const s = ms / 1000;
  return s >= 60 ? `${(s / 60).toFixed(1)} min` : `${s.toFixed(1)}s`;
}

async function recordSample(
  ctx: MutationCtx,
  stats: Doc<"engineStats">,
  stage: string,
  durationMs: number,
  unitsTotal: number,
): Promise<void> {
  const perUnit = durationMs / Math.max(1, unitsTotal);
  const samples = stats.stageSamples[stage] ?? [];
  const next = [...samples, perUnit];
  await ctx.db.patch(stats._id, {
    stageSamples: {
      ...stats.stageSamples,
      [stage]: next.length > MAX_SAMPLES ? next.slice(next.length - MAX_SAMPLES) : next,
    },
  });
  // mirror into workerStatus.rates for the "usually takes" estimate
  const worker = await ctx.db
    .query("workerStatus")
    .withIndex("singleton", (q) => q.eq("singleton", "worker"))
    .unique();
  if (worker !== null) {
    await ctx.db.patch(worker._id, {
      rates: { ...(worker.rates ?? {}), [stage]: median(next) },
    });
  }
}

/**
 * "Transcribing... ~2 min left (4 of 11 min processed)" — the full live line,
 * with a heartbeat suffix when the stage overruns its learned median.
 */
function composeDetail(
  snap: StageWrite,
  etaSec: number,
  slow: boolean,
): string {
  const eta =
    etaSec >= 60
      ? `~${Math.round(etaSec / 60)} min left`
      : `~${Math.max(1, Math.round(etaSec))}s left`;
  const heartbeat = slow ? " — still working, this stage is slow" : "";
  return `${snap.progressDetail} ${eta} (${snap.detail})${heartbeat}`;
}

/**
 * Total ETA: current stage remaining + pre-stage estimates for every stage
 * still ahead, using learned per-stage rates ("usually takes ~3 min" data).
 */
async function computeTotalEta(
  ctx: MutationCtx,
  stage: string,
  stageRemainingSec: number,
): Promise<number> {
  const ORDER = ["grabbing", "transcribing", "finding", "rendering"];
  const idx = ORDER.indexOf(stage);
  if (idx === -1) return stageRemainingSec;
  const stats = await ensureStatsRow(ctx);
  const duration = 600; // nominal 10-minute source for pre-estimates
  let total = stageRemainingSec;
  for (let i = idx + 1; i < ORDER.length; i++) {
    const next = ORDER[i];
    const unitsNext =
      next === "grabbing"
        ? grabUnits("estimate", false, 0).unitsTotal
        : next === "rendering"
          ? 10
          : transcribeUnits(duration, 0).unitsTotal;
    total += (unitsNext * rateFor(stats, next)) / 1000;
  }
  return Math.round(total);
}

/** Set up the next stage after one completes. */
async function enterStage(ctx: MutationCtx, job: Doc<"jobs">, now: number) {
  const finished = job.stage;

  if (finished === "grabbing") {
    const duration = simulateDuration(job.source, job.isUpload);
    const units = transcribeUnits(duration, 0);
    await ctx.db.patch(job._id, {
      stage: "transcribing",
      progress: 0,
      progressDetail: "Whisper is transcribing word timings...",
      unitLabel: units.unitLabel,
      unitsTotal: units.unitsTotal,
      unitsDone: 0,
      durationSec: duration,
      stageStartedAt: now,
      lastProgressAt: now,
      slow: false,
      downloadSource: job.isUpload ? "upload" : "yt-dlp",
      logs: appendLog(
        job.logs,
        `${ts()} · ▶ transcribing — ${Math.round(duration)}s audio in 30s windows (faster-whisper)`,
      ),
    });
    return;
  }

  if (finished === "transcribing") {
    const duration = job.durationSec ?? simulateDuration(job.source, job.isUpload);
    const units = findUnits(duration, 0);
    await ctx.db.patch(job._id, {
      stage: "finding",
      progress: 0,
      progressDetail: "Scoring viral moments...",
      unitLabel: units.unitLabel,
      unitsTotal: units.unitsTotal,
      unitsDone: 0,
      stageStartedAt: now,
      lastProgressAt: now,
      slow: false,
      transcriptWordCount: buildTranscript(job.source, duration).length,
      logs: appendLog(
        job.logs,
        `${ts()} · ▶ finding — ${units.unitsTotal} transcript chunks → Gemini (pipeline, no full-transcript wait)`,
      ),
    });
    return;
  }

  if (finished === "finding") {
    const duration = job.durationSec ?? simulateDuration(job.source, job.isUpload);
    const words = buildTranscript(job.source, duration);
    const candidates = heuristicClipDetection(words, {
      maxClips: 12,
      seedSalt: `${job.source}:${job._id}`,
    });
    const plans = forgeClips(job.source, job.isUpload, candidates, 10);
    const totalSec = Math.round(plans.reduce((acc, p) => acc + (p.end - p.start), 0));
    const captionsTxt = plans
      .map(
        (p, i) =>
          `=== Clip ${i + 1}: ${p.title} (score ${p.score}) ===\n${buildSrt(p.captionCues)}`,
      )
      .join("\n");

    const oldClips = await ctx.db
      .query("clips")
      .withIndex("by_job", (q) => q.eq("jobId", job._id))
      .collect();
    for (const c of oldClips) await ctx.db.delete(c._id);
    for (let i = 0; i < plans.length; i++) {
      const p = plans[i];
      await ctx.db.insert("clips", {
        jobId: job._id,
        userId: job.userId,
        order: i,
        start: p.start,
        end: p.end,
        score: p.score,
        title: p.title,
        hookText: p.hookText,
        transcript: p.transcript,
        hashtags: p.hashtags,
        captionCues: p.captionCues,
        cropBox: p.cropBox,
        silenceCuts: p.silenceCuts,
        aspect: "9:16",
        renderMs: 3200 + i * 250,
      });
    }

    const units = renderUnits(plans.length, 0);
    await ctx.db.patch(job._id, {
      stage: "rendering",
      progress: 0,
      progressDetail: "Rendering 9:16 crops with captions...",
      unitLabel: units.unitLabel,
      unitsTotal: units.unitsTotal,
      unitsDone: 0,
      stageStartedAt: now,
      lastProgressAt: now,
      slow: false,
      clipCount: plans.length,
      totalClipsSec: totalSec,
      clipSource: "heuristic",
      captionsTxt,
      logs: appendLog(
        job.logs,
        `${ts()} · ▶ rendering — ${plans.length} clips, 9:16 1080x1920 libx264/AAC (2-4 worker pool)`,
      ),
    });
    return;
  }

  if (finished === "rendering") {
    await ctx.db.patch(job._id, {
      stage: "complete",
      progress: 100,
      progressDetail: undefined,
      stageEtaSec: 0,
      totalEtaSec: 0,
      slow: false,
      completedAt: now,
      logs: appendLog(job.logs, `${ts()} · ✓ complete — MP4s in the output directory`),
    });
    return;
  }

  await failJob(ctx, job, "Unknown pipeline stage.", { detail: `stage=${finished}` });
}

// ---------------------------------------------------------------------------
// total ETA (current stage remaining + learned estimates for upcoming stages)
// ---------------------------------------------------------------------------

async function failJob(
  ctx: MutationCtx,
  job: Doc<"jobs">,
  error: string,
  opts: { detail?: string; stage?: string } = {},
) {
  const stage = opts.stage ?? job.stage;
  await ctx.db.patch(job._id, {
    stage: "failed",
    error,
    errorDetail: opts.detail,
    progressDetail: undefined,
    stageEtaSec: undefined,
    totalEtaSec: undefined,
    slow: false,
    logs: appendLog(
      job.logs,
      `${ts()} · ✗ ${stage} — ${error}${opts.detail ? ` :: ${opts.detail}` : ""}`,
    ),
  });
}

// ---------------------------------------------------------------------------
// AI clip detection support (the public action lives in detection.ts, which
// runs in the node runtime; mutations cannot call actions directly)
// ---------------------------------------------------------------------------

/** Load a job for ownership checks from actions. */
export const getJobForUser = internalQuery({
  args: { jobId: v.id("jobs"), userId: v.id("users") },
  handler: async (ctx, { jobId, userId }) => {
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) return null;
    return job;
  },
});

/** Resolve the signed-in user's id from actions (getAuthUserId is query-side). */
export const getCurrentUserId = internalQuery({
  args: {},
  handler: async (ctx) => {
    return await getAuthUserId(ctx);
  },
});

/**
 * Replace a completed job's clips with a new detection result (AI or
 * heuristic) and refresh captions.txt. Called by detection.ts after the
 * LLM scoring step.
 */
export const applyDetectedClips = internalMutation({
  args: {
    jobId: v.id("jobs"),
    userId: v.id("users"),
    clipSource: v.union(v.literal("ai"), v.literal("heuristic")),
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
  },
  handler: async (ctx, { jobId, userId, clipSource, candidates }) => {
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) {
      throw new Error("Job not found.");
    }
    if (candidates.length === 0) {
      throw new Error("Clip detection returned no usable clips.");
    }

    const duration = job.durationSec ?? simulateDuration(job.source, job.isUpload);
    const plans = forgeClips(job.source, job.isUpload, candidates, 10);
    const totalSec = Math.round(
      plans.reduce((acc, p) => acc + (p.end - p.start), 0),
    );
    const captionsTxt = plans
      .map(
        (p, i) =>
          `=== Clip ${i + 1}: ${p.title} (score ${p.score}) ===\n${buildSrt(p.captionCues)}`,
      )
      .join("\n");

    const oldClips = await ctx.db
      .query("clips")
      .withIndex("by_job", (q) => q.eq("jobId", jobId))
      .collect();
    for (const c of oldClips) await ctx.db.delete(c._id);
    for (let i = 0; i < plans.length; i++) {
      const p = plans[i];
      await ctx.db.insert("clips", {
        jobId,
        userId,
        order: i,
        start: p.start,
        end: p.end,
        score: p.score,
        title: p.title,
        hookText: p.hookText,
        transcript: p.transcript,
        hashtags: p.hashtags,
        captionCues: p.captionCues,
        cropBox: p.cropBox,
        silenceCuts: p.silenceCuts,
        aspect: "9:16",
        renderMs: 3200 + i * 250,
      });
    }

    await ctx.db.patch(jobId, {
      clipCount: plans.length,
      totalClipsSec: totalSec,
      clipSource,
      captionsTxt,
    });
    return { count: plans.length, clipSource };
  },
});

// ---------------------------------------------------------------------------
// admin / lifecycle
// ---------------------------------------------------------------------------

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
    await ctx.db.patch(jobId, {
      stage: "queued",
      progress: 0,
      progressDetail: "Queued — waiting for the worker...",
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
      durationSec: undefined,
      transcriptWordCount: undefined,
      logs: appendLog(job.logs, `${ts()} · ↻ retry queued`),
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
