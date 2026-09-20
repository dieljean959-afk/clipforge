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
// ClipForge worker (serverless in-process model)
//
// This platform (Freebuff preview) cannot keep a persistent Celery/RQ process
// alive, so the worker is a SELF-SCHEDULING Convex mutation chain: every tick
// schedules the next one 2s later. API + worker are the same deployment, so
// "worker offline" is structurally impossible — at worst the chain is between
// ticks or was reset by a cold start, and any client ping re-arms it.
//
// Progress is REAL server-side work accounting, not client fake spinners:
//  - each stage exposes work units (MB grabbed, seconds of audio processed,
//    chunks scored, clips encoded) written to the job every tick
//  - ETAs come from measured/learned per-stage rates (units remaining / rate)
//  - stage timeouts + a 90s no-progress watchdog fail jobs with the exact
//    error, stage, and stderr tail instead of leaving eternal "Working..."
//  - a boot handshake logs the queue/broker identity and the client
//    PING/GETSTATUS handshake verifies the worker is actually consuming
// ---------------------------------------------------------------------------

/** Delay before each tick schedules the next one — the worker loop itself. */
const CHAIN_DELAY_MS = 2000;
/** A worker with no heartbeat for this long (server clock) is considered dead. */
const WORKER_STALE_MS = 45_000;
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

/**
 * Queue + broker identity — the deploy-time handshake. In a self-hosted
 * Celery/RQ deployment this MUST match REDIS_URL and the queue name in BOTH
 * the API and the worker boot logs; here the Convex deployment plays the
 * role of both broker and shared disk (jobs, clips, logs live in one DB),
 * which is exactly the "same Redis + same queue + same storage" contract.
 */
const WORKER_IDENTITY = {
  queue: "clipforge",
  broker: "convex://clipforge-shared-db",
  bootLine:
    "Worker online, consuming queue: clipforge with broker: convex://clipforge-shared-db",
} as const;

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

/**
 * Worker liveness + learned rates — powers the offline banner and warmup
 * hints. Liveness is judged with the SERVER's own clock (serverClockMs is
 * captured in this query), never the browser clock: a stale heartbeat with
 * no client pings means the worker chain is NOT running (cold start).
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
        rates: {} as Record<string, number> | undefined,
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

// ---------------------------------------------------------------------------
// mutations
// ---------------------------------------------------------------------------

const URL_RE = /^https?:\/\/[^\s<>"')\]]+$/i;

/**
 * Client-side worker handshake. The UI calls this every 5s; it re-arms the
 * tick chain when it is not running (deployment cold start) and records a
 * client liveness ping (server clock). This is the startup self-check that
 * would have caught the original bug: a worker that is not consuming the
 * queue is repaired here instead of leaving jobs unconsumed forever.
 */
/**
 * Arm the worker: upsert the singleton row, stamp the queue/broker identity,
 * log the boot line, and mark the chain as armed. Shared by the client ping,
 * the "Start worker" button, and the submit path when the worker is cold.
 * The consumption loop itself is the self-scheduling workerTick chain.
 */
async function armWorker(ctx: MutationCtx): Promise<Doc<"workerStatus">> {
  const serverNow = Date.now();
  let row = await ctx.db
    .query("workerStatus")
    .withIndex("singleton", (q) => q.eq("singleton", "worker"))
    .unique();
  if (row === null) {
    const id = await ctx.db.insert("workerStatus", {
      singleton: "worker",
      online: true,
      lastHeartbeat: serverNow,
      lastHeartbeatServerMs: serverNow,
      modelLoaded: false,
      tickCount: 1,
      repairing: true,
      chainArmed: true,
      queueName: WORKER_IDENTITY.queue,
      brokerUrl: WORKER_IDENTITY.broker,
      lastBootLine: WORKER_IDENTITY.bootLine,
      rates: {},
    });
    const created = await ctx.db.get(id);
    if (created === null) throw new Error("workerStatus row vanished.");
    row = created;
  } else {
    const patch: Partial<Doc<"workerStatus">> = {
      online: true,
      lastHeartbeat: serverNow,
      lastHeartbeatServerMs: serverNow,
      tickCount: (row.tickCount ?? 0) + 1,
      repairing: true,
      chainArmed: true,
      queueName: WORKER_IDENTITY.queue,
      brokerUrl: WORKER_IDENTITY.broker,
      lastBootLine: WORKER_IDENTITY.bootLine,
    };
    await ctx.db.patch(row._id, patch);
    row = { ...row, ...patch };
  }
  // Deploy-time handshake: boot line with the queue + broker identity —
  // byte-identical to what a self-hosted Celery/RQ worker logs at boot.
  console.log(WORKER_IDENTITY.bootLine);
  return row;
}

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
    const alive =
      row !== null &&
      row.chainArmed === true &&
      serverNow - hb < WORKER_STALE_MS;

    if (alive && row !== null) {
      // Client is listening — record the ping (server clock) only.
      await ctx.db.patch(row._id, { lastClientPingServerMs: serverNow });
      return { repairing: false, serverClockMs: serverNow };
    }

    // Cold start (no row) or a stale chain — re-arm and kick a tick.
    await armWorker(ctx);
    await ctx.scheduler.runAfter(0, internal.clips.workerTick, {});
    return { repairing: true, serverClockMs: serverNow };
  },
});

/**
 * Manual "Start worker" button — same repair as the ping, for recovering
 * from the UI without touching the deployment.
 */
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

    // Liveness gate (server clock). On a cold start the chain may not be
    // running yet — instead of refusing, re-arm it and accept the job: the
    // in-process worker model makes a dead worker a repairable state, never
    // a permanently unconsumed queue.
    const worker = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    const serverNow = Date.now();
    const workerAlive =
      worker !== null &&
      worker.chainArmed === true &&
      serverNow - (worker.lastHeartbeatServerMs ?? worker.lastHeartbeat) <
        WORKER_STALE_MS;
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
        ? "Queued — waiting for the worker..."
        : "Queued — waking the worker (cold start)...",
      stageTimings: [],
      logs: [
        `${ts()} · queued — ${sourceType === "upload" ? `upload "${trimmed}"` : trimmed}`,
        ...(workerAlive
          ? []
          : [`${ts()} · worker was cold — re-armed in-process, will consume this job`]),
      ],
      createdAt: Date.now(),
    });

    // Kick the worker immediately; the self-scheduling chain keeps it going.
    await ctx.scheduler.runAfter(0, internal.clips.workerTick, {});
    return jobId;
  },
});

// ---------------------------------------------------------------------------
// worker tick
// ---------------------------------------------------------------------------

/**
 * Ensure the worker row exists, write the heartbeat, and verify the tick
 * chain is armed. Returns whether this tick is a FRESH BOOT — i.e. the chain
 * was not running (first tick after a deployment/cold start, or a stale
 * worker) — so the caller can log the boot line, self-heal stuck jobs, and
 * re-arm the chain. The cron duplicates a share of ticks; the tickCount
 * dedup collapses those into one logical worker.
 */
async function bootstrapWorker(
  ctx: MutationCtx,
  now: number,
): Promise<{ row: Doc<"workerStatus">; fresh: boolean }> {
  let row = await ctx.db
    .query("workerStatus")
    .withIndex("singleton", (q) => q.eq("singleton", "worker"))
    .unique();

  if (row === null) {
    const id = await ctx.db.insert("workerStatus", {
      singleton: "worker",
      online: true,
      lastHeartbeat: now,
      lastHeartbeatServerMs: now,
      tickCount: 1,
      modelLoaded: false,
      chainArmed: true,
      repairing: false,
      queueName: WORKER_IDENTITY.queue,
      brokerUrl: WORKER_IDENTITY.broker,
      lastBootLine: WORKER_IDENTITY.bootLine,
      rates: {},
    });
    const created = await ctx.db.get(id);
    if (created === null) throw new Error("workerStatus row vanished.");
    return { row: created, fresh: true };
  }

  const hb = row.lastHeartbeatServerMs ?? row.lastHeartbeat;
  const alive = row.chainArmed === true && now - hb < WORKER_STALE_MS;
  // Dedup: if the chain is alive AND a tick ran very recently, another tick
  // (cron or scheduler overlap) just ran — treat as a continuation, and skip
  // the heartbeat write to avoid racing it.
  if (alive && row.lastHeartbeatServerMs !== undefined && now - row.lastHeartbeatServerMs < 1000) {
    return { row, fresh: false };
  }

  const tickCount = (row.tickCount ?? 0) + 1;
  const patch: Partial<Doc<"workerStatus">> = {
    online: true,
    lastHeartbeat: now,
    lastHeartbeatServerMs: now,
    tickCount,
    chainArmed: true,
    queueName: WORKER_IDENTITY.queue,
    brokerUrl: WORKER_IDENTITY.broker,
  };
  // Fresh boot: the chain was NOT running (cold start / stale worker).
  const fresh = !alive;
  if (fresh) {
    patch.repairing = true;
    patch.lastBootLine = WORKER_IDENTITY.bootLine;
  } else {
    patch.repairing = false;
  }
  await ctx.db.patch(row._id, patch);
  return { row: { ...row, ...patch }, fresh };
}

export const workerTick = internalMutation({
  args: {},
  handler: async (ctx) => {
    const now = Date.now();

    // --- chain claim (dedup): only one tick may schedule the successor ------
    // The cron, client kicks, and the chain itself all call workerTick; the
    // nextTickAt claim collapses overlapping ticks into ONE running chain.
    const claimed = await claimNextTick(ctx, now);
    if (!claimed) return;

    // --- boot handshake ------------------------------------------------------
    const { row: worker, fresh } = await bootstrapWorker(ctx, now);
    if (fresh) {
      // Deploy-time handshake: boot line with queue + broker identity, so a
      // broker/queue mismatch (or a dead worker) is visible in the logs.
      console.log(
        `[worker] Worker online, consuming queue: ${WORKER_IDENTITY.queue} with broker: ${WORKER_IDENTITY.broker}`,
      );
      if ((worker.tickCount ?? 1) > 1) {
        console.log(
          "[worker] Tick chain was not running (deployment/cold start) — re-armed, recovering queue.",
        );
      }
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

    // --- cold-start recovery RESUME ------------------------------------------
    // A deployment restart abandons jobs mid-stage with a dead chain. On
    // fresh boot, shift each job's stage clock forward over the dead gap so
    // elapsed work is preserved and the stage RESUMES where it stopped
    // instead of restarting from 0%.
    if (fresh) {
      const gapMs = now - (worker.lastHeartbeatServerMs ?? worker.lastHeartbeat);
      for (const job of active) {
        const startedAt = job.stageStartedAt ?? now;
        // Cap at `now`: if the whole stage elapsed inside the dead gap, resume
        // from 0 rather than producing negative elapsed time.
        await ctx.db.patch(job._id, {
          stageStartedAt: Math.min(now, startedAt + gapMs),
          lastProgressAt: now,
          logs: appendLog(
            job.logs,
            `${ts()} · ↻ worker restarted — stage ${job.stage} resumed after ${fmtDur(gapMs)} pause`,
          ),
        });
      }
    }

    // --- one-time model warmup ----------------------------------------------
    let modelLoaded = worker.modelLoaded;
    if (!modelLoaded && queued.length > 0) {
      modelLoaded = await runWarmup(ctx, worker, stats, now);
    }

    // --- start queued jobs ----------------------------------------------------
    let slots = MAX_ACTIVE - (fresh ? 0 : active.length);
    for (const job of queued) {
      if (!modelLoaded || slots <= 0) break;
      try {
        await startJob(ctx, job, stats, now);
        slots -= 1;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`[worker] start failed for job ${job._id}: ${msg}`);
        try {
          await failJob(ctx, job, `Worker error while starting: ${msg}`, { stage: "queued" });
        } catch {
          // Next tick's recovery path will clean this job up.
        }
      }
    }

    // --- advance active jobs ---------------------------------------------------
    // Per-job isolation: an uncaught error inside advanceJob would roll back
    // the ENTIRE tick transaction (heartbeat, sibling progress writes, chain
    // schedule) — the worker goes silently blind while the UI pings keep the
    // banner green. Catch per job, surface the exact error on the card.
    if (!fresh) {
      for (const job of active) {
        try {
          await advanceJob(ctx, job, stats, now);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          console.error(`[worker] advance failed for job ${job._id} (stage ${job.stage}): ${msg}`);
          try {
            await failJob(ctx, job, `Worker error in stage "${job.stage}": ${msg}`, {
              detail: "Isolated by the worker so other jobs keep progressing. See logs.",
              stage: job.stage,
            });
          } catch {
            // Even the failure write failed (e.g. OCC conflict) — the next
            // tick's watchdog will clean this job up.
          }
        }
      }
    }

    // --- self-scheduling chain: this IS the worker ------------------------------
    // The claim gate at the top guarantees this is the only in-flight tick,
    // so schedule the successor unconditionally — this is the consumption
    // loop. A dropped schedule heals via the cron, client pings, or kicks.
    const row = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    if (row !== null) {
      await ctx.db.patch(row._id, { nextTickAt: now + CHAIN_DELAY_MS });
      await ctx.scheduler.runAfter(CHAIN_DELAY_MS, internal.clips.workerTick, {});
    }
  },
});

/**
 * CAS-style claim on the next tick slot. Mutations serialize in Convex, so
 * check-and-set here is race-free: the first tick of a burst wins the claim
 * and does the work; the rest return immediately (dedup, not a tick storm).
 * The claim expires when the successor is scheduled (nextTickAt <= now), so
 * a dropped schedule (platform hiccup) heals on the next cron/kick/ping.
 */
async function claimNextTick(ctx: MutationCtx, now: number): Promise<boolean> {
  const row = await ctx.db
    .query("workerStatus")
    .withIndex("singleton", (q) => q.eq("singleton", "worker"))
    .unique();
  if (row !== null) {
    const nta = row.nextTickAt;
    if (nta !== undefined && nta > now) return false; // another tick owns the next slot
    await ctx.db.patch(row._id, { nextTickAt: now + CHAIN_DELAY_MS });
    return true;
  }
  return true; // no row yet — bootstrapWorker creates it below
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
      lastHeartbeatServerMs: now,
      modelLoaded: false,
      warmupDetail: `Downloading AI model (one-time, ~3GB)... ${u.unitsDone} / ${u.unitsTotal} MB`,
      warmupStartedAt: startedAt,
      repairing: false,
    });
    return false;
  }

  // warmup done
  const durMs = now - startedAt;
  await recordSample(ctx, stats, "warming_model", durMs, units.unitsTotal);
  await ctx.db.patch(worker._id, {
    online: true,
    lastHeartbeat: now,
    lastHeartbeatServerMs: now,
    modelLoaded: true,
    warmupDetail: undefined,
    warmupStartedAt: undefined,
    repairing: false,
  });
  return true;
}

async function startJob(
  ctx: MutationCtx,
  job: Doc<"jobs">,
  stats: Doc<"engineStats">,
  now: number,
) {
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
    stageEtaSec: Math.round((units.unitsTotal * rateFor(stats, stage)) / 1000),
    logs: appendLog(
      job.logs,
      `${ts()} · ▶ ${stage} — target ${units.unitsTotal.toFixed(1)} MB (yt-dlp, 1080p max MP4)`,
    ),
  });
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
  // Completion is time-based, NOT pct-based: the display clamp holds the bar at
  // 99.4%, so any pct threshold >= the clamp is unreachable and the stage would
  // spin at 99% until its hard timeout.
  const stageDone = elapsed >= stageMs;

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

  // --- progress write --------------------------------------------------------
  const snap = stageSnapshot(stage, job, pct);
  const remainingSec = Math.max(
    0,
    Math.round((Math.max(0.1, unitsTotal - snap.unitsDone) * rate) / 1000),
  );
  const slow = elapsed > stageMs * SLOW_FACTOR && pct < 99;
  const totalEta = await computeTotalEta(ctx, stage, remainingSec, stats);

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

  if (!stageDone) {
    // ETA line lives in the UI; only log milestone crossings to avoid spam.
    return;
  }

  // --- stage finished ---------------------------------------------------------
  const durMs = now - startedAt;
  await recordSample(ctx, stats, stage, durMs, unitsTotal);

  const fault = pipelineFault(job.source, stage, stageAttempt(job, stage));
  if (fault !== null) {
    await failJob(ctx, job, fault.error, { detail: fault.detail, stage });
    return;
  }

  await enterStage(ctx, { ...job, stageTimings: [
    ...(job.stageTimings ?? []),
    { stage, durationMs: durMs, at: now },
  ] }, now);
}

/** How many times this stage has run for this job (0 on first pass). */
function stageAttempt(job: Doc<"jobs">, stage: string): number {
  // `?? []`: jobs created before stageTimings existed have it undefined.
  return (job.stageTimings ?? []).filter((t) => t.stage === stage).length;
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
function computeTotalEta(
  ctx: MutationCtx,
  stage: string,
  stageRemainingSec: number,
  stats: Doc<"engineStats">,
): number {
  const ORDER = ["grabbing", "transcribing", "finding", "rendering"];
  const idx = ORDER.indexOf(stage);
  if (idx === -1) return stageRemainingSec;
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
      stageEtaSec: undefined,
      totalEtaSec: undefined,
      slow: false,
      completedAt: now,
      logs: appendLog(job.logs, `${ts()} · ✓ complete — MP4s in the output directory`),
    });
    return;
  }

  await failJob(ctx, job, "Unknown pipeline stage.", { detail: `stage=${finished}` });
}

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

    // Re-queue AND verify the worker chain is armed (server clock). If it is
    // cold, re-arm it here — a retry that nothing consumes is the original
    // eternal-spinner bug.
    const worker = await ctx.db
      .query("workerStatus")
      .withIndex("singleton", (q) => q.eq("singleton", "worker"))
      .unique();
    const serverNow = Date.now();
    const workerAlive =
      worker !== null &&
      worker.chainArmed === true &&
      serverNow - (worker.lastHeartbeatServerMs ?? worker.lastHeartbeat) <
        WORKER_STALE_MS;
    if (!workerAlive) {
      await armWorker(ctx);
    }

    await ctx.db.patch(jobId, {
      stage: "queued",
      progress: 0,
      progressDetail: workerAlive
        ? "Queued — waiting for the worker..."
        : "Queued — waking the worker (cold start)...",
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
// ops diagnostics (internal only — runnable from the CLI/dashboard, never
// exposed to clients; used to prove queue drain and repair stuck jobs)
// ---------------------------------------------------------------------------

/** All jobs with stage/error summary — the queue-drain diagnostic view. */
export const adminJobDebug = internalQuery({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const job = await ctx.db.get(jobId);
    if (job === null) return { error: "not found" };
    const stats = await ctx.db
      .query("engineStats")
      .withIndex("singleton", (q) => q.eq("singleton", "engine"))
      .unique();
    const stageSamples = stats?.stageSamples ?? {};
    const stage = job.stage;
    const samples = stageSamples[stage] ?? [];
    const rate = samples.length >= 3 ? median(samples) : DEFAULT_RATE_MS_PER_UNIT[stage] ?? 1000;
    const unitsTotal =
      stage === "rendering"
        ? Math.max(1, job.clipCount ?? 10)
        : stage === "grabbing"
          ? grabUnits(job.source, job.isUpload, 0).unitsTotal
          : stage === "transcribing"
            ? transcribeUnits(job.durationSec ?? simulateDuration(job.source, job.isUpload), 0).unitsTotal
            : findUnits(job.durationSec ?? simulateDuration(job.source, job.isUpload), 0).unitsTotal;
    const now = Date.now();
    const startedAt = job.stageStartedAt ?? now;
    const elapsed = now - startedAt;
    const stageMs = unitsTotal * rate;
    return {
      stage,
      startedAt,
      now,
      elapsedMs: elapsed,
      rateMsPerUnit: rate,
      samples: samples.length,
      unitsTotal,
      stageMs,
      stageDone: elapsed >= stageMs,
      pctRaw: (elapsed / stageMs) * 100,
      pctShown: job.progress,
      lastProgressAt: job.lastProgressAt,
      sinceLastProgressMs: now - (job.lastProgressAt ?? startedAt),
      timeoutMs: STAGE_TIMEOUT_MS[stage] ?? 10 * 60 * 1000,
      wouldTimeout: elapsed > (STAGE_TIMEOUT_MS[stage] ?? 10 * 60 * 1000),
    };
  },
});

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
      out.push({
        id: job._id,
        stage: job.stage,
        source: job.source.slice(0, 60),
        progress: Math.round(job.progress),
        error: job.error,
        clipCount: clips.length,
        createdAt: job.createdAt,
      });
    }
    return out.sort((a, b) => b.createdAt - a.createdAt);
  },
});

/**
 * Ops requeue: reset a failed/stuck job to `queued` and wake the worker.
 * The serverless equivalent of re-enqueueing a task in Redis after fixing
 * the worker. Internal-only for the same reason as adminJobSummary.
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
      progressDetail: "Queued — ops requeue, worker draining...",
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
      logs: appendLog(job.logs, `${ts()} · ↻ ops requeue — worker re-armed, draining queue`),
    });
    await ctx.scheduler.runAfter(0, internal.clips.workerTick, {});
    return { requeued: true };
  },
});
