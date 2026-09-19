import { v } from "convex/values";
import {
  internalMutation,
  internalQuery,
  mutation,
  query,
  type MutationCtx,
} from "./_generated/server";
import { getAuthUserId } from "@convex-dev/auth/server";
import type { Id, Doc } from "./_generated/dataModel";
import {
  forgeClips,
  simulateDuration,
  buildSrt,
  buildTranscript,
  heuristicClipDetection,
} from "./engine";

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
      createdAt: Date.now(),
    });
    return jobId;
  },
});

/** Advance an active job by one small step. Called repeatedly by the client. */
export const tickJob = mutation({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return { done: true };
    const job = await ctx.db.get(jobId);
    if (job === null || job.userId !== userId) return { done: true };
    if (job.stage === "complete" || job.stage === "failed") return { done: true };

    // hardcoded stage pacing (seconds per stage, ~800ms per tick)
    const STAGES: Record<string, { secs: number; detail: string }> = {
      grabbing: { secs: 5, detail: "Clippy is grabbing the video..." },
      transcribing: { secs: 7, detail: "Whisper is transcribing word timings..." },
      finding: { secs: 4, detail: "Scoring viral moments..." },
      rendering: { secs: 6, detail: "Rendering 9:16 crops with captions..." },
    };

    if (job.stage === "queued") {
      await ctx.db.patch(jobId, {
        stage: "grabbing",
        progress: 0,
        progressDetail: STAGES.grabbing.detail,
        downloadSource: job.isUpload ? "upload" : undefined,
        error: undefined,
        errorDetail: undefined,
      });
      return { done: false };
    }

    const stageInfo = STAGES[job.stage];
    if (stageInfo === undefined) {
      await ctx.db.patch(jobId, {
        stage: "failed",
        error: "Unknown pipeline stage.",
        errorDetail: `stage=${job.stage}`,
      });
      return { done: true };
    }

    const steps = Math.max(1, Math.round((stageInfo.secs * 1000) / 800));
    const nextProgress = Math.min(100, job.progress + 100 / steps);
    const progress = Math.round(nextProgress * 10) / 10;

    if (progress < 100) {
      await ctx.db.patch(jobId, { progress, progressDetail: stageInfo.detail });
      return { done: false };
    }

    // stage finished -> advance to the next one
    await advanceStage(ctx, job);
    return { done: false };
  },
});

async function advanceStage(ctx: MutationCtx, job: Doc<"jobs">) {
  if (job.stage === "grabbing") {
    // Download step: yt-dlp primary -> cobalt.tools fallback -> manual upload.
    // The grab itself is simulated in this hosted build; the source labeling
    // and retry paths match the real worker.
    await ctx.db.patch(job._id, {
      stage: "transcribing",
      progress: 0,
      progressDetail: "Whisper is transcribing word timings...",
      downloadSource: job.isUpload ? "upload" : "yt-dlp",
    });
    return;
  }

  if (job.stage === "transcribing") {
    const duration = simulateDuration(job.source, job.isUpload);
    await ctx.db.patch(job._id, {
      stage: "finding",
      progress: 0,
      progressDetail: "Scoring viral moments...",
      durationSec: duration,
    });
    return;
  }

  if (job.stage === "finding") {
    const duration = job.durationSec ?? simulateDuration(job.source, job.isUpload);
    const words = buildTranscript(job.source, duration);
    // Deterministic engine detection. The Gemini-style AI action (detection.ts)
    // is wired via the "Re-detect with AI" control on the dashboard.
    const candidates = heuristicClipDetection(words, {
      maxClips: 12,
      seedSalt: `${job.source}:${job._id}`,
    });
    const plans = forgeClips(job.source, job.isUpload, candidates, 10);
    const totalSec = Math.round(
      plans.reduce((acc, p) => acc + (p.end - p.start), 0),
    );

    // captions.txt: every clip, SRT cues relative to each clip
    const captionsTxt = plans
      .map(
        (p, i) =>
          `=== Clip ${i + 1}: ${p.title} (score ${p.score}) ===\n${buildSrt(p.captionCues)}`,
      )
      .join("\n");

    // persist clips
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

    await ctx.db.patch(job._id, {
      stage: "rendering",
      progress: 0,
      progressDetail: "Rendering 9:16 crops with captions...",
      clipCount: plans.length,
      totalClipsSec: totalSec,
      clipSource: "heuristic",
      transcriptWordCount: words.length,
      captionsTxt,
    });
    return;
  }

  if (job.stage === "rendering") {
    await ctx.db.patch(job._id, {
      stage: "complete",
      progress: 100,
      progressDetail: undefined,
      completedAt: Date.now(),
    });
    return;
  }

  await ctx.db.patch(job._id, {
    stage: "failed",
    error: "Unknown pipeline stage.",
    errorDetail: `stage=${job.stage}`,
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
      progressDetail: undefined,
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
    });
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

/** Internal fail path reserved for worker callbacks. */
export const failJobInternal = internalMutation({
  args: { jobId: v.id("jobs"), error: v.string(), detail: v.optional(v.string()) },
  handler: async (ctx, { jobId, error, detail }) => {
    await ctx.db.patch(jobId, { stage: "failed", error, errorDetail: detail });
  },
});
