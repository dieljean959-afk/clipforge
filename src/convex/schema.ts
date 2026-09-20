import { authTables } from "@convex-dev/auth/server";
import { defineSchema, defineTable } from "convex/server";
import { Infer, v } from "convex/values";

// default user roles. can add / remove based on the project as needed
export const ROLES = {
  ADMIN: "admin",
  USER: "user",
  MEMBER: "member",
} as const;

export const roleValidator = v.union(
  v.literal(ROLES.ADMIN),
  v.literal(ROLES.USER),
  v.literal(ROLES.MEMBER),
);
export type Role = Infer<typeof roleValidator>;

const schema = defineSchema(
  {
    // default auth tables using convex auth.
    ...authTables, // do not remove or modify

    // the users table is the default users table that is brought in by the authTables
    users: defineTable({
      name: v.optional(v.string()), // name of the user. do not remove
      image: v.optional(v.string()), // image of the user. do not remove
      email: v.optional(v.string()), // email of the user. do not remove
      emailVerificationTime: v.optional(v.number()), // email verification time. do not remove
      isAnonymous: v.optional(v.boolean()), // is the user anonymous. do not remove

      role: v.optional(roleValidator), // role of the user. do not remove
    }).index("email", ["email"]), // index for the email. do not remove or modify

    // ------------------------------------------------------------------
    // ClipForge
    // ------------------------------------------------------------------

    // one row per user-submitted video (URL or upload)
    jobs: defineTable({
      userId: v.id("users"),
      // "url" = yt-dlp/cobalt grab, "upload" = manual attach
      sourceType: v.string(), // "url" | "upload"
      source: v.string(), // video URL or original file name
      isUpload: v.boolean(),
      // pipeline stages: queued | warming_model | grabbing | transcribing | finding | rendering | complete | failed
      stage: v.string(),
      progress: v.number(), // 0-100 within current stage
      progressDetail: v.optional(v.string()),
      // real progress units, e.g. "MB" during grab, "min of audio" while transcribing
      unitLabel: v.optional(v.string()),
      unitsDone: v.optional(v.number()),
      unitsTotal: v.optional(v.number()),
      // ETA seconds for the current stage (null until a rate is measured)
      stageEtaSec: v.optional(v.number()),
      // total remaining ETA across all unfinished stages (seconds)
      totalEtaSec: v.optional(v.number()),
      // epoch ms of the last worker progress write — the stall watchdog reads this
      lastProgressAt: v.optional(v.number()),
      // "slow" heartbeat flag, set by the worker when a stage overruns its median
      slow: v.optional(v.boolean()),
      // stage start epoch ms (for per-stage duration logging)
      stageStartedAt: v.optional(v.number()),
      // per-stage duration log (ms), written on every stage completion
      stageTimings: v.array(
        v.object({
          stage: v.string(),
          durationMs: v.number(),
          at: v.number(),
        }),
      ),
      // append-only worker log, newest last ("View logs" in the UI)
      logs: v.array(v.string()),      error: v.optional(v.string()),
      errorDetail: v.optional(v.string()),
      durationSec: v.optional(v.number()),
      transcriptWordCount: v.optional(v.number()),
      clipCount: v.optional(v.number()),
      totalClipsSec: v.optional(v.number()),
      clipSource: v.optional(v.string()), // "ai" | "heuristic"
      downloadSource: v.optional(v.string()), // "yt-dlp" | "cobalt" | "upload"
      captionsTxt: v.optional(v.string()),
      createdAt: v.number(),
      completedAt: v.optional(v.number()),
    })
      .index("by_user", ["userId"])
      .index("by_user_stage", ["userId", "stage"])
      .index("by_stage", ["stage"]),

    // one row per rendered 9:16 clip
    clips: defineTable({
      jobId: v.id("jobs"),
      userId: v.id("users"),
      order: v.number(),
      start: v.number(),
      end: v.number(),
      score: v.number(),
      title: v.string(),
      hookText: v.string(),
      transcript: v.string(),
      hashtags: v.array(v.string()),
      captionCues: v.array(
        v.object({
          start: v.number(),
          end: v.number(),
          text: v.string(),
        }),
      ),
      cropBox: v.object({ x: v.number(), y: v.number() }),
      silenceCuts: v.array(v.object({ start: v.number(), end: v.number() })),
      aspect: v.string(), // "9:16"
      renderMs: v.number(),
    })
      .index("by_job", ["jobId"])
      .index("by_user", ["userId"]),

    // Worker liveness singleton — the "Render worker offline" banner source.
    // An API that queues jobs nothing consumes is the #1 cause of eternal
    // spinners, so the worker must heartbeat here to accept work.
    workerStatus: defineTable({
      singleton: v.literal("worker"),
      online: v.boolean(),
      lastHeartbeat: v.number(),
      modelLoaded: v.boolean(),
      warmupDetail: v.optional(v.string()),
      warmupStartedAt: v.optional(v.number()),
      // average per-second processing rate per stage, learned across jobs
      rates: v.optional(v.record(v.string(), v.number())),
    }).index("singleton", ["singleton"]),

    // Learned stage speeds (3+ runs) used for pre-stage ETA estimates.
    engineStats: defineTable({
      singleton: v.literal("engine"),
      stageSamples: v.record(v.string(), v.array(v.number())), // ms durations
    }).index("singleton", ["singleton"]),
  },
  {
    schemaValidation: false,
  },
);

export default schema;
