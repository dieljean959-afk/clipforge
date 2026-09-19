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
      // pipeline stages: queued | grabbing | transcribing | finding | rendering | complete | failed
      stage: v.string(),
      progress: v.number(), // 0-100 within current stage
      progressDetail: v.optional(v.string()),
      error: v.optional(v.string()),
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
      .index("by_user_stage", ["userId", "stage"]),

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
  },
  {
    schemaValidation: false,
  },
);

export default schema;
