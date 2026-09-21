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
      // "url" = cobalt/direct HTTP fetch, "upload" = manual attach
      sourceType: v.string(), // "url" | "upload"
      source: v.string(), // video URL or original file name
      isUpload: v.boolean(),
      // REAL media (no simulation): the fetched source MP4 lives in Convex
      // file storage. A job may not advance past "grabbing" without one.
      sourceStorageId: v.optional(v.id("_storage")),
      sourceBytes: v.optional(v.number()),
      sourceDurationSec: v.optional(v.number()),
      sourceWidth: v.optional(v.number()),
      sourceHeight: v.optional(v.number()),
      // how the bytes were actually obtained: "yt-dlp" | "direct" | "upload"
      fetchSource: v.optional(v.string()),
      // VM worker (Oracle Cloud) claim: the VM fetches URL jobs while awake.
      vmClaimedAt: v.optional(v.number()),
      vmAttempts: v.optional(v.number()),
      fetchError: v.optional(v.string()),
      fetchAttempts: v.array(
        v.object({
          method: v.string(),
          ok: v.boolean(),
          error: v.optional(v.string()),
          bytes: v.optional(v.number()),
          at: v.number(),
        }),
      ),
      // Whisper word-level transcript (real, uploaded by the browser worker)
      transcriptStorageId: v.optional(v.id("_storage")),
      transcriptWordCount: v.optional(v.number()),
      // browser render worker state: "pending" | "running" | "done" | "failed"
      renderState: v.optional(v.string()),
      renderError: v.optional(v.string()),
      renderProgress: v.optional(v.number()),
      detectedSource: v.optional(v.string()), // "gemini" | "none"
      // pipeline stages: queued | warming_model | grabbing | transcribing | finding | rendering | complete | failed
      // queued | grabbing | transcribing | finding | rendering | complete | failed
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
      clipCount: v.optional(v.number()),
      totalClipsSec: v.optional(v.number()),
      clipSource: v.optional(v.string()), // "gemini"
      downloadSource: v.optional(v.string()), // "cobalt" | "direct" | "upload"
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
      // reframe geometry is produced by the real render pass, not fabricated
      cropBox: v.optional(v.object({ x: v.number(), y: v.number() })),
      silenceCuts: v.optional(v.array(v.object({ start: v.number(), end: v.number() }))),
      aspect: v.string(), // "9:16"
      // REAL rendered files (ffmpeg.wasm in the user's browser). A clip card
      // with no mediaStorageId shows an honest "not rendered" state — the UI
      // must never invent frames or labels for a clip that has no video.
      mediaStorageId: v.optional(v.id("_storage")),
      thumbStorageId: v.optional(v.id("_storage")),
      mediaBytes: v.optional(v.number()),
      mediaWidth: v.optional(v.number()),
      mediaHeight: v.optional(v.number()),
      mediaDurationSec: v.optional(v.number()),
      mediaError: v.optional(v.string()),
      // per-clip: "pending" | "running" | "done" | "failed"
      renderState: v.optional(v.string()),
      renderMs: v.optional(v.number()),
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

      // ----------------------------------------------------------------
      // Serverless "in-process worker" plumbing (this platform cannot
      // keep a persistent Celery/RQ process alive, so the worker is a
      // self-scheduling Convex mutation chain — same process semantics:
      // API + worker are one deployment, "worker offline" becomes a
      // cold-start state a client ping repairs, never a broken deploy).
      // ----------------------------------------------------------------
      // secret written by every tick; a client PING that cannot verify it
      // (serverClockMs - lastHeartbeat > WORKER_STALE_MS) sees the worker
      // as dead. Playwright clocks differ from server time, so liveness is
      // judged with SERVER-returned time only.
      lastHeartbeatServerMs: v.optional(v.number()),
      // server-clock ms the client last PINGed — proof a client is listening
      lastClientPingServerMs: v.optional(v.number()),
      // monotonically increasing worker ticks (log "consuming queue" at 1)
      tickCount: v.optional(v.number()),
      // queue/broker identity — the handshake the boot check verifies
      queueName: v.optional(v.string()),
      brokerUrl: v.optional(v.string()),
      // set when the self-scheduling chain detects it is not running and
      // reschedules itself — used by the UI to show "repairing (cold start)"
      repairing: v.optional(v.boolean()),
      // exact server log line the verify step greps for
      lastBootLine: v.optional(v.string()),
      // true once the chain is re-armed after a deployment/cold start
      chainArmed: v.optional(v.boolean()),
      // claim gate for the self-scheduling chain: a tick only schedules the
      // next one when this is unset or past — mutations serialize, so this
      // CAS keeps exactly ONE chain alive even with cron + client kicks
      nextTickAt: v.optional(v.number()),
    }).index("singleton", ["singleton"]),

    // Learned stage speeds (3+ runs) used for pre-stage ETA estimates.
    engineStats: defineTable({
      singleton: v.literal("engine"),
      stageSamples: v.record(v.string(), v.array(v.number())), // ms durations
    }).index("singleton", ["singleton"]),

    // Oracle Cloud worker-VM control plane (see vm.ts): cached REAL OCI
    // lifecycle state + checked-at, worker heartbeat freshness, and a
    // deferred STOP intent that waits until the VM drains its jobs. The
    // dashboard ON/OFF switch renders from THIS truth — never last-click.
    workerVm: defineTable({
      singleton: v.literal("vm"),
      // "RUNNING" | "STOPPED" | "STARTING" | "STOPPING" | ... (OCI verbatim)
      ociState: v.optional(v.union(v.string(), v.null())),
      ociError: v.optional(v.union(v.string(), v.null())),
      ociCheckedAt: v.number(),
      // worker VM heartbeat (fresh within 75s = worker reachable)
      workerOk: v.optional(v.boolean()),
      workerLastSeenAt: v.optional(v.number()),
      workerLastSeenServerMs: v.optional(v.number()),
      modelLoaded: v.optional(v.boolean()),
      // deferred power action (STOP waits for the worker to go idle)
      pendingAction: v.optional(v.union(v.literal("START"), v.literal("STOP"))),
    }).index("singleton", ["singleton"]),

    // OCI setup wizard storage (server-side only — the browser never reads
    // these back after save; vm.ts resolves env first, then this table).
    // PRIVATE_KEY holds the raw PEM contents, never returned to the client.
    vmConfig: defineTable({
      singleton: v.literal("vm-config"),
      tenancyOcid: v.string(),
      userOcid: v.string(),
      fingerprint: v.string(),
      privateKeyPem: v.string(),
      instanceOcid: v.string(),
      region: v.string(),
      savedAt: v.number(),
      skipped: v.optional(v.boolean()), // "Skip for now" hides the wizard
    }).index("singleton", ["singleton"]),
  },
  {
    schemaValidation: false,
  },
);

export default schema;
