import { httpRouter } from "convex/server";
import { httpAction } from "./_generated/server";
import { v } from "convex/values";
import { auth } from "./auth";
import { internal, api } from "./_generated/api";

const http = httpRouter();

auth.addHttpRoutes(http);

// ---------------------------------------------------------------------------
// ClipForge worker VM API (token-gated REST — the Oracle Cloud worker polls
// these; nothing here trusts the browser)
// ---------------------------------------------------------------------------

const TOKEN = process.env.CLIPFORGE_WORKER_TOKEN ?? "";
const MAX_BODY = 50_000; // log lines and progress payloads are small

function checkToken(req: Request): Response | null {
  if (TOKEN === "") {
    return json(
      { error: "CLIPFORGE_WORKER_TOKEN is not set in the deployment env" },
      503,
    );
  }
  const provided = req.headers.get("authorization") ?? "";
  if (provided !== `Bearer ${TOKEN}`) {
    return json({ error: "invalid or missing worker token" }, 401);
  }
  return null;
}
function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Authorization, Content-Type",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    },
  });
}

/** Worker heartbeat: VM identity + model state. Also drains a pending STOP. */
http.route({
  path: "/api/worker/heartbeat",
  method: "POST",
  handler: httpAction(async (ctx, req) => {
    const denied = checkToken(req);
    if (denied) return denied;
    const body = (await req.json().catch(() => ({}))) as {
      modelLoaded?: boolean;
      pendingJobs?: number;
    };
    const modelLoaded = body.modelLoaded === true;
    const pendingJobs = Number.isFinite(body.pendingJobs) ? Number(body.pendingJobs) : 0;
    const out = (await ctx.runAction(api.vm.vmWorkerHeartbeatPublic, {
      modelLoaded,
      pendingJobs,
    })) as { ok?: boolean; stopped?: boolean };
    return json({
      ok: out?.ok === true,
      stopRequested: out?.stopped === true,
      // the worker treats this as "shut down cleanly after the current job"
      drainAndStop: out?.stopped === true,
    });
  }),
});

/**
 * Claim the oldest queued URL job. The VM fetches it with the yt-dlp Python
 * API; progress arrives via /api/worker/progress and the file via storage.
 */
http.route({
  path: "/api/worker/claim",
  method: "GET",
  handler: httpAction(async (ctx, req) => {
    const denied = checkToken(req);
    if (denied) return denied;
    const job = await ctx.runQuery(internal.clips.findQueuedUrlJob, {});
    if (job === null) return json({ job: null }, 200);
    await ctx.runMutation(internal.clips.vmClaimJob, { jobId: job.jobId });
    return json({ job });
  }),
});

/** Fresh Convex storage upload URL for the VM's fetched MP4. */
http.route({
  path: "/api/worker/upload-url",
  method: "POST",
  handler: httpAction(async (ctx, req) => {
    const denied = checkToken(req);
    if (denied) return denied;
    await req.json().catch(() => ({})); // jobId reserved for future per-job scoping
    const url = await ctx.runMutation(internal.clips.generateUploadUrlInternal, {});
    return json({ uploadUrl: url });
  }),
});

/** Fetch progress from the VM: yt-dlp download %, MB done/total. */
http.route({
  path: "/api/worker/progress",
  method: "POST",
  handler: httpAction(async (ctx, req) => {
    const denied = checkToken(req);
    if (denied) return denied;
    const body = (await req.json().catch(() => null)) as {
      jobId?: string;
      bytesDone?: number;
      bytesTotal?: number | null;
    } | null;
    if (body === null || typeof body.jobId !== "string") {
      return json({ error: "jobId required" }, 400);
    }
    await ctx.runMutation(internal.clips.reportFetchProgressInternal, {
      jobId: body.jobId as never,
      bytesDone: Number(body.bytesDone ?? 0),
      bytesTotal:
        body.bytesTotal === null || body.bytesTotal === undefined
          ? null
          : Number(body.bytesTotal),
    });
    return json({ ok: true });
  }),
});

/**
 * The VM fetcher chain exhausted its retries for this job — record the exact
 * yt-dlp/stderr error. The server-side chain then falls through to the next
 * fetcher (direct / upload promotion) on its next run.
 */
http.route({
  path: "/api/worker/fetch-failed",
  method: "POST",
  handler: httpAction(async (ctx, req) => {
    const denied = checkToken(req);
    if (denied) return denied;
    const body = (await req.json().catch(() => null)) as {
      jobId?: string;
      error?: string;
    } | null;
    if (body === null || typeof body.jobId !== "string" || typeof body.error !== "string") {
      return json({ error: "jobId and error required" }, 400);
    }
    await ctx.runMutation(internal.clips.reportVmFetchFailed, {
      jobId: body.jobId as never,
      error: body.error.slice(0, MAX_BODY),
    });
    return json({ ok: true });
  }),
});

export default http;
