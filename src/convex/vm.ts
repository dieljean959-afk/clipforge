"use node";

// Oracle Cloud Always Free worker VM control.
//
// The render worker is a real Linux VM (Oracle Cloud Always Free) that runs
// yt-dlp + ffmpeg. The dashboard's ON/OFF power switch talks to the OCI API
// through this module:
//   - status: GET /20160918/instances/{ocid} -> lifecycleState (REAL state,
//     re-queried from OCI — the toggle never reflects "last clicked" state)
//   - start:  POST /20160918/instances/{ocid} {"action":"START"}
//   - stop:   POST /20160918/instances/{ocid} {"action":"STOP","force":false}
//             (SOFT stop — the VM flushes and shuts down cleanly)
//
// Credentials come from env only (never hardcoded):
//   OCI_TENANCY_OCID, OCI_USER_OCID, OCI_FINGERPRINT, OCI_PRIVATE_KEY (PEM
//   contents), OCI_INSTANCE_OCID, OCI_REGION.
//
// Signing follows Oracle's request-signing spec:
//   keyId   = "<tenancy ocid>/<user ocid>/<fingerprint>"
//   headers = "(request-target) date host"                    (GET)
//             "(request-target) date host x-content-sha256
//              content-type content-length"                   (POST)
// We use node:https (not fetch) because undici forbids setting the Host,
// Date, and Content-Length headers these signatures require.

import { createHash, createSign } from "node:crypto";
import https from "node:https";
import { v } from "convex/values";
import { action, type ActionCtx } from "./_generated/server";
import { internal } from "./_generated/api";

// ---------------------------------------------------------------------------
// env + config
// ---------------------------------------------------------------------------

interface OciConfig {
  tenancy: string;
  user: string;
  fingerprint: string;
  privateKeyPem: string;
  instanceOcid: string;
  region: string;
}

/** PEM may be stored with escaped newlines ("\n") — normalize it. */
function normalizePem(raw: string): string {
  const pem = raw.includes("\\n") ? raw.replace(/\\n/g, "\n") : raw;
  return pem.trim() + "\n";
}

export function ociConfig(): OciConfig | { error: string } {
  const missing = [
    "OCI_TENANCY_OCID",
    "OCI_USER_OCID",
    "OCI_FINGERPRINT",
    "OCI_PRIVATE_KEY",
    "OCI_INSTANCE_OCID",
    "OCI_REGION",
  ].filter((n) => !process.env[n] || (process.env[n] as string).length < 4);
  if (missing.length > 0) {
    return {
      error: `VM control is not configured — missing env vars: ${missing.join(", ")}`,
    };
  }
  return {
    tenancy: process.env.OCI_TENANCY_OCID as string,
    user: process.env.OCI_USER_OCID as string,
    fingerprint: process.env.OCI_FINGERPRINT as string,
    privateKeyPem: normalizePem(process.env.OCI_PRIVATE_KEY as string),
    instanceOcid: process.env.OCI_INSTANCE_OCID as string,
    region: process.env.OCI_REGION as string,
  };
}

// ---------------------------------------------------------------------------
// OCI request signing + HTTPS
// ---------------------------------------------------------------------------

interface OciResponse {
  ok: boolean;
  status: number;
  body: string;
}

function sha256Base64(data: string): string {
  return createHash("sha256").update(data, "utf8").digest("base64");
}

function rsaSign(signingString: string, privateKeyPem: string): string {
  const signer = createSign("RSA-SHA256");
  signer.update(signingString);
  return signer.sign(privateKeyPem, "base64");
}

/** Signed request against the OCI IAAS endpoint (node:https — see header). */
function ociRequest(
  cfg: OciConfig,
  method: "GET" | "POST",
  path: string,
  body: string,
): Promise<OciResponse> {
  return new Promise((resolve, reject) => {
    const host = `iaas.${cfg.region}.oraclecloud.com`;
    const date = new Date().toUTCString();
    const headers: Record<string, string> = { date, host };

    let signingString: string;
    if (method === "GET") {
      headers.accept = "application/json";
      signingString = [
        `(request-target): ${method.toLowerCase()} ${path}`,
        `date: ${date}`,
        `host: ${host}`,
      ].join("\n");
    } else {
      const bodyHash = sha256Base64(body);
      headers["content-type"] = "application/json";
      headers["x-content-sha256"] = bodyHash;
      headers["content-length"] = String(Buffer.byteLength(body, "utf8"));
      headers.accept = "application/json";
      signingString = [
        `(request-target): ${method.toLowerCase()} ${path}`,
        `date: ${date}`,
        `host: ${host}`,
        `x-content-sha256: ${bodyHash}`,
        `content-type: application/json`,
        `content-length: ${Buffer.byteLength(body, "utf8")}`,
      ].join("\n");
    }

    const signature = rsaSign(signingString, cfg.privateKeyPem);
    const headerList =
      method === "GET"
        ? "(request-target) date host"
        : "(request-target) date host x-content-sha256 content-type content-length";
    headers.authorization =
      `Signature version="1",keyId="${cfg.tenancy}/${cfg.user}/${cfg.fingerprint}",` +
      `algorithm="rsa-sha256",headers="${headerList}",signature="${signature}"`;

    const req = https.request(
      { hostname: host, port: 443, path, method, headers, timeout: 15_000 },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () => {
          resolve({
            ok: (res.statusCode ?? 500) >= 200 && (res.statusCode ?? 500) < 300,
            status: res.statusCode ?? 500,
            body: Buffer.concat(chunks).toString("utf8"),
          });
        });
      },
    );
    req.on("timeout", () => req.destroy(new Error("OCI request timed out")));
    req.on("error", reject);
    if (body.length > 0 && method === "POST") req.write(body);
    req.end();
  });
}

function instancePath(cfg: OciConfig): string {
  return `/20160918/instances/${encodeURIComponent(cfg.instanceOcid)}`;
}

// ---------------------------------------------------------------------------
// shared VM state (cached OCI truth + worker heartbeat from the VM)
// ---------------------------------------------------------------------------

const VM_STATE_MAX_AGE_MS = 15_000; // cached OCI state older than this re-queries
const WORKER_HEARTBEAT_FRESH_MS = 75_000;

async function runOciGet(ctx: ActionCtx): Promise<{
  ociState: string | null;
  ociError: string | null;
  ociCheckedAt: number;
}> {
  const cfg = ociConfig();
  if ("error" in cfg) return { ociState: null, ociError: cfg.error, ociCheckedAt: Date.now() };
  const res = await ociRequest(cfg, "GET", instancePath(cfg), "");
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const parsed = JSON.parse(res.body) as { message?: string; code?: string };
      if (parsed.message) detail = `${parsed.code ?? "OCI"}: ${parsed.message}`;
    } catch {
      /* keep HTTP status detail */
    }
    return { ociState: null, ociError: `OCI API error (${detail})`, ociCheckedAt: Date.now() };
  }
  const parsed = JSON.parse(res.body) as { lifecycleState?: string };
  return {
    ociState: parsed.lifecycleState ?? "UNKNOWN",
    ociError: null,
    ociCheckedAt: Date.now(),
  };
}

/** Cached instance truth. Only touches the OCI API when the cache is stale. */
async function vmTruth(
  ctx: ActionCtx,
  force = false,
): Promise<{
  ociState: string | null;
  ociError: string | null;
  ociCheckedAt: number;
  workerOk: boolean;
  workerLastSeenAt: number | null;
  modelLoaded: boolean;
  activeJobs: number;
}> {
  const now = Date.now();
  const row = await ctx.runQuery(internal.vmStore.getState, {});
  const needOci =
    force ||
    row === null ||
    row.ociState === null ||
    now - row.ociCheckedAt > VM_STATE_MAX_AGE_MS;

  const oci = needOci ? await runOciGet(ctx) : {
    ociState: row?.ociState ?? null,
    ociError: null as string | null,
    ociCheckedAt: row?.ociCheckedAt ?? now,
  };

  if (needOci) {
    await ctx.runMutation(internal.vmStore.cacheOciState, {
      ociState: oci.ociState,
      ociError: oci.ociError,
      ociCheckedAt: oci.ociCheckedAt,
    });
  }

  return {
    ociState: oci.ociState,
    ociError: oci.ociError,
    ociCheckedAt: oci.ociCheckedAt,
    workerOk:
      row?.workerOk === true &&
      row.workerLastSeenAt !== undefined &&
      now - row.workerLastSeenAt < WORKER_HEARTBEAT_FRESH_MS,
    workerLastSeenAt: row?.workerLastSeenAt ?? null,
    modelLoaded: row?.modelLoaded === true,
    activeJobs: await ctx.runQuery(internal.vmStore.countActiveJobs, {}),
  };
}

// ---------------------------------------------------------------------------
// actions used by the dashboard power switch
// ---------------------------------------------------------------------------

/**
 * Real instance state for page load + 5s polling. Never lies: an OCI error
 * surfaces as ociError and the UI renders an "unknown" striped state.
 */
export const vmStatus = action({
  args: { force: v.optional(v.boolean()) },
  handler: async (ctx, args) => {
    const cfg = ociConfig();
    if ("error" in cfg) {
      return {
        configured: false as const,
        configureError: cfg.error,
        ociState: null,
        ociError: null,
        ociCheckedAt: Date.now(),
        workerOk: false,
        workerLastSeenAt: null,
        modelLoaded: false,
        activeJobs: 0,
      };
    }
    const truth = await vmTruth(ctx, args.force === true);
    return { configured: true as const, configureError: null, ...truth };
  },
});

/**
 * START or SOFT-STOP the worker VM. If jobs are actively rendering, a stop
 * requires confirmBusy — the UI shows "busy — stop after current job?".
 */
export const vmSetPower = action({
  args: {
    action: v.union(v.literal("START"), v.literal("STOP")),
    confirmBusy: v.optional(v.boolean()),
  },
  handler: async (ctx, args) => {
    const cfg = ociConfig();
    if ("error" in cfg) {
      throw new Error(`Cannot ${args.action === "START" ? "start" : "stop"} the worker VM — ${cfg.error}`);
    }

    // Safety: never hard-kill a rendering job without confirmation.
    const truth = await vmTruth(ctx, true);
    if (args.action === "STOP" && truth.activeJobs > 0 && args.confirmBusy !== true) {
      return { busy: true as const, activeJobs: truth.activeJobs };
    }

    const body =
      args.action === "STOP" ? JSON.stringify({ action: "STOP", force: false }) : JSON.stringify({ action: "START" });
    const res = await ociRequest(cfg, "POST", instancePath(cfg), body);
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const parsed = JSON.parse(res.body) as { message?: string; code?: string };
        if (parsed.message) detail = `${parsed.code ?? "OCI"}: ${parsed.message}`;
      } catch {
        /* keep HTTP status detail */
      }
      throw new Error(`OCI ${args.action} failed (${detail})`);
    }
    await ctx.runMutation(internal.vmStore.cacheIntent, { action: args.action });
    console.log(`[vm] instance ${args.action} accepted by OCI`);
    return { busy: false as const, accepted: true as const };
  },
});

/**
 * Worker heartbeat — called by the VM's worker over HTTP (token-checked in
 * http.ts). Also clears a pending STOP once the VM has drained its jobs.
 */
export const vmWorkerHeartbeatPublic = action({
  args: { modelLoaded: v.boolean(), pendingJobs: v.number() },
  handler: async (ctx, { modelLoaded, pendingJobs }) => {
    await ctx.runMutation(internal.vmStore.cacheHeartbeat, { modelLoaded });
    // A queued STOP waits until the VM reports it is idle.
    const row = await ctx.runQuery(internal.vmStore.getState, {});
    if (row?.pendingAction === "STOP" && pendingJobs <= 0) {
      const cfg = ociConfig();
      if (!("error" in cfg)) {
        const res = await ociRequest(
          cfg,
          "POST",
          instancePath(cfg),
          JSON.stringify({ action: "STOP", force: false }),
        );
        if (res.ok) {
          await ctx.runMutation(internal.vmStore.cacheIntent, { action: "STOP" });
          console.log("[vm] deferred STOP executed — worker drained its queue");
          return { ok: true, stopped: true };
        }
      }
    }
    return { ok: true, stopped: false };
  },
});
