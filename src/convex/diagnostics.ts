"use node";

// Startup self-checks for the LLM integration. Mirrors the requirement:
// "At boot, worker logs: `LLM provider: gemini | key loaded: YES/NO`."
// Only env var NAMES are ever reported — values stay secret.

import { v } from "convex/values";
import { action } from "./_generated/server";

/** Env var names the pipeline understands. Only names are surfaced. */
const RELEVANT_ENV_NAMES = [
  "GEMINI_API_KEY",
  "CLIPFORGE_WORKER_TOKEN",
  "OCI_TENANCY_OCID",
  "OCI_USER_OCID",
  "OCI_FINGERPRINT",
  "OCI_PRIVATE_KEY",
  "OCI_INSTANCE_OCID",
  "OCI_REGION",
  "VLY_INTEGRATION_KEY",
  "VLY_APP_NAME",
  "NODE_ENV",
] as const;

/**
 * Which LLM credential path is available.
 *
 * ONLY a real GEMINI_API_KEY counts. The platform's VLY_INTEGRATION_KEY is a
 * deployment token, not a Gemini credential — claiming "key loaded: YES"
 * through it while detection refuses would be exactly the dishonest state
 * this module exists to prevent. No key -> detection refuses (no fallback).
 */
export function llmKeyState(): {
  provider: "gemini-direct" | "none";
  keyLoaded: boolean;
  keyName: string | null;
} {
  if (process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.length > 8) {
    return { provider: "gemini-direct", keyLoaded: true, keyName: "GEMINI_API_KEY" };
  }
  if (process.env.VLY_INTEGRATION_KEY && process.env.VLY_INTEGRATION_KEY.length > 8) {
    return { provider: "none", keyLoaded: false, keyName: null };
  }
  return { provider: "none", keyLoaded: false, keyName: null };
}

/** Human-readable boot line, e.g. "LLM provider: gemini | key loaded: YES". */
export function llmBootLine(): string {
  const s = llmKeyState();
  return `LLM provider: gemini | key loaded: ${s.keyLoaded ? "YES" : "NO"}${
    s.keyName ? ` (via ${s.keyName})` : " — add GEMINI_API_KEY to environment"
  }`;
}

/** Public action so the UI can run the boot check on demand and show it. */
export const envCheck = action({
  args: {},
  handler: async () => {
    const boot = llmBootLine();
    // Real boot log line — grep target for the acceptance test.
    console.log(`[env-check] ${boot}`);
    return {
      bootLine: boot,
      envNames: RELEVANT_ENV_NAMES.filter((n) => process.env[n] !== undefined),
      llm: llmKeyState(),
      // NOTE on ffmpeg/yt-dlp: this runtime is serverless (Convex) — no
      // subprocess, no filesystem, no yt-dlp/ffmpeg binaries here. URL
      // fetching therefore runs on the worker VM (Oracle Cloud Always Free,
      // yt-dlp Python API — see vm/README.md) and ffmpeg.wasm renders run in
      // the user's browser. cobalt was removed permanently: the public v7 API
      // is shut down and cobalt v10+ dropped YouTube support.
      runtime: "serverless-node (fetch: worker-VM yt-dlp / render: browser ffmpeg.wasm)",
    };
  },
});

/** Cheap action so the dashboard can banner the LLM state on demand. */
export const getLlmStatus = action({
  args: {},
  handler: async () => {
    const s = llmKeyState();
    const bootLine = llmBootLine();
    console.log(`[llm-status] ${bootLine}`);
    return {
      provider: s.provider,
      keyLoaded: s.keyLoaded,
      keyName: s.keyName,
      bootLine,
    };
  },
});

/** Record the boot line on a named surface (used by ops to verify). */
export const recordBootCheck = action({
  args: { surface: v.string() },
  handler: async (ctx, { surface }) => {
    const boot = llmBootLine();
    console.log(`[env-check:${surface}] ${boot}`);
    return boot;
  },
});
