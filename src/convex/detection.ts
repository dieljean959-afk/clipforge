"use node";

// AI clip detection — REAL Gemini calls, NO silent fallback.
//
// Contract (from the no-placeholder mandate):
//  - boot line "LLM provider: gemini | key loaded: YES/NO" (diagnostics.ts)
//  - if NO key: detection REFUSES with "AI detection offline — add
//    GEMINI_API_KEY to environment" — never keyword heuristics
//  - 429 rate limits retry with exponential backoff, 3 attempts
//  - the "Re-detect with AI" button calls Gemini for real

import { v } from "convex/values";
import { action } from "./_generated/server";
import { internal, api } from "./_generated/api";
import {
  buildClipDetectionPrompt,
  parseClipDetectionResponse,
  segmentTranscript,
  type TranscriptWord,
} from "./engine";
import { llmKeyState } from "./diagnostics";

const GEMINI_MODEL = "gemini-2.0-flash";
const GEMINI_URL = (model: string, key: string) =>
  `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(key)}`;
const MAX_CLIPS = 12;
const RETRIES = 3;

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** Extract assistant text from a Gemini generateContent response. */
function geminiText(data: unknown): string | null {
  if (data === null || typeof data !== "object") return null;
  const rec = data as Record<string, unknown>;
  const candidates = rec.candidates;
  if (!Array.isArray(candidates) || candidates.length === 0) return null;
  const first = candidates[0] as Record<string, unknown>;
  const content = first.content as Record<string, unknown> | undefined;
  const parts = content?.parts;
  if (!Array.isArray(parts) || parts.length === 0) return null;
  const texts = parts
    .map((p) => (p as Record<string, unknown>).text)
    .filter((t): t is string => typeof t === "string");
  const joined = texts.join("");
  return joined.length > 0 ? joined : null;
}

async function callGemini(
  apiKey: string,
  prompt: string,
): Promise<
  { ok: true; text: string } | { ok: false; error: string; retryable: boolean }
> {
  let lastError = "unknown error";
  for (let attempt = 0; attempt < RETRIES; attempt++) {
    try {
      const res = await fetch(GEMINI_URL(GEMINI_MODEL, apiKey), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: prompt }] }],
          generationConfig: { temperature: 0.4, maxOutputTokens: 4096 },
        }),
      });

      if (res.status === 429) {
        lastError = "Gemini rate limit (429)";
        if (attempt < RETRIES - 1) {
          const backoff = 4000 * Math.pow(2, attempt); // 4s, 8s
          await sleep(backoff);
          continue;
        }
        return { ok: false, error: lastError, retryable: true };
      }

      if (!res.ok) {
        const body = await res.text().catch(() => "");
        return {
          ok: false,
          error: `Gemini HTTP ${res.status}: ${body.slice(0, 300)}`,
          retryable: res.status >= 500,
        };
      }

      const data: unknown = await res.json();
      const text = geminiText(data);
      if (text === null) {
        return { ok: false, error: "Gemini returned an empty response.", retryable: false };
      }
      return { ok: true, text };
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
      if (attempt < RETRIES - 1) {
        await sleep(4000 * Math.pow(2, attempt));
        continue;
      }
    }
  }
  return { ok: false, error: lastError, retryable: true };
}

/**
 * Run REAL Gemini clip detection for a job whose transcript is stored.
 * Scheduled by reportTranscribe; refuses without a key (no fallback).
 */
export const runClipDetection = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }): Promise<{ detected: boolean; error?: string; count?: number }> => {
    const llm = llmKeyState();
    console.log(`[detection] ${llm.keyLoaded ? llmBoot(llm.keyName) : "LLM provider: gemini | key loaded: NO"}`);

    if (llm.provider !== "gemini-direct") {
      await ctx.runMutation(internal.clips.failFinding, {
        jobId,
        error: "AI detection offline — add GEMINI_API_KEY to environment",
        detail:
          "ClipForge does not fall back to keyword heuristics. Detection runs only with a real Gemini key " +
          "(free tier is enough: 15 RPM covers a 10-minute video). Add GEMINI_API_KEY in the Keys tab, then Retry.",
      });
      return { detected: false as const, error: "AI detection offline — add GEMINI_API_KEY to environment" };
    }

    // Load the stored REAL transcript.
    const job = (await ctx.runQuery(internal.clips.getJobInternal, {
      jobId,
    })) as {
      durationSec?: number;
      transcriptStorageId?: string;
      source?: string;
    } | null;
    if (job === null || job.transcriptStorageId === undefined) {
      await ctx.runMutation(internal.clips.failFinding, {
        jobId,
        error: "No stored transcript — nothing to score.",
      });
      return { detected: false as const, error: "no transcript" };
    }

    const blob = await ctx.storage.get(job.transcriptStorageId);
    if (blob === null) {
      await ctx.runMutation(internal.clips.failFinding, {
        jobId,
        error: "Stored transcript could not be read from file storage.",
      });
      return { detected: false as const, error: "transcript unreadable" };
    }
    const words = JSON.parse(await blob.text()) as TranscriptWord[];
    const duration = job.durationSec ?? words[words.length - 1]?.t1 ?? 600;

    // Stream chunk-level progress: chunks sent / chunks total.
    const segments = segmentTranscript(words);
    await ctx.runMutation(internal.clips.reportFindingProgress, {
      jobId,
      done: 0,
      total: segments.length,
      logLine: `${new Date().toISOString().slice(11, 19)} · ▶ finding — ${segments.length} transcript chunks → ${GEMINI_MODEL}`,
    });

    const apiKey = process.env.GEMINI_API_KEY as string;
    const prompt = buildClipDetectionPrompt(segments, duration);

    const started = Date.now();
    const result = await callGemini(apiKey, prompt);
    await ctx.runMutation(internal.clips.reportFindingProgress, {
      jobId,
      done: segments.length,
      total: segments.length,
    });

    if (!result.ok) {
      await ctx.runMutation(internal.clips.failFinding, {
        jobId,
        error: `Gemini clip detection failed: ${result.error}`,
        detail:
          "Retried with exponential backoff. A 429 usually means the key is missing or shared — " +
          "verify GEMINI_API_KEY. No heuristic fallback was used.",
      });
      return { detected: false as const, error: result.error };
    }

    const candidates = parseClipDetectionResponse(result.text, words, MAX_CLIPS);
    if (candidates.length === 0) {
      await ctx.runMutation(internal.clips.failFinding, {
        jobId,
        error: "Gemini returned no parseable clip windows.",
        detail: `Raw response head: ${result.text.slice(0, 300)}`,
      });
      return { detected: false as const, error: "unparseable response" };
    }

    await ctx.runMutation(internal.clips.applyDetectedClips, {
      jobId,
      clipSource: "gemini",
      candidates,
      words,
      logLines: [
        `${new Date().toISOString().slice(11, 19)} · ✓ finding — Gemini scored ${segments.length} chunks in ${Math.round((Date.now() - started) / 1000)}s`,
      ],
    });
    return { detected: true as const, count: candidates.length };
  },
});

function llmBoot(keyName: string | null): string {
  return `LLM provider: gemini | key loaded: YES (via ${keyName})`;
}

/**
 * "Re-detect with AI" — re-enter the finding stage and run Gemini for real.
 * Throws to the UI when the key is missing (visible error, no fake re-run).
 */
export const redetectClips = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }): Promise<{ detected: boolean; error?: string; count?: number }> => {
    const userId = await ctx.runQuery(internal.clips.getCurrentUserId, {});
    if (userId === null) throw new Error("Sign in first.");

    const job = (await ctx.runQuery(internal.clips.getJobForUser, {
      jobId,
      userId,
    })) as { stage?: string; transcriptStorageId?: string } | null;
    if (job === null) throw new Error("Job not found.");
    if (job.transcriptStorageId === undefined) {
      throw new Error("This job has no stored transcript — retry the job to rebuild one.");
    }

    const llm = llmKeyState();
    if (llm.provider !== "gemini-direct") {
      throw new Error("AI detection offline — add GEMINI_API_KEY to environment");
    }

    await ctx.runMutation(internal.clips.reenterFinding, { jobId });
    const result = await ctx.runAction(api.detection.runClipDetection, { jobId });
    return result;
  },
});
