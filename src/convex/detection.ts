"use node";

// AI clip detection (Gemini free-tier style LLM scoring) with graceful
// fallback to the deterministic engine. Runs in the Convex node runtime.

import { v } from "convex/values";
import { action } from "./_generated/server";
import { internal } from "./_generated/api";
import { vly } from "../lib/vly-integrations";
import {
  buildClipDetectionPrompt,
  buildTranscript,
  heuristicClipDetection,
  parseClipDetectionResponse,
  segmentTranscript,
  simulateDuration,
  type ClipCandidateOutput,
} from "./engine";

interface CompletionRequest {
  model: string;
  messages: { role: string; content: string }[];
  temperature?: number;
  maxTokens?: number;
}

interface CompletionResponse {
  success: boolean;
  data?: unknown;
  error?: string;
}

/** Extract assistant text from the various completion payload shapes. */
function extractText(data: unknown): string | null {
  if (typeof data === "string") return data;
  if (data === null || typeof data !== "object") return null;
  const rec = data as Record<string, unknown>;
  for (const key of ["text", "content", "message", "output", "response"]) {
    const val = rec[key];
    if (typeof val === "string" && val.length > 0) return val;
  }
  // OpenAI-style choices
  const choices = rec.choices;
  if (Array.isArray(choices) && choices.length > 0) {
    const first = choices[0] as Record<string, unknown>;
    const msg = first?.message;
    if (msg && typeof msg === "object") {
      const content = (msg as Record<string, unknown>).content;
      if (typeof content === "string") return content;
    }
  }
  return null;
}

/**
 * Ask the LLM to pick the most viral windows from the transcript. Returns
 * null when the model is unavailable or returns unusable output — the caller
 * then falls back to heuristic detection.
 */
async function aiDetectCandidates(
  source: string,
  durationSec: number,
): Promise<ClipCandidateOutput[] | null> {
  if (!process.env.VLY_INTEGRATION_KEY) return null;

  const ai = vly as unknown as {
    completion?: (req: CompletionRequest) => Promise<CompletionResponse>;
  };
  if (typeof ai.completion !== "function") return null;

  const words = buildTranscript(source, durationSec);
  const segments = segmentTranscript(words);
  const prompt = buildClipDetectionPrompt(segments, durationSec);

  try {
    const result = await ai.completion({
      model: "gemini-2.0-flash",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.4,
      maxTokens: 2000,
    });
    if (!result.success) return null;
    const text = extractText(result.data);
    if (text === null) return null;
    return parseClipDetectionResponse(text, words, 12);
  } catch {
    // Model refused, rate limited, malformed JSON... -> heuristic fallback
    return null;
  }
}

/**
 * Re-run clip detection for a completed job using the LLM, falling back to
 * the deterministic engine when the model is unavailable. Rewrites the job's
 * clips in place.
 */
export const redetectClips = action({
  args: { jobId: v.id("jobs") },
  handler: async (
    ctx,
    { jobId },
  ): Promise<{ clipSource: "ai" | "heuristic"; count: number }> => {
    const userId = await ctx.runQuery(internal.clips.getCurrentUserId, {});
    if (userId === null) throw new Error("Sign in first.");

    const job = await ctx.runQuery(internal.clips.getJobForUser, {
      jobId,
      userId,
    });
    if (job === null) throw new Error("Job not found.");
    if (job.stage !== "complete") {
      throw new Error("Wait for the job to finish before re-detecting.");
    }

    const duration =
      job.durationSec ?? simulateDuration(job.source, job.isUpload);

    const aiCandidates = await aiDetectCandidates(job.source, duration);

    const clipSource: "ai" | "heuristic" =
      aiCandidates !== null && aiCandidates.length >= 3 ? "ai" : "heuristic";
    const candidates =
      clipSource === "ai" && aiCandidates !== null
        ? aiCandidates
        : heuristicClipDetection(buildTranscript(job.source, duration), {
            maxClips: 12,
            seedSalt: `${job.source}:${job._id}:redetect`,
          });

    const { count } = await ctx.runMutation(internal.clips.applyDetectedClips, {
      jobId,
      userId,
      clipSource,
      candidates,
    });

    return { clipSource, count };
  },
});
