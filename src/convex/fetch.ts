"use node";

// REAL video fetching — no simulated downloads.
//
// This platform's serverless runtime has no subprocess/filesystem, so
// yt-dlp/ffmpeg cannot run here. The real fetch chain is:
//   1. cobalt.tools HTTP API (supports YouTube, TikTok, Vimeo, X, ...)
//   2. direct HTTP GET for direct .mp4/.webm/.mov links (or cobalt-provided)
// Bytes are stored in Convex file storage (the shared "disk" for the API,
// watchdog, and browser render worker). A job cannot advance past the
// grabbing stage without a stored, size-verified MP4.

import { v } from "convex/values";
import { action } from "./_generated/server";
import { api, internal } from "./_generated/api";

const COBALT_ENDPOINT = process.env.COBALT_API_URL ?? "https://api.cobalt.tools/api/json";
const MAX_BYTES = 200 * 1024 * 1024; // 200 MB cap for the preview environment
const PROGRESS_CHUNK = 1024 * 512; // report every ~512KB

interface CobaltResponse {
  status?: string;
  url?: string;
  text?: string;
  error?: { code?: string } | string;
}

async function cobaltRequest(source: string): Promise<string> {
  const res = await fetch(COBALT_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      url: source,
      videoQuality: "1080",
      filenameStyle: "basic",
      // audioFormat: keep original audio — clips preserve the creator's voice
    }),
  });
  const data = (await res.json().catch(() => null)) as CobaltResponse | null;
  if (data === null) {
    throw new Error(`cobalt returned non-JSON (HTTP ${res.status})`);
  }
  if (data.status === "stream" || data.status === "redirect" || data.status === "tunnel") {
    if (typeof data.url !== "string" || data.url.length === 0) {
      throw new Error("cobalt returned no stream URL");
    }
    return data.url;
  }
  const errText =
    typeof data.error === "object" && data.error !== null
      ? (data.error.code ?? JSON.stringify(data.error))
      : (data.text ?? data.status ?? `HTTP ${res.status}`);
  throw new Error(`cobalt: ${errText}`);
}

async function downloadToStorage(
  ctx: { storage: { generateUploadUrl: () => Promise<string> } },
  fileUrl: string,
  onProgress: (done: number, total: number | null) => Promise<void> | void,
): Promise<{ storageId: string; bytes: number }> {
  const res = await fetch(fileUrl, { redirect: "follow" });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} while downloading the video stream`);
  }
  const totalHeader = res.headers.get("content-length");
  const total = totalHeader !== null ? Number(totalHeader) : null;
  if (total !== null && Number.isFinite(total) && total > MAX_BYTES) {
    throw new Error(`video is ${(total / (1024 * 1024)).toFixed(0)} MB — over the ${MAX_BYTES / (1024 * 1024)} MB preview limit`);
  }
  if (!res.body) {
    throw new Error("empty response body from the fetcher");
  }

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let done = 0;
  let lastReport = 0;
  for (;;) {
    const { value, done: finished } = await reader.read();
    if (value) {
      chunks.push(value);
      done += value.byteLength;
      if (done > MAX_BYTES) {
        throw new Error(`download exceeded the ${MAX_BYTES / (1024 * 1024)} MB preview limit`);
      }
      if (done - lastReport >= PROGRESS_CHUNK) {
        lastReport = done;
        await onProgress(done, total);
      }
    }
    if (finished) break;
  }
  await onProgress(done, total);

  const blob = new Blob(chunks as BlobPart[], { type: "video/mp4" });
  if (blob.size < 1024) {
    throw new Error(`downloaded file is only ${blob.size} bytes — not a real video`);
  }
  const uploadUrl = await ctx.storage.generateUploadUrl();
  const put = await fetch(uploadUrl, {
    method: "POST",
    body: blob,
    headers: { "Content-Type": "video/mp4" },
  });
  if (!put.ok) {
    throw new Error(`storage upload failed (HTTP ${put.status})`);
  }
  const meta = (await put.json()) as { storageId: string };
  return { storageId: meta.storageId, bytes: blob.size };
}

function extType(url: string): string {
  const lower = url.toLowerCase();
  if (lower.includes(".webm")) return "video/webm";
  if (lower.includes(".mov")) return "video/quicktime";
  return "video/mp4";
}

/**
 * Grab the source video for real. Order: cobalt API -> direct URL.
 * Every failure is reported verbatim to the job (no placeholder content).
 * The browser reports the ffprobe/HTMLVideoElement-verified metadata after
 * decoding the stored file (reportFetch), which advances the pipeline.
 */
export const fetchVideo = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, { jobId }) => {
    const userId = await ctx.runQuery(internal.clips.getCurrentUserId, {});
    if (userId === null) throw new Error("Sign in first.");
    const job = (await ctx.runQuery(internal.clips.getJobForUser, {
      jobId,
      userId,
    })) as { source?: string; isUpload?: boolean; stage?: string } | null;
    if (job === null) throw new Error("Job not found.");
    if (job.isUpload) {
      return { fetched: false as const, reason: "upload" as const };
    }
    if (job.stage !== "grabbing") {
      return { fetched: false as const, reason: "stage" as const };
    }
    const source = job.source as string;

    const attempts: { method: string; error: string }[] = [];
    const reportProgress = async (done: number, total: number | null) => {
      await ctx.runMutation(internal.clips.reportFetchProgressInternal, {
        jobId,
        bytesDone: done,
        bytesTotal: total,
      });
    };

    // 1) cobalt API (platform-friendly fetch for YouTube/TikTok/etc.)
    try {
      const streamUrl = await cobaltRequest(source);
      console.log(`[fetch] cobalt OK for ${source.slice(0, 60)} — downloading stream`);
      const { storageId, bytes } = await downloadToStorage(ctx, streamUrl, reportProgress);
      // Byte truth is stored now; duration/width/height are measured by the
      // browser after decoding and reported via reportFetch, which advances
      // the pipeline only when the file verifies.
      await ctx.runMutation(internal.clips.reportFetchStored, {
        jobId,
        storageId: storageId as never,
        bytes,
        fetchSource: "cobalt",
      });
      return { fetched: true as const, bytes, via: "cobalt" as const, storageId };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      attempts.push({ method: "cobalt", error: msg });
      console.error(`[fetch] cobalt failed: ${msg}`);
    }

    // 2) direct HTTP GET (works for .mp4/.webm/.mov links)
    if (/\.(mp4|webm|mov)(\?|$)/i.test(source)) {
      try {
        const { storageId, bytes } = await downloadToStorage(ctx, source, reportProgress);
        await ctx.runMutation(internal.clips.reportFetchStored, {
          jobId,
          storageId: storageId as never,
          bytes,
          fetchSource: "direct",
        });
        return { fetched: true as const, bytes, via: "direct" as const, storageId };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        attempts.push({ method: "direct", error: msg });
        console.error(`[fetch] direct failed: ${msg}`);
      }
    }

    const summary = attempts.map((a) => `${a.method}: ${a.error}`).join(" | ");
    await ctx.runMutation(internal.clips.reportFetchFailureInternal, {
      jobId,
      error: summary,
      fetcher: attempts.map((a) => a.method).join("→") || "none",
    });
    return { fetched: false as const, error: summary };
  },
});
