"use node";

// REAL video fetching — no simulated downloads, no cobalt.
//
// COBALT REMOVED PERMANENTLY: cobalt.tools shut down its public v7 API
// (github.com/imputnet/cobalt/discussions/860) and modern cobalt (v10+) no
// longer supports YouTube at all. Any cobalt code path is dead weight and a
// source of confusing errors — it is gone from this chain.
//
// The fetcher chain, in exact order (max 2 attempts each, then the next):
//   1. PRIMARY — the ClipForge worker VM (Oracle Cloud Always Free) running
//      yt-dlp via its PYTHON API ("bv*[height<=1080]+ba/b[height<=1080]",
//      merged to MP4, progress_hooks for real download %). The VM claims
//      URL jobs, downloads, and uploads the bytes to Convex storage. If the
//      VM is asleep, this fetcher reports "worker asleep — flip the ON
//      switch" (the dashboard power switch) instead of a cryptic error.
//   2. SECONDARY — direct HTTP GET for direct .mp4/.webm/.mov links.
//   3. LAST RESORT — manual upload, promoted in the UI with the EXACT error
//      from both fetchers: "Direct URL fetching failed on this platform
//      (<error>). Upload the video file instead — clipping works fully from
//      uploads." Nothing is ever faked.
//
// Bytes are stored in Convex file storage (the shared disk for the API,
// watchdog, browser render worker, and the VM). A job cannot advance past
// the grabbing stage without a stored, size-verified file. The browser then
// decode-verifies the stored file (reportFetch), which advances the pipeline.

import { v } from "convex/values";
import { action, type ActionCtx } from "./_generated/server";
import { internal } from "./_generated/api";

const MAX_BYTES = 2 * 1024 * 1024 * 1024; // 2 GB cap (VM fetches are big-file capable)
const PROGRESS_CHUNK = 1024 * 512; // report every ~512KB
const MAX_ATTEMPTS_PER_FETCHER = 2;

type Ctx = ActionCtx;

async function downloadToStorage(
  ctx: Ctx,
  fileUrl: string,
  onProgress: (done: number, total: number | null) => Promise<void>,
): Promise<{ storageId: string; bytes: number }> {
  const res = await fetch(fileUrl, { redirect: "follow" });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} while downloading the video stream`);
  }
  const totalHeader = res.headers.get("content-length");
  const total = totalHeader !== null ? Number(totalHeader) : null;
  if (total !== null && Number.isFinite(total) && total > MAX_BYTES) {
    throw new Error(
      `video is ${(total / (1024 * 1024)).toFixed(0)} MB — over the ${MAX_BYTES / (1024 * 1024)} MB limit`,
    );
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
        throw new Error(`download exceeded the ${MAX_BYTES / (1024 * 1024)} MB limit`);
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

/**
 * Ask the VM worker to claim this job. The worker (clipforge_worker.py on the
 * Oracle VM) polls the /api/worker/claim HTTP route; it downloads with the
 * yt-dlp Python API and POSTs the file to Convex storage itself. This
 * fetcher's job is only to (a) refuse honestly when the VM is asleep, and
 * (b) wait for the VM's "stored" event (with its own per-attempt timeouts —
 * never an eternal await).
 */
async function vmFetch(
  ctx: Ctx,
  jobId: string,
  source: string,
  onProgress: (done: number, total: number | null) => Promise<void>,
): Promise<{ stored: boolean; error?: string; asleep?: boolean }> {
  // Is the VM awake and its worker heartbeating?
  const vm = (await ctx.runQuery(internal.vmStore.getState, {})) as {
    ociState: string | null;
    workerOk: boolean;
    workerLastSeenAt: number | null;
  } | null;

  if (vm === null || vm.ociState !== "RUNNING" || vm.workerOk !== true) {
    const why =
      vm === null
        ? "worker VM unknown (never heartbeated)"
        : vm.ociState !== "RUNNING"
          ? `VM is ${vm.ociState ?? "OFF"}`
          : "VM up but worker not heartbeating";
    return {
      stored: false,
      asleep: true,
      error: `worker asleep (${why}) — flip the ON switch to wake it (~2 min)`,
    };
  }

  // Nudge the worker loop (its poller picks the job up within ~5s) and wait
  // for the fetch to land, keeping progress fed from the VM's writes.
  const startedAt = Date.now();
  const TIMEOUT_MS = 10 * 60 * 1000; // same hard limit the watchdog enforces
  for (;;) {
    await new Promise((r) => setTimeout(r, 3000));
    const job = (await ctx.runQuery(internal.clips.getJobInternal, {
      jobId: jobId as never,
    })) as {
      stage: string;
      sourceStorageId?: string;
      unitsDone?: number;
      unitsTotal?: number;
      vmAttempts?: number;
      fetchError?: string;
    } | null;
    if (job === null) return { stored: false, error: "job vanished while the VM was fetching" };
    if (job.stage !== "grabbing") {
      // advanced (fetch verified by the browser) or failed by the watchdog
      return { stored: job.sourceStorageId !== undefined };
    }
    if (job.sourceStorageId !== undefined) return { stored: true };

    // keep the UI's byte counter moving from the VM's progress writes
    if (job.unitsDone !== undefined && job.unitsDone > 0) {
      await onProgress(
        job.unitsDone * 1024 * 1024,
        job.unitsTotal && job.unitsTotal > 0 ? job.unitsTotal * 1024 * 1024 : null,
      );
    }

    // The VM reports each failed attempt via reportVmFetchFailed; if it has
    // burned both attempts for this fetcher, stop waiting on it.
    if ((job.vmAttempts ?? 0) >= MAX_ATTEMPTS_PER_FETCHER) {
      return { stored: false, error: job.fetchError ?? "yt-dlp failed on the worker VM" };
    }
    if (Date.now() - startedAt > TIMEOUT_MS) {
      return { stored: false, error: "VM fetch exceeded its 10 min hard limit" };
    }
  }
}

// Kept for the VM-free path: re-exported identity so the HTTP worker API can
// report attempts (see http.ts). Marker import to keep the module graph honest.
export const FETCHER_CHAIN = ["vm-yt-dlp", "direct", "upload"] as const;

/**
 * Grab the source video for real. Order: VM yt-dlp -> direct URL. Every
 * failure is reported verbatim to the job (no placeholder content, no
 * cobalt). The browser reports the decode-verified metadata after the stored
 * file lands (reportFetch), which advances the pipeline only when it verifies.
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

    // 1) PRIMARY — the worker VM running yt-dlp (Python API, 1080p max MP4)
    try {
      const r = await vmFetch(ctx, jobId, source, reportProgress);
      if (r.stored) {
        return { fetched: true as const, via: "yt-dlp" as const };
      }
      attempts.push({
        method: "yt-dlp (worker VM)",
        error: r.error ?? "VM fetch did not complete",
      });
      console.error(`[fetch] VM yt-dlp failed: ${r.error ?? "unknown"}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      attempts.push({ method: "yt-dlp (worker VM)", error: msg });
      console.error(`[fetch] VM yt-dlp failed: ${msg}`);
    }

    // 2) SECONDARY — direct HTTP GET (works for .mp4/.webm/.mov links)
    if (/\.(mp4|webm|mov)(\?|$)/i.test(source)) {
      for (let attempt = 1; attempt <= MAX_ATTEMPTS_PER_FETCHER; attempt++) {
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
          attempts.push({ method: `direct (try ${attempt})`, error: msg });
          console.error(`[fetch] direct failed (try ${attempt}): ${msg}`);
        }
      }
    }

    // 3) FAIL LOUDLY with the full fetchers-tried list. The UI promotes
    //    manual upload as the primary input with this exact error.
    const summary = attempts.map((a) => `${a.method}: ${a.error}`).join(" | ");
    const lastError = attempts[attempts.length - 1]?.error ?? "unknown error";
    await ctx.runMutation(internal.clips.reportFetchFailureInternal, {
      jobId,
      error: summary,
      fetcher: attempts.map((a) => a.method).join(" → ") || "none",
      uploadHint: `Direct URL fetching failed on this platform (${lastError}). Upload the video file instead — clipping works fully from uploads.`,
    });
    return { fetched: false as const, error: summary };
  },
});
