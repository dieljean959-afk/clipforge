// useRenderWorker — the dashboard tab IS the render worker.
//
// Drives the real pipeline end to end, one artifact at a time:
//   grabbing     -> upload (real bytes) or the server fetch action, then
//                   DECODE-verify the stored MP4 and report real metadata
//   transcribing -> real Whisper word-level transcript stored to file storage
//   finding      -> server-side Gemini action (scheduled by reportTranscribe)
//   rendering    -> real ffmpeg.wasm renders, uploaded and reported per clip
//
// Every failure path surfaces the EXACT error; nothing falls back to
// placeholder content.

import { useCallback, useEffect, useRef, useState } from "react";
import { useAction, useMutation, useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";
import type { Doc, Id } from "@/convex/_generated/dataModel";
import { buildAssCues } from "@/convex/engine";
import {
  probeVideo,
  extractAudio,
  transcribeWords,
  renderClip,
  renderLog,
  type WordTiming,
} from "@/lib/render-driver";

type JobDoc = Doc<"jobs">;

export function useRenderWorker(): {
  /** true while this tab is actively working a job */
  busy: boolean;
  /** the job this tab claimed, if any */
  claimedJobId: Id<"jobs"> | null;
} {
  const [claimedJobId, setClaimedJobId] = useState<Id<"jobs"> | null>(null);
  const [busy, setBusy] = useState(false);

  // --- protocol endpoints ----------------------------------------------------
  const claimJob = useMutation(api.clips.clientClaimJob);
  const reportFetch = useMutation(api.clips.reportFetch);
  const reportFetchFailure = useMutation(api.clips.reportFetchFailure);
  const reportTranscribe = useMutation(api.clips.reportTranscribe);
  const reportTranscribeFailure = useMutation(api.clips.reportTranscribeFailure);
  const reportTranscribeProgress = useMutation(api.clips.reportTranscribeProgress);
  const reportClipRendered = useMutation(api.clips.reportClipRendered);
  const reportClipRenderFailure = useMutation(api.clips.reportClipRenderFailure);
  const generateUploadUrl = useMutation(api.clips.generateUploadUrl);
  const fetchAction = useAction(api.fetch.fetchVideo);
  const envCheck = useAction(api.diagnostics.envCheck);

  // --- reactive state ----------------------------------------------------------
  const status = useQuery(api.clips.getWorkerStatus, {});
  const online = status?.online === true;
  const job = useQuery(
    api.clips.getJob,
    claimedJobId !== null ? { jobId: claimedJobId } : "skip",
  );
  const clips = useQuery(
    api.clips.listClips,
    claimedJobId !== null ? { jobId: claimedJobId } : "skip",
  );
  const sourceUrl = useQuery(
    api.clips.getSourceMediaUrl,
    claimedJobId !== null ? { jobId: claimedJobId } : "skip",
  );

  // --- shared helpers ----------------------------------------------------------
  const uploadBlob = useCallback(
    async (blob: Blob, type: string): Promise<Id<"_storage">> => {
      const url = await generateUploadUrl();
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": type },
        body: blob,
      });
      if (!res.ok) throw new Error(`storage upload failed (HTTP ${res.status})`);
      const meta = (await res.json()) as { storageId: Id<"_storage"> };
      return meta.storageId;
    },
    [generateUploadUrl],
  );

  const loadSourceBlob = useCallback(async (url: string): Promise<Blob> => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`stored source unreadable (HTTP ${res.status})`);
    return await res.blob();
  }, []);

  // --- stage runner --------------------------------------------------------------
  const runningRef = useRef(false);
  const sourceBlobRef = useRef<{ key: string; blob: Blob; url: string } | null>(null);

  const ensureSourceBlob = useCallback(
    async (j: JobDoc): Promise<{ blob: Blob; url: string }> => {
      const key = `${j._id}:${j.sourceStorageId ?? "nosrc"}`;
      if (sourceBlobRef.current?.key === key) return sourceBlobRef.current;
      if (sourceUrl === undefined || sourceUrl === null) {
        throw new Error("source URL not ready");
      }
      const blob = await loadSourceBlob(sourceUrl);
      const entry = { key, blob, url: URL.createObjectURL(blob) };
      sourceBlobRef.current = entry;
      return entry;
    },
    [loadSourceBlob, sourceUrl],
  );

  const runStage = useCallback(
    async (j: JobDoc) => {
      if (runningRef.current) return;
      runningRef.current = true;
      setBusy(true);
      try {
        // ---------------- grabbing: fetch (server) + DECODE-verify (here) ----
        if (j.stage === "grabbing") {
          if (j.sourceStorageId === undefined) {
            if (j.isUpload) {
              throw new Error("upload job has no stored file — attach the file again");
            }
            renderLog(`fetching ${j.source}`);
            const res = await fetchAction({ jobId: j._id });
            if (!res.fetched) {
              throw new Error(res.error ?? "all fetchers failed");
            }
            // storage id lands via reportFetchStored; the reactive job update
            // re-enters this stage with sourceStorageId set
            renderLog(`fetched ${(res.bytes / (1024 * 1024)).toFixed(1)} MB via ${res.via}`);
            return;
          }
          // URL may still be resolving (query catch-up) — retry on re-entry
          if (sourceUrl === undefined) return;
          const src = await ensureSourceBlob(j);
          const meta = await probeVideo(src.blob);
          URL.revokeObjectURL(meta.url);
          await reportFetch({
            jobId: j._id,
            storageId: j.sourceStorageId,
            bytes: j.sourceBytes ?? src.blob.size,
            durationSec: meta.durationSec,
            width: meta.width,
            height: meta.height,
            fetchSource: j.fetchSource ?? (j.isUpload ? "upload" : "cobalt"),
          });
          renderLog(
            `source verified: ${meta.durationSec.toFixed(1)}s ${meta.width}x${meta.height}`,
          );
          return;
        }

        // ---------------- transcribing: REAL Whisper word timestamps ----------
        if (j.stage === "transcribing") {
          const src = await ensureSourceBlob(j);
          renderLog("decoding audio (WebAudio)...");
          const audio = await extractAudio(src.blob);
          const totalSec = audio.length / 16000;
          renderLog(`whisper starting — ${Math.round(totalSec)}s of audio`);
          const words: WordTiming[] = await transcribeWords(audio, (processed, total) => {
            void reportTranscribeProgress({ jobId: j._id, processedSec: processed }).catch(
              () => undefined,
            );
            if (Math.round(processed) % 30 < 1) {
              renderLog(`whisper ${processed.toFixed(0)}s / ${total.toFixed(0)}s`);
            }
          });
          if (words.length === 0) {
            throw new Error("Whisper produced no words (silent or unsupported audio?)");
          }
          const storageId = await uploadBlob(
            new Blob([JSON.stringify(words)], { type: "application/json" }),
            "application/json",
          );
          await reportTranscribe({
            jobId: j._id,
            storageId,
            wordCount: words.length,
            durationSec: j.durationSec ?? totalSec,
          });
          renderLog(`transcript stored (${words.length} words) — Gemini scoring next`);
          return;
        }

        // ---------------- finding: server-side Gemini ------------------------
        if (j.stage === "finding") {
          renderLog("Gemini is scoring transcript chunks...");
          return;
        }

        // ---------------- rendering: REAL ffmpeg.wasm clips -------------------
        if (j.stage === "rendering" && clips !== undefined) {
          const src = await ensureSourceBlob(j);
          const pending = clips.filter((c) => c.renderState === "pending");
          for (const clip of pending) {
            try {
              const started = performance.now();
              renderLog(
                `rendering clip ${clip.order + 1}/${clips.length} (${(clip.end - clip.start).toFixed(1)}s)...`,
              );
              const out = await renderClip({
                mediaUrl: src.url,
                startSec: clip.start,
                endSec: clip.end,
                ass: buildAssCues(clip.captionCues),
                onProgress: (f) => {
                  if (Math.random() < 0.08) {
                    renderLog(`clip ${clip.order + 1} encode ${(f * 100).toFixed(0)}%`);
                  }
                },
              });
              const mediaId = await uploadBlob(out.mp4, "video/mp4");
              const thumbId =
                out.thumb !== null ? await uploadBlob(out.thumb, "image/jpeg") : undefined;
              await reportClipRendered({
                jobId: j._id,
                clipId: clip._id,
                mediaStorageId: mediaId,
                thumbStorageId: thumbId,
                bytes: out.mp4.size,
                width: out.width,
                height: out.height,
                durationSec: out.durationSec,
                renderMs: Math.round(performance.now() - started),
              });
              renderLog(
                `clip ${clip.order + 1} rendered: ${(out.mp4.size / (1024 * 1024)).toFixed(1)} MB`,
              );
            } catch (err) {
              const msg = err instanceof Error ? err.message : String(err);
              await reportClipRenderFailure({ jobId: j._id, clipId: clip._id, error: msg });
              return;
            }
          }
          return;
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        renderLog(`stage ${j.stage} failed: ${msg}`);
        try {
          if (j.stage === "grabbing") {
            await reportFetchFailure({ jobId: j._id, error: msg, fetcher: "browser" });
          } else if (j.stage === "transcribing") {
            await reportTranscribeFailure({ jobId: j._id, error: msg });
          }
        } catch {
          // the watchdog fails the job if even the failure report fails
        }
      } finally {
        runningRef.current = false;
        setBusy(false);
      }
    },
    [
      clips,
      ensureSourceBlob,
      fetchAction,
      reportClipRenderFailure,
      reportClipRendered,
      reportFetch,
      reportFetchFailure,
      reportTranscribe,
      reportTranscribeFailure,
      reportTranscribeProgress,
      sourceUrl,
      uploadBlob,
    ],
  );

  // --- stage entry effect ---------------------------------------------------
  // Key includes the artifacts that arrive reactively mid-stage, so the stage
  // re-enters when its input becomes available (stored source, rendered clips).
  const doneCount = clips?.filter((c) => c.renderState === "done").length ?? 0;
  const stageKey =
    job === undefined || job === null
      ? ""
      : `${job._id}:${job.stage}:${job.sourceStorageId ?? "nosrc"}:${doneCount}`;
  const stageKeyRef = useRef("");

  useEffect(() => {
    if (job === undefined || job === null) return;
    if (stageKeyRef.current === stageKey) return;
    stageKeyRef.current = stageKey;
    if (job.stage === "complete" || job.stage === "failed") {
      const t = setTimeout(() => {
        setClaimedJobId(null);
        stageKeyRef.current = "";
      }, 1500);
      return () => clearTimeout(t);
    }
    void runStage(job);
  }, [job, stageKey, runStage]);

  // --- claim poll --------------------------------------------------------------
  useEffect(() => {
    if (!online || claimedJobId !== null) return;
    let cancelled = false;
    const poll = setInterval(async () => {
      if (cancelled || runningRef.current) return;
      try {
        const j = await claimJob({});
        if (j !== null && !cancelled) {
          renderLog(`claimed job ${j._id} (${j.isUpload ? "upload" : j.source})`);
          setClaimedJobId(j._id);
        }
      } catch {
        // signed out — the UI gates this anyway
      }
    }, 3000);
    return () => {
      cancelled = true;
      clearInterval(poll);
    };
  }, [online, claimedJobId, claimJob]);

  // boot line: the browser worker logs the LLM state once per session
  const bootedRef = useRef(false);
  useEffect(() => {
    if (bootedRef.current) return;
    bootedRef.current = true;
    void envCheck({})
      .then((r) => renderLog(r.bootLine))
      .catch(() => undefined);
  }, [envCheck]);

  return { busy, claimedJobId };
}
