import { useMemo, useState } from "react";
import { toast } from "sonner";
import { AudioLines, BadgeCheck, Captions, Film, Loader2, RefreshCw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { LogoWordmark } from "@/components/clipforge/LogoMark";
import { LocalWorkerCard } from "@/components/clipforge/LocalWorkerCard";
import { RightsNotice } from "@/components/clipforge/RightsNotice";
import { NewJobForm } from "@/components/clipforge/NewJobForm";
import { JobCard } from "@/components/clipforge/JobCard";
import { ClipCard, ScoreBadge } from "@/components/clipforge/ClipCard";
import { ClipPreview } from "@/components/clipforge/ClipPreview";
import { ExportDialog } from "@/components/clipforge/ExportDialog";
import { ConfirmDeleteJob } from "@/components/clipforge/ConfirmDeleteJob";
import { CaptionStyleDialog } from "@/components/clipforge/CaptionStyleDialog";
import { copyClipMeta, buildClipSrt } from "@/lib/clip-export";
import { useJobs, useClips } from "@/hooks/use-poll";
import { api, type VerifyClipsDto } from "@/lib/api";
import { fmtAgo, fmtClock, fmtDate, sourceLabel } from "@/lib/clipforge";
import { cn } from "@/lib/utils";
import type { ClipDto } from "@/lib/api";

export default function Dashboard() {
  const { jobs, error: apiError, refresh: refreshJobs, checkOne } = useJobs(2000);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [previewClip, setPreviewClip] = useState<ClipDto | null>(null);
  const [redetecting, setRedetecting] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<VerifyClipsDto | null>(null);

  const allJobs = useMemo(() => jobs ?? [], [jobs]);
  const activeJobs = allJobs.filter(
    (j) => j.stage !== "complete" && j.stage !== "failed",
  );
  const doneJobs = allJobs.filter((j) => j.stage === "complete");
  const failedJobs = allJobs.filter((j) => j.stage === "failed");

  // Auto-select the newest completed job without a setState-in-effect:
  // the effective selection is derived, and the user's explicit pick is kept
  // when still valid.
  const effectiveJobId =
    selectedJobId !== null && doneJobs.some((j) => j._id === selectedJobId)
      ? selectedJobId
      : (doneJobs[0]?._id ?? null);
  const selectedJob = doneJobs.find((j) => j._id === effectiveJobId) ?? null;
  const { clips } = useClips(effectiveJobId, 2000);

  // keep the preview dialog's clip object fresh as renders complete
  const freshPreview = useMemo(
    () => (previewClip ? clips.find((c) => c._id === previewClip._id) ?? previewClip : null),
    [previewClip, clips],
  );

  const runRedetect = async (jobId: string) => {
    setRedetecting(jobId);
    try {
      await api.redetect(jobId);
      toast.info("Re-detection queued — Gemini is re-scoring this batch.");
      void refreshJobs();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Re-detection failed.");
    } finally {
      setRedetecting(null);
    }
  };

  // Re-encode with fresh captions, reusing the cached transcript and cuts.
  // This is the repair path for a caption change: the karaoke preset once
  // rendered "bro." + "Did" as "bro.Did" in clips that were ALREADY encoded,
  // and the only way to fix pixels is to encode them again.
  const [rerendering, setRerendering] = useState<string | null>(null);
  const runRerender = async (jobId: string) => {
    setRerendering(jobId);
    try {
      const res = await api.rerender(jobId);
      toast.info(
        `Re-rendering ${res.clipsReset} clip${res.clipsReset === 1 ? "" : "s"} with fresh captions — the transcript is reused, so no re-transcription.`,
      );
      void refreshJobs();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Re-render failed.");
    } finally {
      setRerendering(null);
    }
  };

  // Re-decode the audio with word confidence, then re-score and re-encode.
  // Slower than a re-render (it re-transcribes) but it is the only thing that
  // can change WHAT WAS HEARD, which is what the [muttering] markers depend on.
  const [retranscribing, setRetranscribing] = useState<string | null>(null);
  const runRetranscribe = async (jobId: string) => {
    setRetranscribing(jobId);
    try {
      await api.retranscribe(jobId);
      toast.info(
        "Re-transcribing — the source is reused, the audio is re-decoded with confidence scores, then every clip is re-encoded.",
      );
      void refreshJobs();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Re-transcribe failed.");
    } finally {
      setRetranscribing(null);
    }
  };

  // dev endpoint: ffprobe every rendered clip — honest valid/invalid report
  const runVerify = async (jobId: string) => {
    setVerifying(true);
    try {
      const result = await api.verifyClips(jobId);
      setVerifyResult(result);
      if (result.invalid === 0) {
        toast.success(`All ${result.valid} rendered clips verified with ffprobe.`);
      } else {
        toast.error(
          `${result.invalid} of ${result.total} clips failed verification — see the report.`,
        );
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Verification failed.");
    } finally {
      setVerifying(false);
    }
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* top bar */}
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/85 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-4">
          <a href="/" className="cursor-pointer">
            <LogoWordmark />
          </a>
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm" className="cursor-pointer gap-1.5 text-xs">
              <a href="/">
                Landing
              </a>
            </Button>
          </div>
        </div>
      </header>

      {/* hosted pipeline status (CPU/RAM/queue/warmup) */}
      <div className="mx-auto w-full max-w-6xl px-4 pt-6">
        <LocalWorkerCard />
        {apiError && (
          <div className="mt-3 rounded-xl border border-destructive/50 bg-destructive/10 px-4 py-3">
            <p className="text-sm font-semibold text-destructive">
              API unreachable: {apiError}
            </p>
            <p className="mt-1 text-xs leading-relaxed text-destructive/80">
              {/^(HTTP [45]\d\d)/.test(apiError)
                ? "The pipeline server answered with an HTTP error on /api/health — the response is logged server-side; try again in a minute or check the deployment logs."
                : "The pipeline server did not answer /api/health. If this persists, the deployment is restarting — try again in a minute."}
            </p>
          </div>
        )}
      </div>

      <div className="mx-auto w-full max-w-6xl px-4 py-8">
        {/* header + new job */}
        <section className="grid gap-6 lg:grid-cols-[1fr_380px]">
          <div>
            <p className="text-sm text-muted-foreground">
              Processed on ClipForge's server — videos are deleted after you download them
            </p>
            <h1 className="mt-1 text-3xl font-black tracking-tight">
              The forge
            </h1>
            <p className="mt-3 max-w-lg text-sm leading-relaxed text-muted-foreground">
              Paste a video you own, watch it move through the server pipeline,
              and export vertical clips with captions ready to post.
            </p>
            <RightsNotice className="mt-5 max-w-lg" compact />
          </div>
          <div className="rounded-xl border border-primary/30 bg-card p-5 neon-glow-sm">
            <h2 className="flex items-center gap-2 font-bold tracking-tight">
              <Sparkles className="size-4 text-primary" />
              New batch
            </h2>
            <NewJobForm className="mt-4" />
          </div>
        </section>

        {/* active pipeline */}
        {(activeJobs.length > 0 || failedJobs.length > 0) && (
          <section className="mt-10">
            <h2 className="text-lg font-bold tracking-tight">In the forge</h2>
            <div className="mt-3 grid gap-3">
              {[...activeJobs, ...failedJobs].map((job) => (
                <JobCard
                  key={job._id}
                  job={job}
                  onDeleted={refreshJobs}
                  onCheckNow={async (jobId) => {
                    // "Check now" forces an immediate poll of THIS job instead
                    // of waiting out the 2s interval, so activity can be
                    // confirmed on demand rather than by reloading the page.
                    const fresh = await checkOne(jobId);
                    if (fresh) {
                      toast.success(`Checked — ${fresh.stage} · updated ${
                        typeof fresh.lastProgressAt === "number"
                          ? fmtAgo(Date.now() - fresh.lastProgressAt)
                          : "just now"
                      }`);
                    }
                  }}
                />
              ))}
            </div>
          </section>
        )}

        {/* completed jobs */}
        <section className="mt-10">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold tracking-tight">Your batches</h2>
              {/* Caption style is global, not per batch, so it lives out here
                  rather than inside the selected-job toolbar — it has to be
                  reachable before the first batch exists. */}
              <CaptionStyleDialog />
            </div>
            {selectedJob && (
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  className="cursor-pointer gap-1.5 text-xs"
                  disabled={verifying || redetecting !== null}
                  onClick={() => void runVerify(selectedJob._id)}
                >
                  {verifying ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <BadgeCheck className="size-3.5 text-primary" />
                  )}
                  Verify
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="cursor-pointer gap-1.5 text-xs"
                  disabled={redetecting !== null}
                  onClick={() => void runRedetect(selectedJob._id)}
                >
                  {redetecting === selectedJob._id ? (
                    <RefreshCw className="size-3.5 animate-spin" />
                  ) : (
                    <Sparkles className="size-3.5 text-primary" />
                  )}
                  Re-detect with AI
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="cursor-pointer gap-1.5 text-xs"
                  disabled={
                    retranscribing !== null || rerendering !== null || redetecting !== null
                  }
                  onClick={() => void runRetranscribe(selectedJob._id)}
                >
                  {retranscribing === selectedJob._id ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <AudioLines className="size-3.5 text-primary" />
                  )}
                  Re-transcribe
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="cursor-pointer gap-1.5 text-xs"
                  disabled={rerendering !== null || redetecting !== null}
                  onClick={() => void runRerender(selectedJob._id)}
                >
                  {rerendering === selectedJob._id ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <Captions className="size-3.5 text-primary" />
                  )}
                  Re-render captions
                </Button>
                <ExportDialog job={selectedJob} clips={clips} />
                <ConfirmDeleteJob
                  job={selectedJob}
                  onDeleted={(id) => {
                    if (selectedJobId === id) setSelectedJobId(null);
                    void refreshJobs();
                  }}
                />
              </div>
            )}
          </div>

          {doneJobs.length === 0 ? (
            <div className="mt-3 rounded-xl border border-dashed border-border p-10 text-center">
              <Film className="mx-auto size-8 text-muted-foreground" />
              <p className="mt-3 text-sm font-medium">No batches yet</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Queue your first video with the form above.
              </p>
            </div>
          ) : (
            <>
              <div className="mt-3 flex flex-wrap gap-2">
                {doneJobs.map((job) => (
                  <span
                    key={job._id}
                    className={cn(
                      "inline-flex items-center rounded-full border text-xs font-medium transition",
                      effectiveJobId === job._id
                        ? "border-primary/60 bg-primary/10 text-primary"
                        : "border-border text-muted-foreground hover:text-foreground",
                    )}
                  >
                    <button
                      type="button"
                      onClick={() => setSelectedJobId(job._id)}
                      className="cursor-pointer px-3 py-1.5"
                    >
                      {sourceLabel(job).slice(0, 36)} · {job.clipCount ?? 0} clips
                    </button>
                    {/* Delete lives INSIDE the pill, next to the batch it
                        belongs to. It opens a confirmation and never deletes
                        on the same tap. */}
                    <ConfirmDeleteJob
                      job={job}
                      iconOnly
                      className="mr-1.5"
                      onDeleted={(id) => {
                        if (selectedJobId === id) setSelectedJobId(null);
                        void refreshJobs();
                      }}
                    />
                  </span>
                ))}
              </div>

              {/* job summary */}
              {selectedJob && (
                <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-1 rounded-xl border border-border bg-card px-4 py-3 text-xs text-muted-foreground">
                  <span className="font-mono">{fmtDate(selectedJob.createdAt)}</span>
                  {selectedJob.durationSec != null && (
                    <span>
                      {fmtClock(selectedJob.durationSec)} source ·{" "}
                      {selectedJob.totalClipsSec ?? 0}s of clips
                    </span>
                  )}
                  <span>
                    {selectedJob.detectedSource
                      ? `AI-detected via ${selectedJob.detectedSource}`
                      : "Detection failed"}
                  </span>
                  {selectedJob.transcriptWordCount != null && (
                    <span>{selectedJob.transcriptWordCount} words transcribed</span>
                  )}
                  {selectedJob.downloadSource && (
                    <span>via {selectedJob.downloadSource}</span>
                  )}
                </div>
              )}

              {/* clip grid */}
              <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
                {clips.map((clip) => (
                  <ClipCard
                    key={clip._id}
                    clip={clip}
                    onOpen={() => setPreviewClip(clip)}
                  />
                ))}
              </div>
              {clips.length === 0 && (
                <p className="mt-3 text-sm text-muted-foreground">
                  Clips are loading...
                </p>
              )}

              {/* verify-clips report (dev endpoint results) */}
              {verifyResult && (
                <div className="mt-4 rounded-xl border border-border bg-card p-4">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-sm font-semibold">
                      ffprobe verification — {verifyResult.valid}/{verifyResult.total} valid
                    </p>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 cursor-pointer text-xs text-muted-foreground"
                      onClick={() => setVerifyResult(null)}
                    >
                      Dismiss
                    </Button>
                  </div>
                  <div className="mt-2 max-h-56 overflow-y-auto">
                    {verifyResult.clips.map((r) => (
                      <div
                        key={r.clipId}
                        className="flex items-center justify-between gap-3 border-t border-border/60 py-1.5 text-xs first:border-t-0"
                      >
                        <span className="min-w-0 truncate text-muted-foreground">
                          #{(r.order ?? 0) + 1} {r.title}
                        </span>
                        <span
                          className={cn(
                            "shrink-0 font-mono text-[10px]",
                            r.valid ? "text-primary" : "text-destructive",
                          )}
                        >
                          {r.valid
                            ? `ok · ${r.width}×${r.height} · ${r.durationSec?.toFixed(1)}s · ${((r.bytes ?? 0) / (1024 * 1024)).toFixed(1)} MB`
                            : r.reason}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </div>

      {/* clip detail dialog */}
      <Dialog
        open={previewClip !== null}
        onOpenChange={(open) => {
          if (!open) setPreviewClip(null);
        }}
      >
        <DialogContent className="sm:max-w-md">
          {freshPreview && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2 pr-6">
                  {freshPreview.title}
                </DialogTitle>
                <DialogDescription className="flex items-center gap-2">
                  <ScoreBadge score={freshPreview.score} />
                  <span className="font-mono text-[10px]">
                    {fmtClock(freshPreview.start)} → {fmtClock(freshPreview.end)}
                  </span>
                </DialogDescription>
              </DialogHeader>
              <ClipPreview clip={freshPreview} autoPlay className="w-full" />
              <p className="max-h-24 overflow-auto rounded-lg border border-border bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
                {freshPreview.transcript}
              </p>
              <div className="flex gap-2">
                <Button
                  className="flex-1 cursor-pointer gap-1.5"
                  onClick={async () => {
                    try {
                      await copyClipMeta(freshPreview);
                      toast.success("Caption + hashtags copied.");
                    } catch {
                      toast.error("Clipboard unavailable.");
                    }
                  }}
                >
                  Copy caption
                </Button>
                <Button
                  variant="outline"
                  className="cursor-pointer gap-1.5"
                  onClick={() => buildClipSrt(freshPreview)}
                >
                  .srt
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </main>
  );
}
