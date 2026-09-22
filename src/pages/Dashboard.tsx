import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Film, RefreshCw, Sparkles } from "lucide-react";
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
import { copyClipMeta, buildClipSrt } from "@/lib/clip-export";
import { useJobs, useClips } from "@/hooks/use-poll";
import { fmtClock, fmtDate, sourceLabel } from "@/lib/clipforge";
import { cn } from "@/lib/utils";
import type { ClipDto } from "@/lib/api";

export default function Dashboard() {
  const { jobs, error: apiError, refresh: refreshJobs } = useJobs(2000);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const { clips } = useClips(selectedJobId, 2000);
  const [previewClip, setPreviewClip] = useState<ClipDto | null>(null);
  const [redetecting, setRedetecting] = useState<string | null>(null);

  const allJobs = jobs ?? [];
  const activeJobs = allJobs.filter(
    (j) => j.stage !== "complete" && j.stage !== "failed",
  );
  const doneJobs = allJobs.filter((j) => j.stage === "complete");
  const failedJobs = allJobs.filter((j) => j.stage === "failed");

  // auto-select the newest completed job (or recover after a delete)
  useEffect(() => {
    if (doneJobs.length === 0) {
      if (selectedJobId !== null && !allJobs.some((j) => j._id === selectedJobId)) {
        setSelectedJobId(null);
      }
      return;
    }
    const stillThere = doneJobs.some((j) => j._id === selectedJobId);
    if (selectedJobId === null || !stillThere) {
      setSelectedJobId(doneJobs[0]._id);
    }
  }, [doneJobs, allJobs, selectedJobId]);

  const selectedJob = doneJobs.find((j) => j._id === selectedJobId) ?? null;

  // keep the preview dialog's clip object fresh as renders complete
  const freshPreview = useMemo(
    () => (previewClip ? clips.find((c) => c._id === previewClip._id) ?? previewClip : null),
    [previewClip, clips],
  );

  const runRedetect = async (jobId: string) => {
    setRedetecting(jobId);
    try {
      await fetch(`/api/jobs/${jobId}/redetect`, { method: "POST" });
      toast.info("Re-detection queued — Gemini is re-scoring this batch.");
      void refreshJobs();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Re-detection failed.");
    } finally {
      setRedetecting(null);
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

      {/* local worker status (CPU/RAM/queue/warmup) */}
      <div className="mx-auto w-full max-w-6xl px-4 pt-6">
        <LocalWorkerCard />
        {apiError && (
          <div className="mt-3 rounded-xl border border-destructive/50 bg-destructive/10 px-4 py-3">
            <p className="text-sm font-semibold text-destructive">
              Cannot reach the local API ({apiError})
            </p>
            <p className="mt-1 text-xs leading-relaxed text-destructive/80">
              Start it with <code className="font-mono">./start.sh</code> — the API and
              the worker are one process, so a running server means a working worker.
            </p>
          </div>
        )}
      </div>

      <div className="mx-auto w-full max-w-6xl px-4 py-8">
        {/* header + new job */}
        <section className="grid gap-6 lg:grid-cols-[1fr_380px]">
          <div>
            <p className="text-sm text-muted-foreground">
              Running 100% locally — your videos never leave this machine
            </p>
            <h1 className="mt-1 text-3xl font-black tracking-tight">
              The forge
            </h1>
            <p className="mt-3 max-w-lg text-sm leading-relaxed text-muted-foreground">
              Paste a video you own, watch it move through the four-stage
              pipeline, and export vertical clips with captions ready to post.
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
                <JobCard key={job._id} job={job} onDeleted={refreshJobs} />
              ))}
            </div>
          </section>
        )}

        {/* completed jobs */}
        <section className="mt-10">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-bold tracking-tight">Your batches</h2>
            {selectedJob && (
              <div className="flex items-center gap-2">
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
                <ExportDialog job={selectedJob} clips={clips} />
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
                  <button
                    key={job._id}
                    type="button"
                    onClick={() => setSelectedJobId(job._id)}
                    className={cn(
                      "cursor-pointer rounded-full border px-3 py-1.5 text-xs font-medium transition",
                      selectedJobId === job._id
                        ? "border-primary/60 bg-primary/10 text-primary"
                        : "border-border text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {sourceLabel(job).slice(0, 36)} · {job.clipCount ?? 0} clips
                  </button>
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
                    {selectedJob.detectedSource === "gemini"
                      ? "AI-detected"
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
                  Copy caption + hashtags
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
