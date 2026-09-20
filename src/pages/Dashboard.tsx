import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { useQuery, useAction } from "convex/react";
import { Film, Loader2, LogOut, RefreshCw, Server, ServerOff, Sparkles, Timer } from "lucide-react";
import { useAuth } from "@/hooks/use-auth";
import { api } from "@/convex/_generated/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { LogoWordmark } from "@/components/clipforge/LogoMark";
import { RightsNotice } from "@/components/clipforge/RightsNotice";
import { NewJobForm } from "@/components/clipforge/NewJobForm";
import { JobCard } from "@/components/clipforge/JobCard";
import { ClipCard, ScoreBadge } from "@/components/clipforge/ClipCard";
import { ClipPreview } from "@/components/clipforge/ClipPreview";
import { ExportDialog } from "@/components/clipforge/ExportDialog";
import { copyClipMeta, buildClipSrt } from "@/lib/clip-export";
import {
  fmtClock,
  fmtDate,
  sourceLabel,
  type ClipDoc,
  type JobDoc,
} from "@/lib/clipforge";
import { cn } from "@/lib/utils";

export default function Dashboard() {
  const { user, signOut } = useAuth();
  const jobs = useQuery(api.clips.listJobs, {});
  const clipsByJob = useQuery(api.clips.recentClips, { limit: 48 });
  const worker = useQuery(api.clips.getWorkerStatus, {});

  // Re-evaluate worker liveness between heartbeats (tick every 2s).
  const [nowTs, setNowTs] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNowTs(Date.now()), 2_000);
    return () => clearInterval(t);
  }, []);
  const workerOnline =
    (worker?.online ?? false) && nowTs - (worker?.lastHeartbeat ?? 0) < 45_000;
  // Learned transcribing rate (ms per second of audio) -> "~N min for a 10 min video"
  const rates = worker?.rates as Record<string, number> | undefined;
  const trRate = rates?.transcribing;
  const trHint =
    trRate && trRate > 0
      ? Math.max(1, Math.round((600 * trRate) / 60_000))
      : null;

  const [selectedJobId, setSelectedJobId] = useState<JobDoc["_id"] | null>(null);
  const [previewClip, setPreviewClip] = useState<ClipDoc | null>(null);
  const [redetecting, setRedetecting] = useState<string | null>(null);
  const redetect = useAction(api.detection.redetectClips);

  // Group clips by their job, memoized to avoid re-sorting every render.
  const clipsGrouped = useMemo(() => {
    const map = new Map<string, ClipDoc[]>();
    for (const c of clipsByJob ?? []) {
      const list = map.get(c.jobId) ?? [];
      list.push(c);
      map.set(c.jobId, list);
    }
    for (const list of map.values()) {
      list.sort((a, b) => a.order - b.order);
    }
    return map;
  }, [clipsByJob]);

  const activeJobs = (jobs ?? []).filter(
    (j) => j.stage !== "complete" && j.stage !== "failed",
  );
  const doneJobs = (jobs ?? []).filter((j) => j.stage === "complete");
  const failedJobs = (jobs ?? []).filter((j) => j.stage === "failed");

  // auto-select the newest completed job (or recover after a delete)
  useEffect(() => {
    if (doneJobs.length === 0) return;
    const stillThere = doneJobs.some((j) => j._id === selectedJobId);
    if (selectedJobId === null || !stillThere) {
      setSelectedJobId(doneJobs[0]._id);
    }
  }, [doneJobs, selectedJobId]);

  const selectedJob = doneJobs.find((j) => j._id === selectedJobId) ?? null;
  const selectedClips = selectedJob
    ? clipsGrouped.get(selectedJob._id) ?? []
    : [];

  const runRedetect = async (jobId: JobDoc["_id"]) => {
    setRedetecting(jobId);
    try {
      const res = await redetect({ jobId });
      if (res.clipSource === "ai") {
        toast.success(`AI re-detection complete — ${res.count} clips.`);
      } else {
        toast.info("AI unavailable — used heuristic scoring instead.");
      }
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
          <a href="/dashboard" className="cursor-pointer">
            <LogoWordmark />
          </a>
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm" className="cursor-pointer gap-1.5 text-xs">
              <a href="/">
                Landing
              </a>
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="cursor-pointer gap-1.5 text-xs"
              onClick={async () => {
                await signOut();
              }}
            >
              <LogOut className="size-3.5" />
              Sign out
            </Button>
          </div>
        </div>
      </header>

      {/* worker liveness + warmup (startup self-checks) */}
      <div className="mx-auto w-full max-w-6xl px-4 pt-6">
        {!workerOnline ? (
          <div className="rounded-xl border border-destructive/50 bg-destructive/10 px-4 py-3">
            <p className="flex items-center gap-2 text-sm font-semibold text-destructive">
              <ServerOff className="size-4" />
              Render worker offline
            </p>
            <p className="mt-1 text-xs leading-relaxed text-destructive/80">
              Jobs are refused until a worker is consuming the queue — start it with:{" "}
              <code className="font-mono">celery -A app worker --loglevel=info</code>
            </p>
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-xl border border-border/60 bg-card/60 px-4 py-2.5 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5 font-medium text-primary">
              <Server className="size-3.5" />
              Worker online
            </span>
            {!worker?.modelLoaded ? (
              <span className="flex items-center gap-1.5 text-amber-300">
                <Loader2 className="size-3.5 animate-spin" />
                Downloading AI model (one-time, ~3GB)…
                {worker?.warmupDetail && (
                  <span className="font-mono text-[11px] text-muted-foreground">
                    {worker.warmupDetail}
                  </span>
                )}
              </span>
            ) : (
              trHint !== null && (
                <span className="flex items-center gap-1.5">
                  <Timer className="size-3.5 text-primary/70" />
                  Transcribing usually takes ~{trHint} min for a 10 min video
                </span>
              )
            )}
          </div>
        )}
      </div>

      <div className="mx-auto w-full max-w-6xl px-4 py-8">
        {/* header + new job */}
        <section className="grid gap-6 lg:grid-cols-[1fr_380px]">
          <div>
            <p className="text-sm text-muted-foreground">
              Welcome back{user?.name ? `, ${user.name}` : ""}
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
                <JobCard key={job._id} job={job} />
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
                  onClick={() => runRedetect(selectedJob._id)}
                >
                  {redetecting === selectedJob._id ? (
                    <RefreshCw className="size-3.5 animate-spin" />
                  ) : (
                    <Sparkles className="size-3.5 text-primary" />
                  )}
                  Re-detect with AI
                </Button>
                <ExportDialog job={selectedJob} clips={selectedClips} />
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
                  <span>
                    {fmtClock(selectedJob.durationSec ?? 0)} source ·{" "}
                    {selectedJob.totalClipsSec ?? 0}s of clips
                  </span>
                  <span>
                    {selectedJob.clipSource === "ai"
                      ? "AI-detected"
                      : "Heuristic-detected"}
                  </span>
                  <span>
                    {selectedJob.transcriptWordCount ?? 0} words transcribed
                  </span>
                  <span>via {selectedJob.downloadSource ?? "yt-dlp"}</span>
                </div>
              )}

              {/* clip grid */}
              <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
                {selectedClips.map((clip) => (
                  <ClipCard
                    key={clip._id}
                    clip={clip}
                    onOpen={() => setPreviewClip(clip)}
                  />
                ))}
              </div>
              {selectedClips.length === 0 && (
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
          {previewClip && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2 pr-6">
                  {previewClip.title}
                </DialogTitle>
                <DialogDescription className="flex items-center gap-2">
                  <ScoreBadge score={previewClip.score} />
                  <span className="font-mono text-[10px]">
                    {fmtClock(previewClip.start)} → {fmtClock(previewClip.end)}
                  </span>
                </DialogDescription>
              </DialogHeader>
              <ClipPreview clip={previewClip} autoPlay className="w-full" />
              <p className="max-h-24 overflow-auto rounded-lg border border-border bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
                {previewClip.transcript}
              </p>
              <div className="flex gap-2">
                <Button
                  className="flex-1 cursor-pointer gap-1.5"
                  onClick={async () => {
                    try {
                      await copyClipMeta(previewClip);
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
                  onClick={() => buildClipSrt(previewClip)}
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
