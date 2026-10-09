import { useEffect, useMemo, useRef, useState } from "react";
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
import { ShareReadinessSummary } from "@/components/clipforge/ShareButton";
import { ClipPreview } from "@/components/clipforge/ClipPreview";
import { ExportDialog } from "@/components/clipforge/ExportDialog";
import { ConfirmDeleteJob } from "@/components/clipforge/ConfirmDeleteJob";
import { CleanupJunkDialog } from "@/components/clipforge/CleanupJunkDialog";
import { CaptionStyleDialog } from "@/components/clipforge/CaptionStyleDialog";
import { GroqUsageBadge } from "@/components/clipforge/GroqUsageBadge";
import { copyClipMeta, buildClipSrt } from "@/lib/clip-export";
import { useJobs, useClips, useHealth } from "@/hooks/use-poll";
import { api, type UsageDto, type VerifyClipsDto, waitForSandbox } from "@/lib/api";
import { fmtAgo, fmtClock, fmtDate, fmtDurationLong, batchLabel, sourceLabel } from "@/lib/clipforge";
import { cn } from "@/lib/utils";
import type { ClipDto } from "@/lib/api";

/**
 * Placeholder cards in the clip grid's own proportions, shown while the
 * request for the OPENED batch is genuinely in flight.
 *
 * "Clips are loading..." sat on an empty grid with no shape, so the wait read
 * as a frozen page rather than as arriving content. Four skeleton cards in a
 * 9:16 frame say the same thing as four real ones, so nothing jumps when they
 * land. Once the request has answered, an empty grid says what it means.
 *
 * Its own component (rather than an inline ternary) so the grid element is a
 * complete element: tools that read the grid block by balancing tags cannot
 * follow a JSX ternary and would otherwise match half the page.
 */
function ClipGridSkeleton() {
  return (
    <div
      className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4"
      aria-label="Loading clips"
      aria-busy="true"
    >
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="rounded-xl border border-border bg-card">
          <div className="aspect-[9/16] w-full animate-pulse bg-muted/60" />
          <div className="space-y-2 p-3">
            <div className="h-3 w-3/4 animate-pulse rounded bg-muted" />
            <div className="h-2.5 w-1/2 animate-pulse rounded bg-muted" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const {
    jobs,
    error: apiError,
    stale: apiStale,
    refresh: refreshJobs,
    checkOne,
    loadMore,
    hasMore,
  } = useJobs(2000);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [previewClip, setPreviewClip] = useState<ClipDto | null>(null);
  const [redetecting, setRedetecting] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<VerifyClipsDto | null>(null);
  // Fed by the header badge, read by the submit form. One poll, two consumers.
  const [usage, setUsage] = useState<UsageDto | null>(null);
  const [sandboxState, setSandboxState] = useState<"idle" | "starting" | "failed">("idle");
  const [sandboxMessage, setSandboxMessage] = useState<string | null>(null);

  const allJobs = useMemo(() => jobs ?? [], [jobs]);
  // The backend derives `status` in ONE place (app/jobstatus.py) and clears
  // every running-only field on a terminal job, so filtering here cannot put
  // the same job in two lists. Keyed off `isTerminal`/`status` rather than
  // `stage` so a job can never read as both failed and running.
  const activeJobs = allJobs.filter(
    (j) => !j.isTerminal && (j.status === "running" || j.status === "queued"),
  );
  const doneJobs = allJobs.filter((j) => j.status === "complete");
  const failedJobs = allJobs.filter((j) => j.status === "failed");

  // Auto-select the newest completed job without a setState-in-effect:
  // the effective selection is derived, and the user's explicit pick is kept
  // when still valid.
  const effectiveJobId =
    selectedJobId !== null && doneJobs.some((j) => j._id === selectedJobId)
      ? selectedJobId
      : (doneJobs[0]?._id ?? null);
  const selectedJob = doneJobs.find((j) => j._id === effectiveJobId) ?? null;
  const { clips, loaded: clipsLoaded } = useClips(effectiveJobId, 2000);
  // The worker's real silence budget. A stall banner that quotes a number
  // invented in the frontend will drift from the server the moment
  // CLIPFORGE_STALL_TIMEOUT_SEC is set, so the card reads it from health.
  const { health } = useHealth(10_000);

  // "Load older batches" as the reader reaches the end of the pill list. The
  // server already returns only the newest page; without this the older ones
  // were unreachable on a phone with more than one screen of batches.
  const moreRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const node = moreRef.current;
    if (!node || !hasMore) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) loadMore();
      },
      { rootMargin: "200px" },
    );
    io.observe(node);
    return () => io.disconnect();
  }, [hasMore, loadMore]);

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
            {/* Groq's daily token budget is the constraint on AI detection, and
                it is invisible until a job dies part way through a paid
                transcript. Shown up front, with the pre-submit warning below
                fed from the same number. */}
            <div className="hidden w-40 sm:block">
              <GroqUsageBadge onUsage={setUsage} />
            </div>
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
        {/* min-w-0 lets the card shrink to the viewport instead of widening the
            page from inside a grid column; overflow-x-hidden catches anything
            that still insists on being wider (e.g. an error code string) so a
            phone never gets horizontal scroll from a status readout. */}
        <div className="min-w-0 overflow-x-hidden">
          <LocalWorkerCard />
        </div>
        {/* A 4xx is NOT an outage. The server answered and rejected one request,
            so it is shown as a quiet amber note and the job list below stays on
            screen. Only a genuine outage — repeated 5xx or a dead connection
            with nothing to show — earns the red banner. */}
        {apiError && (
          <div className="mt-3 rounded-xl border border-destructive/50 bg-destructive/10 px-4 py-3">
            <p className="text-sm font-semibold text-destructive">
              API unreachable: {apiError}
            </p>
            <p className="mt-1 text-xs leading-relaxed text-destructive/80">
              {/HTTP 5\d\d/.test(apiError)
                ? "The pipeline server answered /api/jobs with a server error three times in a row — the response is logged server-side; check the deployment logs."
                : "The pipeline server did not answer /api/jobs after repeated retries. If this persists, the deployment is restarting or offline."}
            </p>
          </div>
        )}
        {apiStale && !apiError && (
          <div
            role="status"
            className="mt-3 rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3"
          >
            <p className="text-sm font-semibold text-amber-300">
              Could not refresh the job list
            </p>
            <p className="mt-1 text-xs leading-relaxed text-amber-200/80">
              {apiStale}. The pipeline server is reachable — this is a soft
              blip, and the batches below are the last good reading. It will
              recover on the next poll.
            </p>
          </div>
        )}
      </div>

      <div className="mx-auto w-full max-w-6xl px-4 py-8">
        {/* header + new job */}
        <section className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
          <div className="min-w-0">
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
          <div className="min-w-0 rounded-xl border border-primary/30 bg-card p-5 neon-glow-sm">
            <h2 className="flex items-center gap-2 font-bold tracking-tight">
              <Sparkles className="size-4 text-primary" />
              New batch
            </h2>

            {sandboxState !== "idle" && (
              <p
                className={cn(
                  "mt-3 text-[11px] font-medium leading-relaxed",
                  sandboxState === "starting"
                    ? "text-amber-300"
                    : "text-destructive",
                )}
              >
                {sandboxState === "starting"
                  ? sandboxMessage ?? "Sandbox is starting, please wait…"
                  : sandboxMessage ?? "Sandbox did not start."}
              </p>
            )}

            <NewJobForm
              className="mt-4"
              usage={usage}
              onSandboxState={(state, message) => {
                setSandboxState(state === "ready" ? "idle" : state);
                setSandboxMessage(message);
              }}
            />

            <div className="mt-4 rounded-xl border border-border/60 bg-card p-4">
              <p className="text-xs font-medium text-muted-foreground">
                Sandbox status
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                <span
                  className={cn(
                    "rounded-full px-3 py-1 text-[11px] font-medium",
                    sandboxState === "idle"
                      ? "border-border bg-muted text-muted-foreground"
                      : sandboxState === "starting"
                        ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
                        : "border-destructive/50 bg-destructive/15 text-destructive",
                  )}
                >
                  {sandboxState === "idle"
                    ? "Checked"
                    : sandboxState === "starting"
                      ? "Starting"
                      : "Not running"}
                </span>
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
                {sandboxState === "idle"
                  ? "The backend was reachable the last time it was checked."
                  : sandboxState === "starting"
                    ? sandboxMessage ?? "Sandbox is starting, please wait…"
                    : sandboxMessage ?? "The backend did not respond."}
              </p>
              <Button
                size="sm"
                variant="outline"
                className="mt-3 cursor-pointer"
                onClick={async () => {
                  setSandboxState("starting");
                  setSandboxMessage("Sandbox is starting, please wait…");
                  try {
                    const ready = await waitForSandbox({
                      waitForWorker: true,
                      onUpdate: (state, message) => {
                        setSandboxState(state === "ready" ? "idle" : state);
                        setSandboxMessage(message);
                      },
                    });
                    if (!ready.ok) {
                      toast.error("Sandbox still not running.");
                    } else {
                      toast.success("Sandbox is running.");
                    }
                  } catch (err) {
                    toast.error(
                      err instanceof Error ? err.message : "Could not reach the sandbox.",
                    );
                  } finally {
                    setSandboxState("idle");
                    setSandboxMessage(null);
                  }
                }}
              >
                Restart sandbox
              </Button>
            </div>
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
                  stallTimeoutSec={health?.stallTimeoutSec}
                  onDeleted={refreshJobs}
                  onCheckNow={async (jobId) => {
                    // "Check now" forces an immediate poll of THIS job instead
                    // of waiting out the 2s interval. The server reconciles
                    // stale running state on this route, so the answer is the
                    // authoritative status — and the toast says WHICH status,
                    // because a toast reading "updated just now" on a job that
                    // had already failed was exactly the ambiguity reported.
                    const fresh = await checkOne(jobId);
                    if (!fresh) return;
                    const since =
                      typeof fresh.lastProgressAt === "number"
                        ? fmtAgo(Date.now() - fresh.lastProgressAt)
                        : "just now";
                    if (fresh.isTerminal) {
                      if (fresh.status === "complete") {
                        toast.success(`Checked — done, ${fresh.clipCount ?? 0} clips ready.`);
                      } else if (fresh.status === "paused") {
                        toast.info("Checked — this job is paused. Resume to continue.");
                      } else {
                        toast.error(
                          `Checked — this job has failed${
                            fresh.stageLabel ? ` during ${fresh.stageLabel.toLowerCase()}` : ""
                          }. See the error on the card.`,
                        );
                      }
                    } else if (fresh.status === "queued") {
                      toast.info("Checked — still queued, waiting for a worker.");
                    } else {
                      toast.success(
                        `Checked — ${fresh.stageLabel ?? "running"}${fresh.progressDetail ? `: ${fresh.progressDetail}` : ""} · updated ${since}`,
                      );
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
              {/* One-tap cleanup for the batches that produced nothing.
                  Individual trash icons are for one batch; after a run of test
                  uploads this is what actually clears the dashboard. */}
              {allJobs.length > 0 && (
                <CleanupJunkDialog
                  className="cursor-pointer gap-1.5 text-xs"
                  onDeleted={() => {
                    if (selectedJobId !== null) setSelectedJobId(null);
                    void refreshJobs();
                  }}
                />
              )}
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
                      className="block cursor-pointer px-3 py-1.5 text-left"
                    >
                      {/* Title (or platform + video ID), never a bare host. Three
                          batches all reading "youtube.com/watch · 23 clips" were
                          indistinguishable; the date below disambiguates re-runs. */}
                      <span className="block max-w-[15rem] truncate" title={batchLabel(job)}>
                        {batchLabel(job)}
                      </span>
                      <span className="ml-1.5 font-normal text-muted-foreground">
                        {fmtDate(job.createdAt)} · {job.clipCount ?? 0} clips
                      </span>
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
                {/* Scroll sentinel. Older batches are fetched only when the
                    reader actually reaches this point — and the same control
                    is a real button, so it works without JS observers, with a
                    keyboard, and on a screen reader. */}
                {hasMore && (
                  <div ref={moreRef} className="flex basis-full justify-center pt-1">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-8 cursor-pointer text-xs text-muted-foreground"
                      onClick={loadMore}
                    >
                      Load older batches
                    </Button>
                  </div>
                )}
              </div>

              {/* job summary */}
              {selectedJob && (
                <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-1 rounded-xl border border-border bg-card px-4 py-3 text-xs text-muted-foreground">
                  <span className="font-mono">{fmtDate(selectedJob.createdAt)}</span>
                  {selectedJob.durationSec != null && (
                    <span>
                      {fmtClock(selectedJob.durationSec)} source ·{" "}
                      {/* mm:ss, not the raw float. This read
                          "932.8100000000011s of clips". */}
                      {fmtDurationLong(selectedJob.totalClipsSec)} of clips
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
              {/* Says plainly which clips will share as a video and which still
                  share as a link, so the per-card badge is never a mystery. */}
              {clips.length > 0 && (
                <div className="mt-2">
                  <ShareReadinessSummary clips={clips} />
                </div>
              )}
              {clips.length === 0 &&
                (clipsLoaded ? (
                  <p className="mt-3 text-sm text-muted-foreground">
                    This batch has no clips yet.
                  </p>
                ) : (
                  <ClipGridSkeleton />
                ))}

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
