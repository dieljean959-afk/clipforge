import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { useMutation } from "convex/react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Download,
  Loader2,
  RotateCcw,
  Sparkles,
  Trash2,
  AudioWaveform,
  Clapperboard,
} from "lucide-react";
import { api } from "@/convex/_generated/api";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import {
  STAGES,
  STAGE_LABELS,
  STAGE_LOADING,
  fmtClock,
  fmtDate,
  sourceLabel,
  type JobDoc,
} from "@/lib/clipforge";

const STAGE_ICONS: Record<string, typeof Clock> = {
  queued: Clock,
  grabbing: Download,
  transcribing: AudioWaveform,
  finding: Sparkles,
  rendering: Clapperboard,
  complete: CheckCircle2,
  failed: AlertTriangle,
};

/** Drives an active job by ticking the pipeline ~every 800ms. */
function useJobTicker(job: JobDoc) {
  const tick = useMutation(api.clips.tickJob);
  const active =
    job.stage !== "complete" && job.stage !== "failed" && job.stage !== "queued";
  const stageRef = useRef(job.stage);
  stageRef.current = job.stage;

  useEffect(() => {
    if (!active) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const run = async () => {
      try {
        await tick({ jobId: job._id });
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Pipeline step failed.");
        stopped = true;
      }
      if (!stopped && stageRef.current !== "complete" && stageRef.current !== "failed") {
        timer = setTimeout(run, 800);
        return;
      }
    };
    timer = setTimeout(run, 200);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [active, job._id, tick]);
}

export function JobCard({ job }: { job: JobDoc }) {
  useJobTicker(job);
  const retry = useMutation(api.clips.retryJob);
  const remove = useMutation(api.clips.deleteJob);
  const failed = job.stage === "failed";
  const complete = job.stage === "complete";
  const activeIdx = STAGES.indexOf(job.stage as (typeof STAGES)[number]);

  return (
    <div
      className={cn(
        "rounded-xl border bg-card p-4",
        failed ? "border-destructive/50" : "border-border",
        complete && "border-primary/30",
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold" title={job.source}>
            {sourceLabel(job)}
          </p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {job.isUpload ? "Uploaded file" : "Video URL"} · {fmtDate(job.createdAt)}
            {job.durationSec !== undefined && ` · ${fmtClock(job.durationSec)} source`}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          {complete && (
            <Button
              size="sm"
              variant="ghost"
              className="h-8 cursor-pointer gap-1.5 text-xs text-muted-foreground hover:text-destructive"
              onClick={async () => {
                try {
                  await remove({ jobId: job._id });
                  toast.success("Job deleted.");
                } catch (err) {
                  toast.error(err instanceof Error ? err.message : "Delete failed.");
                }
              }}
            >
              <Trash2 className="size-3.5" />
              Delete
            </Button>
          )}
          {failed && (
            <Button
              size="sm"
              variant="outline"
              className="h-8 cursor-pointer gap-1.5 text-xs"
              onClick={async () => {
                try {
                  await retry({ jobId: job._id });
                  toast.info("Retrying — Clippy is grabbing the video...");
                } catch (err) {
                  toast.error(err instanceof Error ? err.message : "Retry failed.");
                }
              }}
            >
              <RotateCcw className="size-3.5" />
              Retry
            </Button>
          )}
        </div>
      </div>

      {/* stage stepper */}
      <div className="mt-3 flex items-center gap-1">
        {STAGES.slice(1, 5).map((s, i) => {
          const idx = i + 1; // index within STAGES
          const Icon = STAGE_ICONS[s];
          const done = complete || (activeIdx > idx && activeIdx !== -1);
          const current = !complete && !failed && idx === activeIdx;
          return (
            <div key={s} className="flex flex-1 items-center gap-1">
              <div
                className={cn(
                  "flex items-center gap-1.5 rounded-full border px-2 py-1 text-[10px] font-medium",
                  done && "border-primary/40 bg-primary/10 text-primary",
                  current && !done && "border-border bg-muted text-foreground",
                  !done && !current && "border-border/50 text-muted-foreground",
                )}
              >
                {current ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <Icon className="size-3" />
                )}
                {STAGE_LABELS[s]}
              </div>
              {i < 3 && <div className="h-px flex-1 bg-border" />}
            </div>
          );
        })}
      </div>

      {/* progress */}
      {!complete && !failed && (
        <div className="mt-3">
          <div className="mb-1.5 flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              <Loader2 className="size-3 animate-spin text-primary" />
              {job.progressDetail ?? STAGE_LOADING[job.stage] ?? "Working..."}
            </span>
            <span className="font-mono text-muted-foreground">
              {Math.round(job.progress)}%
            </span>
          </div>
          <Progress value={job.progress} className="h-1.5" />
        </div>
      )}

      {complete && (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span className="font-semibold text-primary">
            {job.clipCount ?? 0} clips
          </span>
          <span>{job.totalClipsSec ?? 0}s total</span>
          <span>
            {job.clipSource === "ai" ? "AI-detected" : "Heuristic-detected"}
          </span>
          {job.downloadSource && <span>via {job.downloadSource}</span>}
          <span>{job.transcriptWordCount ?? 0} words transcribed</span>
        </div>
      )}

      {failed && (
        <div className="mt-3 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2">
          <p className="text-sm font-medium text-destructive">{job.error}</p>
          {job.errorDetail && (
            <p className="mt-0.5 font-mono text-[10px] text-destructive/70">
              {job.errorDetail}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
