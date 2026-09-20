import { useEffect, useState } from "react";
import { toast } from "sonner";
import { useMutation, useQuery } from "convex/react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Download,
  Loader2,
  RotateCcw,
  Sparkles,
  Terminal,
  Trash2,
  AudioWaveform,
  Clapperboard,
} from "lucide-react";
import { api } from "@/convex/_generated/api";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import {
  STAGES,
  STAGE_LABELS,
  fmtClock,
  fmtEta,
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

/**
 * Global stall watchdog (requirement 3): if the worker has written no progress
 * event for this job in 90s while it is in an active stage, the pipeline is
 * dead — surface it instead of spinning. 2s poll mirrors the progress cadence.
 */
function useStalled(job: JobDoc): boolean {
  const [stalled, setStalled] = useState(false);
  const active =
    job.stage !== "complete" && job.stage !== "failed" && job.stage !== "queued";
  useEffect(() => {
    if (!active) {
      setStalled(false);
      return;
    }
    const check = () => {
      const last = job.lastProgressAt ?? job.stageStartedAt ?? job.createdAt;
      setStalled(Date.now() - last > 90_000);
    };
    check();
    const timer = setInterval(check, 2_000);
    return () => clearInterval(timer);
  }, [active, job.lastProgressAt, job.stageStartedAt, job.createdAt]);
  return stalled;
}

export function JobCard({ job }: { job: JobDoc }) {
  const retry = useMutation(api.clips.retryJob);
  const remove = useMutation(api.clips.deleteJob);
  const failed = job.stage === "failed";
  const complete = job.stage === "complete";
  const activeIdx = STAGES.indexOf(job.stage as (typeof STAGES)[number]);
  const stalled = useStalled(job);
  const [logsOpen, setLogsOpen] = useState(false);
  const logs = useQuery(api.clips.getJobLogs, { jobId: job._id });

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
          <Button
            size="sm"
            variant="ghost"
            className="h-8 cursor-pointer gap-1.5 text-xs text-muted-foreground"
            onClick={() => setLogsOpen(true)}
          >
            <Terminal className="size-3.5" />
            View logs
          </Button>
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

      {/* live progress with units + ETA */}
      {!complete && !failed && (
        <div className="mt-3">
          <div className="mb-1.5 flex items-center justify-between gap-2 text-xs">
            <span className="flex min-w-0 items-center gap-1.5 text-muted-foreground">
              <Loader2 className="size-3 shrink-0 animate-spin text-primary" />
              <span className="truncate">
                {job.progressDetail ?? `${STAGE_LABELS[job.stage as ClipStage]}...`}
              </span>
            </span>
            <span className="shrink-0 font-mono tabular-nums text-muted-foreground">
              {Math.round(job.progress)}%
            </span>
          </div>
          <Progress value={job.progress} className="h-1.5" />
          <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
            <span>
              {job.unitsDone !== undefined && job.unitsTotal !== undefined
                ? `${fmtUnits(job.unitLabel, job.unitsDone)} of ${fmtUnits(job.unitLabel, job.unitsTotal)}`
                : "Waiting for worker…"}
            </span>
            {job.totalEtaSec !== undefined && job.totalEtaSec > 0 && (
              <span className="font-mono tabular-nums text-primary/80">
                ~{fmtEta(job.totalEtaSec)} left
              </span>
            )}
          </div>
          {stalled && (
            <div className="mt-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              <span className="font-semibold">Pipeline stalled</span> — no progress
              from the worker for 90s at stage{" "}
              <span className="font-mono">{job.stage}</span>. Open the logs, or retry
              the job.
            </div>
          )}
        </div>
      )}

      {complete && (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span className="font-semibold text-primary">{job.clipCount ?? 0} clips</span>
          <span>{job.totalClipsSec ?? 0}s total</span>
          <span>{job.clipSource === "ai" ? "AI-detected" : "Heuristic-detected"}</span>
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

      {/* per-job worker log ("View logs") */}
      <Dialog open={logsOpen} onOpenChange={setLogsOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Job log</DialogTitle>
            <DialogDescription className="font-mono text-xs">
              {sourceLabel(job)}
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-80 overflow-y-auto rounded-lg border bg-black/60 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
            {logs === undefined ? (
              <p>Loading…</p>
            ) : logs.length === 0 ? (
              <p>No worker events logged yet.</p>
            ) : (
              logs.map((line, i) => (
                <p key={i} className="whitespace-pre-wrap">
                  <span className="text-primary/60">{String(i + 1).padStart(2, "0")}</span>{" "}
                  {line}
                </p>
              ))
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

type ClipStage = (typeof STAGES)[number];

/** Unit-aware value formatting: "142 MB", "4 min", "chunk 2". */
function fmtUnits(label: string | undefined, value: number): string {
  switch (label) {
    case "MB":
      return `${Math.round(value)} MB`;
    case "min":
      return `${Math.round(value)} min`;
    case "chunks":
      return `${value} chunks`;
    default:
      return String(value);
  }
}
