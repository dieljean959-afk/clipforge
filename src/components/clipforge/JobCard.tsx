import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  AlertTriangle,
  AudioWaveform,
  CheckCircle2,
  Clock,
  Download,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  RefreshCw,
  Sparkles,
  Terminal,
  Trash2,
  Clapperboard,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  STAGES,
  STAGE_LABELS,
  QUIET_THRESHOLD_MS,
  fmtAgo,
  fmtClock,
  fmtDate,
  fmtEta,
  isActiveJob,
  overallStatus,
  silenceMs,
  sourceLabel,
  type ClipStage,
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
 * Global stall watchdog: mirror the server's silence watchdog so the UI does
 * not claim a long transcription is dead while it is progressing — and, just
 * as importantly, does NOT stay silent for twice as long as the server will
 * actually wait. The server's budget is 10 minutes (CLIPFORGE_STALL_TIMEOUT_SEC);
 * the UI used to hold off for 20, so a job the server had already killed and
 * auto-retried still showed a spinner with no explanation.
 * 2s poll mirrors the progress cadence.
 */
const STALL_THRESHOLD_MS = 600_000;

/** Re-render once a second so "12s ago" actually counts up on its own. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}

function useStalled(job: JobDoc): boolean {
  const [stalled, setStalled] = useState(false);
  const active =
    job.stage !== "complete" && job.stage !== "failed" && job.stage !== "queued";
  const jobRef = useRef(job);
  useEffect(() => {
    jobRef.current = job;
  }, [job]);
  useEffect(() => {
    if (!active) return;
    const check = () => {
      const j = jobRef.current;
      const last = j.lastProgressAt ?? j.stageStartedAt ?? j.createdAt;
      setStalled(Date.now() - last > STALL_THRESHOLD_MS);
    };
    // first evaluation deferred a tick — no synchronous setState on mount
    const initial = setTimeout(check, 0);
    const timer = setInterval(check, 2_000);
    return () => {
      clearTimeout(initial);
      clearInterval(timer);
    };
  }, [active]);
  return active && stalled;
}

export function JobCard({
  job,
  onDeleted,
  onCheckNow,
}: {
  job: JobDoc;
  onDeleted?: () => void;
  onCheckNow?: (jobId: string) => Promise<void> | void;
}) {
  const failed = job.stage === "failed";
  const complete = job.stage === "complete";
  const paused = job.stage === "paused";
  const activeIdx = STAGES.indexOf(job.stage as (typeof STAGES)[number]);
  const overall = overallStatus(job);
  const stalled = useStalled(job);
  const [busy, setBusy] = useState<null | "pause" | "resume" | "check">(null);
  const watching = isActiveJob(job);
  const now = useNow(watching);
  const quiet = watching && silenceMs(job, now) > QUIET_THRESHOLD_MS;

  const doPause = async () => {
    setBusy("pause");
    try {
      const r = await api.pauseJob(job._id);
      toast(
        r.state === "paused"
          ? "Paused. Nothing was mid-write."
          : "Pausing — stopping at the next safe checkpoint.",
        { description: "Completed chunks and clips are kept; Resume picks up from here." },
      );
    } catch (e) {
      toast.error(`Could not pause: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally {
      setBusy(null);
    }
  };

  const doResume = async () => {
    setBusy("resume");
    try {
      await api.resumeJob(job._id);
      toast.success("Resuming from the last completed checkpoint.", {
        description: "Finished chunks and clips are not redone.",
      });
    } catch (e) {
      toast.error(`Could not resume: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally {
      setBusy(null);
    }
  };

  const doCheck = async () => {
    setBusy("check");
    try {
      await onCheckNow?.(job._id);
    } finally {
      setBusy(null);
    }
  };
  const [logsOpen, setLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);

  const openLogs = async () => {
    setLogsOpen(true);
    try {
      setLogs(await api.jobLogs(job._id));
    } catch {
      setLogs(["(could not load logs)"]);
    }
  };

  const removeJob = async () => {
    try {
      await api.deleteJob(job._id);
      toast.success("Job deleted.");
      onDeleted?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Delete failed.");
    }
  };

  const retryJob = async () => {
    try {
      await api.retryJob(job._id);
      toast.info("Retrying — Clippy is grabbing the video...");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Retry failed.");
    }
  };

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
            {job.durationSec !== undefined && job.durationSec !== null && ` · ${fmtClock(job.durationSec)} source`}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <Button
            size="sm"
            variant="ghost"
            className="h-8 cursor-pointer gap-1.5 text-xs text-muted-foreground"
            onClick={() => void openLogs()}
          >
            <Terminal className="size-3.5" />
            View logs
          </Button>
          {complete && (
            <Button
              size="sm"
              variant="ghost"
              className="h-8 cursor-pointer gap-1.5 text-xs text-muted-foreground hover:text-destructive"
              onClick={() => void removeJob()}
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
              onClick={() => void retryJob()}
            >
              <RotateCcw className="size-3.5" />
              Retry
            </Button>
          )}
        </div>
      </div>

      {/* whole-pipeline status: the single answer to "running / done / broken" */}
      {overall.state === "done" ? (
        <div
          data-testid="job-done"
          className="mt-3 flex items-center gap-2 rounded-lg border border-primary/50 bg-primary/10 px-3 py-2.5 text-sm font-semibold text-primary"
        >
          <CheckCircle2 className="size-4 shrink-0" />
          <span>
            Done — {job.clipCount ?? 0} clip{(job.clipCount ?? 0) === 1 ? "" : "s"} ready
          </span>
          <span className="ml-auto font-mono text-[11px] font-normal opacity-80">
            {job.totalClipsSec ? `· ${fmtClock(job.totalClipsSec)} total` : null}
          </span>
        </div>
      ) : job.interrupted ? (
        <div
          data-testid="job-interrupted"
          className="mt-3 flex items-start gap-2 rounded-lg border border-amber-400/50 bg-amber-400/10 px-3 py-2.5 text-sm"
        >
          <RotateCcw className="mt-0.5 size-4 shrink-0 text-amber-300" />
          <div className="min-w-0">
            <div className="font-semibold text-amber-300">
              Interrupted by a host restart — resuming
              {job.interruptCount ? ` (attempt ${job.interruptCount})` : ""}
            </div>
            <div className="mt-0.5 text-xs text-muted-foreground">
              {job.interruptReason ??
                "The worker was stopped externally, not by a pipeline error. Completed stages are reused, so progress is not lost."}
            </div>
          </div>
        </div>
      ) : overall.state === "failed" ? (
        <div
          data-testid="job-failed"
          className="mt-3 flex items-start gap-2 rounded-lg border border-destructive/50 bg-destructive/10 px-3 py-2.5 text-sm"
        >
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
          <div className="min-w-0">
            <div className="font-semibold text-destructive">{overall.label}</div>
            {job.error && (
              <div className="mt-0.5 truncate text-xs text-muted-foreground" title={job.error}>
                {job.error}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="mt-3" data-testid="job-overall">
          <div className="mb-1.5 flex items-center justify-between gap-2 text-xs">
            <span className="flex min-w-0 items-center gap-1.5 text-foreground">
              {overall.state === "running" ? (
                <Loader2 className="size-3 shrink-0 animate-spin text-primary" />
              ) : (
                <Clock className="size-3 shrink-0 text-muted-foreground" />
              )}
              <span className="truncate">{overall.label}</span>
            </span>
            <span className="shrink-0 font-mono tabular-nums text-muted-foreground">
              {overall.percent}% overall
            </span>
          </div>
          <Progress value={overall.percent} className="h-1.5" />
        </div>
      )}

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
            {/* Liveness: a ticking "last updated Ns ago" so it is possible to
                tell "stuck" from "just slow" without opening the logs. It
                moves on its own every second, so a frozen bar is obvious. */}
            <span
              className="flex shrink-0 items-center gap-1 text-muted-foreground tabular-nums"
              title="Time since the worker last reported progress"
            >
              updated {fmtAgo(silenceMs(job, now))}
            </span>
            <span className="flex shrink-0 items-center gap-2 font-mono tabular-nums text-muted-foreground">
              {/* live ETA — measured once progress exists, RTF-prior before that */}
              {typeof job.stageEtaSec === "number" && (
                <span
                  className="text-primary/80"
                  title={
                    job.etaSource === "measured"
                      ? "Estimated from measured progress"
                      : "Estimated from this host's average transcription speed (RTF)"
                  }
                >
                  ~{fmtEta(job.stageEtaSec)} left
                </span>
              )}
              {Math.round(job.progress)}%
            </span>
          </div>
          <Progress value={job.progress} className="h-1.5" />
          <div className="mt-1 flex flex-wrap items-center justify-between gap-x-3 text-[11px] text-muted-foreground">
            <span>
              {job.unitsDone !== undefined && job.unitsDone !== null &&
              job.unitsTotal !== undefined && job.unitsTotal !== null
                ? `${fmtUnits(job.unitLabel ?? undefined, job.unitsDone)} of ${fmtUnits(job.unitLabel ?? undefined, job.unitsTotal)}`
                : job.queuePosition != null
                  ? queuedLabel(job.queuePosition)
                  : "Starting on the server worker…"}
            </span>
            {typeof job.totalEtaSec === "number" && (
              <span>
                whole job ~{fmtEta(job.totalEtaSec)}
                {job.etaSource !== "measured" && " (estimate)"}
              </span>
            )}
            {/* A queued job's own ETA is meaningless (it hasn't started), so
                the card used to fall through to "~any moment" even with two
                jobs in front of it. Prefer the server's queue estimate, and
                only say "any moment" when there is genuinely nothing ahead. */}
            {job.stage === "queued" && (
              <span className="text-primary/80">
                {typeof job.queueEtaSec === "number" && job.queueEtaSec > 0
                  ? `starts in ~${fmtEta(job.queueEtaSec)}`
                  : "next up — no wait"}
              </span>
            )}
          </div>
          {quiet && !stalled && (
            <div className="mt-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200/90">
              No recent activity ({fmtAgo(silenceMs(job, now))}). The job may just be
              inside one long unit of work — a single encode or transcription
              window produces no progress lines for a while. Nothing is wrong
              yet: the server kills and auto-retries a job only after{" "}
              {STALL_THRESHOLD_MS / 60000} minutes of silence.
            </div>
          )}
          {stalled && (
            <div className="mt-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              <span className="font-semibold">Pipeline stalled</span> — no progress
              from the worker for {STALL_THRESHOLD_MS / 60000} minutes at stage{" "}
              <span className="font-mono">{job.stage}</span>. Long videos can legitimately
              spend many minutes inside one transcription window — check the logs for
              the exact stage. If it stays stuck, the server kills the worker and
              retries the job automatically once, resuming from the last completed
              clip.
            </div>
          )}
        </div>
      )}

      {/* Manual control: pause/resume a live job, and force a poll so activity
          can be confirmed without reloading the page. */}
      {(watching || paused) && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {paused ? (
            <Button size="sm" onClick={doResume} disabled={busy !== null}>
              {busy === "resume" ? (
                <Loader2 className="mr-1.5 size-3 animate-spin" />
              ) : (
                <Play className="mr-1.5 size-3" />
              )}
              Resume
            </Button>
          ) : (
            <Button
              size="sm"
              variant="outline"
              onClick={doPause}
              disabled={busy !== null || job.pauseRequested}
            >
              {busy === "pause" || job.pauseRequested ? (
                <Loader2 className="mr-1.5 size-3 animate-spin" />
              ) : (
                <Pause className="mr-1.5 size-3" />
              )}
              {job.pauseRequested ? "Stopping…" : "Pause"}
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={doCheck} disabled={busy !== null}>
            {busy === "check" ? (
              <Loader2 className="mr-1.5 size-3 animate-spin" />
            ) : (
              <RefreshCw className="mr-1.5 size-3" />
            )}
            Check now
          </Button>
          {paused && job.pausedStage && (
            <span className="text-xs text-muted-foreground">
              stopped at <span className="font-mono">{job.pausedStage}</span>
            </span>
          )}
        </div>
      )}

      {complete && (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span className="font-semibold text-primary">{job.clipCount ?? 0} clips</span>
          {job.totalClipsSec !== undefined && job.totalClipsSec !== null && (
            <span>{job.totalClipsSec}s total</span>
          )}
          <span>
            {job.detectedSource ? `AI-detected via ${job.detectedSource}` : "Detection failed"}
          </span>
          {job.downloadSource && <span>via {job.downloadSource}</span>}
          {job.transcriptWordCount !== undefined && job.transcriptWordCount !== null && (
            <span>{job.transcriptWordCount} words transcribed</span>
          )}
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
            {logs.length === 0 ? (
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

/** "Waiting behind 2 other jobs" — a backed-up queue is not a stuck job. */
function queuedLabel(position: number): string {
  if (position <= 0) return "Next up — starting on the server worker…";
  return `Waiting behind ${position} other job${position > 1 ? "s" : ""}…`;
}

/** Unit-aware value formatting: "142 MB", "4 min", "chunk 2". */
function fmtUnits(label: string | undefined, value: number): string {
  switch (label) {
    case "MB":
      return `${Math.round(value)} MB`;
    case "min":
      return `${Math.round(value)} min`;
    case "clips":
      return `${Math.round(value)} clips`;
    default:
      return String(value);
  }
}
