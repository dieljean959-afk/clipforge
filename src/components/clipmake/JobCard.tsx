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
import { ConfirmDeleteJob } from "@/components/clipmake/ConfirmDeleteJob";
import { cn } from "@/lib/utils";
import {
  STAGES,
  STAGE_LABELS,
  QUIET_THRESHOLD_MS,
  FALLBACK_STALL_TIMEOUT_SEC,
  STALL_WARN_FRACTION,
  fmtAgo,
  fmtClock,
  fmtDate,
  fmtDurationLong,
  fmtEta,
  batchLabel,
  overallStatus,
  silenceMs,
  sourceLabel,
  type ClipStage,
  type JobDoc,
} from "@/lib/clipmake";

const STAGE_ICONS: Record<string, typeof Clock> = {
  queued: Clock,
  grabbing: Download,
  transcribing: AudioWaveform,
  finding: Sparkles,
  rendering: Clapperboard,
  complete: CheckCircle2,
  failed: AlertTriangle,
};

/** The stall budget is the SERVER's (app/worker.py STALL_TIMEOUT_SEC), read
 * from health and threaded in as `stallTimeoutSec`. These shared constants
 * from @/lib/clipmake only cover the cold first paint, before /api/health has
 * answered.
 *
 * This card used to hold its own hardcoded 90s threshold and then describe it
 * to the user as "the server restarts and resumes after 1.5 minutes" — while
 * the server's real budget was 180s. So the UI promised a recovery half a
 * minute before the server would act, and the number on screen was not the
 * number the server was running. One source, not two.
 */

/** Render a duration the way a person would say it. The old copy interpolated
 * seconds/60000 raw, so a 90s budget printed "1.5 minutes" and a 180s budget
 * printed "3 minutes" — the first reading as a fraction of a minute is what
 * made the number look like a bug rather than a budget.
 */
function fmtMinutes(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} second${s === 1 ? "" : "s"}`;
  const m = Math.round(s / 60);
  return `${m} minute${m === 1 ? "" : "s"}`;
}

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

function useStalled(job: JobDoc, thresholdMs: number): boolean {
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
      if (typeof last !== "number") return;
      setStalled(Date.now() - last > thresholdMs);
    };
    // first evaluation deferred a tick — no synchronous setState on mount
    const initial = setTimeout(check, 0);
    const timer = setInterval(check, 2_000);
    return () => {
      clearTimeout(initial);
      clearInterval(timer);
    };
    // thresholdMs is a dependency: when health arrives carrying a different
    // real budget, the verdict is re-evaluated against it instead of keeping
    // the fallback's answer until some unrelated re-render happens.
  }, [active, thresholdMs]);
  return active && stalled;
}

export function JobCard({
  job,
  onDeleted,
  onCheckNow,
  stallTimeoutSec,
}: {
  job: JobDoc;
  onDeleted?: () => void;
  onCheckNow?: (jobId: string) => Promise<void> | void;
  /** The server's real silence budget, from health. Falls back to the
   *  worker's own default until health answers. */
  stallTimeoutSec?: number | null;
}) {
  const serverStallSec =
    typeof stallTimeoutSec === "number" && stallTimeoutSec > 0
      ? stallTimeoutSec
      : FALLBACK_STALL_TIMEOUT_SEC;
  const failed = job.status === "failed" || job.stage === "failed";
  const complete = job.status === "complete" || job.stage === "complete";
  const paused = job.status === "paused" || job.stage === "paused";
  const overall = overallStatus(job);
  const isTerminal = job.isTerminal || failed || complete || paused || overall.isTerminal;
  // The stage to highlight: for a live job that is its active stage, and for a
  // dead one the stage it died in, so the stepper marks the right pill rather
  // than leaving every stage grey.
  const stepStage = (job.activeStage ?? job.failedStage ?? job.stage) as
    | (typeof STAGES)[number]
    | undefined;
  const activeIdx = stepStage ? STAGES.indexOf(stepStage) : -1;
  const stalled = useStalled(job, serverStallSec * 1000 * STALL_WARN_FRACTION);
  const [busy, setBusy] = useState<null | "pause" | "resume" | "check" | "retry">(null);
  // Live ONLY while the backend says the job is live. isTerminal is the
  // backend's single verdict, so this can never show a spinner for a job that
  // has already stopped.
  const watching = !isTerminal && !complete && !failed && !paused;
  const now = useNow(watching);
  const quiet = watching && silenceMs(job, now) > QUIET_THRESHOLD_MS;
  const plan = job.stagePlan ?? [];

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

  const retryJob = async () => {
    setBusy("retry");
    try {
      await api.retryJob(job._id);
      toast.info("Running this batch again — finished stages are cached.");
      // Ask for this job's fresh status now instead of waiting out the poll,
      // so the card flips back to "running" the moment the server accepts it.
      await onCheckNow?.(job._id);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Retry failed.");
    } finally {
      setBusy(null);
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
          {/* The title identifies the batch; the URL is the fallback and the
              tooltip. Three batches from three videos all read
              "youtube.com/watch", so the host alone was not a label. */}
          <p className="truncate text-sm font-semibold" title={job.source}>
            {batchLabel(job)}
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
            <ConfirmDeleteJob job={job} onDeleted={() => onDeleted?.()} />
          )}
          {/* A failed job ALWAYS carries an `error` string, so the old
              `job.error === undefined` guard hid this button on exactly the
              jobs that needed it. Show it whenever the job is dead. */}
          {failed ? (
            <Button
              size="sm"
              variant="outline"
              className="h-8 cursor-pointer gap-1.5 text-xs"
              disabled={busy === "retry"}
              onClick={() => void retryJob()}
            >
              {busy === "retry" ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <RotateCcw className="size-3.5" />
              )}
              Run again
            </Button>
          ) : null}
        </div>
      </div>          {/* whole-pipeline status: the single answer to "running / done / broken".
              "done" is for a job whose every promised unit of work has checked out.
              A job can be "complete" (the backend's terminal status) while still
              looking unfinished to a reader when the list only hands back a stale
              snapshot, so this card names the real situation rather than borrowing
              a status word that means something else in another part of the code. */}
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
          <span className="shrink-0 rounded-full bg-background px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
            finished
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
              {/* Clamped: this text comes off the job record, and a derived
                  field must never be able to grow without bound in the card. */}
              {(job.interruptReason ??
                "The worker was stopped externally, not by a pipeline error. Completed stages are reused, so progress is not lost."
              ).slice(0, 220)}
            </div>
          </div>
        </div>
      ) : overall.state === "failed" ? (
        <div
          data-testid="job-failed"
          className="mt-3"
        >
          <FailureNotice job={job} />
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

      {/* live progress with units + ETA. Gated on `watching` — the backend's
          single verdict — rather than on stage equality, so a job the server
          considers terminal can never render a spinner and a countdown. */}
      {watching && (
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
                : job.status === "queued" && job.queuePosition != null
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

          {/* Per-stage plan: which stage this is, and what is left after it.
              A single "whole job ~34 min" number moves for reasons the user
              cannot see — it is the sum of a 20-minute transcription and a
              3-minute render, and it barely moves while the transcription
              runs. This shows the split, and labels each row by where its
              number came from so a prior is never read as a measurement. */}
          {plan.length > 0 && (
            <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
              {plan.map((entry) => (
                <li
                  key={entry.stage}
                  className={cn(
                    "flex items-center gap-1.5 tabular-nums",
                    entry.current
                      ? "font-medium text-primary"
                      : "text-muted-foreground/80",
                  )}
                  title={
                    entry.source === "live"
                      ? "Live estimate from this stage's own progress"
                      : entry.source === "measured"
                        ? "How long this stage actually took on this job"
                        : "Estimated from the source length and this host's speed"
                  }
                >
                  {entry.current ? (
                    <Loader2 className="size-2.5 shrink-0 animate-spin" />
                  ) : (
                    <span className="size-1 shrink-0 rounded-full bg-current opacity-50" />
                  )}
                  {entry.label}
                  <span className="opacity-70">~{fmtEta(entry.sec)}</span>
                </li>
              ))}
            </ul>
          )}
          {quiet && !stalled && (
            /* Name the real state instead of guessing. "One long unit of work"
               was wrong: with the chunk reporter every transcription window
               updates this field, and the render stage reports per clip, so
               whatever is on screen is the exact thing the worker is doing. */
            <div className="mt-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200/90">
              No progress reported for {fmtAgo(silenceMs(job, now))}. The worker is
              still at{" "}
              <span className="font-mono">{job.stage}</span>
              {typeof job.unitsDone === "number" && job.unitsDone > 0
                ? ` — ${job.unitsDone} of ${job.unitsTotal ?? "?"} ${job.unitLabel ?? "units"} finished`
                : ""}
              , last reporting:{" "}
              <span className="font-semibold">
                {job.progressDetail ?? STAGE_LABELS[job.stage as ClipStage]}
              </span>
              . The server restarts and resumes this job after{" "}
              {fmtMinutes(serverStallSec)} of silence, picking up from the
              last finished unit.
            </div>
          )}
          {stalled && (
            <div className="mt-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              <span className="font-semibold">Pipeline stalled</span> — no progress
              from the worker for {fmtMinutes(serverStallSec * STALL_WARN_FRACTION)} at stage{" "}
              <span className="font-mono">{job.stage}</span>
              {typeof job.unitsDone === "number" && job.unitsDone > 0
                ? `, stuck after ${job.unitsDone} of ${job.unitsTotal ?? "?"} ${job.unitLabel ?? "units"}`
                : ""}
              . Last reported:{" "}
              <span className="font-semibold">
                {job.progressDetail ?? STAGE_LABELS[job.stage as ClipStage]}
              </span>
              . The server kills the worker and retries the job automatically once,
              resuming from the last finished unit — finished clips are kept.
            </div>
          )}
        </div>
      )}

      {/* How the batch size was decided. "You asked for 10 and got 10" and
          "the detector found 61 moments, 23 of them were highlights, and the
          length-based ceiling was 23" are different stories, and only the
          second one tells the user whether to change anything. */}
      {complete && typeof job.candidatesDetected === "number" && (
        <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
          {job.candidatesDetected} candidate highlight
          {job.candidatesDetected === 1 ? "" : "s"} found ·{" "}
          {job.clipsAboveScore ?? job.candidatesDetected} scored above{" "}
          {Math.round(job.clipScoreFloor ?? 0)} ·{" "}
          <span className="text-foreground">
            {job.clipCount ?? 0} rendered
          </span>
          {job.clipTarget ? ` (ceiling ${job.clipTarget})` : ""}
          {job.clipTargetReason ? ` — ${job.clipTargetReason}` : ""}
          {typeof job.clipsBelowScore === "number" && job.clipsBelowScore > 0
            ? `. ${job.clipsBelowScore} weaker moment${job.clipsBelowScore === 1 ? "" : "s"} dropped rather than padded in.`
            : "."}
        </p>
      )}

      {/* Manual control: pause/resume a live job, and force a poll so activity
          can be confirmed without reloading the page. A job that the backend has
          already marked terminal does not keep buttons on screen for work it can
          no longer do. */}
      {(overall.isTerminal ? paused : !complete && !failed && !paused) && (
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
            <span>{fmtDurationLong(job.totalClipsSec)} total</span>
          )}
          <span>
            {job.detectedSource ? `AI-detected via ${job.detectedSource}` : "Detection failed"}
          </span>
          {job.downloadSource && <span>via {job.downloadSource}</span>}
          {job.transcriptWordCount !== undefined && job.transcriptWordCount !== null && (
            <span>{job.transcriptWordCount} words transcribed</span>
          )}
          {job.transcriptLabel && (
            <span
              className="rounded-full border border-amber-400/40 bg-amber-400/10 px-2 py-0.5 text-[10px] font-semibold text-amber-300"
              title={
                job.asrPeakRssMb
                  ? `Peak ASR memory ${Math.round(job.asrPeakRssMb)} MB`
                  : "Decoded with a lighter setting to fit this machine's memory"
              }
            >
              {job.transcriptLabel}
            </span>
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

/** A failure, in words a person can act on.
 *
 * The raw Python traceback used to be rendered inline. On a phone it is a wall
 * of "/home/daytona/.local/share/uv/python/…" that pushed the card past the
 * viewport and forced horizontal scrolling, and it buried the one sentence that
 * says what happened. So: a short headline derived from what the pipeline
 * knows, the full text and traceback behind a collapsed "Show details", and
 * wrapping everywhere so nothing can widen the page.
 */
function FailureNotice({ job }: { job: JobDoc }) {
  const where = job.stageLabel?.toLowerCase();
  const headline = friendlyError(job.error, where);
  const isPersistentServerRestartFailure = Boolean(
    job.error?.match(/the server restarted \d+ times/i),
  );
  const retryNote = (() => {
    if (!isPersistentServerRestartFailure || !job.error) return null;
    if (job.error.includes(headline)) return null;
    return (
      <p className="mt-1 break-words text-xs text-destructive/90 [overflow-wrap:anywhere]">
        Finished stages are cached. Retry starts a new run from the last
        checkpoint; the same render stage will be retried.
      </p>
    );
  })();
  return (
    <div
      data-testid="job-failed-detail"
      className="mt-3 min-w-0 overflow-hidden rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2"
    >
      <p className="break-words text-sm font-medium text-destructive [overflow-wrap:anywhere]">
        {headline}
      </p>
      {retryNote}
      {job.error && job.error !== headline && (
        <p className="mt-1 break-words text-xs text-destructive/80 [overflow-wrap:anywhere]">
          {job.error}
        </p>
      )}
      {job.errorDetail && (
        <details className="mt-1.5 cursor-pointer">
          <summary className="cursor-pointer text-xs font-medium text-destructive underline underline-offset-2 list-none">
            Show details
          </summary>
          <pre className="mt-1.5 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded bg-black/40 p-2 font-mono text-[10px] leading-relaxed text-destructive/70 [overflow-wrap:anywhere]">
            {job.errorDetail}
          </pre>
        </details>
      )}
    </div>
  );
}

/** One short sentence for a failure, derived from the message rather than
 * shown raw. Detection is the case that mattered: the backend said
 * "AI detection failed on chunk 3/7" and the raw body underneath was a
 * `json_validate_failed` envelope, which reads as gibberish to anyone who
 * did not write the parser.
 */
function friendlyError(error: string | null | undefined, where?: string | null): string {
  const raw = String(error ?? "").trim();
  if (!raw) return "This job failed.";

  // The exact failure you keep seeing: the server restarted several times
  // during one stage and the job stopped making progress between those
  // restarts, so the backend marked it unrecoverable rather than pretending
  // a retry would resurrect it. We render that message as the headline
  // already, but the surrounding card still read like a live stall warning.
  // Normalize the card text to the same sentence so the job does not look
  // like two different problems at once.
  if (/the server restarted \d+ times/i.test(raw)) {
    return raw;
  }

  // Stage truth BEFORE any chunk parser: a transcription failure names a
  // chunk too ("chunk 2/10"), and the detection rules below would otherwise
  // report it as "AI detection failed", which is the wrong stage entirely.
  if (where && where.startsWith("transcrib")) {
    const head = (raw.split("; tried")[0] ?? raw).trim();
    const sentence = head.replace(/^./, (c) => c.toUpperCase());
    if (/out of memory|mkl_malloc|failed to allocate|memoryerror/i.test(raw)) {
      return `${sentence} — finished chunks are saved, so Retry resumes from the last one.`;
    }
    return sentence.length > 240 ? `${sentence.slice(0, 239)}…` : sentence;
  }

  // Chunk-level detection failures already name the chunk; lead with that.
  // Two orders reach us: the per-chunk skip logged during the run
  // ("chunk 3/7 failed") and the total-failure message from detect_clips
  // ("all 1/7 chunk(s) failed"), where the count PRECEDES the noun. Matching
  // only one order meant the total-failure case fell through to the raw text.
  const chunkFirst = /chunk\s+(\d+)\s*\/\s*(\d+)/i.exec(raw);
  const countFirst = /(\d+)\s*\/\s*(\d+)\s+chunk\(s\)\s+failed/i.exec(raw);
  if (chunkFirst) {
    return `AI detection failed on chunk ${chunkFirst[1]}/${chunkFirst[2]}.`;
  }
  if (countFirst) {
    return `AI detection failed on ${countFirst[1]} of ${countFirst[2]} chunks. ` +
      `The rest of the video was detected normally — hit Re-detect to retry just those.`;
  }
  if (/chain failed to return valid json/i.test(raw) || /json_validate_failed/i.test(raw)) {
    return "The AI model did not return usable JSON for this video. Retrying usually helps.";
  }
  if (/rate limit|daily quota|tokens\/day/i.test(raw)) {
    return "The AI provider's daily token quota is used up. Detection will work again when it resets.";
  }
  // One rule for every hosted provider: the message is the same whichever key
  // is missing, so list the providers rather than growing a branch each time
  // the primary changes.
  if (/(DEEPSEEK|NVIDIA|GROQ|GEMINI)_API_KEY missing/i.test(raw)) {
    return "No AI provider key is configured. Add it in the Keys/API keys tab.";
  }
  if (/no longer available|decommissioned/i.test(raw)) {
    return "The configured AI model is no longer available. Pick a current one in config.toml.";
  }
  if (/no active worker|never reported back/i.test(raw)) {
    // The backend no longer emits this string — reconciliation re-queues the
    // job and names the real cause instead. Kept only so an old stored record
    // still renders as something truthful rather than as a mystery.
    return "The server restarted during this job. Finished stages are cached — Retry resumes from the last checkpoint.";
  }
  // Name the unit, not just the stage: "Clip 7 failed, retrying 2/3" tells the
  // user exactly how far the batch got and what is being done about it.
  const clipRetry = /clip\s+(\d+)(?:\s+of\s+(\d+))?[^.]*\b(retrying|retry)\b\s*(\d+\/\d+)/i.exec(raw);
  if (clipRetry) {
    return `Clip ${clipRetry[1]}${clipRetry[2] ? ` of ${clipRetry[2]}` : ""} failed — retrying ${clipRetry[4]}.`;
  }
  const clipSkip = /clip\s+(\d+)(?:\s+of\s+(\d+))?[^.]*\b(skipped|skipped after)/i.exec(raw);
  if (clipSkip) {
    return `Clip ${clipSkip[1]}${clipSkip[2] ? ` of ${clipSkip[2]}` : ""} could not be rendered and was skipped. Every other clip is ready.`;
  }
  if (/not enough free disk/i.test(raw)) {
    return raw;
  }
  if (where && raw.toLowerCase().includes(where)) {
    // Already names the stage; leave the operator's own text alone.
    return raw;
  }
  if (raw.length > 240) {
    // Too long to be a headline and not a case we recognise. Keep the first
    // sentence, which is where these messages put the reason.
    const first = raw.split(/(?<=[.!?])\s/)[0] ?? raw;
    return first.length > 200 ? `${first.slice(0, 199)}…` : first;
  }
  return raw;
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
