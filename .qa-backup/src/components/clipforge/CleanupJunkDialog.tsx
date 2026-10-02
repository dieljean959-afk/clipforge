import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Eraser, Loader2 } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { api, type JunkJobDto } from "@/lib/api";
import { fmtDate } from "@/lib/clipforge";

/** The server caps one bulk request at 200 ids. */
const MAX_PER_REQUEST = 200;

/** Readable name for a batch, without the pipeline's internal id. */
function label(job: { title?: string | null; _id: string }): string {
  const title = (job.title ?? "").trim();
  return title.length > 0 ? title.slice(0, 48) : job._id.slice(0, 8);
}

/**
 * Clear out the batches that produced nothing.
 *
 * Deleting a batch one at a time means one confirmation dialog per batch, and
 * after a run of failed uploads that is a dozen dialogs for a dozen dead test
 * batches — which is how a dashboard ends up with seven "source.mp4 · 2 clips"
 * pills nobody is going to open again. This is the same delete, asked for as a
 * list, confirmed once, with the count spelled out before anything is removed.
 *
 * The list comes from GET /api/jobs/junk — batches that finished (or died)
 * without a single delivered clip. Batches that DID produce clips are never
 * swept up silently: they sit behind one unticked checkbox that names how many
 * they are, because a rendered clip is the one artifact here that cannot be
 * fetched again from a URL that may be gone.
 */
export function CleanupJunkDialog({
  onDeleted,
  className,
}: {
  onDeleted?: () => void;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [junk, setJunk] = useState<JunkJobDto[]>([]);
  const [alsoWorking, setAlsoWorking] = useState(false);
  const [working, setWorking] = useState<JunkJobDto[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await api.listJunkJobs();
      setJunk(res.jobs ?? []);
    } catch {
      // The server is the authority on what is junk (it knows what was
      // actually rendered), but a backend that has not picked up that route
      // yet must not make cleanup impossible. Deriving the same list from the
      // batch summaries keeps the button working in the meantime: a failed
      // batch, or a finished one that delivered nothing, is junk either way.
      try {
        const all = await api.listAllJobs();
        setJunk(
          (all.jobs ?? [])
            .filter(
              (job) =>
                job.status === "failed" ||
                (job.status === "complete" && (job.clipCount ?? 0) === 0),
            )
            .map((job) => ({
              _id: job._id,
              title: (job.title ?? job.source ?? job._id).trim(),
              stage: job.stage,
              createdAt: job.createdAt,
              reason:
                job.status === "failed"
                  ? (job.error ?? job.failedStage ?? "failed").slice(0, 48)
                  : "no clips rendered",
            })),
        );
      } catch (err) {
        setJunk([]);
        setLoadError(
          err instanceof Error ? err.message : "Could not read the batch list.",
        );
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetched when the dialog OPENS, not on mount: this is a cleanup action, and
  // nobody should pay for its request on every dashboard poll.
  useEffect(() => {
    if (!open) return;
    void load();
  }, [open, load]);

  /**
   * The batches that DID produce clips are not junk, so they are never swept up
   * silently — but a run of test uploads leaves plenty of those too, and the
   * reader is the only one who knows which are worth keeping. Ticking the box
   * loads the full history (the dashboard only holds a page) and names them.
   */
  const loadWorking = useCallback(async (junkIds: string[]) => {
    try {
      const res = await api.listAllJobs();
      setWorking(
        (res.jobs ?? [])
          .filter((job) => !junkIds.includes(job._id))
          .map((job) => ({
            _id: job._id,
            title: (job.title ?? job.source ?? job._id).trim(),
            stage: job.stage,
            createdAt: job.createdAt,
            reason: `${job.clipCount ?? 0} clip${job.clipCount === 1 ? "" : "s"}`,
          })),
      );
    } catch {
      // The checkbox stays honest: without the list we cannot claim to know
      // what else would go, so the option simply stays unavailable.
      setWorking(null);
      toast.error("Could not read the rest of the batches.");
    }
  }, []);

  const targets = alsoWorking && working ? [...junk, ...working] : junk;

  const remove = async () => {
    setBusy(true);
    try {
      let removed = 0;
      const missing: string[] = [];
      // Chunked so a long history cannot exceed the server's per-request cap
      // and fail the whole cleanup on one oversized body.
      for (let i = 0; i < targets.length; i += MAX_PER_REQUEST) {
        const chunk = targets.slice(i, i + MAX_PER_REQUEST);
        const res = await api.bulkDeleteJobs(chunk.map((job) => job._id));
        removed += res.deleted.length;
        missing.push(...res.missing);
      }
      setOpen(false);
      setJunk([]);
      setAlsoWorking(false);
      setWorking(null);
      if (removed > 0) {
        toast.success(
          `Deleted ${removed} batch${removed === 1 ? "" : "es"} and everything they used on disk.`,
        );
      }
      if (missing.length > 0) {
        toast.warning(`${missing.length} batch(es) were already gone.`);
      }
      onDeleted?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Cleanup failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (busy) return;
        setOpen(next);
        if (!next) {
          // Reopening always starts from a fresh reading of the server, never
          // from what was on screen when the dialog was last closed.
          setJunk([]);
          setAlsoWorking(false);
          setWorking(null);
          setLoadError(null);
        }
      }}
    >
      <AlertDialogTrigger asChild>
        <Button
          size="sm"
          variant="outline"
          className={className}
          title="Delete the batches that produced no clips"
        >
          <Eraser className="size-3.5" />
          Clean up
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Clean up dead batches</AlertDialogTitle>
          <AlertDialogDescription>
            {loading ? (
              "Checking which batches produced nothing…"
            ) : loadError ? (
              loadError
            ) : junk.length === 0 ? (
              "Every batch here produced at least one clip. Nothing to clean up."
            ) : (
              <>
                {junk.length} batch{junk.length === 1 ? "" : "es"} produced no clips. Deleting
                them removes the source video, transcript and any half-rendered output.
              </>
            )}
          </AlertDialogDescription>
        </AlertDialogHeader>

        {junk.length > 0 && (
          <div className="max-h-56 space-y-1 overflow-y-auto rounded-lg border border-border p-2">
            {junk.slice(0, 40).map((job) => (
              <div
                key={job._id}
                className="flex items-baseline justify-between gap-3 px-1 py-0.5 text-xs"
              >
                <span className="truncate font-medium">{label(job)}</span>
                <span className="shrink-0 text-muted-foreground">
                  {job.createdAt ? fmtDate(job.createdAt) : ""} ·{" "}
                  {(job.reason ?? "").slice(0, 48) || "no clips"}
                </span>
              </div>
            ))}
            {junk.length > 40 && (
              <p className="px-1 pt-1 text-xs text-muted-foreground">
                and {junk.length - 40} more
              </p>
            )}
            {alsoWorking && working && (
              <p className="px-1 pt-2 text-xs font-medium text-destructive">
                + {working.length} batch{working.length === 1 ? "" : "es"} that did produce clips
              </p>
            )}
          </div>
        )}

        {junk.length > 0 && (
          <div className="flex items-center gap-2">
            <Checkbox
              id="cleanup-also-working"
              checked={alsoWorking}
              onCheckedChange={(v) => {
                const next = v === true;
                setAlsoWorking(next);
                if (next && working === null) void loadWorking(junk.map((job) => job._id));
              }}
            />
            <Label htmlFor="cleanup-also-working" className="text-xs text-muted-foreground">
              Also delete the batches that did produce clips
            </Label>
          </div>
        )}

        <AlertDialogFooter>
          <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
          {junk.length > 0 && (
            <AlertDialogAction
              disabled={busy}
              onClick={(e) => {
                // Keep the dialog mounted while the delete is in flight.
                e.preventDefault();
                void remove();
              }}
              className="bg-destructive text-white hover:bg-destructive/90"
            >
              {busy ? <Loader2 className="size-3.5 animate-spin" /> : null}
              {busy
                ? "Deleting…"
                : alsoWorking && working
                  ? `Delete all ${targets.length} batches`
                  : `Delete ${junk.length} dead batch${junk.length === 1 ? "" : "es"}`}
            </AlertDialogAction>
          )}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
