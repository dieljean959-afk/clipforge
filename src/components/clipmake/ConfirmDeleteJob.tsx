import { useState } from "react";
import { toast } from "sonner";
import { Loader2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { api, type JobDto } from "@/lib/api";
import { sourceLabel } from "@/lib/clipmake";
import { cn } from "@/lib/utils";

/**
 * Delete a batch, with a confirmation the user has to answer.
 *
 * There is exactly ONE delete path in this app and it lives here, because
 * "permanently deletes a batch and all of its clips" is the kind of action
 * that must not exist twice with two different levels of caution. The pill
 * next to a batch name and the Delete button on the job card are the same
 * component with a different size.
 *
 * What the server does on Yes (app/main.py delete_job, in this order):
 *   1. cancels the job if it is queued or running, so no child process is
 *      still working on files that are about to disappear;
 *   2. writes a tombstone, so a late write from a process we could not stop
 *      cannot resurrect the job;
 *   3. unlinks every clip's MP4 and thumbnail, its clip record, the job
 *      record, and the per-stage workspace;
 *   4. touches NOTHING belonging to any other job — deletion is keyed on this
 *      one job id and its own clips.
 */
export function ConfirmDeleteJob({
  job,
  onDeleted,
  iconOnly = false,
  className,
}: {
  job: Pick<JobDto, "_id" | "source" | "isUpload" | "clipCount" | "title">;
  onDeleted?: (jobId: string) => void;
  /** True for the small trash icon rendered inside a batch pill. */
  iconOnly?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const clips = job.clipCount ?? 0;

  const remove = async () => {
    setBusy(true);
    try {
      await api.deleteJob(job._id);
      toast.success(
        clips > 0
          ? `Deleted “${sourceLabel(job).slice(0, 32)}” and its ${clips} clip${clips === 1 ? "" : "s"}.`
          : "Batch deleted.",
      );
      setOpen(false);
      onDeleted?.(job._id);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Delete failed.";
      // A retry that answers 404 means the FIRST attempt actually worked and
      // the retry was the thing that found nothing there. Reporting that as a
      // failure would leave the user tapping a delete that already succeeded.
      if (/not found|404/i.test(msg)) {
        setOpen(false);
        onDeleted?.(job._id);
        toast.success("Batch deleted.");
        return;
      }
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      {iconOnly ? (
        <button
          type="button"
          aria-label={`Delete batch ${sourceLabel(job)}`}
          title="Delete this batch"
          // stopPropagation matters: this icon lives INSIDE the pill that
          // selects the batch, and selecting it while deleting it would be a
          // nasty surprise.
          onClick={(e) => {
            e.stopPropagation();
            setOpen(true);
          }}
          className={cn(
            "ml-1.5 inline-flex size-5 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition hover:bg-destructive/20 hover:text-destructive",
            className,
          )}
        >
          <Trash2 className="size-3" />
        </button>
      ) : (
        <Button
          size="sm"
          variant="ghost"
          className={cn(
            "h-8 cursor-pointer gap-1.5 text-xs text-muted-foreground hover:text-destructive",
            className,
          )}
          onClick={() => setOpen(true)}
        >
          <Trash2 className="size-3.5" />
          Delete
        </Button>
      )}

      <AlertDialog open={open} onOpenChange={(next) => !busy && setOpen(next)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this batch?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently deletes
              {clips > 0 ? (
                <>
                  {" "}
                  <strong>
                    {sourceLabel(job).slice(0, 40)} and all {clips} of its clips
                  </strong>
                </>
              ) : (
                <> this batch</>
              )}
              . This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={busy}
              onClick={(e) => {
                // Keep the dialog mounted while the request is in flight, so a
                // slow delete does not look like it was ignored.
                e.preventDefault();
                void remove();
              }}
              className="bg-destructive text-white hover:bg-destructive/90"
            >
              {busy ? <Loader2 className="size-3.5 animate-spin" /> : null}
              Yes, delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
