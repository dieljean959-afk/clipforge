// One row under the bucket reading: media that belongs to no batch the site
// can show. Those batches are invisible on the dashboard (their job documents
// are gone), so without this line the only way to get the space back was a
// shell command. It lists nothing until the bucket actually has leftovers.

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Loader2, Trash2, TriangleAlert } from "lucide-react";

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
import { api } from "@/lib/api";
import { fmtBytes } from "@/lib/clipmake";

type UnusedBatch = { jobId: string; bytes: number; objects: number; kinds: string[] };

/**
 * "Unused media · 187.1 MB · Delete".
 *
 * The delete is server-verified: the endpoint re-reads the live job ids before
 * removing anything, so a stale list (a batch restarted in another tab) is
 * skipped instead of destroyed. The confirmation names the batches and how
 * much plan the deletion reclaims, and reports the result honestly — including
 * "already gone", which is not a failure.
 */
export function UnusedMediaRow({ onCleaned }: { onCleaned?: () => void }) {
  const [items, setItems] = useState<UnusedBatch[]>([]);
  const [totalBytes, setTotalBytes] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.unusedStorage();
      setItems(res.items ?? []);
      setTotalBytes(res.totalBytes ?? 0);
      setFailed(false);
    } catch {
      // A bucket that cannot be read is not a crash: the rest of the card is
      // still useful, so this row simply stays out of the way.
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const remove = async () => {
    setBusy(true);
    try {
      const ids = items.map((item) => item.jobId);
      const res = await api.deleteUnusedStorage(ids);
      const freed = fmtBytes(res.bytesFreed ?? 0);
      if (res.deleted.length > 0) {
        toast.success(`Deleted ${res.deleted.length} unused batch(es) — freed ${freed}.`, {
          description: `${res.removed} object(s) removed from the bucket.`,
        });
      } else {
        toast.message("Nothing was deleted", {
          description:
            res.skipped.length > 0
              ? "Those batches are live again, so their media was left alone."
              : "The bucket no longer holds unused media.",
        });
      }
      setOpen(false);
      await load();
      onCleaned?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not delete unused media.");
    } finally {
      setBusy(false);
    }
  };

  // Loading and "no leftovers" both render nothing: this is a repair row, not
  // a permanent statistic, and an empty line under the bucket would read as a
  // broken feature.
  if (loading && items.length === 0) return null;
  if (failed || totalBytes === 0) return null;

  const labels = items.slice(0, 5).map((item) => (
    <span key={item.jobId} className="flex items-baseline justify-between gap-3">
      <span className="truncate font-mono text-xs">{item.jobId.slice(0, 12)}</span>
      <span className="shrink-0 text-xs text-muted-foreground">{fmtBytes(item.bytes)}</span>
    </span>
  ));

  return (
    <AlertDialog open={open} onOpenChange={(next) => (busy ? undefined : setOpen(next))}>
      <span
        className="flex items-center gap-1 text-amber-300"
        title="Media in the bucket whose batch no longer exists. It cannot be listed on the dashboard, but it still counts against the plan."
      >
        <TriangleAlert className="size-3.5 shrink-0" />
        <span>
          Unused media {fmtBytes(totalBytes)}
          {items.length > 0 ? ` · ${items.length} deleted batch${items.length === 1 ? "" : "es"}` : ""}
        </span>
        <AlertDialogTrigger asChild>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-6 cursor-pointer gap-1 px-1.5 text-xs text-amber-300 hover:text-amber-200"
          >
            <Trash2 className="size-3" />
            Delete
          </Button>
        </AlertDialogTrigger>
      </span>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete unused batch media?</AlertDialogTitle>
          <AlertDialogDescription>
            These batches are not on the dashboard any more, but their files still occupy the
            bucket. Deleting them frees {fmtBytes(totalBytes)} of your plan. Clips, thumbnails and
            source files for these batches are removed; a batch that is still live is skipped
            server-side.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="max-h-44 space-y-1 rounded-lg border border-border p-2">
          {labels}
          {items.length > 5 && (
            <p className="px-1 pt-1 text-xs text-muted-foreground">
              and {items.length - 5} more
            </p>
          )}
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={busy} className="cursor-pointer">
            Keep them
          </AlertDialogCancel>
          <AlertDialogAction
            disabled={busy}
            className="cursor-pointer gap-1.5"
            onClick={(event) => {
              event.preventDefault();
              void remove();
            }}
          >
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Trash2 className="size-3.5" />}
            Delete {fmtBytes(totalBytes)}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
