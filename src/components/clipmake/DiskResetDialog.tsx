import { useState } from "react";
import { toast } from "sonner";
import { Eraser, Loader2, ShieldCheck, TriangleAlert } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

type Plan = Awaited<ReturnType<typeof api.diskReset>>;

/**
 * Two-step hard reset: the first click only loads a dry-run plan, and the
 * destructive call requires a second, explicitly labelled confirmation.
 */
export function DiskResetDialog({ onApplied }: { onApplied?: () => void }) {
  const [open, setOpen] = useState(false);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);

  const loadPlan = async () => {
    setLoading(true);
    setPlan(null);
    try {
      setPlan(await api.diskReset(false));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not inspect disk usage.");
    } finally {
      setLoading(false);
    }
  };

  const openDialog = () => {
    setOpen(true);
    void loadPlan();
  };

  const apply = async () => {
    setApplying(true);
    try {
      const out = await api.diskReset(true);
      toast.success(`Freed ${out.freedMb.toFixed(1)} MB`, {
        description: `Disk now has ${out.freeAfterMb.toFixed(0)} MB free.`,
      });
      setOpen(false);
      setPlan(null);
      onApplied?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Disk reset failed.");
    } finally {
      setApplying(false);
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={openDialog}
        className="flex cursor-pointer items-center gap-1 rounded-full border border-border px-2 py-0.5 text-[10px] font-bold text-muted-foreground transition hover:border-primary/50 hover:text-foreground"
        title="Preview a hard reset of caches, test debris, scratch and bucket-verified media, then confirm"
      >
        <Eraser className="size-3" />
        Reset disk
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Eraser className="size-4 text-primary" />
              Hard reset disk
            </DialogTitle>
            <DialogDescription>
              Regenerable files only. Nothing is deleted until you confirm below.
            </DialogDescription>
          </DialogHeader>

          {loading && (
            <p className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" /> Scanning caches, scratch and media…
            </p>
          )}

          {plan && (
            <div className="grid gap-3 text-sm">
              <div className="rounded-md border border-border bg-card/50 px-3 py-2">
                <div className="font-semibold text-foreground">
                  Would free {plan.plannedMb.toFixed(1)} MB
                </div>
                <div className="text-xs text-muted-foreground">
                  {plan.freeBeforeMb.toFixed(0)} MB free now
                  {plan.backup ? ` · backup ${plan.backup}` : ""}
                </div>
              </div>

              <div className="grid gap-1.5">
                {plan.categories
                  .filter((cat) => cat.mb > 0)
                  .map((cat) => (
                    <div key={cat.name} className="flex items-start justify-between gap-3">
                      <span className="min-w-0">
                        <span className="block text-foreground">{cat.name}</span>
                        <span className="block text-[11px] leading-snug text-muted-foreground">
                          {cat.note} · {cat.count} item{cat.count === 1 ? "" : "s"}
                        </span>
                      </span>
                      <span className="shrink-0 font-mono text-xs text-foreground">
                        {cat.mb.toFixed(1)} MB
                      </span>
                    </div>
                  ))}
              </div>

              {plan.openDeleted.length > 0 && (
                <p className="flex gap-2 rounded-md border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-[11px] text-amber-200">
                  <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
                  Deleted-but-open files are held by{" "}
                  {plan.openDeleted.map((p) => `${p.cmd} (pid ${p.pid}, ${p.mb.toFixed(1)} MB)`).join(", ")}.
                  Restart that process after the reset to release the space.
                </p>
              )}

              {plan.kept.length > 0 && (
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Kept {plan.kept.length} file{plan.kept.length === 1 ? "" : "s"} that could not be
                  verified in the bucket.
                </p>
              )}

              <p className="flex gap-2 text-[11px] leading-snug text-muted-foreground">
                <ShieldCheck className="mt-0.5 size-3.5 shrink-0 text-primary" />
                Protected: keys/.env, cookies.txt, job documents and backups, venv, app code,
                fonts, emoji assets, the active model, and every running batch.
              </p>
            </div>
          )}

          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setOpen(false)} disabled={applying}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => void apply()}
              disabled={!plan || applying || plan.plannedMb <= 0}
            >
              {applying ? <Loader2 className="mr-1.5 size-3.5 animate-spin" /> : null}
              {applying ? "Resetting…" : `Confirm reset (${plan?.plannedMb.toFixed(0) ?? 0} MB)`}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
