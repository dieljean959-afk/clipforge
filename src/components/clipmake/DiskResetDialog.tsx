import { useState } from "react";
import { toast } from "sonner";
import { Database, Eraser, HardDrive, Loader2, ShieldCheck, TriangleAlert } from "lucide-react";
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
import { cn } from "@/lib/utils";

type Scope = "local" | "bucket" | "all";
type Plan = Awaited<ReturnType<typeof api.diskReset>>;
type Category = NonNullable<Plan["categories"]>[number];

const SCOPES: Array<{ id: Scope; label: string; hint: string; icon: typeof HardDrive }> = [
  {
    id: "local",
    label: "Local disk",
    hint: "Caches, test debris, scratch, bucket-verified media",
    icon: HardDrive,
  },
  {
    id: "bucket",
    label: "Bucket",
    hint: "Orphans, expired clips, finished sources, multipart uploads",
    icon: Database,
  },
  { id: "all", label: "Both", hint: "Local disk then bucket", icon: Eraser },
];

/**
 * One Reset control, two scopes. Every scope loads a dry run first; the
 * destructive call requires a second, explicitly labelled confirmation.
 * The old "Free disk space" button is gone — it called a no-op route.
 */
export function DiskResetDialog({ onApplied }: { onApplied?: (scope: Scope) => void }) {
  const [open, setOpen] = useState(false);
  const [scope, setScope] = useState<Scope>("local");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);

  const loadPlan = async (next: Scope) => {
    setLoading(true);
    setPlan(null);
    try {
      setPlan(await api.diskReset(false, next));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not inspect usage.");
    } finally {
      setLoading(false);
    }
  };

  const openDialog = () => {
    setOpen(true);
    void loadPlan(scope);
  };

  const choose = (next: Scope) => {
    setScope(next);
    void loadPlan(next);
  };

  const apply = async () => {
    setApplying(true);
    try {
      const out = await api.diskReset(true, scope);
      // A bucket-only result is the root object; `out.bucket` is then just the
      // bucket NAME. Only "all" nests the two results. Reading `out.bucket`
      // as the result showed "Freed 0.0 MB bucket" for every real bucket reset.
      const local =
        scope === "bucket"
          ? null
          : ((scope === "local" ? out : out.local) as Plan | undefined) ?? null;
      const bucket =
        scope === "local"
          ? null
          : ((scope === "bucket" ? out : out.bucket) as Plan | undefined) ?? null;
      const localMb = Number(local?.freedMb ?? 0);
      const bucketMb = Number(bucket?.freedMb ?? out.freedMb ?? 0);
      toast.success(
        scope === "bucket"
          ? `Freed ${bucketMb.toFixed(1)} MB bucket`
          : scope === "local"
            ? `Freed ${localMb.toFixed(1)} MB local`
            : `Freed ${localMb.toFixed(1)} MB local, ${bucketMb.toFixed(1)} MB bucket`,
        {
          description:
            scope === "local"
              ? `Disk now has ${Number(out.freeAfterMb ?? 0).toFixed(0)} MB free.`
              : `${Number(bucket?.usageBeforeMb ?? 0).toFixed(1)} → ${Number(bucket?.usageAfterMb ?? 0).toFixed(1)} MB used · ${bucket?.deletedObjects ?? 0} object(s), ${bucket?.abortedUploads ?? 0} multipart upload(s) removed.`,
        },
      );
      setOpen(false);
      setPlan(null);
      onApplied?.(scope);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Reset failed.");
    } finally {
      setApplying(false);
    }
  };

  const catMb = (cat: Category) => Number(cat.mb ?? 0);
  const plannedLocal =
    scope === "local" || scope === "all" ? Number(plan?.plannedMb ?? 0) : 0;
  const ban = plan?.bucket && typeof plan.bucket === "object" ? plan.bucket : null;
  const plannedBucket =
    scope === "bucket" || scope === "all"
      ? Number((scope === "bucket" ? plan?.plannedMb : ban?.plannedMb) ?? 0)
      : 0;

  return (
    <>
      <button
        type="button"
        onClick={openDialog}
        className="flex cursor-pointer items-center gap-1 rounded-full border border-border px-2 py-0.5 text-[10px] font-bold text-muted-foreground transition hover:border-primary/50 hover:text-foreground"
        title="Preview and confirm a scoped reset of the local disk and/or the bucket"
      >
        <Eraser className="size-3" />
        Reset
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Eraser className="size-4 text-primary" />
              Reset
            </DialogTitle>
            <DialogDescription>
              Dry run first. Nothing is deleted until you confirm.
            </DialogDescription>
          </DialogHeader>

          <div className="grid grid-cols-3 gap-1.5">
            {SCOPES.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => choose(item.id)}
                  className={cn(
                    "flex flex-col items-start gap-1 rounded-lg border px-2.5 py-2 text-left transition",
                    scope === item.id
                      ? "border-primary/60 bg-primary/10 text-foreground"
                      : "border-border bg-card/40 text-muted-foreground hover:border-primary/40",
                  )}
                >
                  <span className="flex items-center gap-1.5 text-xs font-semibold">
                    <Icon className="size-3.5" />
                    {item.label}
                  </span>
                  <span className="text-[10px] leading-snug">{item.hint}</span>
                </button>
              );
            })}
          </div>

          {loading && (
            <p className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" /> Scanning…
            </p>
          )}

          {plan && !loading && (
            <div className="grid gap-3 text-sm">
              <div className="rounded-md border border-border bg-card/50 px-3 py-2">
                <div className="font-semibold text-foreground">
                  {scope === "bucket"
                    ? `Would free ${plannedBucket.toFixed(1)} MB bucket`
                    : `Would free ${plannedLocal.toFixed(1)} MB local${plannedBucket ? ` + ${plannedBucket.toFixed(1)} MB bucket` : ""}`}
                </div>
                <div className="text-xs text-muted-foreground">
                  {scope === "bucket" ? (
                    <>
                      {Number(plan.usageBeforeMb ?? 0).toFixed(1)} MB used now ·{" "}
                      {plan.objects ?? 0} object(s) · {plan.multipartCount ?? 0} multipart upload(s)
                    </>
                  ) : (
                    <>
                      {Number(plan.freeBeforeMb ?? 0).toFixed(0)} MB free now
                      {plan.backup ? ` · backup ${plan.backup}` : ""}
                    </>
                  )}
                </div>
              </div>

              <div className="grid gap-1.5">
                {(plan.categories ?? [])
                  .filter((cat) => catMb(cat) > 0)
                  .map((cat) => (
                    <div key={cat.name} className="flex items-start justify-between gap-3">
                      <span className="min-w-0">
                        <span className="block text-foreground">{cat.name}</span>
                        <span className="block text-[11px] leading-snug text-muted-foreground">
                          {cat.note ?? cat.reason} · {cat.count} item{cat.count === 1 ? "" : "s"}
                        </span>
                      </span>
                      <span className="shrink-0 font-mono text-xs text-foreground">
                        {catMb(cat).toFixed(1)} MB
                      </span>
                    </div>
                  ))}
              </div>

              {(plan.multipart?.length ?? 0) > 0 && (
                <div className="rounded-md border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-[11px] text-amber-200">
                  <p className="flex gap-2 font-semibold">
                    <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
                    {plan.multipart?.length} incomplete multipart upload(s)
                  </p>
                  <p className="mt-1 break-all font-mono text-[10px] opacity-80">
                    {plan.multipart?.slice(0, 3).map((up) => up.key).join(", ")}
                  </p>
                </div>
              )}

              {(plan.openDeleted?.length ?? 0) > 0 && (
                <p className="flex gap-2 rounded-md border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-[11px] text-amber-200">
                  <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
                  Deleted-but-open files are held by{" "}
                  {plan.openDeleted?.map((p) => `${p.cmd} (pid ${p.pid}, ${p.mb.toFixed(1)} MB)`).join(", ")}.
                  The worker will restart after the reset to release them.
                </p>
              )}

              {(plan.kept?.length ?? 0) > 0 && (
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Kept {plan.kept?.length} live or unverified item{(plan.kept?.length ?? 0) === 1 ? "" : "s"}.
                </p>
              )}

              {(plan.failures?.length ?? 0) > 0 && (
                <p className="break-words text-[11px] leading-snug text-destructive">
                  {plan.failures?.join(" · ")}
                </p>
              )}

              <p className="flex gap-2 text-[11px] leading-snug text-muted-foreground">
                <ShieldCheck className="mt-0.5 size-3.5 shrink-0 text-primary" />
                Protected: keys/.env, cookies.txt, DB and backups, venv, app code, fonts,
                emoji assets, the active model, every running batch, and every bucket
                object a live batch references. The DB is backed up before bucket deletes.
              </p>
            </div>
          )}

          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setOpen(false)} disabled={applying}>
              Cancel
            </Button>
            <Button onClick={() => void apply()} disabled={applying || loading || !plan}>
              {applying ? <Loader2 className="mr-1 size-4 animate-spin" /> : null}
              {applying ? "Resetting…" : `Confirm ${scope === "all" ? "reset" : `${scope} reset`}`}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
