import { useState } from "react";
import { toast } from "sonner";
import { useQueries } from "convex/react";
import { FileArchive, FileText, Loader2, Package } from "lucide-react";
import { api } from "@/convex/_generated/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { buildExportZip, buildCaptionsTxt } from "@/lib/clip-export";
import { downloadBlob, downloadText } from "@/lib/zip";
import type { JobDoc, ClipDoc } from "@/lib/clipforge";

export function ExportDialog({
  job,
  clips,
  trigger,
}: {
  job: JobDoc;
  clips: ClipDoc[];
  trigger?: React.ReactNode;
}) {
  const [busy, setBusy] = useState(false);
  const [activeMode, setActiveMode] = useState<"zip" | "captions" | null>(null);
  const [zipProgress, setZipProgress] = useState(0);
  const rendered = clips.filter((c) => c.mediaStorageId !== undefined);
  const ready = job.stage === "complete" && rendered.length > 0;

  // One hook call for all rendered clips' signed URLs (dynamic list — useQueries).
  const mediaUrls = useQueries(
    Object.fromEntries(
      rendered.map((c) => [
        c._id,
        {
          query: api.clips.getStorageUrl,
          args: { storageId: c.mediaStorageId! },
        } as const,
      ]),
    ),
  );

  const doExport = async (mode: "zip" | "captions") => {
    if (!ready) return;
    setBusy(true);
    setActiveMode(mode);
    setZipProgress(0);
    try {
      if (mode === "captions") {
        downloadText(buildCaptionsTxt(job, clips), "captions.txt");
        toast.success("captions.txt downloaded.");
        return;
      }
      // ZIP: fetch every rendered clip's REAL bytes from storage — no metadata
      // exports. Clips without a rendered file are skipped and reported.
      const media: { clip: ClipDoc; bytes: Uint8Array }[] = [];
      for (let i = 0; i < rendered.length; i++) {
        const clip = rendered[i];
        if (clip.mediaStorageId === undefined) continue;
        const url = mediaUrls[clip._id];
        if (typeof url !== "string") {
          throw new Error(`storage URL unavailable for clip ${clip.order + 1}`);
        }
        media.push({ clip, bytes: await fetchClipBytes(url) });
        setZipProgress(Math.round(((i + 1) / rendered.length) * 100));
      }
      const { blob, included, skipped } = buildExportZip(job, clips, media);
      downloadBlob(blob, `clipforge-${job._id.slice(-8)}.zip`);
      toast.success(`Export ZIP downloaded — ${included} real MP4${included === 1 ? "" : "s"}.`, {
        description:
          skipped > 0
            ? `${skipped} clip${skipped === 1 ? " was" : "s were"} skipped (no rendered file) — nothing was faked.`
            : "9:16 1080×1920 · H.264/AAC",
      });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Export failed.");
    } finally {
      setBusy(false);
      setActiveMode(null);
    }
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        {trigger ?? (
          <Button size="sm" disabled={!ready} className="cursor-pointer gap-2">
            <Package className="size-4" />
            Export all 9:16
          </Button>
        )}
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Export batch</DialogTitle>
          <DialogDescription>
            {rendered.length} rendered clip{rendered.length === 1 ? "" : "s"} from{" "}
            <span className="font-mono text-xs">{job.source.slice(0, 48)}</span>
            {rendered.length < clips.length && (
              <span className="mt-1 block text-amber-300">
                {clips.length - rendered.length} clip
                {clips.length - rendered.length === 1 ? " has" : "s have"} no rendered
                file — skipped honestly, not faked.
              </span>
            )}
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          {busy && activeMode === "zip" && zipProgress > 0 && (
            <Progress value={zipProgress} className="h-1.5" />
          )}
          <button
            type="button"
            onClick={() => void doExport("zip")}
            disabled={busy}
            className="flex cursor-pointer items-center gap-3 rounded-lg border border-border px-4 py-3 text-left transition hover:border-primary/50 disabled:opacity-50"
          >
            {busy && activeMode === "zip" ? (
              <Loader2 className="size-5 animate-spin text-primary" />
            ) : (
              <FileArchive className="size-5 text-primary" />
            )}
            <span>
              <span className="block text-sm font-semibold">Export all 9:16 (ZIP)</span>
              <span className="text-xs text-muted-foreground">
                Real MP4s + captions.txt + README manifest
              </span>
            </span>
          </button>
          <button
            type="button"
            onClick={() => void doExport("captions")}
            disabled={busy}
            className="flex cursor-pointer items-center gap-3 rounded-lg border border-border px-4 py-3 text-left transition hover:border-primary/50 disabled:opacity-50"
          >
            {busy && activeMode === "captions" ? (
              <Loader2 className="size-5 animate-spin text-primary" />
            ) : (
              <FileText className="size-5 text-primary" />
            )}
            <span>
              <span className="block text-sm font-semibold">captions.txt only</span>
              <span className="text-xs text-muted-foreground">
                Every clip's SRT cues in one text file
              </span>
            </span>
          </button>
        </div>
        <DialogFooter>
          <p className="text-xs text-muted-foreground">
            Only clip content you own or have permission to use.
          </p>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** Fetch a clip's rendered MP4 bytes from its signed storage URL. */
async function fetchClipBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`storage fetch failed (HTTP ${res.status}) — clip file missing?`);
  }
  return new Uint8Array(await res.arrayBuffer());
}
