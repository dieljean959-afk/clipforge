import { useState } from "react";
import { toast } from "sonner";
import { FileArchive, FileText, Loader2, Package } from "lucide-react";
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
  const ready = job.stage === "complete" && clips.length > 0;

  const doExport = (mode: "zip" | "captions") => {
    if (!ready) return;
    setBusy(true);
    try {
      if (mode === "zip") {
        downloadBlob(buildExportZip(job, clips), `clipforge-${job._id.slice(-8)}.zip`);
        toast.success("Export ZIP downloaded.", {
          description: `${clips.length} clips · 9:16 1080×1920`,
        });
      } else {
        downloadText(buildCaptionsTxt(job, clips), "captions.txt");
        toast.success("captions.txt downloaded.");
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Export failed.");
    } finally {
      setBusy(false);
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
            {clips.length} rendered clips from{" "}
            <span className="font-mono text-xs">{job.source.slice(0, 48)}</span>
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          <button
            type="button"
            onClick={() => doExport("zip")}
            disabled={busy}
            className="flex cursor-pointer items-center gap-3 rounded-lg border border-border px-4 py-3 text-left transition hover:border-primary/50 disabled:opacity-50"
          >
            {busy ? (
              <Loader2 className="size-5 animate-spin text-primary" />
            ) : (
              <FileArchive className="size-5 text-primary" />
            )}
            <span>
              <span className="block text-sm font-semibold">Export all 9:16 (ZIP)</span>
              <span className="text-xs text-muted-foreground">
                Per-clip .srt sidecars, captions.txt, README manifest
              </span>
            </span>
          </button>
          <button
            type="button"
            onClick={() => doExport("captions")}
            disabled={busy}
            className="flex cursor-pointer items-center gap-3 rounded-lg border border-border px-4 py-3 text-left transition hover:border-primary/50 disabled:opacity-50"
          >
            <FileText className="size-5 text-primary" />
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
