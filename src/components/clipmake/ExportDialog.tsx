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
import { Progress } from "@/components/ui/progress";
import { buildCaptionsTxt } from "@/lib/clip-export";
import { downloadBlob, downloadText } from "@/lib/zip";
import { api } from "@/lib/api";
import type { JobDto, ClipDto } from "@/lib/api";

export function ExportDialog({
  job,
  clips,
  trigger,
}: {
  job: JobDto;
  clips: ClipDto[];
  trigger?: React.ReactNode;
}) {
  const [busy, setBusy] = useState(false);
  const [activeMode, setActiveMode] = useState<"zip" | "captions" | null>(null);
  const [zipProgress, setZipProgress] = useState(0);
  const rendered = clips.filter((c) => c.videoPath !== null);
  const ready = job.stage === "complete" && rendered.length > 0;

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
      // ZIP: the backend zips the REAL rendered MP4s from ./outputs — no
      // metadata-only exports. Clips without a rendered file are skipped.
      setZipProgress(50);
      const blob = await api.exportZip(rendered.map((c) => c._id));
      setZipProgress(100);
      downloadBlob(blob, `clipmake-${job._id}.zip`);
      toast.success(`Export ZIP downloaded — ${rendered.length} real MP4${rendered.length === 1 ? "" : "s"}.`, {
        description: "9:16 1080×1920 · H.264/AAC · captions.txt included",
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
          {busy && activeMode === "zip" && (
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
