import { toast } from "sonner";
import {
  Copy,
  Download,
  Flame,
  Scissors,
  Thermometer,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ClipPreview } from "./ClipPreview";
import { buildClipSrt, copyClipMeta } from "@/lib/clip-export";
import { scoreBand, fmtDuration } from "@/lib/clipforge";
import { cn } from "@/lib/utils";
import type { ClipDoc } from "@/lib/clipforge";

const BAND_STYLES: Record<string, string> = {
  hot: "border-primary/60 bg-primary/15 text-primary",
  warm: "border-amber-400/50 bg-amber-400/10 text-amber-300",
  mild: "border-border bg-muted text-muted-foreground",
};

const BAND_ICONS: Record<string, typeof Flame> = {
  hot: Flame,
  warm: Thermometer,
  mild: Scissors,
};

export function ClipCard({
  clip,
  onOpen,
}: {
  clip: ClipDoc;
  onOpen?: (clip: ClipDoc) => void;
}) {
  const score = scoreBand(clip.score);
  const BandIcon = BAND_ICONS[score];

  return (
    <div className="group flex flex-col overflow-hidden rounded-xl border border-border bg-card transition hover:border-primary/40">
      <div className="relative p-2">
        <ClipPreview clip={clip} className="w-full" />
        {/* clickable overlay above the control bar opens the detail dialog */}
        <button
          type="button"
          onClick={() => onOpen?.(clip)}
          className="absolute inset-x-2 top-2 bottom-12 cursor-pointer rounded-t-xl"
          aria-label={`Preview ${clip.title}`}
        />
        <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-1">
          <span
            className={cn(
              "flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold",
              BAND_STYLES[score],
            )}
          >
            <BandIcon className="size-3" />
            {clip.score}
          </span>
        </div>
      </div>
      <div className="flex flex-1 flex-col gap-2 px-3 pb-3">
        <div>
          <button
            type="button"
            onClick={() => onOpen?.(clip)}
            className="block w-full cursor-pointer truncate text-left text-sm font-bold tracking-tight hover:text-primary"
            title={clip.title}
          >
            {clip.title}
          </button>
          <p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
            {clip.hookText}
          </p>
        </div>
        <p className="font-mono text-[10px] text-muted-foreground">
          {fmtDuration(clip.start)} → {fmtDuration(clip.end)} ·{" "}
          {fmtDuration(clip.end - clip.start)} · 9:16 1080×1920
        </p>
        <div className="mt-auto flex items-center gap-1.5">
          <Button
            size="sm"
            variant="outline"
            className="h-7 flex-1 cursor-pointer gap-1.5 text-xs"
            onClick={async () => {
              try {
                await copyClipMeta(clip);
                toast.success("Caption + hashtags copied.");
              } catch {
                toast.error("Clipboard unavailable — select the text manually.");
              }
            }}
          >
            <Copy className="size-3.5" />
            Copy caption
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-7 cursor-pointer gap-1.5 text-xs"
            onClick={() => {
              buildClipSrt(clip);
              toast.success("SRT downloaded.");
            }}
          >
            <Download className="size-3.5" />
            .srt
          </Button>
        </div>
      </div>
    </div>
  );
}

/** Re-export for consumers wanting the score badge standalone. */
export function ScoreBadge({ score }: { score: number }) {
  const band = scoreBand(score);
  const Icon = BAND_ICONS[band];
  return (
    <span
      className={cn(
        "flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold",
        BAND_STYLES[band],
      )}
    >
      <Icon className="size-3" />
      {score}
    </span>
  );
}
