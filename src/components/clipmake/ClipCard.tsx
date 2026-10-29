import { useState } from "react";
import { toast } from "sonner";
import {
  Copy,
  Download,
  EarOff,
  Flame,
  Loader2,
  RefreshCw,
  Scissors,
  Thermometer,
  TriangleAlert,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { ClipPreview } from "./ClipPreview";
import { PostMenu } from "./PostMenu";
import { ShareButton, XQuickButton } from "./ShareButton";
import { buildClipSrt, copyClipMeta, downloadClipMp4 } from "@/lib/clip-export";
import { scoreBand, fmtDuration } from "@/lib/clipmake";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import type { ClipDto } from "@/lib/api";

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
  clip: ClipDto;
  onOpen?: (clip: ClipDto) => void;
}) {
  const score = scoreBand(clip.score);
  const BandIcon = BAND_ICONS[score];
  const [downloading, setDownloading] = useState(false);

  const rendered = clip.renderState === "done" && clip.videoPath !== null;
  const failed = clip.renderState === "failed";
  const playable = rendered && clip.mediaBytes !== undefined && clip.mediaBytes !== null && clip.mediaBytes > 0;
  const hasTimestamps = Number.isFinite(clip.start) && Number.isFinite(clip.end) && clip.end > clip.start;

  // A stock "Clip" title is a symptom of the AI having nothing to work from
  // — a one-shot failure where the backend had no hook_text, no transcript
  // text, and no words overlapping the window. The user-facing fix is to
  // surface the clip's own content instead of leaving the title blank: the
  // hook line is the pitch the model already wrote for this moment, and the
  // subtitle track is the spoken text in order, so either one gives a real
  // name to a clip the AI labelled with a placeholder.
  function fancyTitle(): string {
    if (clip.title && clip.title !== "Clip" && clip.title.trim()) {
      return clip.title;
    }
    if (clip.hookText && clip.hookText.trim()) {
      return clip.hookText.trim();
    }
    // The subtitle track is the most complete, ordered text the model ever
    // produced for this clip, so lead with its first real line. The footer
    // credits strip and repeated boilerplate are filtered out before a
    // readable label is chosen.
    const lines = (clip.captionCues ?? [])
      .map((c) => (c.text ?? "").trim())
      .filter((t) => t && t.length >= 3 && t.length <= 90 && !isFooterLine(t));
    if (lines.length) {
      return lines[0];
    }
    return clip.title;
  }

  function isFooterLine(line: string): boolean {
    return /^(music|audio|sound|voiceover|voice over|music by|producer|senior producer)/i.test(line);
  }

  const downloadMp4 = async () => {
    if (!rendered) {
      toast.error("No rendered MP4 exists for this clip yet.");
      return;
    }
    setDownloading(true);
    try {
      await downloadClipMp4(clip);
      toast.success("MP4 downloaded.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Download failed.");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="group flex flex-col overflow-hidden rounded-xl border border-border bg-card transition hover:border-primary/40">
      <div className="relative p-2">
        <ClipPreview clip={clip} className="w-full" />
        {/* No overlay button over the video. One used to sit here covering the
            whole frame to open the detail dialog, and it swallowed every tap:
            the <video> underneath never received a click, so play() was never
            called and no clip could be previewed from the grid at all. The
            detail view is now reached from the expand button in the player's
            own control bar, and from the title below. */}
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
          {/* honest render-state chip — no fake "ready" look for metadata-only clips */}
          {failed && (
            <span className="flex items-center gap-1 rounded-full border border-destructive/50 bg-destructive/15 px-2 py-0.5 text-[10px] font-bold text-destructive">
              <TriangleAlert className="size-3" />
              render failed
            </span>
          )}
          {!failed && !rendered && (
            <span className="flex items-center gap-1 rounded-full border border-border bg-black/60 px-2 py-0.5 text-[10px] font-bold text-muted-foreground">
              <Loader2 className={clip.renderState === "running" ? "size-3 animate-spin" : "size-3"} />
              {hasTimestamps && clip.renderState === "pending"
                ? "queued"
                : clip.renderState === "running"
                  ? "rendering"
                  : hasTimestamps
                    ? "not rendered"
                    : "no timestamps"}
            </span>
          )}
          {failed && (
            <span className="flex items-center gap-1 rounded-full border border-destructive/50 bg-destructive/15 px-2 py-0.5 text-[10px] font-bold text-destructive" title={clip.mediaError ?? "render failed"}>
              <TriangleAlert className="size-3" />
              render failed
            </span>
          )}
          {rendered && !playable && (
            <span className="flex items-center gap-1 rounded-full border border-amber-400/50 bg-amber-400/10 px-2 py-0.5 text-[10px] font-bold text-amber-300">
              <Film className="size-3" />
              render exists, not playable
            </span>
          )}
          {rendered && !playable && (
            <span className="flex items-center gap-1 rounded-full border border-amber-400/50 bg-amber-400/10 px-2 py-0.5 text-[10px] font-bold text-amber-300">
              <Film className="size-3" />
              render exists, not playable
            </span>
          )}
        </div>
      </div>
      <div className="flex flex-1 flex-col gap-2 px-3 pb-3">
        <div>
          <button
            type="button"
            onClick={() => onOpen?.(clip)}
            className="block w-full cursor-pointer truncate text-left text-sm font-bold tracking-tight hover:text-primary"
            title={fancyTitle()}
          >
            {fancyTitle()}
          </button>
          <p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
            {clip.hookText}
          </p>
        </div>
        <p className="mt-2 font-mono text-[10px] text-muted-foreground">
          {hasTimestamps
            ? `${fmtDuration(clip.start)} → ${fmtDuration(clip.end)} · ${fmtDuration(clip.end - clip.start)}`
            : "— → — · —"}
          · 9:16 1080×1920
          {clip.mediaBytes !== undefined && clip.mediaBytes !== null &&
            ` · ${(clip.mediaBytes / (1024 * 1024)).toFixed(1)} MB`}
          {failed && clip.mediaError && (
            ` · ${String(clip.mediaError).slice(0, 60)}`
          )}
        </p>
        {/* Says out loud that some captions are markers, not transcriptions —
            otherwise [muttering] in the video looks like a bug. */}
        {(clip.uncertainCues ?? 0) > 0 && (
          <p
            className="flex items-center gap-1 text-[10px] text-amber-300/90"
            title={`${clip.uncertainCues} caption line(s) were unclear speech, so they show [muttering]/[inaudible] instead of a guess.`}
          >
            <EarOff className="size-3" />
            {clip.uncertainCues} unclear caption{clip.uncertainCues === 1 ? "" : "s"}
          </p>
        )}
        {/* Primary actions. flex-wrap so nothing can ever be pushed outside a
            narrow card: a 4th button here previously pushed MP4 and .srt off
            the edge where they still existed in the DOM but were unreachable. */}
        <div className="mt-auto flex flex-wrap items-center gap-1.5">
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
            disabled={downloading || !playable}
            title={playable ? "Download the rendered MP4" : "No playable MP4 yet"}
            onClick={() => void downloadMp4()}
          >
            {downloading ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Download className="size-3.5" />
            )}
            MP4
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-7 cursor-pointer gap-1.5 text-xs"
            disabled={!playable}
            title={playable ? "Download the rendered MP4" : "No playable MP4 yet"}
            onClick={() => {
              buildClipSrt(clip);
              toast.success("SRT downloaded.");
            }}
          >
            .srt
          </Button>
          {failed && (
            <Button
              size="sm"
              variant="outline"
              className="h-7 cursor-pointer gap-1.5 text-xs"
              onClick={async () => {
                try {
                  await api.retryClipRender(clip._id);
                  toast.success("Re-render requested for this clip.");
                } catch (err) {
                  toast.error(err instanceof Error ? err.message : "Retry render failed.");
                }
              }}
            >
              <RefreshCw className="size-3.5" />
              Retry render
            </Button>
          )}
          {rendered && !playable && (
            <span className="mt-1 block text-[11px] text-amber-300/90">
              Rendered file is there, but this card cannot play or download it yet.
            </span>
          )}
        </div>
        {/* "Share..." gets its OWN row. Social is strictly additive: it can
            never take space from, reorder or hide the download above. */}
        <div className="flex flex-wrap items-center gap-1.5">
          <ShareButton clip={clip} rendered={rendered} />
          <XQuickButton clip={clip} />
          <PostMenu clip={clip} rendered={rendered} />
        </div>
        <CaptionTracks clip={clip} />
      </div>
    </div>
  );
}

/**
 * Sidecar caption tracks, one chip per language. Rendered only when the job
 * actually produced them ([translate] enabled), so an untranslated job shows
 * nothing extra rather than an empty section.
 */
function CaptionTracks({ clip }: { clip: ClipDto }) {
  const tracks = clip.translationTracks ?? [];
  if (tracks.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1 border-t border-border/60 pt-2">
      <span className="pr-1 text-[10px] uppercase tracking-wide text-muted-foreground">
        Subtitles
      </span>
      {tracks.map((track) => {
        const file = track.files?.srt ?? track.files?.vtt;
        if (track.error || !file) {
          return (
            <span
              key={track.lang}
              title={track.error ?? "track unavailable"}
              className="rounded bg-destructive/10 px-1.5 py-0.5 text-[10px] font-medium text-destructive"
            >
              {track.lang}
            </span>
          );
        }
        return (
          <a
            key={track.lang}
            href={`/api/clips/${clip._id}/tracks/${encodeURIComponent(file)}`}
            download
            title={
              track.source
                ? `${track.lang} — original transcript`
                : `${track.lang} translation${track.cached ? " (cached)" : ""}`
            }
            className={cn(
              "rounded border px-1.5 py-0.5 text-[10px] font-medium transition-colors",
              track.source
                ? "border-primary/40 bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:border-primary/40 hover:text-foreground",
            )}
          >
            {track.lang}
            {track.source ? "*" : ""}
          </a>
        );
      })}
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
