import { useEffect, useRef, useState } from "react";
import { useQuery } from "convex/react";
import { Clapperboard, Film, Loader2, Pause, Play, Volume2, VolumeX } from "lucide-react";
import { api } from "@/convex/_generated/api";
import { cn } from "@/lib/utils";
import { fmtClock } from "@/lib/clipforge";
import type { ClipDoc } from "@/lib/clipforge";

/**
 * ClipPreview — plays the REAL rendered MP4 (HTML5 <video>). There is no
 * scene generator here on purpose: a clip without a rendered file shows an
 * honest "not rendered yet" state, never invented frames or labels.
 */
export function ClipPreview({
  clipId,
  className,
  autoPlay = false,
}: {
  clipId: ClipDoc["_id"];
  className?: string;
  autoPlay?: boolean;
}) {
  const videoUrl = useQuery(api.clips.getClipMediaUrl, { clipId });
  const thumbUrl = useQuery(api.clips.getClipThumbUrl, { clipId });
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState<number | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);

  // reset playback state when switching clips
  useEffect(() => {
    setPlaying(autoPlay);
    setTime(0);
    setDuration(null);
    setVideoFailed(false);
  }, [clipId, autoPlay]);

  const hasMedia = videoUrl !== undefined && videoUrl !== null;
  const loading = videoUrl === undefined || thumbUrl === undefined;

  const togglePlay = () => {
    const v = videoRef.current;
    if (v === null) return;
    if (v.paused) {
      void v.play().catch(() => setVideoFailed(true));
    } else {
      v.pause();
    }
  };

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border border-border bg-black",
        className,
      )}
    >
      <div className="relative aspect-9/16 w-full">
        {loading ? (
          <div className="flex h-full w-full items-center justify-center bg-black">
            <Loader2 className="size-6 animate-spin text-primary/60" />
          </div>
        ) : hasMedia && !videoFailed ? (
          <video
            ref={videoRef}
            src={videoUrl}
            poster={thumbUrl ?? undefined}
            className="h-full w-full object-cover"
            playsInline
            loop
            muted={muted}
            autoPlay={autoPlay}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => {
              const d = e.currentTarget.duration;
              setDuration(Number.isFinite(d) ? d : null);
            }}
            onError={() => setVideoFailed(true)}
            onClick={togglePlay}
          />
        ) : thumbUrl ? (
          // rendered file exists but this browser failed to play it — still real
          <img src={thumbUrl} alt="Clip first frame" className="h-full w-full object-cover" />
        ) : (
          // honest empty state — no video, no invented frames
          <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-muted/40 text-center">
            {videoFailed ? (
              <>
                <Film className="size-7 text-muted-foreground/60" />
                <p className="px-4 text-xs text-muted-foreground">
                  This browser could not play the rendered file. Download it
                  instead — the MP4 itself is fine.
                </p>
              </>
            ) : (
              <>
                <Clapperboard className="size-7 text-muted-foreground/60" />
                <p className="px-4 text-xs font-medium text-foreground/80">
                  Not rendered yet
                </p>
                <p className="px-6 text-[11px] leading-relaxed text-muted-foreground">
                  This clip has metadata only — the 9:16 video is encoded when
                  the render worker runs. No preview is shown because none
                  exists.
                </p>
              </>
            )}
          </div>
        )}

        {/* controls only over real media */}
        {hasMedia && !loading && !videoFailed && (
          <div className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-gradient-to-t from-black/80 to-transparent px-3 pb-2 pt-6">
            <button
              type="button"
              onClick={togglePlay}
              className="rounded-full bg-white/10 p-1.5 text-white transition hover:bg-white/20"
              aria-label={playing ? "Pause preview" : "Play preview"}
            >
              {playing ? <Pause className="size-3" /> : <Play className="size-3" />}
            </button>
            <button
              type="button"
              onClick={() => setMuted((m) => !m)}
              className="rounded-full bg-white/10 p-1.5 text-white transition hover:bg-white/20"
              aria-label={muted ? "Unmute preview" : "Mute preview"}
            >
              {muted ? <VolumeX className="size-3" /> : <Volume2 className="size-3" />}
            </button>
            <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/20">
              <div
                className="h-full bg-primary transition-[width] duration-100"
                style={{
                  width: `${duration !== null && duration > 0 ? Math.min(100, (time / duration) * 100) : 0}%`,
                }}
              />
            </div>
            <span className="font-mono text-[9px] text-white/70">
              {fmtClock(time)} / {duration !== null ? fmtClock(duration) : "--:--"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
