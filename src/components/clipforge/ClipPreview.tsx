import { useRef, useState } from "react";
import { Clapperboard, Film, Pause, Play, Volume2, VolumeX } from "lucide-react";
import { cn } from "@/lib/utils";
import { fmtClock } from "@/lib/clipforge";
import { api } from "@/lib/api";
import type { ClipDto } from "@/lib/api";

/**
 * ClipPreview — plays the REAL rendered MP4 streamed from the server via the
 * same-origin /api endpoint. There is no scene generator here on purpose: a
 * clip without a rendered file shows an honest "not rendered yet" state, never
 * invented frames or labels.
 */
export function ClipPreview({
  clip,
  className,
  autoPlay = false,
}: {
  clip: ClipDto;
  className?: string;
  autoPlay?: boolean;
}) {
  const videoUrl = clip.videoPath ? api.clipVideoUrl(clip._id) : null;
  const thumbUrl = clip.thumbPath ? api.clipThumbUrl(clip._id) : null;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState<number | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);

  // Playback state self-heals via DOM/media events; switching clips remounts
  // the media element through the key on the container below, so no effect-
  // based state resets are needed.
  const resetKey = `${clip._id}:${autoPlay ? "1" : "0"}`;

  const hasMedia = videoUrl !== null;
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
      <div className="relative aspect-9/16 w-full" key={resetKey}>
        {hasMedia && !videoFailed ? (
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
        ) : thumbUrl && !videoFailed ? (
          // rendered file exists but the browser failed to play it — still real
          <img src={thumbUrl} alt="Clip first frame" className="h-full w-full object-cover" />
        ) : videoFailed ? (
          <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-muted/40 text-center">
            <Film className="size-7 text-muted-foreground/60" />
            <p className="px-4 text-xs text-muted-foreground">
              This browser could not play the rendered file. Download it
              instead — the MP4 itself is fine.
            </p>
          </div>
        ) : (
          // honest empty state — no video, no invented frames
          <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-muted/40 text-center">
            <Clapperboard className="size-7 text-muted-foreground/60" />
            <p className="px-4 text-xs font-medium text-foreground/80">
              {clip.renderState === "failed"
                ? "Render failed"
                : clip.renderState === "running"
                  ? "Rendering now…"
                  : "Not rendered yet"}
            </p>
            {clip.mediaError && (
              <p className="px-6 text-[11px] leading-relaxed text-muted-foreground">
                {clip.mediaError}
              </p>
            )}
          </div>
        )}

        {/* controls only over real media */}
        {hasMedia && !videoFailed && (
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
