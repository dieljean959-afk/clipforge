import { useRef, useState } from "react";
import {
  Clapperboard,
  Download,
  Film,
  Pause,
  Play,
  Volume2,
  VolumeX,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { fmtClock } from "@/lib/clipforge";
import { api } from "@/lib/api";
import type { ClipDto } from "@/lib/api";

/**
 * ClipPreview — plays the REAL rendered MP4 streamed from the server via the
 * same-origin /api endpoint. There is no scene generator here on purpose: a
 * clip without a rendered file shows an honest "not rendered yet" state, never
 * invented frames or labels.
 *
 * Three things here are load-bearing for playback on a phone, and each was a
 * visible bug before:
 *
 * 1. `preload="metadata"`. With the default `preload="auto"` a grid of a dozen
 *    players each pulls megabytes nobody asked for; with `preload="none"` the
 *    duration never arrives and the scrubber stays at "0:00 / --:--". Metadata
 *    is the right amount.
 * 2. `#t=0.001`. iOS Safari will not paint a first frame for a video whose
 *    current time is exactly 0, so the element renders as a black rectangle
 *    until it is seeked. Nudging the media fragment makes it fetch a frame.
 * 3. The duration fallback. `video.duration` is NaN until metadata loads and
 *    Infinity for a stream it cannot size, so the clip's own length
 *    (`end - start`, which is what the card already prints) is shown until the
 *    element reports a real one. A player that knows it is 17.2s long should
 *    never claim it does not know how long the clip is.
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
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  // 0 = first attempt, 1 = one retry already spent. A single retry covers the
  // real cause (a truncated/aborted first request while the dev server or the
  // backend was restarting) without looping forever on a genuinely broken file.
  const [retryCount, setRetryCount] = useState(0);

  // The clip's own length: what the detection step asked for. This is the
  // fallback whenever the element has not reported a usable duration.
  const clipSeconds = Math.max(0, clip.end - clip.start);
  const duration =
    mediaDuration !== null && Number.isFinite(mediaDuration) && mediaDuration > 0
      ? mediaDuration
      : clipSeconds > 0
        ? clipSeconds
        : null;

  // Playback state self-heals via DOM/media events; switching clips remounts
  // the media element through the key on the container below, so no effect-
  // based state resets are needed.
  const resetKey = `${clip._id}:${autoPlay ? "1" : "0"}`;

  const hasMedia = videoUrl !== null;
  // Cache-busted URL: a re-render replaces the bytes behind the same clip id,
  // and an element that already failed (or already buffered the old encode)
  // would otherwise never re-request them.
  const src = videoUrl
    ? `${videoUrl}#t=0.001${retryCount > 0 ? `&r=${retryCount}` : ""}`
    : null;

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
        {hasMedia && !videoFailed && src !== null ? (
          <video
            ref={videoRef}
            src={src}
            poster={thumbUrl ?? undefined}
            className="h-full w-full object-cover"
            preload="metadata"
            playsInline
            loop
            muted={muted}
            autoPlay={autoPlay}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => {
              const d = e.currentTarget.duration;
              setMediaDuration(Number.isFinite(d) && d > 0 ? d : null);
            }}
            // durationchange fires when the element rescales its duration —
            // after a metadata load, after a source swap, and when a track is
            // added. Listening only to loadedmetadata missed every case but
            // the first, which is why the total could stay "--:--" even though
            // the browser knew it.
            onDurationChange={(e) => {
              const d = e.currentTarget.duration;
              setMediaDuration(Number.isFinite(d) && d > 0 ? d : null);
            }}
            onError={() => {
              if (retryCount === 0) {
                // Retry once, fresh URL, before declaring the file unplayable.
                setRetryCount(1);
                setVideoFailed(false);
                return;
              }
              setVideoFailed(true);
            }}
            onClick={togglePlay}
          />
        ) : thumbUrl && !videoFailed ? (
          // rendered file exists but the browser failed to play it — still real
          <img src={thumbUrl} alt="Clip first frame" className="h-full w-full object-cover" />
        ) : videoFailed ? (
          <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-muted/40 px-4 text-center">
            <Film className="size-7 text-muted-foreground/60" />
            <p className="text-xs text-muted-foreground">
              This browser could not play the rendered file. Download it
              instead — the MP4 itself is fine.
            </p>
            <a
              href={api.clipDownloadUrl(clip._id)}
              className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2.5 py-1.5 text-[11px] font-medium text-foreground transition hover:bg-accent"
            >
              <Download className="size-3" />
              Download MP4
            </a>
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