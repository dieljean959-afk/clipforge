import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Clapperboard,
  Download,
  Film,
  Maximize2,
  Pause,
  Play,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { fmtClock, fmtDuration } from "@/lib/clipforge";
import { api } from "@/lib/api";
import type { ClipDto } from "@/lib/api";

/** Fired whenever one preview starts playing, so every other one pauses. */
const PLAY_EVENT = "clipforge:clip-play";

/**
 * ClipPreview — plays the REAL rendered MP4 streamed from the server via the
 * same-origin /api endpoint. There is no scene generator here on purpose: a
 * clip without a rendered file shows an honest "not rendered yet" state, never
 * invented frames or labels.
 *
 * Everything here exists because of something that was actually broken:
 *
 * 1. TAPS REACHED THE WRONG ELEMENT. The card wrapped this in a transparent
 *    <button> covering the whole video to open the detail dialog, so every tap
 *    meant "open the dialog" and `video.play()` was never called. That overlay
 *    is gone; the detail view is now an explicit expand button, and the video
 *    itself owns the tap.
 *
 * 2. SIZING IS EXPLICIT, not inherited. The frame is `aspect-[9/16]` and the
 *    video is `block h-full w-full object-cover`. `display:block` matters:
 *    <video> is inline by default, so it sits on the text baseline and the
 *    line box underneath it is dead space that reads as a black bar.
 *
 * 3. `preload="metadata"` plus `#t=0.001`. iOS will not paint a first frame
 *    for a video parked at exactly 0, so the element is a black rectangle
 *    until it is seeked. Nudging the media fragment makes it fetch a frame.
 *
 * 4. A duration fallback. `video.duration` is NaN before metadata lands and
 *    Infinity for a stream it cannot size, so the clip's own length
 *    (`end - start`) is shown until the element reports a real one.
 *
 * 5. The burned-in caption is previewed as an overlay driven by currentTime,
 *    so you can judge the clip without squinting at 9px text in a 1080-wide
 *    frame scaled into a phone card.
 */

/** The caption cue visible at `t` seconds (clip-relative). */
function cueAt(clip: ClipDto, t: number): string {
  const cues = clip.captionCues ?? [];
  for (const cue of cues) {
    if (t >= cue.start - 0.05 && t <= cue.end + 0.05) return cue.text;
  }
  return "";
}

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
  const trackRef = useRef<HTMLDivElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [time, setTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [cue, setCue] = useState("");
  // 0 = first attempt, 1 = one retry already spent. A single retry covers the
  // real cause (an aborted first request while the server was restarting)
  // without looping forever on a genuinely broken file.
  const [retryCount, setRetryCount] = useState(0);

  // The clip's own length: what the detection step asked for. Shown whenever
  // the element has not reported a usable duration.
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

  // --- playback ---------------------------------------------------------

  const togglePlay = useCallback(async () => {
    const v = videoRef.current;
    if (v === null) return;
    if (v.paused || v.ended) {
      try {
        // Tell every other preview to stop. Without this, a grid of clips
        // plays all at once and the phone turns into a crowd.
        window.dispatchEvent(new CustomEvent(PLAY_EVENT, { detail: v }));
        await v.play();
      } catch (err) {
        // Autoplay policies reject play() when it is not user-initiated, and
        // a decode error rejects too. Either way the card must say so rather
        // than sitting on a play icon that does nothing.
        setVideoFailed(true);
        console.warn("[ClipForge] playback refused", err);
      }
    } else {
      v.pause();
    }
  }, []);

  const toggleMute = useCallback(() => {
    const v = videoRef.current;
    if (v === null) return;
    v.muted = !v.muted;
    setMuted(v.muted);
  }, []);

  // Pause this clip whenever another one starts.
  useEffect(() => {
    const onOtherPlay = (e: Event) => {
      const other = (e as CustomEvent<HTMLVideoElement>).detail;
      if (other && other !== videoRef.current) other.pause();
    };
    window.addEventListener(PLAY_EVENT, onOtherPlay);
    return () => window.removeEventListener(PLAY_EVENT, onOtherPlay);
  }, []);

  // Close the modal on Escape — the one shortcut a phone user reaches for.
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  // --- seek -------------------------------------------------------------

  const seekTo = useCallback(
    (clientX: number) => {
      const v = videoRef.current;
      const track = trackRef.current;
      if (v === null || track === null || !duration) return;
      const rect = track.getBoundingClientRect();
      if (rect.width <= 0) return;
      const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      const next = ratio * duration;
      // `duration` can still be the fallback here; only seek once the element
      // reports a real one, otherwise we would seek against a guess.
      const target = Number.isFinite(v.duration) && v.duration > 0 ? v.duration : next;
      try {
        v.currentTime = Math.min(target, Math.max(0, next));
      } catch {
        /* seeking before metadata is a no-op, not an error worth surfacing */
      }
      setTime(next);
    },
    [duration],
  );

  const [scrubbing, setScrubbing] = useState(false);
  const [scrubPct, setScrubPct] = useState<number | null>(null);

  const progress = (() => {
    const pct = scrubPct ?? (duration ? (time / duration) * 100 : 0);
    return Math.min(100, Math.max(0, pct));
  })();

  // --- media element handlers -------------------------------------------

  const onLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const d = e.currentTarget.duration;
    setMediaDuration(Number.isFinite(d) && d > 0 ? d : null);
  };

  const onTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const t = e.currentTarget.currentTime;
    setTime(t);
    setCue(cueAt(clip, t));
  };

  const onError = () => {
    if (retryCount === 0) {
      // Retry once, fresh URL, before declaring the file unplayable.
      setRetryCount(1);
      setVideoFailed(false);
      return;
    }
    setVideoFailed(true);
  };

  const controls = hasMedia && !videoFailed;

  return (
    <>
      <div
        className={cn(
          "relative overflow-hidden rounded-xl border border-border bg-black",
          className,
        )}
      >
        <div className="relative aspect-[9/16] w-full" key={resetKey}>
          {hasMedia && !videoFailed && src !== null ? (
            <video
              ref={videoRef}
              src={src}
              poster={thumbUrl ?? undefined}
              // block kills the inline baseline gap; h-full + object-cover fill
              // the 9:16 box edge to edge with no letterbox of our own making
              className="block h-full w-full object-cover"
              style={{ display: "block" }}
              preload="metadata"
              playsInline
              loop={false}
              muted={muted}
              autoPlay={autoPlay && muted}
              onClick={() => void togglePlay()}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
              onTimeUpdate={onTimeUpdate}
              onLoadedMetadata={onLoadedMetadata}
              // durationchange also fires when the element rescales its
              // duration after a source swap; loadedmetadata alone misses it,
              // which is how the total stayed "--:--".
              onDurationChange={onLoadedMetadata}
              onError={onError}
            />
          ) : thumbUrl && !videoFailed ? (
            // rendered file exists but the browser failed to play it — still real
            <img
              src={thumbUrl}
              alt="Clip first frame"
              className="block h-full w-full object-cover"
            />
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

          {/* burned-in caption preview — the same text the render burns in */}
          {controls && cue !== "" && (
            <p className="pointer-events-none absolute inset-x-3 bottom-16 z-10 text-center text-[11px] font-bold leading-snug text-white [text-shadow:0_2px_6px_rgba(0,0,0,0.9)]">
              {cue}
            </p>
          )}

          {/* controls only over real media */}
          {controls && (
            <div className="absolute inset-x-0 bottom-0 z-20 flex items-center gap-2 bg-gradient-to-t from-black/85 to-transparent px-3 pb-2 pt-6">
              <button
                type="button"
                onClick={() => void togglePlay()}
                className="rounded-full bg-white/15 p-1.5 text-white transition hover:bg-white/25"
                aria-label={playing ? "Pause preview" : "Play preview"}
              >
                {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
              </button>
              <button
                type="button"
                onClick={toggleMute}
                className="rounded-full bg-white/15 p-1.5 text-white transition hover:bg-white/25"
                aria-label={muted ? "Unmute preview" : "Mute preview"}
              >
                {muted ? <VolumeX className="size-3.5" /> : <Volume2 className="size-3.5" />}
              </button>

              {/* seek bar: tap to jump, drag to scrub */}
              <div
                ref={trackRef}
                role="slider"
                tabIndex={0}
                aria-label="Seek"
                aria-valuemin={0}
                aria-valuemax={Math.round(duration ?? 0)}
                aria-valuenow={Math.round(time)}
                onPointerDown={(e) => {
                  (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
                  setScrubbing(true);
                  seekTo(e.clientX);
                }}
                onPointerMove={(e) => {
                  if (!scrubbing) return;
                  const rect = e.currentTarget.getBoundingClientRect();
                  if (rect.width > 0) {
                    setScrubPct(
                      Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100)),
                    );
                  }
                }}
                onPointerUp={(e) => {
                  seekTo(e.clientX);
                  setScrubbing(false);
                  setScrubPct(null);
                }}
                onPointerCancel={() => {
                  setScrubbing(false);
                  setScrubPct(null);
                }}
                onKeyDown={(e) => {
                  const v = videoRef.current;
                  if (v === null || !duration) return;
                  if (e.key === "ArrowRight") v.currentTime = Math.min(duration, v.currentTime + 5);
                  if (e.key === "ArrowLeft") v.currentTime = Math.max(0, v.currentTime - 5);
                }}
                className="relative h-6 flex-1 cursor-pointer touch-none"
              >
                <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 overflow-hidden rounded-full bg-white/25">
                  <div
                    className="h-full bg-primary transition-[width] duration-100"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div
                  className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary shadow"
                  style={{ left: `${progress}%` }}
                />
              </div>

              <span className="shrink-0 font-mono text-[9px] text-white/80">
                {fmtClock(time)} / {duration !== null ? fmtClock(duration) : "--:--"}
              </span>

              <button
                type="button"
                onClick={() => setExpanded(true)}
                className="rounded-full bg-white/15 p-1.5 text-white transition hover:bg-white/25"
                aria-label="Open full preview"
              >
                <Maximize2 className="size-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>

      {expanded && (
        <ClipLightbox
          clip={clip}
          src={src}
          poster={thumbUrl}
          duration={duration}
          onClose={() => setExpanded(false)}
        />
      )}
    </>
  );
}

/**
 * The review view: one clip, large, with the browser's OWN controls. Native
 * controls are deliberate — they bring scrub, fullscreen, PiP, playback speed
 * and the audio route for free, all of which a hand-rolled control bar on a
 * phone does badly. Sound is allowed here because playback starts from a real
 * user gesture inside this dialog.
 */
function ClipLightbox({
  clip,
  src,
  poster,
  duration,
  onClose,
}: {
  clip: ClipDto;
  src: string | null;
  poster: string | null;
  duration: number | null;
  onClose: () => void;
}) {
  const ref = useRef<HTMLVideoElement | null>(null);

  // Move focus into the dialog so Escape and the close button are reachable
  // without hunting for them.
  useEffect(() => {
    ref.current?.focus();
  }, []);

  if (src === null) return null;

  // Portalled to <body> on purpose. `position: fixed` is only viewport-relative
  // while no ancestor establishes a containing block — and the dashboard uses
  // framer-motion, whose transforms do exactly that. Rendered in place, the
  // dialog would be clipped by the card's `overflow-hidden` and sized to the
  // card instead of the screen.
  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Preview: ${clip.title}`}
      className="fixed inset-0 z-[100] flex flex-col bg-black/95 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="flex items-center justify-between gap-3 px-3 py-2">
        <p className="min-w-0 flex-1 truncate text-sm font-semibold text-white">
          {clip.title}
        </p>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close preview"
          className="rounded-full bg-white/10 p-2 text-white transition hover:bg-white/20"
        >
          <X className="size-5" />
        </button>
      </div>

      <div className="flex min-h-0 flex-1 items-center justify-center px-2 pb-2">
        <video
          ref={ref}
          src={src}
          poster={poster ?? undefined}
          controls
          autoPlay
          playsInline
          // Unmuted: this is the "judge the clip properly" view and it is only
          // ever reached from a tap, so autoplay policies allow the audio.
          muted={false}
          tabIndex={-1}
          onPlay={(e) => {
            // Still only one clip at a time.
            window.dispatchEvent(new CustomEvent(PLAY_EVENT, { detail: e.currentTarget }));
          }}
          className="block max-h-full w-full max-w-[min(100%,56vh)] object-contain"
          style={{ display: "block" }}
        />
      </div>

      <p className="px-3 pb-3 text-center font-mono text-[10px] text-white/60">
        {duration !== null ? fmtClock(duration) : "--:--"} ·{" "}
        {fmtDuration(clip.start)} → {fmtDuration(clip.end)}
      </p>
    </div>,
    document.body,
  );
}