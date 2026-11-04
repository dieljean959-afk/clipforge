import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
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
import { fmtClock, fmtDuration } from "@/lib/clipmake";
import { api } from "@/lib/api";
import type { ClipDto } from "@/lib/api";

/** Fired whenever one preview starts playing, so every other one pauses. */
const PLAY_EVENT = "clipmake:clip-play";

/** HTMLMediaElement error codes, named. */
const MEDIA_ERR_ABORTED = 1;
const MEDIA_ERR_NETWORK = 2;
const MEDIA_ERR_DECODE = 3;
const MEDIA_ERR_SRC_NOT_SUPPORTED = 4;

/**
 * Whether a load failure is worth retrying.
 *
 * MEDIA_ERR_SRC_NOT_SUPPORTED (4) means the browser has definitively rejected
 * the file — retrying that is pointless and it will just spin. Network (2) and
 * decode (3) are transient far more often than not: a flaky mobile connection,
 * a dev server restarting, a proxy hiccup mid-transfer. Treating those as fatal
 * is what turned one blip into a permanent "this browser could not play the
 * rendered file" card on a file that is perfectly fine — and on a phone that
 * means the clip is unjudgeable with no way back.
 */
function isRetryable(code: number | undefined): boolean {
  return code === undefined || code === MEDIA_ERR_NETWORK || code === MEDIA_ERR_DECODE;
}

/** Human-readable reason, so the card never says "it broke" with no detail. */
function mediaErrorText(code: number | undefined, msg: string): string {
  switch (code) {
    case MEDIA_ERR_NETWORK:
      return "The connection dropped while loading it. The file is fine — tap retry.";
    case MEDIA_ERR_DECODE:
      return "The connection stalled part-way through. The file is fine — tap retry.";
    case MEDIA_ERR_SRC_NOT_SUPPORTED:
      return "This browser rejected the codec. Download it instead.";
    case MEDIA_ERR_ABORTED:
      return "Loading was cancelled. Tap retry.";
    default:
      return msg || "The file could not be loaded.";
  }
}

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
function cueAt(clip: ClipDto, t: number): string | null {
  const cues = clip.captionCues ?? [];
  for (const cue of cues) {
    if (t >= cue.start - 0.05 && t <= cue.end + 0.05) return cue.text;
  }
  return null;
}

/** Bold-stack preview colours, in the same cycle the render burns in. */
const STACK_COLOURS = ["#FFFF00", "#00FF00", "#FF0000"] as const;

/**
 * The bold-stack chunk visible at `t`: two balanced lines, one coloured.
 *
 * Mirrors the render's rules (see app/subtitle.py: split_stack_lines /
 * build_bold_stack_events) so the overlay says what the MP4 will say: ALL
 * CAPS, nearest-equal split, <= 3 words a line, colour cycling per chunk and
 * alternating between line 1 and line 2. It cannot mirror the pixel size —
 * the card is a scaled-down frame — so sizes come from CSS and the POSITIONS
 * come from the same 1340 / 1426 of 1920 the .ass uses.
 */
function stackAt(
  clip: ClipDto,
  t: number,
): { line1: string; line2: string; colour: string; firstIsColoured: boolean } | null {
  const cues = clip.captionCues ?? [];
  const idx = cues.findIndex((c) => t >= c.start - 0.05 && t <= c.end + 0.05);
  if (idx < 0) return null;
  const words = (cues[idx].text || "").toUpperCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return null;
  let bestI = words.length > 1 ? 1 : 0;
  let best = Number.POSITIVE_INFINITY;
  for (let i = 1; i < words.length; i += 1) {
    const a = words.slice(0, i).join(" ");
    const b = words.slice(i).join(" ");
    const over = Math.max(0, a.length - 16) + Math.max(0, b.length - 16);
    const score = over * 100 + Math.abs(a.length - b.length);
    if (score < best) {
      best = score;
      bestI = i;
    }
  }
  return {
    line1: words.slice(0, bestI).join(" "),
    line2: words.slice(bestI).join(" "),
    colour: STACK_COLOURS[idx % STACK_COLOURS.length],
    firstIsColoured: idx % 2 === 0,
  };
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
  const videoUrl = clip.videoPath
    ? api.clipVideoUrl(clip._id, clip.mediaBytes ?? undefined)
    : null;
  const thumbUrl = clip.thumbPath ? api.clipThumbUrl(clip._id) : null;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [time, setTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [cue, setCue] = useState<string | null>(null);
  // Network/decode failures are retried with a REAL cache-busting URL and a
  // backoff before the error card is allowed to appear. The poster and the
  // clip's metadata stay on screen throughout, so the frame is still judgeable
  // even while the bytes are still arriving.
  const [retry, setRetry] = useState(0);
  const [retrying, setRetrying] = useState(false);
  const [mediaError, setMediaError] = useState("");
  // True only when the browser has definitively rejected the file (unsupported
  // codec). Offering "Retry" in that case is a lie — it will never work.
  const [fatalError, setFatalError] = useState(false);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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
  // Cache-busted on retry, and the counter MUST ride in the QUERY STRING, not
  // after the `#`. A fragment is never sent to the server and does not
  // participate in cache identity, so a counter placed there left the retry
  // re-requesting the byte-identical URL — answerable from cache, which is why
  // a failed load stayed failed.
  const src = videoUrl
    ? `${videoUrl}${retry > 0 ? `?r=${retry}` : ""}#t=0.001`
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
        console.warn("[Clipmake] playback refused", err);
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
      const target =
        Number.isFinite(v.duration) && v.duration > 0
          ? v.duration
          : next;
      try {
        v.currentTime = Math.min(target, Math.max(0, next));
      } catch {
        /* seeking before metadata is a no-op, not an error worth surfacing */
      }
      setTime(next);
      // Seek can retire the show for a clip without captions while the user is
      // dragging the bar: the overlay was showing the last line of text. Keep it
      // honest instead of leaving the previous line stuck on a soundless clip.
      if (!cue) {
        setCue(cueAt(clip, next) ?? null);
      }
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
    setCue(cueAt(clip, t) ?? null);
  };

  const onError = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const err = e.currentTarget.error;
    const code = err?.code;
    setFatalError(!isRetryable(code));
    if (isRetryable(code) && retry < 3) {
      // Backoff grows with the attempt so a dead connection is not hammered.
      const wait = [0, 1200, 3000][retry] ?? 3000;
      setRetrying(true);
      setMediaError(mediaErrorText(code, err?.message ?? ""));
      retryTimer.current = setTimeout(() => {
        setRetry((n) => n + 1);
        setRetrying(false);
      }, wait);
      return;
    }
    setVideoFailed(true);
    setMediaError(mediaErrorText(code, err?.message ?? ""));
  };

  // A new clip starts from a clean slate; a pending retry must not fire into it.
  useEffect(() => {
    return () => {
      if (retryTimer.current) clearTimeout(retryTimer.current);
    };
  }, [clip._id]);

  const manualRetry = () => {
    if (retryTimer.current) clearTimeout(retryTimer.current);
    setRetrying(false);
    setVideoFailed(false);
    setFatalError(false);
    setRetry((n) => n + 1);
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
          {hasMedia && src !== null && (!videoFailed || thumbUrl) ? (
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
              onCanPlay={() => {
                // The load succeeded, so any pending backoff is pointless.
                if (retryTimer.current) clearTimeout(retryTimer.current);
                setRetrying(false);
                setVideoFailed(false);
              }}
            />
          ) : videoFailed ? (
            <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-muted/40 px-4 text-center">
              <Film className="size-7 text-muted-foreground/60" />
              <p className="text-xs text-muted-foreground">
                This browser could not play the rendered file.
              </p>
              <p className="px-2 text-[10px] leading-relaxed text-muted-foreground/80">
                {mediaError || "Download it instead — the MP4 itself is fine."}
              </p>
              <div className="mt-1 flex flex-wrap items-center justify-center gap-1.5">
                {!fatalError && (
                  <button
                    type="button"
                    onClick={manualRetry}
                    className="inline-flex items-center gap-1.5 rounded-md bg-primary px-2.5 py-1.5 text-[11px] font-semibold text-primary-foreground transition hover:opacity-90"
                  >
                    Retry
                  </button>
                )}
                <a
                  href={api.clipDownloadUrl(clip._id)}
                  className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2.5 py-1.5 text-[11px] font-medium text-foreground transition hover:bg-accent"
                >
                  <Download className="size-3" />
                  Download MP4
                </a>
              </div>
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

          {/* burned-in caption preview — the same TWO STACKED LINES the render
              burns in, at the same 1340 / 1426 of 1920 positions (about 70 % of
              the frame), one coloured and one white. Visible even while the
              video is still loading, so a clip that is stuck on the poster
              still tells you what it says. */}
          {controls && cue !== null && (
            <div className="pointer-events-none absolute inset-0 z-10">
              {(() => {
                const stack = stackAt(clip, time);
                const lineStyle = (coloured: boolean, top: string): CSSProperties => ({
                  top,
                  color: coloured ? (stack?.colour ?? "#FFFF00") : "#FFFFFF",
                  opacity: cue ? 1 : 0.35,
                  WebkitTextStroke: "1.2px #000",
                  paintOrder: "stroke fill",
                  textShadow: "0 2px 5px rgba(0,0,0,0.6)",
                });
                const base =
                  "absolute inset-x-0 -translate-y-full text-center text-[13px] font-extrabold uppercase leading-none tracking-tight";
                return (
                  <>
                    <p className={base} style={lineStyle(stack?.firstIsColoured ?? true, "69.8%")}>
                      {stack ? stack.line1 : cue}
                    </p>
                    {stack?.line2 ? (
                      <p className={base} style={lineStyle(!stack.firstIsColoured, "74.3%")}>
                        {stack.line2}
                      </p>
                    ) : null}
                  </>
                );
              })()}
            </div>
          )}

          {/* retrying banner — the frame stays, so a blip is not a dead card */}
          {retrying && (
            <div className="pointer-events-none absolute inset-x-0 top-0 z-30 flex justify-center pt-2">
              <span className="rounded-full bg-black/70 px-2.5 py-1 text-[10px] font-medium text-white/90">
                Reconnecting…
              </span>
            </div>
          )}

          {/* controls only over real media */}
          {controls && (
            <div className="absolute inset-x-0 bottom-0 z-20 flex items-center gap-1.5 bg-gradient-to-t from-black/85 to-transparent px-2 pb-2 pt-6">
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
                className="relative h-6 min-w-8 flex-1 cursor-pointer touch-none"
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

              <span className="shrink-0 whitespace-nowrap font-mono text-[9px] tabular-nums text-white/80">
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
  const [t, setT] = useState(0);

  // Move focus into the dialog so Escape and the close button are reachable
  // without hunting for them.
  useEffect(() => {
    ref.current?.focus();
  }, []);

  if (src === null) return null;

  // A phone on a bad connection cannot always stream the clip — and "can I
  // play it" is not the same question as "is this clip any good". The full cue
  // list is on screen either way, so the hook, the wording and the caption
  // timing stay judgeable even when the bytes will not arrive.
  const cues = clip.captionCues ?? [];
  const activeIndex = cues.findIndex(
    (c) => t >= c.start - 0.05 && t <= c.end + 0.05,
  );

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
          onTimeUpdate={(e) => setT(e.currentTarget.currentTime)}
          className="block max-h-full w-full max-w-[min(100%,56vh)] object-contain"
          style={{ display: "block" }}
        />
      </div>

      {cues.length > 0 && (
        <div className="max-h-[28vh] shrink-0 overflow-y-auto border-t border-white/10 px-3 py-2">
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-white/50">
            Captions as rendered · {cues.length} cues
          </p>
          <ol className="space-y-0.5">
            {cues.map((cue, i) => (
              <li
                key={`${cue.start}-${i}`}
                className={cn(
                  "flex gap-2 rounded px-1.5 py-1 text-left text-[12px] leading-snug transition",
                  i === activeIndex ? "bg-primary/25 text-white" : "text-white/70",
                )}
              >
                <span className="shrink-0 font-mono text-[10px] text-white/40">
                  {fmtClock(cue.start)}
                </span>
                <span className="min-w-0">{cue.text}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="flex shrink-0 items-center justify-between gap-3 px-3 pb-3">
        <p className="font-mono text-[10px] text-white/60">
          {duration !== null ? fmtClock(duration) : "--:--"} ·{" "}
          {fmtDuration(clip.start)} → {fmtDuration(clip.end)}
        </p>
        <a
          href={api.clipDownloadUrl(clip._id)}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-white/20 px-2.5 py-1.5 text-[11px] font-medium text-white transition hover:bg-white/10"
        >
          <Download className="size-3" />
          Download MP4
        </a>
      </div>
    </div>,
    document.body,
  );
}