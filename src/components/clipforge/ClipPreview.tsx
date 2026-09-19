import { motion } from "framer-motion";
import { Play, Pause, Volume2, VolumeX } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { clipScenes, fmtClock } from "@/lib/clipforge";
import type { ClipDoc } from "@/lib/clipforge";

// ---------------------------------------------------------------------------
// Word-pop caption overlay (green outline, white bold text, pop animation)
// ---------------------------------------------------------------------------

function CaptionTrack({
  cues,
  time,
}: {
  cues: ClipDoc["captionCues"];
  time: number;
}) {
  const active = cues.find((c) => time >= c.start && time < c.end);
  if (!active) return <div className="h-9" />;
  return (
    <div className="flex h-9 items-center justify-center px-4">
      <div className="flex flex-wrap items-center justify-center gap-x-1.5">
        {active.text.split(" ").map((w, i) => (
          <motion.span
            key={`${active.start}-${i}-${w}`}
            initial={{ scale: 0.3, opacity: 0, y: 4 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            transition={{
              type: "spring",
              stiffness: 620,
              damping: 19,
              delay: i * 0.045,
            }}
            className="animate-word-pop inline-block text-[13px] font-black uppercase tracking-wide text-white"
            style={{ textShadow: "0 0 2px #000, 2px 2px 0 #39FF14, -2px 2px 0 #39FF14, 2px -2px 0 #39FF14, -2px -2px 0 #39FF14" }}
          >
            {w}
          </motion.span>
        ))}
      </div>
    </div>
  );
}

function SceneVisual({
  gradient,
  tint,
  label,
}: {
  gradient: string;
  tint: string;
  label: string;
}) {
  return (
    <div
      className={cn(
        "absolute inset-0 flex items-center justify-center bg-gradient-to-br",
        gradient,
      )}
    >
      {/* speaker silhouette with face-tracking reticle */}
      <div className="relative h-full w-full overflow-hidden">
        <div className="absolute left-1/2 top-[18%] h-16 w-16 -translate-x-1/2 rounded-full bg-black/35 ring-1 ring-white/10" />
        <div className="absolute left-1/2 top-[38%] h-40 w-28 -translate-x-1/2 rounded-t-full bg-black/30 ring-1 ring-white/10" />
        <motion.div
          className="absolute left-1/2 top-[24%] h-20 w-14 -translate-x-1/2 rounded-lg border-2"
          style={{ borderColor: tint, boxShadow: `0 0 18px ${tint}55` }}
          animate={{ x: [-14, 10, -6, 12, -14] }}
          transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>
      <span className="absolute bottom-16 left-3 rounded bg-black/55 px-1.5 py-0.5 text-[9px] font-medium text-white/70">
        {label}
      </span>
    </div>
  );
}

export function ClipPreview({
  clip,
  className,
  autoPlay = false,
}: {
  clip: Pick<ClipDoc, "_id" | "start" | "end" | "captionCues" | "silenceCuts" | "cropBox">;
  className?: string;
  autoPlay?: boolean;
}) {
  const duration = Math.max(0.1, clip.end - clip.start);
  const scenes = useMemo(() => clipScenes(clip._id), [clip._id]);
  const [playing, setPlaying] = useState(autoPlay);
  const [muted, setMuted] = useState(true);
  const [time, setTime] = useState(0);
  const rafRef = useRef<number | null>(null);
  const lastRef = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) {
      lastRef.current = null;
      return;
    }
    const loop = (ts: number) => {
      if (lastRef.current !== null) {
        const dt = (ts - lastRef.current) / 1000;
        setTime((t) => (t + dt > duration ? 0 : t + dt));
      }
      lastRef.current = ts;
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      lastRef.current = null;
    };
  }, [playing, duration]);

  // silence-removed cut accounting (Silero VAD simulation)
  const cutAt = (t: number) => {
    let shift = 0;
    for (const c of clip.silenceCuts) {
      if (t >= c.end) shift += c.end - c.start;
      else if (t >= c.start) return null;
      else break;
    }
    return Math.min(duration, t - shift);
  };

  // active scene given silence-trimmed time
  const visibleSpan = Math.max(0.1, duration - clip.silenceCuts.reduce((a, c) => a + (c.end - c.start), 0));
  const sceneCount = scenes.length;
  const sceneIdx = Math.min(sceneCount - 1, Math.floor((cutAt(time) ?? time) / (visibleSpan / sceneCount)));

  const scene = scenes[Math.max(0, Math.min(sceneIdx, sceneCount - 1))];
  const cutTime = cutAt(time);
  const inCut = cutTime === null;

  return (
    <div className={cn("relative overflow-hidden rounded-xl border border-border bg-black", className)}>
      <div className="aspect-9/16 w-full">
        <SceneVisual gradient={scene.gradient} tint={scene.tint} label={scene.label} />
        {/* crop-box indicator (MediaPipe reframe simulation) */}
        <div
          className="absolute top-2 right-2 rounded border px-1 py-0.5 font-mono text-[8px] text-primary/90"
          style={{ borderColor: "#39FF1444" }}
        >
          crop x{clip.cropBox.x}
        </div>
        {inCut && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-white/70">
              dead air removed
            </span>
        </div>
        )}
        <CaptionTrack cues={clip.captionCues} time={inCut ? 0 : (cutTime ?? 0)} />
        <div className="absolute inset-x-0 bottom-0 flex items-center gap-2 bg-gradient-to-t from-black/80 to-transparent px-3 pb-2 pt-6">
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
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
              style={{ width: `${(time / duration) * 100}%` }}
            />
          </div>
          <span className="font-mono text-[9px] text-white/70">
            {fmtClock(time)} / {fmtClock(duration)}
</span>
        </div>
      </div>
    </div>
  );
}
