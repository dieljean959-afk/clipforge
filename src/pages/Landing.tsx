import { motion } from "framer-motion";
import {
  ArrowRight,
  AudioWaveform,
  Clapperboard,
  Download,
  Flame,
  Github,
  Scan,
  Scissors,
  Sparkles,
  Type as TypeIcon,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useHealth } from "@/hooks/use-poll";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { LogoWordmark } from "@/components/clipforge/LogoMark";
import { RightsNotice } from "@/components/clipforge/RightsNotice";
import { cn } from "@/lib/utils";

const fadeUp = {
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-60px" },
};

function PipelineStep({
  icon: Icon,
  step,
  title,
  body,
  last = false,
}: {
  icon: typeof Download;
  step: string;
  title: string;
  body: string;
  last?: boolean;
}) {
  return (
    <motion.div {...fadeUp} className="relative flex gap-4">
      {!last && (
        <div className="absolute left-[22px] top-11 bottom-[-12px] w-px bg-gradient-to-b from-primary/40 to-transparent" />
      )}
      <div className="relative flex size-11 shrink-0 items-center justify-center rounded-xl border border-primary/30 bg-primary/10 neon-glow-sm">
        <Icon className="size-5 text-primary" />
      </div>
      <div className="pb-6">
        <p className="font-mono text-[10px] uppercase tracking-widest text-primary/70">
          {step}
        </p>
        <h3 className="mt-1 font-bold tracking-tight">{title}</h3>
        <p className="mt-1 max-w-md text-sm leading-relaxed text-muted-foreground">
          {body}
        </p>
      </div>
    </motion.div>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  body,
}: {
  icon: typeof Flame;
  title: string;
  body: string;
}) {
  return (
    <motion.div
      {...fadeUp}
      className="rounded-xl border border-border bg-card p-5 transition hover:border-primary/40"
    >
      <div className="flex size-10 items-center justify-center rounded-lg border border-primary/25 bg-primary/10">
        <Icon className="size-5 text-primary" />
      </div>
      <h3 className="mt-3 font-bold tracking-tight">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">{body}</p>
    </motion.div>
  );
}

/** Mini 9:16 phone mock with static word-pop caption. */
function HeroPhone({ className }: { className?: string }) {
  return (
    <div className={cn("relative", className)}>
      <div className="absolute -inset-6 rounded-full bg-primary/10 blur-3xl" />
      <motion.div
        initial={{ rotate: -3, y: 12 }}
        animate={{ rotate: 0, y: 0 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="relative w-44 overflow-hidden rounded-[1.6rem] border-2 border-primary/40 bg-black neon-glow sm:w-52"
      >
        <div className="flex aspect-9/16 flex-col justify-between bg-gradient-to-br from-[#123318] via-[#0b1d10] to-black p-3">
          <div className="flex justify-between text-[8px] font-mono text-white/50">
            <span>9:16 · 1080×1920</span>
            <span className="text-primary">1080p max</span>
          </div>
          <div className="relative mx-auto my-2 h-24 w-24 self-start rounded-full bg-black/40 ring-1 ring-white/10">
            <motion.div
              className="absolute -inset-2 rounded-lg border-2 border-primary/70"
              animate={{ x: [-6, 6, -6] }}
              transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>
          <div className="space-y-1.5">
            <motion.p
              initial={{ scale: 0.4, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.5, type: "spring", stiffness: 500, damping: 18 }}
              className="inline-block text-sm font-black uppercase text-white"
              style={{ textShadow: "2px 2px 0 #39FF14, -2px 2px 0 #39FF14" }}
            >
              This clip hit
            </motion.p>
            <motion.p
              initial={{ scale: 0.4, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.8, type: "spring", stiffness: 500, damping: 18 }}
              className="inline-block text-sm font-black uppercase text-white"
              style={{ textShadow: "2px 2px 0 #39FF14, -2px 2px 0 #39FF14" }}
            >
              A MILLION
            </motion.p>
          </div>
          <div className="rounded-lg border border-primary/25 bg-black/50 px-2.5 py-2">
            <p className="font-mono text-[8px] uppercase tracking-widest text-primary">
              virality score
            </p>
            <p className="text-lg font-black text-primary neon-text">92</p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default function Landing() {
  // Local app: "signed in" simply means the local API is reachable.
  const { health, error } = useHealth(10_000);
  const [checked, setChecked] = useState(false);
  useEffect(() => {
    if (health !== undefined || error !== null) setChecked(true);
  }, [health, error]);
  const isLoading = !checked;
  const isAuthenticated = health?.ok === true;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="min-h-screen bg-background"
    >
      {/* nav */}
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/85 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-4">
          <LogoWordmark />
          <nav className="hidden items-center gap-6 text-sm text-muted-foreground md:flex">
            <a href="#pipeline" className="transition hover:text-foreground">Pipeline</a>
            <a href="#features" className="transition hover:text-foreground">Features</a>
            <a href="#rights" className="transition hover:text-foreground">Rights</a>
          </nav>
          <div className="flex items-center gap-2">
            {isLoading ? (
              <div className="h-9 w-20 animate-pulse rounded-md bg-muted" />
            ) : isAuthenticated ? (
              <Button asChild size="sm" className="cursor-pointer gap-1.5 font-bold">
                <a href="/dashboard">
                  Dashboard <ArrowRight className="size-3.5" />
                </a>
              </Button>
            ) : (
              <Button asChild size="sm" className="cursor-pointer gap-1.5 font-bold">
                <a href="/auth?returnTo=%2Fdashboard">
                  Start clipping <ArrowRight className="size-3.5" />
                </a>
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* hero */}
      <section className="bg-grid relative overflow-hidden">
        <div className="pointer-events-none absolute -top-24 left-1/2 h-72 w-[42rem] -translate-x-1/2 rounded-full bg-primary/10 blur-3xl" />
        <div className="mx-auto flex w-full max-w-6xl flex-col items-center gap-10 px-4 py-20 md:flex-row md:py-28">
          <div className="max-w-xl flex-1">
            <Badge variant="outline" className="mb-5 gap-1.5 border-primary/40 bg-primary/10 text-primary">
              <Flame className="size-3" />
              MoneyPrinterTurbo fork · MIT · 100% local
            </Badge>
            <h1 className="text-4xl font-black leading-[1.05] tracking-tight sm:text-5xl md:text-6xl">
              Turn one long video into{" "}
              <span className="text-primary neon-text">ten viral clips</span>.
            </h1>
            <p className="mt-5 text-base leading-relaxed text-muted-foreground sm:text-lg">
              ClipForge runs entirely on your machine: yt-dlp grabs the video,
              Whisper transcribes every word with timings, Gemini finds the
              hooks worth posting, and FFmpeg renders vertical 9:16 cuts with
              animated green-outline captions — nothing leaves your computer.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Button asChild size="lg" className="cursor-pointer gap-2 font-bold neon-glow">
                <a href={isAuthenticated ? "/dashboard" : "/auth?returnTo=%2Fdashboard"}>
                  {isAuthenticated ? "Open dashboard" : "Start clipping — free"}
                  <ArrowRight className="size-4" />
                </a>
              </Button>
              <Button asChild size="lg" variant="outline" className="cursor-pointer gap-2">
                <a href="#pipeline">
                  See the pipeline
                </a>
              </Button>
            </div>
            <RightsNotice className="mt-7 max-w-lg" />
          </div>
          <HeroPhone className="flex-1 flex justify-center md:justify-end" />
        </div>
      </section>

      {/* pipeline */}
      <section id="pipeline" className="border-t border-border/60">
        <div className="mx-auto w-full max-w-6xl px-4 py-20">
          <motion.div variants={fadeUp} initial="initial" whileInView="whileInView" viewport={{ once: true }}>
            <p className="font-mono text-xs uppercase tracking-widest text-primary">
              The forge
            </p>
            <h2 className="mt-2 text-3xl font-black tracking-tight sm:text-4xl">
              Four stages, zero timeline scrubbing
            </h2>
          </motion.div>
          <div className="mt-10 grid gap-x-10 md:grid-cols-2">
            <PipelineStep
              icon={Download}
              step="01 — Grab"
              title="Clippy is grabbing the video..."
              body="Paste a YouTube, Vimeo, TikTok or X link — yt-dlp pulls 1080p MP4 straight to your machine, self-updates when YouTube changes, and can borrow your browser cookies for age-gated videos. Manual upload always works as the fallback."
            />
            <PipelineStep
              icon={AudioWaveform}
              step="02 — Transcribe"
              title="Whisper word timings"
              body="faster-whisper produces word-level timestamps, so captions pop exactly on the beat and dead air is measurable."
            />
            <PipelineStep
              icon={Sparkles}
              step="03 — Find"
              title="Hook detection, scored 0-100"
              body="The transcript is chunked and scored for hook strength, pacing and energy. Top 5-15 windows become clips with titles, hooks and hashtags."
            />
            <PipelineStep
              icon={Clapperboard}
              step="04 — Render"
              title="9:16 crops with pop captions"
              body="Face-tracked reframing keeps the speaker centered, Silero VAD cuts the dead air, and captions render bold white with a #39FF14 outline — word by word."
            />
          </div>
        </div>
      </section>

      {/* features */}
      <section id="features" className="border-t border-border/60 bg-card/40">
        <div className="mx-auto w-full max-w-6xl px-4 py-20">
          <motion.div variants={fadeUp} initial="initial" whileInView="whileInView" viewport={{ once: true }}>
            <p className="font-mono text-xs uppercase tracking-widest text-primary">
              Why ClipForge
            </p>
            <h2 className="mt-2 text-3xl font-black tracking-tight sm:text-4xl">
              Built from the MPT bones, aimed at shorts
            </h2>
          </motion.div>
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <FeatureCard
              icon={Scan}
              title="Speaker-tracked reframing"
              body="The 9:16 crop box drifts to follow the active speaker, so talking-head footage survives the vertical cut."
            />
            <FeatureCard
              icon={TypeIcon}
              title="Word-pop captions"
              body="Whisper word timings drive per-word pop-in animation — bold white text, neon green outline, burned into the render."
            />
            <FeatureCard
              icon={Scissors}
              title="Dead-air surgery"
              body="Silence detection trims pauses between sentences per clip, tightening pacing without touching the words."
            />
            <FeatureCard
              icon={Flame}
              title="Virality scores"
              body="Every candidate window gets a 0-100 score with a title and hook line, so you post the best takes first."
            />
            <FeatureCard
              icon={Download}
              title="One-click batch export"
              body="Grab all clips as a ZIP with captions.txt, per-clip SRTs and a manifest — plus hashtags copied to your clipboard."
            />
            <FeatureCard
              icon={Github}
              title="Forked, not from scratch"
              body="ClipForge forked MoneyPrinterTurbo (MIT) and repurposed its subtitle + MoviePy pipeline for clip extraction."
            />
          </div>
        </div>
      </section>

      {/* rights */}
      <section id="rights" className="border-t border-border/60">
        <div className="mx-auto w-full max-w-6xl px-4 py-16">
          <RightsNotice />
        </div>
      </section>

      {/* final CTA */}
      <section className="border-t border-border/60 bg-grid">
        <div className="mx-auto flex w-full max-w-6xl flex-col items-center px-4 py-20 text-center">
          <motion.div variants={fadeUp} initial="initial" whileInView="whileInView" viewport={{ once: true }}>
            <h2 className="max-w-2xl text-3xl font-black tracking-tight sm:text-4xl">
              Your next ten posts are already{" "}
              <span className="text-primary neon-text">in your last upload</span>.
            </h2>
            <p className="mx-auto mt-4 max-w-xl text-muted-foreground">
              One command to start: <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-sm">./start.sh</code>. Paste a link, confirm you own the content, and let the forge run.
            </p>
            <Button asChild size="lg" className="mt-7 cursor-pointer gap-2 font-bold neon-glow">
              <a href={isAuthenticated ? "/dashboard" : "/auth?returnTo=%2Fdashboard"}>
                {isAuthenticated ? "Open dashboard" : "Start clipping — free"}
                <ArrowRight className="size-4" />
              </a>
            </Button>
          </motion.div>
        </div>
      </section>

      <footer className="border-t border-border/60">
        <div className="mx-auto flex w-full max-w-6xl flex-col items-center justify-between gap-3 px-4 py-8 text-xs text-muted-foreground sm:flex-row">
          <p>ClipForge — forked from MoneyPrinterTurbo, MIT licensed, runs locally.</p>
          <p className="font-mono">yt-dlp · faster-whisper · Gemini · FFmpeg</p>
        </div>
      </footer>
    </motion.div>
  );
}
