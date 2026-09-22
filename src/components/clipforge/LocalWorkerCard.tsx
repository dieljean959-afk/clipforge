// Local worker status card — "Worker: LOCAL — running on this machine".
// Shows CPU/RAM (read from /proc on Linux), queue depth, and warmup state.
// No power controls: the worker IS this machine's process.

import { useEffect, useState } from "react";
import { Cpu, HardDrive, Loader2, MemoryStick, ServerCog } from "lucide-react";
import { useHealth } from "@/hooks/use-poll";
import { cn } from "@/lib/utils";

interface ProcStat {
  cores: number;
  load1: number;
  memTotalBytes: number;
  memUsedBytes: number;
}

function useSystemStats(intervalMs = 3000): ProcStat | null {
  const [stats, setStats] = useState<ProcStat | null>(null);
  useEffect(() => {
    let cancelled = false;
    const read = async () => {
      try {
        const [stat, meminfo, loadavg] = await Promise.all([
          fetch("/proc/stat").then((r) => r.text()),
          fetch("/proc/meminfo").then((r) => r.text()),
          fetch("/proc/loadavg").then((r) => r.text()),
        ]);
        if (cancelled) return;
        const cpuLine = stat.split("\n").find((l) => l.startsWith("cpu ")) ?? "";
        const cols = cpuLine.split(/\s+/).slice(1).map(Number);
        const cores = navigator.hardwareConcurrency || 4;
        const load1 = Number.parseFloat(loadavg.split(" ")[0] ?? "0");
        const memTotal = Number(/MemTotal:\s+(\d+)/.exec(meminfo)?.[1] ?? 0) * 1024;
        const memAvail = Number(/MemAvailable:\s+(\d+)/.exec(meminfo)?.[1] ?? 0) * 1024;
        setStats({ cores, load1, memTotalBytes: memTotal, memUsedBytes: memTotal - memAvail });
      } catch {
        // /proc not available (macOS/Windows) — the card hides these lines
      }
    };
    void read();
    const t = setInterval(() => void read(), intervalMs);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [intervalMs]);
  return stats;
}

function fmtBytes(n: number): string {
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(1)} GB`;
}

export function LocalWorkerCard({ className }: { className?: string }) {
  const { health, error } = useHealth(5000);
  const stats = useSystemStats(3000);

  const workerOnline = health?.worker.online === true;
  const warmup = health?.warmup;
  const warming = warmup?.stage === "loading_model";
  const warmupFailed = warmup?.stage === "failed";
  const llmOk = health?.llm.keyLoaded === true;

  return (
    <div
      className={cn(
        "rounded-xl border bg-card px-4 py-3",
        error ? "border-destructive/50" : "border-primary/30",
        className,
      )}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
        <span className="flex items-center gap-1.5 font-semibold">
          {error ? (
            <>
              <span className="size-2 rounded-full bg-destructive" />
              <span className="text-destructive">API offline — run ./start.sh</span>
            </>
          ) : workerOnline ? (
            <>
              <span className="relative flex size-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
                <span className="relative inline-flex size-2 rounded-full bg-primary" />
              </span>
              <span className="text-primary">Worker: LOCAL — running on this machine</span>
            </>
          ) : (
            <>
              <Loader2 className="size-3 animate-spin text-muted-foreground" />
              <span className="text-muted-foreground">Worker starting…</span>
            </>
          )}
        </span>

        {stats && (
          <>
            <span className="flex items-center gap-1 text-muted-foreground">
              <Cpu className="size-3.5" />
              load {stats.load1.toFixed(2)} / {stats.cores} cores
            </span>
            <span className="flex items-center gap-1 text-muted-foreground">
              <MemoryStick className="size-3.5" />
              RAM {fmtBytes(stats.memUsedBytes)} / {fmtBytes(stats.memTotalBytes)}
            </span>
          </>
        )}

        {health && (
          <>
            <span className="flex items-center gap-1 text-muted-foreground">
              <ServerCog className="size-3.5" />
              Whisper: {health.whisper.model} ({health.whisper.compute})
            </span>
            <span
              className={cn(
                "flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold",
                llmOk
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "border-amber-400/50 bg-amber-400/10 text-amber-300",
              )}
              title={llmOk ? "Gemini key loaded from .env" : "Add GEMINI_API_KEY to .env for AI detection"}
            >
              <HardDrive className="size-3" />
              {llmOk ? "AI detection ready" : "AI detection offline — add GEMINI_API_KEY to .env"}
            </span>
            {health.worker.queueDepth > 0 && (
              <span className="text-muted-foreground">
                queue: {health.worker.queueDepth}
              </span>
            )}
          </>
        )}
      </div>

      {warming && (
        <p className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
          <Loader2 className="size-3 animate-spin text-primary" />
          Downloading Whisper model (one-time, ~500MB-3GB)…
          {warmup?.detail && <span className="font-mono">{warmup.detail}</span>}
        </p>
      )}
      {warmupFailed && (
        <p className="mt-2 text-[11px] text-destructive">
          Whisper warmup failed: {warmup?.detail}
        </p>
      )}
    </div>
  );
}
