// Worker status card — shows the REAL state of the hosted backend process:
// API online/offline (precise reason), server CPU/RAM via psutil on the
// server, tailnet IP when Tailscale is enabled, queue depth, warmup, and
// limits. All fields are null-safe: unknown never renders as NaN.

import { useState } from "react";
import { toast } from "sonner";
import {
  Cpu,
  HardDrive,
  KeyRound,
  Loader2,
  MemoryStick,
  RefreshCw,
  ServerCog,
  ShieldCheck,
} from "lucide-react";
import { useHealth } from "@/hooks/use-poll";
import { api } from "@/lib/api";
import { fmtBytes } from "@/lib/clipforge";
import { cn } from "@/lib/utils";

function StatLine({
  icon: Icon,
  children,
  className,
}: {
  icon: typeof Cpu;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span className={cn("flex items-center gap-1 text-muted-foreground", className)}>
      <Icon className="size-3.5 shrink-0" />
      {children}
    </span>
  );
}

export function LocalWorkerCard({ className }: { className?: string }) {
  const { health, error, refresh: refreshHealth } = useHealth(5000);
  const [rechecking, setRechecking] = useState(false);

  const workerOnline = health?.worker.online === true;
  const warmup = health?.warmup;
  const warming = warmup?.stage === "loading_model";
  const warmupFailed = warmup?.stage === "failed";
  const llmOk = health?.llm.keyLoaded === true;

  // null-safe system stats — the server sends null for anything it can't read
  const sys = health?.system ?? null;
  const cpu = typeof sys?.cpuPercent === "number" ? Math.round(sys.cpuPercent) : null;
  const cores = typeof sys?.cores === "number" ? sys.cores : null;
  const memUsed = typeof sys?.memUsedBytes === "number" ? sys.memUsedBytes : null;
  const memTotal = typeof sys?.memTotalBytes === "number" ? sys.memTotalBytes : null;
  const load1 = typeof sys?.load1 === "number" ? sys.load1 : null;

  const dispatch = health?.dispatch;
  const remoteOnline =
    dispatch?.enabled === true && dispatch.remote?.online === true;
  const remoteDetail =
    dispatch?.enabled === true
      ? (dispatch.remote?.detail ?? "no answer")
      : "no answer";
  const maxMin = typeof health?.limits?.maxSourceMin === "number" ? health.limits.maxSourceMin : null;
  const retentionH = typeof health?.limits?.retentionHours === "number" ? health.limits.retentionHours : null;

  // Live Gemini-key state from the server: where it found the key, or exactly
  // why a present-but-unusable value was rejected (encrypted blob, placeholder,
  // malformed). "Re-check" forces the server to re-read the env files now —
  // the natural action right after saving a key in the Keys/API keys tab.
  const llm = health?.llm;
  const llmProblem = llm?.problem ?? null;
  const llmSource = typeof llm?.source === "string" ? llm.source : "";

  const runRecheck = async () => {
    setRechecking(true);
    try {
      const status = await api.reloadSecrets();
      if (status.keyLoaded) {
        toast.success(`Gemini key detected (${status.source}) — AI detection is live.`);
      } else if (status.problem) {
        toast.error(status.problem);
      } else {
        toast.error(
          "No key found. Add GEMINI_API_KEY in the Keys/API keys tab (Backend section), then Re-check again.",
        );
      }
      void refreshHealth();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Re-check failed — is the API up?");
    } finally {
      setRechecking(false);
    }
  };

  return (
    <div
      className={cn(
        "rounded-xl border bg-card px-4 py-3",
        error ? "border-destructive/50" : "border-primary/30",
        className,
      )}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
        {error ? (
          <span className="flex items-center gap-1.5 font-semibold text-destructive">
            <span className="size-2 rounded-full bg-destructive" />
            API unreachable: {error}
          </span>
        ) : health === undefined ? (
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            Checking server status…
          </span>
        ) : (
          <span className="flex items-center gap-1.5 font-semibold">
            <span className="relative flex size-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
              <span className="relative inline-flex size-2 rounded-full bg-primary" />
            </span>
            <span className="text-primary">API online</span>
            <span className="font-normal text-muted-foreground">
              — pipeline runs on ClipForge's server
            </span>
          </span>
        )}

        {/* real server stats, null-safe */}
        {health && (
          <>
            {cpu !== null && (
              <StatLine icon={Cpu}>
                CPU {cpu}%{cores !== null ? ` · ${cores} cores` : ""}
              </StatLine>
            )}
            {cpu === null && load1 !== null && (
              <StatLine icon={Cpu}>load {load1.toFixed(2)}</StatLine>
            )}
            {memUsed !== null && memTotal !== null && memTotal > 0 && (
              <StatLine icon={MemoryStick}>
                RAM {fmtBytes(memUsed)} / {fmtBytes(memTotal)}
              </StatLine>
            )}
            {dispatch?.enabled === true && (
              <StatLine icon={ServerCog}>
                {remoteOnline ? (
                  <span className="text-primary">home worker online</span>
                ) : (
                  <span className="text-amber-300">
                    home worker unreachable ({remoteDetail})
                  </span>
                )}
              </StatLine>
            )}
            {health.tailnetIp && (
              <StatLine icon={ShieldCheck} className="font-mono text-[10px]">
                tailnet {health.tailnetIp}
              </StatLine>
            )}
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
              title={
                llmOk
                  ? `Gemini key loaded on the server (${llmSource})`
                  : llmProblem ?? "Add GEMINI_API_KEY in the Keys/API keys tab (Backend section) for AI detection"
              }
            >
              {llmOk ? <HardDrive className="size-3" /> : <KeyRound className="size-3" />}
              {llmOk ? "AI detection ready" : llmProblem ? "AI detection blocked" : "AI detection offline — add GEMINI_API_KEY"}
            </span>
            {!llmOk && health && (
              <button
                type="button"
                onClick={() => void runRecheck()}
                disabled={rechecking}
                className={cn(
                  "flex cursor-pointer items-center gap-1 rounded-full border border-amber-400/50 px-2 py-0.5 text-[10px] font-bold text-amber-300 transition hover:bg-amber-400/15",
                  rechecking && "cursor-wait opacity-60",
                )}
                title="Re-read the server's env files right now — press this after saving a key"
              >
                {rechecking ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
                Re-check
              </button>
            )}
            {maxMin !== null && (
              <span className="text-muted-foreground">
                max video: {maxMin} min · files deleted after {retentionH ?? "?"}h
              </span>
            )}
            {health.worker.queueDepth > 0 && (
              <span className="text-muted-foreground">queue: {health.worker.queueDepth}</span>
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
      {warmup?.stage === "skipped" && (
        <p className="mt-2 text-[11px] text-muted-foreground">
          {warmup.detail}
        </p>
      )}
      {!llmOk && health && (
        <p className="mt-2 max-w-3xl text-[11px] leading-relaxed text-amber-300/90">
          {llmProblem ? (
            <>
              <span className="font-semibold">Why: </span>{llmProblem}
            </>
          ) : (
            <>
              Save <span className="font-mono">GEMINI_API_KEY</span> in the Keys/API keys
              tab (Backend section) or the project <span className="font-mono">.env</span>,
              then press <span className="font-semibold">Re-check</span> — no restart
              needed. Searched: {llmSource}.
            </>
          )}
        </p>
      )}
    </div>
  );
}
