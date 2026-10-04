// Worker status card — shows the REAL state of the hosted backend process:
// API online/offline (precise reason), server CPU/RAM via psutil on the
// server, tailnet IP when Tailscale is enabled, queue depth, warmup, and
// limits. All fields are null-safe: unknown never renders as NaN.
//
// Includes an in-app Gemini key manager (Set key / Remove) — the platform's
// Keys/API keys page has no Edit/Delete for existing variables, so the app
// writes .env.local directly through the backend. Works on mobile widths.

import { useState } from "react";
import { toast } from "sonner";
import { Stethoscope } from "lucide-react";
import {
  Cpu,
  Eye,
  EyeOff,
  HardDrive,
  KeyRound,
  Loader2,
  MemoryStick,
  Pencil,
  RefreshCw,
  ServerCog,
  ShieldCheck,
  Sparkles,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { useHealth } from "@/hooks/use-poll";
import { SocialAccounts } from "./SocialAccounts";
import { api, type DetectionProvider, type KeyStatusDto } from "@/lib/api";
import { fmtBytes } from "@/lib/clipforge";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

function StatLine({
  icon: Icon,
  children,
  className,
  title,
}: {
  icon: typeof Cpu;
  children: React.ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span
      className={cn("flex items-center gap-1 text-muted-foreground", className)}
      title={title}
    >
      <Icon className="size-3.5 shrink-0" />
      {children}
    </span>
  );
}

export function LocalWorkerCard({ className }: { className?: string }) {
  const { health, error, stale, refresh: refreshHealth } = useHealth(5000);
  const [rechecking, setRechecking] = useState(false);
  const [keyDialogOpen, setKeyDialogOpen] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [savingKey, setSavingKey] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [runningDoctor, setRunningDoctor] = useState(false);

  // doctor: replaces vague "API unreachable" with specific, fixable checks
  const runDoctor = async () => {
    setRunningDoctor(true);
    try {
      const report = await api.doctor();
      const bad = report.checks.filter((c) => c.status === "fail");
      const warn = report.checks.filter((c) => c.status === "warn");
      if (bad.length === 0) {
        toast.success(
          `Diagnostics passed${warn.length > 0 ? ` (${warn.length} warning${warn.length > 1 ? "s" : ""})` : ""}.`,
          { description: warn.map((c) => c.name).join(", ") || "All systems nominal." },
        );
      } else {
        toast.error(`${bad.length} diagnostic failure${bad.length > 1 ? "s" : ""} found.`, {
          description: bad
            .map((c) => `${c.name}: ${c.detail}${c.fix ? ` — ${c.fix}` : ""}`)
            .join(" · "),
          duration: 12_000,
        });
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Diagnostics could not run.");
    } finally {
      setRunningDoctor(false);
    }
  };

  const workerOnline = health?.worker.online === true;
  const warmup = health?.warmup;
  const warming = warmup?.stage === "loading_model";
  const warmupFailed = warmup?.stage === "failed";
  // The host-limit downgrade is invisible unless the label says so: the server
  // may load `small` while config.toml still asks for `medium`.
  const downgraded =
    !!health?.whisper.configuredModel &&
    health.whisper.configuredModel !== health.whisper.model;

  // null-safe system stats — the server sends null for anything it can't read
  const sys = health?.system ?? null;
  const cpu = typeof sys?.cpuPercent === "number" ? Math.round(sys.cpuPercent) : null;
  const cores = typeof sys?.cores === "number" ? sys.cores : null;
  const memUsed = typeof sys?.memUsedBytes === "number" ? sys.memUsedBytes : null;
  const memTotal = typeof sys?.memTotalBytes === "number" ? sys.memTotalBytes : null;
  const load1 = typeof sys?.load1 === "number" ? sys.load1 : null;
  const diskFree =
    typeof health?.disk?.freeBytes === "number" ? health.disk.freeBytes : null;
  const diskTotal =
    typeof health?.disk?.totalBytes === "number" ? health.disk.totalBytes : null;
  // WHICH volume these numbers describe. Once CLIPFORGE_DATA_DIR moves job
  // artefacts to a bigger mount, the sandbox root is irrelevant — saying so
  // keeps the reading honest instead of implying the 4 GB sandbox is full.
  const diskPath = health?.disk?.path ?? null;
  const diskLow = diskFree !== null && diskFree < 1.5 * 1024 ** 3;

  const dispatch = health?.dispatch;
  const remoteOnline =
    dispatch?.enabled === true && dispatch.remote?.online === true;
  const remoteDetail =
    dispatch?.enabled === true
      ? (dispatch.remote?.detail ?? "no answer")
      : "no answer";
  const maxMin = typeof health?.limits?.maxSourceMin === "number" ? health.limits.maxSourceMin : null;
  const retentionH = typeof health?.limits?.retentionHours === "number" ? health.limits.retentionHours : null;

  // AI highlight-DETECTION backend (DeepSeek = hosted default, NVIDIA/Groq =
  // hosted alternatives, Gemini = alternative, Ollama = local). Separate from
  // the ASR engine on purpose: switching this never touches faster-whisper,
  // and the UI says so.
  const detection = health?.detection;

  // Live key state for the ACTIVE detection backend. DeepSeek is the default,
  // so this must follow the selected provider — showing Groq's key while
  // DeepSeek is the one actually making calls would be a lie.
  const activeProvider = detection?.provider ?? "deepseek";
  const KEY_STATUS: Partial<Record<DetectionProvider, KeyStatusDto | undefined>> = {
    deepseek: health?.llmDeepSeek,
    nvidia: health?.llmNvidia,
    groq: health?.llmGroq,
    gemini: health?.llm,
  };
  // No fallback to another vendor's state. Ollama has no key at all, so the
  // old `?? health?.llmGroq` made an unreachable local server report GROQ's key
  // problem ("encrypted dotenvx…", "no key found") and told the user to paste a
  // Groq key for a backend that needs none — contradicting the dialog two
  // lines below, which correctly says Ollama needs no key.
  const llm = KEY_STATUS[activeProvider];
  const isOllama = activeProvider === "ollama";
  // For Ollama the honest "problem" is the server's reachability, not a key.
  const ollamaDetail =
    typeof detection?.detail === "string" ? detection.detail : "Ollama is not responding.";
  const llmProblem = isOllama ? null : (llm?.problem ?? null);
  const llmSource = typeof llm?.source === "string" ? llm.source : "";
  // Per-vendor key facts. Kept as one table rather than nested ternaries so a
  // new provider is one row here instead of a new branch in five places.
  const VENDOR: Record<string, { env: string; prefix: string; name: string; keysUrl: string }> = {
    deepseek: {
      env: "DEEPSEEK_API_KEY",
      prefix: "sk-",
      name: "DeepSeek",
      keysUrl: "platform.deepseek.com/api_keys",
    },
    nvidia: {
      env: "NVIDIA_API_KEY",
      prefix: "nvapi-",
      name: "NVIDIA",
      keysUrl: "build.nvidia.com",
    },
    groq: {
      env: "GROQ_API_KEY",
      prefix: "gsk_",
      name: "Groq",
      keysUrl: "console.groq.com/keys",
    },
    gemini: {
      env: "GEMINI_API_KEY",
      prefix: "AIza",
      name: "Google",
      keysUrl: "console.gemini.google.com/apikey",
    },
  };
  const vendor = VENDOR[activeProvider];
  // With no key to manage there is no vendor to name: these stay blank rather
  // than borrowing DeepSeek's, so no prompt can ever say "DeepSeek API key"
  // while the selected backend is Ollama.
  const llmEnvName = vendor?.env ?? "";
  const llmKeyPrefix = vendor?.prefix ?? "";
  const llmVendor = vendor?.name ?? "Ollama";
  const llmKeyUrl = vendor?.keysUrl ?? "";
  // Ollama needs no key: the local server's reachability IS the check.
  const llmOk = isOllama ? detection?.ready === true : llm?.keyLoaded === true;
  const [switchingBackend, setSwitchingBackend] = useState(false);
  const switchBackend = async (provider: DetectionProvider) => {
    if (!detection || provider === detection.provider) return;
    setSwitchingBackend(true);
    try {
      await api.setDetectionBackend(provider);
      toast.success(
        provider === "ollama"
          ? "AI detection switched to local Ollama — free and private, but slower (it shares the CPU with Whisper)."
          : provider === "deepseek"
            ? "AI detection switched to DeepSeek — OpenAI-compatible, billed against a balance instead of a fixed daily token ceiling, and the cheapest of the hosted options."
            : provider === "nvidia"
              ? "AI detection switched to NVIDIA — same OpenAI-compatible path, hosting its own deepseek-* models on an NVIDIA quota."
              : provider === "groq"
                ? "AI detection switched to Groq — the backup. ~30 req/min and a 200k/day ceiling on the free tier; calls are paced below that."
                : "AI detection switched to Gemini — a hosted alternative when the primary's quota is spent.",
      );
      void refreshHealth();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not switch the detection backend.");
    } finally {
      setSwitchingBackend(false);
    }
  };

  const runRecheck = async () => {
    setRechecking(true);
    try {
      const status = await api.reloadSecrets();
      const active =
        activeProvider === "gemini"
          ? status
          : activeProvider === "deepseek"
            ? status.deepseek
            : activeProvider === "nvidia"
              ? status.nvidia
              : status.groq;
      if (active?.keyLoaded) {
        toast.success(`${llmVendor} key detected (${active.source}) — AI detection is live.`);
      } else if (active?.problem) {
        toast.error(active.problem);
      } else {
        toast.error("No key found — use 'Set key' to save one right here.");
      }
      void refreshHealth();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Re-check failed — is the API up?");
    } finally {
      setRechecking(false);
    }
  };

  const openKeyDialog = () => {
    setKeyInput("");
    setShowKey(false);
    setConfirmDelete(false);
    setKeyDialogOpen(true);
  };

  const saveKey = async () => {
    const raw = keyInput.trim();
    if (raw.length < 20) {
      toast.error(
        `That looks too short for a ${llmVendor} API key (they start with ${llmKeyPrefix}).`,
      );
      return;
    }
    setSavingKey(true);
    try {
      const status =
        activeProvider === "gemini"
          ? await api.setGeminiKey(raw)
          : activeProvider === "deepseek"
            ? await api.setDeepseekKey(raw)
            : activeProvider === "nvidia"
              ? await api.setNvidiaKey(raw)
              : await api.setGroqKey(raw);
      toast.success(
        `Key saved and live. Verify it: starts ${status.storedShape.first6}, ends ${status.storedShape.last4} (${status.storedShape.length} chars).`,
      );
      setKeyDialogOpen(false);
      void refreshHealth();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not save the key.");
    } finally {
      setSavingKey(false);
    }
  };

  const removeKey = async () => {
    setSavingKey(true);
    try {
      const status =
        activeProvider === "gemini"
          ? await api.deleteGeminiKey()
          : activeProvider === "deepseek"
            ? await api.deleteDeepseekKey()
            : activeProvider === "nvidia"
              ? await api.deleteNvidiaKey()
              : await api.deleteGroqKey();
      toast.info(`Key removed (${status.linesRemoved} line(s) cleaned). AI detection is offline until a new key is set.`);
      setKeyDialogOpen(false);
      void refreshHealth();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not delete the key.");
    } finally {
      setSavingKey(false);
    }
  };

  return (
    <div
      className={cn(
        // min-w-0: as a grid/flex child this card must be allowed to shrink
        // below its content's intrinsic width, or a long unbroken string (a
        // URL, an error code) pushes the whole page wide on a 360px phone.
        "min-w-0 rounded-xl border bg-card px-4 py-3",
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
        ) : stale ? (
          // A poll failed but the readings below are real and recent. Saying
          // "API unreachable" here was wrong twice over: the API demonstrably
          // answered, and it erased a fully populated status card.
          <span
            className="flex items-center gap-1.5 text-amber-300/90"
            title={`Last poll failed: ${stale}. The values below are from the most recent successful read.`}
          >
            <span className="size-2 rounded-full bg-amber-400/70" />
            last check failed ({stale}) — showing the most recent reading
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
            {diskFree !== null && (
              <StatLine
                icon={HardDrive}
                className={diskLow ? "text-amber-300" : undefined}
                title={
                  diskPath
                    ? `Measured on ${diskPath} — the volume jobs write to`
                    : "Measured volume unknown"
                }
              >
                {diskLow && <TriangleAlert className="size-3" />}
                Disk {fmtBytes(diskFree)} free
                {diskTotal !== null ? ` / ${fmtBytes(diskTotal)}` : ""}
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
            {/* AI detection backend: the tradeoff is stated in full, not hidden
                behind a tooltip, so nobody picks Ollama expecting hosted speed. */}
            {detection && (
              <span className="flex flex-wrap items-center gap-1.5 text-muted-foreground">
                <Sparkles className="size-3.5" />
                <span className="rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-medium text-foreground">
                  AI detection: {detection.label}
                </span>
                <span
                  className={cn(
                    "text-[10px]",
                    detection.provider === "ollama" ? "text-amber-300" : "text-muted-foreground",
                  )}
                >
                  {detection.provider === "ollama"
                    ? "fully local · free & private, but slower (shares the CPU with Whisper)"
                    : "hosted · fastest, overlaps local CPU work"}
                </span>
                <span className="flex items-center gap-1">
                  {detection.options.map((o) => (
                    <button
                      key={o.value}
                      type="button"
                      disabled={switchingBackend}
                      onClick={() => switchBackend(o.value)}
                      title={o.tradeoff}
                      className={cn(
                        "rounded border px-1.5 py-0.5 text-[10px] font-medium transition-colors",
                        o.value === detection.provider
                          ? "border-primary/50 bg-primary/15 text-primary"
                          : "border-border text-muted-foreground hover:border-border hover:bg-muted hover:text-foreground",
                        switchingBackend && "opacity-60",
                      )}
                    >
                      {o.label}
                    </button>
                  ))}
                </span>
                {detection.provider === "ollama" && detection.ready === false && (
                  <span className="text-[10px] text-destructive">
                    {detection.detail ?? "local model not ready"}
                  </span>
                )}
                {detection.transcriptionEngine && (
                  <span
                    className="text-[10px] text-muted-foreground"
                    title="Switching the detection backend never changes the transcription engine."
                  >
                    transcription stays {detection.transcriptionEngine}
                  </span>
                )}
              </span>
            )}

            <span
              className="flex items-center gap-1 text-muted-foreground"
              title={
                health.asr?.detail ??
                `${health.whisper.engine ?? "engine"}: ${health.whisper.model} on ${health.whisper.compute}`
              }
            >
              <ServerCog className="size-3.5" />
              {health.asr?.engine ? `${health.asr.engine} · ` : ""}
              {health.whisper.model} ({health.whisper.compute})
              {downgraded && (
                <span className="text-amber-300/80">
                  (host limit, configured {health.whisper.configuredModel})
                </span>
              )}
              {typeof health.asr?.estimatedRtf === "number" &&
                ` · RTF ${health.asr.estimatedRtf.toFixed(3)}`}
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
                  ? isOllama
                    ? ollamaDetail
                    : `${llmVendor} key loaded on the server (${llmSource})`
                  : (llmProblem ?? (isOllama ? ollamaDetail : `No ${llmEnvName} on the server yet`))
              }
            >
              {llmOk ? <HardDrive className="size-3" /> : <KeyRound className="size-3" />}
              {llmOk
                ? "AI detection ready"
                : isOllama
                  ? "AI detection offline — Ollama unreachable"
                  : llmProblem
                    ? "AI detection blocked"
                    : "AI detection offline — no key"}
            </span>
            {/* key manager: Set key / Re-check — always tappable, no hover-only UI.
                Hidden for Ollama: there is no key to set, and Re-check only
                re-reads the env files, so both would be dead controls. */}
            {!isOllama && (
            <button
              type="button"
              onClick={openKeyDialog}
              className={cn(
                "flex cursor-pointer items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold transition",
                llmOk
                  ? "border-border text-muted-foreground hover:border-primary/50 hover:text-foreground"
                  : "border-amber-400/50 text-amber-300 hover:bg-amber-400/15",
              )}
              title={`Set or replace the ${llmVendor} API key — saved directly into the server's env`}
            >
              <Pencil className="size-3" />
              {llmOk ? "Replace key" : "Set key"}
            </button>
            )}
            {!llmOk && !isOllama && (
              <button
                type="button"
                onClick={() => void runRecheck()}
                disabled={rechecking}
                className={cn(
                  "flex cursor-pointer items-center gap-1 rounded-full border border-amber-400/50 px-2 py-0.5 text-[10px] font-bold text-amber-300 transition hover:bg-amber-400/15",
                  rechecking && "cursor-wait opacity-60",
                )}
                title="Re-read the server's env files right now"
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
            <button
              type="button"
              onClick={() => void runDoctor()}
              disabled={runningDoctor}
              className={cn(
                "flex cursor-pointer items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold text-muted-foreground transition hover:border-primary/50 hover:text-foreground",
                runningDoctor && "cursor-wait opacity-60",
              )}
              title="Run environment diagnostics (ffmpeg build, GPU, ASR engine, LLM key, disk)"
            >
              {runningDoctor ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Stethoscope className="size-3" />
              )}
              {runningDoctor ? "Checking…" : "Doctor"}
            </button>
            <span
              className="text-muted-foreground"
              title="Each slot runs its own child process, so a hung job cannot block the queue"
            >
              {health.worker.concurrency ?? 1} worker
              {(health.worker.concurrency ?? 1) > 1 ? "s" : ""} ·{" "}
              {health.worker.activeCount ?? 0} running
              {health.worker.queueDepth > 0 ? ` · ${health.worker.queueDepth} queued` : ""}
            </span>
          </>
        )}
      </div>

      {warming && (
        <p className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
          <Loader2 className="size-3 animate-spin text-primary" />
          Loading the {health?.asr?.engine ?? "transcription"} model (one-time,
          ~500MB-3GB)…
          {warmup?.detail && <span className="font-mono">{warmup.detail}</span>}
        </p>
      )}
      {warmupFailed && (
        <p className="mt-2 text-[11px] text-destructive">
          {health?.asr?.engine ?? "ASR"} warmup failed: {warmup?.detail}
        </p>
      )}
      {warmup?.stage === "skipped" && (
        <p className="mt-2 text-[11px] text-muted-foreground">
          {warmup.detail}
        </p>
      )}
      {!llmOk && health && (
        <p className="mt-2 max-w-3xl text-[11px] leading-relaxed text-amber-300/90">
          {isOllama ? (
            <>
              <span className="font-semibold">Why: </span>{ollamaDetail} Detection
              re-runs locally once the server is up — no API key is involved.
            </>
          ) : llmProblem ? (
            <>
              <span className="font-semibold">Why: </span>{llmProblem}{" "}
              Use <span className="font-semibold">Set key</span> to replace it right
              here — no Keys tab needed.
            </>
          ) : (
            <>
              Click <span className="font-semibold">Set key</span> above to paste your
              {` ${llmVendor} API key`} (get one free at {llmKeyUrl}) — it is saved
              directly on the server and AI detection turns on instantly.
            </>
          )}
        </p>
      )}

      {/* Connected social accounts. Purely additive: the download/export paths
          above are untouched, so hiding this entirely changes nothing else. */}
      <div className="mt-3 border-t border-border pt-3">
        <SocialAccounts />
      </div>

      {/* LLM key manager dialog (follows the active detection backend) */}
      <Dialog open={keyDialogOpen} onOpenChange={setKeyDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <KeyRound className="size-4 text-primary" />
              {`${llmVendor} API key`}
            </DialogTitle>
            <DialogDescription>
              {activeProvider === "ollama"
                ? "Ollama runs locally and needs no API key. Switch the detection backend to deepseek, nvidia or groq to use a hosted key here."
                : llmOk
                  ? `A key is active (source: ${llmSource}). Saving replaces it; removing takes AI detection offline.`
                  : llmProblem
                    ? `Current state: ${llmProblem}`
                    : `No key saved yet. Paste one below — it starts with ${llmKeyPrefix}.`}
            </DialogDescription>
          </DialogHeader>

          <div className="relative">
            <input
              type={showKey ? "text" : "password"}
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              placeholder="AIza…"
              autoComplete="off"
              spellCheck={false}
              className="h-11 w-full rounded-md border border-input bg-background px-3 pr-10 font-mono text-sm"
            />
            <button
              type="button"
              onClick={() => setShowKey((v) => !v)}
              className="absolute right-2 top-1/2 -translate-y-1/2 cursor-pointer rounded p-1.5 text-muted-foreground transition hover:text-foreground"
              title={showKey ? "Hide" : "Show"}
            >
              {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
            </button>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Saved directly into the server's env (no restart needed). Free key:{" "}
            <a
              href="https://aistudio.google.com/apikey"
              target="_blank"
              rel="noreferrer"
              className="text-primary underline"
            >
              aistudio.google.com/apikey
            </a>
          </p>

          <DialogFooter className="flex-row flex-wrap items-center gap-2">
            <Button
              className="cursor-pointer gap-1.5"
              disabled={savingKey || keyInput.trim().length < 20}
              onClick={() => void saveKey()}
            >
              {savingKey ? <Loader2 className="size-4 animate-spin" /> : <Pencil className="size-4" />}
              Save key
            </Button>
            <Button
              variant="outline"
              className="cursor-pointer gap-1.5"
              disabled={savingKey}
              onClick={() => void runRecheck()}
            >
              <RefreshCw className="size-4" />
              Re-check
            </Button>
            <Button
              variant="destructive"
              className="cursor-pointer gap-1.5"
              disabled={savingKey || !confirmDelete}
              onClick={() => void removeKey()}
            >
              <Trash2 className="size-4" />
              {confirmDelete ? "Confirm delete" : "Delete"}
            </Button>
            {!confirmDelete && (
              <Button
                variant="ghost"
                className="cursor-pointer text-xs text-muted-foreground"
                disabled={savingKey}
                onClick={() => setConfirmDelete(true)}
              >
                I want to delete the key
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
