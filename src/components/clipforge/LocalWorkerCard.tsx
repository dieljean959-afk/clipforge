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
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { useHealth } from "@/hooks/use-poll";
import { api } from "@/lib/api";
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
  const llmOk = health?.llm.keyLoaded === true;
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

  // Live Gemini-key state from the server: where it found the key, or exactly
  // why a present-but-unusable value was rejected (empty save, encrypted blob,
  // placeholder, malformed).
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
      toast.error("That looks too short for a Google API key (they start with AIza and are ~39 chars).");
      return;
    }
    setSavingKey(true);
    try {
      const status = await api.setGeminiKey(raw);
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
      const status = await api.deleteGeminiKey();
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
            {diskFree !== null && (
              <StatLine
                icon={HardDrive}
                className={diskLow ? "text-amber-300" : undefined}
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
                  ? `Gemini key loaded on the server (${llmSource})`
                  : llmProblem ?? "No GEMINI_API_KEY on the server yet"
              }
            >
              {llmOk ? <HardDrive className="size-3" /> : <KeyRound className="size-3" />}
              {llmOk ? "AI detection ready" : llmProblem ? "AI detection blocked" : "AI detection offline — no key"}
            </span>
            {/* key manager: Set key / Re-check — always tappable, no hover-only UI */}
            <button
              type="button"
              onClick={openKeyDialog}
              className={cn(
                "flex cursor-pointer items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold transition",
                llmOk
                  ? "border-border text-muted-foreground hover:border-primary/50 hover:text-foreground"
                  : "border-amber-400/50 text-amber-300 hover:bg-amber-400/15",
              )}
              title="Set or replace the Gemini API key — saved directly into the server's env"
            >
              <Pencil className="size-3" />
              {llmOk ? "Replace key" : "Set key"}
            </button>
            {!llmOk && (
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
              title="Run environment diagnostics (ffmpeg build, GPU, ASR engine, Gemini key, disk)"
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
          {llmProblem ? (
            <>
              <span className="font-semibold">Why: </span>{llmProblem}{" "}
              Use <span className="font-semibold">Set key</span> to replace it right
              here — no Keys tab needed.
            </>
          ) : (
            <>
              Click <span className="font-semibold">Set key</span> above to paste your
              Gemini API key (get one free at aistudio.google.com/apikey) — it is saved
              directly on the server and AI detection turns on instantly.
            </>
          )}
        </p>
      )}

      {/* Gemini key manager dialog */}
      <Dialog open={keyDialogOpen} onOpenChange={setKeyDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <KeyRound className="size-4 text-primary" />
              Gemini API key
            </DialogTitle>
            <DialogDescription>
              {llmOk
                ? `A key is active (source: ${llmSource}). Saving replaces it; removing takes AI detection offline.`
                : llmProblem
                  ? `Current state: ${llmProblem}`
                  : "No key saved yet. Paste one below — it starts with AIza."}
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
