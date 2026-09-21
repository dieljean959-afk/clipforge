// WorkerPowerSwitch — the ON/OFF switch that controls the Oracle Cloud
// Always Free render VM (see src/convex/vm.ts).
//
// States (rendered from REAL OCI truth polled every 5s, never last-click):
//   OFF    — gray, "Worker asleep — flip the switch to wake it (~2 min)."
//   WAKING — pulsing amber, shown while OCI reports STARTING/STOPPING or
//            between the click and a fresh worker heartbeat
//   ON     — glowing green, only when OCI says RUNNING *and* the worker's
//            heartbeat is fresh
//   UNKNOWN— striped/hatched, shown only when the OCI API call fails. The
//            switch NEVER lies about state — the exact OCI error is shown.
//
// Setup: when OCI is not configured the switch is replaced by the first-run
// setup wizard (OciSetupWizard). "Skip for now" collapses it to a slim
// tap-to-set-up bar; a gear on the configured card reopens it for edits.

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { useAction } from "convex/react";
import { Loader2, Moon, Power, Settings2, Zap } from "lucide-react";
import { api } from "@/convex/_generated/api";
import { cn } from "@/lib/utils";
import { OciSetupWizard } from "@/components/clipforge/OciSetupWizard";

type VmStatus = {
  configured: boolean;
  configureError: string | null;
  wizardSkipped?: boolean;
  ociState: string | null;
  ociError: string | null;
  ociCheckedAt: number;
  workerOk: boolean;
  workerLastSeenAt: number | null;
  modelLoaded: boolean;
  activeJobs: number;
};

type PowerState = "off" | "waking" | "on" | "unknown";

function deriveState(vm: VmStatus | undefined, pending: "START" | "STOP" | null): PowerState {
  if (vm === undefined) return "unknown"; // still loading
  if (!vm.configured) return "unknown";
  if (vm.ociError !== null && vm.ociState === null) return "unknown";

  // A user click shows WAKING immediately; OCI truth then takes over.
  if (pending === "START") {
    if (vm.ociState === "RUNNING" && vm.workerOk) return "on";
    return "waking";
  }
  if (pending === "STOP") {
    return vm.ociState === "STOPPED" ? "off" : "waking";
  }

  switch (vm.ociState) {
    case "RUNNING":
      return vm.workerOk ? "on" : "waking";
    case "STARTING":
    case "STOPPING":
      return "waking";
    case "STOPPED":
    case "TERMINATED":
      return "off";
    default:
      return "unknown";
  }
}

export function WorkerPowerSwitch({ className }: { className?: string }) {
  const vmStatus = useAction(api.vm.vmStatus);
  const vmSetPower = useAction(api.vm.vmSetPower);
  const [vm, setVm] = useState<VmStatus | undefined>(undefined);
  const [pending, setPending] = useState<"START" | "STOP" | null>(null);
  const [confirmStop, setConfirmStop] = useState(false);
  // forceWizard = the gear was tapped on a configured card. When NOT
  // configured the wizard shows by default (unless previously skipped).
  const [forceWizard, setForceWizard] = useState(false);

  const refresh = useCallback(
    async (force = false) => {
      try {
        const s = (await vmStatus({ force })) as VmStatus;
        setVm(s);
      } catch {
        // leave the last truth in place; the poller retries in 5s
      }
    },
    [vmStatus],
  );

  // 5s poll — the toggle always reflects REAL instance state.
  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 5_000);
    return () => clearInterval(t);
  }, [refresh]);

  const wizardOpen =
    forceWizard || (vm !== undefined && !vm.configured && vm.wizardSkipped !== true);

  const state = deriveState(vm, pending);

  const flip = async (action: "START" | "STOP") => {
    setPending(action);
    try {
      const res = (await vmSetPower({ action })) as { busy?: boolean; activeJobs?: number };
      if (res.busy) {
        setConfirmStop(true);
        setPending(null);
        return;
      }
      toast.success(
        action === "START"
          ? "Starting the worker VM (~2 min)..."
          : "Stopping the worker VM — queued jobs stay safe in the queue.",
      );
      // poll faster right after a power action
      setTimeout(() => void refresh(true), 3_000);
      setTimeout(() => void refresh(true), 8_000);
    } catch (err) {
      setPending(null);
      toast.error(err instanceof Error ? err.message : "VM power action failed.");
    }
  };

  // ---- setup wizard (first run, skipped-resume, or gear edit) ----
  if (wizardOpen) {
    return (
      <OciSetupWizard
        className={className}
        onDone={() => {
          setForceWizard(false);
          void refresh(true);
        }}
      />
    );
  }

  // ---- not configured + skipped: slim tap-to-set-up bar (never a fake state) ----
  if (vm !== undefined && !vm.configured) {
    return (
      <button
        type="button"
        onClick={() => setForceWizard(true)}
        className={cn(
          "flex cursor-pointer items-center gap-2 rounded-xl border border-dashed border-border bg-card px-3 py-2 text-xs text-muted-foreground hover:border-primary/50 hover:text-foreground",
          className,
        )}
      >
        <Power className="size-3.5" />
        Worker VM not configured — tap to set up
      </button>
    );
  }

  const label =
    state === "on"
      ? "ON"
      : state === "waking"
        ? pending === "STOP"
          ? "STOPPING…"
          : "WAKING…"
        : state === "off"
          ? "OFF"
          : "—";

  const hint =
    state === "on"
      ? vm?.modelLoaded
        ? "Worker online — yt-dlp + ffmpeg ready"
        : "Worker online — loading the AI model…"
      : state === "waking"
        ? pending === "STOP"
          ? "Stopping — jobs stay queued and resume next wake"
          : "Waking the worker — this takes ~2 min"
        : state === "off"
          ? "Worker asleep — flip the switch to wake it (~2 min)."
          : vm === undefined
            ? "Checking instance state…"
            : vm.ociError
              ? `Instance state unknown — ${vm.ociError}`
              : "Instance state unknown";

  return (
    <div
      className={cn(
        "relative flex items-center gap-3 rounded-xl border px-3 py-2",
        state === "on"
          ? "border-primary/50 bg-primary/10 neon-glow-sm"
          : state === "waking"
            ? "border-amber-400/40 bg-amber-400/5"
            : state === "off"
              ? "border-border bg-card"
              : "border-dashed border-border bg-card",
        className,
      )}
    >
      {/* the switch itself */}
      <button
        type="button"
        role="switch"
        aria-checked={state === "on"}
        aria-label="Worker VM power"
        disabled={state === "unknown" || state === "waking"}
        onClick={() => {
          if (state === "on") {
            if ((vm?.activeJobs ?? 0) > 0) {
              setConfirmStop(true);
            } else {
              void flip("STOP");
            }
          } else if (state === "off") {
            void flip("START");
          }
        }}
        className={cn(
          "relative h-8 w-16 shrink-0 cursor-pointer rounded-full border transition-colors duration-300",
          state === "on" &&
            "border-primary/70 bg-primary/25 shadow-[0_0_18px_rgba(57,255,20,0.45)]",
          state === "waking" && "border-amber-400/60 bg-amber-400/20",
          state === "off" && "border-border bg-muted",
          state === "unknown" && "cursor-not-allowed border-border bg-muted/50",
        )}
      >
        <span
          className={cn(
            "absolute top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-full transition-all duration-300",
            state === "on" &&
              "left-8 bg-primary text-black shadow-[0_0_12px_rgba(57,255,20,0.8)]",
            state === "waking" && "left-8 animate-pulse bg-amber-300 text-black",
            state === "off" && "left-1 bg-muted-foreground/60 text-background",
            state === "unknown" &&
              "left-1 animate-pulse bg-muted-foreground/40 text-background",
          )}
        >
          {state === "on" ? (
            <Zap className="size-3.5" />
          ) : state === "waking" ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : state === "off" ? (
            <Moon className="size-3" />
          ) : (
            <Power className="size-3" />
          )}
        </span>
      </button>

      <div className="min-w-0">
        <p className="flex items-center gap-1.5 text-xs font-bold tracking-wide">
          <span
            className={cn(
              "size-2 rounded-full",
              state === "on" && "bg-primary shadow-[0_0_8px_rgba(57,255,20,0.9)]",
              state === "waking" && "animate-pulse bg-amber-300",
              state === "off" && "bg-muted-foreground/50",
              state === "unknown" && "bg-muted-foreground/30",
            )}
          />
          {label}
          {state === "waking" && pending === null && (
            <Loader2 className="size-3 animate-spin text-amber-300" />
          )}
        </p>
        <p className="truncate text-[11px] text-muted-foreground" title={hint}>
          {hint}
        </p>
      </div>

      {/* settings gear — reopen the wizard to edit credentials */}
      <button
        type="button"
        aria-label="Edit VM connection"
        title="Edit VM connection"
        onClick={() => setForceWizard(true)}
        className="ml-auto cursor-pointer rounded-md p-1.5 text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
      >
        <Settings2 className="size-3.5" />
      </button>

      {/* busy — stop after current job? */}
      {confirmStop && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-sm rounded-xl border border-border bg-card p-5">
            <p className="text-sm font-bold">Busy — stop after the current job?</p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              {vm?.activeJobs ?? 1} job(s) are rendering right now. A hard stop
              would kill the encode; choose “Stop after current job” and the VM
              shuts down cleanly once the worker drains its queue.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                className="cursor-pointer rounded-md border border-border px-3 py-1.5 text-xs"
                onClick={() => setConfirmStop(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="cursor-pointer rounded-md border border-amber-400/50 bg-amber-400/10 px-3 py-1.5 text-xs font-semibold text-amber-200"
                onClick={() => {
                  setConfirmStop(false);
                  void flip("STOP");
                }}
              >
                Stop after current job
              </button>
            </div>
          </div>
        </div>
      )}

      {/* cost hint */}
      <p className="pointer-events-none hidden text-[10px] text-muted-foreground/60 lg:block">
        Oracle Always Free: ~$0/mo while under free limits.
      </p>
    </div>
  );
}
