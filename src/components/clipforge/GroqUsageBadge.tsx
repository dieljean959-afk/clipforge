import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Gauge } from "lucide-react";
import { api, type UsageDto } from "@/lib/api";
import { cn } from "@/lib/utils";

/** How often the badge re-reads the counter while the tab is open. */
const POLL_MS = 30_000;

function compact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}k`;
  return String(n);
}

/** "5h 12m" / "12m 05s" — no days, because a daily window never needs one. */
export function formatCountdown(sec: number | null): string | null {
  if (sec == null || sec < 0) return null;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(sec % 60).padStart(2, "0")}s`;
  return `${sec}s`;
}

/**
 * Colour bands. Green is "plenty of room", red is "a long source will die at
 * chunk 18". The thresholds sit at the points where the badge starts being
 * worth acting on, not at round numbers.
 */
function bandFor(pct: number): { bar: string; text: string; label: string } {
  if (pct >= 85)
    return {
      bar: "bg-red-500",
      text: "text-red-600 dark:text-red-400",
      label: "Nearly spent",
    };
  if (pct >= 60)
    return {
      bar: "bg-amber-500",
      text: "text-amber-600 dark:text-amber-400",
      label: "Over halfway",
    };
  return { bar: "bg-emerald-500", text: "text-muted-foreground", label: "Plenty left" };
}

/** "openai/gpt-oss-120b" -> "GPT-OSS 120B"; unknown ids are left alone. */
export function prettyModel(id: string): string {
  const tail = id.split("/").pop() ?? id;
  if (tail.startsWith("gpt-oss-")) return `GPT-OSS ${tail.slice("gpt-oss-".length).toUpperCase()}`;
  if (tail.startsWith("qwen")) return `Qwen ${tail.replace(/^qwen[.-]?/i, "")}`.trim();
  return tail;
}

export function GroqUsageBadge({
  onUsage,
  className,
}: {
  /** Called after every successful poll, so the submit form can warn on stale data. */
  onUsage?: (u: UsageDto) => void;
  className?: string;
}) {
  const [usage, setUsage] = useState<UsageDto | null>(null);
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const load = useCallback(async () => {
    try {
      const u = await api.usage();
      setUsage(u);
      onUsage?.(u);
    } catch {
      // A missing route or a backend restart must not put an error toast in
      // the user's face every 30s. The badge simply stays empty.
    }
  }, [onUsage]);

  useEffect(() => {
    void load();
    const id = window.setInterval(() => {
      // Don't poll a tab nobody is looking at.
      if (document.visibilityState === "visible") void load();
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [load]);

  // Tapping outside closes the breakdown; a pinned sheet on mobile would fight
  // the sticky header for space.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!usage) return null;

  const pct = Math.max(0, Math.min(100, usage.percentUsed));
  const band = bandFor(pct);
  const countdown = formatCountdown(usage.refillSec);
  // The backend reports which provider's budget this is. Do not hardcode a
  // vendor name here: the same badge describes NVIDIA's allowance and, after a
  // failover, Groq's, and a badge that says the wrong one is worse than none.
  const vendor = usage.providerLabel ?? (usage.provider === "groq" ? "Groq" : "NVIDIA");

  // No daily token budget to meter (local Ollama, or Gemini's key-based
  // billing). Rendering a meter here meant showing a confident "0 / 200,000"
  // against a denominator invented from an unrelated provider, so say what is
  // actually true instead of drawing a bar.
  if (usage.budgeted === false) {
    return (
      <div
        className={cn(
          "flex min-w-0 flex-col gap-0.5 rounded-md border border-border/60 bg-card/60 px-2 py-1",
          className,
        )}
        title={
          usage.provider === "ollama"
            ? "Ollama runs on this machine — no API key, no token budget"
            : `${vendor} bills per token against your key; there is no daily token ceiling to meter`
        }
      >
        <span className="flex items-center gap-1.5 text-[11px] leading-none">
          <Gauge className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
          <span className="truncate font-medium">{vendor}</span>
        </span>
        <span className="truncate text-[10px] leading-none text-muted-foreground">
          {usage.provider === "ollama" ? "Local · no token budget" : "No daily token budget"}
        </span>
      </div>
    );
  }

  return (
    <div ref={wrapRef} className={cn("relative", className)}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title={`${usage.used.toLocaleString()} of ${usage.limit.toLocaleString()} tokens used today`}
        className={cn(
          "flex w-full min-w-0 cursor-pointer flex-col gap-1 rounded-md border border-border/60 bg-card/60 px-2 py-1 text-left transition-colors",
          "hover:border-border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        )}
      >
        <span className="flex items-center gap-1.5 text-[11px] leading-none">
          <Gauge className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
          <span className="font-medium">{vendor}</span>
          <span className={cn("font-mono tabular-nums", band.text)}>
            {compact(usage.used)} / {compact(usage.limit)}
          </span>
          <ChevronDown
            className={cn(
              "ml-auto h-3 w-3 shrink-0 text-muted-foreground transition-transform",
              open && "rotate-180",
            )}
            aria-hidden
          />
        </span>

        <span
          className="h-1 w-full overflow-hidden rounded-full bg-muted"
          role="progressbar"
          aria-valuenow={Math.round(pct)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${vendor} daily token usage`}
        >
          <span
            className={cn("block h-full rounded-full transition-[width] duration-500", band.bar)}
            style={{ width: `${Math.max(2, pct)}%` }}
          />
        </span>

        {countdown && (
          <span className="text-[10px] leading-none text-muted-foreground">
            Refills in {countdown}
          </span>
        )}
      </button>

      {open && (
        <div
          className={cn(
            "absolute right-0 z-50 mt-1.5 w-64 rounded-lg border border-border bg-popover p-3 text-xs shadow-lg",
            "text-popover-foreground",
          )}
        >
          <p className="font-medium">
            {usage.used.toLocaleString()} / {usage.limit.toLocaleString()} tokens today
          </p>
          <p className="mt-0.5 text-muted-foreground">
            {pct.toFixed(1)}% used &middot; {band.label} &middot;{" "}
            {usage.remaining.toLocaleString()} left
          </p>

          <dl className="mt-2 space-y-0.5 text-muted-foreground">
            <div className="flex justify-between gap-2">
              <dt>Source</dt>
              <dd className="text-right">
                {usage.source === "provider-report" ? `${vendor} 429 report` : "local tally"}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>Calls today</dt>
              <dd className="text-right tabular-nums">{usage.calls}</dd>
            </div>
            {countdown && (
              <div className="flex justify-between gap-2">
                <dt>Refills in</dt>
                <dd className="text-right tabular-nums">{countdown}</dd>
              </div>
            )}
          </dl>

          {usage.models.length > 0 && (
            <>
              <p className="mt-3 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                Per model
              </p>
              <ul className="mt-1 space-y-1">
                {usage.models.map((m) => (
                  <li key={m.model} className="flex items-baseline gap-2">
                    <span
                      className={cn(
                        "truncate",
                        m.active ? "font-medium text-foreground" : "text-muted-foreground",
                      )}
                    >
                      {prettyModel(m.model)}
                    </span>
                    <span className="ml-auto shrink-0 font-mono tabular-nums text-muted-foreground">
                      {compact(m.tokens)}
                    </span>
                    <span className="shrink-0 text-[10px] text-muted-foreground">
                      &times;{m.calls}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}

          {usage.source === "local-sum" && (
            <p className="mt-3 text-[10px] leading-snug text-muted-foreground">
              Tallied from each call&apos;s reported usage.
              {usage.provider === "groq"
                ? " Groq's own counter appears here after a quota rejection, and overrides this."
                : " Not all providers report a daily total — this is our own count, and the limit is the budget you configured."}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default GroqUsageBadge;
