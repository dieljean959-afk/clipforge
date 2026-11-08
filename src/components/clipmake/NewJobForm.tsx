import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Globe, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
// LOCKED: do not modify — see LOCKED_SPEC.md §1 (ONE caption style).
import { api, type UsageDto, waitForSandbox } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { CaptionLanguage } from "@/lib/api";

type LanguageChoice = CaptionLanguage;

export function NewJobForm({
  className,
  usage,
  onSandboxState,
}: {
  className?: string;
  usage?: UsageDto | null;
  onSandboxState?: (state: "starting" | "ready" | "failed", message: string) => void;
}) {
  const [mode, setMode] = useState<"url" | "upload">("url");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [sandboxState, setSandboxState] = useState<"idle" | "starting" | "failed">("idle");
  const [sandboxMessage, setSandboxMessage] = useState<string | null>(null);
  // ONE caption style (LOCKED_SPEC §1). No picker: a job cannot choose a look
  // that the renderer would silently resolve to a different one.
  const preset = "bold_stack";
  const [clipCap, setClipCap] = useState<number | undefined>(undefined);
  const [language, setLanguage] = useState<LanguageChoice>("auto");
  // Emoji per batch. Density is the share of chunks allowed to carry one; the
  // selection rules still drop neutral/low-confidence chunks, so "high" is a
  // ceiling rather than a promise.
  const [emojiOn, setEmojiOn] = useState(true);
  const [emojiDensity, setEmojiDensity] = useState<"low" | "medium" | "high">("medium");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSubmit =
    confirmed &&
    !submitting &&
    (mode === "url" ? url.trim().length > 0 : file !== null);

  useEffect(() => {
    if (mode === "url" ? url.trim().length > 0 : file !== null) {
      setSandboxState("idle");
      setSandboxMessage(null);
    }
  }, [mode, url, file]);

  const announceSandbox = (state: "starting" | "ready" | "failed", message: string) => {
    setSandboxState(state === "ready" ? "idle" : state);
    setSandboxMessage(message);
    onSandboxState?.(state, message);
  };

  const canGo = canSubmit && sandboxState !== "starting" && sandboxState !== "failed";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canGo) return;

    setSubmitting(true);
    setSandboxState("starting");
    setSandboxMessage("Sandbox is starting, please wait...");

    try {
      if (mode === "url") {
        await api.createJobWithLanguageFromUrl(
          url,
          preset,
          clipCap ?? undefined,
          language,
          {
            waitForWorker: true,
            onUpdate: announceSandbox,
          },
          { enabled: emojiOn, density: emojiDensity },
        );
      } else if (file !== null) {
        await api.uploadJobWithLanguage(
          file,
          preset,
          clipCap ?? undefined,
          language,
          { enabled: emojiOn, density: emojiDensity },
        );
      }

      toast.success("Job queued — ClipForge's server is on it.", {
        description: mode === "url" ? url.trim() : file?.name,
      });
      setUrl("");
      setFile(null);
      setConfirmed(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Could not queue the job.",
      );
    } finally {
      setSubmitting(false);
      setSandboxState("idle");
      setSandboxMessage(null);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={cn("flex flex-col gap-4", className)}>
      <div className="flex gap-2">
        <Button
          type="button"
          variant={mode === "url" ? "default" : "outline"}
          size="sm"
          onClick={() => setMode("url")}
          className="flex-1 cursor-pointer gap-2"
        >
          <Globe className="size-4" />
          Video URL
        </Button>
        <Button
          type="button"
          variant={mode === "upload" ? "default" : "outline"}
          size="sm"
          onClick={() => {
            setMode("upload");
            fileInputRef.current?.click();
          }}
          className="flex-1 cursor-pointer gap-2"
        >
          <Loader2 className="size-4" />
          Attach file
        </Button>
      </div>

      {mode === "url" ? (
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://youtube.com/watch?v=..."
          inputMode="url"
          autoComplete="off"
          className="h-11 w-full rounded-md border border-input bg-card px-3 text-sm text-foreground"
        />
      ) : (
        <>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/webm"
            className="hidden"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex h-11 w-full cursor-pointer items-center gap-3 rounded-md border border-input bg-card px-3 text-left text-sm text-muted-foreground transition hover:border-primary/50"
          >
            {file?.name || "Choose an MP4 / MOV / WebM file..."}
          </button>
        </>
      )}

      <label className="flex cursor-pointer items-start gap-2 text-xs text-muted-foreground">
        <span className="mt-0.5 size-3.5 shrink-0 text-primary" />
        <span className="flex-1">
          <span className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Clip count
          </span>
          <select
            value={clipCap ?? ""}
            onChange={(e) => setClipCap(e.target.value ? Number(e.target.value) : undefined)}
            className="h-9 w-full cursor-pointer rounded-md border border-input bg-card px-2 text-sm text-foreground"
          >
            <option value="">Auto — based on video length</option>
            {[3, 5, 8, 10, 15, 20, 25, 30, 40].map((n) => (
              <option key={n} value={n}>
                Max {n}
              </option>
            ))}
          </select>
          <span className="mt-1 block text-[11px] leading-relaxed">
            Auto scales with length: 5-8 clips under 10 min, 10-15 up to 25 min,
            15-25 up to an hour, 25-40 beyond. It is a ceiling — the AI only
            keeps moments that score as real highlights, so you may get fewer.
          </span>
        </span>
      </label>

      <label className="flex cursor-pointer items-start gap-2 text-xs text-muted-foreground">
        <Globe className="mt-0.5 size-3.5 shrink-0 text-primary" />
        <span className="flex-1">
          <span className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Caption language
          </span>
          <div className="flex gap-2">
            {(
              [
                { value: "en", label: "ENG" },
                { value: "es", label: "ESP" },
                { value: "auto", label: "Auto" },
              ] as const
            ).map((opt) => (
              <Button
                key={opt.value}
                type="button"
                variant={language === opt.value ? "default" : "outline"}
                size="sm"
                onClick={() => setLanguage(opt.value)}
                className="flex-1 cursor-pointer text-sm font-medium"
              >
                {opt.label}
              </Button>
            ))}
          </div>
          <span className="mt-1 block text-[11px] leading-relaxed text-muted-foreground">
            English-dominant videos → ENG. Spanish-dominant videos → ESP.
            Auto lets the server pick the source language when it can.
          </span>
        </span>
      </label>

      <div className="flex flex-col gap-2 rounded-md border border-input bg-card/50 px-3 py-2">
        <label className="flex cursor-pointer items-center justify-between gap-2 text-xs text-muted-foreground">
          <span className="text-[11px] font-semibold uppercase tracking-wide">
            Emoji per caption
          </span>
          <input
            type="checkbox"
            checked={emojiOn}
            onChange={(e) => setEmojiOn(e.target.checked)}
            className="size-4 cursor-pointer accent-[#39FF14]"
          />
        </label>
        {emojiOn && (
          <>
            <div className="flex gap-2">
              {(["low", "medium", "high"] as const).map((d) => (
                <Button
                  key={d}
                  type="button"
                  size="sm"
                  variant={emojiDensity === d ? "default" : "outline"}
                  onClick={() => setEmojiDensity(d)}
                  className="flex-1 cursor-pointer capitalize"
                >
                  {d}
                </Button>
              ))}
            </div>
            <span className="text-[11px] leading-relaxed text-muted-foreground">
              Only the strongest emotional beat per clip: neutral or unsure
              chunks never get one.
            </span>
          </>
        )}
      </div>

      <label className="flex cursor-pointer items-start gap-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          checked={confirmed}
          onChange={(e) => setConfirmed(e.target.checked)}
          className="mt-0.5 size-3.5 accent-[#39FF14]"
        />
        <span>
          I own this content or have permission to clip it. Clipping other
          creators&apos; videos may violate platform ToS and copyright.
        </span>
      </label>

      <Button type="submit" size="lg" disabled={!canGo} className="cursor-pointer gap-2 font-bold">
        {submitting ? (
          <Loader2 className="size-4 animate-spin" />
        ) : (
          <Globe className="size-4" />
        )}
        Forge clips
      </Button>

      {sandboxState === "starting" && (
        <p className="text-[11px] text-amber-300">
          {sandboxMessage ?? "Sandbox is starting, please wait..."}
        </p>
      )}

      {sandboxState === "failed" && (
        <p className="text-[11px] text-destructive">
          {sandboxMessage ?? "Sandbox did not start."}
        </p>
      )}
    </form>
  );
}
