import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Layers, Loader2, Paperclip, Sparkles, Type, Youtube } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, type CaptionPresetDto } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * New job form: paste a video URL (fetched server-side with yt-dlp) or attach
 * a file. Ownership confirmation is required before submission. Uploads
 * always work — URL fetching can fail (bot checks), and we say so honestly.
 */
export function NewJobForm({ className }: { className?: string }) {
  const [mode, setMode] = useState<"url" | "upload">("url");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [presets, setPresets] = useState<CaptionPresetDto[]>([]);
  const [preset, setPreset] = useState("bold_pop");
  // "" = auto (clip count scaled from the source length). A number is a hard
  // cap the user chose, and it is what the server uses.
  const [maxClips, setMaxClips] = useState("");
  const clipCap = maxClips ? Number(maxClips) : undefined;
  const fileInputRef = useRef<HTMLInputElement>(null);

  // caption styles come from the backend so the UI never drifts from the
  // pipeline's presets
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const list = await api.captionPresets();
        if (!cancelled && list.length > 0) setPresets(list);
      } catch {
        // fall back to the static default below; never block submission
      }
    };
    const t = setTimeout(() => void load(), 0);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, []);

  const canSubmit =
    confirmed &&
    !submitting &&
    (mode === "url" ? url.trim().length > 0 : file !== null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      if (mode === "url") {
        await api.createJob(url.trim(), "url", preset, clipCap);
      } else if (file !== null) {
        await api.uploadJob(file, preset, clipCap);
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
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={cn("flex flex-col gap-4", className)}
    >
      <div className="flex gap-2">
        <Button
          type="button"
          variant={mode === "url" ? "default" : "outline"}
          size="sm"
          onClick={() => setMode("url")}
          className="flex-1 cursor-pointer gap-2"
        >
          <Youtube className="size-4" />
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
          <Paperclip className="size-4" />
          Attach file
        </Button>
      </div>

      {mode === "url" ? (
        <Input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://youtube.com/watch?v=..."
          inputMode="url"
          autoComplete="off"
          className="h-11"
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
            <Paperclip className="size-4 shrink-0 text-primary" />
            {file?.name || "Choose an MP4 / MOV / WebM file..."}
          </button>
        </>
      )}

      <label className="flex cursor-pointer items-start gap-2 text-xs text-muted-foreground">
        <Type className="mt-0.5 size-3.5 shrink-0 text-primary" />
        <span className="flex-1">
          <span className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Caption style
          </span>
          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value)}
            className="h-9 w-full cursor-pointer rounded-md border border-input bg-card px-2 text-sm text-foreground"
          >
            {(presets.length > 0
              ? presets
              : [
                  { key: "bold_pop", label: "Bold pop (default)" },
                  { key: "karaoke", label: "Karaoke fill" },
                  { key: "lower_third", label: "Clean lower-third" },
                  { key: "boxed", label: "Boxed" },
                ]
            ).map((p) => (
              <option key={p.key} value={p.key}>
                {p.label}
              </option>
            ))}
          </select>
        </span>
      </label>

      <label className="flex cursor-pointer items-start gap-2 text-xs text-muted-foreground">
        <Layers className="mt-0.5 size-3.5 shrink-0 text-primary" />
        <span className="flex-1">
          <span className="mb-1 block text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Clip count
          </span>
          <select
            value={maxClips}
            onChange={(e) => setMaxClips(e.target.value)}
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

      <Button type="submit" size="lg" disabled={!canSubmit} className="cursor-pointer gap-2 font-bold">
        {submitting ? (
          <Loader2 className="size-4 animate-spin" />
        ) : (
          <Sparkles className="size-4" />
        )}
        Forge clips
      </Button>
    </form>
  );
}
