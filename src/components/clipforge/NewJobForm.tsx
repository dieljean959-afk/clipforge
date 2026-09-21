import { useRef, useState } from "react";
import { toast } from "sonner";
import { Loader2, Paperclip, ServerOff, Sparkles, Youtube } from "lucide-react";
import { useMutation, useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

/**
 * New job form: paste a video URL (server fetch via cobalt → direct) or
 * attach a local file — the file's REAL bytes are uploaded to storage and
 * attached to the job before it can advance. Ownership confirmation is
 * required before submission.
 */
export function NewJobForm({ className }: { className?: string }) {
  const [mode, setMode] = useState<"url" | "upload">("url");
  const [url, setUrl] = useState("");
  const [fileName, setFileName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const createJob = useMutation(api.clips.createJob);
  const attachUpload = useMutation(api.clips.attachUpload);
  const generateUploadUrl = useMutation(api.clips.generateUploadUrl);

  // Startup self-check: only block submitting on POSITIVE evidence the
  // worker chain is dead. `worker === undefined` covers both "still loading"
  // and "no row yet" (fresh deployment / cold start) — in that case the
  // backend arms the worker and queues the job anyway, so we never refuse
  // work that the in-process worker model can actually consume.
  const worker = useQuery(api.clips.getWorkerStatus, {});
  const workerOffline = worker !== undefined && worker.online === false;
  // The VM's heartbeat is the real fetch capability — show the honest hint
  // under the URL field depending on whether the worker VM is awake.
  const vm = useQuery(api.clips.getWorkerVmPublic, {});

  const canSubmit =
    confirmed &&
    !submitting &&
    !workerOffline &&
    (mode === "url" ? url.trim().length > 0 : file !== null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      const jobId = await createJob({
        sourceType: mode,
        source: mode === "url" ? url.trim() : fileName,
      });
      // Upload jobs: push the REAL bytes to storage and attach them — a job
      // without a stored source can never advance past grabbing.
      if (mode === "upload" && file !== null) {
        const uploadUrl = await generateUploadUrl();
        const res = await fetch(uploadUrl, {
          method: "POST",
          headers: { "Content-Type": file.type || "video/mp4" },
          body: file,
        });
        if (!res.ok) throw new Error(`File upload failed (HTTP ${res.status}).`);
        const { storageId } = (await res.json()) as { storageId: string };
        await attachUpload({
          jobId,
          storageId: storageId as never,
          bytes: file.size,
        });
      }
      toast.success("Job queued — Clippy is on it.", {
        description: mode === "url" ? url.trim() : fileName,
      });
      setUrl("");
      setFileName("");
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
              const f = e.target.files?.[0] ?? null;
              setFile(f);
              setFileName(f ? f.name : "");
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex h-11 w-full cursor-pointer items-center gap-3 rounded-md border border-input bg-card px-3 text-left text-sm text-muted-foreground transition hover:border-primary/50"
          >
            <Paperclip className="size-4 shrink-0 text-primary" />
            {fileName || "Choose an MP4 / MOV / WebM file..."}
          </button>
        </>
      )}

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

      {workerOffline && (
        <p className="flex items-center gap-1.5 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          <ServerOff className="size-3.5 shrink-0" />
          Render worker offline — jobs are refused until it is consuming the queue.
        </p>
      )}
      {!workerOffline && mode === "url" && vm !== undefined && vm.vmUp === false && (
        <p className="flex items-center gap-1.5 rounded-md border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-xs text-amber-200">
          <ServerOff className="size-3.5 shrink-0" />
          Worker asleep — flip the ON switch (top of page) to fetch URLs. Uploads and direct MP4 links still work.
        </p>
      )}

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
