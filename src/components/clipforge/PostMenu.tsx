import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  Check,
  ExternalLink,
  Loader2,
  Send,
  Share2,
  TriangleAlert,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { api, type ClipDto, type SocialPlatformDto, type SocialSlug } from "@/lib/api";
import { buildPostCaption, clipFileName } from "@/lib/clip-export";
import { cn } from "@/lib/utils";

/** One tint per platform so the menu is scannable, not four identical rows. */
const TINTS: Record<SocialSlug, string> = {
  youtube: "text-red-400",
  x: "text-sky-300",
  tiktok: "text-cyan-300",
  instagram: "text-fuchsia-300",
};

/** Platforms with no extra approval step get a calmer hint. */
const NO_FUSS: ReadonlySet<SocialSlug> = new Set(["youtube"]);

/**
 * Connection status is the same for every clip on the page, but this component
 * is mounted once per card — a 10-clip batch used to fire ten identical
 * requests, and each one could land on a different state. One shared promise,
 * cached briefly, is both cheaper and the reason the sheet never opens to an
 * endless "Checking connections...".
 */
const PLATFORMS_TTL_MS = 30_000;
let cachedPlatforms: { at: number; value: SocialPlatformDto[] } | null = null;
let inflight: Promise<SocialPlatformDto[]> | null = null;

function loadPlatforms(): Promise<SocialPlatformDto[]> {
  const now = Date.now();
  if (cachedPlatforms && now - cachedPlatforms.at < PLATFORMS_TTL_MS) {
    return Promise.resolve(cachedPlatforms.value);
  }
  inflight ??= api
    .socialAccounts()
    .then((r) => {
      cachedPlatforms = { at: Date.now(), value: r.platforms };
      return r.platforms;
    })
    .catch(() => {
      // A failed lookup must not be cached: the user is on a phone and the
      // next tap should be able to try again.
      return [] as SocialPlatformDto[];
    })
    .finally(() => {
      inflight = null;
    });
  return inflight;
}

export function PostMenu({ clip, rendered }: { clip: ClipDto; rendered: boolean }) {
  const [open, setOpen] = useState(false);
  const [platforms, setPlatforms] = useState<SocialPlatformDto[] | null>(null);
  const [busy, setBusy] = useState<SocialSlug | null>(null);
  const live = useRef(true);

  // Fetched when the sheet is first opened, not on mount: a card the user
  // never taps should cost nothing, and nothing should be able to leave the
  // sheet stuck on a spinner.
  useEffect(() => {
    if (!open) return;
    live.current = true;
    loadPlatforms().then((p) => live.current && setPlatforms(p));
    return () => {
      live.current = false;
    };
  }, [open]);

  /**
   * "Share via..." — the OS share sheet, NOT a social connector.
   *
   * Deliberately a different thing from "Post to...":
   *   Post to...  uses our OAuth connectors, uploads straight to the platform
   *               and needs that platform configured and approved.
   *   Share via... hands the same file to Android/iOS/desktop to send however
   *               the user likes, with no account and no approval.
   *
   * The file is the REAL rendered clip — the same /api/clips/<id>/download
   * endpoint the MP4 button uses, so it is the finished 1080x1920 H.264 with
   * the burned-in captions, never the source video or a proxy. It is fetched
   * ONCE and reused for the download fallback: a 10-clip batch is 80 MB a
   * piece, and the old code downloaded the file twice on every fallback.
   */
  const shareVia = async () => {
    const caption = buildPostCaption(clip.hookText, clip.hashtags);
    const name = clipFileName(clip);
    const nav = navigator as Navigator & {
      canShare?: (data: ShareData) => boolean;
      share?: (data: ShareData) => Promise<void>;
    };
    let blob: Blob;
    try {
      const res = await fetch(api.clipDownloadUrl(clip._id));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      blob = await res.blob();
    } catch {
      toast.error("Could not load the clip — try the MP4 button.");
      return;
    }
    try {
      const file = new File([blob], name, { type: "video/mp4" });
      if (nav.canShare?.({ files: [file] })) {
        await nav.share!({ files: [file], text: caption, title: clip.title });
        toast.success(`Shared ${name} through your share sheet.`);
        return;
      }
    } catch {
      // The user dismissed the sheet, or the platform has no share target.
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
    toast.success(`Downloaded ${name} — no share sheet on this device.`);
  };

  const connect = async (slug: SocialSlug) => {
    try {
      const { authUrl } = await api.socialConnect(slug);
      window.open(authUrl, "_blank", "noopener,noreferrer");
      toast.info("Finish connecting in the new tab, then reopen this menu.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not start the connect flow.");
    }
  };

  const post = async (slug: SocialSlug, label: string) => {
    setBusy(slug);
    try {
      const res = await api.postClip(clip._id, { platform: slug });
      const extra = res.note ? ` · ${res.note}` : "";
      if (res.url) {
        toast.success(`Posted to ${label} ✓${extra}`, {
          description: res.url,
          action: {
            label: "Open",
            onClick: () => window.open(res.url, "_blank", "noopener,noreferrer"),
          },
        });
      } else {
        toast.success(`Posted to ${label} ✓${extra}`);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      // 401 is the backend saying "your token is gone" -- the one case the UI
      // can act on, so it says reconnect rather than a raw status code.
      toast.error(
        /reconnect/i.test(msg)
          ? `Failed — reconnect ${label} in Settings.`
          : `Failed to post to ${label}: ${msg}`,
      );
    } finally {
      setBusy(null);
    }
  };

  const caption = buildPostCaption(clip.hookText, clip.hashtags);

  return (
    <>
      <Button
        size="sm"
        variant="outline"
        className="h-7 w-full cursor-pointer gap-1.5 text-xs"
        disabled={!rendered}
        title={
          rendered
            ? "Post this clip straight to a platform, or share it to another app"
            : "No rendered MP4 yet"
        }
        // A plain onClick, not a Radix trigger: DropdownMenu opens on
        // pointerdown, and on a touch screen a tap that the browser also reads
        // as a scroll would open and immediately close it again. The user saw
        // that as "sometimes nothing happens".
        onClick={() => setOpen(true)}
      >
        <Send className="size-3.5" />
        Post to...
      </Button>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent
          side="bottom"
          className="max-h-[88vh] gap-3 overflow-y-auto p-4 sm:max-w-md sm:self-end sm:justify-self-end"
        >
          {/* Sheet affordance: the grabber is decorative only. */}
          <div
            aria-hidden
            className="mx-auto h-1 w-10 shrink-0 rounded-full bg-muted-foreground/40"
          />
          <SheetHeader className="text-left">
            <SheetTitle className="text-base">{clip.title}</SheetTitle>
            <SheetDescription className="text-xs">
              Caption: {caption.split("\n")[0] || "(empty)"}
              {caption.includes("\n") ? (
                <span className="mt-1 block text-muted-foreground">
                  {caption.split("\n").slice(1).join(" ")}
                </span>
              ) : null}
            </SheetDescription>
          </SheetHeader>

          <div className="flex flex-col gap-1 pb-1">
            {platforms === null ? (
              <p className="flex items-center gap-2 rounded-lg border border-border px-3 py-2.5 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                Checking connections...
              </p>
            ) : (
              platforms.map((p) => {
                const busyNow = busy === p.slug;
                const canPost = p.connected;
                return (
                  <button
                    key={p.slug}
                    type="button"
                    disabled={busyNow}
                    onClick={() => {
                      // Close FIRST, act second. The old dropdown kept itself
                      // open on purpose (preventDefault on every row), which
                      // held Radix's modal layer over the page: the body stayed
                      // scroll-locked and un-tappable, so the rest of the
                      // batch could not be reached. Every action now dismisses
                      // the sheet, and progress is reported by a toast instead
                      // of by a panel that must be swiped away.
                      setOpen(false);
                      if (busyNow) return;
                      if (canPost) void post(p.slug, p.label);
                      else if (p.status === "not_connected") void connect(p.slug);
                      else void shareVia();
                    }}
                    className="flex w-full cursor-pointer flex-col items-start gap-0.5 rounded-lg border border-border px-3 py-2.5 text-left transition hover:border-primary/50 hover:bg-accent/40 disabled:opacity-50"
                  >
                    <span className="flex w-full items-center gap-2 text-sm">
                      {busyNow ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : canPost ? (
                        <Send className={cn("size-3.5", TINTS[p.slug])} />
                      ) : (
                        <TriangleAlert className="size-3.5 text-muted-foreground" />
                      )}
                      <span className="font-medium">Post to {p.label}</span>
                      <span className="ml-auto text-[10px] text-muted-foreground">
                        {canPost ? (
                          <span className="flex items-center gap-1 text-primary">
                            <Check className="size-3" />
                            {p.accountName || "connected"}
                          </span>
                        ) : p.status === "not_connected" ? (
                          "connect"
                        ) : NO_FUSS.has(p.slug) ? (
                          "not set up"
                        ) : (
                          "needs approval"
                        )}
                      </span>
                    </span>
                    {!canPost && (
                      <span className="pl-6 text-[10px] leading-tight text-muted-foreground">
                        {p.status === "not_connected"
                          ? "Click to connect."
                          : `${p.headline} — shares the file instead.`}
                      </span>
                    )}
                  </button>
                );
              })
            )}

            <div className="my-1 h-px bg-border" />

            <button
              type="button"
              onClick={() => {
                setOpen(false);
                void shareVia();
              }}
              className="flex w-full cursor-pointer flex-col items-start gap-0.5 rounded-lg border border-border px-3 py-2.5 text-left transition hover:border-primary/50 hover:bg-accent/40"
            >
              <span className="flex w-full items-center gap-2 text-sm">
                <Share2 className="size-3.5" />
                <span className="font-medium">Share via...</span>
                <span className="ml-auto text-[10px] text-muted-foreground">
                  no account needed
                </span>
              </span>
              <span className="pl-6 text-[10px] leading-tight text-muted-foreground">
                Sends the rendered MP4 (captions burned in) to this phone&apos;s share
                sheet. Different from posting above.
              </span>
            </button>

            <a
              href="/api/social/accounts"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-muted-foreground transition hover:text-foreground"
            >
              <ExternalLink className="size-3.5" />
              Connection status
            </a>
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
}
