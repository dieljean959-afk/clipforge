import { useEffect, useState } from "react";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { api, type ClipDto, type SocialPlatformDto, type SocialSlug } from "@/lib/api";
import { buildPostCaption } from "@/lib/clip-export";
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

export function PostMenu({ clip, rendered }: { clip: ClipDto; rendered: boolean }) {
  const [platforms, setPlatforms] = useState<SocialPlatformDto[] | null>(null);
  const [busy, setBusy] = useState<SocialSlug | null>(null);

  useEffect(() => {
    let live = true;
    api
      .socialAccounts()
      .then((r) => live && setPlatforms(r.platforms))
      .catch(() => live && setPlatforms([]));
    return () => {
      live = false;
    };
  }, []);

  /**
   * The share-sheet fallback. Every platform that cannot post right now still
   * gets a one-tap path: hand the file to the OS share sheet (or download it on
   * desktop), which is exactly the behaviour that existed before posting was
   * added. Nothing here can fail because a platform is un-approved.
   */
  const shareInstead = async () => {
    const caption = buildPostCaption(clip.hookText, clip.hashtags);
    const nav = navigator as Navigator & {
      canShare?: (data: ShareData) => boolean;
      share?: (data: ShareData) => Promise<void>;
    };
    try {
      const res = await fetch(api.clipDownloadUrl(clip._id));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const file = new File([await res.blob()], `clip-${clip.order + 1}.mp4`, {
        type: "video/mp4",
      });
      if (nav.canShare?.({ files: [file] })) {
        await nav.share!({ files: [file], text: caption, title: clip.title });
        toast.success("Shared.");
        return;
      }
    } catch {
      // No share sheet, or the user dismissed it -- fall through to download.
    }
    try {
      const res = await fetch(api.clipDownloadUrl(clip._id));
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `clip-${clip.order + 1}.mp4`;
      a.click();
      URL.revokeObjectURL(a.href);
      toast.success("Downloaded — paste the caption into the app.");
    } catch {
      toast.error("Download failed.");
    }
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

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          size="sm"
          variant="outline"
          className="h-7 cursor-pointer gap-1.5 text-xs"
          disabled={!rendered}
          title={rendered ? "Post this clip" : "No rendered MP4 yet"}
        >
          <Send className="size-3.5" />
          Post to...
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72">
        <DropdownMenuLabel className="text-xs text-muted-foreground">
          Caption: {buildPostCaption(clip.hookText, clip.hashtags).split("\n")[0] || "(empty)"}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        {platforms === null ? (
          <DropdownMenuItem disabled>
            <Loader2 className="size-3.5 animate-spin" /> Checking connections...
          </DropdownMenuItem>
        ) : (
          platforms.map((p) => {
            const busyNow = busy === p.slug;
            const canPost = p.connected;
            return (
              <DropdownMenuItem
                key={p.slug}
                disabled={busyNow}
                onSelect={(e) => {
                  // A click that must start a flow or fire a post must not
                  // also close the menu.
                  e.preventDefault();
                  if (busyNow) return;
                  if (canPost) void post(p.slug, p.label);
                  else if (p.status === "not_connected") void connect(p.slug);
                  else void shareInstead();
                }}
                className="flex-col items-start gap-0.5 py-2"
              >
                <span className="flex w-full items-center gap-2 text-sm">
                  {busyNow ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : canPost ? (
                    <Send className={cn("size-3.5", TINTS[p.slug])} />
                  ) : (
                    <TriangleAlert className="size-3.5 text-muted-foreground" />
                  )}
                  <span className="font-medium">{p.label}</span>
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
                      : `${p.headline} — downloads instead.`}
                  </span>
                )}
              </DropdownMenuItem>
            );
          })
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={(e) => (e.preventDefault(), void shareInstead())}>
          <Share2 className="size-3.5" />
          Share sheet / download instead
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <a href="/api/social/accounts" target="_blank" rel="noreferrer" className="text-xs">
            <ExternalLink className="size-3.5" />
            Connection status
          </a>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
