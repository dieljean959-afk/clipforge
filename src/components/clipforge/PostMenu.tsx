import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  Check,
  Copy,
  ExternalLink,
  Link2,
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
import { buildPostCaption } from "@/lib/clip-export";
import {
  clipShareUrl,
  intentUrl,
  openIntent,
  visibleIntents,
} from "@/lib/share-intents";
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
   *
   * The fallback ladder is the important part. This used to end in a silent
   * download plus "no share sheet on this device", which on Android Chrome
   * (where `canShare({files})` is version-dependent and often false) meant the
   * user silently got an 8 MB file they never asked for. Now it degrades
   * through progressively cheaper options and only downloads as an explicit,
   * announced last resort.
   */
  const shareVia = async () => {
    const caption = buildPostCaption(clip.hookText, clip.hashtags);
    const link = clipShareUrl(clip._id);
    const nav = navigator as Navigator & {
      canShare?: (data: ShareData) => boolean;
      share?: (data: ShareData) => Promise<void>;
    };

    // 1. Share TEXT + LINK FIRST. Instant, transfers nothing: the clip is
    //    already served from this origin, so the link IS the clip. This must
    //    never wait on bytes — `navigator.share` needs transient user
    //    activation, and a network wait spends that window.
    if (nav.share && link) {
      try {
        await nav.share({ text: caption, url: link, title: clip.title });
        return;
      } catch {
        // Dismissed by the user, or this platform rejects text/url shares.
      }
    }

    // 2. Copy the caption + link. Always works, transfers nothing.
    if (link) {
      try {
        await navigator.clipboard.writeText(`${caption}\n\n${link}`);
        toast.success("Caption and clip link copied — paste it anywhere.", {
          description: link,
        });
        return;
      } catch {
        // Clipboard blocked (needs a secure context / permission).
      }
    }

    // 3. Last resort, and only with the reason stated. The Share button on the
    //    card shares the CACHED small file instead; reaching here means this
    //    clip was never prefetched.
    toast.error("This browser can't share links.", {
      description: "Use the Share button, or the MP4 button above.",
    });
  };

  /** Open one platform's own composer with the caption (and link) filled in. */
  const shareTo = (slug: string, label: string) => {
    const caption = buildPostCaption(clip.hookText, clip.hashtags);
    const link = clipShareUrl(clip._id);
    const intent = visibleIntents(!!link).find((i) => i.slug === slug);
    if (!intent) {
      toast.error(`${label} cannot build a share link here.`);
      return;
    }
    const url = intentUrl(intent, caption, link);
    if (!url) {
      toast.error(`${label} needs a public clip link — copy the link instead.`);
      return;
    }
    openIntent(url);
    toast.success(`Opened ${label} — press Post there to publish.`);
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
            ? "Share this clip to a platform, or post it through a connected account"
            : "No rendered MP4 yet"
        }
        // A plain onClick, not a Radix trigger: DropdownMenu opens on
        // pointerdown, and on a touch screen a tap that the browser also reads
        // as a scroll would open and immediately close it again. The user saw
        // that as "sometimes nothing happens".
        onClick={() => setOpen(true)}
      >
        <Send className="size-3.5" />
        Share...
      </Button>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent
          side="bottom"
          className="max-h-[88vh] w-full max-w-[100vw] gap-3 overflow-x-hidden overflow-y-auto p-4 box-border sm:max-w-md sm:self-end sm:justify-self-end"
        >
          {/* Sheet affordance: the grabber is decorative only. */}
          <div
            aria-hidden
            className="mx-auto h-1 w-10 shrink-0 rounded-full bg-muted-foreground/40"
          />
          <SheetHeader className="text-left">
            <SheetTitle className="text-base break-words">{clip.title}</SheetTitle>
            <SheetDescription className="block whitespace-normal text-xs [overflow-wrap:anywhere]">
              Caption: {caption.split("\n")[0] || "(empty)"}
              {caption.includes("\n") ? (
                <span className="mt-1 block whitespace-normal text-muted-foreground [overflow-wrap:anywhere]">
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
                    className="flex w-full min-w-0 cursor-pointer flex-col items-start gap-0.5 rounded-lg border border-border px-3 py-2.5 text-left transition hover:border-primary/50 hover:bg-accent/40 disabled:opacity-50"
                  >
                    <span className="flex w-full min-w-0 items-center gap-2 text-sm">
                      {busyNow ? (
                        <Loader2 className="size-3.5 shrink-0 animate-spin" />
                      ) : canPost ? (
                        <Send className={cn("size-3.5 shrink-0", TINTS[p.slug])} />
                      ) : (
                        <TriangleAlert className="size-3.5 shrink-0 text-muted-foreground" />
                      )}
                      <span className="min-w-0 font-medium">Post to {p.label}</span>
                      <span className="ml-auto flex shrink-0 items-center gap-1 text-[10px] text-muted-foreground">
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

            <p className="px-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
              Share to a platform — no download, no account
            </p>

            <div className="grid grid-cols-3 gap-1.5">
              {visibleIntents(!!clipShareUrl(clip._id)).map((intent) => (
                <button
                  key={intent.slug}
                  type="button"
                  onClick={() => {
                    setOpen(false);
                    shareTo(intent.slug, intent.label);
                  }}
                  title={`Open ${intent.label} with this clip's caption`}
                  className="flex min-w-0 cursor-pointer items-center justify-center gap-1.5 rounded-lg border border-border px-2 py-2 text-xs transition hover:border-primary/50 hover:bg-accent/40"
                >
                  <Link2 className={cn("size-3.5 shrink-0", intent.tint)} />
                  <span className="truncate">{intent.label}</span>
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={() => {
                setOpen(false);
                const link = clipShareUrl(clip._id);
                void navigator.clipboard
                  .writeText(
                    link
                      ? `${buildPostCaption(clip.hookText, clip.hashtags)}\n\n${link}`
                      : buildPostCaption(clip.hookText, clip.hashtags),
                  )
                  .then(() =>
                    toast.success("Caption and clip link copied.", {
                      description: link || undefined,
                    }),
                  )
                  .catch(() =>
                    toast.error("This browser blocked the clipboard."),
                  );
              }}
              className="flex w-full min-w-0 cursor-pointer items-center gap-2 rounded-lg border border-border px-3 py-2.5 text-left text-sm transition hover:border-primary/50 hover:bg-accent/40"
            >
              <Copy className="size-3.5 shrink-0" />
              <span className="min-w-0 font-medium">Copy caption + link</span>
              <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                paste anywhere
              </span>
            </button>

            <button
              type="button"
              onClick={() => {
                setOpen(false);
                void shareVia();
              }}
              className="flex w-full min-w-0 cursor-pointer flex-col items-start gap-0.5 rounded-lg border border-border px-3 py-2.5 text-left transition hover:border-primary/50 hover:bg-accent/40"
            >
              <span className="flex w-full min-w-0 items-center gap-2 text-sm">
                <Share2 className="size-3.5 shrink-0" />
                <span className="min-w-0 font-medium">Share link...</span>
                <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                  instant, no download
                </span>
              </span>
              <span className="pl-6 text-[10px] leading-tight text-muted-foreground">
                Opens this phone&apos;s share sheet with the caption and a link to the clip.
                Different from posting above.
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
