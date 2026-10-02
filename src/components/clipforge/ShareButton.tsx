import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Check, Link2, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, type ClipDto } from "@/lib/api";
import { buildPostCaption, clipFileName } from "@/lib/clip-export";
import { clipShareUrl } from "@/lib/share-intents";
import * as cache from "@/lib/clip-share-cache";
import { cn } from "@/lib/utils";

/**
 * Share one clip. Fast by construction, not by luck.
 *
 * The single rule everything else serves: **`navigator.share()` is called
 * synchronously inside the click handler.** The Web Share API needs transient
 * user activation, which expires about five seconds after the tap. Every
 * `await` before that call spends that budget — which is exactly why the
 * previous version (fetch 8 MB, then share) appeared broken on a phone: the
 * bytes arrived after activation was gone and `share()` threw.
 *
 * So there are exactly two outcomes, and both are reached without awaiting:
 *
 *   cached  -> navigator.share({ files: [file], text })   the real MP4
 *   not yet -> navigator.share({ text, url })              the link, instantly
 *
 * The link path is not a degraded experience. The clip is served from this
 * origin, so the link IS the clip, and it costs the user nothing to post it.
 * The file path is the upgrade, and the badge says which one a tap will get.
 */

/**
 * Never leave the spinner up forever if something goes wrong upstream.
 *
 * Deliberately short. This copy is fetched from a server that transcodes on a
 * miss, so a clip that is not ready in half a minute is not going to be ready
 * in two — and a ring that runs for minutes reads as a broken button, not as
 * progress. Timing out just puts the card back on the instant link path, which
 * is the outcome a tap produces either way.
 */
const READY_TIMEOUT_MS = 25_000;

/**
 * How long to keep asking the server whether a share link is live.
 *
 * Three attempts with backoff, as promised: the share copy is built by a
 * transcode, and a host with one core can take a few seconds for the first
 * clip. Giving up early would strand a badge on "Preparing…" for a link that
 * becomes valid a moment later.
 */
const SHARE_CHECK_DELAYS_MS = [600, 1_800, 4_000];

/** Copy text and confirm it. Resolves false when the clipboard refused. */
async function copyLink(url: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(url);
      return true;
    }
  } catch {
    /* fall through to the legacy path */
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = url;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

export function ShareButton({ clip, rendered }: { clip: ClipDto; rendered: boolean }) {
  const [entry, setEntry] = useState<cache.ShareEntry>(() => cache.getEntry(clip._id));
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [linkReady, setLinkReady] = useState(clip.shareReady === true);
  const readyRef = useRef(false);

  const fileName = clipFileName(clip);
  const shareUrl = api.clipShareUrl(clip._id);

  // ---- prefetch: start as soon as the card exists, not when it is tapped ---
  useEffect(() => {
    if (!rendered) return;
    let alive = true;

    // A blob cached in a previous session is restored into memory first, so a
    // returning user gets an instant file share with no network at all.
    void cache.hydrate(clip._id, fileName).then((ok) => {
      if (!alive) return;
      if (ok) readyRef.current = true;
      setEntry(cache.getEntry(clip._id));
    });

    // ...and if that missed, fetch it now. Queued and de-duplicated inside the
    // cache, so a 13-card grid opens at most two sockets.
    cache.prefetch(clip._id, shareUrl, fileName);

    const off = cache.subscribe(clip._id, (_id, next) => {
      if (!alive) return;
      if (next.ready) readyRef.current = true;
      setEntry(next);
    });
    return () => {
      alive = false;
      off();
    };
  }, [clip._id, fileName, shareUrl, rendered]);

  // The cap exists so a wedged request cannot leave "Preparing..." on screen
  // for the rest of the session. Hitting it simply means the link path is used.
  useEffect(() => {
    if (!entry.inFlight) return;
    const t = setTimeout(() => {
      if (readyRef.current) return;
      cache.invalidate(clip._id);
      setEntry(cache.getEntry(clip._id));
    }, READY_TIMEOUT_MS);
    return () => clearTimeout(t);
  }, [entry.inFlight, clip._id]);

  // ---- the SERVER's share link, not this browser's cache -----------------
  // The badge has to mean "the public URL works for anyone", not "this phone
  // finished downloading a blob": a per-device cache resets, and a green tick
  // on a link the recipient cannot open is worse than no tick. So the state
  // comes from a real probe of the share copy, and it is re-checked with
  // backoff while the copy is still being built.
  useEffect(() => {
    if (!rendered) return;
    if (clip.shareReady === true) {
      setLinkReady(true);
      return;
    }
    let alive = true;
    let timer: number | undefined;
    for (const delay of SHARE_CHECK_DELAYS_MS) {
      timer = window.setTimeout(() => {
        if (!alive) return;
        void api
          .ensureShareLink(clip._id)
          .then((s) => {
            if (alive) setLinkReady(s.shareReady);
          })
          .catch(() => {
            /* a link that is not ready yet is not an error to shout about */
          });
      }, delay);
    }
    return () => {
      alive = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [clip._id, clip.shareReady, rendered]);

  /** Copy the public MP4 link. This is the whole of "Share link". */
  const onCopyLink = useCallback(() => {
    const sid = clip.shareId;
    if (!sid) {
      toast.error("This clip has no share link yet.");
      return;
    }
    const url = api.clipPublicMp4Url(sid);
    setBusy(true);
    void copyLink(url).then((ok) => {
      setBusy(false);
      if (!ok) {
        toast.error("Could not copy the link.", { description: url });
        return;
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2_000);
      toast.success("Copied ✓", { description: url });
    });
  }, [clip.shareId, clip._id]);

  const onShare = useCallback(() => {
    // --- SYNCHRONOUS FROM HERE. No await, no fetch, no state set. -----------
    setBusy(true);

    const nav = navigator as Navigator & {
      canShare?: (data: ShareData) => boolean;
      share?: (data: ShareData) => Promise<void>;
    };
    if (typeof nav.share !== "function") {
      setBusy(false);
      toast.error("This browser cannot share.", {
        description: "Use the MP4 button, or a platform row below.",
      });
      return;
    }

    const caption = buildPostCaption(clip.hookText, clip.hashtags);
    const publicUrl = clipShareUrl(clip._id, clip.shareId);
    const file = readyRef.current ? cache.get(clip._id, fileName) : null;

    // Some apps drop the text field when a file is attached, so the caption is
    // copied to the clipboard in the same gesture. Fire-and-forget: awaiting it
    // would spend the activation window we are trying to protect.
    if (navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(caption).catch(() => {
        /* the share below still carries the text */
      });
    }

    if (file && cache.canShareFile()) {
      // The File was built during prefetch, so this call is genuinely instant.
      void nav
        .share({ files: [file], text: caption, title: clip.title })
        .then(() => {
          /* shared */
        })
        .catch((err: unknown) => {
          if (isAbort(err)) return; // the user closed the sheet: not an error
          toast.error("Could not share the video file.", {
            description: "Try Share link for an instant link instead.",
          });
        })
        .finally(() => setBusy(false));
      return;
    }

    // Not cached yet: post the LINK right now rather than make them wait.
    void nav
      .share({ text: caption, url: publicUrl, title: clip.title })
      .then(() => {
        /* shared */
      })
      .catch((err: unknown) => {
        if (isAbort(err)) return;
        toast.error("Could not open the share sheet.", {
          description: "Use the MP4 button, or copy the caption.",
        });
      })
      .finally(() => setBusy(false));
    // --- end of the synchronous window --------------------------------------
  }, [clip._id, clip.title, clip.hookText, clip.hashtags, fileName]);

  const pct = cache.progressOf(clip._id);
  const showRing = entry.inFlight && !entry.ready;
  // Two different things, deliberately: `fileReady` is about this device
  // holding the bytes for navigator.share, `linkReady` is about the public URL
  // working for a recipient. Only the second one earns the green tick.
  const fileReady = entry.ready;
  const label = fileReady ? "Share" : "Share link";
  const onClick = fileReady ? onShare : onCopyLink;

  return (
    <Button
      size="sm"
      variant="outline"
      className="h-7 w-full cursor-pointer gap-1.5 text-xs"
      disabled={!rendered || busy}
      title={
        rendered
          ? fileReady
            ? "Share the rendered MP4"
            : "Copy the public share link for this clip"
          : "No rendered MP4 yet"
      }
      onClick={onClick}
    >
      {copied ? (
        <Check className="size-3.5 text-primary" />
      ) : fileReady ? (
        <Check className="size-3.5 text-primary" />
      ) : showRing ? (
        // A determinate ring when the server sent a Content-Length, an
        // indeterminate spinner when it did not (chunked responses).
        pct !== null ? (
          <Ring pct={pct} />
        ) : (
          <span
            className="size-3.5 shrink-0 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-primary"
            aria-hidden
          />
        )
      ) : (
        <Share2 className="size-3.5" />
      )}
      {copied ? "Copied ✓" : label}
      {linkReady && (
        <span className="ml-auto text-[10px] font-normal text-primary">Ready</span>
      )}
      {!linkReady && rendered && (
        <span className="ml-auto text-[10px] font-normal text-muted-foreground">
          Preparing…
        </span>
      )}
    </Button>
  );
}

/** Determinate download ring. */
function Ring({ pct }: { pct: number }) {
  const r = 5.5;
  const c = 2 * Math.PI * r;
  return (
    <svg className="size-3.5 shrink-0 -rotate-90" viewBox="0 0 14 14" aria-hidden>
      <circle
        cx="7"
        cy="7"
        r={r}
        className="stroke-muted-foreground/30"
        strokeWidth="2"
        fill="none"
      />
      <circle
        cx="7"
        cy="7"
        r={r}
        className="stroke-primary"
        strokeWidth="2"
        fill="none"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - Math.max(0, Math.min(1, pct)))}
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * A cancelled share is not a failure.
 *
 * `AbortError` is what the spec returns when the user dismisses the sheet, so
 * showing a red toast for it teaches people that the button is broken.
 */
function isAbort(err: unknown): boolean {
  return (
    err instanceof DOMException && err.name === "AbortError"
  ) || (err instanceof Error && err.name === "AbortError");
}

/** Quick X post. Opens the documented composer intent with the caption. */
export function XQuickButton({ clip }: { clip: ClipDto }) {
  const caption = buildPostCaption(clip.hookText, clip.hashtags);
  return (
    <Button
      size="sm"
      variant="outline"
      className="h-7 cursor-pointer gap-1.5 text-xs"
      title="Open X with this caption"
      onClick={() => {
        const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(caption)}`;
        window.open(url, "_blank", "noopener,noreferrer");
        toast.success("Opened X — press Post there.", {
          action: {
            label: "Copy caption",
            onClick: () => void navigator.clipboard?.writeText(caption),
          },
        });
      }}
    >
      <Link2 className="size-3.5" />
      X
    </Button>
  );
}

/** Batch-level summary so the user knows when sharing will be instant. */
export function ShareReadinessSummary({ clips }: { clips: ClipDto[] }) {
  const [ready, setReady] = useState(0);
  useEffect(() => {
    const update = () =>
      setReady(clips.filter((c) => c.videoPath && cache.isReady(c._id)).length);
    update();
    return cache.subscribeAll(() => update());
  }, [clips]);

  const rendered = clips.filter((c) => c.videoPath);
  if (rendered.length === 0 || ready === 0) return null;

  return (
    <p className={cn("text-[10px] text-muted-foreground")}>
      {ready} of {rendered.length} clips ready to share as video
      {ready < rendered.length && " — the rest share as a link instantly"}
    </p>
  );
}
