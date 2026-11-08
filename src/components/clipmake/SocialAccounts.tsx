import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  BadgeCheck,
  CircleSlash,
  Copy,
  ExternalLink,
  Loader2,
  Plug,
  Unplug,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, type SocialPlatformDto, type SocialSlug } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * The exact env vars each platform needs, in the order you paste them.
 *
 * The "Not configured" state used to be a disabled button with the reason in a
 * tooltip, which is a dead end on a phone: nothing told you WHICH values to
 * get or where to put them. Direct posting needs an app you register with the
 * platform yourself — there is no key we can supply — so the honest, useful
 * thing is to name the values and make them one tap to copy.
 *
 * Cost is the real differentiator and is stated per row, because "works now"
 * and "needs a paid developer account" are very different decisions.
 */
const ENV_VARS: Record<SocialSlug, string[]> = {
  youtube: ["YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET"],
  x: ["X_CLIENT_ID", "X_CLIENT_SECRET"],
  tiktok: ["TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET"],
  instagram: ["META_APP_ID", "META_APP_SECRET"],
};

/** Where to register the app, for the platform you have not done yet. */
const SIGNUP_URLS: Record<SocialSlug, string> = {
  youtube: "https://console.cloud.google.com/apis/credentials",
  x: "https://developer.x.com/en/portal/dashboard",
  tiktok: "https://developers.tiktok.com/apps",
  instagram: "https://developers.facebook.com/apps",
};

/** How hard each approval gate is, as a short badge. */
const GATE_BADGE: Record<
  SocialPlatformDto["gate"],
  { label: string; className: string }
> = {
  none: {
    label: "no approval needed",
    className: "border-primary/50 bg-primary/10 text-primary",
  },
  payment: {
    label: "needs a paid X account",
    className: "border-amber-400/40 bg-amber-400/10 text-amber-300",
  },
  audit: {
    label: "private until audited",
    className: "border-amber-400/40 bg-amber-400/10 text-amber-300",
  },
  app_review: {
    label: "needs app review",
    className: "border-rose-400/40 bg-rose-400/10 text-rose-300",
  },
};

export function SocialAccounts() {
  const [platforms, setPlatforms] = useState<SocialPlatformDto[] | null>(null);
  const [busy, setBusy] = useState<SocialSlug | null>(null);

  const load = useCallback(() => {
    api
      .socialAccounts()
      .then((r) => setPlatforms(r.platforms))
      .catch(() => setPlatforms([]));
  }, []);

  useEffect(load, [load]);

  // Coming back from a provider's consent screen: the callback redirects to
  // `/?social_ok=<slug>`, so re-read the state rather than making the user
  // reload to discover whether the connect worked.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("social_ok")) {
      toast.success(`Connected ${params.get("social_ok")}.`);
      load();
    } else if (params.get("social_error")) {
      toast.error(`Connect failed: ${params.get("social_error")}`);
    }
    if (params.has("social_ok") || params.has("social_error")) {
      params.delete("social_ok");
      params.delete("social_error");
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, [load]);

  const connect = async (slug: SocialSlug) => {
    setBusy(slug);
    try {
      const { authUrl } = await api.socialConnect(slug);
      window.open(authUrl, "_blank", "noopener,noreferrer");
      toast.info("Approve access in the new tab, then reopen Settings.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not start the connect flow.");
    } finally {
      setBusy(null);
    }
  };

  const disconnect = async (slug: SocialSlug) => {
    setBusy(slug);
    try {
      const res = await api.socialDisconnect(slug);
      setPlatforms(res.platforms);
      toast.success("Disconnected.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Disconnect failed.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="space-y-3">
      <header className="space-y-1">
        <h3 className="text-sm font-semibold tracking-tight">Post to social</h3>
        <p className="text-xs text-muted-foreground">
          Optional. ClipForge still downloads and exports exactly as before — these
          accounts only add a one-tap "Post to..." button on each clip.
        </p>
      </header>

      {platforms === null ? (
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" /> Loading connections...
        </p>
      ) : (
        <ul className="space-y-2">
          {platforms.map((p) => {
            const badge = GATE_BADGE[p.gate];
            return (
              <li
                key={p.slug}
                className="min-w-0 rounded-lg border border-border bg-card/40 p-3"
              >
                {/* Stacked top-to-bottom on a phone, one row from sm up. The
                    old single flex-wrap row squeezed the label into a few
                    characters per line on a 360px screen once the two buttons
                    landed beside it — a column gives the text the full width
                    and lets the actions wrap as a group instead. */}
                <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center">
                  <span className="flex min-w-0 flex-wrap items-center gap-2">
                    <span className="text-sm font-medium">{p.label}</span>
                    <span
                      className={cn(
                        "rounded-full border px-2 py-0.5 text-[10px] font-medium",
                        badge.className,
                      )}
                    >
                      {badge.label}
                    </span>
                    {p.connected && (
                      <span className="min-w-0 text-[10px] text-primary">
                        connected{p.accountName ? ` · ${p.accountName}` : ""}
                      </span>
                    )}
                  </span>
                  <span className="flex flex-wrap items-center gap-1 sm:ml-auto">
                    {p.connected ? (
                      <>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 cursor-pointer gap-1 text-xs"
                          disabled={busy === p.slug}
                          onClick={() => void disconnect(p.slug)}
                        >
                          <Unplug className="size-3.5" />
                          Disconnect
                        </Button>
                      </>
                    ) : (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 cursor-pointer gap-1 text-xs"
                      disabled={busy === p.slug || p.status === "needs_app"}
                      title={p.status === "needs_app" ? p.reason : "Connect account"}
                      onClick={() => void connect(p.slug)}
                    >
                      {busy === p.slug ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : (
                        <Plug className="size-3.5" />
                      )}
                      {p.status === "needs_app" ? "Not configured" : "Connect"}
                    </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 cursor-pointer text-xs"
                      asChild
                    >
                      <a href={p.docs} target="_blank" rel="noreferrer">
                        <ExternalLink className="size-3.5" />
                        Docs
                      </a>
                    </Button>
                  </span>
                </div>
                <p className="mt-1.5 text-[11px] leading-snug text-muted-foreground">
                  {p.headline}. {p.detail}
                </p>
                {!p.connected && p.status === "needs_app" && (
                  <div className="mt-1.5 space-y-1.5 rounded-md border border-amber-400/25 bg-amber-400/5 p-2">
                    <p className="flex items-start gap-1.5 text-[11px] text-amber-300/90">
                      <CircleSlash className="mt-0.5 size-3 shrink-0" />
                      {p.reason}
                    </p>
                    {/* Name the exact values and make them copyable. Without
                        this the row is a dead end: the user knows posting is
                        unavailable but not what to go and do about it. */}
                    <div className="flex flex-wrap gap-1">
                      {ENV_VARS[p.slug].map((name) => (
                        <button
                          key={name}
                          type="button"
                          onClick={() => {
                            void navigator.clipboard
                              ?.writeText(name)
                              .then(() => toast.success(`Copied ${name}`))
                              .catch(() => toast.error("Clipboard blocked."));
                          }}
                          title="Copy this variable name"
                          className="flex cursor-pointer items-center gap-1 rounded border border-amber-400/30 bg-background/60 px-1.5 py-0.5 font-mono text-[10px] text-amber-200 transition hover:border-amber-400/60"
                        >
                          {name}
                          <Copy className="size-2.5" />
                        </button>
                      ))}
                      <a
                        href={SIGNUP_URLS[p.slug]}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center gap-1 rounded border border-amber-400/30 bg-background/60 px-1.5 py-0.5 text-[10px] text-amber-200 transition hover:border-amber-400/60"
                      >
                        Register an app
                        <ExternalLink className="size-2.5" />
                      </a>
                    </div>
                    <p className="text-[10px] leading-snug text-muted-foreground">
                      Paste them into the <strong>Keys/API keys</strong> tab, then
                      reload. Until then, the Share button on each clip still
                      posts the MP4 through your phone&apos;s share sheet — no
                      account needed.
                    </p>
                  </div>
                )}
                {p.connected && (
                  <p className="mt-1 flex items-center gap-1.5 text-[11px] text-primary/80">
                    <BadgeCheck className="size-3 shrink-0" />
                    Ready to post.
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
