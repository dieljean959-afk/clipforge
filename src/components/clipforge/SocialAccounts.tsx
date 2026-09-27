import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  BadgeCheck,
  CircleSlash,
  ExternalLink,
  Loader2,
  Plug,
  Unplug,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, type SocialPlatformDto, type SocialSlug } from "@/lib/api";
import { cn } from "@/lib/utils";

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
                className="rounded-lg border border-border bg-card/40 p-3"
              >
                <div className="flex flex-wrap items-center gap-2">
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
                    <span className="text-[10px] text-primary">
                      connected{p.accountName ? ` · ${p.accountName}` : ""}
                    </span>
                  )}
                  <span className="ml-auto flex items-center gap-1">
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
                  <p className="mt-1 flex items-start gap-1.5 text-[11px] text-amber-300/90">
                    <CircleSlash className="mt-0.5 size-3 shrink-0" />
                    {p.reason}
                  </p>
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
