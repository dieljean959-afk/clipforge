// Share a clip to a social platform WITHOUT downloading it.
//
// The "Share via..." row used to hand the file to the OS share sheet and, when
// the browser had none, silently fall back to a download and report "no share
// sheet on this device". That is the wrong fallback on the device people
// actually use this on: Android Chrome exposes `navigator.canShare` for FILES
// inconsistently across versions, so the sheet reported no share target and
// the user got a download they never asked for.
//
// Every platform below has a public WEB INTENT: a URL that opens the
// platform's own composer with the caption and link already filled in. It
// needs no OAuth app, no approval, no download and no account connection, and
// it works on every browser including the ones with no share sheet at all.
//
// Ordering of preference, and why:
//   1. direct API post  — already in PostMenu; needs a connected account
//   2. web intent      — here; works everywhere, costs nothing
//   3. OS share sheet  — only for actually moving the FILE somewhere
//   4. download        — last resort, never silent

/**
 * A platform's share URL builder.
 *
 * `url` receives the already-encoded text and a link that may be empty. Every
 * builder MUST return an absolute https URL or null when it cannot build one;
 * null hides the row rather than opening something broken.
 */
export interface ShareIntent {
  slug: string;
  label: string;
  /** Tailwind text colour, so the rows are scannable rather than identical. */
  tint: string;
  /** Shown when there is no link to share (offline / unknown origin). */
  textOnly: boolean;
  url: (text: string, link: string) => string | null;
}

/**
 * Absolute, publicly reachable URL for a rendered clip.
 *
 * A relative path is useless in a share intent: the platform's server (or the
 * person it is sent to) cannot resolve `/api/clips/x/video` against our origin.
 * So this resolves against `window.location.origin` and returns "" when there
 * is no usable origin, which makes the callers fall back to text-only intents
 * rather than posting a broken link.
 */
export function clipShareUrl(clipId: string): string {
  if (typeof window === "undefined") return "";
  const origin = window.location.origin;
  if (!origin || origin === "null") return "";
  return `${origin}/api/clips/${encodeURIComponent(clipId)}/video`;
}

/**
 * The platform rows, most-used first.
 *
 * These are the share targets that work with no app registration at all. They
 * are deliberately separate from the OAuth connectors in PostMenu: a connector
 * uploads the file for you, an intent opens their composer so you press Post.
 */
export const SHARE_INTENTS: readonly ShareIntent[] = [
  {
    slug: "x",
    label: "X",
    tint: "text-sky-300",
    textOnly: true,
    // /intent/tweet is X's documented composer deep link. `url` is omitted on
    // purpose: X does not reliably unfurl its own media links, and a dead
    // preview card reads worse than no link at all.
    url: (text) =>
      `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`,
  },
  {
    slug: "facebook",
    label: "Facebook",
    tint: "text-blue-400",
    textOnly: false,
    url: (text, link) =>
      `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(link || "")}` +
      (text ? `&quote=${encodeURIComponent(text)}` : ""),
  },
  {
    slug: "threads",
    label: "Threads",
    tint: "text-zinc-200",
    textOnly: true,
    url: (text) => {
      // The Threads text-intent needs the @ prefix to treat it as a mention.
      const t = text.replace(/^@/, "%40");
      return `https://www.threads.net/intent/post?text=${encodeURIComponent(t)}`;
    },
  },
  {
    slug: "reddit",
    label: "Reddit",
    tint: "text-orange-400",
    textOnly: true,
    url: (text) => {
      // Reddit's submit endpoint takes `text`, or `title`+`url`. With only a
      // caption and no link, `text` is the correct field.
      const linkMatch = text.match(/https?:\/\/\S+/);
      const link = linkMatch ? linkMatch[0] : "";
      const rest = text.replace(/https?:\/\/\S+\s*/, "").trim();
      return link
        ? `https://www.reddit.com/submit?url=${encodeURIComponent(link)}&title=${encodeURIComponent(rest)}`
        : `https://www.reddit.com/submit?text=${encodeURIComponent(text)}`;
    },
  },
  {
    slug: "whatsapp",
    label: "WhatsApp",
    tint: "text-emerald-400",
    textOnly: true,
    url: (text) => `https://wa.me/?text=${encodeURIComponent(text)}`,
  },
  {
    slug: "telegram",
    label: "Telegram",
    tint: "text-sky-400",
    textOnly: true,
    url: (text) =>
      `https://t.me/share/url?url=${encodeURIComponent("")}&text=${encodeURIComponent(text)}`,
  },
  {
    slug: "linkedin",
    label: "LinkedIn",
    tint: "text-sky-500",
    textOnly: true,
    // LinkedIn has no public share intent. Sharing the mobile site is the
    // documented-ish path and opens the composer rather than a dead page.
    url: () => "https://www.linkedin.com/feed/",
  },
  {
    slug: "pinterest",
    label: "Pinterest",
    tint: "text-red-500",
    textOnly: false,
    // Pinterest pins a URL. With no link there is nothing to pin, so this row
    // only appears when we have one.
    url: (_text, link) =>
      link ? `https://pinterest.com/pin/create/button/?url=${encodeURIComponent(link)}` : null,
  },
  {
    slug: "email",
    label: "Email",
    tint: "text-amber-400",
    textOnly: true,
    url: (text, link) =>
      `mailto:?subject=${encodeURIComponent("Clipped for you")}` +
      `&body=${encodeURIComponent(link ? `${text}\n\n${link}` : text)}`,
  },
];

/**
 * The rows to actually render, given whether we have a link to share.
 *
 * Pinterest pins a URL and cannot take a bare caption, so it is dropped when
 * there is nothing to pin rather than opening a create form with no image.
 */
export function visibleIntents(hasLink: boolean): ShareIntent[] {
  return SHARE_INTENTS.filter((i) => hasLink || !i.slug.startsWith("pinterest"));
}

/** Build a platform share URL, or null when that platform cannot build one. */
export function intentUrl(
  intent: ShareIntent,
  text: string,
  link: string,
): string | null {
  try {
    const url = intent.url(text, link);
    if (!url) return null;
    // A share intent must never be javascript: or data:. Only https, http and
    // mailto are ever legitimate here.
    if (!/^(https?:|mailto:)/i.test(url)) return null;
    return url;
  } catch {
    return null;
  }
}

/**
 * Open a share intent.
 *
 * Uses a real anchor with rel="noopener noreferrer" rather than
 * window.open: on iOS/Safari a popup opened outside a user gesture is blocked
 * outright, and on Android an <a target=_blank> is handed to the system
 * browser chooser, which is what people expect from "share to X".
 */
export function openIntent(url: string): boolean {
  if (typeof document === "undefined") return false;
  const a = document.createElement("a");
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.click();
  return true;
}
