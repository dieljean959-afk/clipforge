// Cache-first for share-sized clips ONLY.
//
// Scope is deliberately narrow. A service worker that caches application code
// or API JSON in a Vite dev server caches stale modules and breaks HMR, which
// is a far worse failure than a clip being re-fetched. So:
//
//   * ONLY /api/clips/<id>/share is cached.
//   * Everything else — JS, CSS, JSON, the 1080p /video master, thumbnails —
//     is passed straight through to the network, untouched.
//
// IMPORTANT — boot-loop safety:
//   * This worker does NOT precache the app shell. It does NOT fetch any
//     navigation-time JS/CSS during install. It does NOT call skipWaiting or
//     clients.claim. Activation only deletes caches that belong to OLD shell
//     versions so a stale worker never serves a mix of two builds.
//   * Updates apply on the next navigation, not by forcing a mid-load reload.
//
// The cache key includes the full URL, and the server sends
// `Cache-Control: public, max-age=31536000, immutable` with an ETag derived
// from the file's size+mtime, so a re-rendered clip has a new ETag and is
// never served from here as a stale lighter copy of old footage.
//
// The IndexedDB prefetch cache (src/lib/clip-share-cache.ts) is the primary
// mechanism; this is the HTTP-layer backstop for clips prefetched outside a
// live page (share target reopening, a cold start).

const CACHE = "clipmake-share-v1";

/** True only for the small share artifact. */
function isShareClip(url) {
  return (
    url.origin === self.location.origin &&
    /^\/api\/clips\/[^/]+\/share$/.test(url.pathname)
  );
}

self.addEventListener("install", (event) => {
  // No precache: an install that fetches anything would delay activation on a
  // slow connection for no benefit. Clips get cached lazily on first use.
  // Do NOT skipWaiting here — we only activate when the page navigates to a
  // fresh version, never by forcing a reload mid-boot.
});

self.addEventListener("activate", (event) => {
  // Only this worker's own share cache is kept. Anything else is an old app
  // shell cache from a prior build/version and should be removed so a stale
  // worker cannot serve a mix of old and new assets.
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(
        names.filter((n) => n !== CACHE && n.indexOf("clipmake-shell-") !== 0).map((n) => caches.delete(n)),
      );
      // Do NOT claim clients here. Claiming every client on activation can
      // interleave an in-flight load with a new worker and is one of the
      // classic sources of "progress bar keeps resetting" reload loops. The
      // registered page controls its own load; this worker only serves share
      // clips.
    })(),
  );
});

self.addEventListener("message", (event) => {
  // One-way escape hatch used by the boot loader and by main.tsx when the app
  // detects it is stuck. Asking the active worker to clear its own caches is
  // safer than unregistering it from the page while it is serving requests.
  if (event.data && event.data.type === "CLIPMAKE_CLEAR_SHELL") {
    event.waitUntil(
      caches.keys().then((names) => {
        return Promise.all(names.filter((n) => n.indexOf("clipmake-shell-") === 0)
            .map((n) => caches.delete(n)),
        );
      }),
    );
  }
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  let url;
  try {
    url = new URL(req.url);
  } catch {
    return;
  }

  // Everything that is not a share clip is none of this worker's business.
  if (!isShareClip(url)) return;

  event.respondWith(
    (async () => {
      const cached = await caches.match(req);
      if (cached) return cached;

      const res = await fetch(req);
      // Only a complete, successful, range-free response is worth keeping. A
      // 206 is a partial answer to someone else's Range request and must not
      // become the cached whole file.
      if (res && res.status === 200 && res.type === "basic") {
        try {
          const cache = await caches.open(CACHE);
          await cache.put(req, res.clone());
        } catch {
          // Quota or opaque response. Caching is an optimisation.
        }
      }
      return res;
    })(),
  );
});
