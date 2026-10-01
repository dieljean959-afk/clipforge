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
// The cache key includes the full URL, and the server sends
// `Cache-Control: public, max-age=31536000, immutable` with an ETag derived
// from the file's size+mtime, so a re-rendered clip has a new ETag and is
// never served from here as a stale lighter copy of old footage.
//
// The IndexedDB prefetch cache (src/lib/clip-share-cache.ts) is the primary
// mechanism; this is the HTTP-layer backstop for clips prefetched outside a
// live page (share target reopening, a cold start).

const CACHE = "clipforge-share-v1";

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
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(
        names.filter((n) => n !== CACHE).map((n) => caches.delete(n)),
      );
      await self.clients.claim();
    })(),
  );
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
