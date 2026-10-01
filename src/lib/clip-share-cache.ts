// Prefetch the share-sized clip so Share is INSTANT on tap.
//
// The problem this solves, precisely:
//
//   navigator.share({ files }) needs a real `File` in memory, and it must be
//   called while the page still holds TRANSIENT USER ACTIVATION — a window of
//   roughly five seconds from the tap. Fetching 6-9 MB of 1080p inside the
//   handler burns that window on the network: on Slow 3G the sheet either
//   opened many seconds late or, once activation had expired, `share()` threw
//   NotAllowedError and nothing happened at all.
//
// So the bytes are fetched EARLY (when the card is on screen, not when it is
// tapped), stored, and the `File` is constructed up front. The tap handler
// then does nothing but call `navigator.share()` — no await, no fetch.
//
// Storage is IndexedDB rather than memory-only, because a mobile browser can
// evict an in-memory blob and because it survives a navigation: coming back
// to a batch should not re-download it. The Cache API would also work but
// IndexedDB stores the Blob directly, so a hit needs no network round trip at
// all.
//
// Everything here is best-effort by design. A prefetch failure must never
// surface as a broken button: `get()` simply returns null and the UI shares a
// link instead, which always works.

const DB_NAME = "clipforge-share";
const DB_VERSION = 1;
const STORE = "blobs";

/** One clip's share state, as the UI sees it. */
export interface ShareEntry {
  /** Bytes received so far, for the progress ring. */
  loaded: number;
  /** Total bytes, or 0 when the server did not say (chunked). */
  total: number;
  /** Ready to share: a File exists in memory or in the store. */
  ready: boolean;
  /** Last failure, if any. Never thrown at the caller. */
  error: string | null;
  /** In-flight prefetch promise, so taps can await nothing. */
  inFlight: boolean;
}

export type ShareListener = (clipId: string, entry: ShareEntry) => void;

/** Cap on concurrent prefetches. More thrashes a phone radio and slows the page. */
const MAX_CONCURRENT = 2;

/**
 * Cap on how many clips may prefetch at once, queued plus in flight.
 *
 * The share endpoint is not a static file server: on a miss it transcodes, and
 * that competes with the host's CPU for every other job. Prefetching a whole
 * batch therefore queues work that mostly finishes long after the user stopped
 * looking — a 13-clip grid is ~33 MB of mobile data and ~13 transcodes, and
 * measured on the 1-CPU reference host the last card did not reach Ready until
 * well past the point anyone would wait.
 *
 * Past the cap a card stays on the link path. That is not a downgrade: the
 * share link IS this clip served from this origin, and tapping it is instant.
 * The cap only stops the app from eagerly spending a phone's data and a
 * server's CPU on clips the user may never share.
 */
const MAX_PREFETCH = 4;

/**
 * Retries, with backoff. Three attempts is deliberate: a prefetch that keeps
 * retrying forever drains the battery and competes with the video the user is
 * actually watching, and after three failures the link path is a better answer
 * anyway.
 */
const MAX_ATTEMPTS = 3;
const BACKOFF_MS = [0, 600, 2000];

/**
 * Total bytes we will hold in memory as ready Files.
 *
 * The memory tier exists so the tap is synchronous; this bounds it. At 2.5 MB
 * per clip, 48 MB is ~19 clips — comfortably more than one screenful, and far
 * below the point where a phone starts killing the tab.
 */
const MEMORY_BUDGET_BYTES = 48 * 1024 * 1024;

type Listener = ShareListener;

const memory = new Map<string, File>();
const entries = new Map<string, ShareEntry>();
const listeners = new Set<Listener>();
let memoryBytes = 0;
let queue: string[] = [];
let active = 0;

// ---------------------------------------------------------------------------
// IndexedDB
// ---------------------------------------------------------------------------

let dbPromise: Promise<IDBDatabase | null> | null = null;

function openDb(): Promise<IDBDatabase | null> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve) => {
    // Private-browsing modes and locked-down webviews throw on open. A null
    // db is a supported state: prefetch still works from the network, it just
    // does not survive a reload.
    if (typeof indexedDB === "undefined") return resolve(null);
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch {
      return resolve(null);
    }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
    req.onblocked = () => resolve(null);
  });
  return dbPromise;
}

async function idbGet(key: string): Promise<Blob | null> {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(key);
      req.onsuccess = () => resolve((req.result as Blob) ?? null);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function idbPut(key: string, blob: Blob): Promise<void> {
  const db = await openDb();
  if (!db) return;
  await new Promise<void>((resolve) => {
    try {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(blob, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
      tx.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
}

async function idbDelete(key: string): Promise<void> {
  const db = await openDb();
  if (!db) return;
  try {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).delete(key);
  } catch {
    /* best effort */
  }
}

// ---------------------------------------------------------------------------
// entry bookkeeping
// ---------------------------------------------------------------------------

function blank(): ShareEntry {
  return { loaded: 0, total: 0, ready: false, error: null, inFlight: false };
}

function emit(clipId: string, next: Partial<ShareEntry>) {
  const cur = entries.get(clipId) ?? blank();
  const merged = { ...cur, ...next };
  entries.set(clipId, merged);
  for (const fn of listeners) {
    try {
      fn(clipId, merged);
    } catch {
      /* a bad subscriber must not break the cache */
    }
  }
}

/** Subscribe to a clip's prefetch state. Returns an unsubscribe function. */
export function subscribe(clipId: string, fn: ShareListener): () => void {
  const bound: Listener = (id, entry) => {
    if (id === clipId) fn(id, entry);
  };
  listeners.add(bound);
  return () => listeners.delete(bound);
}

/** Subscribe to every clip. Used by the batch-level "ready" indicator. */
export function subscribeAll(fn: ShareListener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Current state for one clip. Never null; a cold cache is simply blank. */
export function getEntry(clipId: string): ShareEntry {
  return entries.get(clipId) ?? blank();
}

/**
 * The ready File, or null.
 *
 * SYNCHRONOUS on purpose: this is called inside the click handler, and an
 * async lookup would burn the very user-activation window this module exists
 * to protect.
 */
export function get(clipId: string, fileName: string): File | null {
  const hit = memory.get(clipId);
  if (hit) return hit;
  return null;
}

/** True when the bytes are already here — drives the "Ready" badge. */
export function isReady(clipId: string): boolean {
  return memory.has(clipId);
}

/**
 * Network conditions, when the browser reports them.
 *
 * `saveData` means the user has explicitly asked not to waste their data, and
 * `2g`/`slow-2g` means a prefetch is likely to cost more than it saves. Both
 * suppress prefetching: the Share button still works instantly via the link
 * path, which transfers nothing until the user actually posts.
 */
export function networkSaysNoPrefetch(): boolean {
  const c = (
    navigator as Navigator & {
      connection?: {
        saveData?: boolean;
        effectiveType?: string;
        type?: string;
      };
    }
  ).connection;
  if (!c) return false;
  if (c.saveData === true) return true;
  const et = (c.effectiveType || c.type || "").toLowerCase();
  return et === "slow-2g" || et === "2g" || et.includes("2g");
}

/** True when prefetching is allowed at all on this connection. */
export function prefetchAllowed(): boolean {
  return !networkSaysNoPrefetch();
}

/** Fraction downloaded, 0..1, or null when the total is unknown. */
export function progressOf(clipId: string): number | null {
  const e = entries.get(clipId);
  if (!e || !e.total) return null;
  return Math.min(1, e.loaded / e.total);
}

// ---------------------------------------------------------------------------
// fetching
// ---------------------------------------------------------------------------

function rememberMemory(clipId: string, file: File) {
  const prev = memory.get(clipId);
  if (prev) memoryBytes -= prev.size;
  memory.set(clipId, file);
  memoryBytes += file.size;
  // Evict the OLDEST entry, never the one just added (Map preserves
  // insertion order, and re-setting an existing key would not move it).
  while (memoryBytes > MEMORY_BUDGET_BYTES && memory.size > 1) {
    const oldest = memory.keys().next();
    if (oldest.done) break;
    const key = oldest.value;
    if (key === clipId) break;
    const f = memory.get(key);
    memory.set(key, undefined as unknown as File);
    memory.delete(key);
    if (f) memoryBytes -= f.size;
  }
}

/**
 * Fetch with progress, resuming a partial download via Range when possible.
 *
 * The resume matters on exactly the connection this feature targets: a 2.5 MB
 * file that dies at 80% on a phone should continue from byte N, not restart.
 * It is opportunistic — a server that ignores Range (answering 200 instead of
 * 206) simply restarts, which is correct, just not optimal.
 */
async function fetchWithProgress(
  url: string,
  clipId: string,
  attempt: number,
  signal: AbortSignal,
): Promise<Blob> {
  const backoff = BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)];
  if (backoff > 0) await new Promise((r) => setTimeout(r, backoff));

  const from = entries.get(clipId)?.loaded ?? 0;
  const headers: Record<string, string> = {};
  if (from > 0) headers.Range = `bytes=${from}-`;

  const res = await fetch(url, { headers, cache: "force-cache", signal });
  if (!res.ok && res.status !== 206) {
    throw new Error(`HTTP ${res.status}`);
  }

  const total = (() => {
    const cr = res.headers.get("Content-Range");
    if (cr) {
      const m = /\/(\d+)$/.exec(cr);
      if (m) return Number(m[1]);
    }
    const len = res.headers.get("Content-Length");
    return len ? Number(len) + (res.status === 206 ? from : 0) : 0;
  })();

  // A 200 to a ranged request means the server ignored Range: the body is the
  // WHOLE file, so any previously accumulated bytes must be discarded or the
  // result is corrupt.
  const appending = res.status === 206 && from > 0;
  let loaded = appending ? from : 0;
  emit(clipId, { loaded, total });

  if (!res.body) return res.blob();

  const reader = res.body.getReader();
  const chunks: BlobPart[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      emit(clipId, { loaded, total });
    }
  }
  return new Blob(chunks, { type: "video/mp4" });
}

async function run(clipId: string, url: string, fileName: string, signal: AbortSignal) {
  let lastError = "";
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      const blob = await fetchWithProgress(url, clipId, attempt, signal);
      if (!blob || blob.size === 0) throw new Error("empty response");
      await idbPut(clipId, blob);
      // Build the File NOW, off the tap path. This is the object the click
      // handler hands straight to navigator.share().
      const file = new File([blob], fileName, { type: "video/mp4" });
      rememberMemory(clipId, file);
      emit(clipId, { ready: true, error: null, loaded: blob.size, total: blob.size });
      return;
    } catch (err) {
      // An abort is a decision, not a failure. `invalidate` stopped this on
      // purpose, so retrying would resurrect the very ring the caller just
      // cancelled and leave it spinning forever. Settle once and stay quiet.
      if (signal.aborted) {
        emit(clipId, { inFlight: false });
        return;
      }
      lastError = err instanceof Error ? err.message : String(err);
      if (attempt < MAX_ATTEMPTS - 1) {
        // A failed attempt leaves `loaded` at whatever it reached, which is
        // what the next attempt's Range header resumes from.
        emit(clipId, { error: lastError });
      }
    }
  }
  emit(clipId, { inFlight: false, error: lastError });
}

function pump() {
  while (active < MAX_CONCURRENT && queue.length > 0) {
    const clipId = queue.shift();
    if (!clipId) break;
    const job = jobs.get(clipId);
    if (!job) continue;
    active++;
    run(clipId, job.url, job.fileName, job.controller.signal)
      .catch(() => {
        /* run() already recorded the error */
      })
      .finally(() => {
        active--;
        // Always clear inFlight here: run() only does so on total failure, so
        // a stuck spinner after a success would otherwise be possible.
        emit(clipId, { inFlight: false });
        jobs.delete(clipId);
        pump();
      });
  }
}

/**
 * Start (or join) a background prefetch for one clip.
 *
 * Safe to call on every render of every card: it is idempotent, de-duplicated
 * by clipId, and queued rather than fired, so a 13-card grid opens at most
 * MAX_CONCURRENT sockets instead of 13.
 */
export function prefetch(clipId: string, url: string, fileName: string): void {
  if (!url || !prefetchAllowed()) return;
  if (memory.has(clipId)) {
    if (!entries.get(clipId)?.ready) emit(clipId, { ready: true, error: null });
    return;
  }
  if (entries.get(clipId)?.inFlight) return;
  if (jobs.has(clipId)) return;
  if (jobs.size >= MAX_PREFETCH) return;

  jobs.set(clipId, { clipId, url, fileName, controller: new AbortController() });
  emit(clipId, { inFlight: true, error: null });
  queue.push(clipId);
  pump();
}

/** clipId -> the url/name the prefetch is using. */
const jobs = new Map<
  string,
  { clipId: string; url: string; fileName: string; controller: AbortController }
>();

/**
 * Restore a prefetched blob from IndexedDB into memory.
 *
 * Called when a card mounts for a clip we may have cached in a previous
 * session. Returns synchronously-usable state via the store, and resolves once
 * the File exists (or immediately if it never will).
 */
export async function hydrate(clipId: string, fileName: string): Promise<boolean> {
  if (memory.has(clipId)) return true;
  const blob = await idbGet(clipId);
  if (!blob || blob.size === 0) return false;
  const file = new File([blob], fileName, { type: "video/mp4" });
  rememberMemory(clipId, file);
  emit(clipId, {
    ready: true,
    error: null,
    loaded: blob.size,
    total: blob.size,
  });
  return true;
}

/** Forget one clip (a re-render, a deleted job). */
export function invalidate(clipId: string): void {
  // Abort BEFORE dropping the bookkeeping. A fetch that is merely forgotten
  // keeps running, keeps emitting progress, and the ring the caller just
  // cancelled comes back — which is exactly how a share button ended up
  // spinning for minutes after it had already given up.
  jobs.get(clipId)?.controller.abort();
  const f = memory.get(clipId);
  if (f) {
    memoryBytes -= f.size;
    memory.delete(clipId);
  }
  entries.delete(clipId);
  jobs.delete(clipId);
  queue = queue.filter((id) => id !== clipId);
  void idbDelete(clipId);
}

/** Drop everything. Used when the user switches batches. */
export function clear(): void {
  for (const job of jobs.values()) job.controller.abort();
  memory.clear();
  entries.clear();
  jobs.clear();
  queue = [];
  memoryBytes = 0;
}

/**
 * Can this browser actually share a File?
 *
 * Checked once per call site rather than cached: `canShare` results differ per
 * file type, and a stale positive here means a share() that throws at the tap,
 * which is precisely the failure mode being fixed.
 */
export function canShareFile(): boolean {
  const nav = navigator as Navigator & {
    canShare?: (data: ShareData) => boolean;
    share?: (data: ShareData) => Promise<void>;
  };
  if (typeof nav?.share !== "function") return false;
  if (typeof nav.canShare !== "function") return false;
  try {
    return nav.canShare({
      files: [new File([new Blob([new Uint8Array([0])], { type: "video/mp4" })], "probe.mp4")],
    });
  } catch {
    return false;
  }
}

/** Whether navigator.share exists at all. */
export function canShareAnything(): boolean {
  const nav = navigator as Navigator & { share?: (data: ShareData) => Promise<void> };
  return typeof nav?.share === "function";
}
