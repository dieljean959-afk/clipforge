// Lightweight polling hooks — the server worker writes progress to the store
// continuously, so 2s polling gives live progress without websockets.
//
// Initial fetches are deferred by one tick (setTimeout 0) so effects never
// call setState synchronously during mount — the first update lands after the
// browser has painted.

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ClipDto, type HealthDto, type JobDto } from "@/lib/api";

/**
 * One invalidation key per request family, so the poller can prove to the
 * caller that it has actually re-read the server's current answer and not
 * merely re-rendered whatever it had cached the last time it was asked.
 *
 * Buttons that change state use the same family key as the request they
 * replace, so a delete/re-render/retry can never win a race against the
 * next 2s poll and paint a stale card back into the list it just left.
 */
export const JOB_DETAIL_KEY = (jobId: string) => `job:${jobId}`;

/**
 * Shared failure triage for the pollers.
 *
 * A 4xx is NOT the API being unreachable. It means the server received the
 * request, understood it, and rejected THIS one — a stale route, a bad id, a
 * malformed header block. Rendering a hard red "API unreachable: HTTP 400"
 * over a dashboard that is otherwise showing live job cards is a false alarm,
 * and treating it as an outage is how this banner kept coming back for
 * something that was never an outage.
 *
 * So a 4xx is recorded as `stale` — a note, never an alarm — while only a 5xx
 * or a genuine network failure counts toward the outage threshold, and even
 * those need several consecutive failures before they are believed, since a
 * deploy restarting under the reader produces a blip or two.
 */
function triage(raw: string): { fatal: boolean; message: string } {
  const status = /HTTP (\d{3})/.exec(raw)?.[1];
  if (status && status.startsWith("4")) {
    return { fatal: false, message: raw };
  }
  return { fatal: true, message: raw };
}

/** Consecutive failures before an outage is believed. */
const FAILURES_BEFORE_ALARM = 3;

/** Ceiling for the retry delay while the API is unavailable. */
const MAX_BACKOFF_MS = 30_000;

/**
 * Self-scheduling poller with exponential backoff.
 *
 * A fixed 2 s interval kept hammering the API while it was restarting — one
 * request per 2 s, each one answered by the proxy with a 500, which is both
 * useless load on a 1-vCPU host and the reason the banner counted "6 failed
 * polls" for one short outage. On failure the wait doubles (2s → 4s → 8s →
 * 16s → 30s cap) and snaps back to the base interval on the first success, so
 * recovery is still detected within one poll.
 */
function usePollLoop(run: () => Promise<boolean>, baseMs: number) {
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    const tick = async () => {
      const ok = await run();
      failures = ok ? 0 : Math.min(failures + 1, 4);
      const wait = ok
        ? baseMs
        : Math.min(baseMs * 2 ** failures, MAX_BACKOFF_MS);
      if (!stopped) timer = setTimeout(() => void tick(), wait);
    };
    timer = setTimeout(() => void tick(), 0);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [run, baseMs]);
}

/** Batches fetched on the first paint. Ten is enough to fill a phone screen. */
export const BATCH_PAGE_SIZE = 10;

export function useJobs(intervalMs = 2000) {
  const [jobs, setJobs] = useState<JobDto[] | undefined>(undefined);
  // How many batches the reader has asked to see. Starts at one page and grows
  // as they scroll; the poll then asks for exactly that many, so "load more"
  // and "keep it fresh" are the same request and never disagree.
  const [limit, setLimit] = useState(BATCH_PAGE_SIZE);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A poll failed but we are still showing the last good list. Same split as
  // useHealth: a soft note, never a hard red "API unreachable" over data that
  // is demonstrably on screen.
  const [stale, setStale] = useState<string | null>(null);
  // Read inside async callbacks only, never during render.
  const haveData = useRef(false);
  const failures = useRef(0);

  const applyFailure = useCallback((err: unknown) => {
    const raw = err instanceof Error ? err.message : "API unreachable";
    // The supervisor said "restarting" (503 + restarting:true). That is a
    // known, self-healing state: say so, keep the last known batches on screen,
    // and let the backoff poller find the API again. It is NOT an error id.
    if (raw.startsWith("API restarting")) {
      setStale("API restarting, retrying…");
      return;
    }
    const { fatal } = triage(raw);
    if (!fatal) {
      // The server answered and rejected one request. That is not an outage,
      // so it must not blank the page or claim the API is gone.
      failures.current = 0;
      setStale(`${raw} — the server answered but rejected this request`);
      return;
    }
    failures.current += 1;
    if (failures.current < FAILURES_BEFORE_ALARM) return;
    const msg =
      failures.current > 1
        ? `${raw} (${failures.current} failed polls of /api/jobs)`
        : raw;
    if (haveData.current) setStale(msg);
    else setError(msg);
  }, []);

  const applySuccess = useCallback(() => {
    haveData.current = true;
    failures.current = 0;
    setError(null);
    setStale(null);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const page = await api.listJobs(limit);
      setJobs((prev) => {
        // The backend's normal list is capped to the newest page, so a finished
        // batch that the reader has already scrolled past can look like it
        // vanished and came back again between polls. Keep anything that was
        // already on screen if the server did not explicitly send it again,
        // rather than blinking it out of existence for a page that has not
        // changed.
        if (!prev) return page.jobs;
        const known = new Set(prev.map((j) => j._id));
        const fresh = new Map(page.jobs.map((j) => [j._id, j]));
        const merged = prev.map((j) => fresh.get(j._id) ?? j);
        for (const j of page.jobs) {
          if (!known.has(j._id)) merged.push(j);
        }
        merged.sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
        return merged;
      });
      setTotal(page.total);
      setHasMore(page.nextBefore !== null);
      applySuccess();
      return true;
    } catch (err) {
      applyFailure(err);
      return false;
    }
  }, [applyFailure, applySuccess, limit]);

  /**
   * Drop deleted batches from the list the instant the delete is confirmed.
   *
   * `refresh` merges each page over what is already on screen and keeps any
   * job that page did not mention. It has to: the list is paged, so a batch
   * the reader scrolled past is not gone merely because the newest page does
   * not repeat it.
   *
   * The cost of that merge was a delete that looked like it did nothing —
   * the server answered 200, the toast said "deleted", and the very next
   * poll (2s) put the pill straight back. A deletion is the one case where
   * absence IS authoritative, so it is applied directly here instead of
   * waiting for a page that by design cannot prove it.
   */
  const forget = useCallback((ids: string | string[]) => {
    const drop = new Set(Array.isArray(ids) ? ids : [ids]);
    if (drop.size === 0) return;
    setJobs((prev) => (prev ? prev.filter((j) => !drop.has(j._id)) : prev));
  }, []);

  // One more page. Not a conditional hook, so it is an explicit call from the
  // scroll sentinel AND from the button — either one works.
  const loadMore = useCallback(() => {
    setLimit((n) => n + BATCH_PAGE_SIZE);
  }, []);

  usePollLoop(refresh, intervalMs);

  // Force an immediate poll of ONE job, so "Check now" answers "is this
  // specific job actually moving?" without waiting out the interval or
  // re-fetching the whole list. Returns the fresh document so the caller can
  // confirm what it saw.
  const checkOne = useCallback(async (jobId: string): Promise<JobDto | null> => {
    try {
      const fresh = await api.getJob(jobId);
      setJobs((prev) =>
        prev
          ? prev.map((j) => (j._id === fresh._id ? fresh : j))
          : [fresh],
      );
      applySuccess();
      return fresh;
    } catch (err) {
      // "Check now" is an explicit request for the truth, so it goes through
      // the same triage rather than becoming an instant red banner.
      applyFailure(err);
      return null;
    }
  }, [applyFailure, applySuccess]);

  return { jobs, error, stale, refresh, forget, checkOne, loadMore, hasMore, total };
}

export function useClips(jobId: string | null, intervalMs = 2000) {
  const [clips, setClips] = useState<ClipDto[]>([]);
  const [loaded, setLoaded] = useState(false);
  // A failed list request used to become "loaded, zero clips", which the
  // dashboard rendered as "This batch has no clips yet" — a server 500 looked
  // exactly like a batch that never rendered. Keep the message instead.
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (jobId === null) {
      setClips([]);
      setError(null);
      setLoaded(true);
      return;
    }
    try {
      const data = await api.listClips(jobId);
      setClips(data);
      setError(null);
      setLoaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load clips");
      setLoaded(true);
    }
  }, [jobId]);

  useEffect(() => {
    // loaded=false is derived from the jobId change below instead of a
    // synchronous setState in the effect body.
    const initial = setTimeout(() => void refresh(), 0);
    const t = setInterval(() => void refresh(), intervalMs);
    return () => {
      clearTimeout(initial);
      clearInterval(t);
    };
  }, [refresh, intervalMs]);

  // reset when switching jobs — keyed on jobId, computed during render
  const [seenJobId, setSeenJobId] = useState(jobId);
  if (seenJobId !== jobId) {
    setSeenJobId(jobId);
    setClips([]);
    setError(null);
    setLoaded(false);
  }

  return { clips, loaded, error, refresh };
}

export function useHealth(intervalMs = 5000) {
  const [health, setHealth] = useState<HealthDto | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  // A poll can fail while the PREVIOUS poll succeeded. The dashboard used to
  // show a hard red "API unreachable: HTTP 400" banner directly above a fully
  // populated status card — real CPU/RAM/disk/worker numbers rendered from the
  // last good response, with one transient failure claiming the API was gone.
  //
  // So the two are separated: `error` means "we have never had a response"
  // (genuinely unreachable, and the card has nothing to show), while `stale`
  // means "a poll failed but we are still displaying the last good reading".
  // A single bad poll must not alarm the user, and the last known state is
  // more useful than a red banner.
  const [stale, setStale] = useState<string | null>(null);
  // tried = first probe completed (ok or failed). Derived from state, never a
  // ref read during render.
  const [tried, setTried] = useState(false);    // Read inside the async callback only (never during render) to decide
    // whether a failure is fatal or transient.
    const haveData = useRef(false);
    // Failed probes since the last success — the banner only shows at 3.
    const failures = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const data = await api.health();
      haveData.current = true;
      failures.current = 0;
      setHealth(data);
      setError(null);
      setStale(null);
      return true;
    } catch (err) {
      const raw = err instanceof Error ? err.message : "API unreachable";
      // Known self-healing state from the supervisor, not a fault to report.
      if (raw.startsWith("API restarting")) {
        setStale("API restarting, retrying…");
        return false;
      }
      // What failed decides whether a banner is honest:
      //   5xx              — the server answered and is broken. Banner-worthy.
      //   no status at all — fetch itself threw (offline, DNS, restart). A
      //                      phone losing signal mid-poll is a real network
      //                      failure, banner-worthy AFTER retries.
      //   4xx              — the server answered and REJECTED one request.
      //                      That is a bug in one route, not an unreachable
      //                      API, and shouting "API unreachable: HTTP 400"
      //                      over a card full of live numbers was exactly the
      //                      false alarm this hook exists to prevent. Never
      //                      a banner.
      const status = /HTTP (\d{3})/.exec(raw)?.[1];
      const fatal = status === undefined || status.startsWith("5");
      failures.current = fatal ? failures.current + 1 : 0;
      // Retry twice (three strikes) before a red banner is allowed: a deploy
      // restarting under the reader produces exactly one or two blips, and
      // none of them deserve the alarm.
      if (!fatal || failures.current < 3) return false;
      const msg = failures.current > 1 ? `${raw} (${failures.current} failed polls)` : raw;
      if (haveData.current) {
        // We are still showing the last good reading; keep it and note the
        // blip instead of blanking the card with a false "unreachable".
        setStale(msg);
      } else {
        setError(msg);
      }
      return false;
    } finally {
      setTried(true);
    }
  }, []);

  usePollLoop(refresh, intervalMs);

  return { health, error, stale, tried, refresh };
}
