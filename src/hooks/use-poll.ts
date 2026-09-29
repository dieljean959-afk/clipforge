// Lightweight polling hooks — the server worker writes progress to the store
// continuously, so 2s polling gives live progress without websockets.
//
// Initial fetches are deferred by one tick (setTimeout 0) so effects never
// call setState synchronously during mount — the first update lands after the
// browser has painted.

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ClipDto, type HealthDto, type JobDto } from "@/lib/api";

export function useJobs(intervalMs = 2000) {
  const [jobs, setJobs] = useState<JobDto[] | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await api.listJobs();
      setJobs(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "API unreachable");
    }
  }, []);

  useEffect(() => {
    const initial = setTimeout(() => void refresh(), 0);
    const t = setInterval(() => void refresh(), intervalMs);
    return () => {
      clearTimeout(initial);
      clearInterval(t);
    };
  }, [refresh, intervalMs]);

  // Force an immediate poll of ONE job, so "Check now" answers "is this
  // specific job actually moving?" without waiting out the interval or
  // re-fetching the whole list. Returns the fresh document so the caller can
  // confirm what it saw.
  const checkOne = useCallback(async (jobId: string): Promise<JobDto | null> => {
    try {
      const fresh = await api.getJob(jobId);
      setJobs((prev) =>
        prev ? prev.map((j) => (j._id === fresh._id ? fresh : j)) : prev,
      );
      setError(null);
      return fresh;
    } catch (err) {
      setError(err instanceof Error ? err.message : "API unreachable");
      return null;
    }
  }, []);

  return { jobs, error, refresh, checkOne };
}

export function useClips(jobId: string | null, intervalMs = 2000) {
  const [clips, setClips] = useState<ClipDto[]>([]);
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    if (jobId === null) {
      setClips([]);
      setLoaded(true);
      return;
    }
    try {
      const data = await api.listClips(jobId);
      setClips(data);
      setLoaded(true);
    } catch {
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
    setLoaded(false);
  }

  return { clips, loaded, refresh };
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
  const [tried, setTried] = useState(false);
  // Read inside the async callback only (never during render) to decide
  // whether a failure is fatal or transient.
  const haveData = useRef(false);
  /** Failed probes since the last success — the banner only shows at 3. */
  const failures = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const data = await api.health();
      haveData.current = true;
      failures.current = 0;
      setHealth(data);
      setError(null);
      setStale(null);
    } catch (err) {
      const raw = err instanceof Error ? err.message : "API unreachable";
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
      // Retry twice (three strikes across the 5s polls, ~10s) before a red
      // banner is allowed: a deploy restarting under the reader produces
      // exactly one or two blips, and none of them deserve the alarm.
      if (!fatal || failures.current < 3) return;
      const msg = failures.current > 1 ? `${raw} (${failures.current} failed polls)` : raw;
      if (haveData.current) {
        // We are still showing the last good reading; keep it and note the
        // blip instead of blanking the card with a false "unreachable".
        setStale(msg);
      } else {
        setError(msg);
      }
    } finally {
      setTried(true);
    }
  }, []);

  useEffect(() => {
    const initial = setTimeout(() => void refresh(), 0);
    const t = setInterval(() => void refresh(), intervalMs);
    return () => {
      clearTimeout(initial);
      clearInterval(t);
    };
  }, [refresh, intervalMs]);

  return { health, error, stale, tried, refresh };
}
