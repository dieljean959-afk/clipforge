// Lightweight polling hooks — the server worker writes progress to the store
// continuously, so 2s polling gives live progress without websockets.
//
// Initial fetches are deferred by one tick (setTimeout 0) so effects never
// call setState synchronously during mount — the first update lands after the
// browser has painted.

import { useCallback, useEffect, useState } from "react";
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

  return { jobs, error, refresh };
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
  // tried = first probe completed (ok or failed). Derived from state, never a
  // ref read during render.
  const [tried, setTried] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const data = await api.health();
      setHealth(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "API unreachable");
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

  return { health, error, tried, refresh };
}
