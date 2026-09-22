// Lightweight polling hooks — the local worker writes progress to the store
// continuously, so 2s polling gives live progress without websockets.

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
    void refresh();
    const t = setInterval(() => void refresh(), intervalMs);
    return () => clearInterval(t);
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
    setLoaded(false);
    void refresh();
    const t = setInterval(() => void refresh(), intervalMs);
    return () => clearInterval(t);
  }, [refresh, intervalMs]);

  return { clips, loaded, refresh };
}

export function useHealth(intervalMs = 5000) {
  const [health, setHealth] = useState<HealthDto | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const triedRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const data = await api.health();
      setHealth(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "API unreachable");
    } finally {
      triedRef.current = true;
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), intervalMs);
    return () => clearInterval(t);
  }, [refresh, intervalMs]);

  return { health, error, tried: triedRef.current, refresh };
}
