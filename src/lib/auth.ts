// Thin auth client against the assumed backend contract:
//
//   GET /api/auth/me   -> { ok: boolean; user?: { id, email, name? } }
//   POST /api/auth/signout -> { ok: boolean }
//
// Anything protected returns 401 when the session is missing/invalid. This
// client does not manage tokens itself — it reads the session from the backend
// and keeps a derived local mirror so the UI does not have to poll on every
// render.
//
// Because this project currently has no real /api/auth/* backend implemented,
// the helper below degrades gracefully: it treats an unreachable backend as
// "no session yet" instead of crashing the app, and it exposes the same
// sign-in / sign-up / sign-out surface so the real backend can be wired in
// later without changing the UI.
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type SessionUser = {
  id: string;
  email: string;
  name?: string | null;
};

export type Session = {
  user: SessionUser | null;
  loaded: boolean;
};

type AuthContextValue = {
  session: Session;
  user: SessionUser | null;
  loaded: boolean;
  signingIn: boolean;
  signingUp: boolean;
  weakError: string | null;
  setSession: (next: Session) => void;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, name: string) => Promise<void>;
  signOut: () => Promise<void>;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

const API_BASE = (typeof window !== "undefined" ? (import.meta.env?.VITE_API_BASE ?? "") : "")
  .replace(/\/+$/, "");

async function fetchMe(): Promise<Session> {
  if (!API_BASE) {
    return { user: null, loaded: true };
  }
  try {
    const res = await fetch(`${API_BASE}/api/auth/me`, {
      method: "GET",
      credentials: "include",
      headers: { Accept: "application/json" },
    });
    if (res.status === 401) {
      return { user: null, loaded: true };
    }
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const body = (await res.json()) as { ok?: boolean; user?: SessionUser };
    return {
      user: body.ok && body.user ? body.user : null,
      loaded: true,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : "network error";
    return {
      user: null,
      loaded: true,
      // Keep going locally even if the backend is not reachable yet. A warm
      // dev preview should not lock everyone out because the container is cold.
      _hint: message,
    };
  }
}

async function postAuth(
  path: string,
  body: Record<string, unknown>,
): Promise<void> {
  if (!API_BASE) {
    throw new Error("no backend configured");
  }
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.status === 401) {
    throw new Error("Session invalid or expired.");
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const json = (await res.json()) as { detail?: string; errorMessage?: string };
      if (typeof json.detail === "string") detail = json.detail;
      if (typeof json.errorMessage === "string") detail = json.errorMessage;
    } catch {
      // keep status text
    }
    throw new Error(detail);
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSessionInner] = useState<Session>({ user: null, loaded: false });
  const [signingIn, setSigningIn] = useState(false);
  const [signingUp, setSigningUp] = useState(false);
  const [weakError, setWeakError] = useState<string | null>(null);

  const setSession = useCallback((next: Session) => {
    setSessionInner(next);
    if (!next.loaded) {
      setWeakError(null);
    }
  }, []);

  const refresh = useCallback(async () => {
    setWeakError(null);
    setSessionInner({ user: null, loaded: false });
    const next = await fetchMe();
    setSessionInner(next);
    if (next.user === null && next.loaded && next._hint) {
      setWeakError(next._hint);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      const next = await fetchMe();
      if (!cancelled) {
        setSessionInner(next);
        if (next.user === null && next.loaded && next._hint) {
          setWeakError(next._hint);
        }
      }
    };
    const t = setTimeout(run, 0);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, []);

  const signIn = useCallback(
    async (email: string, password: string) => {
      setSigningIn(true);
      setWeakError(null);
      try {
        await postAuth("/api/auth/signin", { email, password });
        const next = await fetchMe();
        setSessionInner(next);
        if (next.user === null) {
          throw new Error("Signed in, but no user came back.");
        }
      } finally {
        setSigningIn(false);
      }
    },
    [],
  );

  const signUp = useCallback(
    async (email: string, password: string, name: string) => {
      setSigningUp(true);
      setWeakError(null);
      try {
        await postAuth("/api/auth/signup", { email, password, name });
        const next = await fetchMe();
        setSessionInner(next);
        if (next.user === null) {
          throw new Error("Account created, but no session came back.");
        }
      } finally {
        setSigningUp(false);
      }
    },
    [],
  );

  const signOut = useCallback(async () => {
    setWeakError(null);
    try {
      if (API_BASE) {
        await fetch(`${API_BASE}/api/auth/signout`, {
          method: "POST",
          credentials: "include",
        }).catch(() => {
          /* best effort */
        });
      }
    } finally {
      setSessionInner({ user: null, loaded: true });
    }
  }, []);

  const value = useMemo<AuthContextValue>(() => ({
      session,
      user: session.user,
      loaded: session.loaded,
      signingIn,
      signingUp,
      weakError,
      setSession,
      signIn,
      signUp,
      signOut,
      refresh,
    }),
    [session, signingIn, signingUp, weakError, signIn, signUp, signOut, refresh],
  );


  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useSession(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useSession must be used inside <AuthProvider>");
  }
  return ctx;
}

export function RequireAuth({
  children,
  returnTo,
}: {
  children: React.ReactNode;
  returnTo?: string;
}) {
  const { user, loaded, refresh } = useSession();

  useEffect(() => {
    if (!loaded) return;
    if (!user) {
      const params = new URLSearchParams();
      params.set("returnTo", returnTo ?? "/dashboard");
      window.location.href = `/auth?${params}`;
    } else {
      refresh();
    }
  }, [user, loaded, returnTo, refresh]);

  if (!loaded) {
    return null;
  }

  if (!user) {
    return null;
  }

  return <>{children}</>;
}
