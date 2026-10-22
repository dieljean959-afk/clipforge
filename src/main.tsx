import '@vly-ai/integrations';
import { Toaster } from "@/components/ui/sonner";
import React, { StrictMode, useEffect, lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes, useLocation } from "react-router";
import "./index.css";
import { useSession, AuthProvider } from "@/lib/auth";
import { AuthPage } from "./pages/Auth";


// Lazy load route components for better code splitting
const Landing = lazy(() => import("./pages/Landing.tsx"));
const Dashboard = lazy(() => import("./pages/Dashboard.tsx"));
const NotFound = lazy(() => import("./pages/NotFound.tsx"));

function PublicDashboard() {
  return <Dashboard />;
}

// The Vly dev toolbar is CHROME, not app: it renders nothing unless the
// hostname ends in `.vly.sh` (see isDevDeployment inside it). It was imported
// eagerly, so every visitor — including a phone on a slow connection — paid for
// the toolbar module AND `@zumer/snapdom`, the screenshot library it pulls in,
// before first paint. Deferring it keeps those bytes out of the critical path
// entirely, and the toolbar still appears when it is actually applicable.
const VlyToolbar = lazy(() => import("../vly-toolbar-readonly.tsx"));

// Simple loading fallback for route transitions
function RouteLoading() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="animate-pulse text-muted-foreground">Loading...</div>
    </div>
  );
}

/** Silent error boundary — if VlyToolbar crashes it renders nothing instead of
 *  crashing the whole app (e.g. hook errors in WebContainer environment). */
class ToolbarErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(err: Error) {
    console.warn("[VlyToolbar] Caught error, toolbar disabled:", err.message);
  }
  render() {
    return this.state.hasError ? null : this.props.children;
  }
}

/** Hard guard so runtime errors never leave the preview as a blank page. */
declare global {
  interface Window {
    /** Set by main.tsx the moment React owns the page; see index.html. */
    __clipmakeBooted?: boolean;
  }
}

class RootErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; message: string; stack: string }
> {
  state = { hasError: false, message: "", stack: "" };
  static getDerivedStateFromError(error: Error) {
    return {
      hasError: true,
      message: error.message || "Unknown runtime error",
      stack: error.stack || "",
    };
  }
  componentDidCatch(err: Error) {
    console.error("[WebContainer preview] Root crash:", err);
    document.title = `Clipmake error: ${err.message || "runtime failure"}`;
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-background text-foreground p-6">
          <div className="max-w-lg text-center">
            <p className="text-sm font-semibold">Preview runtime error</p>
            <p className="mt-2 text-xs text-muted-foreground break-words">
              {this.state.message}
            </p>
            {this.state.stack && (
              <pre className="mt-3 text-left text-[10px] leading-4 text-muted-foreground/80 max-h-40 overflow-auto rounded border border-border/60 p-2">
                {this.state.stack}
              </pre>
            )}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function RouteSyncer() {
  const location = useLocation();
  useEffect(() => {
    window.parent.postMessage(
      { type: "iframe-route-change", path: location.pathname },
      "*",
    );
  }, [location.pathname]);

  useEffect(() => {
    function handleMessage(event: MessageEvent) {
      if (event.data?.type === "navigate") {
        if (event.data.direction === "back") window.history.back();
        if (event.data.direction === "forward") window.history.forward();
      }
    }
    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  return null;
}

/** Removes the static boot fallback from index.html, once React has committed.
 *
 *  Deliberately a component rather than a line before `createRoot`: React only
 *  runs effects AFTER it has successfully committed, so the fallback stays on
 *  screen through any render-time throw and is removed only once the app is
 *  genuinely up. A sibling of the error boundary (not a child) so it still
 *  mounts when the boundary catches — otherwise a crash would leave both the
 *  fallback and the error panel on screen.
 */
function BootCleanup() {
  useEffect(() => {
    document.getElementById("boot")?.remove();
  }, []);
  return null;
}

function AuthRouteContent() {
  // The dashboard is public. The auth page stays reachable at /auth for anyone
  // who still wants an account, but the app no longer forces a sign-in gate.
  //
  // `?returnTo=` is honoured here, not in the Landing links: the caller already
  // knows where it wants the visitor to land, so the auth page just relays it
  // to AuthPage after a successful sign-in/up.
  useSession();
  const returnTo = useLocation().search;
  const target = new URLSearchParams(returnTo).get("returnTo");
  return <AuthPage returnTo={target ?? "/dashboard"} />;
}
const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error("Clipmake: #root is missing from index.html");
}

window.__clipmakeBooted = true;

if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/share-sw.js", { scope: "/" })
      .then((reg) => {
        // Track installation so we can clear stale shell caches the first time
        // a new worker takes over. This worker is deliberately narrow and does
        // NOT precache the app shell, but a leftover worker from a prior build
        // can still serve stale assets if it is still active.
        if (reg.installing) {
          const installing = reg.installing;
          installing.addEventListener("statechange", () => {
            if (installing.state === "installed" && navigator.serviceWorker.controller) {
              logBootEvent("worker-activated", { state: installing.state });
            }
          });
        }
      })
      .catch(() => {
        /* the IndexedDB prefetch cache still works without it */
      });
  });
}

// One-time recovery for devices that got stuck in a reload loop.
// This runs once per page load, before React commits, so it can neuter a bad
// worker and old shell cache before the new load starts.
(() => {
  try {
    const stuckFlag = sessionStorage.getItem("clipmake.bootLoop");
    const wasStuck = stuckFlag === "aborted";
    if (wasStuck) {
      console.warn("[clipmake/boot] recovering from prior boot-loop state");
      if (navigator.serviceWorker) {
        navigator.serviceWorker.getRegistrations().then((regs) => {
          regs.forEach((r) => r.unregister());
        }).catch(() => {});
      }
      if (window.caches) {
        caches.keys().then((names) => {
          Promise.all(
            names
              .filter((n) =>
                n.indexOf("clipmake-shell-") === 0 || n === "clipmake-share-v1",
              )
              .map((n) => caches.delete(n)),
          ).catch(() => {});
        }).catch(() => {});
      }
      sessionStorage.removeItem("clipmake.bootLoop");
      logBootEvent("recovered", { clearedWorker: true, clearedShellCaches: true });
    }
  } catch (e) { /* ignore storage errors on some browsers */ }
})();

function logBootEvent(event: string, detail?: Record<string, unknown>) {
  try {
    (window as any).__clipmakeBootEvents =
      (window as any).__clipmakeBootEvents || [];
    (window as any).__clipmakeBootEvents.push({
      event,
      ts: Date.now(),
      detail,
    });
    if (event === "recovered" || event === "worker-activated") {
      console.info("[clipmake/boot] " + event, detail);
    }
  } catch (e) { /* ignore */ }
}

createRoot(rootEl).render(
  <StrictMode>
    <BootCleanup />
    <RootErrorBoundary>
      <ToolbarErrorBoundary>
        <Suspense fallback={null}>
          <VlyToolbar />
        </Suspense>
      </ToolbarErrorBoundary>        <BrowserRouter>
          <RouteSyncer />
          <AuthProvider>
            <Suspense fallback={<RouteLoading />}>
              <Routes>
                <Route path="/" element={<Landing />} />
                <Route path="/auth" element={<AuthRouteContent />} />
                <Route path="/dashboard" element={<PublicDashboard />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Suspense>
          </AuthProvider>
        </BrowserRouter>

        <Toaster />
    </RootErrorBoundary>
  </StrictMode>,
);
