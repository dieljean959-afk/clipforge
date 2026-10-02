import '@vly-ai/integrations';
import { Toaster } from "@/components/ui/sonner";
import React, { StrictMode, useEffect, lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes, useLocation } from "react-router";
import "./index.css";

// Lazy load route components for better code splitting
const Landing = lazy(() => import("./pages/Landing.tsx"));
const Dashboard = lazy(() => import("./pages/Dashboard.tsx"));
const NotFound = lazy(() => import("./pages/NotFound.tsx"));

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
    __clipforgeBooted?: boolean;
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
    // Also surface it in the document title and body class, so a crash is
    // visible even in a screenshot of a stripped-down preview shell.
    console.error("[WebContainer preview] Root crash:", err);
    document.title = `ClipForge error: ${err.message || "runtime failure"}`;
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

const rootEl = document.getElementById("root");
if (!rootEl) {
  // Should be impossible (index.html ships the element), but a missing mount
  // point is otherwise a silent blank page.
  throw new Error("ClipForge: #root is missing from index.html");
}

// Tell index.html's guards that React has taken over, so its 15s timeout and
// its error listeners stop reporting on a page that is working fine.
window.__clipforgeBooted = true;

// HTTP-layer backstop for share-sized clips.
//
// Registered ONLY in a production build. In dev this would sit in front of the
// Vite module graph, and a cache-first worker that ever touched an app module
// would serve stale code and break HMR for the whole session. It also matches
// only /api/clips/<id>/share (see public/share-sw.js) and passes everything
// else straight through, so even in production it cannot cache app code.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/share-sw.js", { scope: "/" })
      .catch(() => {
        /* the IndexedDB prefetch cache still works without it */
      });
  });
}

createRoot(rootEl).render(
  <StrictMode>
    <BootCleanup />
    <RootErrorBoundary>
      <ToolbarErrorBoundary>
        <Suspense fallback={null}>
          <VlyToolbar />
        </Suspense>
      </ToolbarErrorBoundary>
      <BrowserRouter>
        <RouteSyncer />
        <Suspense fallback={<RouteLoading />}>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
      <Toaster />
    </RootErrorBoundary>
  </StrictMode>,
);
