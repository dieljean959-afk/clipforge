// ClipForge backend supervisor plugin.
//
// The Freebuff platform runs ONE managed web process: the Vite dev server
// (vite.config.ts `server` block). There is no Procfile/Docker/systemd on this
// platform — so the FastAPI backend is spawned by Vite itself as a supervised
// child process. The existing dev-server proxy (see vite.config.ts /api +
// /clips routes) makes the browser see the backend at the SAME origin, so the
// frontend keeps using relative URLs — no localhost, no hardcoded IPs.
//
// IMPORTANT: this plugin only does its job if Vite's server_listening event fires
// reliably. On the Freebuff managed preview, that event can be inconsistent, which
// means state.starting may never resolve and every /api call can sit waiting on a
// promise that never settles. The middleware now includes a 4s bailout on a stuck
// boot promise, and attachListener resolves state.starting early when the listener
// fires. If a request still times out, that usually means the backend itself did
// not come up, not that the frontend gave up too early.
//
// Guarantees:
// - backend starts with the platform's web process (not manually)
// - if the backend dies, the next /api request triggers a respawn
// - clean teardown when the platform restarts the web process
// - the pipeline (yt-dlp Python API, faster-whisper, ffmpeg subprocess) runs
//   inside that backend process, ON THE PLATFORM, for every job

import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs"; // side-effect only
import http from "node:http"; // side-effect only
import path from "node:path"; // side-effect only
import type { Plugin } from "vite";

const ROOT = path.resolve(__dirname);
const HEALTH = "http://127.0.0.1:8000/api/health";
const LOG_TAG = "[clipforge-backend]";

const state = {
  proc: null as ChildProcess | null,
  python: null as string | null,
  starting: null as Promise<boolean> | null,
  sawBackend: false,
  // Crash-loop protection. A backend that dies on boot (bad import, missing
  // dependency, OOM at import) must not be respawned by every incoming
  // request, or a wedged app turns into a fork bomb.
  attempts: 0,
  lastAttemptAt: 0,
};

// Backoff schedule, ms. Doubles per failed boot, capped.
const BACKOFF_MS = [1_000, 2_000, 5_000, 10_000, 30_000];
const MAX_ATTEMPTS = 12;
// After the crash-loop budget is spent, keep probing at this slow rate rather
// than giving up for good.
//
// This used to latch: once `attempts` reached MAX_ATTEMPTS the middleware
// logged "giving up on respawn. Reload the app to retry." and never tried
// again for the life of the web process. The only escape was restarting the
// managed web process, which the user cannot do — so a backend that was down
// for a couple of minutes (an OOM at import, a redeploy, a transient crash)
// stayed down until someone restarted the app. That is the "API unreachable"
// that kept coming back.
//
// A cooldown preserves the fork-bomb protection the budget existed for —
// still at most one spawn per COOLDOWN_MS — while making the failure
// self-healing instead of requiring a manual reload.
const COOLDOWN_MS = 60_000;

function backoffFor(attempts: number): number {
  if (attempts >= MAX_ATTEMPTS) return COOLDOWN_MS;
  return BACKOFF_MS[Math.min(attempts, BACKOFF_MS.length - 1)];
}

function log(msg: string) {
  console.log(`${LOG_TAG} ${msg}`);
}

function pickPython(): string | null {
  for (const py of [".venv/bin/python", "python3", "python"]) {
    const p = path.isAbsolute(py)
      ? path.join(ROOT, py)
      : py;
    try {
      fs.accessSync(p, fs.constants.X_OK);
      return p;
    } catch {
      // try next
    }
  }
  return null;
}

async function pingBackend(timeoutMs = 1500): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(HEALTH, { timeout: timeoutMs }, (res) => {
      if (res.statusCode !== 200) {
        // Surface the real status — "backend answered but with 400" is a very
        // different failure from "no answer at all" (crash loop / bad request).
        log(`health probe answered HTTP ${res.statusCode} (expected 200)`);
      }
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
    req.on("error", () => resolve(false));
  });
}

async function startBackend(): Promise<boolean> {
  if (state.proc && state.proc.exitCode === null) {
    return pingBackend();
  }
  if (state.python === null) {
    state.python = pickPython();
    if (state.python === null) {
      log("ERROR: no Python interpreter found (need .venv/bin/python or python3)");
      return false;
    }
  }

  // Backend reads PORT-style env through config.toml; we pin host/port so the
  // proxy target in vite.config.ts always matches.
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    CLIPFORGE_HOST: "127.0.0.1",
    CLIPFORGE_PORT: "8000",
    // hint for logs; real limits enforced by app.config at submit time
    PYTHONUNBUFFERED: "1",
  };

  log(`starting FastAPI via ${state.python} (uvicorn app.main:app on 127.0.0.1:8000)...`);
  state.lastAttemptAt = Date.now();
  const proc = spawn(
    state.python,
    [
      "-m", "uvicorn", "app.main:app",
      "--host", "127.0.0.1", "--port", "8000", "--log-level", "info",
      // h11 rejects requests whose header block exceeds this (default 16KB).
      // Deployments behind platform proxies with large cookie headers can hit
      // it, surfacing to the UI as a bare "HTTP 400" — raise it well past any
      // realistic header set.
      "--h11-max-incomplete-event-size", "1048576",
      // Keep-alive: default 5s closes idle connections aggressively; proxies
      // that reuse a connection right at the boundary can get a 400 on the
      // raced request. 75s is the common reverse-proxy default.
      "--timeout-keep-alive", "75",
    ],
    {
      cwd: ROOT,
      env,
      stdio: ["ignore", "inherit", "inherit"], // boot log lands in the platform web-process log
    },
  );

  state.proc = proc;
  proc.on("exit", (code, signal) => {
    log(`backend exited (code=${code} signal=${signal})`);
    state.proc = null;
  });
  proc.on("error", (err) => {
    log(`backend spawn error: ${err.message}`);
    state.proc = null;
  });

  // wait for the API to answer (imports + whisper import are not trivial)
  const deadline = Date.now() + 30_000;
  let up = false;
  while (Date.now() < deadline) {
    up = await pingBackend(1200);
    if (up) break;
    if (state.proc === null) break; // died during boot
    await new Promise((r) => setTimeout(r, 400));
  }
  if (up) {
    state.sawBackend = true;
    state.attempts = 0; // healthy again — the backoff budget is restored
    log("backend is UP on 127.0.0.1:8000 — /api proxy is live (same-origin)");
  } else {
    state.attempts += 1;
    log(
      `ERROR: backend did not answer /api/health within 30s ` +
        `(attempt ${state.attempts}/${MAX_ATTEMPTS}, next retry in ` +
        `${backoffFor(state.attempts) / 1000}s)`,
    );
  }
  return up;
}

// The platform runs the Vite dev server as a single supervised web process and
// the preview proxy answers 502 for every path the moment that process exits.
// Nothing in the browser can explain a 502 — index.html is never served, so any
// in-app recovery UI (including the boot fallback) is unreachable by definition.
//
// So the only real defence is to keep the process alive. Node's default for an
// unhandled rejection is to throw and terminate; a single stray rejection
// anywhere in a dev-server plugin chain therefore turns a working preview into
// a 502 on every path with no way for the user to recover it. Log loudly and
// keep serving — a degraded app beats a dead origin.
let guardsInstalled = false;
function installCrashGuards() {
  if (guardsInstalled) return;
  guardsInstalled = true;
  process.on("unhandledRejection", (reason) => {
    log(
      `WARNING: unhandled rejection (not fatal, preview stays up): ` +
        `${reason instanceof Error ? (reason.stack ?? reason.message) : String(reason)}`,
    );
  });
  process.on("uncaughtException", (err) => {
    log(
      `WARNING: uncaught exception (not fatal, preview stays up): ` +
        `${err instanceof Error ? (err.stack ?? err.message) : String(err)}`,
    );
  });
}

export function clipforgeBackend(): Plugin {
  return {
    name: "clipforge-backend-supervisor",
    // 'pre' so the backend boots before the first page request can proxy /api
    enforce: "pre",

    configureServer(server) {
      // Synchronous from here on: an async hook turns every throw inside it
      // into a rejected promise, which Node escalates to a process exit — a
      // way for a plugin bug to kill the whole origin.
      installCrashGuards();
      // Freebuff's toolbox proxy sometimes does not fire the Vite
      // server_listening event cleanly, so state.starting can stay
      // pending and /api calls can block on a promise that never
      // settles. Force a known-good listener once the server is ready.
      let listenerAttached = false;
      function onServerListening() {
        if (!state.starting) return;
        // Help the first /api request along: if backend is already up,
        // resolve the pending boot promise immediately.
        pingBackend(800).then((up) => {
          if (up) {
            state.starting = Promise.resolve(true);
          }
        });
      }
      const attachListener = () => {
        if (listenerAttached) return;
        listenerAttached = true;
        // Vite's WebSocket server can be in one of two shapes depending on
        // version/process state. Use whatever "listening" hook is available,
        // and fall back to polling the ready state briefly.
        try {
          if (server.httpServer && server.httpServer.listening) {
            onServerListening();
          } else if (server.httpServer) {
            server.httpServer.on("listening", onServerListening);
          }
        } catch {
          // Best-effort: if this host's Vite wrapper does not expose a
          // usable listener hook, the middleware bailout is the recovery
          // path, not this callback.
        }
      };
      // Start the backend, but do NOT block the dev server on it.
      //
      // This used to `await startBackend()`, which holds Vite's startup for up
      // to 30 seconds while it polls /api/health. The backend's cold boot is
      // not instant — it imports faster-whisper, probes host limits, and on a
      // cold container may pull model files — so a slow boot made the ENTIRE
      // site unreachable, not just the API: Vite was not listening yet, so the
      // browser got no HTML, no JS and no /api at all, and the preview looked
      // dead rather than "warming up".
      //
      // The respawn middleware below already waits on an in-flight boot, so
      // publishing `state.starting` here gives the same guarantee for the
      // first API call without holding the whole dev server hostage.
      //
      // The `.catch` is load-bearing, not defensive noise. `state.starting` is
      // a FLOATING promise until the first /api request arrives and awaits it.
      // Node's default for an unhandled rejection is to THROW, which exits the
      // process — so a boot that rejected (or a future line that throws) would
      // take the whole Vite dev server down, and the platform's proxy would
      // answer 502 on every path with nothing in the browser to explain it.
      // Before this was made non-blocking, `await startBackend()` meant Vite
      // owned that rejection instead.
      state.starting = startBackend()
        .catch((err: unknown) => {
          log(`backend boot failed: ${err instanceof Error ? err.message : String(err)}`);
          return false;
        })
        .finally(() => {
          state.starting = null;
        });

      // Respawn guard: any proxied /api request revives a dead backend first.
      //
      // This used to require `state.sawBackend`, i.e. "only revive a backend
      // this process had already seen alive". That made a boot-time failure
      // PERMANENT: if the first startBackend() crashed, sawBackend stayed
      // false forever, so the on-demand path never fired and every request
      // 500'd until the whole web process was restarted. A crash loop and a
      // failed first boot are the same problem and deserve the same recovery.
      //
      // The reviving request now WAITS for the backend instead of being
      // proxied into a guaranteed 500 — so the first page load after a crash
      // succeeds rather than showing the user an error and needing a refresh.
      server.middlewares.use(async (req, _res, next) => {
        const url = req.url ?? "";
        if (url.startsWith("/api") || url.startsWith("/clips")) {
          const deadline = Date.now() + 4000;
          while (state.starting && Date.now() < deadline) {
            await state.starting;
          }
          if (state.starting === null) {
            state.starting = startBackend()
              .catch((err: unknown) => {
                log(`backend cold-start failed: ${err instanceof Error ? err.message : String(err)}`);
                return false;
              })
              .finally(() => {
                state.starting = null;
              });
            await state.starting;
          }
          if (state.proc === null) {
            const sinceLast = Date.now() - state.lastAttemptAt;
            const wait = backoffFor(state.attempts);
            const exhausted = state.attempts >= MAX_ATTEMPTS;
            if (sinceLast >= wait) {
              log(
                `backend down — respawning on demand (attempt ${state.attempts + 1}` +
                `${exhausted ? `, crash-loop budget spent, retrying every ${COOLDOWN_MS / 1000}s` : ""})`,
              );
              state.starting = state.starting ?? startBackend()
                .catch((err: unknown) => {
                  log(`backend respawn failed: ${err instanceof Error ? err.message : String(err)}`);
                  return false;
                })
                .finally(() => {
                  state.starting = null;
                });
              await state.starting;
            }
          }
        }
        next();
      });

      const cleanup = () => {
        if (state.proc) {
          log("platform web process stopping — terminating backend");
          state.proc.kill("SIGTERM");
          state.proc = null;
        }
      };
      process.on("exit", cleanup);
      process.on("SIGINT", () => {
        cleanup();
        process.exit(0);
      });
      process.on("SIGTERM", () => {
        cleanup();
        process.exit(0);
      });
    },
  };
}