// ClipForge backend supervisor plugin.
//
// The Freebuff platform runs ONE managed web process: the Vite dev server
// (vite.config.ts `server` block). There is no Procfile/Docker/systemd on this
// platform — so the FastAPI backend is spawned by Vite itself as a supervised
// child process. The existing dev-server proxy (see vite.config.ts /api +
// /clips routes) makes the browser see the backend at the SAME origin, so the
// frontend keeps using relative URLs — no localhost, no hardcoded IPs.
//
// Guarantees:
// - backend starts with the platform's web process (not manually)
// - if the backend dies, the next /api request triggers a respawn
// - clean teardown when the platform restarts the web process
// - the pipeline (yt-dlp Python API, faster-whisper, ffmpeg subprocess) runs
//   inside that backend process, ON THE PLATFORM, for every job

import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
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

export function clipforgeBackend(): Plugin {
  return {
    name: "clipforge-backend-supervisor",
    // 'pre' so the backend boots before the first page request can proxy /api
    enforce: "pre",

    async configureServer(server) {
      await startBackend();

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
          if (state.proc === null) {
            const sinceLast = Date.now() - state.lastAttemptAt;
            const wait = backoffFor(state.attempts);
            const exhausted = state.attempts >= MAX_ATTEMPTS;
            if (sinceLast >= wait) {
              log(
                `backend down — respawning on demand (attempt ${state.attempts + 1}` +
                `${exhausted ? `, crash-loop budget spent, retrying every ${COOLDOWN_MS / 1000}s` : ""})`,
              );
              state.starting = state.starting ?? startBackend().finally(() => {
                state.starting = null;
              });
            }
          }
          if (state.starting) {
            await state.starting;
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
