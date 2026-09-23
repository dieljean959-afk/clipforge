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
};

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
  const proc = spawn(
    state.python,
    ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
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
    log("backend is UP on 127.0.0.1:8000 — /api proxy is live (same-origin)");
  } else {
    log("ERROR: backend did not answer /api/health within 30s");
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

      // respawn guard: any proxied /api request revives a dead backend first
      server.middlewares.use((req, _res, next) => {
        const url = req.url ?? "";
        if (url.startsWith("/api") || url.startsWith("/clips")) {
          if (state.proc === null && state.sawBackend) {
            log("backend down — respawning on demand");
            state.starting = state.starting ?? startBackend().finally(() => {
              state.starting = null;
            });
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
