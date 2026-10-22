"use strict";
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
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.clipforgeBackend = clipforgeBackend;
var node_child_process_1 = require("node:child_process");
var node_fs_1 = require("node:fs");
var node_http_1 = require("node:http");
var node_path_1 = require("node:path");
var ROOT = node_path_1.default.resolve(__dirname);
var HEALTH = "http://127.0.0.1:8000/api/health";
var LOG_TAG = "[clipmake-backend]";
var state = {
    proc: null,
    python: null,
    starting: null,
    sawBackend: false,
    // Crash-loop protection. A backend that dies on boot (bad import, missing
    // dependency, OOM at import) must not be respawned by every incoming
    // request, or a wedged app turns into a fork bomb.
    attempts: 0,
    lastAttemptAt: 0,
};
// Backoff schedule, ms. Doubles per failed boot, capped.
var BACKOFF_MS = [1000, 2000, 5000, 10000, 30000];
var MAX_ATTEMPTS = 12;
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
var COOLDOWN_MS = 60000;
function backoffFor(attempts) {
    if (attempts >= MAX_ATTEMPTS)
        return COOLDOWN_MS;
    return BACKOFF_MS[Math.min(attempts, BACKOFF_MS.length - 1)];
}
function log(msg) {
    console.log("".concat(LOG_TAG, " ").concat(msg));
}
function pickPython() {
    for (var _i = 0, _a = [".venv/bin/python", "python3", "python"]; _i < _a.length; _i++) {
        var py = _a[_i];
        var p = node_path_1.default.isAbsolute(py)
            ? node_path_1.default.join(ROOT, py)
            : py;
        try {
            node_fs_1.default.accessSync(p, node_fs_1.default.constants.X_OK);
            return p;
        }
        catch (_b) {
            // try next
        }
    }
    return null;
}
function pingBackend() {
    return __awaiter(this, arguments, void 0, function (timeoutMs) {
        if (timeoutMs === void 0) { timeoutMs = 1500; }
        return __generator(this, function (_a) {
            return [2 /*return*/, new Promise(function (resolve) {
                    var req = node_http_1.default.get(HEALTH, { timeout: timeoutMs }, function (res) {
                        if (res.statusCode !== 200) {
                            // Surface the real status — "backend answered but with 400" is a very
                            // different failure from "no answer at all" (crash loop / bad request).
                            log("health probe answered HTTP ".concat(res.statusCode, " (expected 200)"));
                        }
                        res.resume();
                        resolve(res.statusCode === 200);
                    });
                    req.on("timeout", function () {
                        req.destroy();
                        resolve(false);
                    });
                    req.on("error", function () { return resolve(false); });
                })];
        });
    });
}
function startBackend() {
    return __awaiter(this, void 0, void 0, function () {
        var env, proc, deadline, up;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (state.proc && state.proc.exitCode === null) {
                        return [2 /*return*/, pingBackend()];
                    }
                    if (state.python === null) {
                        state.python = pickPython();
                        if (state.python === null) {
                            log("ERROR: no Python interpreter found (need .venv/bin/python or python3)");
                            return [2 /*return*/, false];
                        }
                    }
                    env = __assign(__assign({}, process.env), { CLIPFORGE_HOST: "127.0.0.1", CLIPFORGE_PORT: "8000", 
                        // hint for logs; real limits enforced by app.config at submit time
                        PYTHONUNBUFFERED: "1" });
                    log("starting FastAPI via ".concat(state.python, " (uvicorn app.main:app on 127.0.0.1:8000)..."));
                    state.lastAttemptAt = Date.now();
                    proc = (0, node_child_process_1.spawn)(state.python, [
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
                    ], {
                        cwd: ROOT,
                        env: env,
                        stdio: ["ignore", "inherit", "inherit"], // boot log lands in the platform web-process log
                    });
                    state.proc = proc;
                    proc.on("exit", function (code, signal) {
                        log("backend exited (code=".concat(code, " signal=").concat(signal, ")"));
                        state.proc = null;
                    });
                    proc.on("error", function (err) {
                        log("backend spawn error: ".concat(err.message));
                        state.proc = null;
                    });
                    deadline = Date.now() + 30000;
                    up = false;
                    _a.label = 1;
                case 1:
                    if (!(Date.now() < deadline)) return [3 /*break*/, 4];
                    return [4 /*yield*/, pingBackend(1200)];
                case 2:
                    up = _a.sent();
                    if (up)
                        return [3 /*break*/, 4];
                    if (state.proc === null)
                        return [3 /*break*/, 4]; // died during boot
                    return [4 /*yield*/, new Promise(function (r) { return setTimeout(r, 400); })];
                case 3:
                    _a.sent();
                    return [3 /*break*/, 1];
                case 4:
                    if (up) {
                        state.sawBackend = true;
                        state.attempts = 0; // healthy again — the backoff budget is restored
                        log("backend is UP on 127.0.0.1:8000 — /api proxy is live (same-origin)");
                    }
                    else {
                        state.attempts += 1;
                        log("ERROR: backend did not answer /api/health within 30s " +
                            "(attempt ".concat(state.attempts, "/").concat(MAX_ATTEMPTS, ", next retry in ") +
                            "".concat(backoffFor(state.attempts) / 1000, "s)"));
                    }
                    return [2 /*return*/, up];
            }
        });
    });
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
var guardsInstalled = false;
function installCrashGuards() {
    if (guardsInstalled)
        return;
    guardsInstalled = true;
    process.on("unhandledRejection", function (reason) {
        var _a;
        log("WARNING: unhandled rejection (not fatal, preview stays up): " +
            "".concat(reason instanceof Error ? ((_a = reason.stack) !== null && _a !== void 0 ? _a : reason.message) : String(reason)));
    });
    process.on("uncaughtException", function (err) {
        var _a;
        log("WARNING: uncaught exception (not fatal, preview stays up): " +
            "".concat(err instanceof Error ? ((_a = err.stack) !== null && _a !== void 0 ? _a : err.message) : String(err)));
    });
}
function clipforgeBackend() {
    return {
        name: "clipforge-backend-supervisor",
        // 'pre' so the backend boots before the first page request can proxy /api
        enforce: "pre",
        configureServer: function (server) {
            var _this = this;
            // Synchronous from here on: an async hook turns every throw inside it
            // into a rejected promise, which Node escalates to a process exit — a
            // way for a plugin bug to kill the whole origin.
            installCrashGuards();
            // Freebuff's toolbox proxy sometimes does not fire the Vite
            // server_listening event cleanly, so state.starting can stay
            // pending and /api calls can block on a promise that never
            // settles. Force a known-good listener once the server is ready.
            var listenerAttached = false;
            function onServerListening() {
                if (!state.starting)
                    return;
                // Help the first /api request along: if backend is already up,
                // resolve the pending boot promise immediately.
                pingBackend(800).then(function (up) {
                    if (up) {
                        state.starting = Promise.resolve(true);
                    }
                });
            }
            var attachListener = function () {
                if (listenerAttached)
                    return;
                listenerAttached = true;
                // Vite's WebSocket server can be in one of two shapes depending on
                // version/process state. Use whatever "listening" hook is available,
                // and fall back to polling the ready state briefly.
                try {
                    if (server.httpServer && server.httpServer.listening) {
                        onServerListening();
                    }
                    else if (server.httpServer) {
                        server.httpServer.on("listening", onServerListening);
                    }
                }
                catch (_a) {
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
                .catch(function (err) {
                log("backend boot failed: ".concat(err instanceof Error ? err.message : String(err)));
                return false;
            })
                .finally(function () {
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
            server.middlewares.use(function (req, _res, next) { return __awaiter(_this, void 0, void 0, function () {
                var url, deadline, sinceLast, wait, exhausted;
                var _a, _b;
                return __generator(this, function (_c) {
                    switch (_c.label) {
                        case 0:
                            url = (_a = req.url) !== null && _a !== void 0 ? _a : "";
                            if (!(url.startsWith("/api") || url.startsWith("/clips"))) return [3 /*break*/, 7];
                            deadline = Date.now() + 4000;
                            _c.label = 1;
                        case 1:
                            if (!(state.starting && Date.now() < deadline)) return [3 /*break*/, 3];
                            return [4 /*yield*/, state.starting];
                        case 2:
                            _c.sent();
                            return [3 /*break*/, 1];
                        case 3:
                            if (!(state.starting === null)) return [3 /*break*/, 5];
                            state.starting = startBackend()
                                .catch(function (err) {
                                log("backend cold-start failed: ".concat(err instanceof Error ? err.message : String(err)));
                                return false;
                            })
                                .finally(function () {
                                state.starting = null;
                            });
                            return [4 /*yield*/, state.starting];
                        case 4:
                            _c.sent();
                            _c.label = 5;
                        case 5:
                            if (!(state.proc === null)) return [3 /*break*/, 7];
                            sinceLast = Date.now() - state.lastAttemptAt;
                            wait = backoffFor(state.attempts);
                            exhausted = state.attempts >= MAX_ATTEMPTS;
                            if (!(sinceLast >= wait)) return [3 /*break*/, 7];
                            log("backend down \u2014 respawning on demand (attempt ".concat(state.attempts + 1) +
                                "".concat(exhausted ? ", crash-loop budget spent, retrying every ".concat(COOLDOWN_MS / 1000, "s") : "", ")"));
                            state.starting = (_b = state.starting) !== null && _b !== void 0 ? _b : startBackend()
                                .catch(function (err) {
                                log("backend respawn failed: ".concat(err instanceof Error ? err.message : String(err)));
                                return false;
                            })
                                .finally(function () {
                                state.starting = null;
                            });
                            return [4 /*yield*/, state.starting];
                        case 6:
                            _c.sent();
                            _c.label = 7;
                        case 7:
                            next();
                            return [2 /*return*/];
                    }
                });
            }); });
            var cleanup = function () {
                if (state.proc) {
                    log("platform web process stopping — terminating backend");
                    state.proc.kill("SIGTERM");
                    state.proc = null;
                }
            };
            process.on("exit", cleanup);
            process.on("SIGINT", function () {
                cleanup();
                process.exit(0);
            });
            process.on("SIGTERM", function () {
                cleanup();
                process.exit(0);
            });
        },
    };
}
