// Clipmake Download Worker
// Standalone Node HTTP service. Real yt-dlp downloads only, never fake/mock.
// One job at a time. Streams the resulting mp4 back to the caller.
// Token from env or merged args. Auth with timing-safe Bearer comparison.
// No dependency on the main app or its database.

import { createServer } from "node:http";
import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { env } from "node:process";

const TOKEN = (() => {
  const envToken = env.WORKER_TOKEN;
  if (envToken) return envToken;
  const argv = env._ || "";
  const stripped = argv.replace(/^['"]+|['"]+$/g, "").trim();
  if (stripped) return stripped;
  return "";
})();
const PORT = Number(env.WORKER_PORT) || 3001;
const YT = env.YTDLP_BIN || "yt-dlp";
const COOKIES = env.YTDLP_COOKIES || "";
const JS_RUNTIMES = (env.YTDLP_JS_RUNTIMES || "node").split(",").map((s) => s.trim()).filter(Boolean);
const REMOTE_COMPONENTS = (env.YTDLP_REMOTE_COMPONENTS || "ejs:github").split(",").map((s) => s.trim()).filter(Boolean);
const BODY_LIMIT = 1024 * 1024;
const TIMEOUT_MS = 25 * 60 * 1000;
const MAX_OUTPUT_SIZE = 10 * 1024 * 1024;
const HEALTH_TIMEOUT_MS = 15000;

if (!TOKEN) {
  console.error("WORKER_TOKEN required. Set env or pass as arg: node start.js \"token\"");
  process.exit(1);
}
if (!JS_RUNTIMES.length) {
  console.error("No JS runtime configured. Set YTDLP_JS_RUNTIMES (e.g. node,deno).");
  process.exit(1);
}

let busy = false;
let ctxSent = false;

function sha256(s) {
  return crypto.createHash("sha256").update(String(s)).digest();
}
function timingSafeEqual(a, b) {
  const ha = sha256(a);
  const hb = sha256(b);
  return crypto.timingSafeEqual(ha, hb);
}
function auth(req) {
  const h = req.headers["authorization"];
  const raw = typeof h === "string" ? h : "";
  const token = raw.startsWith("Bearer ") ? raw.slice(7) : "";
  if (!token) return false;
  return timingSafeEqual(token, TOKEN);
}
function json(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json", "Content-Length": String(Buffer.byteLength(body)) });
  res.end(body);
}
function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = Buffer.alloc(0);
    req.on("data", (chunk) => {
      data = Buffer.concat([data, chunk]);
      if (data.length > BODY_LIMIT) {
        reject(new Error("request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(data.toString()) || {});
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}
function runYt(args, timeoutMs) {
  return new Promise((resolve) => {
    const child = execFile(YT, args, { timeout: timeoutMs, maxBuffer: MAX_OUTPUT_SIZE }, (err, stdout, stderr) => {
      resolve({ err: err == null ? null : err, stdout: stdout || "", stderr: stderr || "" });
    });
    child.on("error", (err) => resolve({ err, stdout: "", stderr: "" }));
    child.on("close", () => {});
  });
}
function baseArgs() {
  const a = ["--no-playlist", "--no-warnings"];
  for (const rt of JS_RUNTIMES) a.push("--js-runtimes", rt);
  for (const rc of REMOTE_COMPONENTS) a.push("--remote-components", rc);
  if (COOKIES && fs.existsSync(COOKIES)) a.push("--cookies", COOKIES);
  return a;
}

function badRequest(res, msg) {
  json(res, 400, { error: msg });
}
function unauthorized(res) {
  json(res, 401, { error: "unauthorized" });
}

function normalizeHeight(raw) {
  if (raw === undefined || raw === null) return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n)) return undefined;
  const h = Math.round(n);
  if (h < 240) return undefined;
  if (h > 2160) return undefined;
  return h;
}

function formatStringForHeight(maxHeight) {
  return `bv*[height<=${maxHeight}]+ba/b[height<=${maxHeight}]/b`;
}

async function download(req, res) {
  if (busy) {
    json(res, 429, { error: "worker busy, try again soon" });
    return;
  }
  let body;
  try {
    body = await readBody(req);
  } catch (e) {
    badRequest(res, e && e.message ? e.message : "bad request body");
    return;
  }
  const urlRaw = body.url;
  if (typeof urlRaw !== "string" || !/^https?:\/\//i.test(urlRaw)) {
    badRequest(res, "url must be a string starting with http(s)://");
    return;
  }
  const url = urlRaw.trim();
  const maxHeight = normalizeHeight(body.maxHeight);
  const proxy = typeof body.proxy === "string" && body.proxy.trim() ? body.proxy.trim() : undefined;

  if (proxy && !/^https?:\/\//i.test(proxy) && !/^socks[45]/i.test(proxy)) {
    badRequest(res, "proxy must be http, https or socks5");
    return;
  }

  busy = true;
  ctxSent = false;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "clip-"));
  const cleanup = () => {
    busy = false;
    ctxSent = false;
    try {
      fs.rm(dir, { recursive: true, force: true });
    } catch {
      /* best effort */
    }
  };
  try {
    const args = baseArgs().concat([
      "-f",
      maxHeight ? formatStringForHeight(maxHeight) : "bv*+ba/b/best",
      "--merge-output-format", "mp4",
      "-o",
      path.join(dir, "video.%(ext)s"),
    ]);
    if (proxy) args.push("--proxy", proxy);
    args.push(url);

    const r = await runYt(args, TIMEOUT_MS);

    let files;
    try {
      files = fs.readdirSync(dir).filter((f) => /^video\./.test(f) && !f.endsWith(".part"));
    } catch {
      files = [];
    }

    if (r.err || !files.length) {
      let detail = (r.stderr || String(r.err || "no file written")).slice(-800);
      if (proxy && detail.includes(proxy)) {
        detail = proxyRedacted(detail, proxy);
      }
      cleanup();
      json(res, 502, { error: "download failed", detail });
      return;
    }

    const file = path.join(dir, files[0]);
    let stat;
    try {
      stat = fs.statSync(file);
    } catch {
      cleanup();
      json(res, 502, { error: "download failed", detail: "file vanished after yt-dlp finished" });
      return;
    }
    if (!stat.isFile()) {
      cleanup();
      json(res, 502, { error: "download failed", detail: "not a regular file" });
      return;
    }

    res.writeHead(200, {
      "Content-Type": "video/mp4",
      "Content-Length": String(stat.size),
      "Cache-Control": "no-store",
    });
    ctxSent = true;
    const rs = fs.createReadStream(file);
    rs.on("error", () => res.destroy());
    rs.pipe(res);
    res.on("close", cleanup);
  } catch (e) {
    cleanup();
    if (!ctxSent) {
      json(res, 500, { error: "worker error", detail: e && e.message ? e.message : String(e) });
    }
  }
}

function proxyRedacted(detail, proxy) {
  return detail.split(proxy).join("[proxy]");
}

async function health() {
  const v = await runYt(["--version"], HEALTH_TIMEOUT_MS);
  const version = (v.stdout || "").trim() || "unknown";
  let foundRuntime = null;
  try {
    const info = await runYt(["-v"], HEALTH_TIMEOUT_MS);
    const text = (info.stderr || info.stdout || "").toString();
    foundRuntime = pickRuntime(text);
  } catch {
    foundRuntime = "unreachable";
  }
  return {
    ok: true,
    ytdlp: version,
    node: process.version,
    js_runtime: foundRuntime,
    busy,
  };
}

function pickRuntime(text) {
  for (const rt of JS_RUNTIMES) {
    if (text.includes(rt)) return rt;
  }
  return JS_RUNTIMES[0] == null ? null : JS_RUNTIMES[0];
}

function maskToken(token) {
  const t = token.trim();
  if (t.length <= 6) return "***";
  return `${t.slice(0, 4)}…${t.slice(-4)}`;
}

const server = createServer(async (req, res) => {
  try {
    if (!auth(req)) {
      unauthorized(res);
      return;
    }
    if (req.method === "GET" && (req.url === "/health" || req.url === "/health/")) {
      const h = await health();
      json(res, 200, h);
      return;
    }
    if (req.method === "POST" && (req.url === "/download" || req.url === "/download/")) {
      await download(req, res);
      return;
    }
    json(res, 404, { error: "not found" });
  } catch (e) {
    if (!ctxSent) {
      json(res, 500, { error: "worker error" });
    }
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Clipmake download worker listening on 127.0.0.1:${PORT}`);
  console.log(`token=${maskToken(TOKEN)} (full token not logged)`);
});
