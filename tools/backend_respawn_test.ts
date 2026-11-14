// Tests for the backend supervisor's recovery logic
// (vite-plugin-clipforge-backend.ts).
//
// The property that matters: a backend that failed to START must still be
// recoverable, and a backend that keeps crashing must not be respawned on
// every request.
//
// The guard used to read:
//
//     if (state.proc === null && state.sawBackend) { /* respawn */ }
//
// `sawBackend` is only set once the API has answered /api/health. So if the
// very first startBackend() crashed — a bad import, a missing dependency, an
// OOM at import time — sawBackend stayed false for the life of the process,
// the on-demand branch never fired, and every proxied /api request returned
// 500 until the entire web process was restarted. A crash loop and a failed
// first boot are the same failure and must recover the same way.
//
// Removing the condition without adding backoff would be worse: a backend
// that dies on boot would be respawned by every incoming request, turning a
// wedged app into a fork bomb. And the reviving request must WAIT for the
// spawn, otherwise the user's first page load after a crash is proxied into
// a guaranteed 500 and they have to refresh.
//
// Run: bun tools/backend_respawn_test.ts

import { readFileSync } from "node:fs";
import path from "node:path";

const SRC = readFileSync(
  path.resolve(import.meta.dir, "..", "vite-plugin-clipforge-backend.ts"),
  "utf8",
);
const GUARD = SRC.slice(SRC.indexOf("server.middlewares.use"));

let failures = 0;

// The argument order here is (name, ok). That is easy to get backwards, and a
// backwards call passes SILENTLY: a boolean lands in `name` and a non-empty
// string lands in `ok`, which is truthy. Two such mistakes shipped in this
// repo's own test suite before. So the helper refuses anything that is not a
// real label, and a miscall now fails loudly instead of reporting a green
// suite that tested nothing.
function check(name: string, ok: boolean, detail = ""): void {
  if (typeof name !== "string" || name.length === 0) {
    failures += 1;
    console.log(
      `   FAIL check() called with a non-string label (${String(name)}) — ` +
        `the arguments are backwards; this assertion did NOT run`,
    );
    return;
  }
  const pass = Boolean(ok);
  console.log(`   ${pass ? "OK " : "FAIL"} ${name}${detail ? " — " + detail : ""}`);
  if (!pass) failures += 1;
}

console.log("== 1. a failed first boot is recoverable");
check(
  "the respawn guard no longer requires sawBackend",
  !/state\.proc === null && state\.sawBackend/.test(GUARD),
);
check(
  "the guard still checks the backend is actually down",
  /state\.proc === null/.test(GUARD),
);
check(
  "sawBackend is still recorded (it drives the boot log, not recovery)",
  /state\.sawBackend\s*=\s*true/.test(SRC),
);

console.log("\n== 2. crash loops are throttled, not forked");
check("a backoff schedule exists", /const BACKOFF_MS/.test(SRC));
check("there is a hard attempt cap", /const MAX_ATTEMPTS/.test(SRC));
check("backoff is computed from the attempt count", /function backoffFor\(/.test(SRC));
check(
  "backoff is capped, so it cannot grow without bound",
  /BACKOFF_MS\[Math\.min\(/.test(SRC),
);
check(
  "the respawn path gives up after the cap instead of looping forever",
  /state\.attempts >= MAX_ATTEMPTS/.test(GUARD),
);
check(
  "giving up tells the operator what to do next",
  /[Rr]eload the app/.test(SRC),
);
check(
  "a respawn is skipped until the backoff window has elapsed",
  /sinceLast >= wait/.test(GUARD),
);

console.log("\n== 3. the reviving request succeeds instead of 500ing");
check("the request awaits the in-flight spawn", /await state\.starting/.test(GUARD));
check(
  "concurrent requests share one spawn rather than racing",
  /state\.starting = state\.starting \?\?/.test(GUARD),
);
check(
  "the starting promise is cleared so later requests can retry",
  /state\.starting = null/.test(SRC),
);
check("a healthy boot restores the backoff budget", /state\.attempts = 0/.test(SRC));

console.log("\n== 4. the plugin still supervises as before");
check("still boots before the first page request", /enforce: "pre"/.test(SRC));
check("an exiting backend clears state.proc", /proc\.on\("exit"/.test(SRC));
check("a spawn error clears state.proc", /proc\.on\("error"/.test(SRC));
check("the backend is terminated on shutdown", /proc\.kill\("SIGTERM"\)/.test(SRC));
check(
  "both /api and /clips trigger revival (the dashboard loads clip media)",
  /url\.startsWith\("\/api"\)/.test(SRC) && /url\.startsWith\("\/clips"\)/.test(SRC),
);
check("the h11 header limit is still raised", /--h11-max-incomplete-event-size/.test(SRC));
check("the keep-alive override is still passed", /--timeout-keep-alive/.test(SRC));
check(
  "the pinned host/port still match the proxy target",
  /CLIPFORGE_HOST: "127\.0\.0\.1"/.test(SRC) && /CLIPFORGE_PORT: "8000"/.test(SRC),
);
check("HMR configuration is untouched by this change", !/hmr\s*:/.test(SRC));

console.log(
  failures === 0
    ? "\nBACKEND RESPAWN: ALL CHECKS PASSED"
    : `\nBACKEND RESPAWN: ${failures} FAILED`,
);
process.exit(failures === 0 ? 0 : 1);
