#!/usr/bin/env python3
"""Regression checks for the preview boot configuration.

Context
-------
The Freebuff Workspace preview panel intermittently sits at

    "Project connected, booting up preview... 97%"

while the origin is already answering. That bar is the platform's, not ours, but
the wait behind it was ours. Measured from the toolbox proxy log across a real
sandbox boot on 2026-10-04:

  21:27:54  dial tcp [::1]:5173: connect: connection refused   (502)
  21:28:03  GET /            -> 200   (0.93s)
  21:28:08  GET /src/main.tsx -> 200  (16.262s)   <-- entry module blocked
  21:28:28  GET /src/index.css -> 200 (3.58s)

So the HTML was served promptly and the app still could not mount for ~16s. Two
contributing causes, both configuration:

1. `cacheDir` defaulted to `node_modules/.vite`. The sandbox restarted 24 times
   in one day, each time presenting a cold optimizer cache, so esbuild
   re-pre-bundled every dependency synchronously with the first module request.

2. `optimizeDeps.include` listed 6 packages while `src/` imports ~47 bare
   specifiers. The remainder were discovered lazily, mid-request, forcing a
   second optimization round while the page was already loading -- and with
   `hmr: false` the browser never gets the reload notice, so it stalls.

These checks pin the invariants so a future edit cannot quietly reintroduce
either. Run: .venv/bin/python tools/preview_boot_config_test.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "vite.config.ts"

failures: list[str] = []
checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))
        failures.append(label)


def strip_comments(src: str) -> str:
    """Remove // and /* */ comments.

    The invariants below are argued in long comments in the config itself.
    Asserting against raw text would let the prose satisfy the check (or trip
    it), so the source is stripped first and only live config is inspected.
    """
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"(?m)^\s*//.*$", "", src)
    return re.sub(r"(?<![:'\"])//[^\n]*$", "", src, flags=re.M)


def array_items(src: str, key: str) -> list[str]:
    """Extract the quoted entries of the top-level `key: [ ... ]` array."""
    start = src.find(f"{key}: [")
    if start == -1:
        return []
    depth = 0
    body_start = src.index("[", start)
    for i in range(body_start, len(src)):
        if src[i] == "[":
            depth += 1
        elif src[i] == "]":
            depth -= 1
            if depth == 0:
                pairs = re.findall(r"'([^']+)'|\"([^\"]+)\"", src[body_start + 1 : i])
                return [a or b for a, b in pairs]
    return []


def main() -> int:
    print("preview boot config")
    if not CONFIG.exists():
        print(f"  FAIL missing {CONFIG}")
        return 1

    raw = CONFIG.read_text(encoding="utf-8")
    live = strip_comments(raw)

    # --- 1. Dep cache must not live in node_modules -------------------------
    m = re.search(r"cacheDir:\s*['\"]([^'\"]+)['\"]", live)
    check("cacheDir is set explicitly", m is not None, "no cacheDir key found")
    cache_dir = m.group(1) if m else ""
    check(
        "cacheDir is outside node_modules",
        bool(cache_dir) and "node_modules" not in cache_dir,
        f"cacheDir={cache_dir!r}",
    )
    check(
        "cacheDir is inside the synced project root",
        bool(cache_dir) and not cache_dir.startswith("/") and ".." not in cache_dir,
        f"cacheDir={cache_dir!r}",
    )
    check(
        "cacheDir is excluded from the watcher",
        ".vite-cache/**" in live or f"{cache_dir}/**" in live,
        "optimizer cache is not in server.watch.ignored",
    )

    # --- 2. No lazy dep discovery ------------------------------------------
    check(
        "optimizeDeps.noDiscovery is not enabled",
        "noDiscovery: true" not in live,
        "lazy discovery reintroduces the mid-load re-optimize",
    )
    check(
        "optimizeDeps.holdUntilCrawlEnd is false",
        "holdUntilCrawlEnd: false" in live,
        "a held crawl queues entry-module requests behind the optimizer",
    )
    check(
        "optimizeDeps.entries is scoped to the app entry",
        "entries: ['index.html']" in live or 'entries: ["index.html"]' in live,
        "an unscoped crawl picks up unrelated html",
    )

    # --- 3. include must cover every bare import in src/ -------------------
    include = set(array_items(live, "include"))
    check("optimizeDeps.include is populated", len(include) > 20, f"only {len(include)}")

    used: set[str] = set()
    for path in sorted((ROOT / "src").rglob("*.ts")) + sorted((ROOT / "src").rglob("*.tsx")):
        text = path.read_text(encoding="utf-8")
        # Drop comments so prose in a docblock is not mistaken for an import.
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"(?m)^\s*//.*$", "", text)
        for spec in re.findall(
            r"""(?:from|import)\s*\(?\s*['"]([^'"]+)['"]""", text
        ):
            if spec.startswith((".", "/", "@/")):
                continue
            parts = spec.split("/")
            used.add("/".join(parts[:2]) if spec.startswith("@") else parts[0])

    missing = sorted(used - include)
    check(
        "every bare import in src/ is declared in include",
        not missing,
        f"not declared: {missing}",
    )
    print(f"       ({len(used)} bare imports, {len(include)} declared)")

    # --- 4. Platform invariants that must NOT regress -----------------------
    check(
        "HMR stays disabled (platform requirement)",
        re.search(r"\bhmr:\s*false\b", live) is not None,
        "server.hmr must remain false",
    )
    check(
        "server binds dual-stack for the [::1] proxy dial",
        'host: "::"' in live or "host: '::'" in live,
        'IPv4-only binding makes the proxy dial [::1]:5173 fail',
    )
    check(
        "preview host header is not rejected",
        "allowedHosts: true" in live,
        "generated *.freebuff.dev Host header would 403",
    )
    check(
        "api proxy still targets the FastAPI backend",
        '"/api"' in live and "127.0.0.1:8000" in live,
        "/api proxy target changed",
    )

    print()
    if failures:
        print(f"FAILED {len(failures)}/{checks}: {failures}")
        return 1
    print(f"OK {checks}/{checks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
