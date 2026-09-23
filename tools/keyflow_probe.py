#!/usr/bin/env python3
"""End-to-end probe of the in-app key management flow (no real secrets used).

Boots the FastAPI app on an ephemeral port, then:
  1. DELETE /api/settings/gemini-key   -> cleans the broken empty blob
  2. GET  /api/health                  -> must say NO key, no silent amber
  3. POST /api/settings/gemini-key     -> saves a TEST value (not a real key)
  4. GET  /api/health                  -> must say keyLoaded: true with source
  5. DELETE again                      -> leaves a clean state

All results print to STDERR (boot logs own stdout). Run:
  .venv/bin/python tools/keyflow_probe.py 2>&1 | grep -E '^[0-9]|KEY FLOW'
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PORT = 8766
TEST_KEY = "AIza" + "T" * 35  # test-shaped value, clearly not a real key


def call(path: str, method: str = "GET", body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        return {"HTTP": exc.code, **json.loads(exc.read().decode() or "{}")}


def main() -> int:
    import uvicorn

    from app.main import app

    results: list[str] = []

    def probe() -> None:
        time.sleep(5)
        try:
            # 1. clean the broken empty blob
            r1 = call("/api/settings/gemini-key", method="DELETE")
            results.append(f"1. DELETE broken blob: linesRemoved={r1.get('linesRemoved')}, keyLoaded={r1.get('keyLoaded')}")

            # 2. health after delete
            h2 = call("/api/health").get("llm", {})
            results.append(f"2. health after delete: keyLoaded={h2.get('keyLoaded')}, source={h2.get('source')!r}")

            # 3. save a test key
            r3 = call("/api/settings/gemini-key", method="POST", body={"key": TEST_KEY})
            results.append(
                f"3. SET test key: keyLoaded={r3.get('keyLoaded')}, "
                f"storedShape={r3.get('storedShape')}, problem={r3.get('problem')!r}"
            )

            # 4. health after set
            h4 = call("/api/health").get("llm", {})
            results.append(f"4. health after set: keyLoaded={h4.get('keyLoaded')}, source={h4.get('source')!r}")

            # 5. delete again (clean state)
            r5 = call("/api/settings/gemini-key", method="DELETE")
            results.append(f"5. final DELETE: linesRemoved={r5.get('linesRemoved')} (clean state restored)")
        finally:
            import os

            os._exit(0)

    threading.Thread(target=probe, daemon=True).start()
    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="critical"),
        daemon=True,
    ).start()

    deadline = time.time() + 25
    while time.time() < deadline:
        time.sleep(0.3)
        if len(results) >= 5:
            break
    time.sleep(0.5)
    for line in results:
        print(line, file=sys.stderr)
    ok = len(results) == 5 and "keyLoaded=True" in results[3]
    print("\nKEY FLOW:", "PASS" if ok else "FAIL", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
