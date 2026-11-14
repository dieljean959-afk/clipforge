#!/usr/bin/env python3
"""Self-contained API acceptance probe for the live-key feature.

Boots the real FastAPI app on an ephemeral local port, waits for warmup,
then exercises:

  GET  /api/health                    -> llm.keyLoaded / llm.source / llm.problem
  POST /api/settings/reload-secrets   -> same, after a forced re-read

Results are written to /tmp/cf-probe.json and printed. Exits 0 when the API
surfaces the live key status (whatever it is), 1 on transport errors.

Run: .venv/bin/python tools/api_probe.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PORT = 8765
OUT = Path("/tmp/cf-probe.json")


def main() -> int:
    import uvicorn

    from app.main import app

    results: dict[str, dict] = {}

    def probe() -> None:
        time.sleep(5)
        for path, method in (("/api/health", "GET"), ("/api/settings/reload-secrets", "POST")):
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}{path}", method=method,
                    data=b"" if method == "POST" else None,
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    results[path] = json.load(r)
            except Exception as exc:
                results[path] = {"ERROR": str(exc)}
        OUT.write_text(json.dumps(results, indent=2))

    threading.Thread(target=probe, daemon=True).start()
    server = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="critical"),
        daemon=True,
    )
    server.start()

    deadline = time.time() + 24
    while time.time() < deadline and not results:
        time.sleep(0.3)
    time.sleep(0.5)

    if OUT.exists():
        print(OUT.read_text()[:2000])
    ok = all("ERROR" not in v for v in results.values()) and len(results) == 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
