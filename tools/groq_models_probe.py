"""Which Groq models are available to this key right now?

Run when a detection chunk is rejected as "Request too large for model X": the
limit is the MODEL's context window, not the account, so the fix is a different
model — and every model carries its own tokens-per-day bucket, so switching is
also how you get a fresh budget when one is spent.

Reads the key through the app's own config rather than the environment so it
behaves like the rest of the pipeline.

    python tools/groq_models_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402


def main() -> int:
    settings = load_config()
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        print("no Groq key configured")
        return 1
    url = settings.groq.base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            payload = json.loads(resp.read())
    except Exception as exc:
        print(f"could not list models: {exc}")
        return 1
    ids = sorted(m.get("id", "") for m in payload.get("data", []))
    print(f"{len(ids)} model(s) available to this key:")
    for model_id in ids:
        print(f"  {model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
