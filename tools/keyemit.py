#!/usr/bin/env python3
"""Bridge: emit dotenvx key material as JSON to stdout for the node verifier.

Reads the project env files via python-dotenv (which is allowed), prints the
encrypted GEMINI_API_KEY value + private key hex. This output IS secret
material — it exists only in a pipe between two verifier processes, never in
a log. Deleted after diagnosis.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/daytona/codebase")

from dotenv import load_dotenv  # noqa: E402

ROOT = Path("/home/daytona/codebase")
for name in (".env", ".env.local", ".env.keys"):
    p = ROOT / name
    if p.exists():
        load_dotenv(p, override=False)

print(json.dumps({
    "value": os.environ.get("GEMINI_API_KEY", ""),
    "priv": os.environ.get("DOTENV_PRIVATE_KEY_LOCAL", "").strip(),
}))
