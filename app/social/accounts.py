"""OAuth token storage for connected social accounts.

Tokens are refresh-token credentials: a leaked access token is a 1-hour
inconvenience, a leaked refresh token is a permanent posting capability on the
user's account. So they are kept in one 0600 file per platform, never in the
job documents, never in a log line, and never returned to the browser.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_MIN_FILE_MODE = 0o600


class AccountStore:
    """Per-platform token documents under ``<data>/accounts/``."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "accounts"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, platform: str) -> Path:
        safe = "".join(c for c in platform if c.isalnum() or c in "-_")
        # Reject rather than sanitise: silently turning "../escape" into
        # "escape.json" would write a token somewhere the caller did not ask for
        # and report success. A traversal attempt should be a loud failure.
        if not safe or safe != platform:
            raise ValueError(f"invalid platform {platform!r}")
        return self._dir / f"{safe}.json"

    def save(self, platform: str, record: dict[str, Any]) -> None:
        record = {**record, "savedAt": time.time()}
        with _LOCK:
            path = self._path(platform)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
            # Write it private from the moment it exists, not after a chmod that
            # a concurrent read could slip between.
            os.chmod(tmp, _MIN_FILE_MODE)
            tmp.replace(path)

    def get(self, platform: str) -> dict[str, Any] | None:
        with _LOCK:
            path = self._path(platform)
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return None

    def delete(self, platform: str) -> bool:
        with _LOCK:
            path = self._path(platform)
            if not path.exists():
                return False
            path.unlink()
            return True

    def connected(self, platform: str) -> bool:
        return bool((self.get(platform) or {}).get("accessToken"))

    def is_expired(self, record: dict[str, Any], *, skew_sec: int = 120) -> bool:
        expires_at = record.get("expiresAt")
        # No expiresAt means a non-expiring token (some platforms omit it).
        if not expires_at:
            return False
        return time.time() >= (float(expires_at) - skew_sec)
