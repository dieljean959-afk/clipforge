"""Cookie sessions for the local ClipMake instance.

The API used to be open to anyone who could reach the port: no login, no
session, every route callable. This module is the missing half of the auth
client in `src/lib/auth.ts` — the same four endpoints it already speaks:

    GET  /api/auth/me        -> { ok, user?, signupOpen }
    POST /api/auth/signup    -> { ok, user }
    POST /api/auth/signin    -> { ok, user }
    POST /api/auth/signout   -> { ok }

Design choices, kept deliberately small:

* **First account only.** The first sign-up bootstraps the owner. Once a user
  exists, further sign-ups are refused (403) — otherwise "anyone on the port"
  could simply register and walk in. `signupOpen` lets the UI hide the tab.
* **scrypt, stdlib only.** No new dependency, no password in the clear, no
  timing leak in verification (`hmac.compare_digest`).
* **Server-side sessions.** A random 32-byte token in an HttpOnly, SameSite=Lax
  cookie; the server stores only its SHA-256. Sessions survive a restart
  (persisted to `data/auth/state.json`) and expire after 30 days.
* **Automation token.** `data/auth/automation.token` (0600) is accepted in
  `X-ClipForge-Token`. It exists so `tools/doctor.py`, `tools/e2e.py` and
  `app/doctor.py` keep working without a password, while still being something
  only a local process can read — unlike a fully open port.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

COOKIE_NAME = "cf_session"
SESSION_DAYS = 30
#: scrypt cost: ~16 MiB and ~50 ms on one core, which is right for a login.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_MIN_PASSWORD = 8

_LOCK = threading.Lock()
_STATE: dict[str, Any] | None = None
_STATE_AT = 0.0


def _auth_dir() -> Path:
    """`data/auth`, honouring CLIPFORGE_DATA_DIR like the rest of the app."""
    env = os.getenv("CLIPFORGE_DATA_DIR")
    if env:
        return Path(env) / "auth"
    try:
        from app.config import load_config  # noqa: PLC0415 - avoids a cycle at import time

        return Path(load_config(quiet=True).app.paths.data_dir) / "auth"
    except Exception:  # noqa: BLE001 - tooling may import this without a config
        return Path("data") / "auth"


def _state_path() -> Path:
    return _auth_dir() / "state.json"


def _read_state() -> dict[str, Any]:
    global _STATE, _STATE_AT
    path = _state_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _STATE is not None and mtime == _STATE_AT:
        return _STATE
    data: dict[str, Any] = {"users": [], "sessions": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data["users"] = list(raw.get("users") or [])
            data["sessions"] = list(raw.get("sessions") or [])
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 - a corrupt file must not lock the app out
        logger.warning(f"[auth] ignoring unreadable {path}: {exc}")
    _STATE, _STATE_AT = data, mtime
    return data


def _write_state(state: dict[str, Any]) -> None:
    global _STATE, _STATE_AT
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    _STATE, _STATE_AT = state, path.stat().st_mtime


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32
    )
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT_N,
        _SCRYPT_R,
        _SCRYPT_P,
        base64.b64encode(salt).decode(),
        base64.b64encode(digest).decode(),
    )


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        got = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
        return hmac.compare_digest(got, expected)
    except Exception:  # noqa: BLE001 - malformed hash means "no match"
        return False


def _public(user: dict[str, Any]) -> dict[str, Any]:
    return {"id": user.get("id"), "email": user.get("email"), "name": user.get("name")}


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _prune(state: dict[str, Any], now: float | None = None) -> bool:
    now = now or time.time()
    before = len(state["sessions"])
    state["sessions"] = [
        s for s in state["sessions"] if float(s.get("expiresAt") or 0) > now
    ]
    return len(state["sessions"]) != before


def user_count() -> int:
    return len(_read_state()["users"])


def signup_open() -> bool:
    return user_count() == 0


def create_user(email: str, password: str, name: str | None) -> dict[str, Any]:
    email = (email or "").strip().lower()
    if "@" not in email or len(email) > 200:
        raise ValueError("Enter a valid email address.")
    if len(password or "") < _MIN_PASSWORD:
        raise ValueError(f"Use at least {_MIN_PASSWORD} characters.")
    with _LOCK:
        state = _read_state()
        if state["users"]:
            raise PermissionError("This instance already has an owner. Sign in instead.")
        if any(u.get("email") == email for u in state["users"]):
            raise ValueError("An account with that email already exists.")
        user = {
            "id": secrets.token_hex(8),
            "email": email,
            "name": (name or "").strip()[:80] or None,
            "password": _hash_password(password),
            "createdAt": int(time.time() * 1000),
        }
        state["users"].append(user)
        _write_state(state)
        logger.info(f"[auth] owner account created ({email})")
        return _public(user)


def authenticate(email: str, password: str) -> dict[str, Any] | None:
    email = (email or "").strip().lower()
    state = _read_state()
    for user in state["users"]:
        if user.get("email") == email and _verify_password(password or "", str(user.get("password") or "")):
            return _public(user)
    return None


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _LOCK:
        state = _read_state()
        _prune(state, now)
        state["sessions"].append(
            {
                "tokenHash": _token_hash(token),
                "userId": user_id,
                "createdAt": int(now * 1000),
                "expiresAt": int((now + SESSION_DAYS * 86400)),
            }
        )
        _write_state(state)
    return token


def session_user(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    with _LOCK:
        state = _read_state()
        if _prune(state):
            _write_state(state)
        wanted = _token_hash(token)
        for session in state["sessions"]:
            if hmac.compare_digest(str(session.get("tokenHash") or ""), wanted):
                for user in state["users"]:
                    if user.get("id") == session.get("userId"):
                        return _public(user)
    return None


def delete_session(token: str | None) -> None:
    if not token:
        return
    with _LOCK:
        state = _read_state()
        wanted = _token_hash(token)
        kept = [s for s in state["sessions"] if s.get("tokenHash") != wanted]
        if len(kept) != len(state["sessions"]):
            state["sessions"] = kept
            _write_state(state)


def automation_token() -> str:
    """A local-only bearer token for scripts. 0600, created on first use."""
    path = _auth_dir() / "automation.token"
    try:
        token = path.read_text(encoding="utf-8").strip()
        if len(token) >= 32:
            return token
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    logger.info("[auth] created data/auth/automation.token for local tooling")
    return token


def is_automation_token(token: str | None) -> bool:
    if not token:
        return False
    try:
        return hmac.compare_digest(token, automation_token())
    except Exception:  # noqa: BLE001 - never fail open on a broken token file
        return False


def reset_for_tests() -> None:
    """Drop the in-process cache (tests point CLIPFORGE_DATA_DIR at a tmpdir)."""
    global _STATE, _STATE_AT
    _STATE, _STATE_AT = None, 0.0
