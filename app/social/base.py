"""Shared plumbing for the "Post to..." connectors.

Everything here is stdlib on purpose: the OAuth dance and the four upload
protocols are all plain HTTPS, and adding an SDK per platform would mean four
new dependencies and four more import-time failure modes for no gain.

The User-Agent is not decoration. These endpoints sit behind bot filters that
reject the default ``Python-urllib/3.x`` UA outright (the same Cloudflare 1010
that broke Groq detection earlier), so every request sends a real one.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = (
    "ClipForge/1.2 (+https://github.com/harry0703/MoneyPrinterTurbo)"
)


class SocialError(Exception):
    """Base class so the API can map one failure mode to one HTTP status."""


class NotConfigured(SocialError):
    """No OAuth app credentials for this platform. UI falls back to download."""


class NeedsReview(SocialError):
    """Platform requires an app review/audit the user has not passed yet."""


class NeedsReconnect(SocialError):
    """Token is expired, revoked or was issued to a different app."""


class PostRejected(SocialError):
    """The platform accepted the request and refused the content."""


def raw_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: int = 60,
) -> tuple[int, bytes, dict[str, str]]:
    """One raw HTTP call. Raises SocialError on transport failure, never 4xx/5xx."""
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("User-Agent", USER_AGENT)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        # An HTTPError IS a response: hand it back so the caller can map the
        # platform's own error shape to a useful message.
        return exc.code, exc.read(), dict(exc.headers or {})
    except urllib.error.URLError as exc:
        raise SocialError(f"could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise SocialError(f"timed out talking to {url}") from exc


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: Any = None,
    form: dict[str, str] | None = None,
    data: bytes | None = None,
    content_type: str = "application/octet-stream",
    timeout: int = 60,
) -> tuple[int, Any, dict[str, str]]:
    """JSON in, JSON out. Non-2xx is returned, not raised.

    ``data`` sends raw bytes (a video upload chunk) rather than an encoded
    body; the response is still parsed as JSON where there is one.
    """
    hdrs = {"Accept": "application/json", **(headers or {})}
    if data is not None:
        hdrs.setdefault("Content-Type", content_type)
    elif form is not None:
        data = urllib.parse.urlencode(form).encode()
        hdrs.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif payload is not None:
        data = json.dumps(payload).encode()
        hdrs.setdefault("Content-Type", "application/json")
    status, raw, resp_headers = raw_request(
        url, method=method, headers=hdrs, body=data, timeout=timeout
    )
    try:
        parsed = json.loads(raw.decode("utf-8", "replace")) if raw else {}
    except json.JSONDecodeError:
        parsed = {"raw": raw[:400].decode("utf-8", "replace")}
    return status, parsed, resp_headers


def basic_auth_header(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def pkce_pair() -> tuple[str, str]:
    """A PKCE verifier/challenge pair.

    The verifier must be RETAINED and replayed at the token exchange -- a
    challenge minted inside the authorize URL builder and discarded would make
    every X/TikTok callback fail. Callers store it alongside the CSRF state.
    """
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


def new_state() -> str:
    """CSRF state for the OAuth round trip. Compared on the way back."""
    return secrets.token_urlsafe(24)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def read_video(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()
