"""X (Twitter) via API v2, OAuth 2.0 with PKCE.

Registering a developer app is self-serve and immediate. The catch is the WRITE
tier: X moved to pay-per-use in 2025, and posting from a free-tier app is no
longer reliably available, so this connector needs a billing-enabled developer
account. That is a payment step on X's side, not an approval queue -- worth
knowing, because it is the difference between "minutes" and "indefinite".

Video goes up in three phases (INIT / APPEND / FINALIZE), then the tweet is
created referencing the returned media id. X transcodes asynchronously, so
FINALIZE returns a processing state that must be polled before posting.
"""

from __future__ import annotations

import time
import urllib.parse

from app.social.base import (
    NeedsReconnect,
    PostRejected,
    bearer,
    pkce_pair,
    request_json,
)

SLUG = "x"
LABEL = "X"
AUTH_URL = "https://twitter.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"
MEDIA_URL = "https://api.x.com/2/media/upload"
TWEET_URL = "https://api.x.com/2/tweets"
SCOPES = "tweet.write media.write offline.access"
# X rejects a tweet over this; the hashtag block can push a long one over.
MAX_TWEET_CHARS = 280


def readiness(cfg, store) -> tuple[str, str]:
    if not cfg.x_client_id or not cfg.x_client_secret:
        return (
            "needs_app",
            "Add X_CLIENT_ID and X_CLIENT_SECRET (X now requires a billing-enabled developer account)",
        )
    if not store.connected(SLUG):
        return "not_connected", "Connect an X account in Settings"
    return "connected", ""


def authorize_url(cfg, *, redirect_uri: str, state: str, code_verifier: str = "") -> str:
    _verifier, challenge = pkce_pair()
    params = {
        "response_type": "code",
        "client_id": cfg.x_client_id,
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(cfg, *, code: str, redirect_uri: str, code_verifier: str) -> dict:
    status, data, _ = request_json(
        TOKEN_URL,
        method="POST",
        form={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": cfg.x_client_id,
            "code_verifier": code_verifier,
        },
    )
    if status >= 400 or "access_token" not in data:
        raise NeedsReconnect(
            f"X rejected the authorization code: {data.get('error_description') or data}"
        )
    return _record(data)


def refresh(cfg, record: dict) -> dict:
    if not record.get("refreshToken"):
        raise NeedsReconnect("Reconnect: the stored X grant has no refresh token")
    status, data, _ = request_json(
        TOKEN_URL,
        method="POST",
        form={
            "grant_type": "refresh_token",
            "refresh_token": record["refreshToken"],
            "client_id": cfg.x_client_id,
        },
    )
    if status >= 400 or "access_token" not in data:
        raise NeedsReconnect("X rejected the refresh token -- reconnect the account")
    return _record({**data, "refresh_token": record["refreshToken"]})


def _record(data: dict) -> dict:
    return {
        "accessToken": data["access_token"],
        "refreshToken": data.get("refresh_token", ""),
        "expiresAt": time.time() + float(data.get("expires_in", 7200)),
        "scope": data.get("scope", ""),
    }


def post_video(
    cfg,
    record: dict,
    *,
    video_bytes: bytes,
    caption: str,
    **_: object,
) -> dict:
    auth = bearer(record["accessToken"])
    total = len(video_bytes)
    chunk = min(cfg.upload_chunk_bytes, total) or 1

    status, data, _ = request_json(
        f"{MEDIA_URL}?{urllib.parse.urlencode({'command': 'INIT', 'total_bytes': total, 'media_category': 'tweet_video'})}",
        method="POST",
        headers=auth,
    )
    if status >= 400:
        raise PostRejected(f"X refused the upload session: {_reason(data)}")
    media_id = str((data.get("data") or {}).get("media_id") or "")
    if not media_id:
        raise PostRejected("X did not return a media id for the upload session")

    sent = 0
    index = 0
    while sent < total:
        part = video_bytes[sent : sent + chunk]
        sent += len(part)
        status, data, _ = request_json(
            f"{MEDIA_URL}?{urllib.parse.urlencode({'command': 'APPEND', 'media_id': media_id, 'segment_index': index})}",
            method="POST",
            headers=auth,
            data=part,  # the chunk itself is the body -- not an encoded form
            timeout=cfg.request_timeout_sec,
        )
        if status >= 400:
            raise PostRejected(f"X refused upload chunk {index}: {_reason(data)}")
        index += 1

    status, data, _ = request_json(
        f"{MEDIA_URL}?{urllib.parse.urlencode({'command': 'FINALIZE', 'media_id': media_id})}",
        method="POST",
        headers=auth,
    )
    if status >= 400:
        raise PostRejected(f"X refused to finalize the upload: {_reason(data)}")

    media_id = _await_ready(cfg, auth, media_id, data)

    status, data, _ = request_json(
        TWEET_URL,
        method="POST",
        headers=auth,
        payload={
            "text": caption[:MAX_TWEET_CHARS],
            "media": {"media_ids": [media_id]},
        },
    )
    if status >= 400:
        raise PostRejected(f"X refused the post: {_reason(data)}")
    tweet_id = str((data.get("data") or {}).get("id") or "")
    return {
        "platform": SLUG,
        "id": tweet_id,
        "url": f"https://x.com/i/status/{tweet_id}" if tweet_id else "",
        "privacy": "public",
    }


def _await_ready(cfg, auth: dict, media_id: str, finalize: dict) -> str:
    """Poll X's async transcode until the media is usable."""
    info = ((finalize.get("data") or {}).get("processing_info")) or {}
    if not info or info.get("state") in {None, "succeeded"}:
        return media_id
    deadline = time.monotonic() + cfg.request_timeout_sec
    while time.monotonic() < deadline:
        time.sleep(3)
        status, data, _ = request_json(MEDIA_URL, headers=auth, timeout=30)
        if status >= 400:
            raise PostRejected(f"X media status check failed: {_reason(data)}")
        info = ((data.get("data") or {}).get("processing_info")) or {}
        state = info.get("state")
        if state == "succeeded":
            return media_id
        if state == "failed":
            raise PostRejected(
                f"X could not process the video: {(info.get('error') or {}).get('message', 'unknown')}"
            )
    raise PostRejected("X was still transcoding the video when the timeout elapsed")


def _reason(data: object) -> str:
    if not isinstance(data, dict):
        return str(data)[:200]
    if "detail" in data:
        return str(data["detail"])[:200]
    if "title" in data:
        return str(data["title"])[:200]
    return str(data)[:200]
