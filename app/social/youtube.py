"""YouTube Shorts via the YouTube Data API v3.

The only one of the four that is fully usable today: creating an OAuth client in
Google Cloud and enabling the API takes minutes, and ``youtube.upload`` needs no
App Review. A Google project also gets a default 10,000-unit/day quota, and one
Shorts upload costs 1600 units -- about six uploads a day. Raising that is a
self-serve quota request, not a review.

Video is sent with a resumable upload: a POST opens the session and returns a
Location, then the bytes go to that Location in a single PUT.
"""

from __future__ import annotations

import json
import urllib.parse

from app.social.base import (
    NeedsReconnect,
    PostRejected,
    bearer,
    request_json,
)

SLUG = "youtube"
LABEL = "YouTube Shorts"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
API_KEY_URL = "https://www.googleapis.com/youtube/v3/channels"
SCOPES = (
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
)
# ONE Shorts upload = 1600 quota units; the default daily grant is 10,000.
QUOTA_COST = 1600


def readiness(cfg, store) -> tuple[str, str]:
    if not cfg.youtube_client_id or not cfg.youtube_client_secret:
        return "needs_app", "Add YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET"
    if not store.connected(SLUG):
        return "not_connected", "Connect a Google account in Settings"
    return "connected", ""


def authorize_url(cfg, *, redirect_uri: str, state: str) -> str:
    params = {
        "client_id": cfg.youtube_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",  # without this there is no refresh token
        "prompt": "consent",  # ditto: a silent re-auth returns no refresh token
        "include_granted_scopes": "true",
        "state": state,
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(cfg, *, code: str, redirect_uri: str) -> dict:
    status, data, _ = request_json(
        TOKEN_URL,
        method="POST",
        form={
            "code": code,
            "client_id": cfg.youtube_client_id,
            "client_secret": cfg.youtube_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
    )
    if status >= 400 or "access_token" not in data:
        raise NeedsReconnect(
            f"Google rejected the authorization code: {data.get('error_description') or data}"
        )
    return _record(data)


def refresh(cfg, record: dict) -> dict:
    if not record.get("refreshToken"):
        raise NeedsReconnect("Reconnect: the stored Google grant has no refresh token")
    status, data, _ = request_json(
        TOKEN_URL,
        method="POST",
        form={
            "refresh_token": record["refreshToken"],
            "client_id": cfg.youtube_client_id,
            "client_secret": cfg.youtube_client_secret,
            "grant_type": "refresh_token",
        },
    )
    if status >= 400 or "access_token" not in data:
        raise NeedsReconnect("Google rejected the refresh token -- reconnect the account")
    # A refresh response omits refresh_token, so the old one is carried over.
    return _record({**data, "refresh_token": record["refreshToken"]})


def _record(data: dict) -> dict:
    import time

    return {
        "accessToken": data["access_token"],
        "refreshToken": data.get("refresh_token", ""),
        "expiresAt": time.time() + float(data.get("expires_in", 3600)),
        "scope": data.get("scope", ""),
    }


def post_video(
    cfg,
    record: dict,
    *,
    video_bytes: bytes,
    caption: str,
    title: str = "",
    privacy: str = "public",
    **_: object,
) -> dict:
    """Upload one Short. ``privacy`` is public | unlisted | private."""
    if privacy not in {"public", "unlisted", "private"}:
        raise PostRejected(f"unknown privacyStatus {privacy!r}")
    body = {
        "snippet": {
            "title": (title or caption)[:100],
            "description": caption,
            # A Short is a vertical video; the API rejects one that is not.
            "categoryId": "22",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }
    query = urllib.parse.urlencode(
        {"part": "snippet,status", "uploadType": "resumable"}
    )
    status, data, headers = request_json(
        f"{UPLOAD_URL}?{query}",
        method="POST",
        headers={
            **bearer(record["accessToken"]),
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(len(video_bytes)),
            "Content-Type": "application/json; charset=UTF-8",
        },
        payload=body,
        timeout=cfg.request_timeout_sec,
    )
    location = headers.get("Location") or headers.get("location") or ""
    if status >= 400 or not location:
        reason = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
        raise PostRejected(f"YouTube refused the upload: {reason or data}")

    from app.social.base import raw_request

    up_status, raw, _ = raw_request(
        location,
        method="PUT",
        headers={"Content-Type": "video/mp4"},
        body=video_bytes,
        timeout=cfg.request_timeout_sec,
    )
    if up_status >= 400:
        try:
            reason = json.loads(raw).get("error", {}).get("message")
        except json.JSONDecodeError:
            reason = raw[:200].decode("utf-8", "replace")
        raise PostRejected(f"YouTube refused the video bytes: {reason}")

    video = json.loads(raw)
    return {
        "platform": SLUG,
        "id": video.get("id", ""),
        "url": f"https://www.youtube.com/shorts/{video.get('id', '')}",
        "privacy": privacy,
    }
