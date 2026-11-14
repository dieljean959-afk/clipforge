"""TikTok via the Content Posting API (Direct Post).

The approval reality, which is the part that matters:
  - Registering the app is self-serve and instant.
  - Until you pass TikTok's **Direct Post audit**, an app is "unaudited", and
    TikTok hard-limits unaudited clients two ways: every post is forced to
    **PRIVATE (SELF_ONLY)** visibility, and only **5 users may post in a rolling
    24h window**.

So this connector works before approval -- you can prove the whole path end to
end -- but nothing you post is public. That is why `privacy` defaults to
SELF_ONLY here and the API reports `audited: false`, rather than the UI letting
someone post a "private" clip and wonder why nobody can see it.
"""

from __future__ import annotations

import json
import time
import urllib.parse

from app.social.base import (
    NeedsReconnect,
    PostRejected,
    bearer,
    pkce_pair,
    request_json,
)

SLUG = "tiktok"
LABEL = "TikTok"
AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
API = "https://open.tiktokapis.com"
SCOPES = "video.publish video.upload user.info.basic"
#: Unaudited clients can only ever publish this. Anything else is rejected.
UNAUDITED_PRIVACY = "SELF_ONLY"
VALID_PRIVACY = {UNAUDITED_PRIVACY, "PUBLIC_TO_FOLLOWERS", "FRIENDS"}


def readiness(cfg, store) -> tuple[str, str]:
    if not cfg.tiktok_client_key or not cfg.tiktok_client_secret:
        return "needs_app", "Add TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET"
    if not store.connected(SLUG):
        return "not_connected", "Connect a TikTok account in Settings"
    return "connected", ""


def authorize_url(cfg, *, redirect_uri: str, state: str, code_verifier: str = "") -> str:
    _verifier, challenge = pkce_pair()
    params = {
        "client_key": cfg.tiktok_client_key,
        "redirect_uri": redirect_uri,
        "response_type": "code",
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
            "client_key": cfg.tiktok_client_key,
            "client_secret": cfg.tiktok_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
    )
    if status >= 400 or not data.get("access_token"):
        raise NeedsReconnect(
            f"TikTok rejected the authorization code: {data.get('description') or data}"
        )
    rec = {
        "accessToken": data["access_token"],
        "refreshToken": data.get("refresh_token", ""),
        "expiresAt": time.time() + float(data.get("expires_in", 86400)),
        "openId": ((data.get("data") or {}).get("open_id")) or "",
    }
    rec["accountName"] = _profile_name(cfg, rec)
    return rec


def _profile_name(cfg, rec: dict) -> str:
    status, data, _ = request_json(
        f"{API}/v2/user/info/?fields=display_name,username",
        headers=bearer(rec["accessToken"]),
        timeout=30,
    )
    if status >= 400:
        return ""
    user = (data.get("data") or {}).get("user") or {}
    return str(user.get("display_name") or user.get("username") or "")


def refresh(cfg, record: dict) -> dict:
    if not record.get("refreshToken"):
        raise NeedsReconnect("Reconnect: the stored TikTok grant has no refresh token")
    status, data, _ = request_json(
        TOKEN_URL,
        method="POST",
        form={
            "client_key": cfg.tiktok_client_key,
            "client_secret": cfg.tiktok_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": record["refreshToken"],
        },
    )
    if status >= 400 or not data.get("access_token"):
        raise NeedsReconnect("TikTok rejected the refresh token -- reconnect the account")
    return {
        **record,
        "accessToken": data["access_token"],
        "refreshToken": data.get("refresh_token", record.get("refreshToken", "")),
        "expiresAt": time.time() + float(data.get("expires_in", 86400)),
    }


def resolve_privacy(audited: bool, requested: str = "") -> str:
    """The privacy level actually sent, given the app's approval state.

    An unaudited client cannot publish publicly no matter what it asks for --
    TikTok rejects anything but SELF_ONLY. Forcing it here rather than trusting
    the caller's request is what stops a UI from offering a "public" toggle
    that would then fail at upload time.
    """
    if not audited:
        return UNAUDITED_PRIVACY
    return requested if requested in VALID_PRIVACY else UNAUDITED_PRIVACY


def post_video(
    cfg,
    record: dict,
    *,
    video_bytes: bytes,
    caption: str,
    audited: bool = False,
    privacy: str = "",
    **_: object,
) -> dict:
    """Direct-post one video.

    ``audited`` comes from the app's real approval state, not a setting. When
    false the privacy level is forced to SELF_ONLY because TikTok rejects
    anything else from an unaudited client.
    """
    level = resolve_privacy(bool(audited), privacy)
    auth = bearer(record["accessToken"])
    total = len(video_bytes)

    status, data, _ = request_json(
        f"{API}/v2/post/publish/video/init/?{urllib.parse.urlencode({'post_info': _post_info(caption, level), 'source_info': {'source': 'PULL_FROM_URL', 'video_size': total}})}",
        method="POST",
        headers=auth,
        timeout=cfg.request_timeout_sec,
    )
    if status >= 400 or not data.get("data"):
        raise PostRejected(f"TikTok refused to start the upload: {_reason(data)}")
    info = data["data"]
    upload_url = info.get("upload_url", "")
    publish_id = info.get("publish_id", "")
    if not upload_url:
        raise PostRejected("TikTok did not return an upload_url")

    # PULL_FROM_URL with an explicit size means TikTok pulls the bytes itself
    # from `upload_url` in chunked form; APPEND is how we hand them over.
    chunk = min(cfg.upload_chunk_bytes, total) or 1
    sent = 0
    index = 0
    while sent < total:
        part = video_bytes[sent : sent + chunk]
        sent += len(part)
        up_status, up_data, _ = request_json(
            f"{upload_url}?{urllib.parse.urlencode({'chunk_size': len(part), 'chunk_number': index + 1, 'total_size': total, 'total_chunk_count': (total + chunk - 1) // chunk})}",
            method="PUT",
            headers=auth,
            data=part,
            content_type="video/mp4",
            timeout=cfg.request_timeout_sec,
        )
        if up_status >= 400:
            raise PostRejected(f"TikTok refused upload chunk {index + 1}: {_reason(up_data)}")
        index += 1

    status, data, _ = request_json(
        f"{API}/v2/post/publish/?{urllib.parse.urlencode({'publish_id': publish_id})}",
        method="POST",
        headers=auth,
        timeout=cfg.request_timeout_sec,
    )
    if status >= 400:
        raise PostRejected(f"TikTok refused to publish: {_reason(data)}")

    return {
        "platform": SLUG,
        "id": str(data.get("data", {}).get("publish_id", publish_id)),
        "url": "",
        "privacy": level,
        "audited": bool(audited),
        "note": (
            "Posted PRIVATE."
            if level == UNAUDITED_PRIVACY
            else "Posted publicly."
        ),
    }


def _post_info(caption: str, level: str) -> str:
    return json.dumps(
        {
            "title": caption[:150],
            "description": caption[:2200],
            "privacy_level": level,
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
        }
    )


def _reason(data: object) -> str:
    if not isinstance(data, dict):
        return str(data)[:200]
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)[:200]
    return str(data.get("description") or data)[:200]
