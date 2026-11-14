"""Instagram Reels via the Instagram Graph API.

The approval reality, which is the part that matters:
  - Creating a Meta app is instant, but ``instagram_content_publish`` is an
    **Advanced Access** permission, so it needs **App Review**: a screencast
    walking through the flow, often days to weeks, and a business/creator
    Instagram account that owns the media.
  - Until it is approved the app can only publish to accounts explicitly added
    as app testers/roles in Development mode, and is rate-limited.

So this is the one that genuinely cannot be made public-facing on demand. The
connector is implemented and the UI reports `needs_review` with a link to the
permission request rather than pretending the button will work; posting falls
back to download + share sheet until the review passes.
"""

from __future__ import annotations

import time
import urllib.parse

from app.social.base import (
    NeedsReconnect,
    PostRejected,
    bearer,
    request_json,
)

SLUG = "instagram"
LABEL = "Instagram Reels"
AUTH_URL = "https://www.facebook.com/v21.0/dialog/oauth"
TOKEN_URL = "https://graph.facebook.com/v21.0/oauth/access_token"
GRAPH = "https://graph.facebook.com/v21.0"
SCOPES = "instagram_basic,instagram_content_publish,pages_show_list"
PERMISSION = "instagram_content_publish"
#: Processing a Reels container takes a moment; the container must report FINISHED.
_POLL_SEC = 3
_POLL_LIMIT = 40  # ~2 minutes


def readiness(cfg, store) -> tuple[str, str]:
    if not cfg.meta_app_id or not cfg.meta_app_secret:
        return "needs_app", "Add META_APP_ID and META_APP_SECRET"
    if not store.connected(SLUG):
        return "not_connected", "Connect an Instagram account in Settings"
    return "connected", ""


def authorize_url(cfg, *, redirect_uri: str, state: str) -> str:
    params = {
        "client_id": cfg.meta_app_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(cfg, *, code: str, redirect_uri: str) -> dict:
    # Long-lived tokens: a 60-day token so the user does not reconnect weekly.
    status, data, _ = request_json(
        f"{TOKEN_URL}?{urllib.parse.urlencode({'client_id': cfg.meta_app_id, 'client_secret': cfg.meta_app_secret, 'redirect_uri': redirect_uri, 'code': code, 'grant_type': 'authorization_code'})}",
        method="GET",
    )
    if status >= 400 or not data.get("access_token"):
        raise NeedsReconnect(f"Meta rejected the authorization code: {_reason(data)}")
    short = data["access_token"]
    long_status, long_data, _ = request_json(
        f"{TOKEN_URL}?{urllib.parse.urlencode({'grant_type': 'fb_exchange_token', 'client_id': cfg.meta_app_id, 'client_secret': cfg.meta_app_secret, 'fb_exchange_token': short})}"
    )
    token = long_data.get("access_token", short) if long_status < 400 else short
    rec = {
        "accessToken": token,
        "refreshToken": "",  # Meta long-lived tokens do not use refresh tokens
        "expiresAt": time.time() + float(long_data.get("expires_in", 5184000)),
        "igUserId": "",
    }
    rec["accountName"] = _account_name(rec)
    return rec


def _account_name(rec: dict) -> str:
    status, data, _ = request_json(
        f"{GRAPH}/me?fields=id,name,username",
        headers=bearer(rec["accessToken"]),
        timeout=30,
    )
    if status >= 400:
        return ""
    rec["igUserId"] = str(data.get("id", ""))
    return str(data.get("username") or data.get("name") or "")


def refresh(cfg, record: dict) -> dict:
    """No-op: Meta issues 60-day long-lived tokens with no refresh endpoint.

    The expiry is still surfaced to the UI as `not_connected` once passed, which
    sends the user through reconnect -- the honest equivalent of a refresh.
    """
    return record


def post_video(
    cfg,
    record: dict,
    *,
    video_bytes: bytes,
    caption: str,
    **_: object,
) -> dict:
    ig_user = record.get("igUserId")
    if not ig_user:
        raise NeedsReconnect("Reconnect Instagram -- the stored token has no IG user id")
    auth = bearer(record["accessToken"])

    # Reels use a two-step container: create the container, push the bytes, poll
    # until Meta finishes processing, then publish the container id.
    status, data, _ = request_json(
        f"{GRAPH}/{ig_user}/media",
        method="POST",
        headers=auth,
        payload={
            "media_type": "REELS",
            "video_url": "",  # binary upload mode needs no public URL
            "caption": caption[:2200],
        },
    )
    if status >= 400 or "id" not in data:
        raise PostRejected(
            f"Instagram refused the Reel container: {_reason(data)}. "
            f"If this says a permission is missing, {PERMISSION} still needs App Review."
        )
    container = str(data["id"])

    from app.social.base import raw_request

    up_status, up_data, _ = raw_request(
        f"{GRAPH}/{container}",
        method="POST",
        headers={**auth, "Content-Type": "application/octet-stream"},
        body=video_bytes,
        timeout=cfg.request_timeout_sec,
    )
    if up_status >= 400:
        raise PostRejected(f"Instagram refused the video bytes: {_reason(up_data)}")

    _await_finished(auth, container)

    status, data, _ = request_json(
        f"{GRAPH}/{ig_user}/media_publish",
        method="POST",
        headers=auth,
        payload={"creation_id": container},
    )
    if status >= 400 or "id" not in data:
        raise PostRejected(f"Instagram refused to publish the Reel: {_reason(data)}")
    media_id = str(data["id"])
    return {
        "platform": SLUG,
        "id": media_id,
        "url": f"https://www.instagram.com/reel/{media_id}/",
        "privacy": "public",
    }


def _await_finished(auth: dict, container: str) -> None:
    for _ in range(_POLL_LIMIT):
        time.sleep(_POLL_SEC)
        status, data, _ = request_json(
            f"{GRAPH}/{container}?fields=status_code,status",
            headers=auth,
            timeout=30,
        )
        if status >= 400:
            raise PostRejected(f"Instagram container check failed: {_reason(data)}")
        if data.get("status_code") == "FINISHED":
            return
        if data.get("status_code") == "ERROR":
            raise PostRejected(f"Instagram could not process the Reel: {data.get('status')}")
    raise PostRejected("Instagram was still processing the Reel when the timeout elapsed")


def _reason(data: object) -> str:
    if not isinstance(data, dict):
        return str(data)[:200]
    err = data.get("error")
    if isinstance(err, dict):
        # (code, subcode) is what actually identifies the missing permission.
        code = err.get("code")
        sub = err.get("error_subcode")
        msg = err.get("message") or ""
        return f"{msg} (code {code}{f', subcode {sub}' if sub else ''})"[:300]
    return str(data)[:200]
