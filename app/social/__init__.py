""""Post to..." registry.

One place that knows, per platform: what it is called, what it needs before it
works, and whether that need is *minutes* (register an app) or *weeks* (an app
review). The UI renders this table verbatim, so the approval reality lives in one
readable block rather than being scattered across the frontend.

The whole feature is additive: every platform degrades to the existing
download + share-sheet path, and nothing here can block or alter a download.
"""

from __future__ import annotations

import time
from typing import Any

from app.social import instagram, tiktok, x, youtube
from app.social.accounts import AccountStore
from app.social.base import (
    NeedsReconnect,
    NeedsReview,
    NotConfigured,
    PostRejected,
    SocialError,
)

#: slug -> adapter module. Every adapter exposes the same four functions.
PLATFORMS: dict[str, Any] = {
    youtube.SLUG: youtube,
    x.SLUG: x,
    tiktok.SLUG: tiktok,
    instagram.SLUG: instagram,
}

#: What a user has to do before a platform can post, and roughly how long.
#: This is the single source of truth for the "which can I actually use?" answer.
APPROVAL_NOTES: dict[str, dict[str, str]] = {
    "youtube": {
        "gate": "none",
        "headline": "Works now",
        "detail": (
            "Enable YouTube Data API v3 in a Google Cloud project and add an OAuth "
            "client. No App Review. Default quota is 10,000 units/day and a Short "
            "costs 1,600 -- about 6 a day, raisable self-serve."
        ),
        "docs": "https://developers.google.com/youtube/v3/guides/uploading_a_video",
    },
    "x": {
        "gate": "payment",
        "headline": "Needs a billing-enabled X developer account",
        "detail": (
            "The app itself is instant, but X moved to pay-per-use in 2025 and "
            "free-tier posting is no longer reliably available. This is a payment "
            "step on X's side, not an approval queue."
        ),
        "docs": "https://developer.x.com/en/docs/x-api/media/upload/introduction",
    },
    "tiktok": {
        "gate": "audit",
        "headline": "Works now, but posts stay PRIVATE until the audit passes",
        "detail": (
            "Direct Post works unaudited, but TikTok forces every post to "
            "SELF_ONLY and caps unaudited clients at 5 users per rolling 24h. "
            "Pass the Direct Post audit to publish publicly."
        ),
        "docs": "https://developers.tiktok.com/doc/content-posting-api-reference-direct-post",
    },
    "instagram": {
        "gate": "app_review",
        "headline": "Needs Meta App Review -- the slowest of the four",
        "detail": (
            "instagram_content_publish is Advanced Access: a screencast plus a "
            "business/creator IG account, typically days to weeks. Until then only "
            "app-role accounts work, so the UI falls back to download."
        ),
        "docs": "https://developers.facebook.com/documentation/instagram-platform/content-publishing",
    },
}

#: Whether the app's approval state has been confirmed. A user who has passed a
#: review flips this, which is what lets TikTok publish publicly.
_APPROVAL_ENV_KEY = {
    "tiktok": "TIKTOK_APP_AUDITED",
}


def approved(cfg_env: dict[str, str] | None, slug: str) -> bool:
    """Read the audited/reviewed flag from the environment (default: not)."""
    env = cfg_env if cfg_env is not None else {}
    key = _APPROVAL_ENV_KEY.get(slug)
    return bool(key and str(env.get(key, "")).strip().lower() in {"1", "true", "yes"})


def redirect_uri(cfg, slug: str) -> str:
    return f"{cfg.public_url.rstrip('/')}/api/social/{slug}/callback"


def describe(slug: str, cfg, store: AccountStore) -> dict[str, Any]:
    """Everything the UI needs for one platform, and nothing secret."""
    module = PLATFORMS[slug]
    status, reason = module.readiness(cfg, store)
    record = store.get(slug) or {}
    note = APPROVAL_NOTES[slug]
    return {
        "slug": slug,
        "label": module.LABEL,
        "status": status,
        "reason": reason,
        "accountName": record.get("accountName", ""),
        "connected": status == "connected",
        "gate": note["gate"],
        "headline": note["headline"],
        "detail": note["detail"],
        "docs": note["docs"],
    }


def describe_all(cfg, store: AccountStore) -> list[dict[str, Any]]:
    # Ordered best-available first so the most likely thing to work is on top.
    order = ("youtube", "x", "tiktok", "instagram")
    return [describe(slug, cfg, store) for slug in order]


def _fresh_record(slug: str, cfg, store: AccountStore) -> dict[str, Any]:
    """Load a record, refreshing the access token first if it has expired."""
    module = PLATFORMS[slug]
    record = store.get(slug)
    if not record or not record.get("accessToken"):
        raise NeedsReconnect("Connect this account in Settings first")
    if store.is_expired(record):
        record = module.refresh(cfg, record)
        store.save(slug, record)
    return record


def post_video(
    slug: str,
    cfg,
    store: AccountStore,
    *,
    video_path: str,
    caption: str,
    title: str = "",
    privacy: str = "",
) -> dict[str, Any]:
    """Upload one rendered clip to one platform.

    Raises a SocialError subclass the API maps to a status code, so the UI can
    say "Failed -- reconnect account" vs "Failed -- platform rejected it".
    """
    if slug not in PLATFORMS:
        raise NotConfigured(f"unknown platform {slug!r}")
    module = PLATFORMS[slug]
    status, reason = module.readiness(cfg, store)
    if status == "needs_app":
        raise NotConfigured(reason)
    if status == "not_connected":
        raise NeedsReconnect(reason)

    record = _fresh_record(slug, cfg, store)
    from app.social.base import read_video

    kwargs: dict[str, Any] = {
        "video_bytes": read_video(video_path),
        "caption": caption,
    }
    if slug == youtube.SLUG:
        kwargs["title"] = title
        kwargs["privacy"] = privacy or "public"
    if slug == tiktok.SLUG:
        import os

        kwargs["audited"] = approved(dict(os.environ), slug)
        kwargs["privacy"] = privacy

    try:
        result = module.post_video(cfg, record, **kwargs)
    except SocialError:
        raise
    except (OSError, ValueError) as exc:
        raise SocialError(f"could not read the clip to upload: {exc}") from exc

    return {
        "postedAt": time.time(),
        "captionLength": len(caption),
        **result,
    }


__all__ = [
    "APPROVAL_NOTES",
    "PLATFORMS",
    "AccountStore",
    "NeedsReconnect",
    "NeedsReview",
    "NotConfigured",
    "PostRejected",
    "SocialError",
    "approved",
    "describe",
    "describe_all",
    "post_video",
    "redirect_uri",
]
