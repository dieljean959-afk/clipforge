"""Permanent public share links.

A clip's share link is the one artefact in this app that has to outlive
everything else: the batch it came from, the host it was rendered on, the
browser that first opened the dashboard, and any temp-file cleanup. It is a
short random id stored on the clip document, and it resolves to two public
URLs:

    /s/<id>.mp4    the share-sized MP4 (no login, no expiry)
    /s/<id>        a page with Open Graph / Twitter Card tags, so a link
                   pasted into a social app unfurls with the video

Why an id and not the clip's own id: `jobId`-based URLs leak the batch id, and
the clip id is a 24-hex document id that says nothing to a human reading a
link. The id here is short, random, and stable for the life of the clip.

Readiness is a real filesystem check, never an assumption. `share_state()`
answers "is the public URL actually serving bytes right now", so the UI can say
"Preparing…" until it is true and only then say Ready.
"""

from __future__ import annotations

import re
import secrets
import threading
from pathlib import Path
from typing import Any

#: 10 characters from a 32-symbol alphabet with the visually ambiguous ones
#: removed (0/O, 1/I/L, 8/B). These ids get read aloud, typed by hand and
#: pasted into apps that mangle case, so the alphabet earns its keep.
_ALPHABET = "2345679ACDEFGHJKMNPQRSTUVWXYZ"
ID_LEN = 10

#: A share id in a URL path segment. Anchored, so a bad path is a 404 rather
#: than a filesystem lookup with user input in it.
ID_RE = re.compile(r"^[2345679ACDEFGHJKMNPQRSTUVWXYZ]{%d}$" % ID_LEN)

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

#: A share state is one of these, and the UI switches on them.
STATE_MISSING = "missing"      # no id yet — the clip predates share links
STATE_PREPARING = "preparing"  # id exists, the share copy is being built
STATE_READY = "ready"          # verified: real bytes on disk, mp4, size > 0
STATE_FAILED = "failed"        # the build failed; the reason is in the state


def new_id() -> str:
    """A fresh random share id."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(ID_LEN))


def valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(ID_RE.match(value))


def lock_for(clip_id: str) -> threading.Lock:
    """One build lock per clip, so two taps cannot transcode it twice."""
    with _locks_guard:
        lock = _locks.get(clip_id)
        if lock is None:
            lock = threading.Lock()
            _locks[clip_id] = lock
        return lock


#: New-style share links. A 7-day expiry is short enough that an accidental
#: post is not permanent, and long enough to be useful; the UI can revoke it at
#: any time. Stored in the clip document, so it survives a restart.
TOKEN_TTL_MS = 7 * 24 * 60 * 60 * 1000


def new_token() -> str:
    """A URL-safe, unguessable token (not the short human-typed id)."""
    return secrets.token_urlsafe(18)


def ensure_token(clip: dict[str, Any]) -> str:
    """Return this clip's live token, minting a fresh 7-day one if needed."""
    import time

    token = clip.get("shareToken")
    expires = int(clip.get("shareExpiresAt") or 0)
    now = int(time.time() * 1000)
    if (
        isinstance(token, str)
        and len(token) >= 16
        and not clip.get("shareRevoked")
        and expires > now
    ):
        return token
    token = new_token()
    clip["shareToken"] = token
    clip["shareExpiresAt"] = now + TOKEN_TTL_MS
    clip["shareRevoked"] = False
    return token


def token_state(clip: dict[str, Any]) -> dict[str, Any]:
    """Whether a stored token is currently redeemable, with the real reason."""
    import time

    token = clip.get("shareToken")
    expires = int(clip.get("shareExpiresAt") or 0)
    now = int(time.time() * 1000)
    if not isinstance(token, str) or len(token) < 16:
        return {"token": None, "state": "missing", "detail": "no share link yet", "expiresAt": None}
    if clip.get("shareRevoked"):
        return {"token": token, "state": "revoked", "detail": "this share link was revoked", "expiresAt": expires or None}
    if expires <= now:
        return {"token": token, "state": "expired", "detail": "this share link expired after 7 days", "expiresAt": expires or None}
    return {"token": token, "state": "ready", "detail": "ready", "expiresAt": expires}


def ensure_id(clip: dict[str, Any]) -> str | None:
    """Return the clip's share id, minting and persisting one if needed.

    Returns None when there is no clip to attach an id to. The write is
    best-effort: an id that could not be persisted is still returned, because
    a link that works now beats an error, and the next call will try again.
    """
    sid = clip.get("shareId")
    if valid_id(sid):
        return sid
    sid = new_id()
    clip["shareId"] = sid
    clip.setdefault("shareState", STATE_PREPARING)
    return sid


def mp4_url(sid: str) -> str:
    return f"/s/{sid}.mp4"


def page_url(sid: str) -> str:
    return f"/s/{sid}"


def probe(path: Path | None) -> tuple[str, str]:
    """Is this file actually servable? Returns `(state, detail)`.

    A size check alone is not enough — a transcoder killed by the host can
    leave a zero-byte or truncated file, and a "Ready" badge on a link that
    404s is worse than no badge at all. So: exists, non-empty, and starts with
    an MP4 box header.
    """
    if path is None:
        # No copy yet is PREPARING, not "missing": "missing" means the clip has
        # no id at all, which is a different problem with a different fix.
        return STATE_PREPARING, "share copy not built yet"
    try:
        if not path.exists():
            return STATE_PREPARING, "share copy not built yet"
        size = path.stat().st_size
        if size <= 0:
            return STATE_FAILED, "share copy is empty (the encode was killed)"
        with path.open("rb") as fh:
            head = fh.read(12)
        if b"ftyp" not in head:
            return STATE_FAILED, "share copy is not a valid MP4"
    except OSError as exc:
        return STATE_FAILED, f"share copy unreadable: {exc}"
    return STATE_READY, "ready"


def share_state(clip: dict[str, Any], share_path: Path | None) -> dict[str, Any]:
    """The per-clip share facts the UI renders.

    `ready` is deliberately the AND of "has an id" and "the bytes check out",
    so a green tick can never be shown for a link that does not serve.
    """
    sid = clip.get("shareId")
    state, detail = probe(share_path)
    if not valid_id(sid):
        state, detail = STATE_MISSING, "no share id yet"
    ready = state == STATE_READY and valid_id(sid)
    return {
        "shareId": sid if valid_id(sid) else None,
        "shareState": state,
        "shareDetail": detail,
        "shareReady": ready,
        "shareUrl": mp4_url(sid) if valid_id(sid) else None,
        "sharePageUrl": page_url(sid) if valid_id(sid) else None,
    }


def backfill(store: Any, *, limit: int = 500) -> dict[str, int]:
    """Give every existing clip a share id.

    Runs in the background at boot so a deployment that gains this feature
    overnight does not leave a wall of "Share link" buttons with nothing behind
    them. Only the id is written here — building the share copy is the
    render/prewarm path's job, so this is cheap and never transcodes.
    """
    minted = 0
    scanned = 0
    try:
        clips = store.list_all_clips(limit=limit)
    except Exception:  # noqa: BLE001 - a backfill must never break boot
        return {"scanned": 0, "minted": 0}
    for clip in clips:
        scanned += 1
        if valid_id(clip.get("shareId")):
            continue
        sid = new_id()
        clip["shareId"] = sid
        clip.setdefault("shareState", STATE_PREPARING)
        try:
            store.write_clip(clip)
            minted += 1
        except Exception:  # noqa: BLE001 - keep going through the backlog
            continue
    return {"scanned": scanned, "minted": minted}
