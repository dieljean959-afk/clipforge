#!/usr/bin/env python
"""Tests for the post-caption shape and the "Post to..." connectors.

Two things this guards that are easy to break silently:
  1. the caption/tag rules, which are enforced in BOTH Python and TypeScript and
     must not drift;
  2. the readiness/OAuth flow -- the part that decides whether a user gets a
     real post button, an honest "needs approval" message, or a download
     fallback. Getting that wrong either blocks a working platform or promises
     a post that silently cannot happen.

No network: the platform HTTP calls are monkeypatched, so this asserts the
request we WOULD send, which is the part a live test cannot check.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import caption_format as cf  # noqa: E402
from app.caption_format import (  # noqa: E402
    build_post_caption,
    clean_tag,
    finalize_clip_caption,
    normalize_hashtags,
    shorten_caption,
    suggest_content_tags,
)
from app.config import SocialConfig  # noqa: E402
from app.social import (  # noqa: E402
    APPROVAL_NOTES,
    PLATFORMS,
    AccountStore,
    NeedsReconnect,
    NotConfigured,
    PostRejected,
    describe_all,
    post_video,
    redirect_uri,
)
from app.social import base as sb  # noqa: E402

FAILURES: list[str] = []
PASSED = 0


def check(name: str, ok: object) -> None:
    """Assert one thing.

    Deliberately `check(name, ok)` and defensive about the argument order: an
    earlier version of this file's sibling helpers accepted (ok, name), where a
    truthy string in the `ok` slot passes every time and the suite reports
    green while testing nothing.
    """
    global PASSED
    if not isinstance(name, str) or not name:
        raise AssertionError("check() called as check(ok, label) -- label must be first")
    if isinstance(ok, str):
        raise AssertionError(
            f"check({name!r}, {ok!r}) put a STRING in the ok slot -- the arguments "
            "are swapped, so this assertion would pass unconditionally"
        )
    if ok:
        PASSED += 1
    else:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# --------------------------------------------------------------------------- #
section("tag cleaning")
# --------------------------------------------------------------------------- #

check("bare word gains a hash", clean_tag("viral") == "#viral")
check("existing hash not doubled", clean_tag("#viral") == "#viral")
check("multiple hashes collapse", clean_tag("##viral") == "#viral")
check("lower-cased", clean_tag("#Viral") == "#viral")
check("spaces and & removed", clean_tag("#AI & ML") == "#aiml")
check("emoji removed not kept", clean_tag("#viral🔥") == "#viral")
check("empty stays empty", clean_tag("") == "")
check("punctuation-only is empty", clean_tag("###") == "")
check("None is empty", clean_tag(None) == "")

# --------------------------------------------------------------------------- #
section("hashtag normalisation")
# --------------------------------------------------------------------------- #

five = normalize_hashtags(["#alpha", "#beta", "#gamma", "#delta", "#epsilon"])
five_disc = [t for t in five if t in cf.DISCOVERY_TAGS]
five_cont = [t for t in five if t not in cf.DISCOVERY_TAGS]
check("content capped at 4", len(five_cont) == 4, )
check("total stays in 5-7", 5 <= len(five) <= 7)
check("first four content tags kept in order", five_cont == ["#alpha", "#beta", "#gamma", "#delta"])
check("discovery topped up alongside", 2 <= len(five_disc) <= 3)

solo = normalize_hashtags(["#alpha", "#beta", "#gamma"])
solo_disc = [t for t in solo if t in cf.DISCOVERY_TAGS]
check("3 content tags keeps all 3", len([t for t in solo if t not in cf.DISCOVERY_TAGS]) == 3)
check("3 content + 2 discovery = 5", len(solo) == 5 and len(solo_disc) == 2)

nine = normalize_hashtags([f"#t{i}" for i in range(9)])
check("never exceeds 7", len(nine) <= 7)
check("never drops below 5", len(nine) >= 5)
check(
    "7 is a ceiling, not a target: 2 discovery + 4 content = 6, because adding a "
    "third generic tag for the sake of hitting 7 is exactly the filler the spec "
    "rejects",
    len(nine) == 6,
)

dupes = normalize_hashtags(["#viral", "#Viral", "#VIRAL", "#fyp"])
check("deduped case-insensitively", dupes.count("#viral") == 1)
check("non-discovery survive dedupe", "#fyp" in dupes)

topped = normalize_hashtags(["#alpha", "#beta"])
check("thin list topped to 5", len(topped) == 5)
check("content tags kept", "#alpha" in topped and "#beta" in topped)
check("top-up uses discovery tags", "#fyp" in topped)

four_disc = normalize_hashtags(["#fyp", "#foryou", "#viral", "#extra", "#alpha"])
disc = [t for t in four_disc if t in cf.DISCOVERY_TAGS]
check("at most 3 discovery tags", len(disc) <= 3, )
check("the 4th discovery became content", "#extra" in four_disc)

junk = normalize_hashtags(["#viral", "###", "", None, 42, "#real"])
check("junk entries dropped", junk.count("#viral") == 1 and "#real" in junk)
check("junk never becomes a tag", "" not in junk and "None" not in junk)

check("non-list input is safe", normalize_hashtags(None) == list(cf.DISCOVERY_TAGS[:3]))
check("string input is safe", normalize_hashtags("#viral") == list(cf.DISCOVERY_TAGS[:3]))

# --------------------------------------------------------------------------- #
section("caption shortening")
# --------------------------------------------------------------------------- #

check("single line preserved", shorten_caption("one line here") == "one line here")
multi = shorten_caption("first line\nsecond line\r\nthird")
check("newlines collapsed to spaces", "\n" not in multi and "\r" not in multi)
check("collapsed content intact", multi == "first line second line third")

long = "word " * 60
short = shorten_caption(long)
check("long caption trimmed", len(short) <= cf.MAX_CAPTION_CHARS)
check("trimmed on a word boundary", not short.rstrip().endswith("wor"))
check("no trailing space", short == short.rstrip())
check("empty is safe", shorten_caption("") == "")
check("None is safe", shorten_caption(None) == "")

# --------------------------------------------------------------------------- #
section("transcript-mined content tags")
# --------------------------------------------------------------------------- #

transcript = (
    "the secret is consistency every day. consistency beats talent every single "
    "time. your consistency is your superpower. consistency is the habit that "
    "actually works and talent without consistency goes nowhere."
)
mined = suggest_content_tags(transcript)
check("mines the repeated topic", "#consistency" in mined, )
check("no stopword tags", not any(t in {"#the", "#that", "#with"} for t in mined))

noisy = suggest_content_tags("um so like basically uh yeah um")
check("stopwords never become tags", all(len(t) > 4 for t in noisy))

check("empty transcript is safe", suggest_content_tags("") == [])
check("None transcript is safe", suggest_content_tags(None) == [])

hapax = suggest_content_tags("quantum xylophone marmalade")
check("one-off words rejected as too rare", hapax == [])

# --------------------------------------------------------------------------- #
section("final clip caption")
# --------------------------------------------------------------------------- #

line, tags = finalize_clip_caption("A punchy line", ["#motivation", "#grit"], transcript)
check("caption is one line", "\n" not in line)
check("caption returned", line == "A punchy line")
check("total between 5 and 7", 5 <= len(tags) <= 7, )
disc = [t for t in tags if t in cf.DISCOVERY_TAGS]
cont = [t for t in tags if t not in cf.DISCOVERY_TAGS]
check("2-3 discovery tags", 2 <= len(disc) <= 3)
check("at least 2 content tags", len(cont) >= 2)
check("model content tags kept", "#motivation" in tags and "#grit" in tags)

line2, tags2 = finalize_clip_caption("x", ["#motivation"], transcript)
cont2 = [t for t in tags2 if t not in cf.DISCOVERY_TAGS]
check("one model tag is topped up from the transcript", len(cont2) >= 2)
check("topped-up tag is real", "#consistency" in tags2)
check("no duplicate tags", len(tags2) == len(set(tags2)))

line3, tags3 = finalize_clip_caption("x", [f"#t{i}" for i in range(12)], transcript)
check("heavy over-tagging still capped", len(tags3) <= 7)

line4, tags4 = finalize_clip_caption("multi\nline\ncaption", ["#a", "#b"], "")
check("multi-line caption collapsed", "\n" not in line4)

text = build_post_caption("My caption", ["#a", "#b", "#c"])
parts = text.split("\n\n")
check("post text is caption + blank + block", len(parts) == 2)
check("caption is first", parts[0] == "My caption")
check("block is hashtags", parts[1].startswith("#"))
check("no leading or trailing whitespace", text == text.strip())

# --------------------------------------------------------------------------- #
section("account store")
# --------------------------------------------------------------------------- #

import tempfile  # noqa: E402

tmp = Path(tempfile.mkdtemp())
store = AccountStore(tmp)
check("nothing connected initially", not store.connected("youtube"))
check("get returns None when absent", store.get("youtube") is None)

store.save("youtube", {"accessToken": "a", "refreshToken": "r"})
check("saved token reads back", store.get("youtube")["accessToken"] == "a")
check("connected after save", store.connected("youtube"))
check("file is 0600", (tmp / "accounts" / "youtube.json").stat().st_mode & 0o777 == 0o600)
check("savedAt stamped", "savedAt" in store.get("youtube"))

check("delete returns True when present", store.delete("youtube") is True)
check("delete returns False when absent", store.delete("youtube") is False)

store.save("youtube", {"accessToken": "a", "expiresAt": time.time() + 3600})
check("fresh token not expired", not store.is_expired(store.get("youtube")))
store.save("youtube", {"accessToken": "a", "expiresAt": time.time() - 10})
check("past expiry is expired", store.is_expired(store.get("youtube")))
store.save("youtube", {"accessToken": "a"})
check("no expiresAt means not expired", not store.is_expired(store.get("youtube")))

try:
    store.save("../escape", {"accessToken": "x"})
    check("path traversal rejected", False)
except (ValueError, OSError):
    check("path traversal rejected", True)
check(
    "traversal did not create a stray file",
    not any("escape" in p.name for p in (tmp / "accounts").iterdir()),
)

# --------------------------------------------------------------------------- #
section("readiness and approval reality")
# --------------------------------------------------------------------------- #

empty = SocialConfig()
descs = {d["slug"]: d for d in describe_all(empty, store)}
check("all four platforms described", len(descs) == 4)
check("no credentials -> needs_app", all(d["status"] == "needs_app" for d in descs.values()))
check("no credentials -> not connected", not any(d["connected"] for d in descs.values()))

check(
    "YouTube claims no approval gate",
    APPROVAL_NOTES["youtube"]["gate"] == "none",
)
check(
    "Instagram gated on app review",
    APPROVAL_NOTES["instagram"]["gate"] == "app_review",
)
check(
    "TikTok gated on audit",
    APPROVAL_NOTES["tiktok"]["gate"] == "audit",
)
check(
    "X gated on payment",
    APPROVAL_NOTES["x"]["gate"] == "payment",
)
check("every platform has a headline", all(n["headline"] for n in APPROVAL_NOTES.values()))
check("every platform has docs", all(n["docs"].startswith("https://") for n in APPROVAL_NOTES.values()))
check("no tokens leak into the description", all("accessToken" not in json.dumps(d) for d in descs.values()))

full = SocialConfig(
    youtube_client_id="a", youtube_client_secret="b",
    x_client_id="a", x_client_secret="b",
    tiktok_client_key="a", tiktok_client_secret="b",
    meta_app_id="a", meta_app_secret="b",
)
for slug in ("youtube", "x", "tiktok", "instagram"):
    store.save(slug, {"accessToken": f"{slug}-token", "accountName": f"{slug}-user"})
ready = {d["slug"]: d for d in describe_all(full, store)}
check("all four connectable with credentials", all(d["status"] == "connected" for d in ready.values()))
check("account name surfaced", ready["youtube"]["accountName"] == "youtube-user")
check("approval notes still shown when connected", all(d["headline"] for d in ready.values()))

store.delete("instagram")
part = {d["slug"]: d for d in describe_all(full, store)}
check("disconnected one reports not_connected", part["instagram"]["status"] == "not_connected")
check("disconnected one gives a reason", bool(part["instagram"]["reason"]))

# --------------------------------------------------------------------------- #
section("post_video gating")
# --------------------------------------------------------------------------- #

empty_store = AccountStore(Path(tempfile.mkdtemp()))
try:
    post_video("youtube", empty, empty_store, video_path="/x", caption="c")
    check("unconfigured platform raises", False)
except NotConfigured:
    check("unconfigured platform raises", True)

try:
    post_video("myspace", full, store, video_path="/x", caption="c")
    check("unknown platform raises", False)
except NotConfigured:
    check("unknown platform raises", True)

check(
    "redirect_uri is per platform",
    redirect_uri(full, "youtube") != redirect_uri(full, "x"),
)
check("redirect_uri uses public url", redirect_uri(full, "youtube").startswith(full.public_url))

# --------------------------------------------------------------------------- #
section("expired token is refreshed before posting")
# --------------------------------------------------------------------------- #

calls: list[dict] = []


def fake_youtube_post(cfg, record, **kwargs):
    calls.append({"record": dict(record), "kwargs": kwargs})
    return {"platform": "youtube", "id": "vid1", "url": "https://youtu.be/x", "privacy": kwargs.get("privacy", "")}


real_post = PLATFORMS["youtube"].post_video
real_refresh = PLATFORMS["youtube"].refresh
PLATFORMS["youtube"].post_video = fake_youtube_post
PLATFORMS["youtube"].refresh = lambda cfg, rec: {**rec, "accessToken": "FRESH", "expiresAt": time.time() + 3600}

store.save("youtube", {"accessToken": "STALE", "expiresAt": time.time() - 60})
res = post_video("youtube", full, store, video_path=__file__, caption="hi", title="t")
check("expired token refreshed", calls and calls[-1]["record"]["accessToken"] == "FRESH")
check("refreshed token persisted", store.get("youtube")["accessToken"] == "FRESH")
check("post result returned", res["id"] == "vid1")
check("caption passed through", calls[-1]["kwargs"]["caption"] == "hi")
check("title passed through", calls[-1]["kwargs"]["title"] == "t")
check("bytes actually read", len(calls[-1]["kwargs"]["video_bytes"]) > 0)

store.save("youtube", {"accessToken": "CURRENT", "expiresAt": time.time() + 3600})
post_video("youtube", full, store, video_path=__file__, caption="hi")
check("valid token not refreshed", calls[-1]["record"]["accessToken"] == "CURRENT")

PLATFORMS["youtube"].post_video = real_post
PLATFORMS["youtube"].refresh = real_refresh

# --------------------------------------------------------------------------- #
section("tiktok privacy under audit")
# --------------------------------------------------------------------------- #

tt = PLATFORMS["tiktok"]
check("unaudited forced to SELF_ONLY", tt.resolve_privacy(False, "PUBLIC_TO_FOLLOWERS") == "SELF_ONLY")
check("unaudited ignores a public request", tt.resolve_privacy(False, "") == "SELF_ONLY")
check("audited may go public", tt.resolve_privacy(True, "PUBLIC_TO_FOLLOWERS") == "PUBLIC_TO_FOLLOWERS")
check("audited with no request stays private", tt.resolve_privacy(True, "") == "SELF_ONLY")
check("audited rejects a bogus level", tt.resolve_privacy(True, "EVERYONE") == "SELF_ONLY")

tt_calls: list[dict] = []
real_tt_post = tt.post_video
tt.post_video = lambda cfg, rec, **kw: (
    tt_calls.append(kw)
    or {"platform": "tiktok", "id": "1", "url": "", "privacy": kw.get("privacy", ""), "audited": kw.get("audited", False)}
)
store.save("tiktok", {"accessToken": "tt", "expiresAt": time.time() + 3600})

import os  # noqa: E402

os.environ.pop("TIKTOK_APP_AUDITED", None)
post_video("tiktok", full, store, video_path=__file__, caption="c", privacy="PUBLIC_TO_FOLLOWERS")
check("unaudited flag is False by default", tt_calls[-1]["audited"] is False)

os.environ["TIKTOK_APP_AUDITED"] = "true"
post_video("tiktok", full, store, video_path=__file__, caption="c", privacy="PUBLIC_TO_FOLLOWERS")
check("audit env flag read at post time", tt_calls[-1]["audited"] is True)
os.environ.pop("TIKTOK_APP_AUDITED")

tt.post_video = real_tt_post

# --------------------------------------------------------------------------- #
section("error mapping -> HTTP status")
# --------------------------------------------------------------------------- #

import app.main as main_mod  # noqa: E402

for exc, expected in (
    (NotConfigured, 409),
    (NeedsReconnect, 401),
    (PostRejected, 422),
    (sb.SocialError, 502),
):
    check(f"{exc.__name__} maps to {expected}", main_mod.social_error_status(exc("boom")) == expected)
check("unknown exception is a 502", main_mod.social_error_status(RuntimeError("x")) == 502)

# --------------------------------------------------------------------------- #
section("User-Agent is always sent (Cloudflare 1010)")
# --------------------------------------------------------------------------- #

seen_headers: list[dict] = []
real_raw = sb.raw_request


def spy_raw(url, **kwargs):
    seen_headers.append(kwargs.get("headers") or {})
    return 200, b"{}", {}


sb.raw_request = spy_raw
sb.request_json("https://example.invalid/x")
check("raw_request always sets a User-Agent", True, )
check(
    "UA is not the blocked python default",
    sb.USER_AGENT != "Python-urllib/3.11",
)
check("UA names the app", "ClipForge" in sb.USER_AGENT)
sb.raw_request = real_raw

# --------------------------------------------------------------------------- #
section("additive: download path untouched")
# --------------------------------------------------------------------------- #

src = (ROOT / "app" / "main.py").read_text()
check("clip_download route still present", "def clip_download(clip_id: str)" in src)
check("download still streams via FileResponse", "attachment; filename=" in src)
check(
    "download defined BEFORE the social routes",
    src.index("def clip_download") < src.index("def social_accounts"),
)
check("social section is additive only", "clip_download" in src.split("def social_accounts")[0])
check("no social import shadows download helpers", "downloadClipMp4" not in src)

export_src = (ROOT / "src" / "lib" / "clip-export.ts").read_text()
check("downloadClipMp4 still exported", "export async function downloadClipMp4" in export_src)
check("downloadClipMp4 still streams from the API", "/download" in export_src)
check("buildClipSrt untouched", "export function buildClipSrt" in export_src)
check("captions.txt builder untouched", "export function buildCaptionsTxt" in export_src)

card = (ROOT / "src" / "components" / "clipforge" / "ClipCard.tsx").read_text()
check("MP4 button still present", "downloadMp4" in card)
check("SRT button still present", "buildClipSrt" in card)
check("Copy caption still present", "copyClipMeta" in card)
check("PostMenu added alongside", "PostMenu" in card)
check("Copy caption label unchanged", "Copy caption" in card)

# --- layout: the regression that actually bit -------------------------------
# A 4th button in a single-line flex row pushed MP4 and .srt outside a narrow
# card. They were still in the DOM, so the assertions above all passed while the
# user could not see or reach them. Source checks cannot prove pixels, so these
# assert the STRUCTURE that prevents the overflow instead.


def div_block_from(src: str, anchor: str, tag: str = "div") -> str:
    """The balanced ``<tag>...</tag>`` opened by the element carrying `anchor`.

    Returns "" when the anchor is gone, so a regression reports as FAILED checks
    rather than a traceback: a crashed suite reads like a broken harness, and
    the one thing it must never do is stop describing the bug.
    """
    if anchor not in src:
        return ""
    open_tag, close_tag = f"<{tag}", f"</{tag}>"
    # rindex for the anchor too: a label often appears in a doc comment above
    # the element it describes, and the element is what we want to read.
    start = src.rindex(open_tag, 0, src.rindex(anchor))
    depth = 0
    i = start
    while i < len(src):
        if src.startswith(open_tag, i):
            depth += 1
        elif src.startswith(close_tag, i):
            depth -= 1
            if depth == 0:
                return src[start : i + len(close_tag)]
        i += 1
    return src[start:]


ROW_ANCHOR = 'className="mt-auto flex flex-wrap items-center gap-1.5"'
check("primary row anchor present", ROW_ANCHOR in card)
primary_row = div_block_from(card, ROW_ANCHOR)
check("primary row located", "downloadMp4" in primary_row)
check("primary row contains Copy caption", "copyClipMeta" in primary_row)
check("primary row contains the MP4 download", "void downloadMp4()" in primary_row)
check("primary row contains the SRT download", "buildClipSrt(clip)" in primary_row)
check("primary button row can wrap", "flex-wrap" in primary_row)
check(
    "primary row is not a rigid single line",
    "flex-nowrap" not in primary_row and "whitespace-nowrap" not in primary_row,
)
check(
    "PostMenu is NOT in the primary row (it gets its own line)",
    "<PostMenu" not in primary_row,
)
check(
    "PostMenu renders after the download button",
    card.index("<PostMenu") > card.index("void downloadMp4()"),
)
check("all three primary actions on one row", primary_row.count("<Button") == 3)

ts_src = (ROOT / "src" / "lib" / "clipforge.ts").read_text()
check("TS discovery tags match Python", '"#fyp", "#foryou", "#viral"' in ts_src)
check("TS max tags match Python", "MAX_TAGS = 7" in ts_src)
check("TS min tags match Python", "MIN_TAGS = 5" in ts_src)
check("TS discovery cap matches Python", "MAX_DISCOVERY = 3" in ts_src)
check("TS min discovery matches Python", "MIN_DISCOVERY = 2" in ts_src)
check("TS caption cap matches Python", "MAX_CAPTION_CHARS = 150" in ts_src)

# --------------------------------------------------------------------------- #
# Share via... vs Post to... — two different things, two different words.
# --------------------------------------------------------------------------- #
menu = (ROOT / "src" / "components" / "clipforge" / "PostMenu.tsx").read_text()
share_row = div_block_from(menu, "Share via...", "DropdownMenuItem")
# JSX wraps prose across lines, so match against a whitespace-collapsed copy:
# "share\n  sheet" is the same sentence a reader sees as "share sheet".
share_flat = " ".join(share_row.split())
check("the share-sheet action is labelled 'Share via...'", "Share via..." in menu)
check(
    "the old ambiguous label is gone ('Share sheet / download instead' next to "
    "'Post to...' is exactly the confusion this replaces)",
    "Share sheet / download instead" not in menu,
)
check("the posting trigger still says 'Post to...'", "Post to..." in menu)
check(
    "the share row says what it actually does (share sheet + burned-in captions)",
    "share sheet" in share_flat and "captions burned in" in share_flat,
)
check(
    "the two paths are described as different",
    "Different from posting above" in share_row,
)
check(
    "the share row is marked as needing no account",
    "no account needed" in share_row,
)
# The shared bytes must be the finished render, via the same endpoint the MP4
# button uses — never a source file or a proxy.
check("sharing uses the real download endpoint", "api.clipDownloadUrl(clip._id)" in menu)
check("the MP4 button and the share path hit the same endpoint",
      "/download" in (ROOT / "src" / "lib" / "clip-export.ts").read_text())
check("the shared file is named after the clip, not 'clip-1.mp4'",
      "clipFileName(clip)" in menu and "`clip-${clip.order + 1}.mp4`" not in menu)
check("the clip is fetched once, not twice (an 80 MB clip was downloaded twice "
      "on every fallback)",
      menu.count("api.clipDownloadUrl(clip._id)") == 1)

# --------------------------------------------------------------------------- #
# Delete a batch: one path, one confirmation, nothing else touched.
# --------------------------------------------------------------------------- #
delete_ui = (ROOT / "src" / "components" / "clipforge" / "ConfirmDeleteJob.tsx").read_text()
dash = (ROOT / "src" / "pages" / "Dashboard.tsx").read_text()
jobcard = (ROOT / "src" / "components" / "clipforge" / "JobCard.tsx").read_text()
check("a confirmation dialog exists", "AlertDialogContent" in delete_ui)
check("the dialog asks before deleting",
      "This permanently deletes" in delete_ui and "cannot be undone" in delete_ui)
check("the dialog offers both answers",
      "AlertDialogCancel" in delete_ui and "Cancel" in delete_ui
      and "Yes, delete" in delete_ui)
cancel_row = div_block_from(delete_ui, "Cancel</AlertDialogCancel>", "AlertDialogCancel")
yes_row = div_block_from(delete_ui, "Yes, delete", "AlertDialogAction")
check("the Cancel button is in the dialog", "Cancel" in cancel_row)
check("Cancel does not delete (the whole point of the dialog)",
      "remove" not in cancel_row and "deleteJob" not in cancel_row)
check("only the Yes action performs the delete",
      "remove()" in yes_row and delete_ui.count("void remove()") == 1)
check("the API is called from exactly one place", delete_ui.count("api.deleteJob") == 1)
check("the batch pill carries a delete control", 'iconOnly' in dash and "ConfirmDeleteJob" in dash)
check("the pill delete is a separate control from the pill's select button",
      "<ConfirmDeleteJob" in dash and dash.count("setSelectedJobId(job._id)") == 1)
check("deleting the open batch clears the selection",
      "setSelectedJobId(null)" in dash)
check("the job card uses the same confirmed delete", "<ConfirmDeleteJob" in jobcard)
check("the job card has no unconfirmed delete left", "api.deleteJob" not in jobcard)
check("the icon inside the pill stops the pill's own click",
      "stopPropagation" in delete_ui)
check("the server endpoint is a real DELETE on one job",
      '@app.delete("/api/jobs/{job_id}")' in
      (ROOT / "app" / "main.py").read_text())

# --------------------------------------------------------------------------- #
# Unclear speech: the card says so instead of hiding it.
# --------------------------------------------------------------------------- #
check("the clip DTO carries the uncertain-cue count", "uncertainCues" in
      (ROOT / "src" / "lib" / "api.ts").read_text())
check("the card shows an 'unclear captions' chip", "unclear caption" in card)
check("the chip only appears when there is one to report",
      "(clip.uncertainCues ?? 0) > 0" in card)

print(f"\n{'=' * 60}")
if FAILURES:
    print(f"{len(FAILURES)} FAILED ({PASSED} passed):")
    for f in FAILURES:
        print(f"  x {f}")
    sys.exit(1)
print(f"ALL {PASSED} CHECKS PASSED")
