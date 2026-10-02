"""Sharing a clip must never quietly download it.

The complaint: tapping "Share..." on a phone produced
`Downloaded clip-01.mp4 - no share sheet on this device.` instead of a way to
post the clip. On Android Chrome `navigator.canShare({files})` is
version-dependent and frequently false, so the old code took its final fallback
(a silent <a download>) and reported success as a download. An 8 MB file the
user never asked for, with no way to post from it.

The ladder is now, in order:
  1. connected account  -> real API upload (unchanged, in PostMenu)
  2. platform intent    -> opens their composer, no download, no account
  3. OS share sheet     -> file if the browser can, else text+link
  4. clipboard          -> caption + link, always works

A download is never a fallback here. These are source-level assertions on the
frontend (there is no browser in this environment), plus a real evaluation of
the intent URLs so a malformed link cannot ship.

    python tools/share_intents_test.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MENU = (ROOT / "src" / "components" / "clipforge" / "PostMenu.tsx").read_text()
# The FILE share moved out of PostMenu and into ShareButton, where the prefetch
# cache makes it instant. PostMenu now only offers links and platform
# composers, so the share-ladder assertions belong to ShareButton.
SHARE_BTN = (ROOT / "src" / "components" / "clipforge" / "ShareButton.tsx").read_text()
INTENTS_PATH = ROOT / "src" / "lib" / "share-intents.ts"
INTENTS_TS = INTENTS_PATH.read_text()

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


def section(title: str) -> None:
    print(f"\n[{title}]")


# --------------------------------------------------------------------------
# 1. the silent download is gone
# --------------------------------------------------------------------------


def test_no_silent_download() -> None:
    section("1. no path silently downloads instead of sharing")
    # Strip comments first: the JSDoc deliberately NAMES the old behaviour in
    # prose ("this used to end in a silent download plus ..."), and matching on
    # the raw file would flag the explanation as if it were still shipped.
    code = re.sub(r"/\*.*?\*/", "", MENU, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    check(
        "the old 'no share sheet on this device' message is gone from the code",
        "no share sheet on this device" not in code,
    )
    # An <a download> / createObjectURL + click is the silent download. It must
    # not appear in the share path at all.
    check(
        "the share path never builds a download anchor before sharing",
        "URL.createObjectURL" not in code.split("const shareTo")[0].split("a.download")[0]
        or code.index("a.download") > code.index("navigator.clipboard.writeText"),
    )
    check(
        "the fallback ends at the clipboard, not a file",
        "navigator.clipboard.writeText" in code,
    )


# --------------------------------------------------------------------------
# 2. share intents: real, evaluated URLs
# --------------------------------------------------------------------------

#: slug -> the platform's own host. Checked against the BUILT url, so a typo in
#: a template literal fails here rather than opening the wrong site.
HOSTS = {
    "x": "twitter.com",
    "facebook": "facebook.com",
    "threads": "threads.net",
    "reddit": "reddit.com",
    "whatsapp": "wa.me",
    "telegram": "t.me",
    "linkedin": "linkedin.com",
    "pinterest": "pinterest.com",
    "email": None,  # mailto:
}

BUILT = ""


def build_intents() -> str:
    """Evaluate the REAL share-intents.ts under bun and dump the results.

    bun runs TypeScript directly, so the module under test is the shipped one
    with no transpile step and no Python re-implementation that could drift
    from the source. Only `window` is stubbed, since clipShareUrl needs an
    origin.
    """
    driver = f"""
import {{ clipShareUrl, intentUrl, visibleIntents, SHARE_INTENTS }} from {json.dumps(str(INTENTS_PATH))};

(globalThis as any).window = {{ location: {{ origin: "https://clipforge.test" }} }};

const bad = {{ slug: "evil", label: "E", tint: "", textOnly: true,
              url: () => "javascript:alert(1)" }};
const dataSource = {{ slug: "s", label: "S", tint: "", textOnly: true,
                     url: () => "data:text/html,<script>alert(1)</script>" }};

console.log(JSON.stringify({{
  clipUrl: clipShareUrl("abc123"),
  clipUrlEncoded: clipShareUrl("a/b?c"),
  withLink: visibleIntents(true).map((i) => i.slug),
  withoutLink: visibleIntents(false).map((i) => i.slug),
  urls: SHARE_INTENTS.map((i) => [i.slug, intentUrl(i, "hello world #fyp", "https://x.test/v")]),
  urlNoLink: SHARE_INTENTS.map((i) => [i.slug, intentUrl(i, "hello #fyp", "")]),
  bad: intentUrl(bad as any, "t", "l"),
  dataUrl: intentUrl(dataSource as any, "t", "l"),
  thrower: intentUrl({{ ...bad, url: () => {{ throw new Error("boom"); }} }} as any, "t", "l"),
}}));
"""
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "drive.ts"
        f.write_text(driver, encoding="utf-8")
        out = subprocess.run(
            ["bun", "run", str(f)],
            capture_output=True, text=True, timeout=60, cwd=str(ROOT),
        )
    if out.returncode != 0:
        print(out.stdout[-1500:])
        print(out.stderr[-1500:])
    check("share-intents.ts evaluates under bun", out.returncode == 0, out.stderr[-400:])
    return out.stdout.strip() if out.returncode == 0 else ""


def test_intent_urls() -> None:
    section("2. every share intent builds a real URL (evaluated, not eyeballed)")
    global BUILT
    BUILT = build_intents()
    if not BUILT:
        return
    try:
        data = json.loads(BUILT)
    except json.JSONDecodeError:
        check("the module returns JSON", False, BUILT[:300])
        return

    urls = dict(data["urls"])
    url_no_link = dict(data["urlNoLink"])

    for slug, host in HOSTS.items():
        got = urls.get(slug)
        if host is None:
            check(f"{slug}: builds a mailto: link", bool(got) and got.startswith("mailto:"),
                  repr(got))
            continue
        check(f"{slug}: builds a URL", bool(got), repr(got))
        if not got:
            continue
        parsed = urlparse(got)
        # Compare on the registrable host: "www.facebook.com" and
        # "facebook.com" are the same destination, and pinning the exact string
        # would fail on a harmless www change.
        bare = parsed.netloc.lower().removeprefix("www.")
        check(f"{slug}: points at {host}", bare == host, parsed.netloc)
        check(f"{slug}: is https", parsed.scheme == "https", parsed.scheme)

    # The caption must survive into the target, encoded — not dropped, not raw.
    check(
        "the caption reaches X's composer, URL-encoded",
        "hello" in parse_qs(urlparse(urls["x"]).query).get("text", [""])[0],
        urls["x"],
    )
    check(
        "the clip link reaches Facebook's sharer",
        parse_qs(urlparse(urls["facebook"]).query).get("u", [""])[0] == "https://x.test/v",
        urls["facebook"],
    )
    check(
        "the clip link reaches Pinterest's pin creator",
        parse_qs(urlparse(urls["pinterest"]).query).get("url", [""])[0] == "https://x.test/v",
        urls["pinterest"],
    )
    check(
        "WhatsApp carries the caption as its share text",
        "hello" in parse_qs(urlparse(urls["whatsapp"]).query).get("text", [""])[0],
        urls["whatsapp"],
    )
    # Pinterest pins a URL. With no link there is nothing to pin, so it must be
    # hidden rather than opening an empty create form.
    check("Pinterest refuses to build a URL with no link to pin",
          url_no_link.get("pinterest") is None, repr(url_no_link.get("pinterest")))
    check("Pinterest is hidden from the row list when there is no link",
          "pinterest" not in data["withoutLink"], str(data["withoutLink"]))
    check("Pinterest is present when there is a link",
          "pinterest" in data["withLink"], str(data["withLink"]))

    # A javascript: URL from any builder must be refused outright.
    check("a javascript: intent is refused", data["bad"] is None, repr(data["bad"]))
    check("a data: intent is refused", data["dataUrl"] is None, repr(data["dataUrl"]))
    check("a builder that throws is contained, not propagated",
          data["thrower"] is None, repr(data["thrower"]))

    # The share link must be absolute and safe to paste anywhere.
    check("clipShareUrl returns an absolute origin-based link",
          data["clipUrl"] == "https://clipforge.test/api/clips/abc123/video",
          repr(data["clipUrl"]))
    check("clipShareUrl encodes a hostile clip id",
          "/api/clips/a%2Fb%3Fc/video" in data["clipUrlEncoded"],
          repr(data["clipUrlEncoded"]))


# --------------------------------------------------------------------------
# 3. the ladder never ends in a download, and the UI offers the real options
# --------------------------------------------------------------------------


def test_ladder_order() -> None:
    section("3. PostMenu never waits on the network to share a link")
    body = MENU[MENU.find("const shareVia"):MENU.find("const shareTo")]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"^\s*//.*$", "", body, flags=re.M)
    check("it tries text+link first", "await nav.share({ text: caption, url: link" in body)
    check("it falls back to the clipboard",
          "navigator.clipboard.writeText" in body)
    check("it no longer fetches the 8 MB master at all",
          "clipDownloadUrl" not in body and "await fetch(" not in body,
          "the file share now lives in ShareButton, on the cached small copy")
    check("it never downloads", "a.download" not in body and "createObjectURL" not in body)
    check("the dead end names the real alternative",
          "This browser can't share links." in body)


def test_share_button_ladder() -> None:
    section("3b. ShareButton shares the CACHED file, never fetches on tap")
    body = SHARE_BTN[SHARE_BTN.find("const onShare"):SHARE_BTN.find("const pct =")]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"^\s*//.*$", "", body, flags=re.M)

    # The whole point: nothing on the tap path may suspend, or transient user
    # activation expires and share() throws NotAllowedError.
    for bad, label in (
        ("await navigator.share", "an await before navigator.share"),
        ("await fetch", "a fetch inside the handler"),
        ("await cache.", "an await on the cache"),
    ):
        check(f"no {label}", bad not in body, repr(bad))

    check("the cached File is read synchronously", "cache.get(" in body)
    check("canShare is checked before sharing a file",
          "cache.canShareFile()" in body)
    check("the uncached path shares text+url instantly",
          "url: publicUrl" in body)
    check("no download on any path", "a.download" not in body)

    # Prefetch must be wired to the CARD (an effect), not to the tap. The
    # handler body deliberately contains neither call.
    check("prefetch is NOT triggered by the click handler",
          "cache.prefetch(" not in body and "cache.hydrate(" not in body,
          "prefetching on tap is the exact bug being fixed")
    check("prefetch starts from a card-mount effect",
          "cache.prefetch(" in SHARE_BTN
          and "cache.prefetch(clip._id" in SHARE_BTN)
    check("a restored blob is hydrated from IndexedDB",
          "cache.hydrate(" in SHARE_BTN)


def test_ui_offers() -> None:
    section("4. the sheet offers the options instead of a dead end")
    check("platform intent rows are rendered",
          "visibleIntents(" in MENU and "shareTo(intent.slug" in MENU)
    check("there is a copy caption + link action",
          "Copy caption + link" in MENU)
    check("the copy action writes both the caption and the link",
          "clipShareUrl(clip._id)" in MENU)
    check("intents open in a new tab without leaking the opener",
          "noopener" in (ROOT / "src" / "lib" / "share-intents.ts").read_text())
    check("the share link is absolute, not a relative path",
          "window.location.origin" in INTENTS_TS)
    check("a relative share link is never produced",
          "return \"\";" in INTENTS_TS)
    # The connected-account path must survive: that is still the only option
    # that uploads the file for you.
    check("the connected-account API post path is untouched",
          "api.postClip(" in MENU and "p.connected" in MENU)
    check("connecting an account is still offered",
          "api.socialConnect(" in MENU)


def main() -> int:
    test_no_silent_download()
    test_intent_urls()
    test_ladder_order()
    test_share_button_ladder()
    test_ui_offers()
    print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
