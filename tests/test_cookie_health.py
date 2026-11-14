"""The cookies.txt plaster: a stale export must not take the whole job down.

Seen on this host: YouTube answered "Sign in to confirm you're not a bot" for
every URL, cookies.txt had been on disk for weeks, and the download ladder
skipped its alternate-client retry precisely *because* a jar was configured —
so the one case the retry exists for (a jar that has gone stale) was the one
case it never ran. The failure message then told the user to export cookies
they had already exported.
"""

from __future__ import annotations

import os
import time
import types
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import load_config
from app.download import COOKIES_STALE_DAYS, DownloadError, cookie_health

NETSCAPE = "# Netscape HTTP Cookie File\n# https://curl.se/docs/http-cookies.html\n"


def _jar(path: Path, *, expiry: int, rows: int = 3, age_days: float = 0) -> Path:
    body = NETSCAPE
    for i in range(rows):
        body += (
            f".youtube.com\tTRUE\t/\tTRUE\t{expiry}\tSID{i}\tsecret{i}\n"
        )
    path.write_text(body)
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))
    return path


def test_missing_and_unconfigured_jars_say_why(tmp_path):
    assert cookie_health(None)["problem"] == "no cookies file configured"
    missing = cookie_health(tmp_path / "cookies.txt")
    assert missing["present"] is False
    assert "missing" in (missing["problem"] or "")


def test_a_json_export_is_not_a_netscape_jar(tmp_path):
    p = tmp_path / "cookies.txt"
    p.write_text('[{"name": "SID", "value": "x"}]')
    info = cookie_health(p)
    assert info["present"] is True
    assert "not a Netscape" in (info["problem"] or "")


def test_session_only_jar_is_called_out(tmp_path):
    info = cookie_health(_jar(tmp_path / "cookies.txt", expiry=0))
    assert info["sessionOnly"] is True
    assert "session cookie" in (info["problem"] or "")


def test_old_jar_is_stale_before_youtube_rejects_it(tmp_path):
    fresh = cookie_health(_jar(tmp_path / "fresh.txt", expiry=2_000_000_000))
    assert fresh["problem"] is None
    assert fresh["stale"] is False
    assert fresh["cookieCount"] == 3

    old = cookie_health(
        _jar(tmp_path / "old.txt", expiry=2_000_000_000, age_days=COOKIES_STALE_DAYS + 5)
    )
    assert old["stale"] is True
    assert "days old" in (old["problem"] or "")


def test_youtu_be_short_links_count_as_youtube():
    from app.download import is_youtube_url

    for url in (
        "https://youtu.be/ShxSGJ95g0c",
        "https://www.youtube.com/watch?v=ShxSGJ95g0c",
        "https://m.youtube.com/watch?v=x",
        "https://music.youtube.com/watch?v=x",
        "https://www.youtube-nocookie.com/embed/x",
    ):
        assert is_youtube_url(url), url
    for url in ("https://vimeo.com/1", "https://notyoutube.com/watch?v=x", "https://example.com"):
        assert not is_youtube_url(url), url


class _FakeYDL:
    """yt-dlp that refuses cookies, the way an expired jar does."""

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):  # noqa: ARG002
        if self.opts.get("cookiefile"):
            raise RuntimeError(
                "Sign in to confirm you're not a bot. Use --cookies for the authentication."
            )
        out = Path(str(self.opts["outtmpl"]).replace("%(ext)s", "mp4"))
        out.write_bytes(b"\x00" * 64)
        return {"title": "fake title"}


def _fake_ytdlp(monkeypatch):
    mod = types.ModuleType("yt_dlp")
    mod.YoutubeDL = _FakeYDL
    mod.version = types.SimpleNamespace(__version__="fake")
    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", mod)


def test_a_stale_jar_falls_back_to_a_cookieless_retry(tmp_path, monkeypatch):
    cookies = _jar(tmp_path / "cookies.txt", expiry=2_000_000_000, age_days=45)
    base = load_config(quiet=True)
    settings = replace(base, download=replace(base.download, cookies_file=str(cookies)))
    _fake_ytdlp(monkeypatch)
    # The probe is ffprobe in a child process; the ladder is what is under test.
    monkeypatch.setattr("app.video.probe", lambda p: {"width": 1280, "height": 720, "duration": 12.0})

    from app.download import download_url

    logs: list[str] = []
    got = download_url(
        "https://youtu.be/example", tmp_path / "dl", settings, on_log=logs.append
    )
    assert got.is_file()
    assert any("without cookies.txt" in line for line in logs)
    assert any("cookies.txt: cookies are 45 days old" in line for line in logs)


def test_an_expired_jar_that_still_fails_names_the_jar_not_the_user(tmp_path, monkeypatch):
    cookies = _jar(tmp_path / "cookies.txt", expiry=2_000_000_000, age_days=40)
    base = load_config(quiet=True)
    settings = replace(base, download=replace(base.download, cookies_file=str(cookies)))
    ytdlp = types.ModuleType("yt_dlp")

    class _AlwaysBlocked(_FakeYDL):
        def extract_info(self, url, download=True):
            raise RuntimeError("Sign in to confirm you're not a bot")

    ytdlp.YoutubeDL = _AlwaysBlocked
    ytdlp.version = types.SimpleNamespace(__version__="fake")
    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", ytdlp)

    from app.download import download_url

    with pytest.raises(DownloadError) as err:
        download_url("https://youtu.be/example", tmp_path / "dl", settings)
    message = str(err.value)
    assert "cookies.txt on this server was refused" in message
    assert "40 days old" in message
