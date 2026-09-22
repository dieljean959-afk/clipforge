"""yt-dlp fetcher service (Python API, not subprocess).

Auto-updates yt-dlp once on failure (YouTube rots), honors
cookies-from-browser, and reports progress via callback.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import Settings


class DownloadError(Exception):
    """Fetch failed — carries the exact yt-dlp error text for the UI."""


def _pip_self_update() -> str:
    """`pip install -U yt-dlp` in-process. Returns the update log tail."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "yt-dlp", "--quiet"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        tail = (proc.stdout + proc.stderr).strip()[-400:]
        logger.info(f"[download] yt-dlp self-update exit={proc.returncode}: {tail[:200]}")
        return tail
    except Exception as exc:
        logger.error(f"[download] yt-dlp self-update failed: {exc}")
        return str(exc)


def probe_url(url: str) -> dict[str, Any] | None:
    """Lightweight yt-dlp metadata probe; None when the URL is unusable."""
    try:
        import yt_dlp
    except Exception:
        return None
    opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": (info or {}).get("title"),
                "duration": (info or {}).get("duration"),
                "uploader": (info or {}).get("uploader"),
            }
    except Exception as exc:
        logger.debug(f"[download] probe failed for {url}: {exc}")
        return None


def download_url(
    url: str,
    dest_dir: Path,
    settings: Settings,
    on_progress: Callable[[float, float | None], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    already_retried: bool = False,
) -> Path:
    """Download a video into dest_dir via the yt-dlp Python API.

    Raises DownloadError with the exact error after one self-update retry.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts: dict[str, Any] = {
        "format": f"bv*[height<={settings.download.max_height}]+ba/b[height<={settings.download.max_height}]",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "progress_hooks": [],
    }
    if settings.download.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (settings.download.cookies_from_browser,)

    def _hook(d: dict[str, Any]) -> None:
        if d.get("status") == "downloading" and on_progress is not None:
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes", 0)
            if total:
                on_progress(done / total, None)
        elif d.get("status") == "finished" and on_progress is not None:
            on_progress(1.0, None)

    ydl_opts["progress_hooks"] = [_hook]

    last_error = ""
    try:
        import yt_dlp
    except Exception as exc:
        raise DownloadError(
            f"yt-dlp is not importable ({exc}); run `pip install yt-dlp`"
        ) from exc

    try:
        logger.info(f"[download] yt-dlp fetching: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = (info or {}).get("title") or "video"
        # locate the merged output file
        for cand in sorted(dest_dir.glob("source.*")):
            if cand.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
                if on_log:
                    on_log(f"yt-dlp finished: {title}")
                logger.info(f"[download] saved {cand.name} ({cand.stat().st_size} bytes)")
                return cand
        raise DownloadError("yt-dlp reported success but no media file was found")
    except Exception as exc:
        last_error = str(exc)
        logger.warning(f"[download] attempt failed: {last_error[:300]}")

    if already_retried or not settings.download.auto_update:
        raise DownloadError(f"yt-dlp failed: {last_error}")

    # rot policy: self-update then retry once
    if on_log:
        on_log("yt-dlp failed — auto-updating and retrying once...")
    _pip_self_update()
    try:
        import importlib

        importlib.reload(__import__("yt_dlp", fromlist=["YoutubeDL"]))
    except Exception as exc:
        logger.warning(f"[download] reload failed ({exc}); continuing with fresh process import")

    return download_url(
        url,
        dest_dir,
        settings,
        on_progress=on_progress,
        on_log=on_log,
        already_retried=True,
    )


def save_upload(src: Path, dest_dir: Path) -> Path:
    """Move an uploaded file into the job's downloads dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"source{src.suffix.lower() or '.mp4'}"
    shutil.move(str(src), str(dest))
    logger.info(f"[download] upload ingested: {dest.name} ({dest.stat().st_size} bytes)")
    return dest
