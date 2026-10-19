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
    """`pip install -U yt-dlp` in-process. Returns the update log tail.

    Uses uv (much faster) when available, falling back to pip. On uv-created
    venvs without pip, `ensurepip` is bootstrapped first so the rot guard can
    never silently no-op.
    """
    import shutil as _shutil

    if _shutil.which("uv") and _pip_missing():
        try:
            proc = subprocess.run(
                ["uv", "pip", "install", "-U", "yt-dlp", "--python", sys.executable, "--quiet"],
                capture_output=True,
                text=True,
                timeout=180,
            )
            tail = (proc.stdout + proc.stderr).strip()[-400:]
            logger.info(f"[download] yt-dlp self-update via uv exit={proc.returncode}: {tail[:200]}")
            return tail
        except Exception as exc:
            logger.warning(f"[download] uv self-update failed ({exc}); falling back to pip")
    if _pip_missing():
        try:
            boot = subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                capture_output=True, text=True, timeout=120,
            )
            if boot.returncode != 0:
                return f"ensurepip failed: {(boot.stderr or boot.stdout).strip()[-200:]}"
        except Exception as exc:
            return f"ensurepip failed: {exc}"
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


def _pip_missing() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return False
    except Exception:
        return True


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
) -> Path:
    """Download a video into dest_dir via the yt-dlp Python API.

    On failure the rot policy kicks in: `pip install -U yt-dlp`, reload, and
    retry ONCE. Raises DownloadError with the exact error when both attempts
    fail (or when the first failure happens with auto-update disabled).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    fetch_log: list[str] = []
    def _log(line: str) -> None:
        fetch_log.append(line)
        if on_log:
            on_log(line)
    max_h = int(settings.download.max_height)
    # Codec preference, in order. H.264 first: it is what the renderer
    # re-encodes to anyway, every decoder handles it, and AV1/VP9 sources cost
    # minutes of CPU decode per clip on a 1-vCPU host. The two fallbacks exist
    # because not every site (Twitch clips, older uploads, members-only HLS)
    # offers an H.264 stream at all — refusing those would mean "this URL does
    # not work" for the whole class. Final fallback is a single progressive
    # file (`b`), which is always muxed and always playable.
    format_prefs = [
        f"bv*[vcodec^=avc1][height<={max_h}]+ba/bv*[vcodec^=h264][height<={max_h}]+ba",
        f"bv*[height<={max_h}]+ba/b[height<={max_h}]",
        f"b[height<={max_h}]/b",
    ]
    ydl_opts: dict[str, Any] = {
        "format": format_prefs[0],
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        # Download DASH fragments concurrently. A single stream is a serial
        # request/response loop that leaves the link idle between segments;
        # eight in flight saturates a normal connection several times over.
        # This is pure network concurrency -- it costs no extra CPU, which is
        # what makes it one of the few speedups that still pays on a host with
        # a single core. Capped at 8 because beyond that the gain flattens and
        # the connection starts spending its time on contention instead.
        "concurrent_fragment_downloads": 8,
        "progress_hooks": [],
    }
    if settings.download.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (settings.download.cookies_from_browser,)
    cookies_file = getattr(settings.download, "cookies_file", "") or ""
    if cookies_file and Path(cookies_file).expanduser().is_file():
        # Netscape cookies.txt. This is the one lever that gets a datacenter
        # IP past YouTube's "Sign in to confirm you're not a bot" page, which
        # otherwise makes every YouTube URL fail from this host.
        ydl_opts["cookiefile"] = str(Path(cookies_file).expanduser())

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

    def _verify_source(path: Path) -> str | None:
        """Why this download is unusable as a render source, or None.

        A directory can hold a file yt-dlp considers finished and the renderer
        cannot use: a video-only stream that merged badly, an audio-only
        variant, or a zero-duration container. Verifying here — once, before a
        twelve-minute transcribe — turns "the clips are black" into an honest
        download error with a fallback already attempted.
        """
        try:
            from app import video as _video

            meta = _video.probe(path)
        except Exception as exc:  # noqa: BLE001 - unreadable == unusable
            return f"unreadable media ({exc})"
        if int(meta.get("width") or 0) <= 0 or int(meta.get("height") or 0) <= 0:
            return "no video stream"
        if float(meta.get("duration") or 0.0) <= 0:
            return "zero duration"
        return None

    def _run_fetch(attempt_label: str) -> Path:
        last_error_local = ""
        last_unplayable = ""
        last_exc: Exception | None = None
        for idx, fmt in enumerate(format_prefs):
            ydl_opts["format"] = fmt
            label = attempt_label if idx == 0 else f"{attempt_label}, format {idx + 1} fallback"
            if on_log:
                on_log(f"yt-dlp fetcher: {label}, {url}")
            logger.info(f"[download] yt-dlp fetching ({label}): {url}")
            ydl_opts["verbose"] = False
            for leftover in dest_dir.glob("source.*"):
                if leftover.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".part"}:
                    leftover.unlink(missing_ok=True)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    title = (info or {}).get("title") or "video"
            except Exception as exc:  # noqa: BLE001 - try the next format
                last_error_local = str(exc)
                last_exc = exc
                logger.warning(f"[download] format {idx + 1} failed: {last_error_local[:300]}")
                continue
            cand = None
            for c in sorted(dest_dir.glob("source.*")):
                if c.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
                    cand = c
                    break
            if cand is None:
                last_error_local = "yt-dlp reported success but no media file was found"
                continue
            unplayable = _verify_source(cand)
            if unplayable is not None:
                last_unplayable = f"{cand.name} is not a usable source ({unplayable})"
                logger.warning(f"[download] {last_unplayable} — trying next format")
                cand.unlink(missing_ok=True)
                continue
            if on_log:
                on_log(f"yt-dlp finished: {title} ({cand.name})")
            logger.info(f"[download] saved {cand.name} ({cand.stat().st_size} bytes)")
            return cand
        # A source that downloaded but cannot be rendered is not a yt-dlp
        # version problem, so surface it directly. An extraction failure IS
        # often the rotating-site problem the auto-update retry exists for, so
        # the original exception is re-raised and the caller's rot policy
        # re-runs every format after updating.
        if last_unplayable:
            raise DownloadError(
                f"{last_unplayable} after trying {len(format_prefs)} formats"
            )
        if last_exc is not None:
            raise last_exc
        raise DownloadError("yt-dlp failed to produce a usable video file")

    try:
        return _run_fetch("attempt 1")
    except DownloadError:
        raise
    except Exception as exc:
        last_error = str(exc)
        logger.warning(f"[download] fetch failed: {last_error[:400]}")
        if on_log:
            on_log(f"yt-dlp failed: {last_error}")
        fetch_log_text = "\n".join(fetch_log[-40:])
        # `yt_dlp.version` is a MODULE, so `getattr(yt_dlp, "version")` is not
        # a string — concatenating it raised TypeError inside the failure
        # handler, which replaced the real yt-dlp error with an unrelated
        # "can only concatenate str (not module) to str".
        version_text = getattr(getattr(yt_dlp, "version", None), "__version__", None)
        if not isinstance(version_text, str):
            version_text = getattr(yt_dlp, "__version__", "unknown")
        ydl_debug = (
            "yt-dlp "
            + str(version_text)
            + "\n"
            + "DOWNLOAD ATTEMPT LOG:\n"
            + fetch_log_text
            + "\n"
            + last_error
        )
        if on_log:
            on_log("yt-dlp FAILED — captured output:\n" + ydl_debug)

        # YouTube's bot check is per client. The default web client is the one
        # that answers "Sign in to confirm you're not a bot" from a server IP;
        # the iOS/Safari/mobile clients frequently do not. One retry with those
        # clients is cheap and turns a dead URL into a working job on hosts
        # where no cookies are configured.
        bot_block = any(
            marker in last_error.lower()
            for marker in ("not a bot", "sign in to confirm", "login required", "confirm your age")
        )
        if bot_block and "youtube" in url.lower() and not cookies_file:
            if on_log:
                on_log("YouTube asked for a sign-in — retrying with alternate clients")
            ydl_opts["extractor_args"] = {
                "youtube": {"player_client": ["ios", "mweb", "web_safari"]}
            }
            try:
                return _run_fetch("youtube alternate client")
            except Exception as exc:
                last_error = str(exc)
                logger.warning(f"[download] alternate client failed: {last_error[:300]}")
            finally:
                ydl_opts.pop("extractor_args", None)
        if bot_block:
            raise DownloadError(
                "This site asked this server to sign in before it would serve the "
                "video (YouTube: \"Sign in to confirm you're not a bot\"). Export "
                "cookies.txt from a logged-in browser and set YT_DLP_COOKIES_FILE, "
                "or attach the video file directly — it is not a ClipForge bug."
            )

    if not settings.download.auto_update:
        raise DownloadError(f"yt-dlp failed: {last_error}")

    # rot policy: self-update then retry exactly once (in-process, same call
    # frame — no recursion, so the retry can never re-trigger itself)
    if on_log:
        on_log("yt-dlp failed — auto-updating and retrying once...")
    _pip_self_update()
    try:
        import importlib

        importlib.reload(__import__("yt_dlp", fromlist=["YoutubeDL"]))
    except Exception as exc:
        logger.warning(f"[download] reload failed ({exc}); continuing with fresh process import")

    try:
        return _run_fetch("attempt 2 after auto-update")
    except Exception as exc:
        raise DownloadError(
            f"yt-dlp failed after auto-update retry. Tried: yt-dlp (attempt 1: {last_error[:200]}) → yt-dlp-updated (attempt 2: {str(exc)[:200]}). Manual upload always works — 'Attach file' on the dashboard."
            + ("") if False else "",
        ) from exc


#: Extensions we accept for an uploaded source. Anything else is stored as
#: .mp4 so the fetch stage never has to guess from a temp filename.
_UPLOAD_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mp3", ".m4a", ".wav"}


def save_upload(src: Path, dest_dir: Path, original_name: str | None = None) -> Path:
    """Move an uploaded file into the job's downloads dir as `source.<ext>`.

    The extension MUST come from the uploaded file's own name, not from the
    temporary staging file. Callers stage the bytes as `upload.bin`, so the
    old `src.suffix` produced `source.bin` and the fetch stage — which looks
    for `source.<ext of job.source>` — only matched by accident, and a job whose
    source was ever recorded with a real extension died with "uploaded file
    missing at .../source.mp4" while a perfectly good file sat next to it.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_name or src.name).suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        suffix = ".mp4"
    dest = dest_dir / f"source{suffix}"
    shutil.move(str(src), str(dest))
    logger.info(f"[download] upload ingested: {dest.name} ({dest.stat().st_size} bytes)")
    return dest
