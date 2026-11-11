"""Capture a URL source straight into the bucket — never a full local copy.

The old fetch stage downloaded the whole video to `downloads/<job>/source.mp4`
(168 MB for a 20-minute source) and the renderer cut clips from that local
file. On a 4 GB disk shared with a 1.6 GB venv that is the entire budget: the
job reached "Transcribing 0%" with 234 MB free and could not finish.

This module removes the file. yt-dlp's stdout (`-o -`) is piped into an S3
multipart upload, so the bytes go network → bucket with only an 8 MiB part
buffer in RAM. Video-only and audio-only are captured as two objects:

    uploads/<job>/video.mp4   (720p cap, video-only)
    uploads/<job>/audio.m4a   (audio-only, ~7-19 MB for 20 min)

Each object is HEAD-verified against the byte count before it is trusted.

When a format cannot be streamed (HLS/fragmented with no seekable container, a
site that only offers muxed `b`), the fallback downloads **5-minute sections**,
uploads and deletes each one immediately, and records the section list so the
renderer can cut a clip out of the one section that contains it. A clip is at
most `max_clip_sec` long, so it always fits inside one section.

`RemoteSource` is what the rest of the pipeline sees instead of a Path: `str()`
gives a fresh presigned URL, and `url_for_span(t)` picks the object (whole
video or the containing section) plus the seek offset inside it.
"""

from __future__ import annotations

import collections
import logging
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

PART_BYTES = 8 * 1024 * 1024
SECTION_SEC = 150.0

#: Sections to try when the source's length is UNKNOWN (yt-dlp metadata
#: failed). 150 s x 48 = 2 h = the largest source this app accepts, so an
#: unreadable metadata field can no longer cut a long video short: the loop
#: still stops at the first section past the end of the video.
UNKNOWN_DURATION_SECTIONS = 48
#: Filebase's free plan refuses MULTIPART video uploads above 25 MB
#: ("Free accounts are limited to 25MB video uploads"), while a single PUT of
#: the same bytes is accepted. Every object that carries video is therefore
#: kept under this and written with ONE put_object.
MAX_VIDEO_OBJECT_BYTES = 24 * 1024 * 1024
#: A source object larger than this is not a "video" any more — it is a bug.
MAX_STREAM_BYTES = 3 * 1024**3

VIDEO_FORMATS = (
    "bv*[height<=720][vcodec^=avc1]",
    "bv*[height<=720]",
    "bv*",
    "b[height<=720]",
    "b",
)
AUDIO_FORMATS = ("ba", "ba*/b")


class StreamError(RuntimeError):
    """Capture failed. `bucket_down` distinguishes "retry later" from "no URL"."""

    def __init__(self, message: str, *, bucket_down: bool = False) -> None:
        self.bucket_down = bucket_down
        super().__init__(message)


def mb(value: int | float | None) -> float | None:
    return None if value is None else round(float(value) / 1048576, 1)


# --------------------------------------------------------------------------- #
# yt-dlp as a pipe
# --------------------------------------------------------------------------- #
def _cli(settings: Any) -> list[str]:
    """The yt-dlp command, with every cache disabled.

    `--no-cache-dir` is the brief's requirement: yt-dlp otherwise keeps the
    player/JS cache under ~/.cache/yt-dlp, which on a 4 GB disk is space the
    render needs. `--no-part` keeps the temporary `.part` file out of the way
    too — stdout has no part file at all.
    """
    exe = shutil.which("yt-dlp") or str(Path(sys.executable).with_name("yt-dlp"))
    cmd = [
        exe,
        "--no-playlist",
        "--no-cache-dir",
        "--no-part",
        "--no-progress",
        "--quiet",
        "--no-warnings",
        "--no-mtime",
    ]
    try:
        runtime = next((n for n in ("deno", "node") if shutil.which(n)), None)
        if runtime:
            cmd += ["--js-runtimes", runtime, "--remote-components", "ejs:github"]
        dl = settings.download
        if getattr(dl, "cookies_from_browser", None):
            cmd += ["--cookies-from-browser", str(dl.cookies_from_browser)]
        cookie_file = getattr(dl, "cookies_file", "") or ""
        if cookie_file and Path(cookie_file).expanduser().exists():
            cmd += ["--cookies", str(Path(cookie_file).expanduser())]
    except Exception:  # noqa: BLE001 - settings shape must never break capture
        pass
    return cmd


def _read_part(stream: Any, size: int) -> bytes:
    """Read exactly `size` bytes (or what is left at EOF)."""
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _upload_pipe(
    stream: Any, key: str, content_type: str, *,
    on_bytes: Callable[[int], None] | None = None,
) -> int:
    """Multipart-upload a readable pipe. Returns the byte count sent."""
    from app import objectstore

    cfg = objectstore.config_from_env()
    s3 = objectstore.client(cfg)
    if s3 is None or not cfg.enabled:
        raise StreamError("bucket is not configured", bucket_down=True)

    upload_id = None
    parts: list[dict[str, Any]] = []
    total = 0
    completed = False
    try:
        created = s3.create_multipart_upload(
            Bucket=cfg.bucket, Key=key, ContentType=content_type,
        )
        upload_id = created["UploadId"]
        number = 0
        while True:
            chunk = _read_part(stream, PART_BYTES)
            if not chunk:
                break
            number += 1
            total += len(chunk)
            if total > MAX_STREAM_BYTES:
                raise StreamError(
                    f"stream exceeded {mb(MAX_STREAM_BYTES)} MB — refusing to "
                    f"keep filling the bucket with {key}"
                )
            resp = s3.upload_part(
                Bucket=cfg.bucket, Key=key, UploadId=upload_id,
                PartNumber=number, Body=chunk,
            )
            parts.append({"ETag": resp["ETag"], "PartNumber": number})
            if on_bytes:
                on_bytes(total)
            if number >= 9000:
                raise StreamError("upload exceeded 9000 parts")
        if not parts:
            raise StreamError("the stream produced no bytes")
        s3.complete_multipart_upload(
            Bucket=cfg.bucket, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        completed = True
        return total
    except StreamError:
        raise
    except Exception as exc:  # noqa: BLE001 - translate into a retryable error
        from app import objectstore as _o

        why = _o.explain_error(exc) if hasattr(_o, "explain_error") else str(exc)
        # A PLAN REFUSAL is not an outage: retrying it 20 times just burns
        # eight minutes before giving the same 403. It means "this object is
        # too big for multipart on this plan", and the caller's answer is the
        # section path, not a wait.
        plan_blocked = "25MB" in why or "AccountProblem" in why
        raise StreamError(
            f"uploading {key} to the bucket failed: {why}",
            bucket_down=not plan_blocked,
        ) from exc
    finally:
        # A failed or interrupted transfer must not leave a half-written object
        # behind: the part list is what makes an object exist, so aborting is
        # the only clean state. A completed upload is never aborted.
        if upload_id and not completed:
            try:
                s3.abort_multipart_upload(
                    Bucket=cfg.bucket, Key=key, UploadId=upload_id,
                )
            except Exception:  # noqa: BLE001 - nothing left to clean up
                pass


def _verify(key: str, sent: int) -> dict[str, Any]:
    """HEAD the object and refuse to believe anything but the exact size."""
    from app import objectstore

    info = objectstore.head(key)
    got = int((info or {}).get("size") or 0)
    if got != sent:
        raise StreamError(
            f"bucket has {got} B for {key} but {sent} B were sent — the object "
            f"is truncated; not using it"
        )
    return {"key": key, "bytes": got}


def _stream_one(
    url: str, fmt: str, key: str, settings: Any, *,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Pipe one format into the bucket. Raises StreamError on any failure."""
    cmd = _cli(settings) + ["-f", fmt, "-o", "-", url]
    logger.info(f"[stream] yt-dlp format {fmt} -> {key}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    tail: collections.deque[str] = collections.deque(maxlen=12)

    def _drain() -> None:
        try:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                tail.append(line)
                if on_log:
                    on_log(line[:200])
        except Exception:  # noqa: BLE001
            pass

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    sent = 0
    if on_progress:
        on_progress(0.0)
    try:
        sent = _upload_pipe(
            proc.stdout, key, "video/mp4" if key.endswith(".mp4") else "audio/mp4",
        )
        code = proc.wait(timeout=120)
        if code != 0:
            raise StreamError(
                f"yt-dlp exited {code} for format {fmt}: "
                + (" | ".join(list(tail)[-3:]) or "no stderr")
            )
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.stdout and proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass
    verified = _verify(key, sent)
    if on_progress:
        on_progress(1.0)
    return verified


def _split_if_big(path: Path, *, target: int, seconds: float) -> list[tuple[Path, float]]:
    """Split a section that came out over the plan's single-object limit.

    A 720p source at ~1 Mbps gives ~19 MB per 150 s, but a high-bitrate one
    does not. Splitting with `-c copy` is a container remux — no re-encode, no
    change to the pixels a clip is cut from — and it keeps every object under
    the 25 MB the free plan accepts.
    """
    size = path.stat().st_size
    if size <= target:
        return [(path, seconds)]
    pieces = min(8, -(-size // target))
    each = seconds / pieces
    out: list[tuple[Path, float]] = []
    for i in range(pieces):
        part = path.with_name(f"{path.stem}_p{i}{path.suffix}")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{i * each:.3f}", "-i", str(path),
            "-t", f"{each:.3f}", "-c", "copy",
            "-avoid_negative_ts", "make_zero", str(part),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0 or not part.exists() or part.stat().st_size == 0:
            raise StreamError(
                f"a {mb(size)} MB section is over the 25 MB upload limit and could "
                f"not be split: {(proc.stderr or '').strip()[-200:]}"
            )
        out.append((part, each))
    path.unlink(missing_ok=True)
    return out


def section_key(prefix: str, job_id: str, index: int) -> str:
    """The bucket key a section is stored under — stable across restarts."""
    return f"{prefix}/uploads/{job_id}/sections/sec_{index:03d}.mp4"


def _section_count(total_sec: float) -> int:
    """How many 150 s sections to request for a source of `total_sec`.

    Unknown length used to mean "6 sections" = 15 minutes, which silently
    replaced a 20-minute source with its first three quarters. Ask for the
    whole allowed maximum instead; `_sections` breaks at the real end.
    """
    if total_sec > 0:
        return max(1, int(-(-total_sec // SECTION_SEC)))
    return UNKNOWN_DURATION_SECTIONS


def source_duration(
    *, probed: float, meta_sec: float, sections: list[dict[str, Any]],
) -> float:
    """Length of the SOURCE, never of the one object that got probed.

    A sections capture probes ONE 150 s object, so ffprobe reports 2:30 for a
    20:37 video. Storing that as `durationSec` cut the batch to the first two
    and a half minutes: auto clip count scaled off 2:30 and the detector never
    looked past the first window. For a whole-object capture the probe IS the
    source, so it wins.
    """
    if sections:
        return max(0.0, float(sections[-1].get("end") or 0.0))
    return float(probed or meta_sec or 0.0)


def _sections(
    url: str, job_id: str, *, total_sec: float, settings: Any,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[dict[str, Any]]:
    """Two-minute sections: download, single-PUT upload, verify, delete.

    This is the primary path on a plan that rejects multipart video. Each
    section is its own object under `uploads/<job>/sections/`, and the local
    file is removed the moment the bucket's HEAD confirms its size — so the
    disk holds at most ONE section while a source is being captured, instead
    of the whole video.
    """
    from app import objectstore

    prefix = objectstore.config_from_env().prefix
    count = _section_count(total_sec)
    out: list[dict[str, Any]] = []
    tmpdir = Path(tempfile.mkdtemp(prefix=f"sec-{job_id[:8]}-"))
    try:
        for i in range(count):
            start = i * SECTION_SEC
            end = min(start + SECTION_SEC, total_sec) if total_sec > 0 else start + SECTION_SEC
            # A section is uploaded to a DETERMINISTIC key and verified before
            # the local copy goes. So a capture that was interrupted by a host
            # restart can keep the sections that already landed instead of
            # re-downloading the video from 0:00 — the difference between
            # finishing a nine-section source and never getting past section 3.
            planned = section_key(prefix, job_id, i)
            try:
                existing = objectstore.head(planned) or {}
            except Exception:  # noqa: BLE001 - an unreadable bucket means re-capture
                existing = {}
            if int(existing.get("size") or 0) > 0:
                out.append({
                    "key": planned,
                    "bytes": int(existing["size"]),
                    "start": start,
                    "end": end,
                })
                if on_log:
                    on_log(
                        f"[stream] section {i + 1}/{count}: {start:.0f}s–{end:.0f}s "
                        "is already in the bucket — reusing it"
                    )
                if on_progress:
                    on_progress(min(1.0, (i + 1) / count))
                continue
            cmd = _cli(settings) + [
                "-f", VIDEO_FORMATS[0],
                "--download-sections", f"*{start:.0f}-{end:.0f}",
                "-o", str(tmpdir / f"sec_{i:03d}.mp4"),
                url,
            ]
            if on_log:
                on_log(f"[stream] section {i + 1}/{count}: {start:.0f}s–{end:.0f}s")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            files = sorted(tmpdir.glob(f"sec_{i:03d}.*"))
            if not files:
                if proc.returncode != 0 and not out:
                    raise StreamError(
                        "yt-dlp could not stream or section this source: "
                        + (proc.stderr or "").strip()[-300:]
                    )
                break
            local = files[0]
            span = end - start
            for k, (piece, piece_sec) in enumerate(
                _split_if_big(local, target=MAX_VIDEO_OBJECT_BYTES, seconds=span)
            ):
                name = f"sec_{i:03d}" if k == 0 and piece == local else f"sec_{i:03d}_p{k}"
                key = f"{prefix}/uploads/{job_id}/sections/{name}{piece.suffix}"
                size = piece.stat().st_size
                if not objectstore.upload_file(piece, key):
                    raise StreamError(
                        f"could not store section {i + 1} in the bucket", bucket_down=True,
                    )
                head = objectstore.head(key) or {}
                if int(head.get("size") or 0) != size:
                    raise StreamError(
                        f"section {i + 1} uploaded {size} B but the bucket reports "
                        f"{head.get('size')} B"
                    )
                piece.unlink(missing_ok=True)
                out.append({
                    "key": key, "bytes": size,
                    "start": start + k * piece_sec,
                    "end": start + (k + 1) * piece_sec,
                })
            local.unlink(missing_ok=True)
            if on_progress:
                on_progress(min(1.0, (i + 1) / count))
        if not out:
            raise StreamError("no sections were captured")
        return out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _probe_remote(key: str) -> dict[str, Any]:
    """ffprobe the object through a presigned URL: headers only, no download."""
    from app import objectstore, video

    url = objectstore.presign(key, expires=600)
    if not url:
        raise StreamError("could not presign the captured object", bucket_down=True)
    try:
        return video.probe(url)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[stream] probe of {key} failed: {exc}")
        return {}


class RemoteSource:
    """The source video as bucket keys. `str()` is a fresh presigned URL."""

    def __init__(
        self, job_id: str, *, video_key: str, audio_key: str,
        probe: dict[str, Any] | None = None, method: str = "stdout-stream",
        sections: list[dict[str, Any]] | None = None, bytes_total: int = 0,
        format_id: str = "", audio_format: str = "",
    ) -> None:
        self.job_id = job_id
        self.video_key = video_key
        self.audio_key = audio_key
        self.probe = probe or {}
        self.method = method
        self.sections = sections or []
        self.bytes_total = bytes_total
        self.format_id = format_id
        self.audio_format = audio_format

    # -- the two things downstream stages need -------------------------
    @property
    def name(self) -> str:
        """What error messages and logs call this source (there is no file)."""
        return f"bucket:{self.job_id}/video"

    def exists(self) -> bool:
        """Duck-type Path.exists() for the guards that ask before seeking."""
        return True

    def __str__(self) -> str:
        from app import objectstore

        return objectstore.presign(self.video_key, expires=3600) or ""

    def url_for_span(self, start_sec: float) -> tuple[str, float]:
        """(object URL, seek offset) for a clip starting at `start_sec`.

        With sections, a clip is cut from the one 5-minute section that holds
        its start — a clip is never longer than that, so no concatenation and
        no local copy is needed.
        """
        from app import objectstore

        if self.sections:
            for sec in self.sections:
                if sec["start"] <= start_sec < sec["end"]:
                    url = objectstore.presign(sec["key"], expires=900)
                    return url or "", start_sec - float(sec["start"])
            last = self.sections[-1]
            url = objectstore.presign(last["key"], expires=900)
            return url or "", max(0.0, start_sec - float(last["start"]))
        return objectstore.presign(self.video_key, expires=900) or "", 0.0

    def audio_url(self, *, expires: int = 900) -> str:
        from app import objectstore

        return objectstore.presign(self.audio_key, expires=expires) or ""

    def describe(self) -> dict[str, Any]:
        return {
            "videoKey": self.video_key,
            "audioKey": self.audio_key,
            "method": self.method,
            "format": self.format_id,
            "audioFormat": self.audio_format,
            "bytes": self.bytes_total,
            "sections": self.sections,
        }


def capture(
    url: str,
    job_id: str,
    settings: Any,
    *,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> RemoteSource:
    """Capture video-only + audio-only into the bucket. No local source file."""
    from app import objectstore

    if not objectstore.configured():
        raise StreamError(
            "no media bucket is configured, and a source is no longer stored "
            "on the local disk",
            bucket_down=True,
        )
    cfg = objectstore.config_from_env()
    base = f"{cfg.prefix}/uploads/{job_id}"

    meta: dict[str, Any] = {}
    try:
        from app import download as _dl

        meta = _dl.probe_url(url) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[stream] probe_url failed ({exc}); continuing blind")
    total_sec = float(meta.get("duration") or 0.0)

    # ---- audio first: it is tiny and transcription cannot start without it
    audio_key = f"{base}/audio.m4a"
    audio: dict[str, Any] | None = None
    audio_err = ""
    for fmt in AUDIO_FORMATS:
        try:
            audio = _stream_one(url, fmt, audio_key, settings, on_log=on_log)
            audio["format"] = fmt
            break
        except StreamError as exc:
            audio_err = str(exc)
            if exc.bucket_down:
                raise
            if on_log:
                on_log(f"[stream] audio format {fmt} could not stream — next")
    if audio is None:
        # Plan limits can also reject a multipart audio object. It is small
        # enough to stage briefly: download, single-PUT, verify, delete.
        from app import objectstore as _o

        tmpdir = Path(tempfile.mkdtemp(prefix=f"aud-{job_id[:8]}-"))
        try:
            out = tmpdir / "audio.m4a"
            cmd = _cli(settings) + ["-f", AUDIO_FORMATS[0], "-o", str(out), url]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
                raise StreamError(f"audio capture failed for every format: {audio_err}")
            if out.stat().st_size > 40 * 1024 * 1024:
                raise StreamError(
                    f"the audio track is {mb(out.stat().st_size)} MB, over the 40 MB "
                    "cap for transcription"
                )
            if not _o.upload_file(out, audio_key):
                raise StreamError("could not store the audio track", bucket_down=True)
            size = out.stat().st_size
            audio = _verify(audio_key, size)
            audio["format"] = AUDIO_FORMATS[0] + " (staged)"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- video, with sections when the container cannot be piped
    video_key = f"{base}/video.mp4"
    method = "stdout-stream"
    sections: list[dict[str, Any]] = []
    chosen = ""
    last_err = ""
    video: dict[str, Any] | None = None
    for fmt in VIDEO_FORMATS:
        try:
            info = _stream_one(
                url, fmt, video_key, settings, on_log=on_log, on_progress=on_progress,
            )
            chosen = fmt
            video = info
            break
        except StreamError as exc:
            last_err = str(exc)
            if "25MB" in last_err or "AccountProblem" in last_err:
                # The plan rejects multipart video above 25 MB: sectioning is
                # the only compliant route, and trying four formats would just
                # push four more rejected uploads.
                if on_log:
                    on_log(
                        "[stream] the bucket plan refuses multipart video over "
                        "25 MB — switching to 150 s sections"
                    )
                break
            if exc.bucket_down:
                raise
            if should_pause and should_pause():
                raise StreamError("paused during capture")
            if on_log:
                on_log(f"[stream] video format {fmt} cannot stream — trying next")

    if video is None:
        if on_log:
            on_log(
                "[stream] piping the whole video is not possible here — "
                "capturing 150 s sections instead"
            )
        sections = _sections(
            url, job_id, total_sec=total_sec, settings=settings,
            on_log=on_log, on_progress=on_progress,
        )
        video = {"key": sections[0]["key"], "bytes": sum(s["bytes"] for s in sections)}
        video_key = sections[0]["key"]
        method = "sections"
        if not last_err:
            last_err = "streaming unsupported"

    probe = _probe_remote(video_key)
    if not probe or not int(probe.get("width") or 0):
        probe = {
            "duration": total_sec,
            "width": 0,
            "height": 0,
            "fps": 30.0,
        }
    if not probe.get("duration") and total_sec:
        probe["duration"] = total_sec
    # The geometry comes from the probed object, the LENGTH comes from the
    # source. With sections those are different things, and confusing them is
    # what made a 20:37 video report "2:30 source" and render only its head.
    probed_dur = float(probe.get("duration") or 0.0)
    whole = source_duration(probed=probed_dur, meta_sec=total_sec, sections=sections)
    if whole > 0 and whole > probed_dur + 1.0:
        probe["duration"] = whole
        if on_log:
            on_log(
                f"[stream] source is {whole / 60:.1f} min — the probed object only "
                f"holds {probed_dur:.0f}s ({len(sections)} section(s) captured), "
                "so the full length is used"
            )
    elif sections and total_sec and whole + 1.0 < total_sec:
        if on_log:
            on_log(
                f"[stream] captured {whole / 60:.1f} min of a {total_sec / 60:.1f} min "
                "source — the batch will work with what was captured"
            )

    source = RemoteSource(
        job_id,
        video_key=video_key,
        audio_key=audio_key,
        probe=probe,
        method=method,
        sections=sections,
        bytes_total=int(video.get("bytes") or 0) + int(audio.get("bytes") or 0),
        format_id=chosen or "(sections)",
        audio_format=str(audio.get("format") or ""),
    )
    if on_log:
        on_log(
            f"[stream] captured video {mb(video.get('bytes'))} MB ({source.format_id}) + "
            f"audio {mb(audio.get('bytes'))} MB ({source.audio_format}) via {method}; "
            f"no local copy of the source"
        )
    return source


def download_audio_local(source: RemoteSource, dest: Path, *, cap_bytes: int) -> Path:
    """Pull ONLY the audio object to disk for transcription, then it is deleted."""
    from app import objectstore

    dest.parent.mkdir(parents=True, exist_ok=True)
    size = int((objectstore.head(source.audio_key) or {}).get("size") or 0)
    if size and size > cap_bytes:
        raise StreamError(
            f"the audio track alone is {mb(size)} MB, over the "
            f"{mb(cap_bytes)} MB local cap for transcription"
        )
    url = source.audio_url(expires=3600)
    if not url:
        raise StreamError("could not presign the audio object", bucket_down=True)
    # ffmpeg copies the container without re-encoding: same bytes, no decode.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", url, "-c", "copy", "-f", "ipod" if dest.suffix == ".m4a" else "mp4",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        # Fall back to a straight HTTP copy through the presigned URL.
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
                shutil.copyfileobj(resp, fh, 1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            raise StreamError(
                f"could not fetch the audio track for transcription: "
                f"{(proc.stderr or str(exc)).strip()[-300:]}",
                bucket_down=True,
            ) from exc
    if dest.stat().st_size > cap_bytes:
        dest.unlink(missing_ok=True)
        raise StreamError(
            f"the audio track is {mb(dest.stat().st_size if dest.exists() else size)} "
            f"MB, over the {mb(cap_bytes)} MB local cap"
        )
    return dest
