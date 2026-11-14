"""Creator-uploaded subtitles: fetch, parse, and hand over the TEXT.

Step 1a of the accuracy plan is "if the video has manual subtitles, use their
text and force-align it". Manual captions are written by a human, so the words
are right; only the timing is coarse (one cue per line, often 3-5 s). That is
exactly the case forced alignment fixes.

Only MANUAL captions are accepted here. YouTube's auto captions are ASR output
from Google's model: reusing them would mean shipping somebody else's errors
under a "creator" label, and the plan already has a better local ASR. They are
still fetched separately, as a WER REFERENCE for the self-check.

yt-dlp writes manual captions with `--write-subs` and auto ones with
`--write-auto-subs`, so the distinction is made by the download flags rather
than by guessing from the filename.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

#: Languages worth fetching, most specific first.
DEFAULT_LANGS = ("en", "en-US", "en-GB", "es", "es-ES", "es-419")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _yt_dlp_bin() -> str:
    import shutil
    import sys

    found = shutil.which("yt-dlp")
    if found:
        return found
    return f"{sys.executable} -m yt_dlp"


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")  # inline <c> / <00:00:01.000> karaoke tags
    text = text.replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_json3(raw: str) -> list[Cue]:
    data = json.loads(raw)
    cues: list[Cue] = []
    for ev in data.get("events") or []:
        segs = ev.get("segs") or []
        text = _clean("".join(str(s.get("utf8") or "") for s in segs))
        if not text:
            continue
        start = float(ev.get("tStartMs") or 0) / 1000.0
        dur = float(ev.get("dDurationMs") or 0) / 1000.0
        cues.append(Cue(start, start + max(dur, 0.2), text))
    return _merge_duplicate_cues(cues)


def parse_vtt(raw: str) -> list[Cue]:
    cues: list[Cue] = []
    lines = raw.replace("\r\n", "\n").split("\n")
    i = 0
    stamp = re.compile(
        r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})"
    )
    while i < len(lines):
        m = stamp.search(lines[i])
        if not m:
            i += 1
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i])
            i += 1
        text = _clean(" ".join(body))
        if text:
            cues.append(Cue(start, max(end, start + 0.2), text))
    return _merge_duplicate_cues(cues)


def parse_srt(raw: str) -> list[Cue]:
    return parse_vtt(raw.replace("\r\n", "\n"))


def parse_file(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    name = path.name.lower()
    if name.endswith(".json3") or name.endswith(".json"):
        return parse_json3(raw)
    return parse_vtt(raw)


def _merge_duplicate_cues(cues: list[Cue]) -> list[Cue]:
    """Drop the rolling-window repeats YouTube emits in auto captions."""
    out: list[Cue] = []
    for cue in cues:
        if out and _norm(cue.text) == _norm(out[-1].text):
            continue
        if out and cue.text.startswith(out[-1].text) and cue.start - out[-1].end < 0.05:
            out[-1] = Cue(out[-1].start, cue.end, cue.text)
            continue
        out.append(cue)
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def find_local_subs(media: Path) -> list[Path]:
    """Subtitle files sitting next to the media (yt-dlp's default layout)."""
    stem = media.with_suffix("")
    out = []
    for ext in (".json3", ".vtt", ".srt"):
        for candidate in media.parent.glob(f"{stem.name}*{ext}"):
            out.append(candidate)
    return sorted(out)


def fetch(
    url: str,
    dest_dir: Path,
    *,
    languages: tuple[str, ...] = DEFAULT_LANGS,
    auto: bool = False,
    timeout: float = 240.0,
) -> list[Path]:
    """Download captions for `url`; returns the files that landed.

    `auto=False` fetches CREATOR subtitles only; `auto=True` fetches the
    auto-generated ones too, for use as a WER reference — never as transcript
    text.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_yt_dlp_bin().split(),
        "--skip-download",
        "--write-subs" if not auto else "--write-auto-subs",
        "--sub-format", "json3/vtt/srt",
        "--sub-langs", ",".join(languages),
        "-o", str(dest_dir / "%(id)s.%(ext)s"),
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"[subs] caption fetch timed out after {timeout:.0f}s")
        return []
    if proc.returncode != 0:
        logger.warning(f"[subs] caption fetch failed: {(proc.stderr or '').strip()[-300:]}")
        return []
    files = sorted(p for p in dest_dir.iterdir() if p.suffix in {".json3", ".vtt", ".srt"})
    return files


def pick_best(paths: list[Path], prefer: str | None = None) -> Path | None:
    """Prefer the requested language, then English, then whatever exists."""
    if not paths:
        return None
    if prefer:
        for p in paths:
            if f".{prefer}" in p.name:
                return p
    for p in paths:
        if ".en" in p.name:
            return p
    return paths[0]


def cues_text(cues: list[Cue]) -> str:
    return " ".join(c.text for c in cues).strip()
