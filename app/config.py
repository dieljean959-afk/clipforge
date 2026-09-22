"""Central configuration for ClipForge local.

``config.toml`` holds the pruned MoneyPrinterTurbo-style settings; ``.env``
(secret keys) is loaded via python-dotenv. Validation fails fast at boot with
readable errors — no half-configured process.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 dev/test environments
    import tomli as tomllib  # type: ignore[no-redef]

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    downloads_dir: Path
    outputs_dir: Path
    data_dir: Path
    static_dir: Path


@dataclass
class AppConfig:
    host: str
    port: int
    paths: Paths


@dataclass
class WhisperConfig:
    model: str
    device: str
    compute_type: str
    language: str

    @property
    def resolved_compute(self) -> str:
        if self.compute_type != "auto":
            return self.compute_type
        try:
            import ctranslate2  # noqa: PLC0415

            if ctranslate2.get_cuda_device_count() > 0:
                return "float16"
        except Exception:
            pass
        return "int8"


@dataclass
class GeminiConfig:
    model: str
    min_clip_sec: int
    max_clip_sec: int
    max_clips: int
    max_retries: int


@dataclass
class SubtitleConfig:
    font: str
    font_fallbacks: list[str] = field(default_factory=list)
    font_size: int = 68
    outline_color: str = "#39FF14"
    outline_width: int = 5
    words_per_cue: int = 2
    position_y: float = 0.78
    pop_animation: bool = True


@dataclass
class VideoConfig:
    width: int
    height: int
    fps: int
    crf: int
    preset: str
    reframe: bool
    silence_trim: bool


@dataclass
class DownloadConfig:
    max_height: int
    cookies_from_browser: str
    auto_update: bool


@dataclass
class Settings:
    app: AppConfig
    whisper: WhisperConfig
    gemini: GeminiConfig
    subtitle: SubtitleConfig
    video: VideoConfig
    download: DownloadConfig
    gemini_api_key: str | None


class ConfigError(Exception):
    """Raised when config.toml or the environment is invalid at boot."""


def _load_toml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config.toml is not valid TOML: {exc}") from exc


def load_config(root: Path = ROOT, quiet: bool = False) -> Settings:
    """Load and validate all settings. Raises ConfigError with readable text."""
    raw = _load_toml(root / "config.toml")
    load_dotenv(root / ".env")

    errors: list[str] = []

    app = raw.get("app", {})
    host = str(app.get("host", "127.0.0.1"))
    try:
        port = int(app.get("port", 8000))
    except (TypeError, ValueError):
        errors.append("app.port must be an integer")
        port = 8000
    if not (1 <= port <= 65535):
        errors.append(f"app.port {port} is out of range (1-65535)")

    paths = Paths(
        downloads_dir=root / str(app.get("downloads_dir", "downloads")),
        outputs_dir=root / str(app.get("outputs_dir", "outputs")),
        data_dir=root / str(app.get("data_dir", "data")),
        static_dir=root / "webui",
    )
    for name in ("downloads_dir", "outputs_dir", "data_dir"):
        try:
            getattr(paths, name).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"cannot create {name}: {exc}")

    w = raw.get("whisper", {})
    model = str(w.get("model", "small"))
    if model not in {"tiny", "base", "small", "medium", "large-v3"}:
        errors.append(
            f'whisper.model "{model}" is not one of tiny/base/small/medium/large-v3'
        )
    whisper = WhisperConfig(
        model=model,
        device=str(w.get("device", "auto")),
        compute_type=str(w.get("compute_type", "auto")),
        language=str(w.get("language", "auto")),
    )

    g = raw.get("gemini", {})
    gemini = GeminiConfig(
        model=str(g.get("model", "gemini-2.0-flash")),
        min_clip_sec=max(5, int(g.get("min_clip_sec", 15))),
        max_clip_sec=max(15, int(g.get("max_clip_sec", 60))),
        max_clips=min(15, max(5, int(g.get("max_clips", 10)))),
        max_retries=max(0, int(g.get("max_retries", 3))),
    )
    if gemini.min_clip_sec >= gemini.max_clip_sec:
        errors.append("gemini.min_clip_sec must be < gemini.max_clip_sec")

    s = raw.get("subtitle", {})
    outline = str(s.get("outline_color", "#39FF14"))
    if not outline.startswith("#") or len(outline) not in (4, 7):
        errors.append(f'subtitle.outline_color "{outline}" must be a hex color like #39FF14')
    subtitle = SubtitleConfig(
        font=str(s.get("font", "Arial Black")),
        font_fallbacks=[str(f) for f in s.get("font_fallbacks", [])],
        font_size=max(20, int(s.get("font_size", 68))),
        outline_color=outline,
        outline_width=max(0, int(s.get("outline_width", 5))),
        words_per_cue=max(1, int(s.get("words_per_cue", 2))),
        position_y=min(0.95, max(0.3, float(s.get("position_y", 0.78)))),
        pop_animation=bool(s.get("pop_animation", True)),
    )

    v = raw.get("video", {})
    video = VideoConfig(
        width=int(v.get("width", 1080)),
        height=int(v.get("height", 1920)),
        fps=int(v.get("fps", 30)),
        crf=int(v.get("crf", 20)),
        preset=str(v.get("preset", "veryfast")),
        reframe=bool(v.get("reframe", True)),
        silence_trim=bool(v.get("silence_trim", True)),
    )
    if video.width <= 0 or video.height <= 0:
        errors.append("video.width/height must be positive")

    d = raw.get("download", {})
    cookies = str(os.getenv("YT_DLP_COOKIES_FROM_BROWSER", d.get("cookies_from_browser", "") or ""))
    download = DownloadConfig(
        max_height=int(d.get("max_height", 1080)),
        cookies_from_browser=cookies,
        auto_update=bool(d.get("auto_update", True)),
    )

    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip() or None

    if errors:
        raise ConfigError(
            "Invalid configuration:\n  - " + "\n  - ".join(errors)
        )

    if not quiet:
        pass  # boot logging happens in main, not at import

    return Settings(
        app=AppConfig(host=host, port=port, paths=paths),
        whisper=whisper,
        gemini=gemini,
        subtitle=subtitle,
        video=video,
        download=download,
        gemini_api_key=gemini_key,
    )


def ffmpeg_bin() -> str:
    custom = (os.getenv("FFMPEG_PATH") or "").strip()
    if custom:
        return custom
    found = shutil.which("ffmpeg")
    if found is None:
        raise ConfigError(
            "ffmpeg not found on PATH. Install it (e.g. `winget install Gyan.FFmpeg`,"
            " `brew install ffmpeg`, or `sudo apt install ffmpeg`) or set FFMPEG_PATH in .env"
        )
    return found


def ffprobe_bin() -> str:
    custom = (os.getenv("FFPROBE_PATH") or "").strip()
    if custom:
        return custom
    found = shutil.which("ffprobe")
    if found is None:
        raise ConfigError(
            "ffprobe not found on PATH. Install ffmpeg (includes ffprobe) or set FFPROBE_PATH"
        )
    return found


def boot_banner(settings: Settings) -> None:
    """Print the fail-fast boot banner with the exact lines ops greps for."""
    key_loaded = settings.gemini_api_key is not None
    print(f"[boot] LLM: gemini | key loaded: {'YES' if key_loaded else 'NO'}")
    if not key_loaded:
        print("[boot] AI detection offline — add GEMINI_API_KEY to .env")
    print(f"[boot] Whisper model: {settings.whisper.model} ({settings.whisper.resolved_compute})")
    print(f"[boot] Output: {settings.video.width}x{settings.video.height} @ {settings.video.fps}fps CRF {settings.video.crf}")
    print(f"[boot] Folders: {settings.app.paths.downloads_dir} / {settings.app.paths.outputs_dir} / {settings.app.paths.data_dir}")


def _self_check() -> int:
    """Compile-time sanity check used by tests: load config, print, exit 0."""
    try:
        settings = load_config()
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1
    boot_banner(settings)
    return 0


if __name__ == "__main__":
    t0 = time.time()
    raise SystemExit(_self_check())
