"""Central configuration for ClipForge local.

``config.toml`` holds the pruned MoneyPrinterTurbo-style settings; ``.env``
(secret keys) is loaded via python-dotenv. Validation fails fast at boot with
readable errors — no half-configured process.

The Gemini API key is resolved LIVE (see `_LiveKeyState`): the hosted platform
writes saved keys into env files without restarting the web process, so a
snapshot taken at boot would stay "missing" forever.
"""

from __future__ import annotations

import base64
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 dev/test environments
    import tomli as tomllib  # type: ignore[no-redef]

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# dotenvx-style secret decryption (Freebuff stores workspace keys encrypted)
# --------------------------------------------------------------------------- #

_ENC_PREFIX = "encrypted:"

# secp256k1 domain parameters (for the dotenvx ECIES shared-point math)
_SECP256K1_P = 2**256 - 2**32 - 977
_SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_SECP256K1_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
_SECP256K1_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def _ec_mul(scalar: int, point: tuple[int, int]) -> tuple[int, int]:
    """Double-and-add scalar multiplication on secp256k1 (affine)."""
    result: tuple[int, int] | None = None
    addend = point
    while scalar:
        if scalar & 1:
            result = _ec_add(result, addend)
        addend = _ec_add(addend, addend)
        scalar >>= 1
    return result or (0, 0)


def _ec_add(p1: tuple[int, int] | None, p2: tuple[int, int]) -> tuple[int, int]:
    """Affine point addition on secp256k1."""
    if p1 is None:
        return p2
    if p1[0] == p2[0] and (p1[1] + p2[1]) % _SECP256K1_P == 0:
        return (0, 0)
    if p1 == p2:
        lam = (3 * p1[0] * p1[0]) * pow(2 * p1[1], -1, _SECP256K1_P) % _SECP256K1_P
    else:
        lam = (p2[1] - p1[1]) * pow(p2[0] - p1[0], -1, _SECP256K1_P) % _SECP256K1_P
    x3 = (lam * lam - p1[0] - p2[0]) % _SECP256K1_P
    return (x3, (lam * (p1[0] - x3) - p1[1]) % _SECP256K1_P)


def _dotenvx_decrypt(private_key_hex: str, encrypted_b64: str) -> str:
    """Decrypt a dotenvx `encrypted:…` value with the matching private key.

    Blob layout (base64): ephemeral_pubkey[65] || nonce[16] || tag[16] || ct.

    Byte-compatible with @dotenvx/primitives (BSD-3-Clause), verified
    bidirectionally (see tools/decrypt_confirm.py):

    - The HKDF input key material is ephemeral_pubkey(65) || FULL uncompressed
      shared point(65) — the shared point INCLUDES the 0x04 prefix. The lib's
      fallback path calls noble's `Point.multiply(scalar).toBytes()`, whose
      default serialization is compressed — but the ECIES config wraps it via
      getSharedPoint(secp256k1, …) which returns the uncompressed point.
    - HKDF-SHA256 with an EMPTY salt (which cryptography's HKDF(salt=b"")
      and @noble/hashes both treat per RFC 5869 as zeros — identical results
      here), info=None, length=32.
    - AES-256-GCM, nonce = blob[65:81], ciphertext+tag = blob[81:].
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    raw = base64.b64decode(encrypted_b64)
    if len(raw) < 97:  # 65 pubkey + 16 nonce + 16 tag minimum
        raise ValueError("dotenvx blob too short to contain a valid envelope")
    eph_pub, rest = raw[:65], raw[65:]
    nonce, rest = rest[:16], rest[16:]
    tag, ct = rest[:16], rest[16:]

    scalar = int(private_key_hex, 16) % _SECP256K1_N
    eph_x = int.from_bytes(eph_pub[1:33], "big")
    eph_y = int.from_bytes(eph_pub[33:65], "big")
    sx, sy = _ec_mul(scalar, (eph_x, eph_y))
    # FULL uncompressed shared point — 0x04 prefix REQUIRED (65 bytes total).
    # Without it the AES key diverges and GCM auth fails (InvalidTag).
    shared_point = b"\x04" + sx.to_bytes(32, "big") + sy.to_bytes(32, "big")

    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"", info=None)
    aes_key = hkdf.derive(eph_pub + shared_point)
    return AESGCM(aes_key).decrypt(nonce, ct + tag, None).decode("utf-8")


def _decrypt_encrypted_values(environ: dict[str, str]) -> dict[str, str]:
    """Decrypt every `encrypted:…` value that has a matching private key.

    dotenvx writes ciphertext into .env files and the private key into
    .env.keys (DOTENV_PRIVATE_KEY_<ENV>). python-dotenv loads both verbatim,
    so values arrive here still encrypted — decrypt them in place. Errors are
    logged and left as-is (the app reports them as unusable rather than
    silently pretending the key works).
    """
    priv_names = [k for k in list(environ.keys()) if k.startswith("DOTENV_PRIVATE_KEY_")]
    if not priv_names:
        return environ
    for name in list(environ.keys()):
        value = environ.get(name, "")
        if not value or not value.startswith(_ENC_PREFIX):
            continue
        for priv_name in priv_names:
            priv_hex = environ.get(priv_name, "").strip()
            if not priv_hex:
                continue
            try:
                environ[name] = _dotenvx_decrypt(priv_hex, value[len(_ENC_PREFIX):])
                break
            except Exception:
                continue
    return environ


GEMINI_KEY_ALIASES = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "GEMINI_KEY",
    "VITE_GEMINI_API_KEY",
)

_PLACEHOLDER_VALUES = {
    "changeme", "change-me", "your-api-key", "your_api_key",
    "your-key-here", "<paste-key-here>", "none", "null", "todo", "xxx",
}


def _fresh_env_snapshot(root: Path) -> tuple[str, dict[str, str]]:
    """Fingerprint + raw contents of the project's env files.

    The fingerprint changes whenever the platform writes a new key into
    .env/.env.local/.env.keys — which is what makes live re-detection possible
    without restarting the web process.
    """
    parts: list[str] = []
    merged: dict[str, str] = {}
    for name in (".env", ".env.local", ".env.keys"):
        p = root / name
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            parts.append(f"{name}:{len(text)}:{int(p.stat().st_mtime)}")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                merged.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    # Decrypt dotenvx-encrypted values in the snapshot itself (.env.keys
    # private keys are part of the merge). Without this, the file fallback
    # would misreport a decryptable platform-saved key as "unusable
    # (encrypted)" even though _load_env_files decrypts it into os.environ.
    _decrypt_encrypted_values(merged)
    return "|".join(parts), merged


def _empty_encrypted(value: str) -> bool:
    """True when an `encrypted:` blob contains ZERO plaintext bytes.

    dotenvx blob = base64(ephemeral_pubkey[65] || nonce[16] || tag[16] || ct);
    ciphertext length == plaintext length, so a 97-byte blob means the
    platform saved an EMPTY value — an envelope with no key inside.
    """
    if not value.startswith(_ENC_PREFIX):
        return False
    try:
        raw = base64.b64decode(value[len(_ENC_PREFIX):] + "=" * (-len(value) % 4))
    except Exception:
        return False
    return len(raw) == 97


def _resolve_gemini_key(environ: dict[str, str]) -> tuple[str | None, str, str | None]:
    """Find the Gemini key under any supported name.

    Returns (key, source, problem). `source` says WHERE the key came from so
    misconfigurations are diagnosable from the UI; `problem` is a
    human-readable reason when a key-shaped value exists but is unusable
    (empty platform save, undecryptable blob, placeholder text, malformed).
    Never returns the secret itself.
    """
    for name in GEMINI_KEY_ALIASES:
        value = (environ.get(name) or "").strip()
        if not value:
            continue
        if value.startswith(_ENC_PREFIX):
            if _empty_encrypted(value):
                return None, f"{name} (empty)", (
                    f"{name} is an EMPTY save — the platform stored an encrypted envelope "
                    "with no key inside. Paste the full key (starts with AIza) in the "
                    "Keys/API keys tab again, then hit Re-check."
                )
            return None, f"{name} (encrypted)", (
                f"{name} is stored in the platform's encrypted dotenvx format and could "
                "not be decrypted inside the app — re-save it in the Keys/API keys tab, "
                "then hit Re-check."
            )
        low = value.lower()
        if low in _PLACEHOLDER_VALUES:
            return None, f"{name} (placeholder)", (
                f"{name} still contains placeholder text — paste the real Gemini API key."
            )
        if " " in value:
            return None, f"{name} (malformed)", (
                f"{name} contains spaces — it should be the raw key with no quotes or spaces."
            )
        if name == "GEMINI_API_KEY" and not low.startswith("aiza") and len(value) < 20:
            # A short non-Google-looking value is almost always a paste error.
            # Still accepted (key shapes can change), but flagged for the UI.
            return value, f"{name} (short — verify)", (
                f"{name} is only {len(value)} characters — a Google API key is longer. "
                "Double-check the paste."
            )
        return value, name, None
    return None, "", None


def _load_env_files(root: Path) -> list[str]:
    """Load .env plus platform-injected env files, decrypting dotenvx values.

    dotenv does not override variables already set in the process environment,
    so OS-provided secrets always win over file contents. Values stored in the
    dotenvx `encrypted:…` format (how the platform saves workspace keys) are
    decrypted in place using the private key from .env.keys.
    """
    loaded: list[str] = []
    for name in (".env", ".env.local", ".env.keys"):
        p = root / name
        if p.exists():
            load_dotenv(p, override=False)
            loaded.append(name)
    # Decrypt any encrypted values that python-dotenv loaded verbatim. Only
    # platform-provided keys are ever in this format; a plaintext GEMINI_API_KEY
    # (starts with AIza) passes through untouched.
    _decrypt_encrypted_values(os.environ)
    return loaded


class _LiveKeyState:
    """Re-resolves the Gemini key when the env files change on disk.

    The platform writes saved keys into .env.local (encrypted dotenvx format)
    WITHOUT restarting the web process. A key captured once at boot would stay
    None forever — so this state re-reads the env files whenever their
    mtime/size fingerprint changes (plus on demand), and records exactly where
    the key was found (or why it was rejected). Never stores logs the secret.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._fingerprint: str | None = None
        self._lock = threading.Lock()
        self.key: str | None = None
        self.source: str = "not checked"
        self.problem: str | None = None
        self.lastCheckedAt: float = 0.0

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            fingerprint, file_values = _fresh_env_snapshot(self._root)
            if not force and fingerprint == self._fingerprint:
                return
            self._fingerprint = fingerprint

            # Re-load env files into the process (idempotent; never overrides
            # real OS env), then decrypt any platform-encrypted values.
            _load_env_files(self._root)

            # Resolution order: a USABLE OS-env key wins (platform-injected
            # secrets). If OS env has a value but it is UNUSABLE (stale
            # encrypted blob, placeholder, malformed), fall through to the
            # env-file contents — the file is fresher than the process env
            # (e.g. after the app itself rewrites .env.local).
            key, source, problem = _resolve_gemini_key(os.environ)
            if key is None:
                fkey, fsource, fproblem = _resolve_gemini_key(file_values)
                if fkey is not None:
                    key, source, problem = fkey, f"{fsource} (.env.local)", None
                elif problem is None and fproblem is not None:
                    key, source, problem = None, fsource, fproblem
            self.key = key
            self.source = source or "not found in env files or OS env"
            self.problem = problem
            self.lastCheckedAt = time.time()
            if key is not None:
                logger.info(f"[config] Gemini key detected ({source})")
            elif problem:
                logger.warning(f"[config] Gemini key rejected: {problem}")

    def status(self) -> dict:
        self.refresh()
        return {
            "keyLoaded": self.key is not None,
            "source": self.source,
            "problem": self.problem,
            "lastCheckedAt": self.lastCheckedAt,
            "aliases": list(GEMINI_KEY_ALIASES),
        }


_live_key: _LiveKeyState | None = None


def live_gemini_key() -> _LiveKeyState:
    """Process-wide live Gemini-key state (created on first use)."""
    global _live_key
    if _live_key is None:
        _live_key = _LiveKeyState(ROOT)
    return _live_key


# --------------------------------------------------------------------------- #
# env-file management (the platform Keys UI has no edit/delete — the app
# provides it for GEMINI_API_KEY by rewriting .env.local directly)
# --------------------------------------------------------------------------- #

ENV_LOCAL = ROOT / ".env.local"


def write_env_var(name: str, value: str) -> None:
    """Add or replace a variable in .env.local (atomic tmp+rename)."""
    import re as _re

    if not _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"invalid variable name: {name!r}")
    lines: list[str] = []
    if ENV_LOCAL.exists():
        lines = ENV_LOCAL.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            existing = stripped.split("=", 1)[0].strip()
            if existing == name:
                if not replaced:
                    out.append(f"{name}={value}")
                    replaced = True
                continue  # drop duplicate definitions
        out.append(line)
    if not replaced:
        out.append(f"{name}={value}")
    tmp = ENV_LOCAL.with_suffix(".local.tmp")
    tmp.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    tmp.replace(ENV_LOCAL)


def sync_gemini_env_after_save(value: str) -> None:
    """Mirror a freshly written GEMINI_API_KEY into the process env.

    The live resolver prefers OS env, so without this a stale copy loaded
    earlier from an older .env.local would shadow the just-saved value — the
    "I replaced the key but it still reports the old one" bug.
    """
    os.environ["GEMINI_API_KEY"] = value


def sync_gemini_env_for_delete() -> list[str]:
    """Drop Gemini key aliases from the process env before a file delete.

    Only GEMINI_API_KEY itself and aliases the env files currently define are
    popped — those are the copies _load_env_files injected. Platform secrets
    injected as real OS env (never via files) are left alone.
    """
    _fingerprint, file_values = _fresh_env_snapshot(ROOT)
    dropped: list[str] = []
    for name in GEMINI_KEY_ALIASES:
        if name == "GEMINI_API_KEY" or name in file_values:
            if os.environ.pop(name, None) is not None:
                dropped.append(name)
    return dropped


def delete_env_var(name: str) -> int:
    """Remove every definition of `name` from .env.local. Returns lines removed."""
    if not ENV_LOCAL.exists():
        return 0
    out: list[str] = []
    removed = 0
    for line in ENV_LOCAL.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            if stripped.split("=", 1)[0].strip() == name:
                removed += 1
                continue
        out.append(line)
    if removed:
        tmp = ENV_LOCAL.with_suffix(".local.tmp")
        tmp.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
        tmp.replace(ENV_LOCAL)
    return removed


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
    # Default caption style preset: bold_pop | karaoke | lower_third | boxed
    preset: str = "bold_pop"


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
class HostedConfig:
    """Hosted-mode limits and remote-worker dispatch settings."""

    # Plan guard: reject sources longer than this at SUBMIT time. Long sources
    # ARE supported — processing time scales proportionally, see [hosted] notes.
    max_source_min: int = 120
    # Delete finished artifacts (outputs/ + downloads/) this long after a job
    # completes — hosted storage is finite and users expect cleanup.
    retention_hours: int = 24
    # Remote worker dispatch: when set (e.g. http://100.x.y.z:8765), the API
    # sends jobs to the user's home machine over Tailscale instead of running
    # the pipeline locally in the sandbox.
    worker_url: str = ""
    # Token the remote worker-agent must present (Authorization: Bearer ...).
    worker_token: str = ""
    # Serve the pipeline API only inside the user's tailnet when true.
    tailscale_only: bool = False


@dataclass
class Settings:
    app: AppConfig
    whisper: WhisperConfig
    gemini: GeminiConfig
    subtitle: SubtitleConfig
    video: VideoConfig
    download: DownloadConfig
    hosted: HostedConfig

    @property
    def gemini_api_key(self) -> str | None:
        """LIVE Gemini key — re-resolved whenever the env files change.

        Never snapshot this at boot: the platform can deliver the key after
        the process has already started.
        """
        state = live_gemini_key()
        state.refresh()
        return state.key


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
    _load_env_files(root)

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

    # Env overrides for the working folders — used by tests/tools to isolate
    # from the real job store (never by the hosted app itself).
    paths = Paths(
        downloads_dir=Path(os.getenv("CLIPFORGE_DOWNLOADS_DIR", root / str(app.get("downloads_dir", "downloads")))),
        outputs_dir=Path(os.getenv("CLIPFORGE_OUTPUTS_DIR", root / str(app.get("outputs_dir", "outputs")))),
        data_dir=Path(os.getenv("CLIPFORGE_DATA_DIR", root / str(app.get("data_dir", "data")))),
        static_dir=root / "webui",
    )
    for name in ("downloads_dir", "outputs_dir", "data_dir"):
        try:
            getattr(paths, name).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"cannot create {name}: {exc}")

    w = raw.get("whisper", {})
    # env override wins (tests / low-memory hosts): CLIPFORGE_WHISPER_MODEL
    model = str(os.getenv("CLIPFORGE_WHISPER_MODEL") or w.get("model", "medium"))
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
        model=str(g.get("model", "gemini-3.6-flash")),
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
        preset=str(s.get("preset", "bold_pop")),
    )
    if subtitle.preset not in {"bold_pop", "karaoke", "lower_third", "boxed"}:
        errors.append(
            f'subtitle.preset "{subtitle.preset}" is not one of '
            "bold_pop/karaoke/lower_third/boxed"
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

    h = raw.get("hosted", {})
    hosted = HostedConfig(
        max_source_min=max(1, int(h.get("max_source_min", os.getenv("CLIPFORGE_MAX_SOURCE_MIN", 120)))),
        retention_hours=max(1, int(h.get("retention_hours", os.getenv("CLIPFORGE_RETENTION_HOURS", 24)))),
        worker_url=(os.getenv("WORKER_URL") or h.get("worker_url", "") or "").strip().rstrip("/"),
        worker_token=(os.getenv("WORKER_TOKEN") or h.get("worker_token", "") or "").strip(),
        tailscale_only=str(os.getenv("TAILSCALE_ONLY", str(h.get("tailscale_only", False)))).strip().lower()
        in {"1", "true", "yes", "on"},
    )

    # Initialize the live key state so the boot banner reports it. The key is
    # NEVER snapshotted into Settings — every read re-resolves.
    live_gemini_key().refresh(force=True)

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
        hosted=hosted,
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
    live = live_gemini_key()
    live.refresh(force=True)
    key_loaded = live.key is not None
    print(f"[boot] LLM: gemini | key loaded: {'YES' if key_loaded else 'NO'}")
    print(f"[boot] Gemini key source: {live.source}")
    if not key_loaded:
        if live.problem:
            print(f"[boot] Gemini key problem: {live.problem}")
        print(
            "[boot] AI detection offline — add GEMINI_API_KEY in the Keys/API keys tab "
            "(or .env), then hit Re-check on the dashboard"
        )
    env_files = _load_env_files(ROOT)
    print(f"[boot] env files loaded: {', '.join(env_files) if env_files else 'none found'}")
    print(f"[boot] Whisper model: {settings.whisper.model} ({settings.whisper.resolved_compute})")
    print(f"[boot] Output: {settings.video.width}x{settings.video.height} @ {settings.video.fps}fps CRF {settings.video.crf}")
    print(f"[boot] Folders: {settings.app.paths.downloads_dir} / {settings.app.paths.outputs_dir} / {settings.app.paths.data_dir}")
    print(f"[boot] Mode: {'tailscale-only' if settings.hosted.tailscale_only else 'public'} | max source: {settings.hosted.max_source_min} min | retention: {settings.hosted.retention_hours}h")
    if settings.hosted.worker_url:
        print(f"[boot] Worker dispatch: {settings.hosted.worker_url} (rendering happens on the remote worker)")
    else:
        print("[boot] Worker dispatch: local (pipeline runs in this process)")


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
