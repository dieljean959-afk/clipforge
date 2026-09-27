"""Central configuration for the hosted ClipForge service.

``config.toml`` holds the pruned MoneyPrinterTurbo-style settings. Secret keys
are supplied by the platform and loaded via python-dotenv. Validation fails
fast at boot with readable errors — no half-configured process.

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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 dev/test environments
    import tomli as tomllib  # type: ignore[no-redef]

from dotenv import load_dotenv

from app import translate as translate_mod  # no cycle: translate imports lazily

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

# Groq is the DEFAULT detection backend (see [llm] provider). Gemini stays
# fully wired as a selectable alternative — a free-tier 429 on one provider
# should not need a redeploy to route around.
GROQ_KEY_ALIASES = (
    "GROQ_API_KEY",
    "GROQ_KEY",
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


def _resolve_groq_key(environ: dict[str, str]) -> tuple[str | None, str, str | None]:
    """Find the Groq key under any supported name.

    Same (key, source, problem) contract as _resolve_gemini_key, kept as a
    separate function rather than a shared parameterised one so the Gemini
    wording that the dashboard and key_flow tests assert on cannot drift.
    Groq keys are `gsk_…`, so the prefix check is the mirror image of
    Gemini's `AIza` check.
    """
    for name in GROQ_KEY_ALIASES:
        value = (environ.get(name) or "").strip()
        if not value:
            continue
        if value.startswith(_ENC_PREFIX):
            if _empty_encrypted(value):
                return None, f"{name} (empty)", (
                    f"{name} is an EMPTY save — the platform stored an encrypted envelope "
                    "with no key inside. Paste the full key (starts with gsk_) in the "
                    "Keys/API keys tab again, then hit Re-check."
                )
            return None, f"{name} (encrypted)", (
                f"{name} is stored in the platform's encrypted dotenvx format and could "
                "not be decrypted inside the app — re-save it in the Keys/API keys tab, "
                "then hit Re-check."
            )
        if value.lower() in _PLACEHOLDER_VALUES:
            return None, f"{name} (placeholder)", (
                f"{name} still contains placeholder text — paste the real Groq API key."
            )
        if " " in value:
            return None, f"{name} (malformed)", (
                f"{name} contains spaces — it should be the raw key with no quotes or spaces."
            )
        if not value.lower().startswith("gsk_") and len(value) < 20:
            return value, f"{name} (short — verify)", (
                f"{name} is only {len(value)} characters and does not start with gsk_ — "
                "double-check the paste from console.groq.com/keys."
            )
        return value, name, None
    return None, "", None


def _as_list(value: object) -> list[object]:
    """Coerce a config value to a list.

    TOML gives real arrays, but env overrides and hand-edited single-line
    values arrive as comma-separated strings — accept both rather than failing
    on a scalar with a confusing error.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


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

    def __init__(
        self,
        root: Path,
        *,
        resolver: Callable[[dict[str, str]], tuple[str | None, str, str | None]] | None = None,
        aliases: tuple[str, ...] = GEMINI_KEY_ALIASES,
        label: str = "Gemini",
    ) -> None:
        self._root = root
        # Resolver/aliases/label default to Gemini so existing callers of
        # live_gemini_key() behave exactly as before; Groq passes its own.
        self._resolver = resolver or _resolve_gemini_key
        self._aliases = aliases
        self._label = label
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
            key, source, problem = self._resolver(os.environ)
            if key is None:
                fkey, fsource, fproblem = self._resolver(file_values)
                if fkey is not None:
                    key, source, problem = fkey, f"{fsource} (.env.local)", None
                elif problem is None and fproblem is not None:
                    key, source, problem = None, fsource, fproblem
            self.key = key
            self.source = source or "not found in env files or OS env"
            self.problem = problem
            self.lastCheckedAt = time.time()
            if key is not None:
                logger.info(f"[config] {self._label} key detected ({source})")
            elif problem:
                logger.warning(f"[config] {self._label} key rejected: {problem}")

    def status(self) -> dict:
        self.refresh()
        return {
            "keyLoaded": self.key is not None,
            "source": self.source,
            "problem": self.problem,
            "lastCheckedAt": self.lastCheckedAt,
            "aliases": list(self._aliases),
        }


_live_key: _LiveKeyState | None = None
_live_groq: _LiveKeyState | None = None


def live_gemini_key() -> _LiveKeyState:
    """Process-wide live Gemini-key state (created on first use)."""
    global _live_key
    if _live_key is None:
        _live_key = _LiveKeyState(ROOT)
    return _live_key


def live_groq_key() -> _LiveKeyState:
    """Process-wide live Groq-key state (created on first use)."""
    global _live_groq
    if _live_groq is None:
        _live_groq = _LiveKeyState(
            ROOT, resolver=_resolve_groq_key, aliases=GROQ_KEY_ALIASES, label="Groq"
        )
    return _live_groq


# --------------------------------------------------------------------------- #
# env-file management (the platform Keys UI has no edit/delete — the app
# provides it for GEMINI_API_KEY by rewriting .env.local directly)
# --------------------------------------------------------------------------- #

# The env file the dashboard's key manager writes to. CLIPFORGE_ENV_LOCAL lets
# the test-suite point key-writing tests at a throwaway file so they can never
# delete a real user's key from the deployment's real env file.
ENV_LOCAL = Path(os.getenv("CLIPFORGE_ENV_LOCAL") or (ROOT / ".env.local"))


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
    return _sync_env_for_delete(GEMINI_KEY_ALIASES, "GEMINI_API_KEY")


def sync_groq_env_after_save(value: str) -> None:
    """Mirror a freshly written GROQ_API_KEY into the process env.

    Same shadowing hazard as the Gemini path: the live resolver prefers OS
    env, so a stale copy from an older .env.local would win over the new key.
    """
    os.environ["GROQ_API_KEY"] = value


def sync_groq_env_for_delete() -> list[str]:
    """Drop Groq key aliases from the process env before a file delete."""
    return _sync_env_for_delete(GROQ_KEY_ALIASES, "GROQ_API_KEY")


def _sync_env_for_delete(aliases: tuple[str, ...], primary: str) -> list[str]:
    _fingerprint, file_values = _fresh_env_snapshot(ROOT)
    dropped: list[str] = []
    for name in aliases:
        if name == primary or name in file_values:
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
    # auto | whisperx | faster-whisper. `auto` prefers WhisperX when its torch
    # stack is installed and otherwise uses faster-whisper's batched pipeline
    # (the same CTranslate2 engine WhisperX runs on).
    engine: str = "auto"
    # How many 30s windows the engine decodes in one batched pass.
    batch_size: int = 8

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
class GroqConfig:
    """Groq specifics for the chat-completions detection backend.

    Groq speaks the OpenAI wire format, so this is just an endpoint + model +
    the free tier's rate budget. `min_interval_sec` paces calls so a long
    transcript (a 2h video is ~60 chunks) stays under the 30 requests/minute
    free-tier ceiling instead of discovering it via 429s.
    """

    model: str = "openai/gpt-oss-120b"
    base_url: str = "https://api.groq.com/openai/v1"
    max_retries: int = 3
    timeout_sec: int = 120
    # 30 rpm -> 2.0s floor. Small headroom below the documented ceiling.
    min_interval_sec: float = 2.1
    temperature: float = 0.4


@dataclass
class LlmConfig:
    """Which backend runs the AI clip/highlight DETECTION stage.

    This is deliberately NOT part of the ASR engine choice: transcription is
    always faster-whisper (app/asr.py). Only the "which moments are viral"
    scoring step is pluggable, because that is the one step a user may want
    to run fully offline on their own hardware.

        groq   — hosted, default. OpenAI-compatible chat completions, ~30 rpm
                 on the free tier, paced client-side so 429s are rare.
        gemini — hosted alternative, kept as a selectable fallback for when
                 the Groq quota is exhausted.
        ollama — self-hosted. Free and private, but it shares the same CPU
                 as Whisper, so detection runs sequentially AFTER the
                 transcript exists (no overlap) and is markedly slower.
    """

    provider: str = "groq"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_sec: int = 600
    ollama_keep_alive: str = "30m"


@dataclass
class TranslateConfig:
    """Multi-language caption tracks (sidecar SRT/VTT).

    Translation reuses the [llm] provider, so there is no separate backend and
    no extra dependency. Whisper's own ``task="translate"`` is NOT used here:
    it targets English only, so it would cover just one of the target languages.
    """

    enabled: bool = False
    #: The language the transcript is actually in. This track is copied
    #: VERBATIM and never machine-translated — round-tripping the original
    #: through a translator would silently reword the source captions.
    source_lang: str = "en"
    #: Target language codes. The source language appears first.
    languages: list[str] = field(default_factory=lambda: ["en"])
    #: Brand/product/speaker terms that must survive translation verbatim.
    glossary: list[str] = field(default_factory=list)
    #: Cues per translation request. Batching carries dialogue context; keeping
    #: it bounded means one malformed reply costs one batch, not the whole job.
    batch_size: int = 30
    max_lines: int = 2
    #: Emit .srt, .vtt, or both.
    formats: list[str] = field(default_factory=lambda: ["srt", "vtt"])


@dataclass
class SubtitleConfig:
    font: str
    font_fallbacks: list[str] = field(default_factory=list)
    font_size: int = 68
    outline_color: str = "#39FF14"
    outline_width: int = 5
    words_per_cue: int = 3
    # Hard cap on a caption burst. A burst also ends early on a pause, so the
    # real range is 1..max_words.
    words_per_cue_max: int = 4
    # A gap of at least this long before the next word ends the burst. This is
    # what makes the captions land on the speech instead of every N words.
    pause_sec: float = 0.30
    # Remove unambiguous vocal fillers ("um", "uh", "hmm") and adjacent
    # stutters from the DISPLAYED captions. The audio is never modified.
    strip_fillers: bool = True
    strip_stutters: bool = True
    position_y: float = 0.78
    pop_animation: bool = True
    # Default caption style preset: bold_pop | karaoke | lower_third | boxed
    preset: str = "bold_pop"
    # --- unclear speech ------------------------------------------------
    # A caption cue whose mean per-word confidence is below `low_confidence`
    # is shown as [muttering] instead of the model's guess, and below
    # `inaudible_below` as [inaudible]. Both are 0..1; `low_confidence = 0`
    # turns the behaviour off entirely. Defaults are measured, not guessed --
    # see subtitle.LOW_CONFIDENCE for the measurements behind 0.45.
    low_confidence: float = 0.45
    inaudible_below: float = 0.30
    # Colour the word being spoken is painted in. Empty means "use the
    # preset's own accent", which is what keeps a preset change intact: the
    # Lower-third style ships a white accent, Bold pop a neon green, and
    # overriding both to the same colour would erase that difference.
    highlight_color: str = ""


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
    # How many jobs the in-process queue drains in parallel. Each job runs in
    # its own child process, so a hung job can never block the whole queue.
    worker_concurrency: int = 2


@dataclass
class SocialConfig:
    """OAuth app credentials + readiness for the "Post to..." connectors.

    Every platform needs its own developer app and its own approval process, and
    they are NOT symmetric -- see PLATFORM_NOTES. Credentials are read live from
    the environment (the platform's Environment vars UI) so a user can add a
    platform without a code change and without the process restarting.

    Nothing here is a secret we mint; these are the user's own registered apps.
    A missing client id simply means that platform reports `needs_app` and the UI
    falls back to download + share sheet, rather than the whole feature being
    blocked behind the slowest approval.
    """

    public_url: str = "http://127.0.0.1:5173"
    request_timeout_sec: int = 180
    # Chunk size for the resumable uploads X and TikTok require for video.
    upload_chunk_bytes: int = 4 * 1024 * 1024

    youtube_client_id: str = ""
    youtube_client_secret: str = ""
    x_client_id: str = ""
    x_client_secret: str = ""
    tiktok_client_key: str = ""
    tiktok_client_secret: str = ""
    meta_app_id: str = ""
    meta_app_secret: str = ""


@dataclass
class Settings:
    app: AppConfig
    whisper: WhisperConfig
    gemini: GeminiConfig
    groq: GroqConfig
    llm: LlmConfig
    translate: TranslateConfig
    subtitle: SubtitleConfig
    video: VideoConfig
    download: DownloadConfig
    hosted: HostedConfig
    social: SocialConfig

    @property
    def gemini_api_key(self) -> str | None:
        """LIVE Gemini key — re-resolved whenever the env files change.

        Never snapshot this at boot: the platform can deliver the key after
        the process has already started.
        """
        state = live_gemini_key()
        state.refresh()
        return state.key

    @property
    def groq_api_key(self) -> str | None:
        """LIVE Groq key — re-resolved whenever the env files change."""
        state = live_groq_key()
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


def _hex_or_blank(value: object, name: str, errors: list[str]) -> str:
    """A hex colour, or "" meaning 'use the preset default'.

    The empty case is the common one: most installs never set a highlight
    colour, and demanding one would turn a blank config line into a boot
    failure.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    if not text.startswith("#") or len(text) not in (4, 7):
        errors.append(f'{name} "{text}" must be a hex colour like #39FF14, or left empty')
        return ""
    return text


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
    engine = str(os.getenv("CLIPFORGE_WHISPER_ENGINE") or w.get("engine", "auto")).strip().lower()
    if engine not in {"auto", "whisperx", "faster-whisper"}:
        errors.append(
            f'whisper.engine "{engine}" is not one of auto/whisperx/faster-whisper'
        )
    try:
        batch_size = int(os.getenv("CLIPFORGE_WHISPER_BATCH_SIZE") or w.get("batch_size", 8))
    except (TypeError, ValueError):
        errors.append("whisper.batch_size must be an integer")
        batch_size = 8
    if not (1 <= batch_size <= 64):
        errors.append(f"whisper.batch_size {batch_size} is out of range (1-64)")
    whisper = WhisperConfig(
        model=model,
        device=str(w.get("device", "auto")),
        compute_type=str(w.get("compute_type", "auto")),
        language=str(w.get("language", "auto")),
        engine=engine,
        batch_size=batch_size,
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

    gq = raw.get("groq", {})
    groq = GroqConfig(
        model=str(os.getenv("GROQ_MODEL") or gq.get("model", "openai/gpt-oss-120b")),
        base_url=str(
            os.getenv("GROQ_BASE_URL") or gq.get("base_url", "https://api.groq.com/openai/v1")
        ).strip().rstrip("/"),
        max_retries=max(0, int(gq.get("max_retries", 3))),
        timeout_sec=max(15, int(gq.get("timeout_sec", 120))),
        min_interval_sec=max(0.0, float(gq.get("min_interval_sec", 2.1))),
        temperature=float(gq.get("temperature", 0.4)),
    )

    lc = raw.get("llm", {})
    provider = str(
        os.getenv("CLIPFORGE_LLM_PROVIDER") or lc.get("provider", "groq")
    ).strip().lower()
    if provider not in {"groq", "gemini", "ollama"}:
        errors.append(
            f'llm.provider "{provider}" is not one of groq/gemini/ollama'
        )
        provider = "groq"
    sc = raw.get("social", {})

    llm = LlmConfig(
        provider=provider,
        ollama_url=str(
            os.getenv("OLLAMA_URL") or lc.get("ollama_url", "http://127.0.0.1:11434")
        ).strip().rstrip("/"),
        ollama_model=str(os.getenv("OLLAMA_MODEL") or lc.get("ollama_model", "llama3.1:8b")),
        ollama_timeout_sec=max(30, int(lc.get("ollama_timeout_sec", 600))),
        ollama_keep_alive=str(lc.get("ollama_keep_alive", "30m")),
    )

    tr = raw.get("translate", {})
    wanted = [c for c in _as_list(tr.get("languages", []))]
    if not wanted:
        wanted = ["en"]
    normalized = translate_mod.normalize_codes(wanted)
    unknown = [c for c in wanted if str(c).strip().lower() not in translate_mod.LANGS]
    if unknown:
        errors.append(
            f"translate.languages contains unsupported code(s): {', '.join(map(str, unknown))}"
        )
    formats = [str(f).strip().lower() for f in _as_list(tr.get("formats", ["srt", "vtt"]))]
    formats = [f for f in formats if f in {"srt", "vtt"}] or ["srt"]
    language_hint = str(w.get("language", "auto")).strip().lower()
    source_lang = str(tr.get("source_lang", "") or "").strip().lower()
    if not source_lang or source_lang not in translate_mod.LANGS:
        # Fall back to the transcription language hint, and only then to en.
        source_lang = language_hint if language_hint in translate_mod.LANGS else "en"
    translate = TranslateConfig(
        enabled=bool(tr.get("enabled", False)) or str(
            os.getenv("CLIPFORGE_TRANSLATE", "")
        ).strip().lower() in {"1", "true", "yes"},
        source_lang=source_lang,
        languages=normalized or ["en"],
        glossary=[str(t).strip() for t in _as_list(tr.get("glossary", [])) if str(t).strip()],
        batch_size=max(1, min(200, int(tr.get("batch_size", 30)))),
        max_lines=max(1, min(4, int(tr.get("max_lines", 2)))),
        formats=formats,
    )

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
        words_per_cue=max(1, int(s.get("words_per_cue", 3))),
        words_per_cue_max=max(1, int(s.get("words_per_cue_max", 4))),
        pause_sec=min(3.0, max(0.0, float(s.get("pause_sec", 0.30)))),
        position_y=min(0.95, max(0.3, float(s.get("position_y", 0.78)))),
        pop_animation=bool(s.get("pop_animation", True)),
        preset=str(s.get("preset", "bold_pop")),
        low_confidence=float(s.get("low_confidence", 0.45)),
        inaudible_below=float(s.get("inaudible_below", 0.30)),
        strip_fillers=bool(s.get("strip_fillers", True)),
        strip_stutters=bool(s.get("strip_stutters", True)),
        highlight_color=_hex_or_blank(s.get("highlight_color", ""), "subtitle.highlight_color", errors),
    )
    if subtitle.preset not in {"bold_pop", "karaoke", "lower_third", "boxed"}:
        errors.append(
            f'subtitle.preset "{subtitle.preset}" is not one of '
            "bold_pop/karaoke/lower_third/boxed"
        )
    for name, value in (("low_confidence", subtitle.low_confidence),
                        ("inaudible_below", subtitle.inaudible_below)):
        if not 0.0 <= value <= 1.0:
            # A confidence is a probability. Silently clamping `low_confidence
            # = 5` to 1.0 would leave it meaning "flag absolutely everything"
            # and, paired with a clamped inaudible tier, produce a config
            # error about the ORDERING of two numbers the user never intended.
            # Saying "this must be 0..1" is the useful message.
            errors.append(
                f"subtitle.{name} must be between 0 and 1 (got {value}) — "
                "it is a confidence, not a percentage"
            )
    if subtitle.words_per_cue > subtitle.words_per_cue_max:
        # Otherwise `max_words` is dead config and the real cap is the target.
        errors.append(
            f"subtitle.words_per_cue ({subtitle.words_per_cue}) cannot be greater than "
            f"subtitle.words_per_cue_max ({subtitle.words_per_cue_max})"
        )
    if subtitle.low_confidence > 0 and subtitle.inaudible_below >= subtitle.low_confidence:
        # Caught here rather than at render time: a mis-ordered pair silently
        # disables the [inaudible] tier (nothing can be below a lower bar than
        # the flag itself), and the user would have no idea why. Skipped when
        # low_confidence is 0 — that is the documented "off" switch, and
        # inaudible_below is then irrelevant, so it must not block it.
        errors.append(
            f"subtitle.inaudible_below ({subtitle.inaudible_below}) must be BELOW "
            f"subtitle.low_confidence ({subtitle.low_confidence})"
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
        worker_concurrency=max(
            1,
            min(
                8,
                int(os.getenv("CLIPFORGE_WORKER_CONCURRENCY") or h.get("worker_concurrency", 2)),
            ),
        ),
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
        groq=groq,
        llm=llm,
        translate=translate,
        subtitle=subtitle,
        video=video,
        download=download,
        hosted=hosted,
        social=SocialConfig(
            public_url=str(
                os.getenv("CLIPFORGE_PUBLIC_URL")
                or sc.get("public_url")
                or "http://127.0.0.1:5173"
            ).rstrip("/"),
            request_timeout_sec=max(
                30, int(os.getenv("CLIPFORGE_SOCIAL_TIMEOUT") or sc.get("request_timeout_sec", 180))
            ),
            upload_chunk_bytes=max(
                262144,
                int(os.getenv("CLIPFORGE_UPLOAD_CHUNK_BYTES") or sc.get("upload_chunk_bytes", 4 * 1024 * 1024)),
            ),
            youtube_client_id=(os.getenv("YOUTUBE_CLIENT_ID") or "").strip(),
            youtube_client_secret=(os.getenv("YOUTUBE_CLIENT_SECRET") or "").strip(),
            x_client_id=(os.getenv("X_CLIENT_ID") or os.getenv("TWITTER_CLIENT_ID") or "").strip(),
            x_client_secret=(os.getenv("X_CLIENT_SECRET") or os.getenv("TWITTER_CLIENT_SECRET") or "").strip(),
            tiktok_client_key=(os.getenv("TIKTOK_CLIENT_KEY") or "").strip(),
            tiktok_client_secret=(os.getenv("TIKTOK_CLIENT_SECRET") or "").strip(),
            meta_app_id=(os.getenv("META_APP_ID") or os.getenv("FACEBOOK_APP_ID") or "").strip(),
            meta_app_secret=(os.getenv("META_APP_SECRET") or os.getenv("FACEBOOK_APP_SECRET") or "").strip(),
        ),
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
    settings = load_config(quiet=True)
    active = settings.llm.provider
    # Report the key for the ACTIVE backend. Groq is the default, so printing
    # only Gemini made a fully-working deployment look unconfigured.
    if active == "groq":
        live: _LiveKeyState = live_groq_key()
        env_name = "GROQ_API_KEY"
    else:
        live = live_gemini_key()
        env_name = "GEMINI_API_KEY"
    live.refresh(force=True)
    key_loaded = live.key is not None
    print(f"[boot] LLM: {active} | key loaded: {'YES' if key_loaded else 'NO'}")
    print(f"[boot] {active.capitalize()} key source: {live.source}")
    if not key_loaded:
        if live.problem:
            print(f"[boot] {active.capitalize()} key problem: {live.problem}")
        print(
            f"[boot] AI detection offline — add {env_name} in the Keys/API keys tab "
            "(or .env), then hit Re-check on the dashboard"
        )
    env_files = _load_env_files(ROOT)
    print(f"[boot] env files loaded: {', '.join(env_files) if env_files else 'none found'}")
    from app import asr as asr_mod  # local import: avoids an import cycle at boot
    from app import hostlimits

    engine = asr_mod.status(settings)
    model_line = f"[boot] Whisper model: {engine.model}"
    if engine.model != settings.whisper.model:
        model_line += f" (configured '{settings.whisper.model}' — does not fit this host)"
    print(
        f"{model_line} | engine: {engine.engine} "
        f"({engine.device}/{engine.compute_type}, batch {engine.batch_size}, batched)"
    )
    print(f"[boot] ASR engine detail: {engine.detail}")
    print(f"[boot] Host limits: {hostlimits.describe()}")
    if engine.model_note:
        print(f"[boot] WARNING: {engine.model_note}")
    print(f"[boot] Output: {settings.video.width}x{settings.video.height} @ {settings.video.fps}fps CRF {settings.video.crf}")
    print(f"[boot] Folders: {settings.app.paths.downloads_dir} / {settings.app.paths.outputs_dir} / {settings.app.paths.data_dir}")
    print(f"[boot] Mode: {'tailscale-only' if settings.hosted.tailscale_only else 'public'} | max source: {settings.hosted.max_source_min} min | retention: {settings.hosted.retention_hours}h")
    print(f"[boot] Worker pool: {settings.hosted.worker_concurrency} parallel job(s) (one child process each)")
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
