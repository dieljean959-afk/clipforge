"""Central configuration for the hosted Clipmake service.

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
from dataclasses import dataclass, field, replace
from pathlib import Path

from loguru import logger

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 dev/test environments
    import tomli as tomllib  # type: ignore[no-redef]

from dotenv import load_dotenv

from app import clipcount as clipcount_mod  # pure math, no cycle
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

# NVIDIA NIM (integrate.api.nvidia.com) hosts DeepSeek models, but with
# NVIDIA's own keys (`nvapi-…`). A DeepSeek-platform key (`sk-…`, from
# platform.deepseek.com) authenticates against DeepSeek's OWN API instead, so
# the two are separate providers with separate keys — hence separate sections
# below. Both speak the same OpenAI chat-completions dialect, so both reuse the
# one hosted call path.
NVIDIA_KEY_ALIASES = (
    "NVIDIA_API_KEY",
    "NVIDIA_KEY",
    "NVIDIA_NIM_API_KEY",
    "NGC_API_KEY",
)

# DeepSeek's own API (api.deepseek.com). Keys are `sk-…`.
DEEPSEEK_KEY_ALIASES = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_KEY",
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


def _resolve_nvidia_key(environ: dict[str, str]) -> tuple[str | None, str, str | None]:
    """Find the NVIDIA NIM key under any supported name.

    Same (key, source, problem) contract as the Gemini/Groq resolvers. NIM
    keys are long `nvapi-…` strings; the prefix is only a warning, never a
    rejection, because the shape is not contractual and a stricter check would
    lock out a future key format.
    """
    for name in NVIDIA_KEY_ALIASES:
        value = (environ.get(name) or "").strip()
        if not value:
            continue
        if value.startswith(_ENC_PREFIX):
            if _empty_encrypted(value):
                return None, f"{name} (empty)", (
                    f"{name} is an EMPTY save — the platform stored an encrypted envelope "
                    "with no key inside. Paste the full key (starts with nvapi-) in the "
                    "Keys/API keys tab again, then hit Re-check."
                )
            return None, f"{name} (encrypted)", (
                f"{name} is stored in the platform's encrypted dotenvx format and could "
                "not be decrypted inside the app — re-save it in the Keys/API keys tab, "
                "then hit Re-check."
            )
        if value.lower() in _PLACEHOLDER_VALUES:
            return None, f"{name} (placeholder)", (
                f"{name} still contains placeholder text — paste the real NVIDIA API key."
            )
        if " " in value:
            return None, f"{name} (malformed)", (
                f"{name} contains spaces — it should be the raw key with no quotes or spaces."
            )
        if not value.lower().startswith("nvapi-") and len(value) < 20:
            return value, f"{name} (short — verify)", (
                f"{name} is only {len(value)} characters and does not start with nvapi- — "
                "double-check the paste from build.nvidia.com."
            )
        return value, name, None
    return None, "", None


def _resolve_deepseek_key(environ: dict[str, str]) -> tuple[str | None, str, str | None]:
    """Find the DeepSeek-platform key under any supported name.

    Same (key, source, problem) contract as the other resolvers. DeepSeek keys
    are `sk-…`; the prefix is a warning, never a rejection, so a future key
    format cannot lock the user out of their own provider.
    """
    for name in DEEPSEEK_KEY_ALIASES:
        value = (environ.get(name) or "").strip()
        if not value:
            continue
        if value.startswith(_ENC_PREFIX):
            if _empty_encrypted(value):
                return None, f"{name} (empty)", (
                    f"{name} is an EMPTY save — the platform stored an encrypted envelope "
                    "with no key inside. Paste the full key (starts with sk-) in the "
                    "Keys/API keys tab again, then hit Re-check."
                )
            return None, f"{name} (encrypted)", (
                f"{name} is stored in the platform's encrypted dotenvx format and could "
                "not be decrypted inside the app — re-save it in the Keys/API keys tab, "
                "then hit Re-check."
            )
        if value.lower() in _PLACEHOLDER_VALUES:
            return None, f"{name} (placeholder)", (
                f"{name} still contains placeholder text — paste the real DeepSeek API key."
            )
        if " " in value:
            return None, f"{name} (malformed)", (
                f"{name} contains spaces — it should be the raw key with no quotes or spaces."
            )
        if not value.lower().startswith("sk-") and len(value) < 20:
            return value, f"{name} (short — verify)", (
                f"{name} is only {len(value)} characters and does not start with sk- — "
                "double-check the paste from platform.deepseek.com/api_keys."
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
_live_nvidia: _LiveKeyState | None = None
_live_deepseek: _LiveKeyState | None = None


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


def live_nvidia_key() -> _LiveKeyState:
    """Process-wide live NVIDIA-key state (created on first use)."""
    global _live_nvidia
    if _live_nvidia is None:
        _live_nvidia = _LiveKeyState(
            ROOT,
            resolver=_resolve_nvidia_key,
            aliases=NVIDIA_KEY_ALIASES,
            label="NVIDIA",
        )
    return _live_nvidia


def live_deepseek_key() -> _LiveKeyState:
    """Process-wide live DeepSeek-key state (created on first use)."""
    global _live_deepseek
    if _live_deepseek is None:
        _live_deepseek = _LiveKeyState(
            ROOT,
            resolver=_resolve_deepseek_key,
            aliases=DEEPSEEK_KEY_ALIASES,
            label="DeepSeek",
        )
    return _live_deepseek


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


def sync_nvidia_env_after_save(value: str) -> None:
    """Mirror a freshly written NVIDIA_API_KEY into the process env."""
    os.environ["NVIDIA_API_KEY"] = value


def sync_nvidia_env_for_delete() -> list[str]:
    """Drop NVIDIA key aliases from the process env before a file delete."""
    return _sync_env_for_delete(NVIDIA_KEY_ALIASES, "NVIDIA_API_KEY")


def sync_deepseek_env_after_save(value: str) -> None:
    """Mirror a freshly written DEEPSEEK_API_KEY into the process env."""
    os.environ["DEEPSEEK_API_KEY"] = value


def sync_deepseek_env_for_delete() -> list[str]:
    """Drop DeepSeek key aliases from the process env before a file delete."""
    return _sync_env_for_delete(DEEPSEEK_KEY_ALIASES, "DEEPSEEK_API_KEY")


#: The four values objectstore.ObjectStoreConfig.enabled requires before it will
#: offload anything. Named here (rather than inlined in the endpoint) because the
#: endpoint, the delete path and the tests all have to agree on the exact set —
#: a name that is off by one produces a config that reports enabled=False with
#: no clue which field is missing.
S3_CREDENTIAL_VARS: tuple[str, ...] = (
    "CLIPFORGE_S3_ENDPOINT",
    "CLIPFORGE_S3_BUCKET",
    "CLIPFORGE_S3_ACCESS_KEY",
    "CLIPFORGE_S3_SECRET_KEY",
)

#: Download-worker token persistence. Stored in a server-only file outside any
#: public/build/zip tree, with 0600 permissions, so the dashboard can persist
#: WORKER_URL + WORKER_TOKEN across restarts without touching env or logs.
DOWNLOAD_WORKER_TOKEN_FILE = ROOT / "data" / "download-worker-token.json"

#: Non-secret store settings. Saved alongside the credentials so a bucket is
#: described in one place, but kept out of S3_CREDENTIAL_VARS because deleting
#: the credentials should not silently reset the operator's chosen prefix.
S3_OPTIONAL_VARS: tuple[str, ...] = ("CLIPFORGE_S3_REGION", "CLIPFORGE_S3_PREFIX")


def sync_s3_env_after_save(values: dict[str, str]) -> list[str]:
    """Mirror freshly written CLIPFORGE_S3_* values into the process env.

    Same reason as every other sync_* helper: objectstore.config_from_env()
    reads os.getenv, and a value written to .env.local alone would not be seen
    by the already-running process until a restart — which is exactly the
    "I saved it and the card still says offload is off" failure. Only the
    names actually written are touched, so a platform-injected value for a
    field the operator left blank is left alone.
    """
    for name, value in values.items():
        os.environ[name] = value
    return sorted(values)


def sync_s3_env_for_delete(names: tuple[str, ...]) -> list[str]:
    """Drop the named CLIPFORGE_S3_* values from the process env."""
    dropped: list[str] = []
    for name in names:
        if os.environ.pop(name, None) is not None:
            dropped.append(name)
    return dropped


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


def default_storage_root(root: Path) -> Path:
    """Where uploads/sources/transcripts/clips live — OUTSIDE the codebase.

    Downloads, outputs and the job store used to sit in `<codebase>/downloads`,
    `<codebase>/outputs` and `<codebase>/data`. That put ~1.6 GB of media
    inside the directory tree the Vite dev server watches, and inside the tree
    the platform syncs back to the editor. Every byte written by a job was a
    file-system event in the watched root, and a restarted job lost its source
    because the source lived somewhere the platform is free to prune.

    The sibling directory `<root>/../clipforge-data` is on the same filesystem
    (so relocation is a rename, not a copy) but is outside the watched project.
    """
    override = os.getenv("CLIPFORGE_STORAGE_DIR")
    if override:
        return Path(override).expanduser()
    return root.parent / "clipforge-data"


def _relocate_to_storage(root: Path, storage: Path, name: str) -> Path:
    """Move `<root>/<name>` under `storage`, leaving a symlink behind.

    The symlink is deliberate: hundreds of absolute paths are already baked
    into `data/work/<id>/*/meta.json` and into clip documents from jobs created
    before the move. Replacing the directory with a symlink keeps every one of
    those paths valid while the bytes now live outside the watched codebase.

    Never raises — a failed relocation just keeps using the old location, which
    is exactly the behaviour we had before.

    NEVER deletes anything, and NEVER moves files one at a time. Two earlier
    versions did, and both damaged real data:

      * `shutil.rmtree(old)` for the "both exist" case destroyed 1.3 GB of job
        sources when the host had restored the codebase tree without the
        external one.
      * A per-child merge then SPLIT a live tree: it moved what it could out of
        `codebase/outputs`, could not `rmdir` the rest, and returned without
        creating the symlink — so half a batch's rendered clips sat at one path
        and half at another, and the absolute paths already stored in clip
        documents stopped resolving.

    The relocation is therefore all-or-nothing on a single `rename`: either the
    whole directory moves and the symlink appears, or nothing moves at all.

    It only runs when `storage` is on the SAME filesystem as the codebase. On
    the hosted sandbox they are not — the project directory is its own mount —
    and there a move is a multi-hundred-megabyte copy with no atomicity at all,
    which is a worse trade than leaving the tree alone. In that case the
    function says so and returns the original path; keeping the media out of
    the Vite watch set is handled by `server.watch.ignored` regardless.
    """
    old = root / name
    new = storage / name
    try:
        storage.mkdir(parents=True, exist_ok=True)
        if old.is_symlink():
            # already relocated (or an operator symlinked it) — just trust it
            return new if old.resolve() == new.resolve() else old
        if old.exists() and old.stat().st_dev != storage.stat().st_dev:
            logger.info(
                f"[config] {name} stays at {old} — {storage} is on a different "
                f"filesystem, so moving it would be a non-atomic copy"
            )
            return old
        if not old.exists():
            new.mkdir(parents=True, exist_ok=True)
            old.symlink_to(new, target_is_directory=True)
            return new
        if old.resolve() == new.resolve():
            return new
        if new.exists():
            # Both trees exist and they are not merged: a running job may be
            # writing into either one, and a piecemeal move is exactly what
            # split `outputs` earlier. `old` stays authoritative — it is the
            # path every stored document already points at — and the operator
            # is told, rather than discovering it later as "missing clips".
            logger.warning(
                f"[config] both {old} and {new} exist — keeping {old} in place. "
                f"Move the contents of {new} into {old} (or delete {new}) and "
                f"restart to finish relocating media to {storage}."
            )
            return old
        os.rename(old, new)
        try:
            old.symlink_to(new, target_is_directory=True)
        except OSError:
            # Undo rather than leave the tree reachable at only one of the two
            # paths: data moved with no symlink IS the split.
            os.rename(new, old)
            logger.warning(f"[config] could not symlink {old} — left in place")
            return old
        return new
    except (OSError, shutil.Error):
        try:
            old.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return old


@dataclass
class Paths:
    downloads_dir: Path
    outputs_dir: Path
    data_dir: Path
    static_dir: Path
    storage_dir: Path | None = None
    #: Where an UPLOADED file is stored. Deliberately inside the codebase
    #: (`data/uploads/`), NOT under `downloads_dir`.
    #:
    #: `downloads_dir` is symlinked to `<codebase>/../clipforge-data`, which
    #: exists to keep media out of the Vite watch set. Measured on this host,
    #: that is exactly the directory that does NOT survive a redeploy: after a
    #: restore, `outputs/` and `data/` (real directories in the codebase) were
    #: intact while the media-storage `downloads/` tree had been emptied,
    #: so 11 of 18 job records pointed at a `downloads/{id}/` folder that no
    #: longer existed and every one of those jobs died at Rendering with
    #: "Source file was lost".
    #:
    #: An upload is the one artefact there is no way to recreate — a URL job can
    #: always be re-downloaded, an uploaded file cannot. So uploads live where
    #: the platform demonstrably restores bytes. `downloads_dir` stays as a
    #: FALLBACK read path, so every job created before this change (and any job
    #: whose source is still there) keeps working untouched.
    uploads_dir: Path | None = None

    def job_source_dirs(self, job_id: str) -> tuple[Path, ...]:
        """Every directory a job's source file may legitimately live in.

        Newest location first. Callers read from ALL of them and write to
        ``uploads_dir`` only — never to a fallback, because writing to two
        places is how the two trees drifted apart before.
        """
        dirs: list[Path] = []
        # `None` means "not configured", not "uploads do not exist" — the
        # default is always `data_dir/uploads`, so a hand-built Paths resolves
        # to the same place load_config does instead of silently losing the
        # primary root.
        uploads = self.uploads_dir or (self.data_dir / "uploads")
        dirs.append(uploads / job_id)
        dirs.append(self.downloads_dir / job_id)
        return tuple(dirs)


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
    # Ceiling for the duration-scaled clip count, and the score below which a
    # candidate is not a highlight. `auto_clip_count` scales the target with the
    # source length (see app/clipcount.py) instead of always asking for the
    # same number; `min_score` is what stops a long source from being padded
    # out to its ceiling with the leftovers.
    auto_clip_count: bool = True
    min_score: float = 60.0


@dataclass
class GroqConfig:
    """Groq specifics for the chat-completions detection backend.

    Groq speaks the OpenAI wire format, so this is just an endpoint + model +
    the free tier's rate budget. `min_interval_sec` paces calls so a long
    transcript (a 2h video is ~20 bounded chunks) stays under the 30
    requests/minute free-tier ceiling instead of discovering it via 429s.
    """

    model: str = "openai/gpt-oss-120b"
    base_url: str = "https://api.groq.com/openai/v1"
    max_retries: int = 3
    timeout_sec: int = 120
    # 30 rpm -> 2.0s floor. Small headroom below the documented ceiling.
    min_interval_sec: float = 2.1
    # 0.0, not 0.4. Detection is an EXTRACTION task: the model is copying word
    # indices out of a transcript, not writing prose. There is no creativity to
    # sample, and every degree of temperature is a chance to emit a malformed
    # or truncated JSON body — which is exactly how this stage used to fail.
    temperature: float = 0.0
    # `json_schema` is the only mode that resolves the original bug. The prompt
    # asks for a top-level ARRAY; `json_object` requires a top-level OBJECT, so
    # the model had two contradictory instructions and failed validation
    # non-deterministically (reproduced live: same config, 200 and 400 on
    # consecutive calls). Strict json_schema uses constrained decoding, so the
    # shape is guaranteed by the decoder and cannot be violated.
    #   "json_object" — weaker, kept selectable for models without schema support
    #   "plain"       — no response_format; the reply is repaired locally
    response_format: str = "json_schema"
    # Output budget per detection call. The observed failure mode is truncation
    # at the token ceiling, which is why this is raised on each JSON retry.
    max_tokens: int = 3200
    #: Hard ceiling for the retry ladder, never exceeded however long a chunk
    #: takes to produce valid JSON.
    max_tokens_ceiling: int = 8192
    #: Retries for a JSON-validation failure, each at lower temperature and a
    #: higher max_tokens. Capped at 3 because a model that cannot emit valid
    #: JSON for this chunk will not manage it on the fourth attempt, and each
    #: attempt costs real tokens against a daily quota.
    json_retries: int = 3
    #: Temperature CEILING for the ladder. Each JSON retry uses
    #: min(temperature, json_retry_temperature), so it can only go DOWN — with
    #: the 0.0 default above, retries are deterministic as well.
    json_retry_temperature: float = 0.2
    #: Tried in order when the primary model cannot produce valid JSON for a
    #: chunk even after the ladder. All three support strict json_schema per
    #: Groq's docs. Kept out of the code so the chain is a config decision.
    fallback_models: tuple[str, ...] = ("openai/gpt-oss-20b", "qwen/qwen3.8-27b")
    # Self-imposed share of the account's tokens-per-minute allowance. The
    # client paces itself to this so a long source does not walk into a 429
    # part way through detection — and, more to the point, does not walk into
    # it having already spent most of a DAILY token budget. 0 disables the
    # token pacer and leaves only the per-request floor, which is the right
    # setting for a key on a much larger tier.
    tokens_per_min: int = 6000
    # The free tier's tokens-per-DAY ceiling, shown by the dashboard badge and
    # used as the denominator when Groq has not told us the real limit. A 429
    # overwrites this for the day with the provider's own number, which is the
    # only authoritative one — Groq never reports the daily balance in a header.
    daily_token_limit: int = 200000
    #: Provider name used in log lines and error text, so one shared call path
    #: can say "Groq" or "NVIDIA" without hardcoding either.
    #: Stable provider identity ("groq"). NOT the display label: telemetry,
    #: pacing ledgers and usage buckets are keyed by this, because a label is
    #: prose that someone can reword. Inferring identity from `label.lower()`
    #: happens to work today and silently misfiles spend the day it does not.
    provider: str = "groq"
    label: str = "Groq"
    #: Where an operator goes to pick a current model after a 404. Named in the
    #: error rather than guessed at: a retired model has no safe auto-replacement.
    docs_url: str = "console.groq.com/docs/models"


@dataclass
class NvidiaConfig:
    """NVIDIA NIM (integrate.api.nvidia.com) — an ALTERNATIVE detection backend.

    NIM is OpenAI-compatible, so these fields are deliberately the SAME names
    and with the SAME meanings as GroqConfig: one call path serves both
    providers and only the endpoint, model and rate budget differ. Do not fork
    the names here — `app.llm` passes this object straight into the
    Groq-shaped ladder.

    Where the defaults differ from Groq, and why:

      * ``min_interval_sec`` is 0.0, not 2.1. Groq's free tier caps at ~30
        requests/minute, so pacing there is mandatory; NIM's limits are far
        higher and pacing a 20-chunk run at 2.1s would cost 40s for nothing.
        0.0 disables the request floor while leaving the token pacer on.
      * ``response_format`` is ``json_object``, not ``json_schema``. Strict
        Structured Outputs are a Groq feature; NIM's compatibility layer does
        not guarantee them, and asking for a mode the model cannot do earns a
        permanent 400 rather than valid JSON. The ladder detects that and drops
        the field, so ``json_schema`` is still selectable if NIM grows support.
      * ``tokens_per_min`` is generous. NIM rate-limits per model and per
        account in requests/minute, not a small daily token wall like Groq's
        200k TPD, so the self-imposed cap here is about bursts, not survival.
      * ``daily_token_limit`` is only the DENOMINATOR the usage badge divides
        by. NIM reports credits rather than a per-model daily token ceiling, so
        this is an operator-set figure, not a provider-stated one — and unlike
        Groq there is no 429 body that overrides it with the truth.
    """

    model: str = "deepseek-ai/deepseek-v4.1-flash"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    max_retries: int = 3
    timeout_sec: int = 120
    min_interval_sec: float = 0.0
    temperature: float = 0.0
    response_format: str = "json_object"
    max_tokens: int = 3200
    max_tokens_ceiling: int = 8192
    json_retries: int = 3
    json_retry_temperature: float = 0.2
    #: Tried in order when the primary model cannot produce valid JSON for a
    #: chunk even after the ladder. Left empty by default: NIM's catalog rotates
    #: and a wrong name here would add a hard 404 to every chunk's ladder. Fill
    #: it in from build.nvidia.com once you know the model you want second.
    fallback_models: tuple[str, ...] = ()
    tokens_per_min: int = 120000
    daily_token_limit: int = 2000000
    #: Provider name used in log lines and error text, so a single call path can
    #: say "NVIDIA" or "Groq" without hardcoding either.
    provider: str = "nvidia"
    label: str = "NVIDIA"
    #: Where an operator goes to pick a current model after a 404.
    docs_url: str = "build.nvidia.com/models"


@dataclass
class DeepSeekConfig:
    """DeepSeek's OWN API (api.deepseek.com) — the PRIMARY detection backend.

    Distinct from [nvidia] even though both serve DeepSeek MODELS: NVIDIA NIM
    is NVIDIA's hosted copy authenticated with an `nvapi-` key, while this is
    DeepSeek's own service authenticated with an `sk-` key. A key from one does
    not work against the other, which is why they are separate sections rather
    than one section with a swappable URL.

    Two defaults here are not cosmetic — they come from DeepSeek's own docs and
    both would otherwise silently weaken this stage:

      * ``thinking_mode = "disabled"``. DeepSeek enables thinking mode BY
        DEFAULT at high effort. Thinking mode returns a chain of thought in
        `reasoning_content` before the answer: pure waste for a task that is
        copying word indices out of a transcript, billed as output tokens.
      * Because of that, ``temperature`` MUST be sent with thinking OFF. The
        docs are explicit that thinking mode ignores `temperature` —
        "will not trigger an error but will also have no effect". So a
        temperature-0.0 request against a default (thinking) DeepSeek is a
        no-op that still LOOKS deterministic, and the extraction quietly goes
        non-deterministic — which is precisely the failure mode the JSON retry
        ladder exists to absorb, at full price.

    `max_tokens` stays modest (3200) for the same reason: CoT would eat that
    budget and truncate the JSON, which is the most common real cause of a
    malformed reply.
    """

    model: str = "deepseek-flash"
    #: The documented OpenAI-format base. NOT `/v1` — our transport appends
    #: `/chat/completions` itself, and the docs name this exact root.
    base_url: str = "https://api.deepseek.com"
    max_retries: int = 3
    timeout_sec: int = 120
    min_interval_sec: float = 0.0
    temperature: float = 0.0
    response_format: str = "json_object"
    max_tokens: int = 3200
    max_tokens_ceiling: int = 8192
    json_retries: int = 3
    json_retry_temperature: float = 0.2
    #: Empty by default: DeepSeek publishes two current model names and rotates
    #: them, so a stale hard-coded second entry would 404 on every chunk.
    fallback_models: tuple[str, ...] = ()
    #: DeepSeek documents a CONCURRENCY ceiling (not a token-per-minute one),
    #: so this is a burst guard only. Detection is sequential anyway.
    tokens_per_min: int = 120000
    #: Only the denominator the usage badge divides by. DeepSeek bills from a
    #: topped-up balance rather than publishing a daily token ceiling, and there
    #: is no 429 body carrying one, so this is an operator-set soft budget that
    #: drives the pre-submit warning — not a limit the provider enforces.
    daily_token_limit: int = 2000000
    #: "disabled" (recommended) or "enabled". Sent as the OpenAI-compat
    #: `thinking: {type: ...}` control parameter.
    thinking_mode: str = "disabled"
    #: Provider name for logs/errors, so one call path can say "DeepSeek".
    provider: str = "deepseek"
    label: str = "DeepSeek"
    #: Where an operator goes to pick a current model after a 404.
    docs_url: str = "api-docs.deepseek.com/quick_start/pricing"


@dataclass
class LlmConfig:
    """Which backend runs the AI clip/highlight DETECTION stage.

    This is deliberately NOT part of the ASR engine choice: transcription is
    always faster-whisper (app/asr.py). Only the "which moments are viral"
    scoring step is pluggable, because that is the one step a user may want
    to run fully offline on their own hardware.

        nvidia  — hosted, default. NVIDIA NIM, OpenAI-compatible chat
                  completions. The primary because Groq's free tier enforces a
                  hard tokens-per-DAY ceiling per model, and a long source
                  exhausts it part way through detection.
        groq    — hosted backup. Same wire format, same code path; ~30 rpm on
                  the free tier, paced client-side so 429s are rare.
        gemini  — hosted alternative, selectable without a redeploy.
        ollama  — self-hosted. Free and private, but it shares the same CPU
                  as Whisper, so detection runs sequentially AFTER the
                  transcript exists (no overlap) and is markedly slower.

    `fallback_provider` names a second backend to try for a chunk when the
    primary FAILS (quota spent, endpoint down, key rejected). It is opt-in and
    off by default, because silently scoring clips with a model the operator
    did not pick is worse than a loud failure. When it is set, the swap is
    logged per chunk so the provenance is never a surprise.
    """

    provider: str = "deepseek"
    #: "" disables automatic failover. Otherwise one of PROVIDERS.
    fallback_provider: str = ""
    #: How many detection chunks may be in flight at once. 1 = strictly
    #: sequential. See load_config for why overlapping is safe here.
    detect_concurrency: int = 4
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
    #: Path to a Netscape-format cookies.txt, for sites that refuse anonymous
    #: requests from a datacenter IP (YouTube's "Sign in to confirm you're not
    #: a bot" is the common one). Empty means no cookie file.
    cookies_file: str = ""
    #: HTTP/SOCKS proxy used for yt-dlp metadata AND media download.
    #: Set YTDLP_PROXY at runtime or persist in config.toml.
    proxy: str = ""
    #: Comma-separated secondary proxies rotated per attempt (optional).
    proxy_list: str = ""


@dataclass
class HostedConfig:
    """Hosted-mode limits and remote-worker dispatch settings."""

    # Plan guard: reject sources longer than this at SUBMIT time. Long sources
    # ARE supported — processing time scales proportionally, see [hosted] notes.
    max_source_min: int = 120
    # Largest accepted upload in MB. 0 = no limit. Checked at submit time.
    max_source_mb: int = 4096
    # Delete finished artifacts (outputs/ + downloads/) this long after a job
    # completes — hosted storage is finite and users expect cleanup.
    #
    # FOUR DAYS, not "never". The sandbox disk is 4 GB and a single 15-clip
    # batch is ~300 MB of renders plus a ~100 MB scratch workspace, so an
    # unlimited library fills the disk in about a dozen jobs and every job after
    # that is refused at submit. Four days is long enough to download and share
    # what you made, short enough that the disk stays usable. Override with
    # CLIPFORGE_RETENTION_HOURS; 0 disables the sweeper entirely.
    retention_hours: int = 96
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
    #
    # This project renders ONE clip at a time per job and checkpoints after every
    # finished clip, so there is no parallelism gain worth the resume/state
    # complexity here. Keep it at 1 unless the host is explicitly sized for
    # parallel renders.
    worker_concurrency: int = 1


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
    nvidia: NvidiaConfig
    deepseek: DeepSeekConfig
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

    @property
    def nvidia_api_key(self) -> str | None:
        """LIVE NVIDIA key — re-resolved whenever the env files change."""
        state = live_nvidia_key()
        state.refresh()
        return state.key

    @property
    def deepseek_api_key(self) -> str | None:
        """LIVE DeepSeek key — re-resolved whenever the env files change."""
        state = live_deepseek_key()
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


def _load_openai_provider(
    section: dict,
    name: str,
    *,
    defaults: "GroqConfig | NvidiaConfig",
    errors: list[str],
    env_model: str,
    env_base_url: str,
) -> "GroqConfig | NvidiaConfig":
    """Build one OpenAI-compatible provider config from its TOML section.

    Groq and NVIDIA NIM speak the same chat-completions dialect, so they share
    every knob: response format, the JSON-retry ladder, the model chain, the
    token budget. Only the DEFAULTS differ, and those come from the dataclass
    rather than being written out twice here — which is the point: a field
    added to one provider cannot silently go missing from the other.

    `defaults` is a dataclass INSTANCE so every unset key falls back to that
    provider's own shipped value instead of to the other one's.
    """
    _rf = str(section.get("response_format", defaults.response_format)).strip().lower()
    if _rf not in {"json_schema", "json_object", "plain"}:
        errors.append(
            f'{name}.response_format "{_rf}" is not one of json_schema/json_object/plain'
        )
        _rf = defaults.response_format
    _mt = max(256, int(section.get("max_tokens", defaults.max_tokens)))
    _ceiling = max(_mt, int(section.get("max_tokens_ceiling", defaults.max_tokens_ceiling)))
    # A fallback equal to the primary model is a no-op that would burn a whole
    # retry ladder proving nothing, so it is dropped at load time.
    _primary = str(os.getenv(env_model) or section.get("model", defaults.model))
    _fallbacks = tuple(
        m for m in (
            str(x) for x in _as_list(section.get("fallback_models", defaults.fallback_models))
            if str(x).strip()
        )
        if m and m != _primary
    )
    return replace(
        defaults,
        model=_primary,
        base_url=str(
            os.getenv(env_base_url) or section.get("base_url", defaults.base_url)
        ).strip().rstrip("/"),
        max_retries=max(0, int(section.get("max_retries", defaults.max_retries))),
        timeout_sec=max(15, int(section.get("timeout_sec", defaults.timeout_sec))),
        min_interval_sec=max(
            0.0, float(section.get("min_interval_sec", defaults.min_interval_sec))
        ),
        temperature=float(section.get("temperature", defaults.temperature)),
        response_format=_rf,
        max_tokens=_mt,
        max_tokens_ceiling=_ceiling,
        json_retries=max(0, min(3, int(section.get("json_retries", defaults.json_retries)))),
        json_retry_temperature=max(
            0.0,
            min(
                float(section.get("temperature", defaults.temperature)),
                float(section.get("json_retry_temperature", defaults.json_retry_temperature)),
            ),
        ),
        fallback_models=_fallbacks,
        tokens_per_min=max(0, int(section.get("tokens_per_min", defaults.tokens_per_min))),
        daily_token_limit=max(
            1000, int(section.get("daily_token_limit", defaults.daily_token_limit))
        ),
    )


def _env_int(name: str, default: int) -> int:
    """An integer from the environment that cannot crash the boot.

    A typo in one env var (`CLIPFORGE_RETENTION_HOURS=six`) used to raise
    ValueError straight out of load_config, which takes the whole API down and
    serves 500 on every request — a far worse outcome than ignoring the value.
    Unparseable input falls back to the default and says so out loud; the
    value is never secret, so naming it in a log is safe.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        logger.warning(
            f"[config] {name}={raw!r} is not a whole number — using {default}"
        )
        return default


def load_config(root: Path = ROOT, quiet: bool = False) -> Settings:
    """Load and validate all settings. Raises ConfigError with readable text."""
    raw = _load_toml(root / "config.toml")
    _load_env_files(root)

    errors: list[str] = []

    app = raw.get("app", {})
    # Render and the managed preview inject PORT and expect a public bind.
    # The server must listen on 0.0.0.0, never on 127.0.0.1, because the
    # platform probes and the same-host proxy reach it by IP, not loopback.
    # config.toml app.host is only a bare-metal default and must never override
    # an explicit env value.
    host = (
        (os.getenv("CLIPFORGE_HOST") or "").strip()
        or (os.getenv("HOST") or "").strip()
        or "0.0.0.0"
    )
    # PORT is the platform-assigned listen port. CLIPFORGE_PORT is a legacy
    # alias some local scripts set. config.toml app.port is only a fallback for
    # bare-metal runs and must not win over the platform port.
    try:
        port = int(
            (os.getenv("PORT") or "").strip()
            or (os.getenv("CLIPFORGE_PORT") or "").strip()
            or app.get("port", 8000)
        )
    except (TypeError, ValueError):
        errors.append("app.port must be a whole number")
        port = 8000
    if not (1 <= port <= 65535):
        errors.append(f"app.port {port} is out of range (1-65535)")

    # Env overrides for the working folders — used by tests/tools to isolate
    # from the real job store (never by the hosted app itself).
    #
    # Anything WITHOUT an explicit override is relocated out of the codebase
    # first — but only `downloads` and `outputs`, the two trees that hold
    # gigabytes of media. `data` (the job store, transcripts and clip
    # documents) deliberately STAYS in the codebase: it is small, it has been
    # stable there for weeks, and it is the one directory whose contents are
    # read by absolute path from documents written long ago. Moving it buys
    # nothing the watcher ignore rules do not already buy.
    #
    # The env overrides exist so tests/tools can pin everything inside a temp
    # dir; when all three are given we honour them verbatim and never touch
    # the real folders.
    pinned = all(
        os.getenv(v)
        for v in (
            "CLIPFORGE_DOWNLOADS_DIR",
            "CLIPFORGE_OUTPUTS_DIR",
            "CLIPFORGE_DATA_DIR",
        )
    )
    storage = default_storage_root(root)
    if pinned:
        media: dict[str, Path] = {}
        storage_dir: Path | None = None
    else:
        media = {
            name: _relocate_to_storage(root, storage, name)
            for name in ("downloads", "outputs")
        }
        storage_dir = storage
        logger.info(f"[config] media storage: {storage}")

    data_dir = Path(os.getenv("CLIPFORGE_DATA_DIR", root / str(app.get("data_dir", "data"))))
    uploads_override = os.getenv("CLIPFORGE_UPLOADS_DIR", "").strip()

    paths = Paths(
        downloads_dir=Path(os.getenv("CLIPFORGE_DOWNLOADS_DIR", media.get("downloads", storage / "downloads"))),
        outputs_dir=Path(os.getenv("CLIPFORGE_OUTPUTS_DIR", media.get("outputs", storage / "outputs"))),
        data_dir=data_dir,
        static_dir=root / "webui",
        storage_dir=storage_dir,
        # Uploads live INSIDE the codebase on purpose — see Paths.uploads_dir
        # for the measurement that motivated it.
        uploads_dir=Path(uploads_override) if uploads_override else data_dir / "uploads",
    )
    for name in ("downloads_dir", "outputs_dir", "data_dir", "uploads_dir"):
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
        max_clips=min(clipcount_mod.HARD_CEILING, max(1, int(g.get("max_clips", 40)))),
        max_retries=max(0, int(g.get("max_retries", 3))),
        auto_clip_count=bool(g.get("auto_clip_count", True)),
        min_score=float(g.get("min_score", 60.0)),
    )
    if not 0.0 <= gemini.min_score <= 100.0:
        errors.append(f'gemini.min_score {gemini.min_score} is out of range (0-100)')
    if gemini.min_clip_sec >= gemini.max_clip_sec:
        errors.append("gemini.min_clip_sec must be < gemini.max_clip_sec")

    gq = raw.get("groq", {})
    groq = _load_openai_provider(
        gq, "groq", defaults=GroqConfig(), errors=errors, env_model="GROQ_MODEL",
        env_base_url="GROQ_BASE_URL",
    )

    nv = raw.get("nvidia", {})
    nvidia = _load_openai_provider(
        nv, "nvidia", defaults=NvidiaConfig(), errors=errors, env_model="NVIDIA_MODEL",
        env_base_url="NVIDIA_BASE_URL",
    )

    ds = raw.get("deepseek", {})
    _thinking = str(ds.get("thinking_mode", "disabled")).strip().lower()
    if _thinking not in {"enabled", "disabled"}:
        errors.append(
            f'deepseek.thinking_mode "{_thinking}" is not one of enabled/disabled'
        )
        _thinking = "disabled"
    deepseek = replace(
        _load_openai_provider(
            ds, "deepseek", defaults=DeepSeekConfig(), errors=errors,
            env_model="DEEPSEEK_MODEL", env_base_url="DEEPSEEK_BASE_URL",
        ),
        thinking_mode=_thinking,
    )

    lc = raw.get("llm", {})
    provider = str(
        os.getenv("CLIPFORGE_LLM_PROVIDER") or lc.get("provider", "deepseek")
    ).strip().lower()
    if provider not in {"deepseek", "nvidia", "groq", "gemini", "ollama"}:
        errors.append(
            f'llm.provider "{provider}" is not one of '
            "deepseek/nvidia/groq/gemini/ollama"
        )
        provider = "deepseek"
    # Automatic failover is opt-in (see LlmConfig). An unset, blank, or
    # self-referential value means "no failover" rather than a config error:
    # pointing the fallback at the primary would burn a whole second ladder
    # proving what the first one already proved.
    fallback_provider = str(lc.get("fallback_provider", "") or "").strip().lower()
    if fallback_provider == provider:
        fallback_provider = ""
    if fallback_provider and fallback_provider not in {
        "deepseek", "nvidia", "groq", "gemini", "ollama"
    }:
        errors.append(
            f'llm.fallback_provider "{fallback_provider}" is not one of '
            "deepseek/nvidia/groq/gemini/ollama"
        )
        fallback_provider = ""
    sc = raw.get("social", {})

    llm = LlmConfig(
        provider=provider,
        fallback_provider=fallback_provider,
        # Detection chunks are INDEPENDENT network calls, so overlapping them
        # is a real win on a host with one CPU: the work is waiting on a round
        # trip, not on a core. Capped at 4 because that is where the gain
        # flattens and rate limits begin to bite -- and 4 is what the provider
        # tiers actually sustain. 1 restores the strictly sequential behaviour.
        detect_concurrency=max(
            1,
            min(
                8,
                int(os.getenv("CLIPFORGE_DETECT_CONCURRENCY")
                    or lc.get("detect_concurrency", 4)),
            ),
        ),
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
    if subtitle.preset not in {"bold_pop", "karaoke", "lower_third", "boxed", "vertical_916"}:
        errors.append(
            f'subtitle.preset "{subtitle.preset}" is not one of '
            "bold_pop/karaoke/lower_third/boxed/vertical_916"
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
        width=int(v.get("width", 720)),
        height=int(v.get("height", 1280)),
        fps=int(v.get("fps", 30)),
        crf=int(v.get("crf", 23)),
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
        cookies_file=str(
            os.getenv("YT_DLP_COOKIES_FILE", d.get("cookies_file", "") or "")
        ).strip(),
        proxy=(os.getenv("YTDLP_PROXY") or d.get("proxy", "") or "").strip(),
        proxy_list=(os.getenv("YTDLP_PROXY_LIST") or d.get("proxy_list", "") or "").strip(),
    )

    h = raw.get("hosted", {})
    hosted = HostedConfig(
        max_source_min=max(1, int(h.get("max_source_min", os.getenv("CLIPFORGE_MAX_SOURCE_MIN", 120)))),
        # 0 = no limit. Enforced at submit time so an oversized upload is
        # refused before its bytes occupy the disk.
        max_source_mb=max(0, int(
            os.getenv("CLIPFORGE_MAX_SOURCE_MB")
            or h.get("max_source_mb", 4096)
        )),
        # ENV FIRST, then config.toml, then the 4-day default.
        #
        # This was `h.get("retention_hours", os.getenv(...))`, which evaluates
        # the env fallback ONLY when the key is absent from config.toml. The
        # key is present (shipped as `retention_hours = 0`), so
        # CLIPFORGE_RETENTION_HOURS was silently dead: setting it changed
        # nothing, and the sweeper stayed off because the file said 0. Every
        # other tunable in this block already reads `os.getenv(...) or
        # h.get(...)` — this one line was the odd one out.
        retention_hours=max(0, _env_int(
            "CLIPFORGE_RETENTION_HOURS", h.get("retention_hours", 96)
        )),
        worker_url=(os.getenv("WORKER_URL") or h.get("worker_url", "") or "").strip().rstrip("/"),
        worker_token=(os.getenv("WORKER_TOKEN") or h.get("worker_token", "") or "").strip(),
        tailscale_only=str(os.getenv("TAILSCALE_ONLY", str(h.get("tailscale_only", False)))).strip().lower()
        in {"1", "true", "yes", "on"},
        worker_concurrency=max(
            1,
            min(
                8,
                int(os.getenv("CLIPFORGE_WORKER_CONCURRENCY") or h.get("worker_concurrency", 1)),
            ),
        ),
    )

    # Initialize the live key state so the boot banner reports it. The key is
    # NEVER snapshotted into Settings — every read re-resolves.
    live_gemini_key().refresh(force=True)
    live_nvidia_key().refresh(force=True)
    live_deepseek_key().refresh(force=True)

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
        nvidia=nvidia,
        deepseek=deepseek,
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
    # Report the key for the ACTIVE backend. Printing only Gemini made a
    # fully-working deployment look unconfigured, and printing only Groq would
    # now do the same for an NVIDIA-primary one.
    if active == "deepseek":
        live: _LiveKeyState = live_deepseek_key()
        env_name = "DEEPSEEK_API_KEY"
    elif active == "nvidia":
        live = live_nvidia_key()
        env_name = "NVIDIA_API_KEY"
    elif active == "groq":
        live = live_groq_key()
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
    # Key HYGIENE, not key presence: the active-provider line above only answers
    # "is there a key", so a superseded or placeholder DeepSeek key still printed
    # a clean banner and then failed every detection with a 401. Runs for the
    # DeepSeek key regardless of which provider is active, because rotating it is
    # the step most likely to leave the phone-side app calling a revoked key.
    try:
        from app import keyaudit  # local import: config loads before app.main

        audit = keyaudit.audit_live()
        if audit["ok"]:
            print(f"[boot] DeepSeek key audit: OK ({audit['state']}, {audit['masked']})")
        else:
            print(f"[boot] WARNING: DeepSeek key audit: {audit['state'].upper()} — {audit['warning']}")
        if audit.get("note"):
            print(f"[boot] DeepSeek key audit note: {audit['note']}")
    except Exception as exc:  # noqa: BLE001 - an audit must never fail the boot
        print(f"[boot] DeepSeek key audit skipped: {type(exc).__name__}: {exc}")
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
    print(
        f"[boot] Folders: uploads {settings.app.paths.uploads_dir} (new sources) "
        f"/ downloads {settings.app.paths.downloads_dir} (legacy sources, read-only fallback) "
        f"/ outputs {settings.app.paths.outputs_dir} / data {settings.app.paths.data_dir}"
    )
    retention = (
        "disabled (jobs are never auto-deleted)"
        if settings.hosted.retention_hours <= 0
        else f"{settings.hosted.retention_hours}h"
    )
    print(f"[boot] Mode: {'tailscale-only' if settings.hosted.tailscale_only else 'public'} | max source: {settings.hosted.max_source_min} min / {settings.hosted.max_source_mb} MB | retention: {retention}")
    # Media durability. Off unless a bucket and keys are configured, so this
    # line is the honest answer to "will my clips outlive this sandbox?".
    from app import objectstore

    off = objectstore.status()
    if off["enabled"]:
        local = "kept locally too" if off["deleteLocal"] is False else "local copy dropped after push"
        print(f"[boot] Durable media: ON — {off['bucket']} @ {off['endpoint']} ({local})")
    else:
        print("[boot] Durable media: OFF — no bucket configured, media dies with the sandbox")
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
