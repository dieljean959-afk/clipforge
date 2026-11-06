"""The env snapshot cache must never hide a newly saved key.

/api/health re-resolves keys on every poll so a key saved in the platform is
picked up without a restart. That is only true while the cached snapshot is
invalidated when the env files change — and this cache exists because the
decryption it replaces (hundreds of ms per poll) made health look dead to the
supervisor, which restarted the API in a loop.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.config import _env_files_fingerprint, _fresh_env_snapshot


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    # Filesystems with 1s mtime granularity would otherwise hide the change;
    # the fingerprint includes size too, but be explicit about the intent.
    now = time.time() + 2
    os.utime(path, (now, now))


def test_snapshot_is_cached(tmp_path: Path) -> None:
    _write(tmp_path / ".env.local", "ALPHA=1\n")
    first_fp, first = _fresh_env_snapshot(tmp_path)
    second_fp, second = _fresh_env_snapshot(tmp_path)
    assert first_fp == second_fp
    assert first["ALPHA"] == second["ALPHA"] == "1"


def test_new_value_is_seen_without_a_restart(tmp_path: Path) -> None:
    _write(tmp_path / ".env.local", "ALPHA=1\n")
    _fresh_env_snapshot(tmp_path)

    _write(tmp_path / ".env.local", "ALPHA=1\nBETA=saved-in-the-ui\n")
    assert _env_files_fingerprint(tmp_path) != ""
    fp, values = _fresh_env_snapshot(tmp_path)
    assert values["BETA"] == "saved-in-the-ui", "cache hid a freshly written key"

    # And the cheap fingerprint tracks the same change, so _LiveKeyState.refresh
    # can decide to re-resolve without paying for decryption.
    before = _env_files_fingerprint(tmp_path)
    _write(tmp_path / ".env.local", "ALPHA=1\nBETA=saved-in-the-ui\nGAMMA=3\n")
    assert _env_files_fingerprint(tmp_path) != before
    _fp, values = _fresh_env_snapshot(tmp_path)
    assert values["GAMMA"] == "3"
    assert fp != _env_files_fingerprint(tmp_path)
