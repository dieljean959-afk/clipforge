"""Model accounting honesty + the multilingual-only ladder.

Two live bugs:

* Reset (Local disk) promised "Would free 2263.1 MB: local model copies" and then
  freed 2.1 MB, because the preview summed every byte under data/models while the
  apply could only release what the modelstore would verify. The big item was an
  ORPHAN HuggingFace blob that belongs to no model, which per-model accounting
  never saw at all.
* The transcription ladder used an ENGLISH-ONLY model (`distil-medium.en`) for
  language="auto", so a Spanish source was transcribed as mangled English
  ("Guegles are in car's...") and the detected language was locked to "en".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import modelstore
from app.asr_accurate import model_ladder

# --------------------------------------------------------------------------- #
# ladder: never English-only unless the language is known to be English
# --------------------------------------------------------------------------- #

def test_auto_never_picks_an_english_only_model() -> None:
    for language in (None, "", "auto", "AUTO", "es", "pt"):
        ladder = model_ladder(language)
        assert "distil-medium.en" not in ladder, (
            f"language={language!r} must not use English-only weights"
        )
        assert ladder[0] == "small", "multilingual small must lead the ladder"


def test_english_uses_the_configured_multilingual_model() -> None:
    """distil-medium.en is gone from the ladder: a forced-English batch pulled
    its 752 MB, filled the disk and left the job waiting for space, for no
    quality gain (4.7% vs 10.1% WER on this host)."""
    for language in ("en", "EN", "english"):
        ladder = model_ladder(language)
        assert "distil-medium.en" not in ladder
        assert ladder[0] == "small"


# --------------------------------------------------------------------------- #
# modelstore accounting
# --------------------------------------------------------------------------- #

def _touch(path: Path, mb: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * (mb * 1024 * 1024))


def test_releasable_counts_an_unused_model_and_keeps_the_active_one(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "models"
    monkeypatch.setattr(modelstore, "_root", lambda: root)
    monkeypatch.setattr(modelstore, "_hf_cache_root", lambda: tmp_path / "none")

    _touch(modelstore.local_dir("small") / "model.bin", 2)
    _touch(modelstore.local_dir("medium") / "model.bin", 3)

    plan = modelstore.releasable("small", keep_active_local=True)
    # Only the unused 3 MB model is offered; the active one is kept with a reason.
    assert plan["mb"] == 3.0
    assert any("medium" in item["what"] for item in plan["items"])
    assert any("active model" in kept["why"] for kept in plan["kept"])


def test_releasable_counts_orphan_blobs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "models"
    cache = tmp_path / "hf"
    monkeypatch.setattr(modelstore, "_root", lambda: root)
    monkeypatch.setattr(modelstore, "_hf_cache_root", lambda: cache)

    blob = cache / "hub" / "blobs" / "ab" / "abcdef"
    _touch(blob, 4)
    plan = modelstore.releasable("small", keep_active_local=True)
    assert plan["mb"] == 4.0
    assert any("orphan" in item["what"] for item in plan["items"])


def test_releasable_is_zero_when_only_the_active_model_exists(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "models"
    monkeypatch.setattr(modelstore, "_root", lambda: root)
    monkeypatch.setattr(modelstore, "_hf_cache_root", lambda: tmp_path / "none")
    _touch(modelstore.local_dir("small") / "model.bin", 2)

    plan = modelstore.releasable("small", keep_active_local=True)
    assert plan["mb"] == 0.0
    assert plan["kept"], "an honest zero must explain why nothing is freed"


def test_diskreset_uses_the_releasable_plan() -> None:
    src = (Path(__file__).resolve().parents[1] / "app/diskreset.py").read_text(encoding="utf-8")
    assert "modelstore.releasable(" in src
    # The old preview summed status()['localBytes'], which is the 2263 MB bug.
    assert "localBytes" not in src


def test_release_refuses_the_active_model_when_keep_local(
    tmp_path: Path, monkeypatch,
) -> None:
    """The 464 MB re-download bug: a bucket-restored copy had no persist marker,
    so the next job start's release_all() deleted it."""
    root = tmp_path / "models"
    monkeypatch.setattr(modelstore, "_root", lambda: root)
    monkeypatch.setattr(modelstore, "_hf_cache_root", lambda: tmp_path / "none")
    _touch(modelstore.local_dir("small") / "model.bin", 1)
    monkeypatch.setattr(modelstore, "active_model_name", lambda: ("small", ""))

    from types import SimpleNamespace

    from app import config

    monkeypatch.setattr(
        config, "load_config",
        lambda **kw: SimpleNamespace(whisper=SimpleNamespace(keep_local=True)),
    )
    result = modelstore.release("small")
    assert result["ok"] is False
    assert "active model" in result["reason"]
    assert modelstore.local_dir("small").is_dir(), "the kept copy must survive"


def test_a_retry_cannot_download_a_model_that_is_not_configured(
    tmp_path: Path, monkeypatch,
) -> None:
    """The gap-retry ladder pulled base (421 MB) and tiny (221 MB) on one
    19-second clip. A retry may reuse weights, never fetch new ones."""
    from types import SimpleNamespace

    from app import asr, config

    monkeypatch.setattr(modelstore, "_root", lambda: tmp_path / "models")
    monkeypatch.setattr(
        config, "load_config",
        lambda **kw: SimpleNamespace(whisper=SimpleNamespace(model="small")),
    )
    with pytest.raises(asr.AsrError, match="refusing to download 'base'"):
        asr._refuse_silent_download("base")
    # The configured model is always allowed: it is what a job exists to load.
    asr._refuse_silent_download("small")
    # A copy already on disk may be reused, even when it is not the configured one.
    local = modelstore.local_dir("base")
    local.mkdir(parents=True)
    for name in ("model.bin", "config.json"):
        (local / name).write_bytes(b"x")
    asr._refuse_silent_download("base")
