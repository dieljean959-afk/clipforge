"""One active batch per video + settings, and stable source identity."""

from __future__ import annotations

from types import SimpleNamespace

from app import main


def test_share_tokens_do_not_make_a_new_video():
    a = main.source_ident("https://youtu.be/627Fvn2fEjU?si=MMQ3k1JaLHyOfjyu")
    b = main.source_ident("https://www.youtube.com/watch?v=627Fvn2fEjU&si=other")
    c = main.source_ident("https://m.youtube.com/shorts/627Fvn2fEjU")
    assert a == b == c == "youtube:627Fvn2fEjU"


def test_different_videos_stay_different():
    assert main.source_ident("https://youtu.be/aaaaaaaaaaa") != main.source_ident(
        "https://youtu.be/bbbbbbbbbbb"
    )


def _body(**kw):
    base = dict(
        sourceType="url", source="https://youtu.be/627Fvn2fEjU?si=x",
        minClipSec=None, maxClipSec=None, maxClips=15, captionPreset=None,
        language="es", emojiEnabled=True, emojiDensity="medium",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _job(**kw):
    job = {
        "_id": "job1", "sourceType": "url", "source": "https://www.youtube.com/watch?v=627Fvn2fEjU",
        "stage": "transcribing", "options": {"minClipSec": 20, "maxClipSec": 60, "maxClips": 15},
        "language": "es", "captionPreset": "bold_stack", "emojiEnabled": True,
        "emojiDensity": "medium",
    }
    job.update(kw)
    return job


def test_an_active_job_for_the_same_source_is_reused(monkeypatch):
    monkeypatch.setattr(main, "store", SimpleNamespace(list_jobs=lambda: [_job()]))
    assert main._active_job_for(_body()).get("_id") == "job1"


def test_terminal_jobs_do_not_block_a_new_run(monkeypatch):
    monkeypatch.setattr(
        main, "store", SimpleNamespace(list_jobs=lambda: [_job(stage="failed")])
    )
    assert main._active_job_for(_body()) is None


def test_different_settings_are_a_different_batch(monkeypatch):
    monkeypatch.setattr(
        main, "store", SimpleNamespace(list_jobs=lambda: [_job(language="en")])
    )
    assert main._active_job_for(_body()) is None


def test_different_video_is_a_different_batch(monkeypatch):
    monkeypatch.setattr(
        main, "store",
        SimpleNamespace(list_jobs=lambda: [_job(source="https://youtu.be/zzzzzzzzzzz")]),
    )
    assert main._active_job_for(_body()) is None
