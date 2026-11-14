"""Patch write_clip so a re-render of one clip does not clobber its resolved
captionSettings. The per-clip render write landed the store's upsert path,
which dropped 'renderState' because the dict already had it, so any
field missing from the job only was left empty."""
from __future__ import annotations
import pathlib, tempfile
import pytest
from app.store import LocalStore
@pytest.fixture
def clips_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    pd = tmp_path / "clips"
    pd.mkdir()
    return pd
def make_store(tmp_path: pathlib.Path) -> LocalStore:
    class P: data_dir=tmp_path; jobs_dir=tmp_path/'jobs'; clips_dir=tmp_path/'clips'
    store = LocalStore(pathlib.Path("/tmp/nonexistent"))
    object.__setattr__(store, "_paths", P())
    return store
def test_write_clip_keeps_existing_caption_settings_on_partial_write(clips_dir: pathlib.Path) -> None:
    store = make_store(clips_dir.parent)
    # first write the whole document the pipeline would produce for a done clip
    doc = {
        "_id": "c1",
        "jobId": "j1",
        "order": 3,
        "title": "Clip",
        "start": 10.0,
        "end": 15.0,
        "renderState": "done",
        "captionSettings": {
            "presetKey": "bold_stack",
            "fontFamily": "Montserrat-ExtraBold",
            "fontSize": 54,
            "outline": 6,
            "primaryColor": "#FFFFFF",
            "highlightColor": "#FFFF00",
            "positionY": 0.72,
            "lines": 2,
            "maxCharsPerLine": 16,
            "position_y": 0.72,
            "highlight_color": "#FFFF00",
            "box": False,
            "uppercase": True,
        },
        "mediaWidth": 1080,
        "mediaHeight": 1920,
        "mediaDurationSec": 11.7,
        "mediaBytes": 5199000,
        "mediaRemote": True,
        "videoPath": "/data/clip_003.mp4",
        "styleId": "a8091ebd9132",
        "ffmpegCommand": "ffmpeg ...",
        "ffmpegStderr": "ffmpeg text",
    }
    store.write_clip(doc)
    # now a re-render is attempted; if that path lands a partial patch,
    # captionSettings must survive
    merged = store.write_clip("c1", {"renderState": "pending"})
    assert merged["renderState"] == "pending"
    cap = merged["captionSettings"]
    assert cap["presetKey"] == "bold_stack"
    assert cap["fontFamily"] == "Montserrat-ExtraBold"
    assert cap["fontSize"] == 54
    assert cap["primaryColor"] == "#FFFFFF"
    assert cap["highlightColor"] == "#FFFF00"
    assert cap["position_y"] == 0.72
    assert merged["mediaWidth"] == 1080
    assert merged["mediaHeight"] == 1920
    assert merged["mediaDurationSec"] == 11.7
    assert merged["mediaBytes"] == 5199000
    assert merged["mediaRemote"] is True
    assert merged["videoPath"] == "/data/clip_003.mp4"
    assert merged["styleId"] == "a8091ebd9132"
    assert merged["ffmpegCommand"] == "ffmpeg ..."
    assert merged["ffmpegStderr"] == "ffmpeg text"
