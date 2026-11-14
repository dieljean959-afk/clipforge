from app import translate, subtitle
from app.task import PipelineTask
from app.store import LocalStore
from pathlib import Path
import json

# Minimal stubs to find the real API surface of translate.build_cues vs task._stage_detect
print(json.dumps({
    "build_cues_sig": getattr(translate.build_cues, "__annotations__", None),
    "write_tracks_sig": getattr(translate.write_tracks, "__annotations__", None),
    "subtitle_has_write_tracks": hasattr(subtitle, "write_tracks"),
}, indent=2))
