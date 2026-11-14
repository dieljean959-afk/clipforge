"""A rendered clip is only "done" when it is the right file.

`verify_output` already proves the encode produced something playable and long
enough. It cannot know which frame size the BATCH asked for, so a graph that
skipped the final 9:16 scale could publish a valid-looking 16:9 file and the
job would report it as a finished clip.
"""

from __future__ import annotations

import pytest

from app.encode import check_geometry
from app.video import RenderError


def test_the_configured_9x16_geometry_passes():
    check_geometry({"width": 1080, "height": 1920}, 1080, 1920, "clip_000.mp4")


@pytest.mark.parametrize(
    "width, height",
    [(1920, 1080), (1080, 1080), (0, 0), (1078, 1920)],
)
def test_any_other_geometry_fails_the_render(width, height):
    with pytest.raises(RenderError) as caught:
        check_geometry({"width": width, "height": height}, 1080, 1920, "clip_003.mp4")
    assert "expected 1080x1920" in str(caught.value)
    assert "clip_003.mp4" in str(caught.value)


def test_a_missing_probe_is_a_failure_not_a_pass():
    with pytest.raises(RenderError):
        check_geometry({}, 1080, 1920)
