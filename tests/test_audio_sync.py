"""Regression tests for the three real-media failure classes.

No media is generated here: these are pure plan/fingerprint tests. The real
YouTube source was blocked in this sandbox by YouTube's bot check, so the
end-to-end evidence is reported separately.
"""

from __future__ import annotations

from app import audiosync, clipcount, encode


def test_whole_source_clip_seeks_to_its_absolute_start():
    inputs, audio_map = encode.source_input_args(
        "https://bucket/video.mp4",
        "https://bucket/audio.m4a",
        seek=114.07,
        audio_seek=114.07,
        clip_seconds=38.0,
    )
    assert audio_map == "1:a:0?"
    assert inputs[inputs.index("-ss") + 1] == "114.070"
    # Two inputs, two independent -ss/-t pairs.
    assert inputs.count("-ss") == 2
    assert inputs.count("-t") == 2
    second = inputs.index("-i", inputs.index("-i") + 1)
    assert inputs[second - 3] == "114.070"


def test_section_relative_video_does_not_shift_audio():
    # Real numbers from job c840...: clip 2 starts at 114.07 s, its section
    # starts at 75 s. The old code used 39.07 for BOTH inputs, which put clip
    # A's audio (13.67 s earlier) under clip 2's video.
    inputs, _ = encode.source_input_args(
        "https://bucket/sec_000_p1.mp4",
        "https://bucket/audio.m4a",
        seek=39.07,
        audio_seek=114.07,
        clip_seconds=38.0,
    )
    seeks = [inputs[i + 1] for i, value in enumerate(inputs) if value == "-ss"]
    assert seeks == ["39.070", "114.070"]


def test_audio_start_time_is_removed_from_the_audio_seek():
    # `a_seek` is computed before source_input_args is called; keep that
    # arithmetic explicit and testable via the helper's own contract.
    inputs, _ = encode.source_input_args(
        "video", "audio", seek=39.07, audio_seek=113.07, clip_seconds=38.0
    )
    seeks = [inputs[i + 1] for i, value in enumerate(inputs) if value == "-ss"]
    assert seeks == ["39.070", "113.070"]


def test_negative_audio_offset_never_becomes_a_negative_seek():
    inputs, _ = encode.source_input_args(
        "video", "audio", seek=-4.0, audio_seek=-2.0, clip_seconds=38.0
    )
    seeks = [inputs[i + 1] for i, value in enumerate(inputs) if value == "-ss"]
    assert seeks == ["0.000", "0.000"]


def test_clamp_emoji_windows_removes_the_erange_negative_start():
    items = [
        {"path": "a.png", "start": -0.13, "end": 0.81, "x": 470, "y": 1132},
        {"path": "b.png", "start": 30.53, "end": 31.81},
        {"path": "c.png", "start": 38.2, "end": 39.0},
    ]
    out, changed = encode.clamp_emoji_windows(items, 38.0)
    assert changed == 2
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 0.81
    assert out[1]["end"] == 31.81
    assert all(item["start"] >= 0 for item in out)
    assert all(item["end"] <= 38.0 for item in out)


def test_span_validation_clamps_eof_and_negative_start():
    start, end, note = encode.clamp_span(-2.0, 999.0, 120.0)
    assert (start, end) == (0.0, 120.0)
    assert note and "clamped" in note


def test_span_validation_rejects_an_empty_plan():
    try:
        encode.clamp_span(76.1, 76.1, 120.0)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an empty span must be rejected before ffmpeg")


def test_video_and_audio_seeks_are_independent():
    args, audio_map = encode.source_input_args(
        "sec_000_p1.mp4",
        "audio.m4a",
        seek=39.07,       # section-relative: 75 + 39.07 = 114.07
        audio_seek=114.07,  # whole audio: absolute
        clip_seconds=38.0,
    )
    assert args.count("-ss") == 2
    assert args[args.index("-ss") + 1] == "39.070"
    second = args.index("-ss", args.index("-ss") + 1)
    assert args[second + 1] == "114.070"
    assert audio_map == "1:a:0?"
    assert "-accurate_seek" in args


def test_requested_count_asks_for_replacement_candidates():
    assert clipcount.candidate_target(10, exact=True) == 14
    assert clipcount.candidate_target(1, exact=True) == 4
    # Auto counts are ceilings, not quotas: no extra render candidates.
    assert clipcount.candidate_target(10, exact=False) == 10


def test_retry_recovers_the_old_static_crop():
    from app import video

    command = "ffmpeg -vf crop=608:1080:656:0,scale=1080:1920 ..."
    assert video._crop_from_command(command) == {
        "w": 608, "h": 1080, "x": 656, "y": 0,
    }
    assert video._crop_from_command("ffmpeg -vf crop=iw/2:ih:0:0") is None


def test_presigned_query_strings_are_redacted_from_errors():
    from app import video

    message = "ffprobe https://s3.example/bucket/key.mp4?X-Amz-Signature=secret&X-Amz-Credential=AKIA"
    safe = video.redact_urls(message)
    assert "secret" not in safe and "AKIA" not in safe
    assert safe.endswith("?<redacted>")


def test_duplicate_requires_both_frames_and_audio_to_match():
    fp = {
        "video": [10.0] * 256,
        "audio": [0.1] * 128,
    }
    same_audio_other_frames = {
        "clipId": "a",
        "fingerprint": {"video": [200.0] * 256, "audio": [0.1] * 128},
    }
    same_frames_other_audio = {
        "clipId": "b",
        "fingerprint": {
            "video": [10.0] * 256,
            "audio": [0.9 if i % 2 else 0.0 for i in range(128)],
        },
    }
    assert audiosync.duplicate_of(fp, [same_audio_other_frames])["ok"] is True
    assert audiosync.duplicate_of(fp, [same_frames_other_audio])["ok"] is True
    exact = {
        "clipId": "c",
        "fingerprint": {"video": [10.0] * 256, "audio": [0.1] * 128},
    }
    dup = audiosync.duplicate_of(fp, [exact])
    assert dup["ok"] is False
    assert dup["duplicateOf"] == "c"
