"""The source never becomes a local file; these tests pin that behaviour."""

from __future__ import annotations

import io

import pytest

from app import objectstore, source_stream


class FakeBucket:
    """Multipart-capable S3 stand-in."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.parts: dict[str, dict[int, bytes]] = {}
        self.aborted: list[str] = []
        self._n = 0

    # ---- multipart ----
    def create_multipart_upload(self, **kw):
        self._n += 1
        uid = f"up-{self._n}"
        self.parts[uid] = {}
        return {"UploadId": uid}

    def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):  # noqa: N803
        self.parts[UploadId][PartNumber] = Body
        return {"ETag": f"etag-{PartNumber}"}

    def complete_multipart_upload(self, *, Bucket, Key, UploadId, MultipartUpload):  # noqa: N803
        body = b"".join(self.parts[UploadId][n] for n in sorted(self.parts[UploadId]))
        self.objects[Key] = body
        return {}

    def abort_multipart_upload(self, *, Bucket, Key, UploadId):  # noqa: N803
        self.aborted.append(Key)
        return {}

    # ---- reads ----
    def head_object(self, *, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def generate_presigned_url(self, op, Params=None, ExpiresIn=900):  # noqa: N803
        return f"https://bucket.example/{Params['Key']}?sig=1&exp={ExpiresIn}"

    # ---- plain upload (sections fallback / fixtures) ----
    def upload_file(self, path, bucket, key, ExtraArgs=None):  # noqa: N803
        from pathlib import Path as _P

        self.objects[key] = _P(path).read_bytes()

    def copy_object(self, **kw):
        self.objects[kw["Key"]] = self.objects.pop(kw["CopySource"]["Key"])

    def delete_object(self, **kw):
        self.objects.pop(kw["Key"], None)

    def list_objects_v2(self, **kw):
        prefix = kw.get("Prefix") or ""
        keys = [k for k in self.objects if k.startswith(prefix)]
        return {"Contents": [{"Key": k, "Size": len(self.objects[k])} for k in keys]}


@pytest.fixture()
def bucket(monkeypatch, tmp_path):
    fake = FakeBucket()
    monkeypatch.setattr(objectstore, "_CLIENT_OVERRIDE", fake)
    monkeypatch.setenv("CLIPFORGE_S3_ENDPOINT", "https://s3.filebase.io")
    monkeypatch.setenv("CLIPFORGE_S3_BUCKET", "clipforge-clips")
    monkeypatch.setenv("CLIPFORGE_S3_ACCESS_KEY", "AKIA-test")
    monkeypatch.setenv("CLIPFORGE_S3_SECRET_KEY", "secret")
    monkeypatch.setenv("CLIPFORGE_S3_PREFIX", "clipforge")
    return fake


def test_a_pipe_is_uploaded_in_parts_and_verified(bucket):
    payload = b"x" * (source_stream.PART_BYTES * 2 + 17)
    sent = source_stream._upload_pipe(io.BytesIO(payload), "clipforge/uploads/j/video.mp4", "video/mp4")

    assert sent == len(payload)
    assert bucket.objects["clipforge/uploads/j/video.mp4"] == payload
    assert source_stream._verify("clipforge/uploads/j/video.mp4", sent)["bytes"] == len(payload)


def test_a_truncated_object_is_refused(bucket):
    bucket.objects["clipforge/uploads/j/video.mp4"] = b"short"
    with pytest.raises(source_stream.StreamError):
        source_stream._verify("clipforge/uploads/j/video.mp4", 999)


def test_the_capture_records_no_local_path(monkeypatch, bucket):
    source = source_stream.RemoteSource(
        "j",
        video_key="clipforge/uploads/j/video.mp4",
        audio_key="clipforge/uploads/j/audio.m4a",
        probe={"width": 1280, "height": 720, "duration": 1200.0, "fps": 30.0},
        method="stdout-stream",
        format_id="bv*[height<=720]",
        audio_format="ba",
    )
    desc = source.describe()
    assert "path" not in desc
    assert desc["videoKey"].startswith("clipforge/uploads/j/")
    assert str(source).startswith("https://bucket.example/clipforge/uploads/j/video.mp4")


def test_a_clip_is_cut_from_the_section_that_contains_it(bucket):
    """A clip inside one section is cut there, with no concatenation."""
    source = source_stream.RemoteSource(
        "j",
        video_key="clipforge/uploads/j/sections/sec_000.mp4",
        audio_key="clipforge/uploads/j/audio.m4a",
        method="sections",
        sections=[
            {"key": "clipforge/uploads/j/sections/sec_000.mp4", "start": 0.0, "end": 300.0},
            {"key": "clipforge/uploads/j/sections/sec_001.mp4", "start": 300.0, "end": 600.0},
        ],
    )
    url, seek = source.url_for_span(412.5)
    assert url.endswith("sections/sec_001.mp4?sig=1&exp=900")
    assert seek == pytest.approx(112.5)
    url0, seek0 = source.url_for_span(12.0)
    assert url0.endswith("sections/sec_000.mp4?sig=1&exp=900")
    assert seek0 == pytest.approx(12.0)


def test_extra_decoded_seconds_in_a_section_are_usable(bucket, monkeypatch):
    """A 75 s object can really contain ~78 s; do not reject a clip at 152 s."""
    source = source_stream.RemoteSource(
        "j",
        video_key="s0",
        audio_key="a",
        method="sections",
        sections=[
            {"key": "s0", "start": 75.0, "end": 150.0},
            {"key": "s1", "start": 150.0, "end": 225.0},
        ],
    )
    monkeypatch.setattr(source, "_actual_section_end", lambda sec: 153.6)
    url, seek = source.url_for_span(114.07, 152.07)
    assert seek == pytest.approx(39.07)
    assert "s0" in url


def test_a_clip_crossing_a_section_boundary_fails_instead_of_truncating(bucket):
    source = source_stream.RemoteSource(
        "j",
        video_key="clipforge/uploads/j/sections/sec_000.mp4",
        audio_key="clipforge/uploads/j/audio.m4a",
        method="sections",
        sections=[
            {"key": "s0", "start": 0.0, "end": 750.0},
            {"key": "s1", "start": 750.0, "end": 900.0},
        ],
    )
    with pytest.raises(source_stream.StreamError, match="crosses source section"):
        source.url_for_span(747.4, 785.4)
    # Sampling/reframe still resolves the start without asking for the end.
    _, seek = source.url_for_span(747.4)
    assert seek == pytest.approx(747.4)


# --- the length of the SOURCE, not of the object that was probed ---------
#
# A sections capture stores 150 s objects and probes ONE of them. Taking that
# probe as the video's duration reported a 20:37 source as "2:30 source",
# scaled the auto clip count off 2:30, and stopped detection at the first
# window — the whole batch worked on the video's head.


def test_sections_source_reports_the_whole_video_not_the_first_section():
    sections = [
        {"key": "s0", "start": 0.0, "end": 150.0},
        {"key": "s1", "start": 150.0, "end": 300.0},
        {"key": "s8", "start": 1200.0, "end": 1237.0},
    ]
    got = source_stream.source_duration(probed=150.08, meta_sec=1237.0, sections=sections)
    assert got == pytest.approx(1237.0), f"20:37 source reported as {got:.0f}s"


def test_sections_win_even_when_the_metadata_probe_failed():
    sections = [{"key": "s0", "start": 0.0, "end": 150.0},
                {"key": "s1", "start": 150.0, "end": 300.0}]
    assert source_stream.source_duration(probed=150.0, meta_sec=0.0, sections=sections) == 300.0


def test_a_whole_object_probe_is_the_source():
    assert source_stream.source_duration(probed=1237.4, meta_sec=0.0, sections=[]) == 1237.4
    assert source_stream.source_duration(probed=0.0, meta_sec=1237.0, sections=[]) == 1237.0


def test_an_unknown_length_no_longer_caps_the_capture_at_15_minutes():
    # 6 x 150 s = 15 min used to be the whole capture for a source whose
    # metadata could not be read — a 20-minute video silently lost 5 minutes.
    assert source_stream._section_count(0.0) == source_stream.UNKNOWN_DURATION_SECTIONS
    assert source_stream.UNKNOWN_DURATION_SECTIONS * source_stream.SECTION_SEC >= 2 * 3600


def test_a_known_length_still_asks_for_exactly_what_is_needed():
    assert source_stream._section_count(1237.0) == 17
    assert source_stream._section_count(120.0) == 2


def test_section_size_stays_frame_accurate():
    # 150 s + `-c copy` split started 3.5 s early in the real batch. Keep the
    # direct section at 75 s unless a frame-accuracy test replaces this.
    assert source_stream.SECTION_SEC <= 75.0


def test_oversize_section_split_reencodes_instead_of_copying(monkeypatch, tmp_path):
    from types import SimpleNamespace

    src = tmp_path / "sec.mp4"
    src.write_bytes(b"x" * 15)
    calls = []

    def fake_run(cmd, **kw):
        from pathlib import Path

        out = Path(cmd[-1])
        out.write_bytes(b"y" * 5)
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(source_stream.subprocess, "run", fake_run)
    pieces = source_stream._split_if_big(src, target=10, seconds=2.0)
    assert len(pieces) == 2
    for cmd in calls:
        assert "copy" not in cmd
        assert "libx264" in cmd


def test_a_whole_object_source_is_seeked_to_the_clip_not_zero(bucket):
    """The old whole-object branch returned seek=0 for every clip."""
    source = source_stream.RemoteSource(
        "j", video_key="clipforge/uploads/j/video.mp4",
        audio_key="clipforge/uploads/j/audio.m4a",
    )
    url, seek = source.url_for_span(412.5)
    assert url.endswith("uploads/j/video.mp4?sig=1&exp=900")
    assert seek == pytest.approx(412.5)


def test_presigned_urls_are_fresh_per_call(bucket):
    source = source_stream.RemoteSource(
        "j", video_key="clipforge/uploads/j/video.mp4",
        audio_key="clipforge/uploads/j/audio.m4a",
    )
    a, _ = source.url_for_span(10.0)
    b, _ = source.url_for_span(20.0)
    assert a.startswith("https://bucket.example/") and b.startswith("https://bucket.example/")


# --- a capture interrupted by a host restart keeps what already landed -----
#
# Sections are uploaded to deterministic keys, so re-downloading the video from
# 0:00 after every restart is wasted work. On a host that bounces every few
# minutes a nine-section source never finished at all.


def test_a_capture_reuses_sections_already_in_the_bucket(monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace

    from app import objectstore

    uploaded: dict[str, int] = {
        source_stream.section_key("clipforge", "job-1", 0): 1000,
        source_stream.section_key("clipforge", "job-1", 1): 1001,
    }
    calls: list[str] = []

    monkeypatch.setattr(
        objectstore, "config_from_env",
        lambda: SimpleNamespace(prefix="clipforge"),
    )
    monkeypatch.setattr(
        objectstore, "head",
        lambda key, *a, **k: ({"size": uploaded[key]} if key in uploaded else None),
    )

    def fake_upload(path, key, *a, **k):
        uploaded[key] = path.stat().st_size
        return True

    monkeypatch.setattr(objectstore, "upload_file", fake_upload)
    monkeypatch.setattr(
        source_stream, "_split_if_big",
        lambda local, target, seconds: [(local, seconds)],
    )

    def fake_run(cmd, **kw):
        out = cmd[cmd.index("-o") + 1]
        Path(out).write_bytes(b"x" * 512)
        calls.append(cmd[cmd.index("--download-sections") + 1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(source_stream.subprocess, "run", fake_run)

    sections = source_stream._sections(
        "https://youtu.be/x", "job-1", total_sec=1237.0,
        settings=SimpleNamespace(),
    )

    assert len(sections) == 17, sections
    assert len(calls) == 15, f"re-downloaded {len(calls)} sections instead of 15"
    assert "*0-75" not in calls, "section 1 was downloaded again"
    assert sections[0]["key"].endswith("sec_000.mp4")
    assert sections[0]["bytes"] == 1000
    assert [s["start"] for s in sections] == [i * 75.0 for i in range(17)]
    assert sections[-1]["end"] == 1237.0, "the last span must end at the real duration"


def test_section_keys_are_stable():
    assert source_stream.section_key("p", "j", 3) == "p/uploads/j/sections/sec_003.mp4"
    assert source_stream.section_key("p", "j", 3) == source_stream.section_key("p", "j", 3)


def test_a_known_duration_beats_an_empty_probe(monkeypatch):
    """The provider probe can fail; the job's own duration must still size it.

    With `total_sec` at 0 the section planner used the unknown-length default
    of 48 x 150 s, so a 20:37 source captured ten times the needed sections and
    kept going past the end of the video.
    """
    from app import source_stream

    assert source_stream._section_count(0.0) == source_stream.UNKNOWN_DURATION_SECTIONS
    assert source_stream._section_count(1237.0) == 17


def test_the_multipart_attempt_stops_at_the_plan_cap(monkeypatch):
    """Do not download a whole video for an upload that cannot succeed.

    Filebase's free plan refuses multipart video above 25 MB. The old code
    streamed the entire source (470 MB on the observed 17-minute job) and only
    learned that at completion, then downloaded it again as 22 sections. The
    cap must abort on the first part over the limit so the fallback is quick.
    """
    from types import SimpleNamespace

    fake = FakeBucket()
    monkeypatch.setattr(
        objectstore, "config_from_env",
        lambda *a, **k: SimpleNamespace(enabled=True, bucket="b"),
    )
    monkeypatch.setattr(objectstore, "client", lambda *a, **k: fake)

    stream = io.BytesIO(b"x" * (10 * source_stream.PART_BYTES))
    with pytest.raises(source_stream.StreamError) as caught:
        source_stream._upload_pipe(
            stream, "clipforge/uploads/j/video.mp4", "video/mp4",
            max_bytes=source_stream.MAX_VIDEO_OBJECT_BYTES,
        )

    assert "25MB" in str(caught.value)
    uploaded = sum(len(body) for body in next(iter(fake.parts.values())).values())
    assert uploaded <= source_stream.MAX_VIDEO_OBJECT_BYTES + source_stream.PART_BYTES
    assert fake.aborted == ["clipforge/uploads/j/video.mp4"]


def test_a_clip_straddling_a_section_boundary_is_joined_not_failed(tmp_path, monkeypatch):
    """A moment 1 s before a section end and 1 s after is a valid clip.

    Detection works on the source timeline, so this happens on every long
    source. The two objects are already in the bucket; failing the candidate
    threw away a good clip. Join them with a stream copy and seek into it.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required to join real sections")
    sections = []
    for i in range(2):
        part = tmp_path / f"sec{i}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1.5:r=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(part),
            ],
            check=True, capture_output=True,
        )
        sections.append({"key": str(part), "start": i * 1.5, "end": (i + 1) * 1.5})

    monkeypatch.setattr(objectstore, "presign", lambda key, expires=900: key)
    src = source_stream.RemoteSource(
        "job1", video_key="v", audio_key="a", method="sections", sections=sections,
    )

    joined, offset = src.span_file(1.0, 2.6, tmp_path / "joined.mp4")

    assert joined.exists() and joined.stat().st_size > 0
    assert offset == pytest.approx(1.0)
    from app import video

    assert float(video.probe(joined)["duration"]) >= 2.9
    assert list(tmp_path.glob(".join-*.txt")) == [], "the concat list must be cleaned up"
