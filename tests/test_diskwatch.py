"""The disk ledger has to be honest, because the job's self-check trusts it."""

from __future__ import annotations

from pathlib import Path

from app import diskwatch


def _ledger(tmp_path: Path, roots: list[Path], cap: int = diskwatch.SCRATCH_CAP_BYTES):
    return diskwatch.JobLedger(
        "job_disk", roots, data_dir=tmp_path, cap_bytes=cap, sample_sec=0.05,
    )


def test_a_stage_records_its_delta_and_a_log_line(tmp_path):
    lines: list[str] = []
    work = tmp_path / "work"
    work.mkdir()
    ledger = _ledger(tmp_path, [work])
    ledger.on_log = lines.append

    with ledger.stage("fetch"):
        (work / "source.mp4").write_bytes(b"x" * (12 * 1024 * 1024))

    row = ledger.stages[-1]
    assert row["stage"] == "fetch"
    assert row["jobMb"] == 12.0
    assert row["deltaMb"] is not None and row["deltaMb"] <= 0
    assert any("disk delta:" in line for line in lines)


def test_crossing_the_cap_stops_the_stage_and_names_the_file(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    ledger = _ledger(tmp_path, [work], cap=1024 * 1024)
    big = work / "clip_000.mp4"
    big.write_bytes(b"x" * (2 * 1024 * 1024))

    try:
        ledger.check("export clip 1")
    except diskwatch.ScratchCapExceeded as exc:
        assert "clip_000.mp4" in str(exc)
        assert exc.path.endswith("clip_000.mp4")
        assert "over the" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("the cap did not stop the stage")


def test_cleanup_frees_the_scratch_and_the_self_check_passes(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    ledger = _ledger(tmp_path, [work])
    with ledger.stage("fetch"):
        (work / "source.mp4").write_bytes(b"x" * (8 * 1024 * 1024))
        freed = diskwatch.cleanup([work])
    assert freed >= 8 * 1024 * 1024
    assert not work.exists()
    ok, why = ledger.recovered()
    assert ok, why


def test_a_leftover_over_the_tolerance_fails_the_self_check(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    ledger = _ledger(tmp_path, [work])
    (work / "leftover.mp4").write_bytes(b"x" * (120 * 1024 * 1024))

    ok, why = ledger.recovered()
    assert not ok
    assert "leftover.mp4" in why


def test_mb_is_the_only_conversion_bytes_in_mb_out():
    assert diskwatch.mb(0) == 0.0
    assert diskwatch.mb(1024 * 1024) == 1.0
    assert diskwatch.mb(1536 * 1024) == 1.5
    assert diskwatch.mb(None) is None
