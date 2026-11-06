"""The memory budget: cgroup bytes in, MB out, and a plan that fits.

The failure these cover: `mkl_malloc: failed to allocate memory` on a 2 GiB
container whose process believed it had the host's 3 GiB. A budget that is read
from the CGROUP, keeps headroom for the API, and is re-checked before every
chunk is the difference between "lighter attempt" and "dead job".
"""

from __future__ import annotations

import pytest

from app import membudget
from app.memguard import MB


def test_budget_reads_the_cgroup_limit_not_the_host_total():
    b = membudget.budget_from(limit_bytes=2048 * MB, used_bytes=1000 * MB, available_bytes=None)
    assert b.source == "cgroup"
    assert b.limit_mb == 2048
    assert b.used_mb == 1000
    assert b.available_mb == 1048
    assert b.usable_mb == 1048 - membudget.HEADROOM_MB


def test_headroom_is_reserved_for_the_api_and_never_handed_to_asr():
    b = membudget.budget_from(2048 * MB, 1400 * MB, None, headroom_mb=320)
    assert b.usable_mb == 328
    assert b.fits(300) is True
    assert b.fits(400) is False
    assert b.shortfall_mb(400) == pytest.approx(72)


def test_an_unlimited_cgroup_falls_back_to_memavailable():
    b = membudget.budget_from(None, None, available_bytes=1500 * MB)
    assert b.source == "proc"
    assert b.limit_mb is None
    assert b.usable_mb == 1500 - membudget.HEADROOM_MB


def test_unreadable_memory_never_claims_a_shortage():
    b = membudget.budget_from(None, None, None)
    assert b.source == "unknown"
    assert b.available_mb == 0


def test_requirement_grows_with_the_window_and_the_model():
    small_60 = membudget.asr_requirement_mb("small", 60)
    small_120 = membudget.asr_requirement_mb("small", 120)
    base_60 = membudget.asr_requirement_mb("base", 60)
    assert small_120 > small_60
    assert base_60 < small_60
    # 900 MB is the int8 `small` floor from memguard; a 60 s chunk adds windows.
    assert small_60 >= 900


def test_plan_picks_the_longest_chunk_that_fits():
    roomy = membudget.budget_from(4096 * MB, 500 * MB, None)      # ~3.2 GB usable
    tight = membudget.budget_from(2048 * MB, 1200 * MB, None)     # ~528 MB usable
    assert membudget.plan_chunk_seconds("small", roomy, preferred=120) == 120
    assert membudget.plan_chunk_seconds("small", roomy) == 60
    # 60 s does not fit ~528 MB, and neither does 30 s: the floor is returned and
    # the ladder's next rungs (beam 1, VAD off, smaller model) take over.
    assert membudget.plan_chunk_seconds("small", tight) == membudget.MIN_CHUNK_SEC


def test_plan_never_returns_less_than_the_floor():
    impossible = membudget.budget_from(2048 * MB, 2000 * MB, None)
    assert membudget.plan_chunk_seconds("small", impossible) == membudget.MIN_CHUNK_SEC


def test_describe_is_readable_and_states_every_unit():
    b = membudget.budget_from(2048 * MB, 1024 * MB, None)
    text = b.describe()
    assert "limit=2048 MB" in text and "used=1024 MB" in text and "usable=" in text


def test_live_budget_is_never_negative():
    b = membudget.current_budget()
    assert b.usable_mb >= 0
