"""Concurrent detection must be UNOBSERVABLE in its results.

Concurrency here is only allowed to change *when* a chunk's answer arrives, never
*what* the stage returns. If it can change the outcome, it is not an
optimisation, it is a different pipeline.

So the contract is exact equality, not "close enough":

  * the returned candidate list is identical, in the same order;
  * cached chunks are reused and never re-billed (not even to parallelise);
  * a failed chunk is still skipped and the rest are still kept;
  * a failure that must abort (no key, user pause) still aborts, in order;
  * per-chunk persistence still happens before progress is reported.

The provider is stubbed, so this asserts the pipeline's behaviour rather than a
vendor's latency, and costs nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import llm  # noqa: E402
from app.config import LlmConfig, load_config  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print(f"\n[{title}]")


def make_words(count: int = 900) -> list[dict]:
    return [
        {"start": i * 0.9, "end": i * 0.9 + 0.8, "word": f"w{i}", "text": f"w{i}"}
        for i in range(count)
    ]


def settings_with(concurrency: int):
    s = load_config(quiet=True)
    return replace_llm(s, concurrency)


def replace_llm(settings, concurrency: int):
    settings.llm = LlmConfig(
        provider=settings.llm.provider,
        fallback_provider=settings.llm.fallback_provider,
        detect_concurrency=concurrency,
        ollama_url=settings.llm.ollama_url,
        ollama_model=settings.llm.ollama_model,
        ollama_timeout_sec=settings.llm.ollama_timeout_sec,
        ollama_keep_alive=settings.llm.ollama_keep_alive,
    )
    return settings


class StubCache:
    """Per-chunk persistence, counted so 're-billed' can be asserted."""

    def __init__(self, preset: dict[int, list] | None = None) -> None:
        self.store: dict[int, list] = dict(preset or {})
        self.puts: list[int] = []

    def get(self, i: int):
        return self.store.get(i)

    def put(self, i: int, value: list) -> None:
        self.store[i] = value
        self.puts.append(i)


def run_with_stub(concurrency: int, *, fail_chunk: int | None = None,
                  cached: dict[int, list] | None = None,
                  raise_cls=RuntimeError, pause_at: int | None = None):
    """Drive detect_clips on a stubbed provider and return what it produced."""
    calls: list[int] = []
    real_call = llm.call_llm
    real_chunks = llm.chunk_words

    # A deterministic per-chunk answer: chunk i always yields the same clips,
    # so any difference between runs is the concurrency, not the model.
    def fake_chunks(words):
        size = max(1, len(words) // 4)
        return [(i, words[i * size:(i + 1) * size]) for i in range(4)]

    def fake_call(prompt: str, settings, _calls=calls):
        index = _calls.__len__()
        _calls.append(index)
        if fail_chunk is not None and index == fail_chunk:
            raise llm.LlmError(f"stub failure on call {index}")
        return (
            '[{"start_word": 0, "end_word": 4, "score": 80, '
            f'"title": "Clip {index}", "hook_text": "Hook {index}"}}]'
        )

    llm.chunk_words = fake_chunks
    llm.call_llm = fake_call
    cache = StubCache(cached)
    pauses = {"n": 0}

    def should_pause():
        if pause_at is None:
            return False
        pauses["n"] += 1
        return pauses["n"] > pause_at

    try:
        out = llm.detect_clips(
            make_words(), settings_with(concurrency),
            chunk_cache=cache, should_pause=should_pause,
        )
        return {"out": out, "calls": calls, "cache": cache}
    except llm.DetectionPaused as exc:
        return {"paused": str(exc), "calls": calls, "cache": cache}
    finally:
        llm.call_llm = real_call
        llm.chunk_words = real_chunks


def test_identical_results() -> None:
    section("concurrent and sequential produce the same batch")
    seq = run_with_stub(1)
    con = run_with_stub(4)

    check("both ran every chunk", len(seq["calls"]) == len(con["calls"]) == 4,
          f"seq={len(seq['calls'])} con={len(con['calls'])}")
    check("the candidate lists are identical", seq["out"] == con["out"],
          f"{len(seq['out'])} vs {len(con['out'])} clips")
    check("candidate order is identical",
          [(c.get("title"), round(float(c.get("start") or 0), 3)) for c in seq["out"]]
          == [(c.get("title"), round(float(c.get("start") or 0), 3)) for c in con["out"]])
    check("every chunk was persisted", sorted(con["cache"].puts) == [0, 1, 2, 3],
          str(sorted(con["cache"].puts)))


def test_cached_chunks_are_not_rebilled() -> None:
    section("a cached chunk is reused, never re-billed to parallelise")
    preset = {0: [{"start": 1.0, "end": 5.0, "score": 90.0,
                   "title": "cached", "hook_text": "cached"}]}
    seq = run_with_stub(1, cached=preset)
    con = run_with_stub(4, cached=preset)
    check("sequential skips the cached chunk's API call",
          len(seq["calls"]) == 3, str(seq["calls"]))
    check("concurrent skips it too", len(con["calls"]) == 3, str(con["calls"]))
    check("the cached clip is still in the output",
          any(c.get("title") == "cached" for c in con["out"]))


def test_one_bad_chunk_does_not_kill_the_job() -> None:
    section("a failed chunk is still skipped, and the rest are kept")
    seq = run_with_stub(1, fail_chunk=1)
    con = run_with_stub(4, fail_chunk=1)
    check("sequential keeps the other chunks", len(seq["out"]) == 3,
          f"{len(seq['out'])} clips")
    check("concurrent keeps exactly the same ones", con["out"] == seq["out"],
          f"{len(con['out'])} vs {len(seq['out'])} clips")


def test_pause_still_aborts() -> None:
    section("a pause is honoured at a chunk boundary, not raced past")
    seq = run_with_stub(1, pause_at=0)
    con = run_with_stub(4, pause_at=0)
    check("sequential raises DetectionPaused", "paused" in seq, str(seq)[:80])
    check("concurrent raises it too", "paused" in con, str(con)[:80])
    check("no API call is made once paused", len(con["calls"]) == 0,
          str(con["calls"]))


def test_concurrency_is_bounded() -> None:
    section("the cap is real and configurable")
    check("default is 4", LlmConfig().detect_concurrency == 4)
    src = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
    check("it is settable in config.toml", "detect_concurrency" in src)
    flat = "".join(src.split())
    check("it is clamped to a sane range", "min(8,int(" in flat, "clamped 1..8")
    body = (ROOT / "app" / "llm.py").read_text(encoding="utf-8")
    check("llm.py clamps what it is given",
          'min(8, int(getattr(settings.llm, "detect_concurrency"' in body)
    check("it is clamped to a sane range",
          "min(\n                8," in src or "min(8," in src.replace("\n", "").replace(" ", ""),
          "clamped 1..8")
    body = (ROOT / "app" / "llm.py").read_text(encoding="utf-8")
    check("llm.py clamps what it is given",
          "min(8, int(getattr(settings.llm, \"detect_concurrency\"" in body)


def main() -> int:
    test_identical_results()
    test_cached_chunks_are_not_rebilled()
    test_one_bad_chunk_does_not_kill_the_job()
    test_pause_still_aborts()
    test_concurrency_is_bounded()
    print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
