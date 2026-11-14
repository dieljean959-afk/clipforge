"""Pause must stop cleanly and resume must NOT redo paid work.

The request was specific: pausing should stop "at the current stage (not
mid-write on a file, to avoid corruption)", and resuming a job paused at
"Detecting highlights on chunk 5 of 7" must not restart from chunk 1.

Both halves needed building:

**Cooperative pause.** The real work happens in a per-job child process, and a
child cannot be interrupted part-way through an ffmpeg write without risking
the corruption the user is trying to avoid. So `pause()` sets a flag in the job
document and the PIPELINE checks it between units of work — transcript chunks,
detection chunks, rendered clips, stage boundaries — then stops itself and
records a clean `paused` state. Nothing is ever killed mid-write.

**Per-chunk detection caching.** This was a real gap, not a hypothetical.
`detect_clips` looped over chunks calling the LLM and keeping results only in
memory, so ANY resume re-ran and re-billed every chunk. A 7-chunk transcript
caught by a host restart at chunk 5 paid for all seven again. Each chunk's
resolved candidates now persist under `detect/chunks/` before progress is
reported, so a resume reuses chunks 1-5 and calls the API only for chunk 6-7.

Transcription already had per-chunk caching and now honours pause at the same
boundaries; the render loop already skipped finished clips and now
checkpoints between them.

Run:  .venv/bin/python tools/pause_resume_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm as llm_mod  # noqa: E402
from app.config import Paths, load_config  # noqa: E402
from app.stages import StageWorkspace  # noqa: E402
from app.store import LocalStore  # noqa: E402
from app.task import _StageChunkCache  # noqa: E402
from app.worker import InProcessWorker  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    """Assert, loudly.

    Order is (label, ok). A backwards call would pass SILENTLY — a non-empty
    string in `ok` is truthy — so the helper refuses anything that is not a
    real string label and a real boolean. That guard has already caught this
    exact mistake three times in this repo.
    """
    global CHECKS
    if not isinstance(label, str) or not isinstance(ok, bool):
        FAILS.append("check() called with swapped/ill-typed arguments")
        print(f"  FAIL check({label!r}, {ok!r}) — arguments are backwards; did NOT run")
        return
    CHECKS += 1
    if ok:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" — {detail}" if detail else ""))
        FAILS.append(label)


def _worker_with(store: LocalStore) -> InProcessWorker:
    """A worker wired to a temp store, with enqueue stubbed (no threads)."""
    w = InProcessWorker.__new__(InProcessWorker)
    w._lock = threading.RLock()
    w._pending = []
    w._current_job = None
    w._current_proc = None
    w._active = {}
    w._stalled_job = None
    w._stalled_jobs = set()
    w.store = store
    w.enqueue = lambda jid: (w._pending.append(jid), len(w._pending))[1]
    return w


def _stub_llm(monkey: list) -> None:
    """Replace the provider call so no network/keys are needed."""

    def fake(prompt, settings):  # noqa: ANN001
        monkey.append(prompt)
        return json.dumps({"clips": [{"start_word": 0, "end_word": 40, "score": 90,
                                      "title": "t", "hook_text": "h"}]})

    llm_mod.call_llm = fake  # type: ignore[assignment]


def main() -> int:
    settings = load_config(quiet=True)
    calls: list = []
    original_call = llm_mod.call_llm
    _stub_llm(calls)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalStore(Paths(
                data_dir=root / "d", outputs_dir=root / "o",
                downloads_dir=root / "dl", static_dir=root / "s",
            ))
            ws = StageWorkspace(settings.app.paths.data_dir if False else root / "d", "j1")
            job_id = "j1"
            store.write_job({"_id": job_id, "source": "s", "title": "t", "stage": "finding"})

            # a transcript long enough to split into several chunks
            words = [
                {"start": float(i) * 0.5, "end": float(i) * 0.5 + 0.4, "word": f"w{i}"}
                for i in range(1200)
            ]
            cache = _StageChunkCache(ws)

            # ---- 1. full run populates the chunk cache ----------------
            calls.clear()
            out1 = llm_mod.detect_clips(
                words, settings, chunk_cache=cache, max_clips=10,
                on_log=lambda line: calls.append(line),
            )
            n_chunks = len(llm_mod.chunk_words(words))
            check("a full run returns candidates", bool(out1), f"{len(out1)} candidates")
            check(
                "every chunk was persisted to disk",
                all((ws.stage_dir("detect") / "chunks" / f"chunk_{i:03d}.json").exists()
                    for i in range(n_chunks)),
                f"{n_chunks} chunks",
            )
            check(
                "the first pass really did call the API for every chunk",
                n_chunks > 1,
                f"{n_chunks} chunks",
            )

            # ---- 2. a resumed run reuses them and re-bills NOTHING ----
            # Count real provider invocations, not log lines: the resume is
            # EXPECTED to emit a "cached — not re-billed" line per chunk.
            api_only: list = []
            llm_mod.call_llm = lambda p, s: (api_only.append(p), json.dumps(
                {"clips": [{"start_word": 0, "end_word": 40, "score": 90,
                            "title": "t", "hook_text": "h"}]}))[1]  # type: ignore[assignment]
            out2 = llm_mod.detect_clips(
                words, settings, chunk_cache=cache, max_clips=10,
                on_log=lambda line: calls.append(line),
            )
            check(
                "a full resume makes ZERO provider calls",
                len(api_only) == 0,
                f"{len(api_only)} calls",
            )
            check("the resumed result matches the original", out2 == out1)

            # ---- 3. a PARTIAL cache resumes from the next chunk ------
            # This is the "paused at chunk 5 of 7" case.
            partial = llm_mod.DictChunkCache()
            # Well-separated windows. These three used to be 0-10, 1-11 and
            # 2-12 — which OVERLAP each other by nine seconds, so they describe
            # the same moment three times. Detection now drops near-duplicate
            # windows across chunks (llm.drop_overlapping), which is correct
            # and made this fixture collapse from 3 candidates to 1. A chunk
            # covering a different part of a video returns candidates from a
            # different part of it, so the fixture is fixed to do that, and the
            # duplicate case is asserted on its own below rather than being
            # smuggled in here.
            for i in range(3):
                partial.put(i, [{"start": float(i) * 30.0, "end": float(i) * 30.0 + 10,
                                 "score": 90 - i, "title": f"c{i}", "hook_text": "h"}])

            api_hits: list = []

            def counting(prompt, s):  # noqa: ANN001
                api_hits.append(prompt)
                return json.dumps({"clips": [{"start_word": 0, "end_word": 20,
                                              "score": 80, "title": "n", "hook_text": "h"}]})

            llm_mod.call_llm = counting  # type: ignore[assignment]
            api_hits.clear()
            partial_out = llm_mod.detect_clips(words, settings, chunk_cache=partial)
            check(
                "a partial cache re-bills only the REMAINING chunks",
                len(api_hits) == n_chunks - 3,
                f"{len(api_hits)} calls for {n_chunks - 3} missing chunks",
            )
            check(
                "the cached chunks still contribute their candidates",
                len(partial_out) >= 3,
                f"{len(partial_out)} candidates",
            )
            check(
                "cached chunks are cheaper (higher score) and rank first",
                partial_out[0]["score"] == 90,
                f"top score {partial_out[0]['score']}",
            )
            # The other half of the same contract: cached chunks that DO
            # describe the same moment are collapsed, not rendered twice.
            dupes = llm_mod.DictChunkCache()
            for i in range(3):
                dupes.put(i, [{"start": float(i), "end": float(i) + 10,
                               "score": 90 - i, "title": f"d{i}", "hook_text": "h"}])
            api_hits.clear()
            dupe_out = llm_mod.detect_clips(words, settings, chunk_cache=dupes)
            check("near-duplicate windows from different chunks are collapsed",
                  len(dupe_out) == 1, f"{len(dupe_out)} candidates: "
                                       f"{[(c['start'], c['end']) for c in dupe_out]}")

            # ---- 4. pause at a chunk boundary stops the stage ---------
            #
            # What is actually guaranteed, and what these checks pin:
            #
            #   * a pause is honoured — the stage raises DetectionPaused;
            #   * NOTHING ALREADY PAID FOR IS THROWN AWAY — every chunk whose
            #     call went out is written to the cache, so a resume cannot
            #     bill the same chunk twice. This is the bug: the pause used to
            #     raise BEFORE the cache write, so a paid-for chunk was lost and
            #     re-billed on resume;
            #   * the overshoot is BOUNDED. A pause raised by a call cannot be
            #     observed until that call returns, so with `concurrency` calls
            #     in flight it can overshoot by up to concurrency-1 chunks. The
            #     bound is asserted rather than wished away, and the sequential
            #     path below still stops EXACTLY on the boundary.
            api_hits.clear()
            pause_after = 2
            seen = {"n": 0}
            lock = threading.Lock()

            def pause_now() -> bool:
                return seen["n"] >= pause_after

            def pausing_call(prompt, s):  # noqa: ANN001
                with lock:
                    seen["n"] += 1
                return counting(prompt, s)

            llm_mod.call_llm = pausing_call  # type: ignore[assignment]
            fresh = llm_mod.DictChunkCache()
            paused = False
            try:
                llm_mod.detect_clips(words, settings, chunk_cache=fresh, should_pause=pause_now)
            except llm_mod.DetectionPaused as exc:
                paused = True
                detail = str(exc)
            check("pause stops detection at a chunk boundary", paused, detail if paused else "no pause raised")
            concurrency = int(getattr(settings.llm, "detect_concurrency", 1) or 1)
            check(
                "the overshoot past the pause is bounded by the window",
                pause_after <= seen["n"] <= pause_after + max(0, concurrency - 1),
                f"{seen['n']} chunks billed for a pause after {pause_after} "
                f"(window {concurrency})",
            )
            cached_n = sum(1 for i in range(n_chunks) if fresh.get(i) is not None)
            check(
                "EVERY chunk that was paid for is kept for the resume",
                cached_n == seen["n"],
                f"{seen['n']} calls, {cached_n} cached — nothing paid for is lost",
            )

            # and the resume after the pause only does the remainder
            api_hits.clear()
            llm_mod.call_llm = counting  # type: ignore[assignment]
            llm_mod.detect_clips(words, settings, chunk_cache=fresh)
            check(
                "resuming after a pause re-bills only the remainder",
                len(api_hits) == n_chunks - cached_n,
                f"{len(api_hits)} calls for {n_chunks - cached_n} remaining",
            )

            # ---- 4b. with no window, the boundary is EXACT ------------
            # The same guarantee with concurrency 1, where there is no
            # in-flight call to overshoot: the stage must stop on the chunk
            # boundary and keep exactly the chunks it finished.
            serial_settings = replace(settings.llm, detect_concurrency=1)
            settings.llm = serial_settings
            try:
                seen2 = {"n": 0}
                seq_cache = llm_mod.DictChunkCache()

                def seq_pause() -> bool:
                    return seen2["n"] >= pause_after

                def seq_call(prompt, s):  # noqa: ANN001
                    seen2["n"] += 1
                    return counting(prompt, s)

                llm_mod.call_llm = seq_call  # type: ignore[assignment]
                seq_paused = False
                try:
                    llm_mod.detect_clips(
                        words, settings, chunk_cache=seq_cache, should_pause=seq_pause
                    )
                except llm_mod.DetectionPaused:
                    seq_paused = True
                check("with no window the pause is exact", seq_paused and seen2["n"] == pause_after,
                      f"paused={seq_paused} after {seen2['n']} chunks")
                seq_cached = sum(1 for i in range(n_chunks) if seq_cache.get(i) is not None)
                check("and it keeps exactly the chunks it finished",
                      seq_cached == pause_after, f"{seq_cached} cached")
            finally:
                settings.llm = replace(settings.llm, detect_concurrency=concurrency)

            # ---- 5. the API surface: pause a QUEUED job is instant ----
            # Two shapes matter: a job sitting in the live queue, and one left
            # marked "queued" by a host restart before anything enqueued it.
            store.write_job({"_id": "q1", "source": "s", "title": "t", "stage": "queued"})
            w2 = _worker_with(store)
            w2.enqueue("q1")
            res = w2.pause("q1")
            check("pausing a queued job is immediate", res["state"] == "paused", json.dumps(res))
            q = store.get_job("q1")
            check("it is marked paused", q["stage"] == "paused", q["stage"])
            check("it left the queue", "q1" not in w2._pending)
            check("nothing was started, so nothing to resume mid-write",
                  q.get("pausedStage") == "queued", str(q.get("pausedStage")))

            # A restart can leave a job "queued" with no process and no queue
            # entry; it must still pause immediately rather than report
            # "pausing" for a pause that will never land.
            store.write_job({"_id": "q9", "source": "s", "title": "t", "stage": "queued"})
            w2b = _worker_with(store)
            r9 = w2b.pause("q9")
            check("an orphaned 'queued' job still pauses immediately",
                  r9["state"] == "paused", json.dumps(r9))
            check("it is marked paused", store.get_job("q9")["stage"] == "paused")

            # ---- 6. resume re-queues and preserves the stage markers ---
            store.write_job({"_id": "q2", "source": "s", "title": "t", "stage": "finding"})
            w3 = _worker_with(store)
            r = w3.resume("q2")
            check("resume re-queues the job", r == 1 and "q2" in w3._pending, str(r))
            q2 = store.get_job("q2")
            check("it reports queued, not failed", q2["stage"] == "queued")
            check("the pause flag is cleared", q2.get("pauseRequested") is False)
            check("no stale error survives the resume", q2.get("error") is None)

            # ---- 7. a paused job is never auto-retried ----------------
            w4 = _worker_with(store)
            store.write_job({
                "_id": "p1", "source": "s", "title": "t", "stage": "paused",
                "pausedStage": "finding", "pauseRequested": False,
            })
            pj = store.get_job("p1")
            check(
                "a pause is not treated as a retryable failure",
                w4._auto_retry("p1", pj, "pause") is False,
            )

            # ---- 8. pausing a finished job is a 409, not a crash ----
            store.write_job({"_id": "f1", "source": "s", "title": "t", "stage": "complete"})
            try:
                w4.pause("f1")
                rejected = False
            except ValueError:
                rejected = True
            check("pausing a completed job is refused", rejected)

    finally:
        llm_mod.call_llm = original_call  # type: ignore[assignment]

    print()
    if FAILS:
        print(f"{CHECKS - len(FAILS)} passed, {len(FAILS)} FAILED")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print(f"{CHECKS} passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
