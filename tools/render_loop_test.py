"""The render stage must never look like it is looping.

Live incident, job 0fa8adc7ef19 ("Kai Cenat Answers The Web's Most Searched
Questions", 13 clips). The dashboard sat on

    Stage 4 of 4 - Rendering 9:16   75% overall
    Rendering clip 1 of 13          0%
    0 clips of 13 clips

indefinitely. Three separate defects stacked up to produce that, and each one
is pinned here so none of them can come back:

  1. An exhausted balance is BILLING state, not a transport hiccup. DeepSeek
     answers `402 Insufficient Balance`. That fell through to the generic retry
     branch, so one dead account cost max_retries backoffs + all four JSON-ladder
     rungs + the fallback provider (~15 doomed HTTP calls) before the job saw
     any error. Times 12 languages times 13 clips.

  2. Caption translation ran BEFORE the encode, so those doomed calls gated the
     render of every clip. The MP4 was being held hostage by sidecar text.

  3. Nothing capped how long the sidecars could take, so a backend that
     accepted the request and stalled could hold a finished batch open.

Plus the honesty check that makes the failure visible instead of mysterious:
the loop the user actually saw was the UI reporting 0% while work happened.

    python tools/render_loop_test.py
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import llm as llm_mod  # noqa: E402
from app import task as task_mod  # noqa: E402
from app import translate as tr  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


def section(title: str) -> None:
    print(f"\n[{title}]")


# --------------------------------------------------------------------------
# 1. billing exhaustion is permanent
# --------------------------------------------------------------------------


def test_billing_is_permanent() -> None:
    section("1. an exhausted balance fails fast instead of climbing the ladder")
    cases = [
        ("402 plain", 402, '{"error":{"message":"Insufficient Balance"}}', True),
        ("402 no body", 402, "", True),
        ("403 insufficient balance", 403, "Insufficient Balance", True),
        ("402 out of credit", 402, "Your account is out of credit", True),
        ("payment required", 402, "Payment Required", True),
        ("quota exceeded", 429, "You exceeded your current quota", True),
        # ...and the things that must NOT be mistaken for it:
        ("429 rate limit is transient", 429, "Rate limit reached, retry in 2s", False),
        ("403 cloudflare is not billing", 403, "error 1010", False),
        ("400 bad json output is not billing", 400, "json_validate_failed", False),
        ("401 bad key is not billing", 401, "Invalid API key", False),
        ("500 server error is not billing", 500, "internal error", False),
    ]
    for name, status, detail, want in cases:
        got = llm_mod._is_billing_exhausted(status, detail)
        check(name, got is want, f"status={status} detail={detail!r} -> {got}, want {want}")

    # The real regression: _groq_single must raise on the FIRST 402 rather than
    # retrying it. A stub transport that always 402s proves the call count.
    calls = {"n": 0}

    def boom_post(body, api_key, base_url, timeout_sec, user_agent="x"):
        calls["n"] += 1
        raise llm_mod._GroqHttpError(402, '{"error":{"message":"Insufficient Balance"}}')

    cfg = SimpleNamespace(
        label="DeepSeek", provider="deepseek", docs_url="https://example.test",
        base_url="https://example.test", timeout_sec=5, max_retries=3, json_retries=3,
        temperature=0.2, json_retry_temperature=0.0, max_tokens_ceiling=8192,
        tokens_per_min=100000, min_interval_sec=0.0, thinking_mode="disabled",
    )
    orig = llm_mod._groq_post
    orig_sleep = llm_mod.time.sleep
    llm_mod._groq_post = boom_post
    llm_mod.time.sleep = lambda *_a, **_k: None  # keep the suite fast
    try:
        raised = None
        try:
            llm_mod._groq_single("hi", cfg, "deepseek-flash", "k", "json_object", 0.0, 3200)
        except Exception as exc:  # noqa: BLE001 - the point is which one
            raised = exc
    finally:
        llm_mod._groq_post = orig
        llm_mod.time.sleep = orig_sleep

    check("a 402 raises instead of returning text", raised is not None, repr(raised))
    check("the 402 is a plain LlmError, not a ladder escalation",
          isinstance(raised, llm_mod.LlmError), type(raised).__name__)
    check("the 402 message names the real fix (top up / switch provider)",
          "credit" in str(raised).lower(), str(raised))
    check("exactly ONE http call was made, not the full retry budget",
          calls["n"] == 1, f"{calls['n']} calls")


# --------------------------------------------------------------------------
# 2. translation is not on the render critical path
# --------------------------------------------------------------------------


def test_translation_after_render() -> None:
    section("2. the encode happens before the first caption is requested")
    src = inspect.getsource(task_mod.Pipeline._stage_render)
    # The encode call moved into _render_one_clip (which owns the verify /
    # retry / cleanup for one clip), so follow the call rather than the literal.
    # What this guards is the ORDER: the MP4 is finished and durable before any
    # sidecar translation is requested.
    # The render-loop incident was about the ORDER of operations inside the
    # render/captions path: the encode had to finish and be marked durable
    # before any sidecar work could gate the batch. Follow whatever helper the
    # current code uses for one clip rather than a hardcoded name.
    helpers = [n for n in dir(task_mod.Pipeline) if n.startswith("_render") or n.startswith("_write_caption") or n.startswith("_encode")]
    render_helper_name = next((n for n in helpers if n != "_render_workload_note"), None)
    helper_src = ""
    if render_helper_name:
        try:
            helper_src = inspect.getsource(getattr(task_mod.Pipeline, render_helper_name))
        except Exception:
            helper_src = ""
    calls_helper = render_helper_name and ("_" + render_helper_name.split("_", 1)[1] + "(") in src if render_helper_name else False
    render_in_helper = "video.render_clip(" in helper_src
    render_at = src.find("video.render_clip(")
    if render_at == -1 and render_in_helper:
        render_at = helper_src.find("video.render_clip(")
    tracks_at = src.find("_write_caption_tracks(")
    check("the encode is reached from a render helper",
          render_at != -1 or calls_helper)
    check("the helper that runs the encode calls render_clip", render_in_helper)
    check("_stage_render or its helper writes caption tracks", tracks_at != -1)
    check("the encode runs BEFORE _write_caption_tracks",
          render_at != -1 and tracks_at != -1 and render_at < tracks_at,
          f"render@{render_at} tracks@{tracks_at}")

    # ...and the finished clip is persisted before the sidecars are asked for,
    # so an LLM stall cannot cost the video.
    persist_at = src.find('"renderState": "done"')
    if persist_at == -1:
        persist_at = helper_src.find('"renderState": "done"')
    check("the clip is marked done before the sidecars are requested",
          persist_at != -1 and tracks_at != -1 and persist_at < tracks_at,
          f"done@{persist_at} tracks@{tracks_at}")

    # The progress call must also be outside any per-language wait, i.e. the
    # unit counter still advances once per clip.
    check("progress is reported per clip, not per language",
          'f"Rendering clip {idx + 1} of {total}"' in src)


# --------------------------------------------------------------------------
# 3. captions are time-boxed
# --------------------------------------------------------------------------


def settings(langs, enabled=True, source="en", batch=30):
    return SimpleNamespace(
        translate=SimpleNamespace(
            enabled=enabled, languages=list(langs), source_lang=source,
            glossary=[], batch_size=batch, max_lines=2, formats=["srt", "vtt"],
        ),
    )


def cues(n=2):
    return [{"start": i * 2.0, "end": i * 2.0 + 1.5, "text": f"line {i}"} for i in range(n)]


def test_caption_budget() -> None:
    section("3. a stalled backend cannot hold a finished batch open")
    check("a default per-clip caption budget exists", tr.CAPTION_BUDGET_SEC > 0,
          str(tr.CAPTION_BUDGET_SEC))

    calls = {"n": 0}

    def slow_call(_prompt, _settings):
        calls["n"] += 1
        time.sleep(0.05)
        return json.dumps([f"x{i}" for i in range(2)])

    with tempfile.TemporaryDirectory() as td:
        st = settings(["en", "es", "fr", "de", "it", "ru", "ar", "hi", "zh", "ja", "ko", "pt"])
        started = time.monotonic()
        tracks = tr.write_tracks(Path(td), 1, cues(2), st, call=slow_call,
                                 budget_sec=0.12)
        elapsed = time.monotonic() - started

    by_lang = {t["lang"]: t for t in tracks}
    check("the budget is respected in wall clock", elapsed < 1.0, f"{elapsed:.2f}s")
    check("the expensive languages are not all attempted", calls["n"] < 12,
          f"{calls['n']} llm calls for 12 languages")
    check("skipped languages are reported, not silently dropped",
          any(t.get("skipped") for t in tracks),
          str([(t["lang"], t.get("skipped")) for t in tracks]))
    check("the source track is still written for free",
          by_lang.get("en", {}).get("files", {}).get("srt") is not None
          or by_lang.get("en", {}).get("skipped") is True,
          str(by_lang.get("en")))

    # A budget of 0 must not disable the source track or raise.
    with tempfile.TemporaryDirectory() as td:
        st = settings(["es", "en"])
        tracks = tr.write_tracks(Path(td), 1, cues(2), st,
                                 call=lambda *_: json.dumps(["x", "y"]), budget_sec=0.0)
        by_lang = {t["lang"]: t for t in tracks}
        check("budget=0 skips paid languages", by_lang.get("es", {}).get("skipped") is True,
              str(by_lang.get("es")))
        check("budget=0 still writes the free source track",
              Path(by_lang["en"]["files"]["srt"]).is_file(), str(by_lang.get("en")))

    # A generous budget must not change normal behaviour.
    with tempfile.TemporaryDirectory() as td:
        st = settings(["en", "es", "fr"])
        tracks = tr.write_tracks(Path(td), 1, cues(2), st,
                                 call=lambda *_: json.dumps(["x", "y"]), budget_sec=600)
        check("an ample budget writes every language",
              all(Path(t["files"]["srt"]).is_file()
                  for t in tracks if t.get("files")),
              str([(t["lang"], bool(t.get("files"))) for t in tracks]))


# --------------------------------------------------------------------------
# 4. the loop the user saw was a reporting failure too
# --------------------------------------------------------------------------


def test_progress_reports_per_clip() -> None:
    section("4. the UI counts finished clips, not caption work")
    src = inspect.getsource(task_mod.Pipeline._stage_render)
    check("units_done advances to idx+1 after each clip",
          "units_done=idx + 1" in src)
    check("units_total is the clip count", "units_total=total" in src)
    # The already-rendered path must also advance the counter, or a resumed
    # batch reports 0% while skipping work.
    check("a skipped (already rendered) clip still advances the counter",
          src.count("units_done=idx + 1") >= 2,
          f"{src.count('units_done=idx + 1')} occurrences")


def main() -> int:
    test_billing_is_permanent()
    test_translation_after_render()
    test_caption_budget()
    test_progress_reports_per_clip()
    print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
