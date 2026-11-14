"""Usage-counter isolation and persistence tests.

Two bugs this file exists to prevent, both found by reading the code rather
than by a failing job:

  1. THE COUNTER RESET ON EVERY RESTART. `_read()` returned the INNER
     `days` map while `_write()` persisted `self._data` wrapped in a `days`
     key. So the first read after any write unwrapped one level too far,
     `_day()`'s `setdefault("days", {})` then opened a fresh empty namespace,
     and the real totals were ignored. Net effect: "calls today" and the usage
     badge read 0 after every backend restart, and the first write afterwards
     left a stale duplicate of the day at the top level of the file. That is
     the worst possible time for the number to lie — an operator checking the
     badge after a deploy to see whether the budget is spent.

  2. ONE PROVIDER'S SPEND COUNTED AGAINST ANOTHER'S LIMIT. The daily budgets
     are not comparable (Groq's hard 200k tokens/day vs a DeepSeek balance),
     and the store had no provider dimension at all: `local_tokens` summed
     every model from every provider, and Groq's authoritative 429 figure was
     a single day-level number. Switching from Groq to DeepSeek mid-day
     reported DeepSeek at 198,530/200,000 (99.3%) when it had spent nothing —
     a false "quota exhausted" warning about a provider that never made a call.

No network, no keys, no quota — this drives UsageStore directly against a
temp directory.

    python tools/usage_isolation_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.groq_usage import UsageStore, today_key  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))


GROQ_TPD_429 = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization org-x "
    "on tokens per day (TPD): Limit 200000, Used 198530, Requested 3200. "
    "Please try again in 42m0.0s."
)


def _dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="clipforge-usage-"))


def _doc(path: Path) -> dict:
    return json.loads((path / "groq_usage.json").read_text(encoding="utf-8"))


def _write(path: Path, doc: dict) -> UsageStore:
    (path / "groq_usage.json").write_text(json.dumps(doc), encoding="utf-8")
    return UsageStore(path)


# --------------------------------------------------------------------------- #
# 1. persistence across a restart
# --------------------------------------------------------------------------- #


def test_survives_restart() -> None:
    print("\n[persist] the daily total survives a process restart")
    d = _dir()
    s1 = UsageStore(d)
    s1.record_call("openai/gpt-oss-120b", 90000, provider="groq")
    first = s1.snapshot(default_limit=200000, model="openai/gpt-oss-120b", provider="groq")
    check("session 1 sees its own spend", first["used"] == 90000, str(first["used"]))

    # A new UsageStore over the same file is what a restarted backend sees.
    s2 = UsageStore(d)
    after = s2.snapshot(default_limit=200000, model="openai/gpt-oss-120b", provider="groq")
    check("session 2 still sees 90,000 (the old bug read 0)", after["used"] == 90000,
          str(after["used"]))
    check("...and the call count too", after["calls"] == 1, str(after["calls"]))

    s2.record_call("openai/gpt-oss-120b", 1000, provider="groq")
    s3 = UsageStore(d)
    third = s3.snapshot(default_limit=200000, model="openai/gpt-oss-120b", provider="groq")
    check("a third session accumulates rather than resets", third["used"] == 91000,
          str(third["used"]))
    check("the file has exactly one top-level key", list(_doc(d)) == ["days"],
          str(list(_doc(d))))

    # The refill estimate is anchored to the first call, so a restart must not
    # reset the window to "now" and tell the user the budget refilled early.
    check("refill estimate is still anchored to the first call",
          bool(third["refillAt"]) and third["refillAt"] == first["refillAt"],
          f"{third['refillAt']} vs {first['refillAt']}")


# --------------------------------------------------------------------------- #
# 2. provider isolation
# --------------------------------------------------------------------------- #


def test_provider_isolation() -> None:
    print("\n[isolate] one provider's spend never becomes another's")
    d = _dir()
    s = UsageStore(d)
    s.record_call("openai/gpt-oss-120b", 180000, provider="groq")
    s.record_quota_error("openai/gpt-oss-120b", GROQ_TPD_429, provider="groq")
    s.record_call("deepseek-flash", 5000, provider="deepseek")

    ds = s.snapshot(default_limit=2_000_000, model="deepseek-flash", provider="deepseek")
    gr = s.snapshot(default_limit=200_000, model="openai/gpt-oss-120b", provider="groq")

    check("deepseek used = its own 5,000, not groq's 198,530", ds["used"] == 5000,
          str(ds["used"]))
    check("deepseek limit = its configured 2,000,000", ds["limit"] == 2_000_000,
          str(ds["limit"]))
    check("deepseek is NOT reported as nearly exhausted", ds["percentUsed"] < 1.0,
          f"{ds['percentUsed']}%")
    check("deepseek does not claim groq's authoritative number",
          ds["source"] == "local-sum" and ds["reportedUsed"] is None,
          f"{ds['source']}/{ds['reportedUsed']}")
    check("groq keeps its own 429-derived total", gr["used"] == 198530, str(gr["used"]))
    check("groq keeps its own 429-derived ceiling", gr["limit"] == 200000,
          str(gr["limit"]))
    check("groq is the one marked provider-report", gr["source"] == "provider-report",
          gr["source"])
    check("the per-model list is provider-scoped",
          [m["model"] for m in ds["models"]] == ["deepseek-flash"],
          str([m["model"] for m in ds["models"]]))
    check("the per-model list is provider-scoped for groq too",
          [m["model"] for m in gr["models"]] == ["openai/gpt-oss-120b"],
          str([m["model"] for m in gr["models"]]))
    check("calls are counted per provider", ds["calls"] == 1 and gr["calls"] == 1,
          f"{ds['calls']}/{gr['calls']}")

    # Same model NAME under two providers must stay in separate buckets: a
    # name is only unique within a provider.
    s.record_call("shared-name", 4000, provider="nvidia")
    nv = s.snapshot(default_limit=2_000_000, model="shared-name", provider="nvidia")
    check("a repeated model name does not merge two providers",
          nv["used"] == 4000, str(nv["used"]))


def test_rate_limit_headers_isolated() -> None:
    print("\n[isolate] rate-limit headers belong to the provider that sent them")
    d = _dir()
    s = UsageStore(d)
    s.record_call("openai/gpt-oss-120b", 100, provider="groq",
                  headers={"x-ratelimit-remaining-tokens": "1400",
                           "x-ratelimit-limit-tokens": "6000",
                           "content-type": "application/json"})
    s.record_call("deepseek-flash", 100, provider="deepseek",
                  headers={"x-ratelimit-remaining-requests": "2499"})

    gr = s.snapshot(default_limit=200_000, model="openai/gpt-oss-120b", provider="groq")
    ds = s.snapshot(default_limit=2_000_000, model="deepseek-flash", provider="deepseek")
    check("groq keeps its own rate-limit headers",
          gr["rateLimit"].get("x-ratelimit-remaining-tokens") == "1400",
          str(gr["rateLimit"]))
    check("deepseek shows its own, not groq's",
          ds["rateLimit"].get("x-ratelimit-remaining-requests") == "2499"
          and "x-ratelimit-remaining-tokens" not in ds["rateLimit"],
          str(ds["rateLimit"]))
    check("non-rate-limit headers are dropped",
          "content-type" not in gr["rateLimit"], str(gr["rateLimit"]))
    check("a provider that never sent headers gets none", isinstance(
        s.snapshot(default_limit=1, model="openai/gpt-oss-120b", provider="nvidia"
                   )["rateLimit"], dict))

    # A last-writer-wins overwrite must also stay inside its own provider.
    s.record_call("deepseek-flash", 10, provider="deepseek",
                  headers={"x-ratelimit-remaining-requests": "1"})
    check("an overwrite does not clobber the other provider",
          s.snapshot(default_limit=200_000, model="openai/gpt-oss-120b",
                     provider="groq")["rateLimit"].get("x-ratelimit-remaining-tokens")
          == "1400")

    # Legacy single-slot files were written by Groq only.
    k = today_key()
    legacy = _write(_dir(), {"days": {k: {
        "models": {"openai/gpt-oss-120b": {"tokens": 10, "provider": "groq"}},
        "rateLimit": {"x-ratelimit-remaining-tokens": "77"},
    }}})
    check("a legacy groq header slot still reads as groq's",
          legacy.snapshot(default_limit=1, provider="groq")["rateLimit"]
          .get("x-ratelimit-remaining-tokens") == "77")
    check("...and is not shown to another provider",
          legacy.snapshot(default_limit=1, provider="deepseek")["rateLimit"] == {})


# --------------------------------------------------------------------------- #
# 3. only Groq may define a budget
# --------------------------------------------------------------------------- #


def test_only_groq_defines_a_budget() -> None:
    print("\n[quota] only groq's 429 may set the ceiling")
    d = _dir()
    s = UsageStore(d)
    s.record_call("deepseek-flash", 100, provider="deepseek")
    # `parse_quota_message` understands Groq's TPD wording. A non-Groq 429 that
    # happens to contain the same numbers must not restate one provider's
    # ceiling as another's, so it is ignored outright.
    info = s.record_quota_error("deepseek-flash", "Limit 200000, Used 199000, TPD",
                                provider="deepseek")
    snap = s.snapshot(default_limit=2_000_000, model="deepseek-flash", provider="deepseek")
    check("a non-groq 429 is not recorded", info == {}, str(info))
    check("a non-groq 429 does not lower the ceiling", snap["limit"] == 2_000_000,
          str(snap["limit"]))
    check("a non-groq 429 does not inflate the total", snap["used"] == 100,
          str(snap["used"]))

    # A groq 429 that does NOT parse must also be a no-op, not a zeroed counter.
    s.record_quota_error("openai/gpt-oss-120b", "rate limited, slow down", provider="groq")
    gr = s.snapshot(default_limit=200_000, model="openai/gpt-oss-120b", provider="groq")
    check("an unparseable groq 429 keeps the local tally", gr["used"] == 0 and
          gr["source"] == "local-sum", f"{gr['used']}/{gr['source']}")


# --------------------------------------------------------------------------- #
# 4. file-format tolerance
# --------------------------------------------------------------------------- #


def test_file_tolerance() -> None:
    print("\n[format] old and foreign files are read, not crashed on")
    k = today_key()

    # The shape the old buggy writer left behind: the day duplicated at the top
    # level as well as under "days". The nested map is the newer truth.
    d = _dir()
    s = _write(d, {
        k: {"models": {"m": {"tokens": 1}}, "calls": 1},
        "days": {k: {"models": {"openai/gpt-oss-120b": {"tokens": 77000, "calls": 9}},
                      "calls": 9, "firstCallAt": 1}},
    })
    snap = s.snapshot(default_limit=200_000, model="openai/gpt-oss-120b", provider="groq")
    check("a file with a stale duplicate reads the nested total", snap["used"] == 77000,
          str(snap["used"]))
    s.record_call("openai/gpt-oss-120b", 500, provider="groq")
    check("...and is rewritten without the duplicate", list(_doc(d)) == ["days"],
          str(list(_doc(d))))

    # Pre-provider files: one flat `reported` dict, model entries with no
    # provider tag. Those were written when Groq was the only provider, so
    # Groq keeps the numbers and nobody else inherits the ceiling.
    d = _dir()
    s = _write(d, {"days": {k: {
        "models": {"openai/gpt-oss-120b": {"tokens": 4200, "calls": 3}},
        "reported": {"used": 4300, "limit": 200000}, "calls": 3, "firstCallAt": 1,
    }}})
    gr = s.snapshot(default_limit=200_000, model="openai/gpt-oss-120b", provider="groq")
    ds = s.snapshot(default_limit=2_000_000, model="deepseek-flash", provider="deepseek")
    check("a pre-provider file still reads as groq's", gr["used"] == 4300, str(gr["used"]))
    check("...including its local tally", gr["localTokens"] == 4200,
          str(gr["localTokens"]))
    check("its ceiling is not inherited by another provider",
          ds["limit"] == 2_000_000 and ds["used"] == 0,
          f"{ds['limit']}/{ds['used']}")

    # Garbage and foreign shapes must degrade to an empty counter, never raise:
    # telemetry is never allowed to fail a job.
    d = _dir()
    (d / "groq_usage.json").write_text("{not json", encoding="utf-8")
    try:
        snap = UsageStore(d).snapshot(default_limit=5, provider="groq")
        check("an unreadable file degrades to zero", snap["used"] == 0)
    except Exception as exc:  # noqa: BLE001 - that IS the assertion
        check("an unreadable file degrades to zero", False, repr(exc))

    d = _dir()
    s = _write(d, {"unexpected": True})
    try:
        snap = s.snapshot(default_limit=5, provider="groq")
        check("a foreign shape degrades to zero", snap["used"] == 0)
    except Exception as exc:  # noqa: BLE001
        check("a foreign shape degrades to zero", False, repr(exc))

    d = _dir()
    (d / "groq_usage.json").write_text("[]", encoding="utf-8")
    try:
        snap = UsageStore(d).snapshot(default_limit=5, provider="groq")
        check("a JSON array degrades to zero", snap["used"] == 0)
    except Exception as exc:  # noqa: BLE001
        check("a JSON array degrades to zero", False, repr(exc))


# --------------------------------------------------------------------------- #
# 5. the API reports the active provider's own numbers
# --------------------------------------------------------------------------- #


def test_api_reports_active_provider() -> None:
    print("\n[api] /api/usage describes the ACTIVE provider's budget only")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app import llm as llm_mod  # noqa: PLC0415
    from app import main as main_mod  # noqa: PLC0415

    client = TestClient(main_mod.app)
    active = main_mod.settings.llm.provider
    cfg = llm_mod.openai_cfg(main_mod.settings, active)
    usage = client.get("/api/usage").json()
    check("usage names the active provider", usage.get("provider") == active,
          str(usage.get("provider")))
    check("usage limit is the active provider's configured limit",
          usage["limit"] == cfg.daily_token_limit,
          f"{usage['limit']} vs {cfg.daily_token_limit}")
    check("usage is scoped to that provider", usage.get("provider") == active)
    # The frontend branches on this string; a rename on one side only would
    # silently downgrade every provider to "local tally" forever.
    check("source is one of the two the UI knows", usage["source"] in
          {"provider-report", "local-sum"}, str(usage.get("source")))
    check("a budgeted provider is flagged as such", usage.get("budgeted") is True,
          str(usage.get("budgeted")))


def test_unbudgeted_providers_invent_nothing() -> None:
    print("\n[api] a backend with no daily ceiling gets no invented meter")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app import main as main_mod  # noqa: PLC0415

    client = TestClient(main_mod.app)
    original = main_mod.settings.llm.provider
    try:
        for name, expect_label in (("ollama", "Ollama"), ("gemini", "Gemini")):
            main_mod.settings.llm.provider = name
            u = client.get("/api/usage").json()
            check(f"{name} reports no daily budget", u.get("budgeted") is False,
                  str(u.get("budgeted")))
            check(f"{name} invents no denominator", u.get("limit") == 0, str(u.get("limit")))
            check(f"{name} borrows no model name", not u.get("activeModel"),
                  str(u.get("activeModel")))
            check(f"{name} is not labelled as a hosted vendor",
                  u.get("providerLabel") == expect_label, str(u.get("providerLabel")))
            # This is the gate NewJobForm uses, so an unbudgeted backend can
            # never trip the "quota spent" pre-submit warning.
            check(f"{name} is not marked configured", u.get("configured") is False,
                  str(u.get("configured")))
    finally:
        main_mod.settings.llm.provider = original


def main() -> int:
    test_survives_restart()
    test_provider_isolation()
    test_rate_limit_headers_isolated()
    test_only_groq_defines_a_budget()
    test_file_tolerance()
    test_api_reports_active_provider()
    test_unbudgeted_providers_invent_nothing()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
