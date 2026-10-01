"""What the Groq free tier has left today, and when it comes back.

The daily token budget is the only hard wall between a working ClipForge and a
job that dies at detection chunk 18 of 20, and it is invisible until you hit it.
Three sources of truth, in descending order of trust:

  1. **A 429 rejection.** Groq states the real numbers in the body —
     ``Limit 200000, Used 198530, Requested 5360. Please try again in 28m0.48s``
     — and that is the ONLY place the actual daily counter appears. Nothing
     else, including the rate-limit headers, reports tokens-per-day.
  2. **Per-call usage.** Every completion returns ``usage.total_tokens``, which
     is what you were actually billed. Summing these tracks consumption closely
     and is available on the happy path, with no 429 needed.
  3. **A local day counter.** What (2) accumulated since midnight, which is what
     the badge shows when no 429 has happened.

Source (1) wins when it is fresher than the local sum, because Groq's window is
a ROLLING 24 hours, not a calendar day: a local midnight boundary is an
approximation, and the number in the 429 is not.

Persisted as JSON next to the other run state (``data/groq_usage.json``) rather
than SQLite: this is a handful of counters, written at most once per API call,
and it has to survive a host restart mid-detection. Only the last few days are
kept so the file cannot grow without bound.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

#: How many days of history to retain. The UI only ever shows today, but keeping
#: a few days makes a sudden usage spike explainable after the fact.
_KEEP_DAYS = 7

#: Groq's phrasing, e.g.
#:   "Rate limit reached for model `openai/gpt-oss-120b` in organization
#:    `org_x` service tier `on_demand` on tokens per day (TPD):
#:    Limit 200000, Used 198530, Requested 5360. Please try again in 28m0.48s."
#: Only the first three numbers are interesting here, and the ordering has been
#: stable, but each is optional so a reworded message degrades instead of
#: failing the whole parse.
_LIMIT_RE = re.compile(r"\bLimit\s+(\d+)", re.I)
_USED_RE = re.compile(r"\bUsed\s+(\d+)", re.I)
_REQUESTED_RE = re.compile(r"\bRequested\s+(\d+)", re.I)
_RETRY_RE = re.compile(
    r"try again in\s+(\d+(?:\.\d+)?)\s*(m(?:in(?:ute)?s?)?|h(?:our(?:s)?)?|s(?:ec(?:ond)?s?)?)",
    re.I,
)

_UNIT_SECONDS = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
                 "m": 60.0, "min": 60.0, "minute": 60.0, "minutes": 60.0,
                 "h": 3600.0, "hour": 3600.0, "hours": 3600.0}


def today_key(now: float | None = None) -> str:
    return date.fromtimestamp(now if now is not None else time.time()).isoformat()


def parse_quota_message(detail: str) -> dict[str, Any]:
    """Pull the real daily-quota numbers out of a 429 body.

    Returns `{}` when the message is not a quota rejection (a per-minute limit,
    or any other error), so the caller can tell "no information" apart from
    "zero".
    """
    if not detail:
        return {}
    limit_m, used_m = _LIMIT_RE.search(detail), _USED_RE.search(detail)
    if not limit_m or not used_m:
        return {}
    out: dict[str, Any] = {"limit": int(limit_m.group(1)), "used": int(used_m.group(1))}
    req_m = _REQUESTED_RE.search(detail)
    if req_m:
        out["requested"] = int(req_m.group(1))
    retry_m = _RETRY_RE.search(detail)
    if retry_m:
        try:
            out["retry_after_sec"] = float(retry_m.group(1)) * _UNIT_SECONDS.get(
                retry_m.group(2).lower(), 60.0
            )
        except (TypeError, ValueError):
            pass
    return out


def _rate_limit_headers(headers: Any) -> dict[str, str]:
    """Keep only `x-ratelimit-*`, lowercased.

    These are per-minute/per-request counters — Groq never exposes the daily
    token balance in a header — but they are the cheapest way to show that a
    limit is being approached, and they cost nothing to retain.
    """
    out: dict[str, str] = {}
    if headers is None:
        return out
    try:
        items = headers.items()
    except AttributeError:
        return out
    for key, value in items:
        name = str(key).lower()
        if name.startswith("x-ratelimit-"):
            out[name] = str(value)
    return out


class UsageStore:
    """Per-model daily token counters, persisted as JSON."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / "groq_usage.json"
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._read()

    def _read(self) -> dict[str, Any]:
        """Load the whole document. `self._data["days"]` is ALWAYS the day map.

        Returning the OUTER document matters. An earlier version returned
        `loaded["days"]` — the inner map — while `_write` persisted `self._data`
        wrapped in a `days` key. So the first read after any write unwrapped one
        level too far, `_day()`'s `setdefault("days", {})` then opened a fresh
        empty namespace, and the real totals were ignored: the daily counter
        silently reset to 0 on every process restart, which is exactly when an
        operator is checking whether the badge is lying about their budget.
        """
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    # `days` is the only legitimate top-level key, so normalise
                    # to exactly that. A file from the buggy writer also carries
                    # day keys at the top level; keeping them would re-introduce
                    # the same unwrap confusion one reader away.
                    nested = loaded.get("days")
                    return {"days": nested if isinstance(nested, dict) else {}}
        except Exception as exc:  # noqa: BLE001 - never break a job over telemetry
            logger.warning(f"[groq-usage] unreadable usage file ({exc}) — starting fresh")
        return {"days": {}}

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[groq-usage] could not persist: {exc}")

    def _day(self, key: str) -> dict[str, Any]:
        days = self._data.setdefault("days", {})
        day = days.setdefault(key, {"models": {}, "calls": 0, "firstCallAt": 0})
        day.setdefault("models", {})
        return day

    def _prune(self) -> None:
        days = self._data.get("days") or {}
        if len(days) <= _KEEP_DAYS:
            return
        for key in sorted(days)[: len(days) - _KEEP_DAYS]:
            days.pop(key, None)

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_call(
        self,
        model: str,
        total_tokens: int | None = None,
        headers: Any = None,
        provider: str = "groq",
    ) -> None:
        """Add one successful call's billed tokens to today's counter.

        `provider` tags the bucket. The daily budgets are not comparable — a
        200k tokens/day Groq ceiling and a DeepSeek balance — so a shared
        counter would report one provider's spend against another's limit and
        show a false "quota exhausted" warning on a provider that never spent
        anything.
        """
        now_ms = int(time.time() * 1000)
        with self._lock:
            day = self._day(today_key())
            entry = day["models"].setdefault(model, {"tokens": 0, "calls": 0})
            # A model name is only unique within a provider, so keep the
            # provider on the entry rather than trusting the name to be global.
            entry["provider"] = provider
            entry["calls"] = int(entry.get("calls", 0)) + 1
            if total_tokens and total_tokens > 0:
                entry["tokens"] = int(entry.get("tokens", 0)) + int(total_tokens)
            day["calls"] = int(day.get("calls", 0)) + 1
            if not day.get("firstCallAt"):
                day["firstCallAt"] = now_ms
            # Also per-model, so the refill estimate can use THIS provider's
            # first call of the day rather than whichever provider happened to
            # call first — otherwise switching providers mid-day reports a
            # window that opened before the provider was even in use.
            if not entry.get("firstCallAt"):
                entry["firstCallAt"] = now_ms
            rl = _rate_limit_headers(headers)
            if rl:
                day["rateLimit"] = rl
            self._prune()
            self._write()

    def record_quota_error(
        self, model: str, detail: str, provider: str = "groq"
    ) -> dict[str, Any]:
        """Record Groq's authoritative daily numbers from a 429 body.

        This is the only source that reports the REAL daily counter, so it is
        stored verbatim rather than being added to the local sum — adding it
        would double-count every token already tallied from `usage`.

        Groq ONLY. `parse_quota_message` understands Groq's tokens-per-DAY
        wording, and it overwrites the denominator for the whole day, so
        letting another provider's 429 land here would restate one provider's
        ceiling as another's. A non-Groq 429 still backs off normally; it just
        does not get to define a budget.
        """
        if provider != "groq":
            return {}
        info = parse_quota_message(detail)
        if not info:
            return {}
        now_ms = int(time.time() * 1000)
        with self._lock:
            day = self._day(today_key())
            day.setdefault("reported", {})
            day["reported"][provider] = {
                "used": int(info["used"]),
                "limit": int(info["limit"]),
                "at": now_ms,
            }
            if info.get("requested") is not None:
                day["reported"][provider]["requested"] = int(info["requested"])
            if info.get("retry_after_sec") is not None:
                day["reported"][provider]["refillAt"] = (
                    now_ms + int(float(info["retry_after_sec"]) * 1000)
                )
            entry = day["models"].setdefault(model, {"tokens": 0, "calls": 0})
            entry["provider"] = provider
            entry["quotaErrorAt"] = now_ms
            self._prune()
            self._write()
        logger.warning(
            f"[groq-usage] {model}: Groq reports {info['used']}/{info['limit']} "
            f"tokens used today"
            + (
                f", refills in {info['retry_after_sec'] / 60:.0f}m"
                if info.get("retry_after_sec")
                else ""
            )
        )
        return info

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def snapshot(
        self, *, default_limit: int, model: str = "", provider: str = "groq"
    ) -> dict[str, Any]:
        """Today's usage for ONE provider, preferring that provider's own number.

        Scoped to `provider` on purpose. The budgets are not comparable, so a
        combined figure would divide one provider's spend by another's ceiling.
        """
        now = time.time()
        with self._lock:
            day = dict(self._day(today_key()))
            models = {k: dict(v) for k, v in (day.get("models") or {}).items()}
        # Entries written before providers were distinguished carry no tag;
        # Groq is the correct owner because it was the only provider that could
        # write them, so today's file keeps its numbers instead of zeroing.
        mine = {
            k: v
            for k, v in models.items()
            if str(v.get("provider") or "groq") == provider
        }
        reported = self._reported_for(day, provider)

        local_tokens = 0
        for entry in mine.values():
            local_tokens += int(entry.get("tokens", 0))

        # Groq's number is authoritative while it is same-day. If the local sum
        # is HIGHER, it is still the better number: it can only exceed the
        # provider's by counting calls the provider has not yet counted.
        used = int(reported.get("used", 0))
        if local_tokens > used:
            used = local_tokens
        limit = int(reported.get("limit") or default_limit or 0)
        remaining = max(0, limit - used) if limit else 0

        refill_at: int | None = reported.get("refillAt")
        first_call = next(
            (int(e["firstCallAt"]) for e in mine.values() if e.get("firstCallAt")),
            0,
        ) or int(day.get("firstCallAt") or 0)
        if not refill_at and first_call:
            # No 429 seen, so no provider estimate. The window is a rolling 24
            # hours, so the best available guess is that the first call of the
            # local day begins ageing out then.
            refill_at = first_call + 24 * 3600 * 1000

        active = mine.get(model) if model else None
        return {
            "used": used,
            "limit": limit,
            "remaining": remaining,
            "percentUsed": round((used / limit) * 100, 1) if limit else 0.0,
            "localTokens": local_tokens,
            "reportedUsed": int(reported.get("used", 0)) if reported else None,
            "source": "provider-report" if reported else "local-sum",
            "provider": provider,
            "refillAt": refill_at,
            "refillSec": max(0, int((refill_at - now * 1000) / 1000)) if refill_at else None,
            "calls": sum(int(e.get("calls", 0)) for e in mine.values()),
            "activeModel": model,
            "activeModelTokens": int((active or {}).get("tokens", 0)),
            "models": [
                {
                    "model": name,
                    "tokens": int(entry.get("tokens", 0)),
                    "calls": int(entry.get("calls", 0)),
                    "active": name == model,
                }
                for name, entry in sorted(mine.items(), key=lambda kv: -int(kv[1].get("tokens", 0)))
            ],
            "rateLimit": day.get("rateLimit") or {},
        }

    @staticmethod
    def _reported_for(day: dict[str, Any], provider: str) -> dict[str, Any]:
        """This provider's authoritative 429 numbers, tolerating the old shape.

        `day["reported"]` used to be a single flat dict written only by Groq.
        Reading a legacy file must not crash and must not hand Groq's ceiling
        to whichever provider is active now.
        """
        reported = day.get("reported") or {}
        if not isinstance(reported, dict):
            return {}
        if "used" in reported or "limit" in reported:
            # Legacy flat record: Groq's, by construction.
            return reported if provider == "groq" else {}
        scoped = reported.get(provider)
        return scoped if isinstance(scoped, dict) else {}

    def reset(self) -> None:
        with self._lock:
            self._data = {"days": {}}
            self._write()


_STORE: UsageStore | None = None
_STORE_LOCK = threading.Lock()


def store(data_dir: Path) -> UsageStore:
    """Process-wide usage store for a data dir (mirrors `eta.store`)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None or _STORE.path.parent != Path(data_dir):
            _STORE = UsageStore(Path(data_dir))
        return _STORE


#: Measured cost of DETECTION per second of source audio.
#:
#: Taken from a real 54.6-minute job: 20 chunks of ~120s each, ~2,400 input
#: tokens per chunk plus ~800 of actual output (the provider reserves
#: `max_tokens` but only bills what it generates), so ~3,200 billed tokens per
#: chunk = 26.7 tokens per second of source. Transcription and rendering cost
#: nothing here — only detection talks to Groq.
#:
#: Deliberately an over-estimate: the warning it drives is better a little
#: pessimistic than a job that dies at chunk 18.
_TOKENS_PER_SOURCE_SECOND = 26.7


def estimate_detection_tokens(duration_sec: float) -> int:
    """Rough token cost of detecting highlights for a source of this length."""
    return int(max(0.0, float(duration_sec or 0.0)) * _TOKENS_PER_SOURCE_SECOND)


__all__ = [
    "UsageStore",
    "estimate_detection_tokens",
    "parse_quota_message",
    "store",
    "today_key",
]
