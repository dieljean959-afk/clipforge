"""Salvage clip candidates from a model reply that is not clean JSON.

Why this module exists
----------------------
A detection chunk is a ~120s transcript asking for a list of highlights. That
list is the ONLY thing standing between a 20-chunk job and its clips, and the
model does not reliably hand it over as parseable JSON. Observed against the
live Groq API while fixing the `json_validate_failed` job failure:

  * the reply is wrapped in ```json fences, or has a sentence of prose in
    front of it and a sign-off after;
  * the reply is TRUNCATED mid-array because `max_tokens` ran out — the single
    most common cause, and the one the retry ladder exists for;
  * a trailing comma survives (`[{"a":1},]`);
  * `json_object` mode is in play, which coerces the requested ARRAY into a
    single clip OBJECT, so a one-clip answer arrives where a list was asked
    for;
  * smart quotes, full-width punctuation and `None`/`True` in the Python sense.

The rule throughout: never raise because a reply was ugly. Repair what can be
repaired, and hand back whatever survived, so a single malformed chunk costs a
few clips instead of the whole job. `parse_candidates` in app/llm.py is the
only caller, and it is deliberately the only place that knows reply shape.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Keys a clip may use for its start/end word index. Models drift to
#: `start`, `from`, `begin`, `start_index` — accept the common spellings
#: rather than losing an otherwise-good clip to a naming difference.
_START_KEYS = ("start_word", "startWord", "start_index", "startIndex", "start", "from", "begin")
_END_KEYS = ("end_word", "endWord", "end_index", "endIndex", "end", "to", "until")

#: Fenced block, the prose-wrapped case, or a bare payload.
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

#: `None` / `True` / `False` leaking from a Python-flavoured model.
_PY_LITERALS = re.compile(r"\b(None|True|False)\b")

#: Curly quotes and the ellipsis a model uses for "truncated here".
_SMART = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "…": "...",
})


def strip_fences(text: str) -> str:
    """Return the contents of the first fenced block, else the text itself."""
    if not text:
        return ""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def _balanced_span(text: str, opener: str, closer: str) -> str | None:
    """The first balanced `opener`…`closer` span, ignoring brackets in strings.

    Scans with a string/escape state machine rather than a regex so a `]` inside
    a title ("the [good] part") cannot close the array early — which is exactly
    how a naive `re.search(r"\\[.*\\]")` truncates a real reply.
    """
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unbalanced — a truncated reply. Hand back what we have; the caller closes
    # it and drops any partial final element.
    return text[start:]


def extract_payload(text: str) -> str | None:
    """Pull the outermost JSON array or object out of a noisy reply."""
    if not text:
        return None
    cleaned = strip_fences(text)
    candidates = [
        _balanced_span(cleaned, "[", "]"),
        _balanced_span(cleaned, "{", "}"),
        cleaned,
    ]
    for cand in candidates:
        if cand:
            return cand
    return None


def _close_truncated(payload: str) -> str:
    """Best-effort close of a payload cut off by the token limit.

    Drops the incomplete trailing element, then balances the brackets. A reply
    truncated at 3 of 5 clips still yields 3 usable clips instead of nothing.
    """
    out = payload.rstrip()
    # Cut back to the last point where an element was definitely complete, so a
    # half-written final object is dropped rather than poisoning the whole list.
    last_obj = out.rfind("}")
    if last_obj == -1:
        # Nothing finished yet — the reply was cut inside the FIRST clip, very
        # often in the middle of a string. Giving up here threw the whole chunk
        # away even though closing the open string and the open brackets below
        # recovers it: the fields written before the cut are all still there,
        # and every one the schema needs is written before the model reaches
        # the caption fields. This is the single most common truncation shape
        # at a tight max_tokens, so it is worth the branch.
        return _balance(out)
    # Everything after the last '}' is either a partial element or the tail of
    # the array; keep up to and including it.
    out = out[: last_obj + 1]
    return _balance(out)


def _balance(out: str) -> str:
    """Close whatever the model left open: the string, then the brackets.

    A stack, not two counters. Two counters force a close order (arrays before
    objects) that is only accidentally right for a flat payload; the moment the
    payload nests — which is every `{"clips": [ ... ]}` envelope — it produces
    `[{"clips": [...]` closed as `]}` and the JSON is still unparseable.

    A reply cut mid-VALUE (`"score": 0.`) cannot be closed, only trimmed, so
    the incomplete trailing `, "key": value` is dropped until it parses.
    """
    stack: list[str] = []
    in_str = False
    escape = False
    quote = ""
    for ch in out:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if in_str:
        # A cut inside a string literal: close it. A dangling escape would
        # escape the quote we are about to add, so it is dropped.
        out = out[:-1] if out.endswith("\\") else out
        out += quote
    tail = "".join(reversed(stack))
    for candidate in (out + tail, _drop_partial_tail(out) + tail):
        try:
            json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        return candidate
    return out + tail


def _drop_partial_tail(payload: str) -> str:
    """Cut a reply back to its last complete `, "key": value`.

    Only reached when the reply was cut inside a value, where closing brackets
    are not enough: `"score": 0.` has to lose its key as well. Repeated until
    the tail ends on a complete member, which json.loads then confirms.
    """
    out = payload.rstrip()
    for _ in range(64):
        cut = out.rfind(",")
        if cut == -1:
            break
        out = out[:cut].rstrip()
        try:
            json.loads(_balance(out))
        except (json.JSONDecodeError, ValueError, RecursionError):
            continue
        return out
    return payload.rstrip()


def _drop_trailing_commas(s: str) -> str:
    """`{"a":1},]` -> `{"a":1}]`. A comma before a closing bracket."""
    return re.sub(r",(\s*[\]}])", r"\1", s)


def _fix_literals(s: str) -> str:
    return _PY_LITERALS.sub(lambda m: {"None": "null", "True": "true", "False": "false"}[m.group(1)], s)


def loads_lenient(text: str) -> Any:
    """Parse a possibly-noisy reply. Returns None if nothing usable is found.

    Tries, in order: as-is, stripped of fences/prose, trailing commas removed,
    truncated-and-closed, and finally a per-element sweep that keeps only the
    elements which parse — that last one is what recovers a list where the
    model emitted one malformed object among good ones.
    """
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None
    attempts: list[str] = []
    stripped = strip_fences(raw).translate(_SMART)
    attempts.append(stripped)
    for base in (stripped, _extract_only(stripped)):
        if not base:
            continue
        no_commas = _drop_trailing_commas(base)
        attempts.append(no_commas)
        fixed = _fix_literals(no_commas)
        attempts.append(fixed)
        attempts.append(_close_truncated(fixed))
    seen: set[str] = set()
    for attempt in attempts:
        if not attempt or attempt in seen:
            continue
        seen.add(attempt)
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
        # Some replies are several concatenated JSON values.
    # Per-element salvage: an array whose 3rd object is broken.


def _is_timestampish_string(v: Any) -> tuple[bool, float]:
    """True when `v` looks like a timestamp string we can parse to seconds.

    Accepted forms:
      * "12.5"            -> 12.5
      * "12:34"           -> 12*60 + 34 = 754
      * "00:12:34"        -> 12*60 + 34 = 754 (leading 00 = hours)
      * "1:02:03.4"       -> 1*3600 + 2*60 + 3.4
    Anything else returns (False, 0.0).
    """
    s = str(v or "").strip()
    if not s:
        return False, 0.0
    if _IS_NUMERIC.match(s):
        try:
            f = float(s)
            if f != f:
                return False, 0.0
            return True, f
        except (ValueError, TypeError):
            return False, 0.0
    m = _TIME_RE.match(s)
    if not m:
        return False, 0.0
    h = float(m.group(1) or 0)
    mm = float(m.group(2))
    sec = float(m.group(3) or 0)
    total = h * 3600 + mm * 60 + sec
    if total != total:
        return False, 0.0
    return True, total


_IS_NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")
_TIME_RE = re.compile(r"^(\d{1,3}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$|^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def _coerce_timestamp(v: Any) -> float | None:
    """Best-effort coerce a model field to a numeric seconds value.

    Accepts numbers and the timestamp strings listed in ``_is_timestampish_string``
    so a model that emits "12:34" or "754.2" is not silently dropped. Returns
    ``None`` when the value is genuinely unusable (null, boolean, object, array,
    empty string, nan).
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if f != f:
            return None
        return f
    ok, total = _is_timestampish_string(v)
    if ok:
        return total
    return None
    return _salvage_array(stripped)


def _extract_only(text: str) -> str | None:
    return _balanced_span(text, "[", "]") or _balanced_span(text, "{", "}")


def _salvage_array(text: str) -> Any:
    """Parse a list element-by-element, keeping the ones that are valid.

    Uses `JSONDecoder.raw_decode` per element instead of a bracket-counting
    state machine. An unescaped quote inside a title ("The "best" moment")
    desynchronises a string-tracking scanner for the REST of the payload, so
    one bad element used to cost every good element after it as well. raw_decode
    is driven by the real JSON grammar: it consumes exactly one well-formed
    value, raises on a malformed one, and the resync below simply steps to the
    next '{'. So a list with three clips and one broken title yields the two
    good ones.
    """
    body = _balanced_span(text, "[", "]")
    if body is None:
        return None
    inner = body[1:-1]
    if not inner.strip():
        return []
    decoder = json.JSONDecoder()
    out: list[Any] = []
    i = 0
    n = len(inner)
    while i < n:
        if inner[i] != "{":
            i += 1
            continue
        try:
            value, end = decoder.raw_decode(inner, i)
        except (json.JSONDecodeError, ValueError):
            # Malformed element: resync on the next object opener.
            nxt = inner.find("{", i + 1)
            if nxt == -1:
                break
            i = nxt
            continue
        out.append(value)
        i = end
    return out


def _as_list(value: Any) -> list[Any]:
    """Coerce a parsed payload into a list of clip dicts.

    Handles the three shapes the models actually produce:
      - a list of clip objects (what the prompt asks for);
      - a single clip object (what `json_object` mode coerces a one-clip answer
        into — the array is lost entirely, so this recovers it);
      - `{"clips": [...]}` / `{"results": [...]}` (the object envelope, which
        is the shape strict json_schema produces).
    """
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        for key in ("clips", "results", "highlights", "candidates", "data", "items"):
            inner = value.get(key)
            if isinstance(inner, list):
                return [v for v in inner if isinstance(v, dict)]
        # A bare clip object.
        if any(k in value for k in _START_KEYS) or "title" in value:
            return [value]
    return []


def _coerce_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            value = value.strip().rstrip("s")
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_key(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in item and item[k] is not None:
            return item[k]
    return None


def normalize_clip(item: Any) -> dict[str, Any] | None:
    """Validate one clip against the schema; return it normalized, or None.

    The required fields are the start/end word indices. A clip without usable
    indices cannot be timed — a clip with a missing title or an odd score is
    still a real moment and gets a default, because dropping it would silently
    shrink the batch for a cosmetic reason.
    """
    if not isinstance(item, dict):
        return None
    start = _coerce_int(_first_key(item, _START_KEYS))
    end = _coerce_int(_first_key(item, _END_KEYS))
    if start is None or end is None:
        return None
    if end <= start:
        # Some models emit the range reversed; accept it rather than lose it.
        start, end = end, start
    if end <= start:
        return None
    try:
        score = float(item.get("score", item.get("rating", 0)) or 0)
    except (TypeError, ValueError):
        score = 0.0
    title = str(item.get("title") or item.get("name") or "Untitled clip").strip()
    if not title:
        title = "Untitled clip"
    out: dict[str, Any] = {
        "start_word": start,
        "end_word": end,
        "score": max(0.0, min(100.0, score)),
        "title": title[:120],
    }
    for key in ("hook_text", "hashtags"):
        if key in item:
            out[key] = item[key]
    return out


def repair_candidates(text: str) -> list[dict[str, Any]]:
    """Turn a raw model reply into validated clip dicts. Never raises.

    Returns an empty list when the reply holds nothing usable — the caller
    treats that as "this chunk yielded no clips" and moves on, which is the
    whole point: a chunk that cannot be parsed must not fail the job.
    """
    try:
        parsed = loads_lenient(text)
    except Exception:  # noqa: BLE001 - a bad reply is not a crash
        return []
    if parsed is None:
        return []
    out: list[dict[str, Any]] = []
    for item in _as_list(parsed):
        clip = normalize_clip(item)
        if clip is not None:
            out.append(clip)
    return out
