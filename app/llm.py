"""Clip/highlight detection (LLM service).

Chunked transcript → LLM → JSON [{start_word, end_word, score, title,
hook_text}] with 429-aware exponential backoff. The model returns WORD
INDICES, never timestamps — fragile float math is the LLM's weak spot, so we
resolve the real start/end from Whisper's word-level timings afterwards.

Two interchangeable hosted backends plus a local one, chosen by `[llm]
provider` / CLIPFORGE_LLM_PROVIDER:
  - groq   (default) — OpenAI-compatible chat completions. Free tier is
                       ~30 req/min, so calls are PACED client-side and 429s
                       are retried with Retry-After-aware backoff.
  - gemini           — hosted alternative, kept selectable as a fallback when
                       the Groq quota is exhausted.
  - ollama           — self-hosted. Free and fully private, but it competes for
                       the SAME CPU as Whisper, so it runs sequentially and is
                       markedly slower. Requires no API key.

The contract is deliberately narrow: every backend takes a prompt STRING and
returns a reply STRING. `parse_candidates` is the only thing that knows what
the reply looks like, which is what makes swapping providers safe.

Only transcript text is ever sent to any backend — never audio or video.
The key-missing case is surfaced loudly: no silent heuristic fallback exists
in this module. Likewise there is NO automatic failover between providers:
a 429 on Groq fails the stage visibly rather than quietly producing clips
scored by a different model than the one the operator configured.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from loguru import logger

from app.config import GeminiConfig, Settings, live_gemini_key, live_groq_key

PROVIDER_GROQ = "groq"
PROVIDER_GEMINI = "gemini"
PROVIDER_OLLAMA = "ollama"
PROVIDERS = (PROVIDER_GROQ, PROVIDER_GEMINI, PROVIDER_OLLAMA)


class LlmNotConfigured(Exception):
    """Selected backend is unusable (no key / no server) — surface, never fall back."""


class DetectionPaused(Exception):
    """The user asked to pause, and detection stopped at a chunk boundary.

    Not a failure. Completed chunks are already cached on disk, so resuming
    continues from the next unfinished chunk rather than re-running — and
    re-paying for — the whole transcript.
    """


class LlmError(Exception):
    """The selected backend's call failed after retries."""


def provider(settings: Settings) -> str:
    """Which detection backend is active. Defaults to the hosted one."""
    return getattr(getattr(settings, "llm", None), "provider", PROVIDER_GROQ)


def provider_label(settings: Settings) -> str:
    """Human-readable backend name for logs and the dashboard."""
    active = provider(settings)
    if active == PROVIDER_OLLAMA:
        return f"ollama ({settings.llm.ollama_model})"
    if active == PROVIDER_GEMINI:
        return f"gemini ({settings.gemini.model})"
    return f"groq ({settings.groq.model})"


# Model names rot: Google retires old flash models with a 404 that names the
# recommended replacement ("models/gemini-X-flash"). When that happens the
# call self-heals to the recommended model and persists it to config.toml so
# every later chunk uses it too.
_MODEL_RETIRED = re.compile(r"models/([A-Za-z0-9._-]+)\s+is no longer available", re.IGNORECASE)
_RECOMMENDED_MODEL = re.compile(r"use\s+models/([A-Za-z0-9._-]+)", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# optional SDK import (kept lazy so the API can boot without the package)
# --------------------------------------------------------------------------- #

def _load_google_module():
    try:
        from google import genai  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise LlmError(
            f"google-genai is not importable ({exc}); run `pip install google-genai`"
        ) from exc
    return genai


def build_prompt(chunk: list[dict], cfg: GeminiConfig, start_index: int = 0) -> str:
    """Prompt with globally-numbered words; the model picks index ranges."""
    lines = ["NUMBERED TRANSCRIPT (word index, seconds):"]
    for i, w in enumerate(chunk):
        lines.append(f'[{start_index + i}] {w["start"]:.2f}s {w["word"]}')
    transcript_text = "\n".join(lines)
    return f"""You select the most viral short-form clips from a video transcript.
Every transcript line starts with a [word_index]. Identify clip-worthy moments
BY WORD INDEX RANGE — do NOT output timestamps, the indices are the contract.
Timing window: clips must be {cfg.min_clip_sec}-{cfg.max_clip_sec} seconds long
(as shown next to each word). Pick {cfg.max_clips} best candidates max — fewer
is fine. Do NOT invent indices that are not in the transcript. Score 0-100 for
hook strength, pacing and emotional energy. hook_text is a short punchy phrase
that would work as the opening caption. hashtags: 3-6 short tags for the
platform.

Return ONLY a JSON array (no prose, no markdown fences):
[{{"start_word": 120, "end_word": 310, "score": 92, "title": "Short punchy title",
   "hook_text": "WAIT FOR IT", "hashtags": ["#viral", "#podcast"]}}]

{transcript_text}
""".strip()


def parse_candidates(text: str) -> list[dict[str, Any]]:
    """Extract word-index candidates from a model reply, tolerating fences."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m is None:
        raise LlmError(f"no JSON array in model reply: {text[:200]!r}")
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise LlmError(f"invalid JSON from model: {exc}") from exc
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start_word = int(item["start_word"])
            end_word = int(item["end_word"])
            score = float(item.get("score", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if end_word <= start_word:
            continue
        out.append(
            {
                "start_word": start_word,
                "end_word": end_word,
                "score": max(0.0, min(100.0, score)),
                "title": str(item.get("title", "Untitled clip"))[:120],
                "hook_text": str(item.get("hook_text", ""))[:200],
                "hashtags": [
                    str(t) for t in (item.get("hashtags") or []) if str(t).strip()
                ][:8],
            }
        )
    return out


def resolve_candidates(
    parsed: list[dict[str, Any]],
    words: list[dict],
    start_index: int,
    cfg: GeminiConfig,
) -> list[dict[str, Any]]:
    """Map word indices → real timestamps via Whisper timings.

    The transcript quote is derived LOCALLY from the selected words (the model
    is not asked to reproduce text, so it cannot paraphrase or hallucinate it).
    """
    out: list[dict[str, Any]] = []
    n = len(words)
    if n == 0:
        return out
    lo = start_index
    hi = start_index + n - 1
    for c in parsed:
        si = max(lo, min(hi, int(c["start_word"])))
        ei = max(lo, min(hi, int(c["end_word"])))
        if ei <= si:
            continue
        start = float(words[si]["start"])
        end = float(words[ei]["end"])
        # enforce the duration window (indices can drift out of range)
        if end - start < cfg.min_clip_sec:
            end = min(float(words[-1]["end"]), start + cfg.max_clip_sec)
        if end - start > cfg.max_clip_sec:
            end = start + cfg.max_clip_sec
        if end - start < 1.0:
            continue
        quote = " ".join(words[j]["word"] for j in range(si, ei + 1))[:300]
        out.append(
            {
                "start": start,
                "end": end,
                "score": c["score"],
                "title": c["title"],
                "hook_text": c["hook_text"],
                "transcript": quote,
                "hashtags": c["hashtags"],
            }
        )
    return out


def chunk_words(words: list[dict], target_seconds: float = 120.0) -> list[tuple[int, list[dict]]]:
    """Split a word-level transcript into ~2-minute chunks.

    Returns [(global_start_index, chunk_words), ...] so prompts can carry the
    same word numbering the resolver uses.
    """
    if not words:
        return []
    chunks: list[tuple[int, list[dict]]] = []
    current: list[dict] = []
    start_index = 0
    t0 = words[0]["start"]
    for i, w in enumerate(words):
        current.append(w)
        if w["end"] - t0 >= target_seconds:
            chunks.append((start_index, current))
            start_index += len(current)
            current = []
            t0 = w["end"]
    if current:
        chunks.append((start_index, current))
    return chunks


def _model_rotated(msg: str) -> str | None:
    """Return the recommended replacement model when Google retires the current one.

    The 404 body looks like: 'This model models/gemini-2.0-flash is no longer
    available. Please update your code to use models/gemini-3.6-flash ...'
    """
    if "no longer available" not in msg:
        return None
    m = _RECOMMENDED_MODEL.search(msg)
    if m:
        return m.group(1)
    m = _MODEL_RETIRED.search(msg)
    if m:
        return re.sub(r"-\d+$", "", m.group(1))
    return None


def _persist_model_fix(settings: Settings, new_model: str) -> None:
    """Persist a working model name into the [gemini] section of config.toml.

    Only the `model =` line inside `[gemini]` is touched — the whisper section
    has its own `model =` line that must stay intact.
    """
    from pathlib import Path as _Path

    from app.config import ROOT

    cfg_path = _Path(ROOT) / "config.toml"
    try:
        lines = cfg_path.read_text(encoding="utf-8").splitlines()
        section = ""
        replaced = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped[1:-1].strip().lower()
                continue
            if section == "gemini" and not replaced and re.match(r"^model\s*=", stripped):
                indent = line[: len(line) - len(line.lstrip())]
                comment = ""
                in_q = False
                for j, ch in enumerate(stripped):
                    if ch == '"':
                        in_q = not in_q
                    elif ch == "#" and not in_q:
                        comment = "  " + stripped[j:].rstrip()
                        break
                lines[i] = f'{indent}model = "{new_model}"{comment}'
                replaced = True
        if replaced:
            cfg_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
            logger.info(f"[llm] persisted model fix to config.toml [gemini]: {new_model}")
        else:
            logger.warning("[llm] could not find model = line under [gemini] in config.toml")
    except Exception as exc:
        logger.warning(f"[llm] could not persist model fix to config.toml: {exc}")
    # keep this process consistent immediately
    settings.gemini.model = new_model


def call_gemini(chunk_text_prompt: str, settings: Settings) -> str:
    """Single Gemini call with 429 backoff. Raises LlmError after retries."""
    genai = _load_google_module()
    model = settings.gemini.model
    # Generous per-call HTTP timeout: long transcripts (2h video = ~60 chunks)
    # can take a while per request, and SDK default timeouts abort mid-call.
    from google.genai import types as _types  # noqa: PLC0415

    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=_types.HttpOptions(timeout=120_000),
    )

    last_err: Exception | None = None
    for attempt in range(settings.gemini.max_retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=chunk_text_prompt)
            text = resp.text or ""
            if not text.strip():
                raise LlmError("Gemini returned an empty reply")
            return text
        except Exception as exc:  # noqa: BLE001 — SDK raises many types
            msg = str(exc)
            rate_limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            last_err = exc
            if attempt >= settings.gemini.max_retries:
                break
            if rate_limited:
                wait = min(60.0, (2 ** attempt) + random.random() * 2)
                logger.warning(f"[llm] 429 rate limited, backing off {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
            else:
                # Model retirement is not a transient error: retrying with the
                # same name can never succeed. Switch to Google's recommended
                # replacement immediately and keep going.
                replacement = _model_rotated(msg)
                if replacement and replacement != model:
                    logger.warning(
                        f"[llm] model {model!r} was retired by Google — self-healing to {replacement!r}"
                    )
                    model = replacement
                    _persist_model_fix(settings, replacement)
                    continue
                wait = 2.0 * (attempt + 1)
                logger.warning(f"[llm] gemini error ({msg[:140]}), retry {attempt + 1} in {wait:.0f}s")
                time.sleep(wait)
    raise LlmError(f"Gemini failed after retries: {last_err}")


def ollama_status(settings: Settings) -> tuple[bool, str]:
    """Is the local Ollama server reachable, and is the model pulled?

    Returns (ok, reason). Never raises — the dashboard renders this verbatim.
    """
    url = f"{settings.llm.ollama_url}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 - local http
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any transport error is "unreachable"
        return False, f"ollama server unreachable at {settings.llm.ollama_url} ({exc})"
    names = {
        str(m.get("name", ""))
        for m in (data.get("models") or [])
        if isinstance(m, dict)
    }
    want = settings.llm.ollama_model
    if names and not any(n == want or n.split(":")[0] == want.split(":")[0] for n in names):
        return False, f"model {want!r} is not pulled (have: {', '.join(sorted(names)) or 'none'})"
    return True, f"ollama ready: {want} @ {settings.llm.ollama_url}"


def call_ollama(chunk_text_prompt: str, settings: Settings) -> str:
    """Single local Ollama call over its plain HTTP API.

    TEXT ONLY: `chunk_text_prompt` is the numbered transcript built by
    build_prompt(). No audio, no video, no file handles ever leave this call.
    Uses urllib (stdlib) so the backend adds no Python dependency.
    """
    llm = settings.llm
    endpoint = f"{llm.ollama_url}/api/generate"
    payload = json.dumps(
        {
            "model": llm.ollama_model,
            # `format: json` makes Ollama constrain decoding to a JSON document.
            "prompt": chunk_text_prompt,
            "stream": False,
            "format": "json",
            "keep_alive": llm.ollama_keep_alive,
            "options": {"temperature": 0.4},
        }
    ).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - local http by design
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=llm.ollama_timeout_sec) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LlmNotConfigured(
            f"ollama server unreachable at {llm.ollama_url} ({exc.reason}) — "
            f"start it with `ollama serve`, or switch [llm] provider back to gemini"
        ) from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise LlmError(f"ollama call failed: {exc}") from exc
    text = str(body.get("response") or "")
    if not text.strip():
        raise LlmError("ollama returned an empty reply")
    return text


# Cloudflare (in front of api.groq.com) bans the default Python-urllib
# User-Agent with "403 Forbidden: error code: 1010" — a request-signature
# ban applied at the edge, so the API key is never even checked. Any
# non-Python-urllib UA gets through. Never remove this.
GROQ_USER_AGENT = "ClipForge/1.1 (+https://github.com/harry0703/MoneyPrinterTurbo)"


def call_groq(chunk_text_prompt: str, settings: Settings) -> str:
    """Single Groq chat-completions call with pacing + 429-aware backoff.

    TEXT ONLY: `chunk_text_prompt` is the numbered transcript built by
    build_prompt(). No audio, no video, no file handles ever leave this call.

    Two rate-limit defences, because the free tier is 30 req/min:
      1. Pacing — a process-wide floor between Groq calls, so a 60-chunk
         transcript never bursts into the limit in the first place.
      2. Backoff — on a 429 we honour the server's Retry-After when it sends
         one, else exponential + jitter, exactly like the Gemini path.

    Uses urllib (stdlib) so the backend adds no Python dependency, matching
    call_ollama. Returns the assistant message text; raises on failure.
    """
    cfg = settings.groq
    api_key = settings.groq_api_key
    if not api_key:
        raise LlmNotConfigured("GROQ_API_KEY missing — add it in the Keys/API keys tab.")

    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a video-clipping assistant. Reply with a single JSON "
                    "object and nothing else — no prose, no markdown fence."
                ),
            },
            {"role": "user", "content": chunk_text_prompt},
        ],
        "temperature": cfg.temperature,
    }

    last_err: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        _groq_wait_turn(cfg.min_interval_sec)
        try:
            data = _groq_post(body, api_key, cfg.base_url, cfg.timeout_sec)
            text = _groq_reply_text(data)
            if not text.strip():
                raise LlmError("Groq returned an empty reply")
            return text
        except _GroqHttpError as exc:
            msg = exc.detail
            last_err = exc
            # A model that no longer exists can never succeed on retry, and
            # Groq does not name a replacement the way Google does. Fail with
            # the exact reason rather than burning the retry budget.
            if exc.status in (400, 404) and _groq_model_problem(msg):
                raise LlmError(
                    f"Groq rejected model {cfg.model!r}: {msg[:200]}. "
                    f"Pick a current model from console.groq.com/docs/models and set "
                    f"[groq] model in config.toml."
                ) from exc
            if attempt >= cfg.max_retries:
                break
            if exc.status == 403 and "1010" in msg:
                # Cloudflare edge ban on the request signature, NOT a bad key.
                # Retrying cannot help, so fail immediately with the real cause.
                raise LlmError(
                    "Groq returned 403 / Cloudflare error 1010. This is NOT a bad API key "
                    f"— Cloudflare blocked the request before it reached Groq. "
                    f"Sent User-Agent: {GROQ_USER_AGENT!r}."
                ) from exc
            if exc.status == 429:
                wait = exc.retry_after or min(60.0, (2 ** attempt) + random.random() * 2)
                logger.warning(
                    f"[llm] groq 429 rate limited (free tier ~30 rpm), "
                    f"backing off {wait:.1f}s (attempt {attempt + 1})"
                )
            else:
                wait = 2.0 * (attempt + 1)
                logger.warning(
                    f"[llm] groq error ({exc.status} {msg[:140]}), "
                    f"retry {attempt + 1} in {wait:.0f}s"
                )
            time.sleep(wait)
        except (TimeoutError, json.JSONDecodeError, LlmError) as exc:
            last_err = exc
            if attempt >= cfg.max_retries:
                break
            wait = 2.0 * (attempt + 1)
            logger.warning(f"[llm] groq transport error ({exc}), retry {attempt + 1} in {wait:.0f}s")
            time.sleep(wait)
    raise LlmError(f"Groq failed after retries: {last_err}")


class _GroqHttpError(Exception):
    """A non-2xx from Groq, carrying the status and any Retry-After."""

    def __init__(self, status: int, detail: str, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


# Groq names a model that is gone in words like "decommissioned" / "no longer
# available" / "does not exist". Matched loosely because the exact phrasing
# has changed between API versions.
_GROQ_MODEL_PROBLEM = re.compile(
    r"decommissioned|no longer (?:available|supported|exists)|does not exist|is not a (?:valid|supported) model",
    re.IGNORECASE,
)


def _groq_model_problem(msg: str) -> bool:
    return bool(_GROQ_MODEL_PROBLEM.search(msg))


_groq_lock = threading.Lock()
_groq_last_call: float = 0.0


def _groq_wait_turn(floor_sec: float) -> None:
    """Block until the configured floor between Groq calls has elapsed.

    Process-wide (not per-job) on purpose: a long transcript is many calls
    and they must be paced against the ACCOUNT limit, not each other.
    """
    if floor_sec <= 0:
        return
    global _groq_last_call
    with _groq_lock:
        now = time.monotonic()
        wait = _groq_last_call + floor_sec - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _groq_last_call = now


def _groq_post(
    body: dict[str, Any], api_key: str, base_url: str, timeout_sec: int
) -> dict[str, Any]:
    """POST to {base_url}/chat/completions and return the decoded JSON body."""
    base = str(base_url).rstrip("/")
    last_400 = ""
    # Try with a JSON-mode hint first; not every Groq model accepts it, so a
    # 400 that names response_format degrades to a plain call rather than
    # failing the whole job over an output-formatting nicety.
    for attempt_json_mode in (True, False):
        req_body = dict(body)
        if attempt_json_mode:
            req_body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(req_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                # REQUIRED. api.groq.com sits behind Cloudflare, which returns
                # "HTTP 403: error code: 1010" — a signature ban on the
                # default `Python-urllib/3.x` User-Agent — BEFORE the request
                # reaches Groq, so a perfectly valid key still fails. Verified:
                # urllib default -> 403/1010, an explicit UA -> 401 (reaches
                # the API). Do not remove this header.
                "User-Agent": GROQ_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - body is best-effort context only
                raw = ""
            detail = raw or exc.reason or ""
            retry_after: float | None = None
            try:
                if exc.headers is not None:
                    hdr = exc.headers.get("Retry-After")
                    if hdr:
                        retry_after = float(hdr)
            except (TypeError, ValueError):
                retry_after = None
            if exc.code == 400 and "response_format" in detail and attempt_json_mode:
                last_400 = detail
                continue
            raise _GroqHttpError(exc.code, detail, retry_after) from exc
    raise _GroqHttpError(400, last_400 or "groq rejected the request")


def _groq_reply_text(data: dict[str, Any]) -> str:
    """Pull the assistant message out of an OpenAI-shaped chat completion."""
    choices = data.get("choices") or []
    if not choices:
        # Some gateway errors come back 200 with an error envelope.
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            raise _GroqHttpError(200, str(err["message"]))
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def call_llm(chunk_text_prompt: str, settings: Settings) -> str:
    """Dispatch one detection call to the configured backend."""
    active = provider(settings)
    if active == PROVIDER_OLLAMA:
        return call_ollama(chunk_text_prompt, settings)
    if active == PROVIDER_GEMINI:
        return call_gemini(chunk_text_prompt, settings)
    return call_groq(chunk_text_prompt, settings)


class ChunkCache:
    """Per-chunk detection cache (duck-typed: needs `get` and `put`).

    Detection is the only stage that costs MONEY and takes minutes: a 15-minute
    source is 7 Groq calls. Without a per-chunk cache, a host restart at chunk 5
    threw away chunks 1-5 and re-paid for all 7 on the retry — which is both
    slow and, on a free tier, how you run out of quota.

    Keeping the interface this small means `detect_clips` stays free of
    filesystem concerns and remains unit-testable with a dict.
    """

    def get(self, index: int) -> list[dict[str, Any]] | None:
        raise NotImplementedError

    def put(self, index: int, candidates: list[dict[str, Any]]) -> None:
        raise NotImplementedError


class DictChunkCache(ChunkCache):
    """In-memory chunk cache (tests, and the no-persistence default)."""

    def __init__(self) -> None:
        self._data: dict[int, list[dict[str, Any]]] = {}

    def get(self, index: int) -> list[dict[str, Any]] | None:
        return self._data.get(index)

    def put(self, index: int, candidates: list[dict[str, Any]]) -> None:
        self._data[index] = candidates


def detect_clips(
    words: list[dict],
    settings: Settings,
    *,
    min_clip_sec: int | None = None,
    max_clip_sec: int | None = None,
    max_clips: int | None = None,
    on_log=None,
    chunk_cache: ChunkCache | None = None,
    should_pause=None,
) -> list[dict[str, Any]]:
    """Full detection: chunk → LLM (word indices) → resolve timings → top-N.

    Returns MPT-style dicts: start, end, score, title, hook_text.
    Raises LlmNotConfigured when no key, LlmError when the API fails after
    retries. There is deliberately no heuristic fallback here.

    `chunk_cache` persists each chunk's resolved candidates so a resume re-runs
    only the chunks that never finished. `should_pause` is polled at each chunk
    boundary — the one safe point in this stage — so a user-requested pause can
    stop the job between API calls instead of mid-write.
    """
    if provider(settings) == PROVIDER_OLLAMA:
        # No API key by design. The only precondition is a reachable server.
        ok, reason = ollama_status(settings)
        if not ok:
            raise LlmNotConfigured(
                f"Ollama backend selected but unusable — {reason}. "
                f"Run `ollama serve` and `ollama pull {settings.llm.ollama_model}`, "
                f"or switch the detection backend back to the hosted one."
            )
    elif provider(settings) == PROVIDER_GROQ:
        if not settings.groq_api_key:
            live = live_groq_key()
            live.refresh(force=True)
            detail = live.problem or (
                f"no key found under any supported name ({', '.join(live.status()['aliases'])})"
            )
            raise LlmNotConfigured(
                f"GROQ_API_KEY missing — add it in the Keys/API keys tab (or .env), "
                f"then hit Re-check. Detail: {detail}"
            )
    elif not settings.gemini_api_key:
        live = live_gemini_key()
        live.refresh(force=True)
        detail = live.problem or (
            f"no key found under any supported name ({', '.join(live.status()['aliases'])})"
        )
        raise LlmNotConfigured(
            f"GEMINI_API_KEY missing — add it in the Keys/API keys tab (or .env), "
            f"then hit Re-check. Detail: {detail}"
        )

    cfg = settings.gemini
    if min_clip_sec is not None:
        cfg = GeminiConfig(cfg.model, min_clip_sec, cfg.max_clip_sec, cfg.max_clips, cfg.max_retries)
    if max_clip_sec is not None:
        cfg = GeminiConfig(cfg.model, cfg.min_clip_sec, max_clip_sec, cfg.max_clips, cfg.max_retries)
    if max_clips is not None:
        cfg = GeminiConfig(cfg.model, cfg.min_clip_sec, cfg.max_clip_sec, max_clips, cfg.max_retries)

    chunks = chunk_words(words)
    label = provider_label(settings)
    logger.info(f"[llm] detecting on {len(chunks)} chunk(s), {len(words)} words via {label}")
    if on_log:
        on_log(f"Detecting highlights on {len(chunks)} chunk(s) via {label}...")

    candidates: list[dict[str, Any]] = []
    reused = 0
    for i, (start_index, chunk) in enumerate(chunks):
        # Pause is honoured at a chunk boundary: never between the API call
        # and the cache write, which is the window where a kill would lose
        # work we already paid for.
        if should_pause is not None and should_pause():
            raise DetectionPaused(
                f"paused after chunk {i}/{len(chunks)} — {reused} chunk(s) reused, "
                f"{len(candidates)} candidate(s) kept"
            )
        cached = chunk_cache.get(i) if chunk_cache is not None else None
        if cached is not None:
            candidates.extend(cached)
            reused += 1
            if on_log:
                on_log(
                    f"Detection: chunk {i + 1}/{len(chunks)} → {len(cached)} candidates "
                    f"(cached — not re-billed)"
                )
            logger.info(
                f"[llm] chunk {i + 1}/{len(chunks)}: reusing {len(cached)} cached candidates"
            )
            continue
        prompt = build_prompt(chunk, cfg, start_index=start_index)
        text = call_llm(prompt, settings)
        parsed = parse_candidates(text)
        resolved = resolve_candidates(parsed, words, start_index, cfg)
        # Persist BEFORE reporting progress, so a kill in the next instant
        # still keeps this chunk.
        if chunk_cache is not None:
            chunk_cache.put(i, resolved)
        candidates.extend(resolved)
        if on_log:
            on_log(f"Detection: chunk {i + 1}/{len(chunks)} → {len(resolved)} candidates")
        logger.info(f"[llm] chunk {i + 1}/{len(chunks)}: {len(resolved)} candidates")
    if reused:
        logger.info(
            f"[llm] detection resumed: reused {reused}/{len(chunks)} chunk(s) from cache, "
            f"re-billed only the remainder"
        )

    # keep only the top N by score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[: cfg.max_clips]
    if not top:
        raise LlmError(f"{label} returned no valid clip candidates")
    logger.info(f"[llm] selected {len(top)} clips (scores {top[0]['score']:.0f}…{top[-1]['score']:.0f})")
    return top
