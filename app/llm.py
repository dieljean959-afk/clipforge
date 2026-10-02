"""Clip/highlight detection (LLM service).

Chunked transcript → LLM → JSON [{start_word, end_word, score, title,
hook_text}] with 429-aware exponential backoff. The model returns WORD
INDICES, never timestamps — fragile float math is the LLM's weak spot, so we
resolve the real start/end from Whisper's word-level timings afterwards.

Two interchangeable hosted backends plus a local one, chosen by `[llm]
provider` / CLIPFORGE_LLM_PROVIDER:
  - nvidia  (default) — NVIDIA NIM, OpenAI-compatible chat completions. The
                       primary because it has no per-model DAILY token ceiling
                       like Groq's free tier, so a long source cannot die half
                       way through detection on an invisible quota.
  - groq             — the backup. Same wire format, same code path; its free
                       tier is ~30 req/min, so calls are PACED client-side and
                       429s are retried with Retry-After-aware backoff.
  - gemini           — hosted alternative, kept selectable without a redeploy.
  - ollama           — self-hosted. Free and fully private, but it competes for
                       the SAME CPU as Whisper, so it runs sequentially and is
                       markedly slower. Requires no API key.

`[llm] fallback_provider` (default: groq) names a second backend to try for a
chunk when the primary fails — quota spent, endpoint down, key rejected. It is
opt-in, it is logged per swap, and a missing key on the PRIMARY is still a hard
config error rather than a silent downgrade: a key mistake should be fixed, not
worked around invisibly.

The contract is deliberately narrow: every backend takes a prompt STRING and
returns a reply STRING. `parse_candidates` is the only thing that knows what
the reply looks like, which is what makes swapping providers safe.

Only transcript text is ever sent to any backend — never audio or video.
The key-missing case is surfaced loudly: no silent heuristic fallback exists
in this module. Provider FAILOVER, by contrast, is explicit and configured
(`[llm] fallback_provider`): a chunk that fails on the primary is retried on
the named backup, the swap is logged with both provider names, and every chunk
of that job records which model actually answered.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent import futures as cf
from dataclasses import replace
from typing import Any

from loguru import logger

from app import clipcount, jsonfix
from app.caption_format import finalize_clip_caption, normalize_hashtags, shorten_caption
from app.config import (
    GeminiConfig,
    Settings,
    live_deepseek_key,
    live_gemini_key,
    live_groq_key,
    live_nvidia_key,
)

PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_NVIDIA = "nvidia"
PROVIDER_GROQ = "groq"
PROVIDER_GEMINI = "gemini"
PROVIDER_OLLAMA = "ollama"
PROVIDERS = (
    PROVIDER_DEEPSEEK,
    PROVIDER_NVIDIA,
    PROVIDER_GROQ,
    PROVIDER_GEMINI,
    PROVIDER_OLLAMA,
)

#: The hosted OpenAI-compatible backends, in the order they are preferred.
#: `key_attr` is the Settings property holding the live key; `live()` is the
#: resolver used for the "no key" error message, so a new provider is one line
#: here plus one in `openai_cfg` rather than a scatter of if/elif branches.
OPENAI_PROVIDER_KEYS: dict[str, tuple[str, str]] = {
    PROVIDER_DEEPSEEK: ("deepseek_api_key", "DEEPSEEK_API_KEY"),
    PROVIDER_NVIDIA: ("nvidia_api_key", "NVIDIA_API_KEY"),
    PROVIDER_GROQ: ("groq_api_key", "GROQ_API_KEY"),
}
_LIVE_KEY_RESOLVERS = {
    PROVIDER_DEEPSEEK: live_deepseek_key,
    PROVIDER_NVIDIA: live_nvidia_key,
    PROVIDER_GROQ: live_groq_key,
}


#: Ceiling on detection API calls per job, whatever the source length, so a
#: 2-hour interview costs the same number of requests as a 20-minute one.
MAX_DETECTION_CHUNKS = 20


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
    return getattr(getattr(settings, "llm", None), "provider", PROVIDER_DEEPSEEK)


def provider_label(settings: Settings) -> str:
    """Human-readable backend name for logs and the dashboard."""
    active = provider(settings)
    if active == PROVIDER_OLLAMA:
        return f"ollama ({settings.llm.ollama_model})"
    if active == PROVIDER_GEMINI:
        return f"gemini ({settings.gemini.model})"
    return f"{active} ({openai_cfg(settings, active).model})"


def settings_llm_fallback(settings: Settings) -> str:
    """The configured backup provider, or "" when failover is off.

    Normalised here rather than at each use site so "unset", "blank" and
    "pointed at the primary" all mean the same thing: no failover. A fallback
    equal to the primary would burn a second full ladder proving what the first
    one already proved.
    """
    backup = str(getattr(settings.llm, "fallback_provider", "") or "").strip().lower()
    if not backup or backup == provider(settings):
        return ""
    return backup


def openai_cfg(settings: Settings, which: str | None = None) -> Any:
    """The OpenAI-compatible config for a hosted provider (deepseek/nvidia/groq).

    All three share one call path, so everything downstream takes the CONFIG
    rather than the provider name — the name is only needed for text (logs,
    errors) and for pacing isolation, both of which the config carries.
    """
    name = which or provider(settings)
    if name == PROVIDER_GROQ:
        return settings.groq
    if name == PROVIDER_NVIDIA:
        return settings.nvidia
    return settings.deepseek


def openai_api_key(which: str, settings: Settings) -> str | None:
    """The live key for a hosted provider, resolved on every read."""
    attr, _ = OPENAI_PROVIDER_KEYS.get(which, ("", ""))
    return getattr(settings, attr) if attr else None


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
    """Prompt with globally-numbered words; the model picks index ranges.

    The per-chunk ask is deliberately NOT the global clip cap. Asking a 2-minute
    chunk for 40 clips invites the model to pad the chunk with its weakest
    moments so the quota is met — and since selection is a global top-N by
    score afterwards, those padded candidates are the ones that crowd out good
    ones from other chunks. A small per-chunk number and a large global cap
    gives a much better pool to rank.
    """
    per_chunk = max(3, min(8, cfg.max_clips))
    lines = ["NUMBERED TRANSCRIPT (word index, seconds):"]
    for i, w in enumerate(chunk):
        lines.append(f'[{start_index + i}] {w["start"]:.2f}s {w["word"]}')
    transcript_text = "\n".join(lines)
    return f"""You select the most viral short-form clips from a video transcript.
Every transcript line starts with a [word_index]. Identify clip-worthy moments
BY WORD INDEX RANGE — do NOT output timestamps, the indices are the contract.
Timing window: clips must be {cfg.min_clip_sec}-{cfg.max_clip_sec} seconds long
(as shown next to each word). Pick at most {per_chunk} of the best moments in
this section — fewer is fine, and a section with nothing strong should return
an empty list rather than its least bad moment. Do NOT invent indices that are
not in the transcript. Score 0-100 for hook strength, pacing and emotional
energy; only moments you would genuinely call a highlight belong above 60.

hook_text is the POST CAPTION: ONE line, under 150 characters, no newlines, no
repetition of the title, no "in this video..." preamble. It must work on its own
as the line a viewer reads under the clip.

hashtags: 5 to 7 tags total, mixing exactly:
  - 2-3 discovery tags from: #fyp #foryou #viral
  - 2-4 CONTENT tags naming what this specific clip is actually about, drawn
    from the transcript. Generic filler that would suit any video (#love,
    #instagood, #explore) is forbidden -- every tag must be one you would only
    use for THIS clip. No duplicates, lowercase, no punctuation.

Return ONLY a JSON array (no prose, no markdown fences):
[{{"start_word": 120, "end_word": 310, "score": 92, "title": "Short punchy title",
   "hook_text": "one line of post copy",
   "hashtags": ["#fyp", "#foryou", "#viral", "#<topic>", "#<topic>"]}}]

{transcript_text}
""".strip()


def _raw_reply_excerpt(text: str, limit: int = 1200) -> str:
    """The model's reply, verbatim but bounded, for the job log.

    Detection failures used to be undiagnosable because the only evidence was
    "no valid clip candidates" — which does not distinguish an empty reply, a
    markdown fence around a perfectly good object, a reply truncated by the
    token limit, prose with the JSON glued to the end, or a rate-limit page
    served with a200. The raw text answers that in one line.
    """
    if not text:
        return "<empty reply>"
    body = text.strip()
    if len(body) <= limit:
        return body
    head = body[: limit // 2]
    tail = body[-limit // 2 :]
    return f"{head}\\n…[{len(body) - limit} chars elided]…\\n{tail}"


def parse_candidates(text: str) -> list[dict[str, Any]]:
    """Extract word-index candidates from a possibly-noisy model reply.

    Delegates the hard part to `app.jsonfix`, which strips fences and prose,
    closes a reply truncated by the token limit, tolerates trailing commas, and
    unwraps whichever envelope the reply used — including the single clip OBJECT
    that `json_object` mode produces when the prompt asked for an array.

    NEVER raises. An unreadable reply yields an empty list and the caller treats
    that as "this chunk had no highlights" rather than a failed job: the
    previous implementation raised on a fenced or truncated reply, and one such
    chunk at 3/7 killed the whole 20-chunk run.
    """
    out: list[dict[str, Any]] = []
    for item in jsonfix.repair_candidates(text):
        # The prompt asks for one line of copy and 5-7 tags; these two calls
        # enforce that rather than trusting the reply, which is routinely
        # multi-line and over-tagged.
        out.append(
            {
                **item,
                "hook_text": shorten_caption(item.get("hook_text", "")),
                "hashtags": normalize_hashtags(item.get("hashtags")),
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
        # Applied HERE rather than in parse_candidates: this is the first point
        # where the clip's real transcript text exists, and the content-tag
        # top-up is mined from it. Doing it earlier could only pad with generic
        # discovery tags.
        hook_text, hashtags = finalize_clip_caption(c["hook_text"], c["hashtags"], quote)
        out.append(
            {
                "start": start,
                "end": end,
                "score": c["score"],
                "title": c["title"],
                "hook_text": hook_text,
                "transcript": quote,
                "hashtags": hashtags,
            }
        )
    return out


def chunk_words(
    words: list[dict],
    target_seconds: float = 120.0,
    max_chunks: int = MAX_DETECTION_CHUNKS,
) -> list[tuple[int, list[dict]]]:
    """Split a word-level transcript into chunks for the detector.

    Returns [(global_start_index, chunk_words), ...] so prompts can carry the
    same word numbering the resolver uses.

    The chunk length is not fixed: it grows for long sources so the NUMBER of
    API calls stays bounded. A 2-minute target meant 28 calls for a 55-minute
    interview and 60 for a 2-hour one, at ~4k tokens each — which walks
    straight into a tokens-per-minute limit and then dies on a 429 part way
    through, having already paid for most of the work. Detection is a
    fixed-cost-per-chunk operation, so a bounded number of slightly larger
    chunks is both cheaper and far more likely to finish. Clip candidates are
    picked by word index, so a wider window costs resolution in *search*, not
    in timing accuracy.
    """
    if not words:
        return []
    if max_chunks > 0:
        span = float(words[-1].get("end") or 0.0) - float(words[0].get("start") or 0.0)
        if span > 0:
            target_seconds = max(target_seconds, span / float(max_chunks))
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
                # Google states the wait explicitly — "Please retry in
                # 47.352748833s", and a structured RetryInfo.retryDelay — so
                # honour it as a floor. The old fixed 1+2+3s backoff spent the
                # whole retry budget against a per-minute quota that needed
                # 47s, and turned a recoverable 429 into a failed job.
                asked = _retry_hint(msg)
                if asked is None:
                    m = re.search(r"retryDelay'?\"?\s*:\s*'?\"?(\d+)", msg)
                    if m:
                        asked = float(m.group(1))
                if asked is not None and asked > MAX_INLINE_RATE_LIMIT_WAIT:
                    raise LlmError(
                        f"Gemini quota needs {asked / 60:.0f} minutes before it will "
                        f"accept another call — that is a daily limit, not a burst "
                        f"one. {msg[:220]}"
                    )
                fallback = min(60.0, (2 ** attempt) * 2.0 + random.random() * 2)
                wait = max(asked or 0.0, fallback)
                logger.warning(
                    f"[llm] 429 rate limited"
                    f"{f', provider asked for {asked:.0f}s' if asked else ''}"
                    f" — backing off {wait:.1f}s (attempt {attempt + 1})"
                )
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

#: The clip contract, as a JSON Schema for Groq's strict Structured Outputs.
#:
#: The top level is an OBJECT holding a `clips` array, and that is the entire
#: fix for `json_validate_failed`. The prompt has always asked for a top-level
#: array, which `json_object` cannot represent; under strict schema the shape
#: is enforced by constrained decoding, so the model cannot emit a mismatched
#: payload no matter how the prompt is worded. `parse_candidates` unwraps the
#: envelope.
CLIP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_word": {"type": "integer"},
                    "end_word": {"type": "integer"},
                    "score": {"type": "number"},
                    "title": {"type": "string"},
                    "hook_text": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                # Strict mode requires every property listed as required and
                # additionalProperties:false. The natural-language fields are
                # required too — parse_candidates defaults them, but asking for
                # them keeps the decoder from skipping them.
                "required": [
                    "start_word", "end_word", "score", "title", "hook_text", "hashtags",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

#: Groq error text that means "the model produced JSON we could not accept",
#: as opposed to "your request was wrong". These are RETRYABLE and, crucially,
#: must not be treated as permanent 400s — the previous code failed the whole
#: job on any 400, which is how one bad chunk at 3/7 killed a 20-chunk run.
_JSON_VALIDATION = re.compile(
    r"json_validate_failed|failed to validate json|failed to generate json|"
    r"invalid json|json_schema_validation|does not match the expected schema",
    re.IGNORECASE,
)

#: A model that does not support Structured Outputs at all. Not a JSON-content
#: problem and not a bad request either: the feature is simply unavailable for
#: it, so the fix is to drop `response_format` and repair the reply locally.
#: Retrying the identical request can only ever earn the same 400.
_NO_RESPONSE_FORMAT = re.compile(
    r"response_format.{0,40}not supported|"
    r"structured output.{0,40}not supported|"
    r"does not support (?:json_schema|response_format|structured output)",
    re.IGNORECASE,
)


def _is_json_validation_error(detail: str) -> bool:
    """True when a 400 means 'bad model output', not 'bad request'."""
    return bool(_JSON_VALIDATION.search(detail or ""))


#: An exhausted balance is a BILLING state, not a transport hiccup. DeepSeek
#: answers 402 "Insufficient Balance" and OpenAI-compatible gateways answer 403
#: with the same words; neither is fixed by waiting, and the wording is what
#: distinguishes them from a real permission problem, where a different key
#: WOULD help and the operator deserves to be told exactly that.
_BILLING_EXHAUSTED = re.compile(
    r"insufficient[_ ]balance|insufficient[_ ]funds|insufficient[_ ]credit|"
    r"billing|out of credit|payment required|quota exceeded|"
    r"exceeded your current quota",
    re.IGNORECASE,
)


def _is_billing_exhausted(status: int, detail: str) -> bool:
    """True when the account is out of credit — permanent until topped up.

    This is the single most expensive non-failure in the ladder. A 402 falls
    through to the generic retry branch, so one exhausted balance cost
    ``max_retries`` backoffs, then all four JSON-ladder rungs, then the
    fallback provider — roughly 15 doomed HTTP calls before the job saw the
    error. Called once per caption language per clip, that turned a 13-clip
    batch into an apparent infinite loop pinned at "clip 1 of 13, 0%".
    """
    if _BILLING_EXHAUSTED.search(detail or ""):
        return True
    # 402 is Payment Required and means exactly one thing on these APIs.
    return status == 402


def _format_unsupported(detail: str) -> bool:
    """True when the model cannot do Structured Outputs at all."""
    return bool(_NO_RESPONSE_FORMAT.search(detail or ""))


def _response_format_for(mode: str) -> dict[str, Any] | None:
    """The wire `response_format` for a configured mode name."""
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "clips", "strict": True, "schema": CLIP_SCHEMA},
        }
    return None


class _JsonRetry(Exception):
    """Internal: this attempt produced unusable JSON, escalate the ladder.

    `transient` separates the two reasons a rung can fail, and they need
    OPPOSITE treatment. A provider that rejected the OUTPUT (a 400
    json_validate_failed) is healthy — the next rung is a different STRATEGY
    (lower temperature, bigger budget, no response_format, another model) and
    waiting changes nothing. A provider that is rate limiting, 5xx-ing or
    dropping the connection is UNHEALTHY, and only a pause gives it room.

    Getting this backwards cost 7 seconds of dead time per doomed model on
    every chunk: a run that could not possibly succeed took 22.6s to admit it.
    """

    transient: bool = False

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


class _FormatUnsupported(Exception):
    """Internal: this model has no Structured Outputs; retry without it."""


class _ModelExhausted(Exception):
    """Internal: every model in the chain failed for this chunk."""


def call_groq(chunk_text_prompt: str, settings: Settings) -> str:
    """One detection call to Groq. See `_call_openai_chat` for the ladder."""
    return _call_openai_chat(chunk_text_prompt, settings, PROVIDER_GROQ)


def call_nvidia(chunk_text_prompt: str, settings: Settings) -> str:
    """One detection call to NVIDIA NIM.

    Identical contract and identical ladder to `call_groq` — NIM and Groq are
    the same wire format, so this is the same function with a different config
    object. Kept as its own name so the dispatch in `call_llm` and the tests
    read the same way they do for Groq.
    """
    return _call_openai_chat(chunk_text_prompt, settings, PROVIDER_NVIDIA)


def call_deepseek(chunk_text_prompt: str, settings: Settings) -> str:
    """One detection call to the DeepSeek platform.

    Same OpenAI-compatible wire format and the same ladder as Groq/NIM, so this
    is the same function pointed at a different config object. DeepSeek is a
    provider in its own right, NOT the NVIDIA-hosted `deepseek-*` models: its
    keys are `sk-…` from platform.deepseek.com and are billed against a
    balance, not an NVIDIA quota.
    """
    return _call_openai_chat(chunk_text_prompt, settings, PROVIDER_DEEPSEEK)


def _call_openai_chat(chunk_text_prompt: str, settings: Settings, which: str) -> str:
    """One detection call to an OpenAI-compatible backend, with a repair ladder.

    TEXT ONLY: `chunk_text_prompt` is the numbered transcript built by
    build_prompt(). No audio, no video, no file handles ever leave this call.

    THE LADDER
    ----------
    A chunk used to be all-or-nothing: one malformed reply and
    `json_validate_failed` (400) failed the entire job, discarding the 19
    chunks already paid for and cached. Detection now degrades instead:

      1. the configured `response_format` (strict `json_schema` on Groq,
         `json_object` on NIM);
      2. on a JSON-validation failure, up to `json_retries` attempts at LOWER
         temperature and HIGHER `max_tokens` — truncation at the token ceiling
         is the usual real cause, so the budget grows;
      3. then a `response_format`-free attempt at the ceiling budget, whose
         reply is repaired locally by `app.jsonfix` (fences, prose, trailing
         commas, truncation);
      4. then the same ladder against each model in `fallback_models`.

    A 400 is PERMANENT only when it is not a JSON-validation error, so a
    genuinely malformed request still fails fast rather than re-billing.

    Rate-limit defences: a process-wide request floor and a token pacer, both
    ISOLATED PER PROVIDER (an NIM call must never be throttled by Groq's
    budget). On 429 the server's own wait is honoured, and a daily-quota wait
    fails fast rather than sleeping inline for an hour.

    Returns the assistant message text. Raises LlmNotConfigured with no key,
    LlmError when the whole chain is exhausted.
    """
    cfg = openai_cfg(settings, which)
    api_key = openai_api_key(which, settings)
    if not api_key:
        raise LlmNotConfigured(
            f"{cfg.label.upper()}_API_KEY missing — add it in the Keys/API keys tab."
        )

    prompt = _groq_envelope_prompt(chunk_text_prompt, cfg.response_format)
    chain = [cfg.model, *[m for m in cfg.fallback_models if m and m != cfg.model]]
    problems: list[str] = []

    for model in chain:
        try:
            return _groq_attempt_chain(prompt, cfg, model, api_key, problems)
        except _ModelExhausted:
            logger.warning(
                f"[llm] {cfg.label} model {model} could not produce valid JSON for this "
                f"chunk after the full ladder; trying the next fallback model"
            )
        except LlmError:
            # A non-JSON failure (bad key, dead model, exhausted quota) is not
            # worth re-asking a different model — surface it immediately.
            raise

    detail = "; ".join(problems[-3:])
    raise LlmError(
        f"Every model in the {cfg.label} detection chain failed to return valid JSON "
        f"for this chunk. Tried: {', '.join(chain)}. Last problems: {detail[:400]}"
    )


def _groq_envelope_prompt(prompt: str, response_format: str) -> str:
    """Align the ASK with the envelope the response_format actually enforces.

    Under `json_schema`/`json_object` the top level must be an object, so a
    prompt saying "return a JSON array" contradicts the wire contract — the
    exact contradiction that produced the original 400s. Under `plain` the
    legacy array wording is kept, because there is no envelope to satisfy and
    `jsonfix` unwraps either shape.
    """
    if response_format == "plain":
        return prompt
    return prompt.replace(
        "Return ONLY a JSON array (no prose, no markdown fences):",
        "Return ONLY a JSON object shaped {\"clips\": [...]} — the top level must "
        "be an object whose \"clips\" key holds the array (no prose, no fences):",
    )


def _groq_attempt_chain(
    prompt: str, cfg: Any, model: str, api_key: str, problems: list[str]
) -> str:
    """Run the full ladder against ONE model. Raises _ModelExhausted on failure.

    The last rung deliberately drops `response_format` and asks the model to
    repair its own JSON, with the largest token budget. That is the rung that
    survives a model whose structured-output path is broken but whose plain
    generation is fine — and `app.jsonfix` cleans up whatever comes back.
    """
    last_exc: _JsonRetry | None = None
    for attempt in range(cfg.json_retries + 1):
        # temperature can only go DOWN from the configured base
        temp = max(0.0, min(cfg.temperature, cfg.json_retry_temperature))
        if attempt >= cfg.json_retries:
            # Final rung: no response_format, largest budget.
            mode, budget = "plain", cfg.max_tokens_ceiling
        else:
            # Grow the budget geometrically: 3200 -> 6400 -> 8192 (capped).
            mode = cfg.response_format
            budget = min(
                cfg.max_tokens_ceiling,
                cfg.max_tokens * (2 ** attempt) if attempt else cfg.max_tokens,
            )
        # Back off between ladder steps ONLY when the provider is the problem.
        #
        # The steps used to be fired back to back with no pause (a provider
        # that rejected one JSON reply got the identical retry inside the same
        # second, and the whole ladder burned itself out in under a second),
        # so a blanket 1s, 2s, 4s… was added. That fixed the spin and created
        # a worse problem: the common failure is a model whose output does not
        # satisfy the schema, which no amount of waiting can fix, so every
        # doomed chunk paid 7s per model before admitting it.
        #
        # Now there is no ladder sleep at all, because a provider-health
        # failure no longer continues up the ladder: it breaks out to the NEXT
        # MODEL (see the transient branch below). A rate limited model is not
        # going to start honouring json_schema after a sleep, so climbing its
        # remaining rungs was four wasted calls and ~30s of sleeping per model.
        if attempt and last_exc is not None and last_exc.transient:
            # Unreachable in practice (the transient branch breaks), but kept so
            # a future rung that decides to continue a rate-limited model has a
            # stated policy rather than an accidental instant retry.
            backoff = min(16.0, 2.0 ** (attempt - 1)) * (0.75 + random.random() * 0.5)
            logger.info(
                f"[llm] backing off {backoff:.1f}s before JSON ladder step "
                f"{attempt + 1}/{cfg.json_retries + 1} "
                f"(previous rung failed on a rate limit or transport error)"
            )
            time.sleep(backoff)
        try:
            return _groq_single(prompt, cfg, model, api_key, mode, temp, budget)
        except _FormatUnsupported as exc:
            # The model cannot do Structured Outputs at all. Skip straight to
            # the no-format rung rather than repeating the same rejected call.
            problems.append(f"{model}/{mode}: {str(exc)[:140]}")
            logger.warning(
                f"[llm] {model} does not support {mode} — retrying without "
                f"response_format and repairing the JSON locally"
            )
            try:
                return _groq_single(
                    prompt, cfg, model, api_key, "plain", temp, cfg.max_tokens_ceiling
                )
            except (_JsonRetry, _FormatUnsupported) as inner:
                problems.append(f"{model}/plain: {str(inner)[:140]}")
                if isinstance(inner, _JsonRetry):
                    last_exc = inner
                break
        except _JsonRetry as exc:
            last_exc = exc
            reason = str(exc)
            problems.append(f"{model}/{mode}/t{temp}/mt{budget}: {reason[:140]}")
            logger.warning(
                f"[llm] JSON ladder step {attempt + 1}/{cfg.json_retries + 1} for "
                f"{model} (mode={mode}, temp={temp}, max_tokens={budget}): {reason[:180]}"
            )
            if exc.transient:
                # Rate limited, 5xx-ing or dropping the connection — the MODEL
                # is unhealthy, its JSON strategy is irrelevant. Climbing the
                # remaining rungs of this model would be three more calls and
                # three more sleeps to learn the same thing. Go to the next
                # model in the chain, which is the only move that can help.
                logger.warning(
                    f"[llm] {model} is rate limited or unreachable — skipping its "
                    f"remaining {cfg.json_retries - attempt} ladder step(s) and "
                    f"trying the next model instead"
                )
                break
    raise _ModelExhausted(problems[-1] if problems else "no usable JSON")


def _groq_single(
    prompt: str, cfg: Any, model: str, api_key: str, mode: str,
    temperature: float, max_tokens: int,
) -> str:
    """One HTTP rung with pacing, 429 backoff and JSON-aware 400 handling.

    Raises `_JsonRetry` when the provider rejected the OUTPUT (retryable — the
    ladder should escalate), and `LlmError` for anything else (bad key, dead
    model, a quota wait too long to sit through).

    The config decides which provider this is: `cfg.label` names it in every
    message and keys its pacing, so NIM and Groq share this function without
    either one leaking its name or budget into the other.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a video-clipping assistant. Reply with a single JSON "
                    "object and nothing else — no prose, no markdown fence."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    fmt = _response_format_for(mode)
    if fmt is not None:
        body["response_format"] = fmt
    # DeepSeek-only wire field, guarded by getattr so the other providers are
    # untouched. Thinking mode is ON by default at high effort and, while on,
    # it IGNORES temperature/presence_penalty/frequency_penalty ("will not
    # trigger an error but will also have no effect") — so with it enabled our
    # temperature-0 ladder would be decorative and every retry would bill the
    # chain-of-thought as output tokens. Disabling it is what actually makes
    # the determinism apply. Raw wire shape is a top-level object.
    thinking = getattr(cfg, "thinking_mode", "")
    if thinking:
        body["thinking"] = {"type": thinking}

    last_err: Exception | None = None
    est = _est_tokens(prompt, max_tokens)
    # Pacing is per PROVIDER, not global: an NIM call must never wait out
    # Groq's 30-rpm floor, or a job that failed over to the backup would run
    # at half speed for no reason.
    pace_key = _provider_key(cfg)
    for attempt in range(cfg.max_retries + 1):
        _groq_wait_turn(cfg.min_interval_sec, pace_key)
        # Pace on TOKENS as well as requests: the request floor keeps a burst
        # under the rpm limit, this under the token limit that actually stops a
        # long source. Only on the first attempt — a retry after a 429 has
        # already waited out the limit, and pacing again would double every
        # backoff.
        token_wait = _groq_token_wait(est, cfg.tokens_per_min, pace_key) if attempt == 0 else 0.0
        if token_wait > 0:
            logger.info(
                f"[llm] {cfg.label} token budget: waiting {token_wait:.0f}s before a "
                f"~{est}-token call ({cfg.tokens_per_min}/min self-imposed)"
            )
            time.sleep(token_wait)
        _groq_record_tokens(est, pace_key)
        try:
            data, resp_headers = _groq_post(body, api_key, cfg.base_url, cfg.timeout_sec)
            text = _groq_reply_text(data)
            if not text.strip():
                raise LlmError(f"{cfg.label} returned an empty reply")
            try:
                _usage_store().record_call(
                    model,
                    total_tokens=(data.get("usage") or {}).get("total_tokens"),
                    headers=resp_headers,
                    provider=_provider_key(cfg),
                )
            except Exception as exc:  # noqa: BLE001 - telemetry must never fail a job
                logger.debug(f"[groq-usage] could not record usage: {exc}")
            return text
        except _GroqHttpError as exc:
            msg = exc.detail
            last_err = exc
            # A model that no longer exists can never succeed on retry, and
            # neither Groq nor NIM names a replacement the way Google does.
            if exc.status in (400, 404) and _groq_model_problem(msg):
                raise LlmError(
                    f"{cfg.label} rejected model {model!r}: {msg[:200]}. "
                    f"Pick a current model from {cfg.docs_url} and set "
                    f"[{pace_key}] model in config.toml."
                ) from exc
            # Checked BEFORE the retry budget and before the ladder can
            # escalate: no amount of backing off or reformatting revives an
            # empty balance, and failing fast here is what stops one dead
            # account from looking like a hung render.
            if _is_billing_exhausted(exc.status, msg):
                raise LlmError(
                    f"{cfg.label} account is out of credit ({msg[:200]}). "
                    f"This is a billing limit, not a bad request — retrying "
                    f"cannot fix it. Top up the {cfg.label} account or set "
                    f"[llm] fallback_provider to a funded backend."
                ) from exc
            if exc.status == 403 and "1010" in msg:
                # Cloudflare edge ban on the request signature, NOT a bad key.
                raise LlmError(
                    f"{cfg.label} returned 403 / Cloudflare error 1010. This is NOT a bad "
                    f"API key — Cloudflare blocked the request before it reached "
                    f"{cfg.label}. Sent User-Agent: {GROQ_USER_AGENT!r}."
                ) from exc
            # ---- CLASSIFY BEFORE THE RETRY BUDGET ----------------------
            # This block used to sit BELOW `if attempt >= cfg.max_retries:
            # break`, which meant that on the final attempt every 4xx fell out
            # of the loop and was re-reported as "transport/rate failures
            # exhausted" — the generic catch-all at the bottom of the function.
            # That mislabelling is invisible on its own and catastrophic once
            # something reads it: a DeepSeek model with no Structured Outputs
            # was reported as a rate-limited transport failure, so the ladder
            # escalated to the next MODEL instead of trying the very next rung
            # (the same model without response_format) that exists to handle
            # exactly this. Classification is not a retry decision, so it does
            # not belong under the retry budget.
            if exc.status == 400 and _is_json_validation_error(msg):
                # The point of the ladder: a 400 that means "your OUTPUT was
                # bad" is retryable with a different strategy (lower temp, bigger
                # budget, no format, another model). Do not spend the in-attempt
                # backoff on it, and do not fail the job over it.
                raise _JsonRetry(f"400 {msg[:200]}")
            if exc.status in (400, 422) and _format_unsupported(msg):
                # The model has no Structured Outputs. Drop response_format
                # entirely and repair the reply locally rather than failing the
                # whole job over an output-formatting nicety.
                raise _FormatUnsupported(f"{exc.status} {msg[:200]}")
            if exc.status == 400:
                # A genuinely permanent 400: the same request fails again. Fail
                # fast rather than re-bill it.
                raise LlmError(
                    f"{cfg.label} rejected the request permanently (400): {msg[:220]}"
                )
            # Everything left IS retryable (429, 413, 5xx, transport), so this
            # is where the budget applies.
            if attempt >= cfg.max_retries:
                break
            if exc.status == 429:
                # The daily-quota rejection is the only place Groq states the real
                # tokens-per-day balance — record it before deciding what to do.
                try:
                    _usage_store().record_quota_error(model, msg, provider=_provider_key(cfg))
                except Exception as qexc:  # noqa: BLE001 - telemetry only
                    logger.debug(f"[groq-usage] could not record quota error: {qexc}")
                fallback = min(90.0, (2 ** attempt) * 4.0 + random.random() * 2)
                asked = exc.retry_after
                # The server's own instruction WINS when it gave one — and
                # `Retry-After: 0` is an instruction ("now"), not an absence.
                # This was `max(asked, fallback)`, which treated Retry-After as
                # a floor to exceed rather than an answer: a provider that said
                # "retry immediately" still cost 4.8s, then 9.6s, and a rate
                # limited chunk spent minutes asleep inside one rung. Only a
                # provider that says NOTHING gets the local exponential.
                wait = fallback if asked is None else max(0.0, asked)
                if asked is not None and asked > MAX_INLINE_RATE_LIMIT_WAIT:
                    # A DAILY token limit answers with an hour-scale wait
                    # ("1h25m40s"). Sleeping that inside a retry loop looks like
                    # a hung job, burns the retries, and fails an hour later with
                    # no useful reason. Stop now and say so.
                    raise LlmError(
                        f"{cfg.label} rate limit needs {asked / 60:.0f} minutes before it "
                        f"will accept another call — this is a daily quota, not a "
                        f"burst one. {msg[:220]}"
                    ) from exc
                logger.warning(
                    f"[llm] {cfg.label} 429 rate limited"
                    f"{' (tokens/min)' if 'tokens per minute' in msg else ''}"
                    f"{f', server asked for {asked:.0f}s' if asked else ''}"
                    f" — backing off {wait:.1f}s (attempt {attempt + 1})"
                )
            else:
                # 413 is NOT permanent — Groq uses it for "Request too large ...
                # on input tokens per minute", a rate limit that clears in
                # seconds. Only the unfixable 4xx fail fast.
                wait = 2.0 * (attempt + 1)
                logger.warning(
                    f"[llm] {cfg.label} error ({exc.status} {msg[:140]}), "
                    f"retry {attempt + 1} in {wait:.0f}s"
                )
            time.sleep(wait)
        except (TimeoutError, json.JSONDecodeError, LlmError, urllib.error.URLError) as exc:
            last_err = exc
            if attempt >= cfg.max_retries:
                break
            wait = 2.0 * (attempt + 1)
            logger.warning(
                f"[llm] {cfg.label} transport error ({exc}), retry {attempt + 1} in {wait:.0f}s"
            )
            time.sleep(wait)
        except _FormatUnsupported:
            raise  # handled by the ladder; never spend the backoff on it
    raise _JsonRetry(
        f"transport/rate failures exhausted: {last_err}", transient=True
    )


#: Providers that rate-limit by TOKENS state the wait in prose rather than a
#: header, and the number is the whole difference between backing off enough
#: and burning every retry against a limit that has not reset yet. The phrasings
#: differ per provider: Groq says "Please try again in 15.6375s", Google says
#: "Please retry in 47.352748833s" and also ships a structured RetryInfo.
_RETRY_HINT = re.compile(
    r"(?:try again in|retry in|retry after|try again after)\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:s\b|sec|secs|seconds?)",
    re.IGNORECASE,
)


def _retry_hint(detail: str) -> float | None:
    """Seconds to wait, if the provider said so in the error body."""
    m = _RETRY_HINT.search(detail or "")
    if not m:
        return None
    try:
        return max(0.0, float(m.group(1)))
    except (TypeError, ValueError):
        return None


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


#: REENTRANT on purpose. `_groq_wait_turn` / `_groq_record_tokens` /
#: `_groq_token_wait` hold this lock across their whole critical section AND
#: call `_pacer()`, which takes the same lock to fetch their per-provider
#: state. A plain Lock deadlocks the first detection call — which is exactly
#: what a 60-second hang on a fresh process looks like.
_groq_lock = threading.RLock()
#: Per-PROVIDER pacing state, keyed by `_provider_key(cfg)` — the config's own
#: `provider` field, so "deepseek" / "nvidia" / "groq".
#:
#: These used to be module globals, which was fine with one provider and is
#: wrong with several: Groq's 30-rpm floor and 6000 tok/min cap are properties
#: of GROQ's free tier, not of the process. Sharing one ledger would make every
#: other call wait out a budget it does not share — and, worse, a job that
#: failed over to the backup would run at half speed for no reason.
_PACERS: dict[str, dict[str, Any]] = {}


def _pacer(key: str) -> dict[str, Any]:
    """The pacing state for one provider, created on first use."""
    with _groq_lock:
        state = _PACERS.get(key)
        if state is None:
            state = {"last": 0.0, "ledger": []}
            _PACERS[key] = state
        return state

#: Conservative share of the published per-minute token allowance this client
#: will spend. Groq's free tier allows 8000 tokens/min for this model; leaving
#: a quarter of it unused costs a little wall time and removes essentially all
#: of the 429s, which cost far more (a retried call is a re-billed call).
#: Overridable per deployment via [groq] tokens_per_min.
GROQ_TOKEN_BUDGET = 6000

#: Longest a single retry may sleep in-line. Past this the limit is a daily
#: quota that will not clear inside a job, and the honest answer is to stop and
#: report it rather than sit in a sleep loop the watchdog will kill.
MAX_INLINE_RATE_LIMIT_WAIT = 150.0


def _est_tokens(text: str, max_tokens: int) -> int:
    """Rough token cost of one call: the prompt plus the output reservation."""
    return max(64, len(text or "") // 4) + max(64, int(max_tokens))


def _usage_store() -> "Any":
    """The Groq usage counter for the configured data dir.

    A wrapper rather than the store itself so a missing/odd data dir can never
    turn telemetry into a detection failure — callers already guard the call,
    and this keeps that guarantee in one place.
    """
    from app import groq_usage

    paths = _SETTINGS_HOLDER[0]
    if paths is None:
        from app.config import load_config

        paths = load_config().app.paths.data_dir
        _SETTINGS_HOLDER[0] = paths
    return groq_usage.store(paths)


#: Cached data dir for the usage store. ``llm`` is imported by tooling that
#: never boots the app, so this is filled lazily on first Groq call.
_SETTINGS_HOLDER: list[Any] = [None]


def note_usage_call(
    model: str, total_tokens: Any = None, headers: Any = None, provider: str = "groq"
) -> None:
    """Record one hosted detection call. Public so tests can drive it."""
    try:
        _usage_store().record_call(model, total_tokens, headers, provider=provider)
    except Exception as exc:  # noqa: BLE001 - telemetry must never fail a job
        logger.debug(f"[groq-usage] could not record call: {exc}")


def _groq_token_wait(est: int, budget: int, key: str = "groq") -> float:
    """Seconds to wait so `est` more tokens fit inside the rolling minute."""
    if est <= 0 or budget <= 0:
        return 0.0
    now = time.monotonic()
    with _groq_lock:
        ledger = _pacer(key)["ledger"]
        recent = [(t, n) for t, n in ledger if now - t < 60.0]
        ledger[:] = recent
        used = sum(n for _, n in recent)
        if used + est <= budget:
            return 0.0
        # Drop the oldest entries until it fits, and wait for them to age out.
        need = used + est - budget
        freed = 0
        for t, n in sorted(recent):
            freed += n
            if freed >= need:
                return max(0.0, 60.0 - (now - t) + 0.5)
        return 30.0


def _groq_record_tokens(est: int, key: str = "groq") -> None:
    with _groq_lock:
        ledger = _pacer(key)["ledger"]
        ledger.append((time.monotonic(), est))
        ledger[:] = [(t, n) for t, n in ledger if time.monotonic() - t < 60.0]


def _groq_wait_turn(floor_sec: float, key: str = "groq") -> None:
    """Block until the configured floor between this provider's calls elapsed.

    Per-provider and process-wide (not per-job) on purpose: a long transcript
    is many calls and they must be paced against the ACCOUNT limit, not each
    other — while a call to the OTHER provider must not be counted against it.
    """
    if floor_sec <= 0:
        return
    with _groq_lock:
        state = _pacer(key)
        now = time.monotonic()
        wait = state["last"] + floor_sec - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        state["last"] = now


def _groq_post(
    body: dict[str, Any],
    api_key: str,
    base_url: str,
    timeout_sec: int,
    user_agent: str = GROQ_USER_AGENT,
) -> tuple[dict[str, Any], dict[str, str]]:
    """POST to {base_url}/chat/completions with the body exactly as given.

    Returns (body, headers). The headers are only there for the
    `x-ratelimit-*` counters, which are per-minute rather than per-day — the
    daily balance is never in a header, only in a 429 body.

    `user_agent` defaults to the Groq constant because api.groq.com sits behind
    Cloudflare and rejects the urllib default outright (see below). NIM does not
    need it, but sending the same explicit UA is harmless and keeps one code
    path rather than a conditional.

    `response_format` is the CALLER's decision now. It used to be hardcoded to
    `json_object` here, which is the root of the `json_validate_failed` death:
    that mode forces a top-level JSON *object* while the detection prompt asks
    for a top-level *array*, so the model had two contradictory instructions
    and failed validation non-deterministically. The retry ladder in call_groq
    owns the format now because choosing it is a strategy, not a transport
    detail.
    """
    base = str(base_url).rstrip("/")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # REQUIRED. api.groq.com sits behind Cloudflare, which returns
            # "HTTP 403: error code: 1010" — a signature ban on the
            # default `Python-urllib/3.x` User-Agent — BEFORE the request
            # reaches Groq, so a perfectly valid key still fails. Verified:
            # urllib default -> 403/1010, an explicit UA -> 401 (reaches
            # the API). Do not remove this header.
            "User-Agent": user_agent,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8")), dict(resp.headers.items())
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
        if retry_after is None:
            # Groq's token-per-minute limit does NOT send Retry-After — it
            # puts the wait in the body: "Please try again in 15.6375s".
            # Parsing it matters more than it looks: a 55-minute source is
            # 20 bounded detection chunks of ~3k tokens each, which overruns
            # an 8000 TPM limit within the first few calls, and a fallback
            # that backed off 1+2+4s against a limit needing 15.6s spent the
            # retry budget and died on a limit it could have waited out.
            retry_after = _retry_hint(detail)
        raise _GroqHttpError(exc.code, detail, retry_after) from exc


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
    """Dispatch one detection call, with optional provider failover.

    The primary is tried first. If it FAILS (quota spent, endpoint down, key
    rejected, unparseable replies after the whole ladder) and
    `[llm] fallback_provider` names a different backend that has a usable key,
    the same chunk is retried there.

    Two things deliberately do NOT fail over:

      * `LlmNotConfigured` on the PRIMARY. A missing key is a mistake to fix,
        not a transient blip to route around — silently downgrading on every
        chunk would hide the misconfiguration until someone read the logs.
        (A missing key on the FALLBACK is different: it just means there is
        nothing to fail over to, and the original error stands.)
      * `DetectionPaused`, which is a user action rather than a failure.
    """
    active = provider(settings)
    try:
        return _call_provider(active, chunk_text_prompt, settings)
    except (LlmNotConfigured, DetectionPaused):
        raise
    except LlmError as exc:
        backup = settings_llm_fallback(settings)
        if not backup or not _provider_ready(backup, settings):
            raise
        logger.warning(
            f"[llm] {active} failed for this chunk ({exc}) — retrying it on the "
            f"configured fallback provider {backup!r}"
        )
        return _call_provider(backup, chunk_text_prompt, settings)


def _call_provider(which: str, chunk_text_prompt: str, settings: Settings) -> str:
    """One detection call to a named backend (no failover logic here)."""
    if which == PROVIDER_OLLAMA:
        return call_ollama(chunk_text_prompt, settings)
    if which == PROVIDER_GEMINI:
        return call_gemini(chunk_text_prompt, settings)
    if which == PROVIDER_DEEPSEEK:
        return call_deepseek(chunk_text_prompt, settings)
    if which == PROVIDER_NVIDIA:
        return call_nvidia(chunk_text_prompt, settings)
    return call_groq(chunk_text_prompt, settings)


def _provider_key(cfg: Any) -> str:
    """Stable provider identity for telemetry buckets (usage, pacing ledgers).

    Taken from the config's own `provider` field, never from its display
    `label`. The label is prose an operator-facing string may be reworded to
    ("NVIDIA NIM"), and a usage bucket that silently changes key when that
    happens reports one provider's spend against another's ceiling. Falls back
    to the label for any config that predates the field, then to "groq".
    """
    name = str(getattr(cfg, "provider", "") or "").strip().lower()
    if name:
        return name
    return str(getattr(cfg, "label", "") or "groq").strip().lower()


def _provider_ready(which: str, settings: Settings) -> bool:
    """Can this backend actually be called right now?

    Key presence only — no network. Used to decide whether a failover is even
    possible, so a missing backup key does not turn one clear error into two.
    """
    if which == PROVIDER_OLLAMA:
        ok, _ = ollama_status(settings)
        return ok
    if which == PROVIDER_GEMINI:
        return settings.gemini_api_key is not None
    if which in OPENAI_PROVIDER_KEYS:
        return bool(openai_api_key(which, settings))
    return False


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


def heuristic_candidates(
    words: list[dict],
    duration_sec: float,
    cfg: GeminiConfig,
) -> list[dict[str, Any]]:
    """Rank transcript segments locally when the model gives us nothing.

    Detection is the one stage whose failure used to be fatal: the AI provider
    returned nothing usable, and the user got a failed job with zero clips even
    though the transcript — the expensive part — was sitting right there
    finished on disk. A batch of nothing is strictly worse than a batch of
    decent clips, so the pipeline now picks segments itself.

    With a transcript: sliding windows scored by WORDS PER SECOND (speech
    density). Dense stretches are where the argument, the number and the
    emphasis live; silence and filler fall to the bottom.

    Without one (a silent or music-only source, where Whisper returns no
    words): evenly spaced segments across the duration, which is the honest
    answer — there is no highlight to find in a source with no speech.
    """
    lo = float(cfg.min_clip_sec or 15)
    hi = float(cfg.max_clip_sec or 60)
    if hi < lo:
        lo, hi = hi, lo
    length = max(lo, min(hi, (lo + hi) / 2.0))
    cap = max(1, int(cfg.max_clips or 3))

    scored: list[tuple[float, float, float]] = []  # (score, start, end)
    if words:
        span = float(words[-1].get("end") or 0.0) - float(words[0].get("start") or 0.0)
        if span > 0:
            stride = max(1.0, length / 3.0)
            start = float(words[0].get("start") or 0.0)
            while start < span + float(words[0].get("start") or 0.0):
                end = min(start + length, float(words[-1].get("end") or 0.0))
                inside = [
                    w for w in words
                    if float(w.get("start") or 0.0) >= start
                    and float(w.get("end") or 0.0) <= end
                ]
                dur = max(0.001, end - start)
                scored.append((len(inside) / dur, start, end))
                start += stride
    else:
        total = float(duration_sec or 0.0)
        if total <= 0:
            return []
        # Evenly spaced, centred so a short source still yields usable clips.
        wanted = max(1, min(cap, int(total // max(length, 1.0))))
        for i in range(wanted):
            start = (total / wanted) * i
            scored.append((1.0 - (i * 0.01), start, min(start + length, total)))

    scored.sort(key=lambda t: t[0], reverse=True)
    picked: list[dict[str, Any]] = []
    for score, start, end in scored:
        if len(picked) >= cap:
            break
        if end - start < min(lo, length) * 0.6:
            continue
        # Keep the chosen windows apart so the batch is not five copies of the
        # same twelve seconds.
        if any(abs(start - p["start"]) < length * 0.8 for p in picked):
            continue
        picked.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "score": round(min(1.0, 0.5 + score / 20.0), 3),
            "title": "Best moment",
            "hook_text": "",
            "hashtags": [],
            "quote": "",
            "detection_source": "transcript_heuristic",
        })
    picked.sort(key=lambda c: c["start"])
    return picked


def detect_clips(
    words: list[dict],
    settings: Settings,
    *,
    min_clip_sec: int | None = None,
    max_clip_sec: int | None = None,
    max_clips: int | None = None,
    duration_sec: float | None = None,
    on_log=None,
    on_progress=None,
    chunk_cache: ChunkCache | None = None,
    should_pause=None,
    report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Full detection: chunk → LLM (word indices) → resolve timings → gate.

    Returns MPT-style dicts: start, end, score, title, hook_text.
    Raises LlmNotConfigured when no key, LlmError when the API fails after
    retries. There is deliberately no heuristic fallback here.

    `chunk_cache` persists each chunk's resolved candidates so a resume re-runs
    only the chunks that never finished. `should_pause` is polled at each chunk
    boundary — the one safe point in this stage — so a user-requested pause can
    stop the job between API calls instead of mid-write.

    `report`, if given, is filled in with how many candidates were detected,
    how many cleared the score floor and how many were kept — the three numbers
    that explain a batch size the user did not choose.

    `on_progress(fraction, detail)` is called once per chunk. Without it a
    28-chunk detection run sat on a hardcoded 50% for two and a half minutes,
    which looks exactly like a stall to anyone watching the job card.
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
    elif provider(settings) in OPENAI_PROVIDER_KEYS:
        # The PRIMARY must have a key even when a fallback is configured: a
        # missing primary key is a misconfiguration to fix, not a transient
        # failure to route around.
        if not _provider_ready(provider(settings), settings):
            live = _LIVE_KEY_RESOLVERS[provider(settings)]()
            live.refresh(force=True)
            var = OPENAI_PROVIDER_KEYS[provider(settings)][1]
            detail = live.problem or (
                f"no key found under any supported name ({', '.join(live.status()['aliases'])})"
            )
            raise LlmNotConfigured(
                f"{var} missing — add it in the Keys/API keys tab (or .env), "
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
        cfg = replace(cfg, min_clip_sec=min_clip_sec)
    if max_clip_sec is not None:
        cfg = replace(cfg, max_clip_sec=max_clip_sec)
    if max_clips is not None:
        cfg = replace(cfg, max_clips=max_clips)

    chunks = chunk_words(words)
    label = provider_label(settings)
    logger.info(f"[llm] detecting on {len(chunks)} chunk(s), {len(words)} words via {label}")
    if on_log:
        on_log(f"Detecting highlights on {len(chunks)} chunk(s) via {label}...")

    #: Written into the caller's `report` on EVERY exit path, including the
    #: heuristic fallbacks below. It used to be built only at the successful
    #: return, so a run that fell back to transcript density reported NO chunk
    #: accounting at all — and the one number that explains a short batch
    #: ("3 of 4 chunks failed") is exactly the number a fallback hides.
    def _chunk_report() -> dict[str, Any]:
        return {
            "chunksTotal": len(chunks),
            "chunksSkipped": len(skipped),
            "chunksSkippedAt": [i + 1 for i in skipped],
            "chunksReused": reused,
        }

    def _finish_report(extra: dict[str, Any] | None = None) -> None:
        if report is None:
            return
        report.update(_chunk_report())
        if extra:
            report.update(extra)

    if not chunks:
        # The transcript is empty, so there is nothing to send and nothing to
        # repair. Saying "returned no valid clip candidates" here blamed the AI
        # backend for what is actually a property of the SOURCE — a 40-second
        # test clip with no speech. Say what is true, and still deliver clips
        # from the duration instead of failing the whole batch.
        why = (
            f"no speech was detected in this video ({duration_sec:.0f}s), so there "
            "was nothing for the AI to score"
            if duration_sec
            else "the transcript came back empty, so there was nothing for the AI to score"
        )
        logger.warning(f"[llm] {why} — using duration-based fallback selection")
        if on_log:
            on_log("No speech detected in this video — picking clips by timing instead of AI scoring.")
        local = heuristic_candidates(words, duration_sec or 0.0, cfg)
        if local:
            if on_log:
                on_log(
                    f"Fallback: {len(local)} clip(s) chosen from the video's timing "
                    "instead of AI scoring. Add captions manually if needed."
                )
            if report is not None:
                report["detectionFallback"] = "transcript_heuristic"
                report["detectionFallbackReason"] = why
                # No chunks existed, so there is nothing to skip — but the keys
                # are still written, so a consumer never has to distinguish
                # "zero chunks" from "the report forgot to say".
                report["chunksTotal"] = 0
                report["chunksSkipped"] = 0
                report["chunksSkippedAt"] = []
            return local
        # Neither a transcript NOR a duration: there is no input of any kind to
        # fall back to. This used to `return local` — an empty list — so the
        # job went on to "succeed" with zero clips, no error, and an empty
        # grid the user could only interpret as a broken app. Say what is
        # actually missing instead.
        raise LlmError(
            f"{why}, and the video's length is unknown too, so no clips can be "
            f"chosen from it at all. Re-submit the video (Re-transcribe to try "
            f"the audio again), or check that the file actually contains sound."
        )

    candidates: list[dict[str, Any]] = []
    reused = 0
    skipped: list[int] = []

    # ---- concurrent chunk pre-fetch -------------------------------------
    # Detection chunks are independent network round trips, so on a host with
    # a single core the win is real: the stage spends its life WAITING on a
    # provider, not computing, and four calls in flight hide three quarters of
    # that latency. This is the only parallelism in the pipeline that pays on a
    # 1-CPU box.
    #
    # It deliberately does NOT restructure the loop below. The concurrent part
    # only performs the API call; the sequential loop still decides cache hits,
    # ordering, merging, per-chunk persistence, pause checks and the
    # skip-and-continue policy, and it re-raises each pre-fetched failure at
    # exactly the point the sequential call would have raised it. So a chunk
    # that failed still fails the same way, in the same order, with the same
    # log line -- the concurrency cannot change an outcome, only when it lands.
    concurrency = max(
        1, min(8, int(getattr(settings.llm, "detect_concurrency", 1) or 1))
    )
    prefetched: dict[int, str] = {}
    failures: dict[int, BaseException] = {}

    def _prefetch(index: int) -> None:
        start_index, chunk_words_for_chunk = chunks[index]
        try:
            prefetched[index] = call_llm(
                build_prompt(chunk_words_for_chunk, cfg, start_index=start_index),
                settings,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised in order below
            failures[index] = exc

    if concurrency > 1 and len(chunks) > 1:
        todo: list[int] = []
        for index in range(len(chunks)):
            if should_pause is not None and should_pause():
                break
            if chunk_cache is not None and chunk_cache.get(index) is not None:
                continue  # already paid for; never re-bill it to be parallel
            todo.append(index)
        if len(todo) > 1:
            logger.info(
                f"[llm] detecting {len(todo)} chunk(s) {concurrency}-way concurrently"
            )
            # Pause has to be re-checked per LAUNCH, not once for the whole
            # window. The window used to be fired off with no gate at all, so a
            # pause requested during chunk 1 still paid for all five chunks and
            # the pause only took effect after every one of them — then the
            # loop below raised BEFORE writing any of them to the cache, so the
            # resume re-billed work that had already been paid for.
            #
            # The dispatcher below is the fix, and it must be ORDERED. A lock
            # alone is not enough: it serialises the launches but says nothing
            # about which chunk gets the next turn, so the calls that did go out
            # could be chunks 0 and 3 — and the loop then stopped at chunk 1,
            # which had never been paid for, discarding chunk 0's paid answer
            # and re-billing it on resume.
            #
            # The honest limit of this: a pause raised by a call cannot be seen
            # until that call returns, so with `concurrency` calls in flight a
            # pause can overshoot by up to concurrency-1 chunks. That is the
            # price of overlapping the round trips, and it is bounded and
            # asserted. What is NOT acceptable — and what this fixes — is work
            # that has already been paid for being thrown away.
            with cf.ThreadPoolExecutor(
                max_workers=min(concurrency, len(todo))
            ) as pool:
                launched: list[cf.Future] = []
                for index in todo:
                    if should_pause is not None and should_pause():
                        break  # a pause is in: do not bill another chunk
                    launched.append(pool.submit(_prefetch, index))
                for future in launched:
                    # _prefetch records its own outcome; a raise here would be a
                    # bug in it, and must not be swallowed.
                    future.result()
    for i, (start_index, chunk) in enumerate(chunks):
        # Pause is honoured at a chunk boundary: never between the API call
        # and the cache write, which is the window where a kill would lose
        # work we already paid for.
        #
        # A chunk whose call already went out is finished WITH that pause or
        # without it — the money is spent and the only question is whether the
        # answer is kept. Pausing before its cache write threw that answer
        # away and made the resume pay for it twice.
        already_paid = i in prefetched or i in failures
        if should_pause is not None and should_pause() and not already_paid:
            raise DetectionPaused(
                f"paused after chunk {i}/{len(chunks)} — {reused} chunk(s) reused, "
                f"{len(candidates)} candidate(s) kept"
            )
        cached = chunk_cache.get(i) if chunk_cache is not None else None
        if cached is not None:
            candidates.extend(cached)
            reused += 1
            if on_progress:
                on_progress((i + 1) / len(chunks), f"Detection: chunk {i + 1}/{len(chunks)} (cached)")
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
        # ONE BAD CHUNK MUST NOT FAIL THE JOB. This is the single most
        # important behaviour change in this stage: a 20-chunk run that lost
        # chunk 3 to a malformed reply threw away 19 chunks that had already
        # been paid for and cached, and reported "Failed during AI clip
        # detection" for a job that was 15% broken rather than unusable.
        #
        # LlmError is still raised when EVERY chunk failed — then the stage
        # genuinely has nothing and the user is right to be told. A partial
        # result is reported as a warning and the job continues.
        try:
            if i in prefetched:
                text = prefetched.pop(i)
            elif i in failures:
                raise failures.pop(i)
            else:
                text = call_llm(prompt, settings)
        except LlmNotConfigured:
            raise  # a missing key affects every chunk; never hide it
        except DetectionPaused:
            raise
        except LlmError as exc:
            skipped.append(i)
            logger.warning(
                f"[llm] chunk {i + 1}/{len(chunks)} failed ({exc}) — skipping it and "
                f"keeping the other chunks"
            )
            if on_log:
                on_log(
                    f"⚠ AI detection failed on chunk {i + 1}/{len(chunks)} "
                    f"(retried with fallbacks) — skipping it and keeping the rest"
                )
            if on_progress:
                on_progress((i + 1) / len(chunks), f"Detection: chunk {i + 1}/{len(chunks)} (skipped)")
            continue
        parsed = parse_candidates(text)
        if not parsed:
            # The call SUCCEEDED and returned nothing usable. Nothing raised,
            # so the provider failover in call_llm() never fired and the
            # configured backup model was never asked — this chunk was simply
            # marked skipped, and if every chunk behaved this way the job
            # failed with "returned no valid clip candidates" while a funded
            # backend sat idle with a key loaded. Ask the backup explicitly,
            # once, before giving up on the chunk.
            backup_reply = None
            backup = settings_llm_fallback(settings)
            if backup and _provider_ready(backup, settings):
                try:
                    logger.warning(
                        f"[llm] chunk {i + 1}/{len(chunks)}: {label} returned "
                        f"nothing parseable — retrying this chunk on the "
                        f"fallback provider {backup!r}"
                    )
                    backup_reply = _call_provider(backup, prompt, settings)
                    parsed = parse_candidates(backup_reply)
                    if parsed and on_log:
                        on_log(
                            f"Chunk {i + 1}/{len(chunks)} came back unusable from "
                            f"{label} — the fallback provider recovered it."
                        )
                except (LlmError, LlmNotConfigured) as exc:
                    logger.warning(
                        f"[llm] fallback provider {backup!r} could not rescue "
                        f"chunk {i + 1}: {exc}"
                    )
                    backup_reply = None
            if parsed:
                resolved = resolve_candidates(parsed, words, start_index, cfg)
                if chunk_cache is not None:
                    chunk_cache.put(i, resolved)
                candidates.extend(resolved)
                if on_progress:
                    on_progress(
                        (i + 1) / len(chunks), f"Detection: chunk {i + 1}/{len(chunks)}"
                    )
                if on_log:
                    on_log(f"Detection: chunk {i + 1}/{len(chunks)} → {len(resolved)} candidates")
                continue
            # The call succeeded but nothing usable came back (prose only, or
            # every clip failed schema validation). Same treatment: skip, keep
            # going, and say so, rather than failing the run.
            #
            # The RAW reply is logged here because this branch was previously
            # undiagnosable: the job said "returned no valid clip candidates"
            # and nothing in the log said whether the model returned prose, an
            # empty string, a truncated array or prose-and-JSON mixed
            # together. Without the raw text, every such failure looked like
            # "the AI is broken" and nobody could tell which of those it was.
            skipped.append(i)
            shown = backup_reply if backup_reply is not None else text
            logger.warning(
                f"[llm] chunk {i + 1}/{len(chunks)} returned no valid clips — skipping\n"
                f"[llm] RAW REPLY chunk {i + 1} ({len(shown)} chars):\n{_raw_reply_excerpt(shown)}"
            )
            if on_log:
                on_log(
                    f"⚠ Chunk {i + 1}/{len(chunks)} returned no usable clips "
                    f"(raw reply: {_raw_reply_excerpt(shown, 200)}) — skipping it"
                )
            if on_progress:
                on_progress((i + 1) / len(chunks), f"Detection: chunk {i + 1}/{len(chunks)} (skipped)")
            continue
        resolved = resolve_candidates(parsed, words, start_index, cfg)
        # Persist BEFORE reporting progress, so a kill in the next instant
        # still keeps this chunk.
        if chunk_cache is not None:
            chunk_cache.put(i, resolved)
        candidates.extend(resolved)
        if on_progress:
            on_progress((i + 1) / len(chunks), f"Detection: chunk {i + 1}/{len(chunks)}")
        if on_log:
            on_log(f"Detection: chunk {i + 1}/{len(chunks)} → {len(resolved)} candidates")
        logger.info(f"[llm] chunk {i + 1}/{len(chunks)}: {len(resolved)} candidates")
    if reused:
        logger.info(
            f"[llm] detection resumed: reused {reused}/{len(chunks)} chunk(s) from cache, "
            f"re-billed only the remainder"
        )
    if skipped:
        # Say it out loud. A silently short batch is indistinguishable from a
        # genuinely quiet source, and the user deserves to know that a chunk was
        # dropped rather than left to wonder why a 54-minute video produced 11
        # clips instead of 23.
        where = ", ".join(str(i + 1) for i in skipped)
        logger.warning(
            f"[llm] detection completed with {len(skipped)}/{len(chunks)} chunk(s) "
            f"skipped after model failures (chunk {where}); every other chunk was kept"
        )
        if on_log:
            on_log(
                f"Finished with {len(skipped)} of {len(chunks)} chunk(s) skipped "
                f"(chunk {where}). The rest of the video was detected normally — "
                f"hit Re-detect to retry just those."
            )

    # Keep only the strong ones, up to the cap. The cap is a ceiling and the
    # score floor is the gate: a source with three genuinely good moments
    # returns three clips whatever its length, and the report says so.
    top, above_floor, below_floor = clipcount.gate(
        candidates, cfg.max_clips, cfg.min_score
    )
    if not top:
        # Every chunk failed, so the model gave us nothing at all. That used to
        # be a hard job failure — which is the worst possible outcome, because
        # the transcript was already transcribed and paid for, and the user
        # ended up with an empty batch instead of a mediocre one. Fall back to
        # ranking the transcript locally so the job still delivers clips, and
        # say plainly that the clips were not AI-picked.
        local = heuristic_candidates(words, duration_sec or 0.0, cfg)
        if local:
            logger.warning(
                f"[llm] {label} produced no usable candidates across "
                f"{len(chunks)} chunk(s) — falling back to transcript density "
                f"({len(local)} clip(s))"
            )
            if on_log:
                on_log(
                    f"⚠ {label} returned no usable clips after all chunks — "
                    f"picked {len(local)} clip(s) by transcript density instead. "
                    "Re-detect with AI once the provider recovers."
                )
            if report is not None:
                report["detectionFallback"] = "transcript_heuristic"
                report["detectionFallbackReason"] = (
                    f"{label} returned no valid clip candidates across "
                    f"{len(chunks)} chunk(s)"
                )
            _finish_report()
            return local
        # Nothing to rank either: no transcript AND no duration. Fail, but name
        # the chunk count so the user sees why nothing happened.
        if skipped:
            raise LlmError(
                f"{label} returned no valid clip candidates — all "
                f"{len(skipped)}/{len(chunks)} chunk(s) failed after retries and "
                f"fallback models. The transcript is fine; the detection backend is not "
                f"responding with usable JSON right now."
            )
        raise LlmError(f"{label} returned no valid clip candidates")
    if report is not None:
        report.update({
            "detected": len(candidates),
            "selected": len(top),
            "cap": cfg.max_clips,
            "floor": cfg.min_score,
            "aboveFloor": above_floor,
            "belowFloor": below_floor,
            "topScore": float(top[0]["score"]),
            "lowestKept": float(top[-1]["score"]),
        })
        _finish_report()
    logger.info(
        f"[llm] selected {len(top)} clip(s) (cap {cfg.max_clips}, score floor "
        f"{cfg.min_score:.0f}): {above_floor} candidate(s) cleared the floor, "
        f"{below_floor} did not; scores {top[0]['score']:.0f}…{top[-1]['score']:.0f}"
    )
    return top
