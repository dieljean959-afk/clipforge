"""Gemini clip detection (LLM service).

Chunked transcript → Gemini → JSON [{start_word, end_word, score, title,
hook_text}] with 429-aware exponential backoff. The model returns WORD
INDICES, never timestamps — fragile float math is the LLM's weak spot, so we
resolve the real start/end from Whisper's word-level timings afterwards.

Only transcript text is ever sent to the hosted provider — never audio or
video. The key-missing case is surfaced loudly: no silent heuristic fallback
exists in this module.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from loguru import logger

from app.config import GeminiConfig, Settings, live_gemini_key


class LlmNotConfigured(Exception):
    """GEMINI_API_KEY missing — the caller must surface this, not fall back."""


class LlmError(Exception):
    """Gemini call failed after retries."""


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
        raise LlmError(f"no JSON array in Gemini reply: {text[:200]!r}")
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise LlmError(f"invalid JSON from Gemini: {exc}") from exc
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
            logger.warning(f"[llm] could not find model = line under [gemini] in config.toml")
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


def detect_clips(
    words: list[dict],
    settings: Settings,
    *,
    min_clip_sec: int | None = None,
    max_clip_sec: int | None = None,
    max_clips: int | None = None,
    on_log=None,
) -> list[dict[str, Any]]:
    """Full detection: chunk → Gemini (word indices) → resolve timings → top-N.

    Returns MPT-style dicts: start, end, score, title, hook_text.
    Raises LlmNotConfigured when no key, LlmError when the API fails after
    retries. There is deliberately no heuristic fallback here.
    """
    if not settings.gemini_api_key:
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
    logger.info(f"[llm] detecting on {len(chunks)} chunk(s), {len(words)} words")
    if on_log:
        on_log(f"Gemini: scoring {len(chunks)} chunk(s)...")

    candidates: list[dict[str, Any]] = []
    for i, (start_index, chunk) in enumerate(chunks):
        prompt = build_prompt(chunk, cfg, start_index=start_index)
        text = call_gemini(prompt, settings)
        parsed = parse_candidates(text)
        resolved = resolve_candidates(parsed, words, start_index, cfg)
        candidates.extend(resolved)
        if on_log:
            on_log(f"Gemini: chunk {i + 1}/{len(chunks)} → {len(resolved)} candidates")
        logger.info(f"[llm] chunk {i + 1}/{len(chunks)}: {len(resolved)} candidates")

    # keep only the top N by score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[: cfg.max_clips]
    if not top:
        raise LlmError("Gemini returned no valid clip candidates")
    logger.info(f"[llm] selected {len(top)} clips (scores {top[0]['score']:.0f}…{top[-1]['score']:.0f})")
    return top
