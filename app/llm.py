"""Gemini clip detection (LLM service).

Chunked transcript → Gemini → JSON [{start, end, score, title, hook_text}]
with 429-aware exponential backoff. The key-missing case is surfaced loudly:
no silent heuristic fallback exists in this module.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from loguru import logger

from app.config import GeminiConfig, Settings


class LlmNotConfigured(Exception):
    """GEMINI_API_KEY missing — the caller must surface this, not fall back."""


class LlmError(Exception):
    """Gemini call failed after retries."""


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


def build_prompt(chunk: list[dict], cfg: GeminiConfig) -> str:
    lines = ["TRANSCRIPT (word-level, seconds):"]
    for w in chunk:
        lines.append(f'{w["start"]:.2f}-{w["end"]:.2f} {w["word"]}')
    return f"""
You select the most viral short-form clips from a video transcript.
Timing window: clips must be {cfg.min_clip_sec}-{cfg.max_clip_sec} seconds long.
Pick {cfg.max_clips} best candidates max — fewer is fine. Do NOT invent times
that are not backed by the transcript. Score 0-100 for hook strength, pacing
and emotional energy. 2-3 word hook_text that would work as the opening caption.

Return ONLY a JSON array (no prose, no markdown fences):
[{{"start": 12.5, "end": 48.2, "score": 92, "title": "Short punchy title",
   "hook_text": "WAIT FOR IT"}}]

TRANSCRIPT (word-level, seconds):
""".strip()


def parse_candidates(text: str) -> list[dict[str, Any]]:
    """Extract the JSON array from a model reply, tolerating fences/prose."""
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
            start = float(item["start"])
            end = float(item["end"])
            score = float(item.get("score", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start or start < 0:
            continue
        out.append(
            {
                "start": start,
                "end": end,
                "score": max(0.0, min(100.0, score)),
                "title": str(item.get("title", "Untitled clip")),
                "hook_text": str(item.get("hook_text", "")),
            }
        )
    return out


def chunk_words(words: list[dict], target_seconds: float = 120.0) -> list[list[dict]]:
    """Split a word-level transcript into ~2-minute chunks for the LLM."""
    if not words:
        return []
    chunks: list[list[dict]] = []
    current: list[dict] = []
    t0 = words[0]["start"]
    for w in words:
        current.append(w)
        if w["end"] - t0 >= target_seconds:
            chunks.append(current)
            current = []
            t0 = w["end"]
    if current:
        chunks.append(current)
    return chunks


def call_gemini(chunk_text_prompt: str, settings: Settings) -> str:
    """Single Gemini call with 429 backoff. Raises LlmError after retries."""
    genai = _load_google_module()
    model = settings.gemini.model
    client = genai.Client(api_key=settings.gemini_api_key)

    last_err: Exception | None = None
    for attempt in range(settings.gemini.max_retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=chunk_text_prompt)
            text = resp.text or ""
            if not text.strip():
                raise LlmError("Gemini returned an empty reply")
            return text
        except Exception as exc:  # noqa: BLE001 - SDK raises many types
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
    """Full detection: chunk → Gemini per chunk → merge → top-N by score.

    Returns MPT-style dicts: start, end, score, title, hook_text.
    Raises LlmNotConfigured when no key, LlmError when the API fails after
    retries. There is deliberately no heuristic fallback here.
    """
    if not settings.gemini_api_key:
        raise LlmNotConfigured("GEMINI_API_KEY missing — add it to .env for AI detection")

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
    for i, chunk in enumerate(chunks):
        prompt = build_prompt(chunk, cfg)
        text = call_gemini(prompt, settings)
        parsed = parse_candidates(text)
        candidates.extend(parsed)
        if on_log:
            on_log(f"Gemini: chunk {i + 1}/{len(chunks)} → {len(parsed)} candidates")
        logger.info(f"[llm] chunk {i + 1}/{len(chunks)}: {len(parsed)} candidates")

    # keep only the top N by score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[: cfg.max_clips]
    if not top:
        raise LlmError("Gemini returned no valid clip candidates")
    logger.info(f"[llm] selected {len(top)} clips (scores {top[0]['score']:.0f}…{top[-1]['score']:.0f})")
    return top
