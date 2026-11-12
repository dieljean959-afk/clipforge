"""ASR engine layer — WhisperX preferred, batched faster-whisper fallback.

WhisperX (github.com/m-bain/whisperX) is the requested transcription backend.
It runs on the very same faster-whisper / CTranslate2 engine, but adds:

  * VAD-based **batched** inference — many 30s windows are decoded in one
    forward pass instead of one at a time, which is where the big speed win
    comes from, and
  * a wav2vec2 **forced-alignment** pass that produces genuinely accurate
    word-level timestamps (base Whisper's word timings are known to drift by
    seconds, which directly hurts caption sync and clip-highlight scoring).

Installing WhisperX pulls in torch + torchaudio + pyannote-audio (multi-GB).
On a host where that stack is not installed this module falls back to
faster-whisper's own ``BatchedInferencePipeline`` — identical CTranslate2
engine, identical batching strategy, minus the alignment pass. The active
engine is always reported (boot banner, ``/api/health``, ``/api/doctor`` and
the per-job log) so a fallback is visible, never silent.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from app import objectstore

from app import hostlimits

ENGINE_WHISPERX = "whisperx"
ENGINE_FASTER_WHISPER = "faster-whisper"
ENGINES = (ENGINE_WHISPERX, ENGINE_FASTER_WHISPER)

# Requested batch size when decoding. WhisperX/CT2 parallelise this many 30s
# windows at once. 8 is the WhisperX default and is safe on CPU int8 as well.
DEFAULT_BATCH_SIZE = 8

# Silero VAD knobs: strip silence and music before the model sees it — silence
# is where Whisper hallucinates.
VAD_PARAMETERS = {"min_silence_duration_ms": 300, "speech_pad_ms": 60}

# Single source of truth for the anti-repetition decode settings, shared by both
# engines (subtitle.py re-exports this as WHISPER_KWARGS):
#   condition_on_previous_text=False — a hallucination cannot anchor the next
#     window (WhisperX sets this False by default for the same reason).
#   compression_ratio_threshold=2.2 — overly repetitive segments are flagged
#     and discarded instead of emitted 40 times.
#   no_speech_threshold=0.6 — near-silent segments are skipped, not invented.
#   vad_filter=True — Silero VAD removes non-speech before the encoder sees it.
ANTI_REPETITION = {
    "condition_on_previous_text": False,
    "compression_ratio_threshold": 2.2,
    "no_speech_threshold": 0.6,
    "vad_filter": True,
    "vad_parameters": VAD_PARAMETERS,
}

# WhisperX configures decoding when its pipeline is loaded. Its transcribe()
# method does not accept faster-whisper's VAD keys (and older 3.3.x releases do
# not accept the decoder keys there either), so keep the two surfaces separate.
WHISPERX_ASR_OPTIONS = {
    key: value
    for key, value in ANTI_REPETITION.items()
    if key not in {"vad_filter", "vad_parameters"}
}


class AsrError(Exception):
    """Any engine-level failure with a message a user can act on."""


@dataclass(frozen=True)
class EngineStatus:
    """What the engine layer is actually running right now."""

    engine: str
    device: str
    compute_type: str
    # `model` is the model that will actually load. `configured_model` is what
    # config.toml says, and `model_note` explains the difference when the
    # configured one does not fit this host's memory.
    model: str
    batch_size: int
    batched: bool
    detail: str
    configured_model: str = ""
    model_note: str | None = None

    @property
    def is_whisperx(self) -> bool:
        return self.engine == ENGINE_WHISPERX

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "device": self.device,
            "computeType": self.compute_type,
            "model": self.model,
            "configuredModel": self.configured_model or self.model,
            "modelNote": self.model_note,
            "batchSize": self.batch_size,
            "batched": self.batched,
            "detail": self.detail,
        }


def resolved_model(settings: Any) -> tuple[str, str | None]:
    """The whisper model that can actually run here, plus any explanation.

    Delegates to hostlimits: a configured model that exceeds the container's
    memory limit is an OOM kill waiting to happen, not a slow run.
    """
    return hostlimits.resolved_model(settings.whisper.model)


@dataclass(frozen=True)
class AsrWord:
    """One word with absolute (source-relative) timings in seconds.

    ``conf`` is the model's OWN confidence in this token, 0..1: faster-whisper
    exposes it per word as ``Word.probability``, and where that is unavailable
    (WhisperX forced-alignment) we fall back to ``exp(avg_logprob)`` of the
    segment the word came from. It defaults to 1.0 so a transcript written
    before this field existed reads as "never measured" rather than as
    "certain" — see the note in subtitle.words_to_cues, which is where the
    number actually decides anything.
    """

    start: float
    end: float
    word: str
    conf: float = 1.0


def _segment_confidence(seg: Any) -> float:
    """``exp(avg_logprob)`` for one decoded segment, clamped to 0..1.

    Whisper reports the mean per-token log probability; exponentiating it puts
    it on the same 0..1 scale as the per-word probability, so a cue can be
    judged by one rule regardless of which engine produced it. Accepts both
    engine shapes (faster-whisper objects, WhisperX dicts). A missing or
    non-numeric value returns 1.0 — no measurement, so do not invent doubt.
    """
    import math  # local: only this helper needs it

    raw = seg.get("avg_logprob") if isinstance(seg, dict) else getattr(seg, "avg_logprob", None)
    try:
        return min(1.0, max(0.0, math.exp(float(raw))))
    except (TypeError, ValueError):
        return 1.0


def _word_confidence(word: Any, seg_conf: float) -> float:
    """Per-token probability when the engine gives one, else the segment's."""
    try:
        prob = float(getattr(word, "probability", None))
    except (TypeError, ValueError):
        return seg_conf
    if prob <= 0.0:
        return seg_conf
    return min(1.0, max(0.0, prob))


def _spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def whisperx_available() -> tuple[bool, str]:
    """(usable, human reason). Importing is the only honest availability test."""
    if not _spec("whisperx"):
        return False, "whisperx is not installed (pip install 'clipforge[whisperx]')"
    try:
        wx = importlib.import_module("whisperx")
    except Exception as exc:  # torch/pyannote missing, broken wheel, ...
        return False, f"whisperx import failed: {type(exc).__name__}: {exc}"
    return True, f"whisperx {getattr(wx, '__version__', '?')}"


def _requested_engine(settings: Any) -> str:
    want = str(getattr(settings.whisper, "engine", "auto") or "auto").lower()
    if want in ENGINES:
        return want
    return "auto"


def _resolve_device(settings: Any) -> tuple[str, str]:
    """Return (device, compute_type), resolving 'auto' like the config does."""
    device = settings.whisper.device
    compute = settings.whisper.compute_type
    if device == "auto":
        try:
            import ctranslate2  # noqa: PLC0415

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def choose_engine(settings: Any) -> tuple[str, str]:
    """Pick the engine to use and explain the choice (never silent).

    A configured engine is honoured; when it is unavailable we fall back to the
    other one and say exactly why.
    """
    want = _requested_engine(settings)
    ok, reason = whisperx_available()
    if want == ENGINE_WHISPERX:
        if ok:
            return ENGINE_WHISPERX, reason
        return ENGINE_FASTER_WHISPER, f"whisperx requested but unavailable — {reason}"
    if want == ENGINE_FASTER_WHISPER:
        return ENGINE_FASTER_WHISPER, "faster-whisper batched pipeline (engine pinned in config)"
    if ok:
        return ENGINE_WHISPERX, f"auto-selected — {reason}"
    return ENGINE_FASTER_WHISPER, f"auto-selected faster-whisper — {reason}"


def batch_size(settings: Any) -> int:
    """Configured batch size, clamped to what this host can hold in memory.

    A batch is decoded in parallel, so its memory scales with the batch — on a
    small container an unclamped batch is an OOM in waiting.
    """
    try:
        value = int(getattr(settings.whisper, "batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE)
    except (TypeError, ValueError):
        value = DEFAULT_BATCH_SIZE
    return max(1, min(64, hostlimits.batch_size_for_host(value)))


def status(settings: Any) -> EngineStatus:
    """Report the engine that WOULD be used for the next transcription."""
    engine, detail = choose_engine(settings)
    device, compute = _resolve_device(settings)
    model, note = resolved_model(settings)
    if note:
        detail = f"{detail} | {note}"
    return EngineStatus(
        engine=engine,
        device=device,
        compute_type=compute,
        model=model,
        batch_size=batch_size(settings),
        batched=True,
        detail=detail,
        configured_model=settings.whisper.model,
        model_note=note,
    )


# --------------------------------------------------------------------------- #
# loading (cached per model+device+compute+batch)
# --------------------------------------------------------------------------- #

@dataclass
class LoadedEngine:
    """A loaded, ready-to-use engine handle."""

    engine: str
    device: str
    compute_type: str
    model_name: str
    batch_size: int
    model: Any
    whisperx: Any = None
    align: dict[str, Any] = field(default_factory=dict)  # language -> (model_a, metadata)

    def asr_words(
        self,
        audio: Any,
        language: str | None,
        on_segment: Callable[[float], None] | None = None,
    ) -> tuple[list[AsrWord], str | None]:
        """Transcribe one audio buffer into word-level timings.

        Returns (words, detected_language). Timings are relative to the start of
        `audio`; the caller offsets them for chunked decoding.
        """
        if self.engine == ENGINE_WHISPERX:
            return self._whisperx_words(audio, language, on_segment)
        return self._faster_whisper_words(audio, language, on_segment)

    # ------------------------------------------------------------------ #
    # faster-whisper (batched) — always available
    # ------------------------------------------------------------------ #

    def _faster_whisper_words(
        self,
        audio: Any,
        language: str | None,
        on_segment: Callable[[float], None] | None,
    ) -> tuple[list[AsrWord], str | None]:
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            batch_size=self.batch_size,
            beam_size=5,
            word_timestamps=True,
            **ANTI_REPETITION,
        )
        words: list[AsrWord] = []
        for seg in segments:
            seg_conf = _segment_confidence(seg)
            for w in (seg.words or []):
                text = (w.word or "").strip()
                if not text:
                    continue
                words.append(AsrWord(
                    start=float(w.start), end=float(w.end), word=text,
                    conf=_word_confidence(w, seg_conf),
                ))
            if on_segment is not None:
                try:
                    on_segment(float(seg.end))
                except Exception:
                    pass
        return words, getattr(info, "language", None)

    # ------------------------------------------------------------------ #
    # WhisperX (batched + forced alignment)
    # ------------------------------------------------------------------ #

    def _whisperx_words(
        self,
        audio: Any,
        language: str | None,
        on_segment: Callable[[float], None] | None,
    ) -> tuple[list[AsrWord], str | None]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "language": language,
            "task": "transcribe",
            "print_progress": False,
        }
        transcribe_params = inspect.signature(self.model.transcribe).parameters
        progress_sent = False
        if "progress_callback" in transcribe_params:
            audio_seconds = len(audio) / 16_000.0

            def report_progress(percent: float) -> None:
                nonlocal progress_sent
                progress_sent = True
                if on_segment is not None and audio_seconds > 0:
                    try:
                        on_segment(audio_seconds * max(0.0, min(100.0, float(percent))) / 100.0)
                    except Exception:
                        pass

            kwargs["progress_callback"] = report_progress
        try:
            result = self.model.transcribe(audio, **kwargs)
        except TypeError as exc:
            if "progress_callback" not in kwargs or "progress_callback" not in str(exc):
                raise
            # A wrapper can advertise a callback in its signature but reject it
            # at runtime. Retry once; the aligned segments below still update
            # progress after decode on these older builds.
            logger.warning(f"[asr] whisperx progress callback rejected ({exc}) — retrying without it")
            kwargs.pop("progress_callback", None)
            progress_sent = False
            result = self.model.transcribe(audio, **kwargs)
        segments = list(result.get("segments") or [])
        detected = result.get("language") or language

        # WhisperX 3.3.x has no decode callback, so publish each returned
        # segment as soon as decoding finishes. Newer builds already reported
        # live decode progress above.
        if on_segment is not None and not progress_sent:
            for seg in segments:
                try:
                    on_segment(float(seg.get("end") or 0.0))
                except Exception:
                    pass

        # Forced alignment: wav2vec2 gives real per-word timings. Without it we
        # would be guessing inside each segment, which is exactly the caption
        # drift WhisperX exists to fix.
        aligned = self._align_segments(segments, detected, audio)

        words: list[AsrWord] = []
        interpolated = 0
        for seg in aligned:
            # WhisperX aligns timings but does not emit per-token
            # probabilities, so the segment's own confidence is the finest
            # signal available. It is still far better than guessing.
            seg_conf = _segment_confidence(seg)
            seg_words = seg.get("words") or []
            if seg_words:
                for w in seg_words:
                    text = str(w.get("word") or "").strip()
                    start, end = w.get("start"), w.get("end")
                    if not text:
                        continue
                    if start is None or end is None:
                        # WhisperX cannot align tokens it has no phonemes for
                        # (numbers, symbols) — interpolate inside the segment
                        # and count it so the warning is honest about scope.
                        start, end = _interpolate_span(seg, seg_words, w)
                        interpolated += 1
                    words.append(AsrWord(
                        start=float(start), end=float(end), word=text, conf=seg_conf,
                    ))
            else:
                # no alignment data at all for this segment — spread its words
                words.extend(_segment_fallback(seg))
        if interpolated:
            logger.warning(
                f"[asr] whisperx alignment had no timing for {interpolated} token(s) "
                "(numbers/symbols) — interpolated inside their segment"
            )
        return words, detected

    def _align_segments(self, segments: list[dict], language: str | None, audio: Any) -> list[dict]:
        if not segments:
            return segments
        lang = language or "en"
        try:
            model_a, metadata = self._align_model(lang)
        except Exception as exc:
            logger.warning(f"[asr] alignment model unavailable for '{lang}' ({exc}) — using unaligned segments")
            return segments
        try:
            result = self.whisperx.align(
                segments, model_a, metadata, audio, self.device, return_char_alignments=False
            )
            return list(result.get("segments") or segments)
        except Exception as exc:
            logger.warning(f"[asr] whisperx alignment failed ({exc}) — using unaligned segments")
            return segments

    def _align_model(self, language: str) -> tuple[Any, Any]:
        if language not in self.align:
            logger.info(f"[asr] loading wav2vec2 alignment model for '{language}'")
            self.align[language] = self.whisperx.load_align_model(
                language_code=language, device=self.device
            )
        return self.align[language]


def _interpolate_span(seg: dict, seg_words: list[dict], target: dict) -> tuple[float, float]:
    """Estimate a token's timing from its aligned neighbours in the same segment.

    WhisperX leaves `start`/`end` as None for tokens its phoneme dictionary
    cannot cover ("2014.", "£13.60"). Falling back to the segment bounds would
    make a caption flip late, so we clamp the token between the previous
    aligned word's end and the next aligned word's start.
    """
    idx = seg_words.index(target)
    prev_end: float | None = None
    next_start: float | None = None
    for w in reversed(seg_words[:idx]):
        if w.get("end") is not None:
            prev_end = float(w["end"])
            break
    for w in seg_words[idx + 1:]:
        if w.get("start") is not None:
            next_start = float(w["start"])
            break

    seg_start = float(seg.get("start") or 0.0)
    seg_end = float(seg.get("end") or seg_start + 0.5)
    start = prev_end if prev_end is not None else seg_start
    end = next_start if next_start is not None else seg_end
    if end <= start:
        end = start + 0.05
    return start, end


def _segment_fallback(seg: dict) -> list[AsrWord]:
    """Spread a segment's text evenly across its own duration (documented approximation)."""
    text = str(seg.get("text") or "").strip()
    if not text:
        return []
    start = float(seg.get("start") or 0.0)
    end = float(seg.get("end") or start)
    tokens = text.split()
    if not tokens or end <= start:
        return []
    span = (end - start) / len(tokens)
    conf = _segment_confidence(seg)
    return [
        AsrWord(start=start + i * span, end=start + (i + 1) * span, word=tok, conf=conf)
        for i, tok in enumerate(tokens)
    ]


_ENGINE_CACHE: dict[tuple, LoadedEngine] = {}
_ENGINE_LOCK = threading.Lock()


def load(settings: Any, on_log: Callable[[str], None] | None = None) -> LoadedEngine:
    """Load (and therefore download) the configured model. Cached per config.

    Raises AsrError with a readable message — never a bare traceback.
    """
    engine, detail = choose_engine(settings)
    device, compute = _resolve_device(settings)
    size = batch_size(settings)
    model_name, note = resolved_model(settings)
    key = (engine, model_name, device, compute, size)
    with _ENGINE_LOCK:
        if key in _ENGINE_CACHE:
            return _ENGINE_CACHE[key]
        if on_log:
            on_log(f"Engine: {engine} | {detail}")
            if note:
                on_log(f"Model: {model_name} (configured '{settings.whisper.model}') — {note}")
        loaded = _load_fresh(engine, model_name, device, compute, size, settings, on_log)
        _ENGINE_CACHE[key] = loaded
        return loaded


def _load_fresh(
    engine: str,
    model_name: str,
    device: str,
    compute: str,
    size: int,
    settings: Any,
    on_log: Callable[[str], None] | None,
) -> LoadedEngine:
    if engine == ENGINE_WHISPERX:
        ok, reason = whisperx_available()
        if not ok:  # raced/removed — fall back instead of crashing the job
            logger.warning(f"[asr] {reason} — falling back to batched faster-whisper")
            engine = ENGINE_FASTER_WHISPER
        else:
            try:
                return _load_whisperx(model_name, device, compute, size, settings, on_log)
            except Exception as exc:
                logger.warning(
                    f"[asr] whisperx load failed ({type(exc).__name__}: {exc}) — "
                    "falling back to batched faster-whisper"
                )
                if on_log:
                    on_log(f"WhisperX load failed ({exc}) — using batched faster-whisper instead")
    return _load_faster_whisper(model_name, device, compute, size, settings, on_log)


def _load_whisperx(
    model_name: str,
    device: str,
    compute: str,
    size: int,
    settings: Any,
    on_log: Callable[[str], None] | None,
) -> LoadedEngine:
    wx = importlib.import_module("whisperx")
    language = None if settings.whisper.language == "auto" else settings.whisper.language
    base = {
        "device": device,
        "compute_type": compute,
        "language": language,
        # container-aware: never spawn one inference thread per HOST core
        "threads": hostlimits.inference_threads(),
    }
    load_kwargs = dict(base)
    load_params = inspect.signature(wx.load_model).parameters
    if "asr_options" in load_params:
        load_kwargs["asr_options"] = dict(WHISPERX_ASR_OPTIONS)
    if "vad_method" in load_params:
        # New WhisperX releases ship a local Silero VAD implementation. Older
        # 3.3.x releases have no selector and use their bundled default VAD.
        load_kwargs["vad_method"] = "silero"
    try:
        model = wx.load_model(model_name, **load_kwargs)
    except TypeError as exc:
        # Defensive compatibility for wrappers whose signature is incomplete.
        # Never retry after an arbitrary model/OOM failure: that would load the
        # multi-GB model twice. The outer loader falls back to faster-whisper.
        if "asr_options" not in str(exc) and "vad_method" not in str(exc):
            raise
        load_kwargs.pop("asr_options", None)
        load_kwargs.pop("vad_method", None)
        model = wx.load_model(model_name, **load_kwargs)
    if on_log:
        on_log(
            f"WhisperX loaded: {model_name} on {device} ({compute}), "
            f"{base['threads']} inference thread(s), batch {size}"
        )
    return LoadedEngine(
        engine=ENGINE_WHISPERX,
        device=device,
        compute_type=compute,
        model_name=model_name,
        batch_size=size,
        model=model,
        whisperx=wx,
    )


def _load_faster_whisper(
    model_name: str,
    device: str,
    compute: str,
    size: int,
    settings: Any,
    on_log: Callable[[str], None] | None,
) -> LoadedEngine:
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.transcribe import BatchedInferencePipeline
    except Exception as exc:
        raise AsrError(f"faster-whisper is not usable on this host: {exc}") from exc
    # Container-aware thread count. os.cpu_count() reports the HOST's cores,
    # which in a cgroup-capped sandbox is 48x the real quota — that both
    # thrashes the single available core and multiplies CTranslate2's per-thread
    # buffers until the container is OOM-killed.
    threads = hostlimits.inference_threads()
    # Weights come from the bucket: this restores them for the decode and they
    # are removed from disk again once the batch is done.
    from app import modelstore  # noqa: PLC0415

    model_ref = modelstore.resolve(model_name, on_log)
    try:
        model = WhisperModel(model_ref, device=device, compute_type=compute, cpu_threads=threads)
    except Exception as exc:
        raise AsrError(
            f"Whisper model load failed ({model_name} on {device}/{compute}, {threads} threads): {exc}"
        ) from exc
    pipeline = BatchedInferencePipeline(model=model)
    if on_log:
        on_log(
            f"faster-whisper loaded: {model_name} on {device} ({compute}), "
            f"{threads} inference thread(s) of {hostlimits.effective_cpus()} usable, "
            f"batched decode (batch {size})"
        )
    return LoadedEngine(
        engine=ENGINE_FASTER_WHISPER,
        device=device,
        compute_type=compute,
        model_name=model_name,
        batch_size=size,
        model=pipeline,
    )


def model_cache_dir() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415

        return str(HF_HUB_CACHE)
    except Exception:
        return str(os.path.expanduser("~/.cache/huggingface"))


def model_status(settings: Any) -> dict[str, Any]:
    """What will ACTUALLY run, where it lives, and whether it is on disk.

    One honest record for /api/status, the Doctor button and the boot banner.
    It never loads the model (that is ~1 GiB of RSS in the API process); it
    reports the resolved model, the label the UI must show, and the mismatch
    between configured and actual so a downshift can never be silent.
    """
    import os
    from pathlib import Path

    engine, detail = choose_engine(settings)
    device, compute = _resolve_device(settings)
    model, note = resolved_model(settings)
    configured = str(getattr(settings.whisper, "model", "") or "")
    cache = model_cache_dir()
    needle = model.replace("/", "--")
    paths: list[str] = []
    size = 0
    files_present = False
    local_present = False
    bucket_present = False

    def _measure(base: Path) -> None:
        """Add one model directory to the report (path + real bytes)."""
        nonlocal size, files_present
        paths.append(str(base))
        for f in base.rglob("*"):
            try:
                if f.is_file():
                    size += f.stat().st_size
            except OSError:
                continue
        if all((base / f).is_file() and (base / f).stat().st_size > 0 for f in ("model.bin", "config.json")):
            files_present = True

    try:
        # The app keeps its streamed copies in <data>/models/<name>; the HF hub
        # cache is only where a one-off download lands. Checking the hub alone
        # reported "0 files / 0 MB" while 464 MB of `small` sat on disk.
        from app import modelstore  # noqa: PLC0415

        local = modelstore.local_dir(model)
        if local.is_dir():
            _measure(local)
            local_present = files_present
        base = Path(cache)
        if base.is_dir():
            for p in base.glob("models--*"):
                if needle in p.name:
                    _measure(p)
    except Exception as exc:  # noqa: BLE001 - a status card never breaks the API
        logger.warning(f"[asr] model paths could not be measured: {exc}")

    # Between batches the weights are deliberately released (they are 464 MB and
    # only needed while decoding); the durable copy is in the bucket. Reporting
    # "0 files" for a model that loads in 1 s would be a lie, so ask the bucket.
    if not files_present:
        try:
            from app import modelstore  # noqa: PLC0415

            meta = objectstore.head(modelstore._key(model, "model.bin")) or {}
            bucket_present = bool(meta.get("size"))
        except Exception as exc:  # noqa: BLE001 - offline is not a status error
            logger.info(f"[asr] bucket copy of {model} not checked: {exc}")
        files_present = bucket_present
        if bucket_present:
            size = int(meta.get("size") or 0)
    threads = None
    try:
        from app import hostlimits as _hl  # noqa: PLC0415

        threads = int(_hl.inference_threads())
    except Exception:
        threads = None
    label = f"{model} ({compute})"
    if note:
        label = f"{model} ({compute}) — configured {configured} does not fit this host"
    if not local_present and bucket_present:
        label += " · weights in bucket, restored for each batch"
    return {
        "engine": engine,
        "engineDetail": detail,
        "device": device,
        "computeType": compute,
        "cpuThreads": threads,
        "model": model,
        "configured": configured,
        "mismatch": note,
        "label": label,
        "cacheDir": cache,
        "paths": paths,
        "filesPresent": files_present,
        "localPresent": local_present,
        "bucketPresent": bucket_present,
        "idle": not local_present,
        "sizeMb": round(size / 1048576, 1) if size else 0,
        "batchSize": batch_size(settings),
        "threads": threads,
        "usableCpus": os.cpu_count(),
    }
