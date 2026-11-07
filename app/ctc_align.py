"""CTC forced alignment with a small wav2vec2, on onnxruntime.

Why not stable-ts / ctc-forced-aligner: both pull in PyTorch. This host has
1.0 GB of free disk, no swap and a 2 GiB cgroup, so installing torch (+ a
wav2vec2 checkpoint) is not possible while a Whisper model is resident. The
acoustic model those tools use is the same one used here — wav2vec2-base-960h —
just exported to ONNX (95 MB quantized) and aligned by the CTC Viterbi pass
implemented below, which is what "forced alignment" means for a CTC model.

What it buys: word boundaries come from the ACOUSTICS and the known TEXT, not
from Whisper's attention timestamps. Given the text of a creator-uploaded
subtitle, the words land where they are spoken; for a Whisper transcript it
sharpens the boundaries that decide when a caption appears and disappears.

The aligner never changes text. It only assigns [start, end) to words that are
already there, and it refuses (rather than guesses) when the text cannot be
represented by the model's vocabulary.
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: 20 ms per frame: the wav2vec2 CNN stride is 320 samples at 16 kHz.
FRAME_SEC = 0.02
SAMPLE_RATE = 16000

REPO = "Xenova/wav2vec2-base-960h"
FILES = ("onnx/model_quantized.onnx", "vocab.json", "config.json")
MODEL_BYTES = 95 * 1024 * 1024

#: 960h's vocabulary. Anything outside it (digits, accents, non-Latin scripts)
#: cannot be aligned acoustically; those transcripts keep their original times.
_SUBSTITUTIONS = {
    "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...",
}

_session_cache: dict[str, object] = {}


def model_dir(cache_root: Path) -> Path:
    return Path(cache_root) / "wav2vec2-base-960h"


def model_ready(cache_root: Path) -> bool:
    d = model_dir(cache_root)
    return (d / "model_quantized.onnx").exists() and (d / "vocab.json").exists()


def ensure_model(cache_root: Path, *, timeout: float = 900.0) -> Path:
    """Download the ONNX checkpoint once, into the app's own model cache."""
    dest = model_dir(cache_root)
    dest.mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        out = dest / Path(rel).name
        if out.exists() and out.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{REPO}/resolve/main/{rel}"
        tmp = out.with_suffix(out.suffix + ".part")
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.replace(out)
    return dest


def _session(cache_root: Path):
    key = str(model_dir(cache_root))
    sess = _session_cache.get(key)
    if sess is not None:
        return sess
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(model_dir(cache_root) / "model_quantized.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    _session_cache[key] = sess
    return sess


def _vocab(cache_root: Path) -> tuple[dict[str, int], int]:
    raw = json.loads((model_dir(cache_root) / "vocab.json").read_text(encoding="utf-8"))
    blank = int(raw.get("<pad>", 0))
    return {str(k): int(v) for k, v in raw.items()}, blank


def _tokens(text: str, vocab: dict[str, int]) -> list[tuple[str, int, int]]:
    """(char, token id, word index) for every alignable character."""
    out: list[tuple[str, int, int]] = []
    word = 0
    in_word = False
    for ch in text:
        ch = _SUBSTITUTIONS.get(ch, ch)
        if ch.isspace():
            in_word = False
            continue
        token = ch.upper()
        if token not in vocab:
            # Keep the word boundary but drop the character: a number or an
            # accented letter makes that character unalignable, not the word.
            in_word = True
            continue
        if not in_word:
            word += 1
            in_word = True
        out.append((ch, vocab[token], word))
    return out


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _viterbi(emissions: np.ndarray, labels: list[int], blank: int) -> list[int]:
    """Monotonic CTC alignment of `labels` to frames. Returns per-frame state."""
    frames, _ = emissions.shape
    ext: list[int] = []
    for lab in labels:
        ext.append(blank)
        ext.append(lab)
    ext.append(blank)
    S = len(ext)
    neg = -1e30
    alpha = np.full(S, neg, dtype=np.float64)
    alpha[0] = emissions[0, ext[0]]
    if S > 1:
        alpha[1] = emissions[0, ext[1]]
    back = np.zeros((frames, S), dtype=np.uint8)
    for t in range(1, frames):
        prev = alpha
        best = prev.copy()
        choice = np.zeros(S, dtype=np.uint8)
        cand = np.full(S, neg)
        cand[1:] = prev[:-1]
        take = cand > best
        best = np.where(take, cand, best)
        choice = np.where(take, 1, choice).astype(np.uint8)
        if S > 2:
            skip_ok = np.array(
                [
                    s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]
                    for s in range(S)
                ]
            )
            cand2 = np.full(S, neg)
            cand2[2:] = prev[:-2]
            take2 = skip_ok & (cand2 > best)
            best = np.where(take2, cand2, best)
            choice = np.where(take2, 2, choice).astype(np.uint8)
        alpha = best + emissions[t, np.array(ext)]
        back[t] = choice
    end = S - 1
    if S >= 2 and alpha[S - 2] > alpha[S - 1]:
        end = S - 2
    path = [0] * frames
    state = end
    for t in range(frames - 1, -1, -1):
        path[t] = state
        if t:
            state = state - int(back[t][state])
    return path


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float


def align_text(
    samples: np.ndarray,
    text: str,
    *,
    cache_root: Path,
    offset: float = 0.0,
    speech: list[tuple[float, float]] | None = None,
) -> list[AlignedWord] | None:
    """Force-align `text` to `samples` (16 kHz mono float32).

    Returns word spans in absolute seconds, or None when this text cannot be
    aligned (no alignable characters, or a span too short for the frame rate).

    `speech` is a list of (start, end) SILERO regions in the clock of
    `samples` itself (0 = first sample of `samples`), not the file's. Frames
    outside them may only emit blank.

    Without that restriction the lattice will happily place speech into music:
    measured on this app's own clip, the first 6.7 s are a music bed that
    Whisper and Silero both reject, yet raw wav2vec2 still finds letters in it
    (greedy decode: "NA YOU RA RA WA WA") and the aligner pulls the opening
    words there. Masking is what makes the alignment agree with the audio the
    captions are actually shown over.
    """
    text = (text or "").strip()
    if not text or samples.size < SAMPLE_RATE // 10:
        return None
    if not model_ready(cache_root):
        return None
    vocab, blank = _vocab(cache_root)
    toks = _tokens(text, vocab)
    if len(toks) < 1:
        return None
    labels = [t[1] for t in toks]
    x = np.asarray(samples, dtype=np.float32)
    mean = float(x.mean())
    std = float(x.std())
    x = (x - mean) / math.sqrt(std * std + 1e-7)
    sess = _session(cache_root)
    logits = sess.run(None, {"input_values": x[None, :]})[0]
    emissions = _log_softmax(np.asarray(logits, dtype=np.float32)[0])
    frames = emissions.shape[0]
    if speech is not None:
        mask = np.zeros(frames, dtype=bool)
        for lo, hi in speech:
            a = max(0, int(round(float(lo) / FRAME_SEC)))
            b = min(frames, int(round(float(hi) / FRAME_SEC)) + 1)
            if b > a:
                mask[a:b] = True
        if not mask.any():
            return None
        emissions = np.where(mask[:, None], emissions, -1e30)
        emissions[:, blank] = np.where(mask, emissions[:, blank], -60.0)
    if frames < len(labels) + 2:
        return None
    path = _viterbi(emissions, labels, blank)

    # frame range per token index (chars are tokens[1::2] in the extended path)
    spans: dict[int, list[int]] = {}
    for frame, state in enumerate(path):
        if state % 2 == 1:
            idx = (state - 1) // 2
            lo_hi = spans.setdefault(idx, [frame, frame])
            lo_hi[1] = frame
    words: dict[int, list[float]] = {}
    order: list[int] = []
    for idx, (ch, _lab, word) in enumerate(toks):
        if idx not in spans:
            continue
        lo, hi = spans[idx]
        if word not in words:
            words[word] = [lo * FRAME_SEC, (hi + 1) * FRAME_SEC, ch]
            order.append(word)
        else:
            words[word][1] = (hi + 1) * FRAME_SEC
    if not order:
        return None
    # Rebuild the word text in reading order so no character is invented.
    word_text: dict[int, str] = {}
    for ch, _lab, word in toks:
        word_text[word] = word_text.get(word, "") + ch
    out = [
        AlignedWord(word_text[w], words[w][0] + offset, words[w][1] + offset)
        for w in order
    ]
    out.sort(key=lambda a: a.start)
    return out


def align_words_to_audio(
    samples: np.ndarray,
    words: list[dict],
    *,
    cache_root: Path,
    offset: float = 0.0,
) -> list[dict]:
    """Convenience wrapper: align a transcript's own words, keep original text.

    On any refusal the original timings are returned unchanged — a transcript
    with coarse timings is still a transcript; a transcript with invented ones
    is not.
    """
    text = " ".join(str(w.get("word", "")) for w in words)
    aligned = align_text(samples, text, cache_root=cache_root, offset=offset)
    if not aligned or len(aligned) != len(words):
        return words
    out = []
    for src, ali in zip(words, aligned):
        out.append({**src, "start": round(ali.start, 3), "end": round(ali.end, 3)})
    return out
