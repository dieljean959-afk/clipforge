#!/usr/bin/env python3
"""The subtitle-accuracy proof, end to end, on this host.

Everything printed here is measured:
    models     load each Whisper model in a guarded subprocess, record peak RSS
               and WER against a known transcript (LibriSpeech dev-clean).
    transcribe run the real pipeline (app.asr_accurate) over a real clip and
               print the sorted word list, the WER against whatever reference
               exists, the flagged words and the no-reference sync report.
    frames     burn the transcript into 3 clips and save a frame from each.
    identity   the same word list must produce byte-identical .ass and
               pixel-identical frames (diff = 0).

    python tools/verify_subtitles.py models
    python tools/verify_subtitles.py transcribe --media data/work/<job>/transcribe/chunks/source.wav
    python tools/verify_subtitles.py frames --media <source.mp4>
    python tools/verify_subtitles.py identity --media <source.mp4>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import asr_quality, asr_accurate, subs_source  # noqa: E402
from app.config import ffmpeg_bin, load_config  # noqa: E402

DEFAULT_MEDIA = Path(
    "/home/daytona/clipforge-data/downloads/e9206bc370324eb3a89d531c1e9f5142/source.mp4"
)
DEFAULT_AUDIO = ROOT / "data/work/e9206bc370324eb3a89d531c1e9f5142/transcribe/chunks/source.wav"
WORK = ROOT / "data/work/verify"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit(f"FAILED {' '.join(cmd[:6])}\n{proc.stderr[-1500:]}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

def librispeech_clips(n: int = 6) -> list[tuple[Path, str]]:
    """Public speech with a human transcript: the only honest WER reference."""
    import urllib.request

    work = WORK / "librispeech"
    work.mkdir(parents=True, exist_ok=True)
    url = (
        "https://datasets-server.huggingface.co/rows?dataset=hf-internal-testing"
        f"%2Flibrispeech_asr_dummy&config=clean&split=validation&offset=0&length={n}"
    )
    rows = json.load(urllib.request.urlopen(url, timeout=60))["rows"]
    clips = []
    for row in rows:
        item = row["row"]
        flac = work / f"{item['id']}.flac"
        wav = work / f"{item['id']}.wav"
        if not flac.exists():
            urllib.request.urlretrieve(item["audio"][0]["src"], flac)
        if not wav.exists():
            _run([ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(flac),
                  "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
        clips.append((wav, item["text"]))
    return clips


def cmd_models(args) -> int:
    print(f"cgroup memory.max: {(Path('/sys/fs/cgroup/memory.max').read_text().strip())} bytes")
    print(f"RSS cap for a decode child: {asr_quality.rss_cap_mb():.0f} MB "
          f"(cgroup limit - {asr_quality.HEADROOM_MB:.0f} MB headroom)")
    for model in ("large-v3-turbo", "medium", "distil-medium.en", "small.en", "small"):
        print(f"  fit table: {model:18s} measured peak {asr_quality.peak_for(model):6.0f} MB "
              f"-> {'fits' if asr_quality.fits(model) else 'does NOT fit'}")

    clips = librispeech_clips(args.clips)
    print(f"\nWER reference: {len(clips)} LibriSpeech dev-clean clips "
          f"({sum(len(t.split()) for _, t in clips)} words)")
    for model in args.models:
        if not asr_quality.fits(model):
            print(f"\n{model}: SKIPPED — measured peak {asr_quality.peak_for(model):.0f} MB "
                  f"exceeds the {asr_quality.rss_cap_mb():.0f} MB cap")
            continue
        err_words = 0.0
        total = 0
        peak = 0.0
        sample = ""
        try:
            with asr_quality.DecodeProcess(model, log_path=WORK / "asr_child.log") as proc:
                for wav, ref in clips:
                    rep = proc.decode(wav, language="en", beam_size=5,
                                      temperature=[0.0, 0.2, 0.4], initial_prompt=None)
                    hyp = " ".join(w["word"] for w in rep["words"])
                    err = asr_accurate.wer(ref, hyp) or 0.0
                    n = len(ref.split())
                    err_words += err * n
                    total += n
                    peak = max(peak, rep["peak_rss_mb"])
                    sample = sample or hyp
        except asr_quality.DecodeError as exc:
            print(f"\n{model}: FAILED — {exc}")
            continue
        print(f"\n{model}: WER {err_words / max(1, total):.1%} | peak {peak:.0f} MB "
              f"| load+decode OK")
        print(f"  sample: {sample[:120]}")
    return 0


# --------------------------------------------------------------------------- #
# transcribe
# --------------------------------------------------------------------------- #

def _reference_for(media: Path, cache: Path, url: str | None) -> tuple[str, str]:
    if url:
        for auto in (False, True):
            files = subs_source.fetch(url, cache / ("subs" if not auto else "auto"), auto=auto)
            best = subs_source.pick_best(files)
            if best:
                cues = subs_source.parse_file(best)
                if cues:
                    return subs_source.cues_text(cues), ("creator captions" if not auto else "auto captions")
    return "", ""


def cmd_transcribe(args) -> int:
    settings = load_config(quiet=True)
    media = Path(args.media)
    cache = WORK / "transcribe"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"media: {media}  ({media.stat().st_size / 1e6:.1f} MB)")
    print(f"model ladder for this host: {', '.join(asr_accurate.model_ladder(args.language))}")
    transcript = asr_accurate.transcribe_source(
        media, settings, cache_dir=cache, language=args.language,
        source_url=args.url, title=args.title,
    )
    (cache / "words.json").write_text(json.dumps({
        "words": [w.as_dict() for w in transcript.words],
        "sync": transcript.sync, "model": transcript.model_size,
        "label": transcript.label, "language": transcript.language,
    }, indent=2), encoding="utf-8")
    print(f"\nengine: {transcript.engine} | model: {transcript.model_size} "
          f"| label: {transcript.label!r} | peak RSS {transcript.peak_rss_mb:.0f} MB")
    print(f"words: {len(transcript.words)} | flagged 'check this': {transcript.check_words} "
          f"| aligned: {transcript.aligned} | needs review: {transcript.needs_review}")

    print("\nsorted word list, first 15 s (word, start, end, confidence, flag):")
    for w in transcript.words:
        if w.start > 15.0:
            break
        flag = "  CHECK" if w.check else ""
        print(f"  {w.word:>14s} {w.start:7.2f} {w.end:7.2f}  {w.conf:4.2f}{flag}")

    flagged = [(w.word, round(w.start, 2)) for w in transcript.words if w.check]
    print(f"\nflagged words: {flagged if flagged else 'none'}")

    print("\nsync report (Silero VAD vs caption times; no reference needed):")
    for key, value in transcript.sync.items():
        print(f"  {key:22s} {value}")

    reference, kind = "", ""
    if args.ref_text:
        reference, kind = args.ref_text, "text given on the command line"
    else:
        reference, kind = _reference_for(media, cache, args.url)
    if reference:
        rate = asr_accurate.wer(reference, " ".join(w.word for w in transcript.words))
        print(f"\nWER against {kind}: {rate:.1%} "
              f"({'LOW-CONFIDENCE TRANSCRIPT' if rate and rate > asr_accurate.MAX_WER else 'ok'})")
    else:
        print("\nWER: BLOCKED — no reference captions available offline for this clip")
    return 0


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #

def _cues_for(words: list[dict], start: float, end: float) -> list[dict]:
    from app import subtitle as sub

    inside = [
        {**w, "start": float(w["start"]) - start, "end": float(w["end"]) - start}
        for w in words if start - 0.05 <= float(w["start"]) <= end
    ]
    cues = sub.words_to_cues(inside, words_per_cue=4, max_words=5, max_chars=36,
                             max_duration=2.6, pause_sec=0.4)
    sub.extend_cue_gaps(cues, 0.5)
    return cues


def _render(src: Path, start: float, end: float, cues: list[dict], dest: Path,
            settings, title: str) -> dict:
    from app import encode, video

    probe = video.probe(src)
    cut = {
        "start": start, "end": end, "order": 0,
        "cropBox": {"x": 0, "y": 0, "w": int(probe["width"]), "h": int(probe["height"])},
        "framingSegments": [{"start": 0.0, "end": end - start, "mode": "fit"}],
    }
    return encode.render_clip(src, cut, dest, settings, cues=cues,
                              preset_key="bold_stack", emoji_selection=[], title=title)


def cmd_frames(args) -> int:
    settings = load_config(quiet=True)
    src = Path(args.media)
    cache = WORK / "transcribe"
    data = json.loads((cache / "words.json").read_text()) if (cache / "words.json").exists() else None
    if data is None:
        transcript = asr_accurate.transcribe_source(src, settings, cache_dir=cache,
                                                    language=args.language, title=args.title)
        words = [w.as_dict() for w in transcript.words]
    else:
        words = data["words"]
    # Build the SAME cues the app builds, once, over the whole transcript; then
    # render a short clip around three confident cues. Picking a word and hoping
    # a cue covers it renders frames with no caption at all.
    duration = max(float(w["end"]) for w in words) + 1.0
    all_cues = _cues_for(words, 0.0, duration)

    def cue_conf(cue: dict) -> float:
        inside = [w["conf"] for w in words
                  if float(w["start"]) >= float(cue["start"]) - 1e-6
                  and float(w["end"]) <= float(cue["end"]) + 1e-6]
        return sum(inside) / len(inside) if inside else 0.0

    spoken = [c for c in all_cues if cue_conf(c) >= 0.75] or all_cues
    picks = [spoken[min(int(i), len(spoken) - 1)] for i in args.pick]
    out_dir = WORK / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"source: {src}  ({len(all_cues)} cues, {len(spoken)} confident)")
    for n, cue in enumerate(picks, start=1):
        start = max(0.0, float(cue["start"]) - 0.8)
        end = min(float(cue["end"]) + 1.6, start + 3.2)
        cues = [
            {**c, "start": float(c["start"]) - start, "end": float(c["end"]) - start}
            for c in all_cues
            if float(c["end"]) > start and float(c["start"]) < end
        ]
        clip = out_dir / f"clip_{n}.mp4"
        _render(src, start, end, cues, clip, settings, args.title)
        at = max(0.05, float(cue["start"]) - start + 0.30)
        png = out_dir / f"frame_{n}.png"
        _run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
              "-ss", f"{at:.3f}", "-i", str(clip), "-frames:v", "1", str(png)])
        on_screen = [c for c in cues if float(c["start"]) <= at <= float(c["end"])]
        print(f"\n[{n}] cue at {cue['start']:.2f}-{cue['end']:.2f}s "
              f"(word conf {cue_conf(cue):.2f}) -> clip {start:.2f}-{end:.2f}s")
        print(f"    cue on screen at +{at:.2f}s: {on_screen[0].get('text') if on_screen else '(none)'}")
        print(f"    frame: {png}  ass: {clip.with_suffix('.ass')}")
    return 0


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #

def cmd_identity(args) -> int:
    """Same word list in, same bytes out: .ass and pixels."""
    from PIL import Image, ImageChops

    from app import subtitle as sub

    settings = load_config(quiet=True)
    src = Path(args.media)
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
              "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=12",
              "-f", "lavfi", "-i", "sine=frequency=330:duration=12",
              "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
              "-c:a", "aac", "-shortest", str(src)])
    words = [
        {"word": "THE", "start": 0.20, "end": 0.55, "conf": 0.9},
        {"word": "WHOLE", "start": 0.55, "end": 0.90, "conf": 0.9},
        {"word": "GAME", "start": 0.90, "end": 1.30, "conf": 0.9},
        {"word": "WAS", "start": 1.30, "end": 1.60, "conf": 0.9},
        {"word": "WON", "start": 1.60, "end": 2.00, "conf": 0.9},
    ]
    cues = sub.words_to_cues(words, words_per_cue=5, max_words=5, max_chars=36,
                             max_duration=2.6, pause_sec=0.4)
    out = WORK / "identity"
    out.mkdir(parents=True, exist_ok=True)
    start, end = 5.0, 8.0
    digests, rendered = [], []
    for run in (1, 2):
        clip = out / f"render_{run}.mp4"
        _render(src, start, end, [dict(c) for c in cues], clip, settings, args.title)
        ass = clip.with_suffix(".ass")
        digests.append(sha256(ass))
        frame = out / f"frame_{run}.png"
        _run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
              "-ss", "1.200", "-i", str(clip), "-frames:v", "1", str(frame)])
        rendered.append(frame)
        print(f"run {run}: ass sha256 {digests[-1]}  ({ass.stat().st_size} bytes)")
    same_bytes = digests[0] == digests[1]
    a, b = Image.open(rendered[0]).convert("RGB"), Image.open(rendered[1]).convert("RGB")
    diff = ImageChops.difference(a, b)
    changed = sum(1 for px in diff.getdata() if sum(px) > 0)
    print(f"\n.ass byte-identical: {same_bytes}")
    print(f"frame pixels differing: {changed} (of {a.width * a.height}) -> diff = {changed}")
    return 0 if same_bytes and changed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub_p = ap.add_subparsers(dest="cmd", required=True)

    m = sub_p.add_parser("models")
    m.add_argument("--clips", type=int, default=6)
    m.add_argument("--models", nargs="+",
                   default=["large-v3-turbo", "medium", "distil-medium.en", "small.en", "small"])
    m.set_defaults(func=cmd_models)

    t = sub_p.add_parser("transcribe")
    t.add_argument("--media", type=Path, default=DEFAULT_AUDIO)
    t.add_argument("--url", default=None)
    t.add_argument("--title", default="")
    t.add_argument("--language", default=None)
    t.add_argument("--ref-text", default=None)
    t.set_defaults(func=cmd_transcribe)

    f = sub_p.add_parser("frames")
    f.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    f.add_argument("--title", default="")
    f.add_argument("--language", default="en")
    f.add_argument("--pick", nargs="+", type=int, default=[3, 12, 20])
    f.set_defaults(func=cmd_frames)

    i = sub_p.add_parser("identity")
    i.add_argument("--media", type=Path, default=Path("/tmp/cap/src16x9.mp4"))
    i.add_argument("--title", default="")
    i.set_defaults(func=cmd_identity)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
