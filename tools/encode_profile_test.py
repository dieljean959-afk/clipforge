#!/usr/bin/env python3
"""Tests for render graceful degradation (app/encode.py).

The behaviour under test is what makes a CPU-only host finish rather than
stall: a faster x264 preset, coarser frame sampling for analysis, a deadline
sized to the mode, and hardware-encoder detection that does NOT trust a bare
device node.

Run: python3 tools/encode_profile_test.py
"""
from __future__ import annotations

import os as _os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import encode, hostlimits  # noqa: E402
from app.config import load_config  # noqa: E402


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"   {'OK ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    settings = load_config(quiet=True)
    settings.video.preset = "veryfast"
    settings.video.crf = 20

    # --- 1. preset clamping ---------------------------------------------
    print("== 1. x264 preset falls back on a CPU-only host")
    preset, note = encode.resolve_preset("veryfast")
    check("veryfast is lowered on CPU", preset == "ultrafast", f"veryfast → {preset}")
    check("the downgrade is explained, not silent", note != "", note[:60])
    preset, note = encode.resolve_preset("medium")
    check("a slow preset is lowered further", preset == "ultrafast", f"medium → {preset}")
    preset, note = encode.resolve_preset("ultrafast")
    check("ultrafast is left alone", preset == "ultrafast" and note == "")
    preset, note = encode.resolve_preset("superfast")
    check("superfast is left alone (the floor)", preset == "superfast" and note == "")

    # --- 2. the per-clip deadline is derived and bounded ------------------
    print()
    print("== 2. the render deadline scales with the clip and is bounded")
    # This used to assert a flat 90-minute CPU deadline was "generous enough".
    # It was not generous, it was unbounded: a hung ffmpeg held the only worker
    # slot for an hour and a half and every other job sat at 0% behind it.
    d = encode.render_deadline_sec
    check("a zero-length clip still gets the full floor",
          d(0, hardware=False) == encode.RENDER_DEADLINE_FLOOR_SEC,
          f"{d(0, hardware=False) // 60} min")
    check("a short clip never drops below the floor",
          d(1, hardware=False) >= encode.RENDER_DEADLINE_FLOOR_SEC,
          f"{d(1, hardware=False) // 60} min")
    check("a longer clip gets a proportionally longer budget",
          d(60, hardware=False) > d(10, hardware=False),
          f"{d(10, hardware=False) // 60}min -> {d(60, hardware=False) // 60}min")
    check("the budget is monotonic in clip length",
          all(d(a, hardware=False) <= d(b, hardware=False) for a, b in ((5, 20), (20, 90), (90, 300))))
    check("an absurdly long clip is still capped",
          d(100_000, hardware=False) == encode.RENDER_DEADLINE_CEILING_SEC,
          f"{d(100_000, hardware=False) // 60} min")
    check("a typical clip lands in the 5-15 minute band the user asked for",
          5 <= d(52, hardware=False) // 60 <= 15, f"{d(52, hardware=False) // 60} min")
    check("no CPU clip can be given 90 minutes again",
          d(1800, hardware=False) < 60 * 60, f"{d(1800, hardware=False) // 60} min")
    check("the GPU deadline is unchanged",
          d(52, hardware=True) == encode.RENDER_TIMEOUT_GPU_SEC,
          f"{d(52, hardware=True) // 60} min")
    check("the default CPU deadline is now minutes, not an hour and a half",
          encode.RENDER_TIMEOUT_CPU_SEC <= 20 * 60,
          f"{encode.RENDER_TIMEOUT_CPU_SEC // 60} min")

    encode.reset_cache()
    real_hw = encode.hardware_encoder
    try:
        # software host
        encode.hardware_encoder = lambda: ("", "x264 (software)", "")  # type: ignore[assignment]
        cpu = encode.choose(settings)
        check("CPU host reports software encoding", not cpu.hardware, cpu.label)
        check("CPU host uses the long deadline",
              cpu.timeout_sec == encode.RENDER_TIMEOUT_CPU_SEC)
        check("CPU host records the preset downgrade",
              cpu.degraded_from == "veryfast" and cpu.preset == "ultrafast")
        check("CPU host explains the deadline in a note",
              any("deadline" in n for n in cpu.notes), cpu.notes[-1][:60])

        # hardware host
        encode.hardware_encoder = lambda: ("h264_nvenc", "NVENC (NVIDIA, hardware)", "cuda")  # type: ignore[assignment]
        gpu = encode.choose(settings)
        check("GPU host reports hardware encoding", gpu.hardware, gpu.label)
        check("GPU host keeps the requested preset", gpu.preset == "veryfast", gpu.preset)
        check("GPU host uses the strict deadline",
              gpu.timeout_sec == encode.RENDER_TIMEOUT_GPU_SEC)
        check("GPU host records no downgrade", gpu.degraded_from == "")

        # forced software (e.g. an operator escape hatch)
        forced = encode.choose(settings, force_software=True)
        check("force_software overrides detection", not forced.hardware, forced.label)
        check("forced software still gets the CPU deadline",
              forced.timeout_sec == encode.RENDER_TIMEOUT_CPU_SEC)
    finally:
        encode.hardware_encoder = real_hw  # type: ignore[assignment]
        encode.reset_cache()

    # --- 3. codec args per encoder --------------------------------------
    print()
    print("== 3. each encoder gets correct ffmpeg args")
    libx = encode.EncoderChoice(preset="ultrafast", crf=20)
    args = libx.ffmpeg_codec_args()
    check("x264 path uses libx264 + preset + crf",
          args[:2] == ["-c:v", "libx264"] and "ultrafast" in args and "20" in args, " ".join(args))

    nvenc = encode.EncoderChoice(encoder="h264_nvenc", hardware=True, preset="veryfast", crf=20)
    args = nvenc.ffmpeg_codec_args()
    check("NVENC path uses h264_nvenc, not libx264",
          args[:2] == ["-c:v", "h264_nvenc"] and "libx264" not in args, " ".join(args))
    check("NVENC path uses -cq (NVENC has no CRF)", "-cq" in args)
    check("NVENC path uses an NVENC preset, not an x264 one",
          args[args.index("-preset") + 1] == "p5", "preset p5 (from veryfast)")
    check("NVENC path never sends an x264 CRF", "-crf" not in args, " ".join(args))
    # ultrafast is the CPU-fallback preset; on NVENC it maps to p4 (fastest).
    nvenc_fast = encode.EncoderChoice(
        encoder="h264_nvenc", hardware=True, preset="ultrafast", crf=20
    ).ffmpeg_codec_args()
    check("an ultrafast preset maps to NVENC p4",
          nvenc_fast[nvenc_fast.index("-preset") + 1] == "p4", "preset p4")

    vaapi = encode.EncoderChoice(encoder="h264_vaapi", hardware=True, crf=20)
    args = vaapi.ffmpeg_codec_args()
    check("VAAPI path uses a bitrate, not CRF",
          "-b:v" in args and "-crf" not in args, " ".join(args))

    qsv = encode.EncoderChoice(encoder="h264_qsv", hardware=True, crf=20)
    args = qsv.ffmpeg_codec_args()
    check("QSV path uses global_quality", "-global_quality" in args, " ".join(args))

    # --- 4. frame sampling for analysis ---------------------------------
    print()
    print("== 4. analysis samples fewer frames on a CPU-only host")
    hw_hz = encode.analysis_sample_hz(60.0, hardware=True)
    cpu_hz = encode.analysis_sample_hz(60.0, hardware=False)
    check("hardware keeps the dense sampling", hw_hz == 2.0, f"{hw_hz}/s")
    check("CPU-only sampling is coarser", cpu_hz < hw_hz, f"{cpu_hz}/s < {hw_hz}/s")
    check("CPU-only sampling is still frequent enough to find shots",
          cpu_hz >= 0.5, f"{cpu_hz}/s")

    # scdet must actually subsample frames, or the sampling is cosmetic.
    cpu_filter = encode.scdet_sample_filter(10, cpu_hz, 60.0)
    check("scdet drops frames on a CPU-only host", "select=" in cpu_filter, cpu_filter)
    hw_filter = encode.scdet_sample_filter(10, hw_hz, 60.0)
    check("scdet still subsamples on hardware (less often)",
          "select=" in hw_filter and hw_filter != cpu_filter, hw_filter)
    check("scdet filter keeps the real detector", "scdet=" in cpu_filter and "scdet=" in hw_filter)
    # 60fps and 30fps inputs should analyse at a similar wall-clock rate.
    f60 = encode.scdet_sample_filter(10, cpu_hz, 60.0)
    f30 = encode.scdet_sample_filter(10, cpu_hz, 30.0)
    check("stride scales with source fps (60fps skips more frames than 30fps)",
          int(f60.split("\\,")[1].split(")")[0]) > int(f30.split("\\,")[1].split(")")[0]),
          f"60fps {f60} vs 30fps {f30}")
    check("a high sample rate degenerates to no select (every frame)",
          "select=" not in encode.scdet_sample_filter(10, 999.0, 30.0))

    # --- 5. a listed-but-unusable encoder is never selected -------------
    print()
    print("== 5. a listed hardware encoder is verified before use")
    check("the functional probe exists", callable(encode._encoder_works))
    # A codec that definitely is not compiled in must fail the probe.
    check("a bogus encoder fails the functional probe",
          encode._encoder_works("definitely_not_an_encoder", "") is False)
    # x264 is a real software encoder and must pass — proves the probe is not
    # just failing everything.
    check("a real encoder passes the functional probe",
          encode._encoder_works("libx264", "") is True)

    # summary should name the mode so logs are greppable
    check("summary names the deadline", "deadline" in encode.summary(cpu),
          encode.summary(cpu))

    # --- 6. adaptive bitrate -------------------------------------------
    print()
    print("== 6. the VBR ceiling is derived from the output geometry")
    t = encode.target_bitrate_kbps
    check("1080x1920@60 targets a sane ceiling", t(1080, 1920, 60) == 7465, f"{t(1080, 1920, 60)} kbps")
    check("30fps gets roughly half the 60fps ceiling",
          abs(t(1080, 1920, 30) - t(1080, 1920, 60) / 2) < 40,
          f"{t(1080, 1920, 30)} vs {t(1080, 1920, 60)}")
    check("4K is allowed more than 1080p", t(2160, 3840, 60) > t(1080, 1920, 60),
          f"{t(2160, 3840, 60)} vs {t(1080, 1920, 60)}")
    check("a tiny output is floored, not starved",
          t(320, 240, 15) == encode.MIN_MAXRATE_KBPS, f"{t(320, 240, 15)}")
    check("a huge output is capped", t(7680, 4320, 120) == encode.MAX_MAXRATE_KBPS)
    check("a nonsense fps falls back instead of exploding",
          t(1080, 1920, 0) == t(1080, 1920, 30), f"{t(1080, 1920, 0)}")

    print()
    print("== 7. the ceiling reaches ffmpeg, and the CRF adapts to the preset")
    adaptive = encode.EncoderChoice(preset="ultrafast", crf=20).for_output(1080, 1920, 60)
    a = adaptive.ffmpeg_codec_args()
    check("libx264 gets -maxrate", "-maxrate" in a, " ".join(a))
    check("libx264 gets -bufsize", "-bufsize" in a)
    check("bufsize is twice maxrate (VBV wants 2x)",
          a[a.index("-bufsize") + 1] == f"{adaptive.maxrate_kbps * 2}k")
    check("the emitted maxrate matches the computed target",
          a[a.index("-maxrate") + 1] == f"{adaptive.maxrate_kbps}k")
    check("ultrafast raises the CRF (it is the least efficient preset)",
          adaptive.crf == 23, f"crf {adaptive.crf}")
    medium = encode.EncoderChoice(preset="medium", crf=20).for_output(1080, 1920, 60)
    check("a capable preset keeps or lowers the CRF", medium.crf <= 20, f"crf {medium.crf}")
    check("the ceiling is the same regardless of preset",
          medium.maxrate_kbps == adaptive.maxrate_kbps)

    # every encoder family must carry the ceiling, or the saving silently
    # evaporates on a host with a GPU.
    for name in ("h264_nvenc", "h264_qsv"):
        hw = encode.EncoderChoice(
            encoder=name, hardware=True, preset="ultrafast", crf=20
        ).for_output(1080, 1920, 60)
        hw_args = hw.ffmpeg_codec_args()
        check(f"{name} carries the ceiling", "-maxrate" in hw_args, " ".join(hw_args))
    vaapi_adaptive = encode.EncoderChoice(
        encoder="h264_vaapi", hardware=True, crf=20
    ).for_output(1080, 1920, 60)
    vaapi_args = vaapi_adaptive.ffmpeg_codec_args()
    check("VAAPI uses the adaptive target as its bitrate",
          f"{vaapi_adaptive.maxrate_kbps}k" in vaapi_args, " ".join(vaapi_args))
    check("VAAPI does not also get -maxrate (it IS the bitrate)",
          "-maxrate" not in vaapi_args)

    uncapped = encode.EncoderChoice(preset="ultrafast", crf=20)
    check("without for_output() the behaviour is unchanged (no cap)",
          "-maxrate" not in uncapped.ffmpeg_codec_args()
          and uncapped.crf == 20,
          " ".join(uncapped.ffmpeg_codec_args()))
    check("summary reports the ceiling",
          "maxrate" in encode.summary(adaptive), encode.summary(adaptive))
    check("as_dict reports the ceiling",
          encode.as_dict(adaptive).get("maxrateKbps") == adaptive.maxrate_kbps)

    # --- 8. the encoder is told how many CPUs it really has ------------
    print()
    print("== 8. ffmpeg threads come from the container, not from nproc")
    # ffmpeg sizes its thread pools from nproc, which inside a cgroup-limited
    # host reports the HOST's cores. This host reports 64 but is allowed 1, so
    # every render was spawning ~64 frame + ~64 slice threads to share a
    # single CPU. Measured on a real 1080x1920 clip: 10s -> 4s (2.5x faster)
    # for byte-identical output (SSIM 0.9999).
    settings = load_config(quiet=True)
    cpu_choice = encode.choose(settings)
    check("a thread count is always chosen", cpu_choice.threads >= 1, str(cpu_choice.threads))
    check(
        "it never exceeds the container's CPU allowance",
        cpu_choice.threads <= max(1, hostlimits.effective_cpus()),
        f"threads={cpu_choice.threads} cpus={hostlimits.effective_cpus()}",
    )
    check(
        "it is NOT taken from nproc on a limited host",
        cpu_choice.threads < _os.cpu_count() or hostlimits.effective_cpus() >= _os.cpu_count(),
        f"threads={cpu_choice.threads} nproc={_os.cpu_count()}",
    )
    targs = cpu_choice.threads_args()
    check("-threads is emitted", "-threads" in targs, " ".join(targs))
    check("-filter_threads is emitted", "-filter_threads" in targs)
    check("-filter_complex_threads is emitted", "-filter_complex_threads" in targs)
    check("each thread count matches the allowance",
          targs[targs.index("-threads") + 1] == str(cpu_choice.threads))
    full = cpu_choice.for_output(1080, 1920, 60).ffmpeg_codec_args()
    check("the thread limits reach the real ffmpeg command line", "-threads" in full, " ".join(full))
    # The thread cap must reach the hardware paths too, or a GPU host would
    # keep the unbounded pools. `threads` is set by choose(); constructing an
    # EncoderChoice directly leaves it at 0 (no cap), which is the safe default.
    check(
        "thread limits apply to hardware encoders too (NVENC/QSV)",
        all("-threads" in encode.EncoderChoice(encoder=n, hardware=True, crf=20, threads=1)
            .for_output(1080, 1920, 60).ffmpeg_codec_args()
            for n in ("h264_nvenc", "h264_qsv")),
    )
    check(
        "an unset thread count emits nothing (no bogus -threads 0)",
        encode.EncoderChoice(threads=0).threads_args() == [],
    )

    print()
    if failures:
        print(f"ENCODE PROFILE: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("ENCODE PROFILE: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
