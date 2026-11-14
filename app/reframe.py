"""MediaPipe face-tracking reframe (optional dependency).

Produces a time series of 9:16 crop windows that follow the active speaker.
Shot-aware: given the source's detected cut boundaries, each shot is framed
independently and the sample at a cut is flagged so the crop snaps (no
interpolated pan through a cut). If mediapipe is not installed, reframe falls
back to a centered crop and logs it — no fabricated tracking data.
"""

from __future__ import annotations

import os
import threading
import time

# One process, two cores. Set these before OpenCV/MediaPipe spawn their
# thread pools; the old reframe let each library choose cores independently and
# oversubscribed the host while the single worker slot was already busy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")

from loguru import logger

_MP_AVAILABLE: bool | None = None
_FACE_LOCK = threading.Lock()
_FACE_DETECTOR = None

#: Hard wall-clock budget for ONE clip's face sampling. The loop is finite by
#: construction, but a corrupt/remote stream can still make each grab slow.
CLIP_ANALYSIS_BUDGET_SEC = float(os.getenv("CLIPFORGE_REFRAME_CLIP_BUDGET_SEC", "120"))


def _load_mediapipe():
    global _MP_AVAILABLE
    if _MP_AVAILABLE is False:
        return None
    try:
        import mediapipe as mp  # type: ignore

        _MP_AVAILABLE = True
        return mp
    except Exception as exc:
        logger.info(f"[reframe] MediaPipe unavailable ({type(exc).__name__}) — center crop fallback")
        _MP_AVAILABLE = False
        return None


def _center_window(src_width: int, src_height: int, target_aspect: float, t: float) -> dict:
    w = min(int(src_height * target_aspect), src_width)
    return {"t": t, "x": (src_width - w) // 2, "y": 0, "w": w, "h": src_height}


# --------------------------------------------------------------------------- #
# TARGET A: shot-level FIT/FILL framing decisions.
#
# The singal that a shot needs FIT (full frame over blurred pad, not a tight
# crop) is ALWAYS one of:
#   * the face band + 20% padding does not fit the 9:16 window
#     (a 9:16 window over a 16:9 source is 9/16 of the height, which is
#     31.6% of the width); OR
#   * 3 or more detected people; OR
#   * no faces detected at all (wide shot, screen share, B-roll).
#
# The decision is made per SCENE SHOT (the detected cut boundaries), never
# per clip, so a mode switch only ever happens at a real shot boundary, as a
# hard cut, and never mid-word (cuts come from ffmpeg scdet on the picture;
# every switch is between two shots).
# --------------------------------------------------------------------------- #

#: Faces + this much padding must fit the 9:16 crop for the crop to keep the
#: whole group. 20% of the face-band width.
FACE_PAD_FRACTION = 0.20

#: People count at which the shot is declared a group shot (FIT).
GROUP_FACE_COUNT = 3

#: Minimum seconds between framing-mode switches.
MIN_SWITCH_GAP_SEC = 1.0


def fit_or_fill_for_shot(
    face_count: int,
    face_band: tuple[float, float] | None,
    src_width: int,
    src_height: int,
) -> tuple[str, str]:
    """Framing mode + logged reason for one shot.

    face_band is (xmin, xmax) in pixels of the widest face span in the shot
    (None when no faces were detected). Returns ("fit"|"fill", reason).
    """
    faceless_reason = "no faces detected — wide shot / screen share / B-roll"
    if face_count <= 0 or face_band is None:
        return "fit", faceless_reason
    if face_count >= GROUP_FACE_COUNT:
        return "fit", f"{face_count} people in shot — group shot"
    # Single/duo speaker: FILL only if the padded band actually fits the crop.
    crop_w = min(int(src_height * (9 / 16)), src_width)
    span = face_band[1] - face_band[0]
    padded = span * (1 + FACE_PAD_FRACTION)
    if padded > crop_w * 0.98:
        return "fit", (
            f"face band {int(span)}px + 20% pad = {int(padded)}px exceeds the "
            f"{crop_w}px 9:16 crop — faces would be cut"
        )
    return "fill", f"single speaker, face band {int(span)}px fits the crop"


def _face_detector(mp_face):
    """Load MediaPipe's face detector ONCE per worker process.

    The old code constructed a FaceDetection graph for every clip, which is a
    model load per clip. That was hidden inside the reframe stage and made a
    twenty-clip batch look like it was re-downloading/reloading a model on every
    boundary. The graph is immutable and used serially under _FACE_LOCK.
    """
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.4,
        )
    return _FACE_DETECTOR


def _center_fallback(
    clip_start: float, clip_end: float, src_width: int, src_height: int,
    target_aspect: float, shot_stats: list[dict] | None, reason: str,
) -> tuple[list[dict], bool]:
    """Safe path: same locked blurred-background FIT layout, centre framed.

    A missing/failed face model must never silently switch a clip to the old
    tight FILL crop. The fallback records FIT and the real reason in the job
    details, so the visual contract is unchanged.
    """
    if shot_stats is not None:
        shot_stats.append({
            "start": clip_start, "end": clip_end,
            "mode": "fit", "reason": f"{reason} — centred FIT fallback",
        })
    return ([_center_window(src_width, src_height, target_aspect, clip_start)], False)


def face_crop_windows(
    video_path: str,
    clip_start: float,
    clip_end: float,
    src_width: int,
    src_height: int,
    target_aspect: float = 9 / 16,
    sample_hz: float = 2.0,
    scene_cuts: list[float] | None = None,
    shot_stats: list[dict] | None = None,
) -> tuple[list[dict], bool]:
    """Sample faces between clip_start..clip_end, per shot.

    The decoder is opened ONCE at clip_start and advanced sequentially. The old
    loop called `cap.set(POS_MSEC, t)` for every 0.5 s sample; on a sectioned
    bucket source each of those seeks re-decoded from the section start. That is
    the measured reframing time sink, and it is why a 21-clip batch spent 1591 s
    in this stage.

    Returns (windows, used_mediapipe). Falls back to the locked FIT layout when
    MediaPipe is unavailable, the per-clip time budget expires, or sampling
    fails. The loop has a hard sample cap and a wall-clock deadline.
    """
    mp = _load_mediapipe()
    if mp is None:
        return _center_fallback(
            clip_start, clip_end, src_width, src_height, target_aspect,
            shot_stats, "MediaPipe unavailable",
        )

    import cv2  # mediapipe solutions require opencv

    # This host has one usable core. OpenCV's own pool is set explicitly so it
    # cannot create one worker per reported core while ffmpeg/Whisper are up.
    try:
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001 - thread hint must never kill analysis
        pass

    mp_face = mp.solutions.face_detection
    crop_w = min(int(src_height * target_aspect), src_width)
    windows: list[dict] = []
    step = 1.0 / max(0.1, float(sample_hz))
    span = max(0.0, float(clip_end) - float(clip_start))
    max_samples = max(1, int(span / step) + 2)
    deadline = time.monotonic() + max(5.0, CLIP_ANALYSIS_BUDGET_SEC)
    # Per-shot face stats feeding the FIT/FILL decision (TARGET A). Faces are
    # counted per SAMPLEd frame; the band spans the widest face span seen.
    # Frames with zero faces contribute nothing — that is what makes a
    # genuine wide shot report "no faces" rather than "one face in one frame".
    current_shot: dict = {"start": clip_start, "faces": [], "band": None}
    shots: list[dict] = []
    timed_out = False

    def _close_shot(end_t: float) -> None:
        if current_shot["faces"]:
            n = len(current_shot["faces"])
            band = current_shot["band"] or (0.0, float(src_width))
            mode, reason = fit_or_fill_for_shot(n, band, src_width, src_height)
        else:
            mode, reason = fit_or_fill_for_shot(0, None, src_width, src_height)
        shots.append({
            "start": current_shot["start"], "end": end_t,
            "mode": mode, "reason": reason,
            "faces": len(current_shot["faces"]),
        })

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return _center_fallback(
            clip_start, clip_end, src_width, src_height, target_aspect,
            shot_stats, "video could not be opened for face analysis",
        )
    try:
        # One seek per clip. The loop below advances with grab(), never set().
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(clip_start)) * 1000.0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_dt = 1.0 / max(1.0, fps)
        current_time = float(clip_start)
        with _FACE_LOCK:
            fd = _face_detector(mp_face)
            for sample_i in range(max_samples):
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                t = float(clip_start) + sample_i * step
                if t >= float(clip_end):
                    break
                # Advance to the sample timestamp sequentially. Each grab is a
                # decoded frame from the SAME open stream, not a new seek.
                while current_time + frame_dt * 0.5 < t:
                    if not cap.grab():
                        break
                    current_time += frame_dt
                ok, frame = cap.read()
                if not ok:
                    break
                current_time += frame_dt
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fd.process(rgb)
                cx = src_width / 2
                n_faces = len(res.detections or [])
                if n_faces:
                    current_shot["faces"].append(n_faces)
                    xs = [
                        (d.location_data.relative_bounding_box.xmin * src_width,
                         (d.location_data.relative_bounding_box.xmin
                          + d.location_data.relative_bounding_box.width) * src_width)
                        for d in res.detections
                    ]
                    bmin = min(x[0] for x in xs)
                    bmax = max(x[1] for x in xs)
                    if current_shot["band"] is None:
                        current_shot["band"] = (bmin, bmax)
                    else:
                        current_shot["band"] = (
                            min(current_shot["band"][0], bmin),
                            max(current_shot["band"][1], bmax),
                        )
                    best = max(
                        res.detections,
                        key=lambda d: d.location_data.relative_bounding_box.width
                        * d.location_data.relative_bounding_box.height,
                    )
                    bb = best.location_data.relative_bounding_box
                    cx = (bb.xmin + bb.width / 2) * src_width
                # TARGET A3: EMA smoothing + dead-zone + capped pan speed.
                # The face centre is noisy frame-to-frame; the crop must not
                # jitter. Dead-zone ~4% of crop width, max pan speed 25%/s of
                # the crop width, and the result is CLAMPED so the crop never
                # lets the primary face touch the frame edge.
                prev = windows[-1].get("x") if windows else None
                target_x = cx - crop_w / 2
                if prev is not None:
                    delta = target_x - prev
                    if abs(delta) < crop_w * 0.04:
                        target_x = float(prev)          # dead zone
                    else:
                        target_x = float(prev) + delta * 0.4   # EMA-like ease
                    max_step = crop_w * 0.25 * (1.0 / sample_hz)
                    target_x = float(prev) + max(-max_step, min(max_step, target_x - prev))
                x = int(max(0, min(src_width - crop_w, target_x)))
                windows.append({"t": t, "x": x, "y": 0, "w": crop_w, "h": src_height})
    except Exception as exc:  # noqa: BLE001 - a failed optional step uses FIT
        logger.warning(f"[reframe] face sampling failed: {type(exc).__name__}: {exc}")
        cap.release()
        return _center_fallback(
            clip_start, clip_end, src_width, src_height, target_aspect,
            shot_stats, f"face sampling failed ({type(exc).__name__})",
        )
    finally:
        cap.release()

    if timed_out:
        return _center_fallback(
            clip_start, clip_end, src_width, src_height, target_aspect,
            shot_stats, f"face analysis exceeded {CLIP_ANALYSIS_BUDGET_SEC:.0f}s",
        )

    # close the trailing shot
    _close_shot(clip_end)

    # merge shots closer than MIN_SWITCH_GAP_SEC so a mode switch never fires
    # within 1s of the previous one (TARGET A4)
    merged: list[dict] = []
    for s in shots:
        if merged and (s["start"] - merged[-1]["end"]) < MIN_SWITCH_GAP_SEC and merged[-1]["mode"] == s["mode"]:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s)
    if shot_stats is not None:
        shot_stats.extend(merged)

    if not windows:
        return ([_center_window(src_width, src_height, target_aspect, clip_start)], True)

    # Flag the first sample at/after each cut (within the clip) as a hard snap.
    for cut in scene_cuts or []:
        if cut <= clip_start or cut > clip_end:
            continue
        # first sample whose time is at or after the cut
        for wdw in windows:
            if wdw["t"] >= cut - 1e-6:
                wdw["cut"] = True
                break
    return (windows, True)


def find_cuts(
    video_path,
    transcript,
    clips,
    settings,
    *,
    on_log=None,
    on_progress=None,
    should_pause=None,
) -> list[dict]:
    """Convert AI clip candidates into per-clip render windows.

    This is the reframe stage's public entry point. It takes the candidate clips
    chosen by detection and produces a deterministic list of render cuts, one per
    clip, with start/end and the crop window for each frame range.

    Delegates to MediaPipe face-tracking when it is available; otherwise every
    clip gets a centered 9:16 window. Either path returns the same shape, so the
    render stage does not branch on ``used_mediapipe``.

    `clips` is the list returned by the detection stage (each item has at least
    ``start`` and ``end``). `transcript` and `settings` are accepted for API
    compatibility with the call site; they are used later when captioning/timing
    needs them, and this function keeps them for forward compatibility.
    """
    if on_log:
        on_log("Reframing clips...")

    src = str(video_path)
    from app import video as _video
    try:
        from app.encode import analysis_sample_hz
    except ImportError:  # defensive: keep reframe standalone
        def analysis_sample_hz(_fps: float, *, hardware: bool = False) -> float:
            return 2.0

    probe = _video.probe(video_path)
    src_width = int(probe["width"])
    src_height = int(probe["height"])
    src_fps = float(probe.get("fps", 30.0) or 30.0)
    target_aspect = 9.0 / 16.0

    cuts: list[dict] = []
    total = len(clips)

    # A bucket source has no single file to analyse: it is either one object or
    # a list of 150 s sections. `url_for_span` resolves a timestamp to the
    # object that holds it plus the offset inside that object, so face sampling
    # reads the RIGHT section for each clip instead of seeking past the end of
    # the first one and silently falling back to a centered crop.
    resolver = getattr(video_path, "url_for_span", None)

    # Scene cuts are shared across the whole source; reframe so each clip can
    # snap at the nearest boundary. With sections there is no whole-source
    # pass to be had (each section is its own container), so this is skipped
    # rather than reading 150 s and pretending it covers 20 minutes.
    scene_cuts = [] if resolver is not None else _video.detect_scene_cuts(video_path)
    if resolver is not None and total > 1:
        if on_log:
            on_log("[framing] sectioned bucket source — shot boundaries detected per clip")

    for i, clip in enumerate(clips):
        if should_pause and should_pause():
            raise PipelinePaused("reframe paused by user")

        start = float(clip["start"])
        end = float(clip["end"])

        # Per-shot framing log (TARGET A): one line per shot naming the mode
        # and the reason, visible in the job log so the operator can spot a
        # wrong FIT/FILL call without guessing.
        shot_stats: list[dict] = []
        _clip_t0 = time.monotonic()
        clip_src = src
        offset = 0.0
        if resolver is not None:
            clip_src, offset = resolver(start)
            clip_src = clip_src or src
        windows, used_mp = face_crop_windows(
            video_path=clip_src,
            # Times are relative to the object being read; they are shifted
            # back to absolute source seconds below, which is what the cut and
            # the render expect.
            clip_start=start - offset,
            clip_end=end - offset,
            src_width=src_width,
            src_height=src_height,
            target_aspect=target_aspect,
            sample_hz=analysis_sample_hz(src_fps, hardware=False),
            scene_cuts=scene_cuts,
            shot_stats=shot_stats,
        )
        if on_log:
            on_log(
                f"[reframe] clip {i + 1}/{total}: {end - start:.1f}s "
                f"analysed in {time.monotonic() - _clip_t0:.2f}s"
            )
        if offset:
            for w in windows:
                w["t"] = float(w.get("t", 0.0)) + offset
            for s in shot_stats:
                s["start"] = float(s.get("start", start - offset)) + offset
                s["end"] = float(s.get("end", end - offset)) + offset
        for s in shot_stats:
            if on_log:
                on_log(
                    f"[framing] {start:.1f}s-{s.get('end', end):.1f}s "
                    f"shot chose {s.get('mode', 'fill').upper()} — {s.get('reason', '')}"
                )

        # First window anchors the cut; remaining windows are the per-frame crop
        # samples the render filter graph interpolates between.
        first = windows[0] if windows else {
            "t": start, "x": 0, "y": 0, "w": src_width, "h": src_height,
        }
        cuts.append({
            "order": i,
            # Carry the candidate's real clip id (minted by store.create_clips)
            # through to the export stage — export writes render state onto the
            # clip document with this id, so a fabricated one crashes.
            "clipId": clip.get("_id"),
            "start": round(start, 3),
            "end": round(end, 3),
            "cropBox": {
                "x": int(first.get("x", 0)),
                "y": int(first.get("y", 0)),
                "w": int(first.get("w", src_width)),
                "h": int(first.get("h", src_height)),
            },
            "aspect": "9:16",
            "sampleWindows": [
                {
                    "t": float(w["t"]),
                    "x": int(w.get("x", 0)),
                    "y": int(w.get("y", 0)),
                    "w": int(w.get("w", src_width)),
                    "h": int(w.get("h", src_height)),
                    "cut": bool(w.get("cut", False)),
                }
                for w in windows
            ],
            "usedMediapipe": used_mp,
            # TARGET A1: per-shot framing mode — read by encode.render_clip.
            # Times are absolute (source seconds) exactly like start/end.
            "framingSegments": [
                {
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "mode": str(s["mode"]),
                    "reason": str(s.get("reason", "")),
                }
                for s in shot_stats
            ],
            # Optional hook title (TARGET optional), filled by the caption
            # stage if detection produced one.
            "hookTitle": clip.get("hook_text") or clip.get("title") or "",
        })

        if on_progress:
            on_progress((i + 1) / max(1, total), f"Reframing clip {i + 1} of {total}")

    if on_log:
        on_log(f"Reframe complete: {len(cuts)} cuts")
    return cuts


class PipelinePaused(Exception):
    """Reframe was asked to stop at a safe checkpoint."""

    pass
