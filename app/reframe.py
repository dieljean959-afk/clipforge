"""MediaPipe face-tracking reframe (optional dependency).

Produces a time series of 9:16 crop windows that follow the active speaker.
Shot-aware: given the source's detected cut boundaries, each shot is framed
independently and the sample at a cut is flagged so the crop snaps (no
interpolated pan through a cut). If mediapipe is not installed, reframe falls
back to a centered crop and logs it — no fabricated tracking data.
"""

from __future__ import annotations

from loguru import logger

_MP_AVAILABLE: bool | None = None


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


def face_crop_windows(
    video_path: str,
    clip_start: float,
    clip_end: float,
    src_width: int,
    src_height: int,
    target_aspect: float = 9 / 16,
    sample_hz: float = 2.0,
    scene_cuts: list[float] | None = None,
) -> tuple[list[dict], bool]:
    """Sample faces between clip_start..clip_end, per shot.

    `scene_cuts` are absolute timestamps of shot boundaries (from
    video.detect_scene_cuts). The sample nearest each boundary inside the clip
    is marked ``"cut": True`` so video.crop_pan_filter hard-snaps the crop at
    the cut instead of interpolating across it.

    Returns (windows, used_mediapipe). Falls back to a single centered window
    when MediaPipe is unavailable or sampling fails.
    """
    mp = _load_mediapipe()
    if mp is None:
        return ([_center_window(src_width, src_height, target_aspect, clip_start)], False)

    import cv2  # mediapipe solutions require opencv

    mp_face = mp.solutions.face_detection
    crop_w = min(int(src_height * target_aspect), src_width)
    windows: list[dict] = []
    step = 1.0 / sample_hz

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:
        cap = cv2.VideoCapture(video_path)
        t = clip_start
        try:
            while t < clip_end:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fd.process(rgb)
                cx = src_width / 2
                if res.detections:
                    best = max(
                        res.detections,
                        key=lambda d: d.location_data.relative_bounding_box.width
                        * d.location_data.relative_bounding_box.height,
                    )
                    bb = best.location_data.relative_bounding_box
                    cx = (bb.xmin + bb.width / 2) * src_width
                x = int(max(0, min(src_width - crop_w, cx - crop_w / 2)))
                windows.append({"t": t, "x": x, "y": 0, "w": crop_w, "h": src_height})
                t += step
        finally:
            cap.release()

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

    probe = _video.probe(video_path)
    src_width = int(probe["width"])
    src_height = int(probe["height"])
    target_aspect = 9.0 / 16.0

    cuts: list[dict] = []
    total = len(clips)

    # Scene cuts are shared across the whole source; reframe so each clip can
    # snap at the nearest boundary.
    scene_cuts = _video.detect_scene_cuts(video_path)

    for i, clip in enumerate(clips):
        if should_pause and should_pause():
            raise PipelinePaused("reframe paused by user")

        start = float(clip["start"])
        end = float(clip["end"])

        windows, used_mp = face_crop_windows(
            video_path=src,
            clip_start=start,
            clip_end=end,
            src_width=src_width,
            src_height=src_height,
            target_aspect=target_aspect,
            sample_hz=2.0,
            scene_cuts=scene_cuts,
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
        })

        if on_progress:
            on_progress((i + 1) / max(1, total), f"Reframing clip {i + 1} of {total}")

    if on_log:
        on_log(f"Reframe complete: {len(cuts)} cuts")
    return cuts


class PipelinePaused(Exception):
    """Reframe was asked to stop at a safe checkpoint."""

    pass
