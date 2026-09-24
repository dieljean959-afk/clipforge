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
