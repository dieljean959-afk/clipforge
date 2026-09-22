"""MediaPipe face-tracking reframe (optional dependency).

Produces a time series of 9:16 crop windows that follow the active speaker.
If mediapipe is not installed, reframe falls back to a centered crop and logs
it — no fabricated tracking data.
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


def face_crop_windows(
    video_path: str,
    clip_start: float,
    clip_end: float,
    src_width: int,
    src_height: int,
    target_aspect: float = 9 / 16,
    sample_hz: float = 2.0,
) -> tuple[list[dict], bool]:
    """Sample faces between clip_start..clip_end.

    Returns (windows, used_mediapipe) where windows are
    [{t, x, y, w, h}] in source pixels. Falls back to a single centered
    window when MediaPipe is unavailable.
    """
    mp = _load_mediapipe()
    if mp is None:
        w = int(src_height * target_aspect)
        w = min(w, src_width)
        x = (src_width - w) // 2
        return ([{"t": clip_start, "x": x, "y": 0, "w": w, "h": src_height}], False)

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
        w = min(int(src_height * target_aspect), src_width)
        return ([{"t": crop_w and clip_start, "x": (src_width - w) // 2, "y": 0, "w": w, "h": src_height}], True)
    return (windows, True)
