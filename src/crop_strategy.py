"""
src/crop_strategy.py
--------------------
Video Cropping Strategy Selector for 9:16 Vertical Shorts.
Supports:
1. Smart Face / Subject Tracking (dynamic or smoothed face center)
2. Fixed Center Crop
3. Manual Position (user-selected horizontal X offset)
"""

import logging
from typing import Dict, Any, Optional

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("CropStrategy")


def compute_916_crop_coordinates(
    video_w: int,
    video_h: int,
    crop_mode: str = "smart",
    manual_x: float = 0.5,
    face_track_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes exact pixel crop dimensions and offsets for a 9:16 vertical short.
    Guarantees even dimensions for FFmpeg and libx264 compatibility.
    """
    # 9:16 target width
    crop_w = int(round(video_h * 9.0 / 16.0 / 2.0) * 2)
    crop_h = video_h

    # If video is already narrower than or equal to 9:16
    if crop_w >= video_w:
        return {
            "crop_mode": "native",
            "crop_w": video_w,
            "crop_h": video_h,
            "crop_x": 0,
            "crop_y": 0,
            "filter": f"crop={video_w}:{video_h}:0:0",
            "scale_filter": "scale=1080:1920"
        }

    max_x = video_w - crop_w
    default_center_x = int((max_x // 2) * 2)

    crop_mode_lower = (crop_mode or "smart").lower()

    if crop_mode_lower == "manual":
        # Manual offset: 0.0 is left edge, 1.0 is right edge
        ratio = max(0.0, min(1.0, float(manual_x)))
        raw_x = int(round(ratio * max_x))
        crop_x = int((raw_x // 2) * 2)
        applied_mode = "manual"

    elif crop_mode_lower == "center":
        crop_x = default_center_x
        applied_mode = "center"

    else:
        # Smart Face Tracking mode
        applied_mode = "smart"
        if face_track_info and face_track_info.get("has_face"):
            detected_x = face_track_info.get("crop_x")
            if detected_x is not None:
                clamped_x = max(0, min(max_x, int(detected_x)))
                crop_x = int((clamped_x // 2) * 2)
            else:
                crop_x = default_center_x
        else:
            # Fall back safely to center crop if no face was detected
            crop_x = default_center_x
            applied_mode = "smart_fallback_center"

    crop_x = max(0, min(max_x, crop_x))

    return {
        "crop_mode": applied_mode,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "crop_x": crop_x,
        "crop_y": 0,
        "filter": f"crop={crop_w}:{crop_h}:{crop_x}:0",
        "scale_filter": "scale=1080:1920"
    }
