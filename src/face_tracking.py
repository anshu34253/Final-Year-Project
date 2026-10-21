"""
src/face_tracking.py
--------------------
Face Detection & Dynamic Subject Tracking for Smart 9:16 Cropping.
Integrates UniFace (SCRFD 500M) and ByteTrack (BYTETracker) to track
the dominant speaker/subject and calculate a smooth, temporally stable
9:16 vertical crop window.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# Ensure Face Detection directory is on sys.path for bytetrack
_FACE_DETECTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Face Detection"))
if os.path.isdir(_FACE_DETECTION_DIR) and _FACE_DETECTION_DIR not in sys.path:
    sys.path.insert(0, _FACE_DETECTION_DIR)

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("FaceTracking")

_DETECTOR_INSTANCE = None


def get_face_detector():
    """Lazily loads and caches the UniFace SCRFD detector."""
    global _DETECTOR_INSTANCE
    if _DETECTOR_INSTANCE is None:
        try:
            from uniface import SCRFD
            from uniface.constants import SCRFDWeights
            logger.info("Initializing UniFace SCRFD detector (SCRFD_500M_KPS)...")
            _DETECTOR_INSTANCE = SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS)
        except Exception as e:
            logger.warning(f"Could not load UniFace SCRFD ({e}). Falling back to None.")
            _DETECTOR_INSTANCE = None
    return _DETECTOR_INSTANCE


class FaceSubjectTracker:
    """
    Tracks faces across video frames using UniFace and ByteTrack,
    calculating a temporally smooth horizontal crop center for 9:16 vertical framing.
    """

    def __init__(
        self,
        conf_thresh: float = 0.45,
        track_thresh: float = 0.4,
        track_buffer: int = 30,
        match_thresh: float = 0.7,
        smoothing_alpha: float = 0.18
    ):
        self.conf_thresh = conf_thresh
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.smoothing_alpha = smoothing_alpha

    def analyze_clip(
        self,
        video_path: str,
        start_sec: float = 0.0,
        duration_sec: Optional[float] = None,
        sample_fps: float = 6.0,
        max_detect_dim: int = 640
    ) -> Dict[str, Any]:
        """
        Samples video frames between start_sec and start_sec + duration_sec,
        runs face detection and ByteTrack tracking, and computes smoothed crop offsets.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_dur = total_frames / source_fps if source_fps > 0 else 0.0

        if duration_sec is None or duration_sec <= 0:
            duration_sec = max(1.0, video_dur - start_sec)
        end_sec = min(video_dur, start_sec + duration_sec)

        # 9:16 crop dimensions
        target_crop_w = int(round(video_h * 9.0 / 16.0 / 2.0) * 2)
        target_crop_h = video_h

        # Default center crop (guaranteed even integer for codec alignment)
        default_center_x = int((max(0, (video_w - target_crop_w) // 2) // 2) * 2)

        # Check if already 9:16 or narrower (no horizontal crop possible)
        if target_crop_w >= video_w:
            cap.release()
            return {
                "has_face": False,
                "dominant_track_id": None,
                "crop_x": 0,
                "crop_y": 0,
                "crop_w": video_w,
                "crop_h": video_h,
                "primary_x_center": video_w / 2.0,
                "confidence_score": 1.0,
                "frame_samples": []
            }

        detector = get_face_detector()
        if detector is None:
            cap.release()
            return {
                "has_face": False,
                "dominant_track_id": None,
                "crop_x": default_center_x,
                "crop_y": 0,
                "crop_w": target_crop_w,
                "crop_h": target_crop_h,
                "primary_x_center": video_w / 2.0,
                "confidence_score": 0.0,
                "frame_samples": []
            }

        # Initialize ByteTracker
        try:
            from bytetrack import BYTETracker
            tracker = BYTETracker(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                low_thresh=0.1
            )
        except Exception as err:
            logger.warning(f"Could not initialize BYTETracker ({err}). Continuing without tracker.")
            tracker = None

        frame_step = max(1, int(round(source_fps / sample_fps)))
        start_frame = int(start_sec * source_fps)
        end_frame = int(end_sec * source_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame_idx = start_frame

        sampled_centers = []
        track_stats = {}  # track_id -> {"count": int, "area_sum": float, "last_center": float}

        while cap.isOpened() and current_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if (current_frame_idx - start_frame) % frame_step == 0:
                frame_time = current_frame_idx / source_fps

                # Downscale for ultra-fast face detection if frame is large
                h, w = frame.shape[:2]
                scale_factor = 1.0
                if max(h, w) > max_detect_dim:
                    scale_factor = max_detect_dim / float(max(h, w))
                    small_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
                else:
                    small_frame = frame

                # UniFace detection
                faces = detector.detect(small_frame)
                dets = []
                for face in faces:
                    if face.confidence >= self.conf_thresh:
                        # Scale bbox back to original video dimensions
                        x1, y1, x2, y2 = face.bbox
                        if scale_factor != 1.0:
                            x1 /= scale_factor
                            y1 /= scale_factor
                            x2 /= scale_factor
                            y2 /= scale_factor
                        dets.append([x1, y1, x2, y2, face.confidence])

                dets_np = np.array(dets, dtype=np.float32) if len(dets) > 0 else np.empty((0, 5), dtype=np.float32)

                frame_center = None
                best_track_id = None

                if tracker is not None and len(dets_np) > 0:
                    tracks = tracker.update(dets_np)
                    if len(tracks) > 0:
                        # Find track with largest bounding box area
                        best_area = -1.0
                        for t_box in tracks:
                            tx1, ty1, tx2, ty2, tid = t_box
                            area = (tx2 - tx1) * (ty2 - ty1)
                            tid_int = int(tid)
                            if tid_int not in track_stats:
                                track_stats[tid_int] = {"count": 0, "area_sum": 0.0, "last_center": (tx1 + tx2) / 2.0}
                            track_stats[tid_int]["count"] += 1
                            track_stats[tid_int]["area_sum"] += area
                            track_stats[tid_int]["last_center"] = (tx1 + tx2) / 2.0

                            if area > best_area:
                                best_area = area
                                frame_center = (tx1 + tx2) / 2.0
                                best_track_id = tid_int
                elif len(dets_np) > 0:
                    # Tracker unavailable, use largest detected face
                    areas = (dets_np[:, 2] - dets_np[:, 0]) * (dets_np[:, 3] - dets_np[:, 1])
                    best_idx = np.argmax(areas)
                    frame_center = (dets_np[best_idx, 0] + dets_np[best_idx, 2]) / 2.0

                sampled_centers.append({
                    "time": round(frame_time, 3),
                    "detected_x": frame_center,
                    "track_id": best_track_id
                })

            current_frame_idx += 1

        cap.release()

        # Determine dominant track ID across the clip
        dominant_track_id = None
        if track_stats:
            dominant_track_id = max(track_stats.keys(), key=lambda tid: track_stats[tid]["count"] * 1000 + track_stats[tid]["area_sum"])

        # Temporal smoothing with EMA (Exponential Moving Average)
        smoothed_positions = []
        valid_centers = [s["detected_x"] for s in sampled_centers if s["detected_x"] is not None]

        if not valid_centers:
            # No faces detected in clip -> fall back cleanly to center crop
            fallback_samples = [
                {"time": s["time"], "smoothed_x": video_w / 2.0, "crop_x": default_center_x}
                for s in sampled_centers
            ]
            return {
                "has_face": False,
                "dominant_track_id": None,
                "crop_x": default_center_x,
                "crop_y": 0,
                "crop_w": target_crop_w,
                "crop_h": target_crop_h,
                "primary_x_center": video_w / 2.0,
                "confidence_score": 0.0,
                "frame_samples": fallback_samples
            }

        # Initialize smoothing at the median of initial valid detections
        current_smoothed = np.median(valid_centers)
        last_valid_x = current_smoothed

        for sample in sampled_centers:
            raw_x = sample["detected_x"]
            if raw_x is not None:
                current_smoothed = (self.smoothing_alpha * raw_x) + ((1.0 - self.smoothing_alpha) * current_smoothed)
                last_valid_x = raw_x
            else:
                # Face temporarily disappeared: hold position with slow drift towards center
                drift_target = (video_w / 2.0 + last_valid_x) / 2.0
                current_smoothed = (0.05 * drift_target) + (0.95 * current_smoothed)

            # Clamp smoothed center
            crop_x_clamped = max(0, min(video_w - target_crop_w, int(round(current_smoothed - (target_crop_w / 2.0)))))
            # Ensure crop_x is even for libx264 compatibility
            crop_x_even = int((crop_x_clamped // 2) * 2)

            smoothed_positions.append({
                "time": sample["time"],
                "smoothed_x": round(current_smoothed, 2),
                "crop_x": crop_x_even
            })

        # Calculate primary stable crop_x across the clip (median of smoothed crop_x)
        all_crop_x = [p["crop_x"] for p in smoothed_positions]
        primary_crop_x = int(np.median(all_crop_x)) if all_crop_x else default_center_x
        primary_crop_x = int((primary_crop_x // 2) * 2)

        return {
            "has_face": True,
            "dominant_track_id": dominant_track_id,
            "crop_x": primary_crop_x,
            "crop_y": 0,
            "crop_w": target_crop_w,
            "crop_h": target_crop_h,
            "primary_x_center": float(primary_crop_x + target_crop_w / 2.0),
            "confidence_score": round(len(valid_centers) / max(1, len(sampled_centers)), 2),
            "frame_samples": smoothed_positions
        }


def get_cached_or_compute_face_track(
    video_path: str,
    start_sec: float,
    duration_sec: float,
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns face tracking result from cache if available, otherwise computes and caches it.
    """
    cache_path = None
    if cache_dir and os.path.isdir(cache_dir):
        cache_key = f"face_tracks_{int(start_sec)}_{int(duration_sec)}.json"
        cache_path = os.path.join(cache_dir, cache_key)
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    tracker = FaceSubjectTracker()
    track_result = tracker.analyze_clip(video_path, start_sec=start_sec, duration_sec=duration_sec)

    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(track_result, f, indent=2)
        except Exception:
            pass

    return track_result
