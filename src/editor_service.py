"""
src/editor_service.py
---------------------
Service for managing the VideoHighlightAI browser-based video editor project state.
Handles project loading, saving, resetting, validation, and safe audio asset uploads.
Guarantees job isolation, path traversal protection, and persistence without rerunning AI inference.
"""

import os
import sys
import json
import uuid
import logging
from typing import Dict, Any, Optional, Tuple
from werkzeug.utils import secure_filename

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("EditorService")

WEB_OUTPUTS_FOLDER = os.path.join(BASE_DIR, "web_outputs")
ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "aac", "m4a", "ogg"}
MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

SUPPORTED_ASPECT_RATIOS = ["9:16", "16:9", "1:1", "4:5"]
SUPPORTED_CROP_MODES = ["smart", "center", "manual"]
SUPPORTED_SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
SUPPORTED_FILTER_PRESETS = ["normal", "vivid", "warm", "cool", "dramatic", "bw"]


def is_safe_id(identifier: Any) -> bool:
    """Validates that job_id or clip_id contains only alphanumeric characters, dashes, and underscores."""
    if identifier is None:
        return False
    s = str(identifier).strip()
    if not s or ".." in s or "/" in s or "\\" in s:
        return False
    return all(c.isalnum() or c in ("_", "-") for c in s)


def get_job_dir(job_id: str) -> str:
    """Returns absolute path to job output directory, creating it if needed."""
    if not is_safe_id(job_id):
        raise ValueError("Invalid job ID")
    job_dir = os.path.join(WEB_OUTPUTS_FOLDER, job_id)
    return job_dir


def get_project_file_path(job_id: str, clip_id: Any) -> str:
    """Returns path to the project JSON file for a specific clip."""
    job_dir = get_job_dir(job_id)
    return os.path.join(job_dir, f"project_{clip_id}.json")


def load_job_segments(job_id: str) -> Dict[str, Any]:
    """Loads segments.json for a job."""
    job_dir = get_job_dir(job_id)
    meta_path = os.path.join(job_dir, "segments.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"segments.json not found for job '{job_id}'")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_reel_data(job_data: Dict[str, Any], clip_id: Any) -> Tuple[int, Dict[str, Any]]:
    """Finds target reel and its index in job data by reel_num or index."""
    reels = job_data.get("reels", [])
    try:
        clip_int = int(clip_id)
    except (ValueError, TypeError):
        clip_int = 1

    for idx, r in enumerate(reels):
        if r.get("reel_num") == clip_int:
            return idx, r

    # Fallback to index if reel_num not explicitly matching
    if 0 <= clip_int - 1 < len(reels):
        return clip_int - 1, reels[clip_int - 1]

    # If single highlight mode
    if not reels:
        default_reel = {
            "reel_num": 1,
            "filename": "highlight.mp4",
            "start_sec": 0.0,
            "end_sec": float(job_data.get("original_duration", 30.0)),
            "duration_sec": float(job_data.get("original_duration", 30.0)),
            "transcript": job_data.get("transcript", ""),
            "speech_segments": job_data.get("speech_segments", [])
        }
        return 0, default_reel

    raise KeyError(f"Reel #{clip_id} not found in job metadata")


def resolve_source_video_path(job_id: str, job_data: Dict[str, Any], reel: Dict[str, Any]) -> str:
    """
    Locates the best source video path on disk.
    Priority:
    1. job_data.get("source_video_path")
    2. job_data.get("youtube_meta", {}).get("video_path")
    3. Uploads directory matching job_id
    4. Existing reel video file in web_outputs/<job_id>
    """
    candidates = []
    
    src1 = job_data.get("source_video_path")
    if src1:
        candidates.append(src1)

    yt_meta = job_data.get("youtube_meta") or {}
    src2 = yt_meta.get("video_path")
    if src2:
        candidates.append(src2)

    # Check uploads directory
    upload_dir = os.path.join(BASE_DIR, "uploads")
    if os.path.isdir(upload_dir):
        for fname in os.listdir(upload_dir):
            if fname.startswith(job_id) and fname.endswith(".mp4"):
                candidates.append(os.path.join(upload_dir, fname))
                break

    # Check reel video file in job directory
    job_dir = get_job_dir(job_id)
    reel_fname = reel.get("filename", f"reel_{int(reel.get('reel_num', 1)):02d}.mp4")
    reel_path = os.path.join(job_dir, reel_fname)
    if os.path.isfile(reel_path):
        candidates.append(reel_path)

    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)

    # If none found, return the most descriptive candidate or fallback to reel_path
    return reel_path


def initialize_default_project(job_id: str, clip_id: Any) -> Dict[str, Any]:
    """
    Initializes a new project state dictionary from the original reel and job metadata.
    Does NOT rerun AI models.
    """
    job_data = load_job_segments(job_id)
    _, reel = find_reel_data(job_data, clip_id)
    source_video = resolve_source_video_path(job_id, job_data, reel)

    # Duration calculations
    source_dur = float(job_data.get("original_duration", 0.0))
    start_sec = max(0.0, float(reel.get("start_sec", 0.0)))
    end_sec = float(reel.get("end_sec", start_sec + 30.0))
    if end_sec <= start_sec:
        end_sec = start_sec + 30.0
    if source_dur <= 0:
        source_dur = max(end_sec, 60.0)

    # Subtitles
    raw_segments = reel.get("speech_segments") or []
    if not raw_segments:
        # Check if full job has speech segments that overlap
        full_speech = job_data.get("speech_segments") or []
        for s in full_speech:
            if s.get("end", 0.0) >= start_sec and s.get("start", 0.0) <= end_sec:
                raw_segments.append(s)

    clean_subtitles = []
    for s in raw_segments:
        clean_subtitles.append({
            "id": f"sub_{uuid.uuid4().hex[:6]}",
            "start": round(float(s.get("start", 0.0)), 2),
            "end": round(float(s.get("end", 0.0)), 2),
            "text": str(s.get("text", "")).strip()
        })

    # If no timed segments but transcript exists, provide fallback segment
    if not clean_subtitles and reel.get("transcript"):
        clean_subtitles.append({
            "id": f"sub_{uuid.uuid4().hex[:6]}",
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "text": str(reel.get("transcript")).strip()
        })

    aspect_ratio = reel.get("aspect_ratio") or job_data.get("aspect_ratio") or "9:16"
    if aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
        aspect_ratio = "9:16"

    crop_mode = str(reel.get("crop_mode") or "smart").lower()
    if crop_mode not in SUPPORTED_CROP_MODES:
        crop_mode = "smart"

    manual_x = float(reel.get("manual_x", 0.5))

    captions_enabled = reel.get("captions", True)
    if isinstance(captions_enabled, str):
        captions_enabled = captions_enabled.lower() in ("true", "1", "on")

    sub_style = str(reel.get("subtitle_style") or "bold").lower()
    sub_pos = str(reel.get("subtitle_position") or "bottom").lower()

    emojis_enabled = bool(reel.get("emojis", False))

    project = {
        "job_id": job_id,
        "clip_id": int(reel.get("reel_num", clip_id)),
        "reel_filename": reel.get("filename", f"reel_{int(clip_id):02d}.mp4"),
        "source_video_path": source_video,
        "source_duration": round(source_dur, 2),
        "original_clip": {
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "duration_sec": round(end_sec - start_sec, 2),
            "crop_mode": crop_mode,
            "manual_x": manual_x,
            "captions": captions_enabled,
            "emojis": emojis_enabled
        },
        "trim": {
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "duration": round(end_sec - start_sec, 2)
        },
        "crop": {
            "mode": crop_mode,
            "aspect_ratio": aspect_ratio,
            "manual_x": max(0.0, min(1.0, manual_x))
        },
        "subtitles": {
            "enabled": captions_enabled,
            "style": sub_style if sub_style in ["bold", "clean", "minimal"] else "bold",
            "position": sub_pos if sub_pos in ["bottom", "center", "top"] else "bottom",
            "font_family": "Arial",
            "font_size": 48,
            "color": "#FFFFFF",
            "outline_color": "#000000",
            "bg_color": "rgba(0,0,0,0.5)",
            "bold": True,
            "emojis_enabled": emojis_enabled,
            "segments": clean_subtitles
        },
        "overlays": [],
        "audio": {
            "original_enabled": True,
            "original_volume": 1.0,
            "music_filename": None,
            "music_volume": 0.3,
            "fade_in": 0.0,
            "fade_out": 0.0
        },
        "speed": 1.0,
        "adjustments": {
            "brightness": 0.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "filter_preset": "normal"
        },
        "exported_version": None,
        "updated_at": None
    }

    # Check if an edited version already exists on disk
    job_dir = get_job_dir(job_id)
    edited_fname = f"reel_{int(project['clip_id']):02d}_edited.mp4"
    if os.path.isfile(os.path.join(job_dir, edited_fname)):
        size_mb = round(os.path.getsize(os.path.join(job_dir, edited_fname)) / (1024 * 1024), 2)
        project["exported_version"] = {
            "filename": edited_fname,
            "file_size_mb": size_mb,
            "ready": True
        }

    return project


def get_or_create_project(job_id: str, clip_id: Any) -> Dict[str, Any]:
    """Loads existing project JSON from disk or initializes a new one."""
    if not is_safe_id(job_id) or not is_safe_id(clip_id):
        raise ValueError("Invalid job or clip ID")

    proj_file = get_project_file_path(job_id, clip_id)
    if os.path.isfile(proj_file):
        try:
            with open(proj_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Recheck if edited version file exists
            job_dir = get_job_dir(job_id)
            edited_fname = f"reel_{int(saved.get('clip_id', clip_id)):02d}_edited.mp4"
            if os.path.isfile(os.path.join(job_dir, edited_fname)):
                size_mb = round(os.path.getsize(os.path.join(job_dir, edited_fname)) / (1024 * 1024), 2)
                saved["exported_version"] = {
                    "filename": edited_fname,
                    "file_size_mb": size_mb,
                    "ready": True
                }
            return saved
        except Exception as e:
            logger.warning(f"Could not read existing project JSON ({e}). Re-initializing...")

    project = initialize_default_project(job_id, clip_id)
    save_project(job_id, clip_id, project)
    return project


def validate_project_payload(payload: Dict[str, Any], max_duration: float = 3600.0) -> Dict[str, Any]:
    """
    Validates and sanitizes editor state fields to prevent malformed values and injection.
    """
    cleaned: Dict[str, Any] = {}

    # Trim validation
    trim = payload.get("trim", {})
    try:
        start = max(0.0, float(trim.get("start", 0.0)))
    except (ValueError, TypeError):
        start = 0.0

    try:
        end = float(trim.get("end", start + 15.0))
    except (ValueError, TypeError):
        end = start + 15.0

    if end <= start:
        end = start + 5.0
    if max_duration > 0 and end > max_duration:
        end = min(max_duration, end)
        if end <= start:
            start = max(0.0, end - 5.0)

    dur = max(0.5, end - start)
    cleaned["trim"] = {
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(dur, 2)
    }

    # Crop & Aspect Ratio validation
    crop = payload.get("crop", {})
    crop_mode = str(crop.get("mode", "smart")).lower()
    if crop_mode not in SUPPORTED_CROP_MODES:
        crop_mode = "smart"

    aspect = str(crop.get("aspect_ratio", "9:16"))
    if aspect not in SUPPORTED_ASPECT_RATIOS:
        aspect = "9:16"

    try:
        manual_x = max(0.0, min(1.0, float(crop.get("manual_x", 0.5))))
    except (ValueError, TypeError):
        manual_x = 0.5

    cleaned["crop"] = {
        "mode": crop_mode,
        "aspect_ratio": aspect,
        "manual_x": round(manual_x, 3)
    }

    # Subtitles validation
    subs = payload.get("subtitles", {})
    sub_enabled = bool(subs.get("enabled", True))
    sub_style = str(subs.get("style", "bold")).lower()
    if sub_style not in ["bold", "clean", "minimal"]:
        sub_style = "bold"

    sub_pos = str(subs.get("position", "bottom")).lower()
    if sub_pos not in ["bottom", "center", "top"]:
        sub_pos = "bottom"

    font_fam = str(subs.get("font_family", "Arial")).strip()[:40] or "Arial"
    try:
        font_sz = max(16, min(96, int(subs.get("font_size", 48))))
    except (ValueError, TypeError):
        font_sz = 48

    color = str(subs.get("color", "#FFFFFF")).strip()[:10]
    out_color = str(subs.get("outline_color", "#000000")).strip()[:10]
    bg_color = str(subs.get("bg_color", "rgba(0,0,0,0.5)")).strip()[:25]
    bold = bool(subs.get("bold", True))
    emojis_enabled = bool(subs.get("emojis_enabled", False))

    clean_segs = []
    for s in subs.get("segments", []):
        try:
            s_start = max(0.0, float(s.get("start", 0.0)))
            s_end = max(s_start + 0.1, float(s.get("end", s_start + 1.0)))
            text = str(s.get("text", "")).strip()[:300]
            if text:
                clean_segs.append({
                    "id": str(s.get("id") or f"sub_{uuid.uuid4().hex[:6]}"),
                    "start": round(s_start, 2),
                    "end": round(s_end, 2),
                    "text": text
                })
        except Exception:
            continue

    cleaned["subtitles"] = {
        "enabled": sub_enabled,
        "style": sub_style,
        "position": sub_pos,
        "font_family": font_fam,
        "font_size": font_sz,
        "color": color,
        "outline_color": out_color,
        "bg_color": bg_color,
        "bold": bold,
        "emojis_enabled": emojis_enabled,
        "segments": clean_segs
    }

    # Overlays validation (text & emoji overlays)
    raw_overlays = payload.get("overlays", [])
    clean_overlays = []
    for ov in raw_overlays:
        try:
            ov_id = str(ov.get("id") or f"ov_{uuid.uuid4().hex[:6]}")
            ov_type = "emoji" if ov.get("type") == "emoji" else "text"
            ov_content = str(ov.get("content", "")).strip()[:100]
            if not ov_content:
                continue
            ov_fs = max(14, min(120, int(ov.get("font_size", 36))))
            ov_color = str(ov.get("color", "#FFD700")).strip()[:10]
            ov_x = max(0.0, min(100.0, float(ov.get("x_pct", 50.0))))
            ov_y = max(0.0, min(100.0, float(ov.get("y_pct", 20.0))))
            ov_start = max(0.0, float(ov.get("start_sec", start)))
            ov_end = max(ov_start + 0.2, float(ov.get("end_sec", end)))
            clean_overlays.append({
                "id": ov_id,
                "type": ov_type,
                "content": ov_content,
                "font_size": ov_fs,
                "color": ov_color,
                "x_pct": round(ov_x, 1),
                "y_pct": round(ov_y, 1),
                "start_sec": round(ov_start, 2),
                "end_sec": round(ov_end, 2)
            })
        except Exception:
            continue
    cleaned["overlays"] = clean_overlays

    # Audio validation
    audio = payload.get("audio", {})
    orig_enabled = bool(audio.get("original_enabled", True))
    try:
        orig_vol = max(0.0, min(2.0, float(audio.get("original_volume", 1.0))))
    except (ValueError, TypeError):
        orig_vol = 1.0

    music_fname = audio.get("music_filename")
    if music_fname:
        music_fname = secure_filename(str(music_fname))
        if not (music_fname.endswith(tuple(f".{ext}" for ext in ALLOWED_AUDIO_EXTENSIONS))):
            music_fname = None

    try:
        music_vol = max(0.0, min(1.0, float(audio.get("music_volume", 0.3))))
    except (ValueError, TypeError):
        music_vol = 0.3

    try:
        fade_in = max(0.0, min(5.0, float(audio.get("fade_in", 0.0))))
    except (ValueError, TypeError):
        fade_in = 0.0

    try:
        fade_out = max(0.0, min(5.0, float(audio.get("fade_out", 0.0))))
    except (ValueError, TypeError):
        fade_out = 0.0

    cleaned["audio"] = {
        "original_enabled": orig_enabled,
        "original_volume": round(orig_vol, 2),
        "music_filename": music_fname,
        "music_volume": round(music_vol, 2),
        "fade_in": round(fade_in, 2),
        "fade_out": round(fade_out, 2)
    }

    # Speed validation
    try:
        speed = float(payload.get("speed", 1.0))
        if speed not in SUPPORTED_SPEEDS:
            # Snap to closest supported
            speed = min(SUPPORTED_SPEEDS, key=lambda x: abs(x - speed))
    except (ValueError, TypeError):
        speed = 1.0
    cleaned["speed"] = speed

    # Adjustments validation
    adj = payload.get("adjustments", {})
    try:
        brightness = max(-1.0, min(1.0, float(adj.get("brightness", 0.0))))
    except (ValueError, TypeError):
        brightness = 0.0

    try:
        contrast = max(0.0, min(2.0, float(adj.get("contrast", 1.0))))
    except (ValueError, TypeError):
        contrast = 1.0

    try:
        saturation = max(0.0, min(2.0, float(adj.get("saturation", 1.0))))
    except (ValueError, TypeError):
        saturation = 1.0

    preset = str(adj.get("filter_preset", "normal")).lower()
    if preset not in SUPPORTED_FILTER_PRESETS:
        preset = "normal"

    cleaned["adjustments"] = {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "saturation": round(saturation, 2),
        "filter_preset": preset
    }

    return cleaned


def save_project(job_id: str, clip_id: Any, project_data: Dict[str, Any]) -> Dict[str, Any]:
    """Persists project state to web_outputs/<job_id>/project_<clip_id>.json."""
    if not is_safe_id(job_id) or not is_safe_id(clip_id):
        raise ValueError("Invalid job or clip ID")

    job_dir = get_job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    proj_file = get_project_file_path(job_id, clip_id)

    # Preserve essential metadata
    project_data["job_id"] = job_id
    project_data["clip_id"] = int(clip_id)

    import datetime
    project_data["updated_at"] = datetime.datetime.now().isoformat()

    with open(proj_file, "w", encoding="utf-8") as f:
        json.dump(project_data, f, indent=2, ensure_ascii=False)

    return project_data


def reset_project(job_id: str, clip_id: Any) -> Dict[str, Any]:
    """Resets project edits back to original reel configuration."""
    if not is_safe_id(job_id) or not is_safe_id(clip_id):
        raise ValueError("Invalid job or clip ID")

    proj_file = get_project_file_path(job_id, clip_id)
    if os.path.isfile(proj_file):
        try:
            os.remove(proj_file)
        except OSError:
            pass

    return initialize_default_project(job_id, clip_id)


def save_uploaded_audio(job_id: str, file_storage) -> str:
    """
    Saves an uploaded background music file securely into the job directory.
    Returns the saved filename.
    """
    if not is_safe_id(job_id):
        raise ValueError("Invalid job ID")

    if not file_storage or not file_storage.filename:
        raise ValueError("No audio file provided")

    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio format. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}")

    job_dir = get_job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)

    safe_name = f"bgm_{uuid.uuid4().hex[:8]}.{ext}"
    out_path = os.path.join(job_dir, safe_name)
    file_storage.save(out_path)

    # Validate file size
    if os.path.getsize(out_path) > MAX_AUDIO_SIZE_BYTES:
        os.remove(out_path)
        raise ValueError("Audio file exceeds maximum allowed size (50 MB)")

    return safe_name
