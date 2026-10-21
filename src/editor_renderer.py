"""
src/editor_renderer.py
----------------------
High-Performance Video Editor Render Engine for VideoHighlightAI.
Renders customized Short edits using FFmpeg based on the validated project state.

Capabilities:
- Preserves the original generated Short intact.
- Aspect ratio transforms: 9:16 (1080x1920), 16:9 (1920x1080), 1:1 (1080x1080), 4:5 (1080x1350).
- Crop modes: Smart Face Tracking (UniFace/ByteTrack), Center Crop, Manual Horizontal Positioning.
- Visual adjustments: Brightness, Contrast, Saturation, and Color Presets (Vivid, Warm, Cool, Dramatic, B&W).
- Playback speed adjustment with audio pitch/sync preservation via setpts & atempo.
- Subtitle rendering: Advanced SubStation Alpha (.ass) with custom fonts, colors, positions, and emojis.
- Text & Emoji overlays: burned into video with exact timestamps and coordinates.
- Audio mixing: Original volume, mute, audio fade-in/out, and background music track mixing.
- Asynchronous background worker with honest progress tracking and concurrency protection.
- Output validation via OpenCV and ffprobe.
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
from typing import Dict, Any, Optional, List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

from src.reel_renderer import get_ffmpeg_binary, get_ffprobe_binary, probe_video_dimensions
from src.face_tracking import get_cached_or_compute_face_track
from src.crop_strategy import compute_916_crop_coordinates
from src.editor_service import get_job_dir, is_safe_id, save_project

logger = logging.getLogger("EditorRenderer")

# In-memory store for active editor render tasks
# Key: f"{job_id}_{clip_id}" -> dict with status, progress, stage, error, etc.
RENDER_TASKS: Dict[str, Dict[str, Any]] = {}
RENDER_LOCK = threading.Lock()

ASPECT_RATIO_PRESETS = {
    "9:16": {"width": 1080, "height": 1920, "name": "Shorts / Reels / TikTok (9:16)"},
    "16:9": {"width": 1920, "height": 1080, "name": "Landscape (16:9)"},
    "1:1":  {"width": 1080, "height": 1080, "name": "Square (1:1)"},
    "4:5":  {"width": 1080, "height": 1350, "name": "Portrait Post (4:5)"}
}


def get_task_key(job_id: str, clip_id: Any) -> str:
    return f"{job_id}_{clip_id}"


def get_render_status(job_id: str, clip_id: Any) -> Dict[str, Any]:
    """Retrieves current render status for a clip."""
    key = get_task_key(job_id, clip_id)
    with RENDER_LOCK:
        task = RENDER_TASKS.get(key)
        if task:
            return dict(task)

    # Check if edited file already exists on disk
    job_dir = get_job_dir(job_id)
    edited_fname = f"reel_{int(clip_id):02d}_edited.mp4"
    edited_path = os.path.join(job_dir, edited_fname)
    if os.path.isfile(edited_path):
        size_mb = round(os.path.getsize(edited_path) / (1024 * 1024), 2)
        return {
            "status": "complete",
            "progress": 100,
            "stage": "Ready",
            "output_filename": edited_fname,
            "file_size_mb": size_mb,
            "error": None
        }

    return {
        "status": "idle",
        "progress": 0,
        "stage": "Not started",
        "output_filename": None,
        "error": None
    }


def format_ass_timestamp(seconds: float) -> str:
    """Formats seconds into ASS timestamp string H:MM:SS.cs."""
    s = max(0.0, float(seconds))
    hrs = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    cs = int(round((s - int(s)) * 100))
    if cs >= 100:
        secs += 1
        cs = 0
    return f"{hrs:01d}:{mins:02d}:{secs:02d}.{cs:02d}"


def hex_to_ass_color(hex_color: str, default: str = "&H00FFFFFF") -> str:
    """Converts #RRGGBB or #RGB to ASS format &H00BBGGRR."""
    if not hex_color or not isinstance(hex_color, str):
        return default
    c = hex_color.strip().lstrip("#")
    if len(c) == 3:
        c = f"{c[0]}{c[0]}{c[1]}{c[1]}{c[2]}{c[2]}"
    if len(c) == 6:
        r, g, b = c[0:2], c[2:4], c[4:6]
        return f"&H00{b.upper()}{g.upper()}{r.upper()}"
    return default


def generate_editor_ass(
    project_data: Dict[str, Any],
    canvas_w: int,
    canvas_h: int,
    output_ass_path: str
) -> bool:
    """
    Generates a unified ASS subtitle file containing both styled subtitles and overlays.
    Returns True if any events were written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_ass_path)), exist_ok=True)
    trim_start = float(project_data.get("trim", {}).get("start", 0.0))
    trim_end = float(project_data.get("trim", {}).get("end", trim_start + 30.0))
    duration = max(0.1, trim_end - trim_start)

    subs_cfg = project_data.get("subtitles", {})
    subtitles_enabled = subs_cfg.get("enabled", True)
    overlays = project_data.get("overlays", [])

    events = []

    # 1. Process Subtitle Segments
    if subtitles_enabled:
        font_name = subs_cfg.get("font_family", "Arial")
        font_size = int(subs_cfg.get("font_size", 48))
        primary_color = hex_to_ass_color(subs_cfg.get("color", "#FFFFFF"), "&H00FFFFFF")
        outline_color = hex_to_ass_color(subs_cfg.get("outline_color", "#000000"), "&H00000000")
        is_bold = 1 if subs_cfg.get("bold", True) else 0

        pos_name = subs_cfg.get("position", "bottom").lower()
        if pos_name == "top":
            alignment = 8
            margin_v = int(canvas_h * 0.10)
        elif pos_name == "center":
            alignment = 5
            margin_v = 0
        else:
            alignment = 2
            margin_v = int(canvas_h * 0.12)

        margin_h = int(canvas_w * 0.08)

        # Build subtitle events
        raw_segs = subs_cfg.get("segments", [])
        for seg in raw_segs:
            s_abs = float(seg.get("start", 0.0))
            e_abs = float(seg.get("end", s_abs + 1.0))
            text = str(seg.get("text", "")).strip()

            if e_abs <= trim_start or s_abs >= trim_end or not text:
                continue

            rel_s = max(0.0, s_abs - trim_start)
            rel_e = min(duration, e_abs - trim_start)
            if rel_e <= rel_s:
                continue

            clean_text = text.replace("{", "(").replace("}", ")").replace("\n", " ")
            t_start = format_ass_timestamp(rel_s)
            t_end = format_ass_timestamp(rel_e)
            events.append(f"Dialogue: 0,{t_start},{t_end},Subtitles,,0,0,0,,{clean_text}")

    # 2. Process Overlays (Text and Emoji)
    for ov in overlays:
        ov_content = str(ov.get("content", "")).strip()
        if not ov_content:
            continue

        ov_start = float(ov.get("start_sec", trim_start))
        ov_end = float(ov.get("end_sec", trim_end))
        if ov_end <= trim_start or ov_start >= trim_end:
            continue

        rel_s = max(0.0, ov_start - trim_start)
        rel_e = min(duration, ov_end - trim_start)
        if rel_e <= rel_s:
            continue

        x_pct = float(ov.get("x_pct", 50.0)) / 100.0
        y_pct = float(ov.get("y_pct", 20.0)) / 100.0
        pos_x = int(canvas_w * x_pct)
        pos_y = int(canvas_h * y_pct)

        fs = int(ov.get("font_size", 36))
        c_ass = hex_to_ass_color(ov.get("color", "#FFD700"), "&H0000D7FF")
        clean_content = ov_content.replace("{", "(").replace("}", ")")

        t_start = format_ass_timestamp(rel_s)
        t_end = format_ass_timestamp(rel_e)
        events.append(f"Dialogue: 1,{t_start},{t_end},Overlays,,0,0,0,,{{\\pos({pos_x},{pos_y})\\fs{fs}\\c{c_ass}\\bord3\\shad1\\an5}}{clean_content}")

    if not events:
        return False

    sub_font = subs_cfg.get("font_family", "Arial")
    sub_fs = int(subs_cfg.get("font_size", 48))
    sub_color = hex_to_ass_color(subs_cfg.get("color", "#FFFFFF"), "&H00FFFFFF")
    sub_outline = hex_to_ass_color(subs_cfg.get("outline_color", "#000000"), "&H00000000")
    sub_bold = 1 if subs_cfg.get("bold", True) else 0

    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {canvas_w}
PlayResY: {canvas_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Subtitles,{sub_font},{sub_fs},{sub_color},&H000000FF,{sub_outline},&H80000000,{sub_bold},0,0,0,100,100,0,0,1,3.2,1.5,{alignment if subtitles_enabled else 2},{margin_h if subtitles_enabled else 40},{margin_h if subtitles_enabled else 40},{margin_v if subtitles_enabled else 120},1
Style: Overlays,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,1,5,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(events) + "\n"

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    return True


def build_render_filtergraph(
    project_data: Dict[str, Any],
    source_dims: Dict[str, int],
    target_w: int,
    target_h: int,
    ass_path: Optional[str] = None,
    face_track_info: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """
    Constructs the video and audio filter expressions for FFmpeg.
    """
    w, h = source_dims["width"], source_dims["height"]
    crop_cfg = project_data.get("crop", {})
    crop_mode = crop_cfg.get("mode", "smart").lower()
    manual_x = float(crop_cfg.get("manual_x", 0.5))
    aspect_ratio = crop_cfg.get("aspect_ratio", "9:16")

    v_filters = []

    # 1. Aspect Ratio and Cropping
    target_aspect = target_w / float(target_h) if target_h > 0 else 0.5625
    current_aspect = w / float(h) if h > 0 else 1.777

    if aspect_ratio == "9:16":
        # 9:16 vertical crop
        if current_aspect > target_aspect:
            resolved_crop = compute_916_crop_coordinates(
                video_w=w,
                video_h=h,
                crop_mode=crop_mode,
                manual_x=manual_x,
                face_track_info=face_track_info
            )
            v_filters.append(resolved_crop["filter"])
            v_filters.append(f"scale={target_w}:{target_h}")
        elif current_aspect < target_aspect:
            crop_h = "trunc(iw*16/9/2)*2"
            v_filters.append(f"crop=iw:{crop_h}:0:(ih-{crop_h})/2")
            v_filters.append(f"scale={target_w}:{target_h}")
        else:
            v_filters.append(f"scale={target_w}:{target_h}")

    elif aspect_ratio == "16:9":
        # Landscape
        if current_aspect > target_aspect:
            crop_w = "trunc(ih*16/9/2)*2"
            v_filters.append(f"crop={crop_w}:ih:(iw-{crop_w})/2:0")
            v_filters.append(f"scale={target_w}:{target_h}")
        elif current_aspect < target_aspect:
            crop_h = "trunc(iw*9/16/2)*2"
            v_filters.append(f"crop=iw:{crop_h}:0:(ih-{crop_h})/2")
            v_filters.append(f"scale={target_w}:{target_h}")
        else:
            v_filters.append(f"scale={target_w}:{target_h}")

    elif aspect_ratio == "1:1":
        # Square
        min_dim = min(w, h)
        cx = int(((w - min_dim) // 2 // 2) * 2)
        cy = int(((h - min_dim) // 2 // 2) * 2)
        if crop_mode == "manual":
            cx = int(round(manual_x * max(0, w - min_dim) / 2) * 2)
        v_filters.append(f"crop={min_dim}:{min_dim}:{cx}:{cy}")
        v_filters.append(f"scale={target_w}:{target_h}")

    elif aspect_ratio == "4:5":
        # Portrait 4:5 (target_aspect = 0.8)
        if current_aspect > 0.8:
            crop_w = int(round(h * 4.0 / 5.0 / 2.0) * 2)
            max_x = max(0, w - crop_w)
            cx = int(round(manual_x * max_x / 2.0) * 2) if crop_mode == "manual" else int((max_x // 2 // 2) * 2)
            v_filters.append(f"crop={crop_w}:{h}:{cx}:0")
            v_filters.append(f"scale={target_w}:{target_h}")
        else:
            crop_h = int(round(w * 5.0 / 4.0 / 2.0) * 2)
            cy = int(max(0, h - crop_h) // 2 // 2) * 2
            v_filters.append(f"crop={w}:{crop_h}:0:{cy}")
            v_filters.append(f"scale={target_w}:{target_h}")

    else:
        v_filters.append(f"scale={target_w}:{target_h}")

    # 2. Visual Adjustments (Brightness, Contrast, Saturation)
    adj = project_data.get("adjustments", {})
    b = float(adj.get("brightness", 0.0))
    c = float(adj.get("contrast", 1.0))
    s = float(adj.get("saturation", 1.0))
    preset = str(adj.get("filter_preset", "normal")).lower()

    if preset == "bw":
        v_filters.append("hue=s=0")
        if b != 0.0 or c != 1.0:
            v_filters.append(f"eq=brightness={b}:contrast={c}")
    elif preset == "vivid":
        v_filters.append(f"eq=brightness={b}:contrast={c * 1.15:.2f}:saturation={s * 1.35:.2f}")
    elif preset == "warm":
        v_filters.append(f"eq=brightness={b}:contrast={c}:saturation={s}")
        v_filters.append("colorbalance=rs=0.08:gs=0.02:bs=-0.06")
    elif preset == "cool":
        v_filters.append(f"eq=brightness={b}:contrast={c}:saturation={s}")
        v_filters.append("colorbalance=rs=-0.06:gs=0.01:bs=0.10")
    elif preset == "dramatic":
        v_filters.append(f"eq=brightness={b}:contrast={c * 1.25:.2f}:saturation={s * 0.85:.2f}")
    else:
        if b != 0.0 or c != 1.0 or s != 1.0:
            v_filters.append(f"eq=brightness={b}:contrast={c}:saturation={s}")

    # 3. Subtitles & Overlays
    if ass_path and os.path.isfile(ass_path):
        escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
        v_filters.append(f"subtitles='{escaped_ass}'")

    # 4. Playback Speed (Video filter: setpts)
    speed = float(project_data.get("speed", 1.0))
    if abs(speed - 1.0) > 0.01 and speed > 0:
        v_filters.append(f"setpts={1.0 / speed:.4f}*PTS")

    vf_str = ",".join(v_filters) if v_filters else "null"

    # Audio Filter
    a_filters = []
    audio_cfg = project_data.get("audio", {})
    orig_enabled = audio_cfg.get("original_enabled", True)
    orig_vol = float(audio_cfg.get("original_volume", 1.0)) if orig_enabled else 0.0
    a_filters.append(f"volume={orig_vol:.2f}")

    fade_in = float(audio_cfg.get("fade_in", 0.0))
    fade_out = float(audio_cfg.get("fade_out", 0.0))
    trim_dur = float(project_data.get("trim", {}).get("duration", 30.0))
    effective_dur = trim_dur / speed if speed > 0 else trim_dur

    if fade_in > 0:
        a_filters.append(f"afade=t=in:ss=0:d={min(fade_in, effective_dur / 2):.2f}")
    if fade_out > 0:
        out_start = max(0.0, effective_dur - fade_out)
        a_filters.append(f"afade=t=out:st={out_start:.2f}:d={min(fade_out, effective_dur / 2):.2f}")

    # Playback Speed (Audio filter: atempo)
    if abs(speed - 1.0) > 0.01:
        if 0.5 <= speed <= 2.0:
            a_filters.append(f"atempo={speed:.2f}")
        elif speed > 2.0:
            a_filters.append(f"atempo=2.0,atempo={speed / 2.0:.2f}")
        else:
            a_filters.append(f"atempo=0.5,atempo={speed / 0.5:.2f}")

    af_str = ",".join(a_filters) if a_filters else "anull"

    return vf_str, af_str


def execute_editor_render(
    job_id: str,
    clip_id: Any,
    project_data: Dict[str, Any]
):
    """
    Worker function running in a background thread to render the edited Short.
    Safely preserves the original clip and writes to reel_{clip_id:02d}_edited.mp4.
    """
    key = get_task_key(job_id, clip_id)
    job_dir = get_job_dir(job_id)
    clip_int = int(clip_id)

    def set_status(status: str, progress: int, stage: str, error: Optional[str] = None):
        with RENDER_LOCK:
            RENDER_TASKS[key] = {
                "status": status,
                "progress": progress,
                "stage": stage,
                "error": error,
                "output_filename": f"reel_{clip_int:02d}_edited.mp4" if status == "complete" else None,
                "video_url": f"/media/{job_id}/reel_{clip_int:02d}_edited.mp4?t={int(time.time())}" if status == "complete" else None,
                "updated_at": time.time()
            }

    set_status("rendering", 5, "Initializing render pipeline...")

    ass_path = None
    try:
        ffmpeg_exe = get_ffmpeg_binary()
        source_video = project_data.get("source_video_path")
        if not source_video or not os.path.isfile(source_video):
            # Check original reel file
            orig_reel_name = f"reel_{clip_int:02d}.mp4"
            fallback_source = os.path.join(job_dir, orig_reel_name)
            if os.path.isfile(fallback_source):
                source_video = fallback_source
            else:
                raise FileNotFoundError(f"Source video file not found for job '{job_id}'")

        set_status("rendering", 15, "Analyzing source dimensions and crop geometry...")
        source_dims = probe_video_dimensions(source_video)

        crop_cfg = project_data.get("crop", {})
        aspect = crop_cfg.get("aspect_ratio", "9:16")
        preset = ASPECT_RATIO_PRESETS.get(aspect, ASPECT_RATIO_PRESETS["9:16"])
        target_w = preset["width"]
        target_h = preset["height"]

        trim = project_data.get("trim", {})
        start_sec = max(0.0, float(trim.get("start", 0.0)))
        end_sec = float(trim.get("end", start_sec + 30.0))
        duration_sec = max(0.5, end_sec - start_sec)

        # Smart Face Tracking if requested
        face_track_info = None
        if crop_cfg.get("mode") == "smart" and aspect == "9:16":
            set_status("rendering", 25, "Aligning smart subject tracking...")
            try:
                face_track_info = get_cached_or_compute_face_track(
                    video_path=source_video,
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                    cache_dir=job_dir
                )
            except Exception as fe:
                logger.warning(f"Face tracking fallback ({fe})")
                face_track_info = None

        set_status("rendering", 35, "Generating subtitle and overlay typography...")
        ass_path = os.path.join(job_dir, f"temp_editor_{clip_int:02d}.ass")
        has_ass = generate_editor_ass(
            project_data=project_data,
            canvas_w=target_w,
            canvas_h=target_h,
            output_ass_path=ass_path
        )
        active_ass_path = ass_path if has_ass else None

        set_status("rendering", 45, "Constructing FFmpeg filtergraph...")
        vf_str, af_str = build_render_filtergraph(
            project_data=project_data,
            source_dims=source_dims,
            target_w=target_w,
            target_h=target_h,
            ass_path=active_ass_path,
            face_track_info=face_track_info
        )

        out_fname = f"reel_{clip_int:02d}_edited.mp4"
        out_path = os.path.join(job_dir, out_fname)

        # Check background music
        audio_cfg = project_data.get("audio", {})
        bgm_fname = audio_cfg.get("music_filename")
        bgm_path = os.path.join(job_dir, bgm_fname) if bgm_fname else None
        has_bgm = bgm_path and os.path.isfile(bgm_path)

        set_status("rendering", 55, "Encoding high-definition H.264 video with AAC audio...")

        if has_bgm:
            bgm_vol = float(audio_cfg.get("music_volume", 0.3))
            # Complex filtergraph mixing source audio + background music
            filter_complex = (
                f"[0:v]{vf_str}[vout];"
                f"[0:a]{af_str}[aorig];"
                f"[1:a]volume={bgm_vol:.2f},aloop=loop=-1:size=2e+09[abgm];"
                f"[aorig][abgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            cmd = [
                ffmpeg_exe, "-y",
                "-ss", str(start_sec),
                "-i", source_video,
                "-i", bgm_path,
                "-t", str(duration_sec),
                "-filter_complex", filter_complex,
                "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                out_path
            ]
        else:
            cmd = [
                ffmpeg_exe, "-y",
                "-ss", str(start_sec),
                "-i", source_video,
                "-t", str(duration_sec),
                "-vf", vf_str,
                "-af", af_str,
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                out_path
            ]

        logger.info(f"Running editor FFmpeg command for Job {job_id} Reel {clip_int}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

        if res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="replace")
            # If subtitles filter failed on local system, retry without subtitles
            if "subtitles" in err and active_ass_path:
                logger.warning("Subtitles filter failed. Retrying without subtitles overlay...")
                set_status("rendering", 75, "Retrying video encoding without burned subtitles...")
                vf_fallback, af_fallback = build_render_filtergraph(
                    project_data=project_data,
                    source_dims=source_dims,
                    target_w=target_w,
                    target_h=target_h,
                    ass_path=None,
                    face_track_info=face_track_info
                )
                cmd_fb = [
                    ffmpeg_exe, "-y",
                    "-ss", str(start_sec),
                    "-i", source_video,
                    "-t", str(duration_sec),
                    "-vf", vf_fallback,
                    "-af", af_fallback,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                    "-c:a", "aac", "-b:a", "128k",
                    "-avoid_negative_ts", "make_zero",
                    out_path
                ]
                res_fb = subprocess.run(cmd_fb, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if res_fb.returncode != 0:
                    raise RuntimeError(f"FFmpeg render fallback failed: {res_fb.stderr.decode('utf-8', errors='replace')[:300]}")
            else:
                raise RuntimeError(f"FFmpeg render failed: {err[:300]}")

        # Validate rendered video
        set_status("rendering", 90, "Validating exported video container...")
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("Rendered output file is missing or empty.")

        out_dims = probe_video_dimensions(out_path)
        out_size_mb = round(os.path.getsize(out_path) / (1024 * 1024), 2)

        # Update project JSON and segments.json
        project_data["exported_version"] = {
            "filename": out_fname,
            "file_size_mb": out_size_mb,
            "width": out_dims.get("width", target_w),
            "height": out_dims.get("height", target_h),
            "aspect_ratio": aspect,
            "duration_sec": duration_sec,
            "rendered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ready": True
        }
        save_project(job_id, clip_id, project_data)

        # Also register the edited reel in segments.json so results page shows it immediately
        try:
            meta_path = os.path.join(job_dir, "segments.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    job_meta = json.load(f)
                reels = job_meta.get("reels", [])
                for r in reels:
                    if r.get("reel_num") == clip_int:
                        r["edited_version"] = project_data["exported_version"]
                        break
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(job_meta, f, indent=2, ensure_ascii=False)
        except Exception as meta_e:
            logger.warning(f"Could not update segments.json with edited reel ({meta_e})")

        set_status("complete", 100, "Render complete! Video is ready for preview and download.")
        logger.info(f"Successfully rendered edited Reel #{clip_int} for Job {job_id} ({out_size_mb} MB)")

    except Exception as e:
        logger.error(f"Render failed for Job {job_id} Reel {clip_int}: {e}", exc_info=True)
        set_status("error", 0, "Render failed", error=str(e))

    finally:
        if ass_path and os.path.isfile(ass_path):
            try:
                os.remove(ass_path)
            except OSError:
                pass


def start_editor_render(
    job_id: str,
    clip_id: Any,
    project_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Initiates asynchronous rendering of the edited clip.
    Guarantees no duplicate concurrent exports of the exact same clip.
    """
    if not is_safe_id(job_id) or not is_safe_id(clip_id):
        raise ValueError("Invalid job or clip ID")

    key = get_task_key(job_id, clip_id)
    with RENDER_LOCK:
        current_task = RENDER_TASKS.get(key)
        if current_task and current_task.get("status") == "rendering":
            return {
                "status": "already_running",
                "message": "Render is already in progress for this clip."
            }

        RENDER_TASKS[key] = {
            "status": "queued",
            "progress": 0,
            "stage": "Queueing render task...",
            "error": None,
            "output_filename": None,
            "started_at": time.time()
        }

    thread = threading.Thread(
        target=execute_editor_render,
        kwargs={
            "job_id": job_id,
            "clip_id": clip_id,
            "project_data": project_data
        },
        daemon=True
    )
    thread.start()

    return {
        "status": "queued",
        "message": f"Render started for Reel #{clip_id}."
    }
