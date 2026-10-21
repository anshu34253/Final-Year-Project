"""
src/reel_renderer.py
--------------------
Phase 13: 9:16 Vertical Video Renderer with Professional Burned-In Captions.

Extracts selected Reels from source video, crops video to exact 9:16 vertical
aspect ratio (1080x1920), aligns subtitle timestamps relative to reel start,
burns in styled subtitles via FFmpeg, and writes reel_01.mp4 ... reel_N.mp4 and manifest.json.

Guarantees:
- OpenCV + FFprobe hybrid probing to determine source video dimensions accurately.
- Center crop to 9:16 aspect ratio (1080x1920) without black bars or letterboxing.
- Advanced SubStation Alpha (.ass) format with explicit resolution canvas (PlayResX=1080, PlayResY=1920).
- Dynamically scaled font size (~46px for 1080x1920).
- Max 1-2 balanced lines per caption block (3-7 words per phrase).
- Safe lower-middle area positioning (12% margin from bottom, 10% side margins).
- Sleek white text with subtle dark background box/outline.
"""

import os
import sys
import json
import shutil
import logging
import subprocess
from typing import List, Dict, Any, Optional

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("ReelRenderer")


def get_ffmpeg_binary() -> str:
    """Returns absolute path to ffmpeg executable, checking PATH then imageio_ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def get_ffprobe_binary() -> str:
    """Returns absolute path to ffprobe executable, checking PATH then imageio_ffmpeg."""
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = os.path.dirname(ffmpeg_bin)
        candidate = os.path.join(bin_dir, "ffprobe.exe")
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    return "ffprobe"


def probe_video_dimensions(video_path: str) -> Dict[str, int]:
    """Probes video width and height using OpenCV first, then ffprobe fallback."""
    # Method 1: Try OpenCV (fastest and most reliable)
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return {"width": w, "height": h}
    except Exception as cv_err:
        logger.debug(f"OpenCV probe failed ({cv_err}). Trying ffprobe...")

    # Method 2: Fallback to ffprobe
    ffprobe_exe = get_ffprobe_binary()
    cmd = [
        ffprobe_exe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        data = json.loads(res.stdout)
        stream = data["streams"][0]
        return {"width": int(stream["width"]), "height": int(stream["height"])}
    except Exception as e:
        logger.warning(f"Could not probe video dimensions ({str(e)}). Defaulting to 1920x1080.")
        return {"width": 1920, "height": 1080}


def format_srt_timestamp(seconds: float) -> str:
    """Formats seconds into SRT timestamp string HH:MM:SS,mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def generate_reel_srt(
    speech_segments: List[Dict[str, Any]],
    reel_start_sec: float,
    reel_end_sec: float,
    output_srt_path: str
) -> bool:
    """
    Creates an SRT subtitle file with timestamps shifted relative to reel_start_sec.
    """
    if not speech_segments:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)), exist_ok=True)
    count = 1

    with open(output_srt_path, "w", encoding="utf-8") as f:
        for seg in speech_segments:
            s = seg["start"]
            e = seg["end"]
            text = seg["text"].strip()

            if e <= reel_start_sec or s >= reel_end_sec or not text:
                continue

            rel_s = max(0.0, s - reel_start_sec)
            rel_e = min(reel_end_sec - reel_start_sec, e - reel_start_sec)

            if rel_e <= rel_s:
                continue

            f.write(f"{count}\n")
            f.write(f"{format_srt_timestamp(rel_s)} --> {format_srt_timestamp(rel_e)}\n")
            f.write(f"{text}\n\n")
            count += 1

    return count > 1


def generate_reel_ass(
    speech_segments: List[Dict[str, Any]],
    reel_start_sec: float,
    reel_end_sec: float,
    output_ass_path: str,
    video_w: int = 1080,
    video_h: int = 1920,
    fallback_transcript: Optional[str] = None,
    font_name: str = "Arial",
    subtitle_style: str = "bold",
    subtitle_position: str = "bottom",
    enable_emojis: bool = False
) -> bool:
    """
    Generates a beautifully styled Advanced SubStation Alpha (.ass) subtitle file.
    Delegates to src.subtitle_service.create_styled_ass with support for style, position, and emojis.
    """
    from src.subtitle_service import create_styled_ass
    return create_styled_ass(
        speech_segments=speech_segments,
        reel_start_sec=reel_start_sec,
        reel_end_sec=reel_end_sec,
        output_ass_path=output_ass_path,
        video_w=video_w,
        video_h=video_h,
        fallback_transcript=fallback_transcript,
        style_name=subtitle_style,
        position_name=subtitle_position,
        enable_emojis=enable_emojis
    )


def render_single_reel(
    source_video_path: str,
    start_sec: float,
    duration_sec: float,
    output_path: str,
    aspect_ratio: str = "9:16",
    speech_segments: Optional[List[Dict[str, Any]]] = None,
    captions: bool = True,
    transcript: Optional[str] = None,
    crop_mode: str = "smart",
    manual_x: float = 0.5,
    subtitle_style: str = "bold",
    subtitle_position: str = "bottom",
    emojis: bool = False,
    face_track_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Renders an individual reel using FFmpeg with 9:16 vertical crop, smart face tracking,
    and professional burned-in captions with customizable style, position, and emojis.
    """
    from src.crop_strategy import compute_916_crop_coordinates
    from src.face_tracking import get_cached_or_compute_face_track

    ffmpeg_exe = get_ffmpeg_binary()
    dims = probe_video_dimensions(source_video_path)
    w, h = dims["width"], dims["height"]

    target_w, target_h = 1080, 1920
    video_filters = []

    # Enforce 9:16 Vertical crop
    resolved_crop_info = None
    if aspect_ratio == "9:16":
        current_ratio = w / float(h) if h > 0 else 1.777
        target_ratio = 9.0 / 16.0  # 0.5625

        if current_ratio > target_ratio:
            # Video is wider than 9:16 -> determine horizontal crop offset
            cache_dir = os.path.dirname(os.path.abspath(output_path))
            track_info = face_track_info
            if crop_mode.lower() == "smart" and track_info is None:
                try:
                    track_info = get_cached_or_compute_face_track(
                        video_path=source_video_path,
                        start_sec=start_sec,
                        duration_sec=duration_sec,
                        cache_dir=cache_dir
                    )
                except Exception as track_err:
                    logger.warning(f"Face tracking error ({track_err}). Falling back to center crop.")
                    track_info = None

            resolved_crop_info = compute_916_crop_coordinates(
                video_w=w,
                video_h=h,
                crop_mode=crop_mode,
                manual_x=manual_x,
                face_track_info=track_info
            )
            video_filters.append(resolved_crop_info["filter"])
            video_filters.append(f"scale={target_w}:{target_h}")

        elif current_ratio < target_ratio:
            # Video is taller than 9:16 -> crop top/bottom
            crop_h = "trunc(iw*16/9/2)*2"
            video_filters.append(f"crop=iw:{crop_h}:0:(ih-{crop_h})/2")
            video_filters.append(f"scale={target_w}:{target_h}")
        else:
            # Already 9:16 -> scale to exact 1080x1920
            video_filters.append(f"scale={target_w}:{target_h}")

    # Burned-in captions via Advanced SubStation Alpha (.ass)
    ass_path = None
    if captions:
        ass_path = output_path.replace(".mp4", ".ass")
        has_subs = generate_reel_ass(
            speech_segments=speech_segments or [],
            reel_start_sec=start_sec,
            reel_end_sec=start_sec + duration_sec,
            output_ass_path=ass_path,
            video_w=target_w,
            video_h=target_h,
            fallback_transcript=transcript,
            subtitle_style=subtitle_style,
            subtitle_position=subtitle_position,
            enable_emojis=emojis
        )
        if has_subs:
            escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
            video_filters.append(f"subtitles='{escaped_ass}'")

    vf_arg = ",".join(video_filters) if video_filters else "null"

    cmd = [
        ffmpeg_exe, "-y",
        "-ss", str(start_sec),
        "-i", source_video_path,
        "-t", str(duration_sec),
        "-vf", vf_arg,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="replace")
            # If subtitles filter failed, retry without subtitles
            if "subtitles" in err and ass_path:
                logger.warning("Subtitles filter failed in FFmpeg. Retrying without burned-in captions...")
                no_sub_filters = [f for f in video_filters if not f.startswith("subtitles")]
                vf_fallback = ",".join(no_sub_filters) if no_sub_filters else "null"
                fallback_cmd = [
                    ffmpeg_exe, "-y",
                    "-ss", str(start_sec),
                    "-i", source_video_path,
                    "-t", str(duration_sec),
                    "-vf", vf_fallback,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                    "-c:a", "aac", "-b:a", "128k",
                    "-avoid_negative_ts", "make_zero",
                    output_path
                ]
                res_fb = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if res_fb.returncode != 0:
                    err_fb = res_fb.stderr.decode("utf-8", errors="replace")
                    raise RuntimeError(f"FFmpeg reel rendering fallback failed: {err_fb[:300]}")
            else:
                raise RuntimeError(f"FFmpeg reel rendering failed: {err[:300]}")

    finally:
        # Clean up temporary ASS file
        if ass_path and os.path.isfile(ass_path):
            try:
                os.remove(ass_path)
            except OSError:
                pass

    file_size_bytes = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    return {
        "output_path": output_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "crop_info": resolved_crop_info,
        "captions_burned": bool(captions and ass_path),
        "emojis_applied": bool(emojis and captions)
    }


def render_multiple_reels(
    source_video_path: str,
    selected_reels: List[Dict[str, Any]],
    output_dir: str,
    aspect_ratio: str = "9:16",
    captions: bool = True,
    crop_mode: str = "smart",
    manual_x: float = 0.5,
    subtitle_style: str = "bold",
    subtitle_position: str = "bottom",
    emojis: bool = False,
    stage_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Renders all selected reels into output_dir and writes manifest.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_reels = []

    total_reels = len(selected_reels)
    logger.info(
        f"Rendering {total_reels} Reels (aspect: {aspect_ratio}, crop: {crop_mode}, "
        f"captions: {'ON' if captions else 'OFF'}, emojis: {'ON' if emojis else 'OFF'})..."
    )

    for i, reel in enumerate(selected_reels):
        reel_num = reel["reel_num"]
        out_filename = f"reel_{reel_num:02d}.mp4"
        out_path = os.path.join(output_dir, out_filename)

        if stage_callback:
            stage_callback(f"Smart 9:16 Rendering Reel {reel_num}/{total_reels} (Face Tracking & Captions)...")

        render_info = render_single_reel(
            source_video_path=source_video_path,
            start_sec=reel["start_sec"],
            duration_sec=reel["duration_sec"],
            output_path=out_path,
            aspect_ratio=aspect_ratio,
            speech_segments=reel.get("speech_segments"),
            captions=captions,
            transcript=reel.get("transcript"),
            crop_mode=crop_mode,
            manual_x=manual_x,
            subtitle_style=subtitle_style,
            subtitle_position=subtitle_position,
            emojis=emojis
        )

        manifest_item = {
            "reel_num": reel_num,
            "filename": out_filename,
            "output_path": out_path,
            "start_sec": reel["start_sec"],
            "end_sec": reel["end_sec"],
            "start_formatted": reel["start_formatted"],
            "end_formatted": reel["end_formatted"],
            "duration_sec": reel["duration_sec"],
            "score": reel["score"],
            "transcript": reel["transcript"],
            "speech_segments": reel.get("speech_segments", []),
            "crop_mode": crop_mode,
            "manual_x": manual_x,
            "subtitle_style": subtitle_style,
            "subtitle_position": subtitle_position,
            "emojis": emojis,
            "captions": captions,
            "crop_info": render_info.get("crop_info"),
            "file_size_mb": render_info["file_size_mb"]
        }
        manifest_reels.append(manifest_item)

    if stage_callback:
        stage_callback("Finalizing reels & manifest...")

    # Write manifest.json
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest_data = {
        "source_video": os.path.basename(source_video_path),
        "source_video_path": os.path.abspath(source_video_path),
        "total_reels": total_reels,
        "aspect_ratio": aspect_ratio,
        "captions": captions,
        "crop_mode": crop_mode,
        "subtitle_style": subtitle_style,
        "subtitle_position": subtitle_position,
        "emojis": emojis,
        "reels": manifest_reels
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully rendered all {total_reels} Reels! Manifest saved to {manifest_path}")

    return manifest_data

