"""
src/subtitle_service.py
-----------------------
Subtitle Styling, Positioning, and Formatting Service.
Generates SubStation Alpha (.ass) files with selectable visual styles
(Clean, Bold, Minimal), vertical positions (Bottom, Center, Top),
and optional contextual emoji enhancement.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from src.utf8_utils import setup_utf8_environment
from src.emoji_service import enhance_segments_with_emojis
setup_utf8_environment()

logger = logging.getLogger("SubtitleService")

# Subtitle visual styles dictionary
SUBTITLE_STYLES = {
    "bold": {
        "font_name": "Arial",
        "font_size_ratio": 0.026,   # ~50px for 1080x1920
        "primary_color": "&H00FFFFFF",  # Pure White
        "outline_color": "&H00000000",  # Opaque Black
        "back_color": "&H80000000",     # 50% Shadow
        "bold": 1,
        "outline_width": 3.8,
        "shadow_depth": 1.8,
    },
    "clean": {
        "font_name": "Arial",
        "font_size_ratio": 0.023,   # ~44px for 1080x1920
        "primary_color": "&H00FFFFFF",  # Pure White
        "outline_color": "&H00151515",  # Dark Charcoal
        "back_color": "&H60000000",     # Soft Shadow
        "bold": 1,
        "outline_width": 2.2,
        "shadow_depth": 1.0,
    },
    "minimal": {
        "font_name": "Arial",
        "font_size_ratio": 0.021,   # ~40px for 1080x1920
        "primary_color": "&H00FFFFFF",  # Pure White
        "outline_color": "&H00202020",  # Subtle outline
        "back_color": "&H00000000",
        "bold": 0,
        "outline_width": 1.5,
        "shadow_depth": 0.0,
    }
}

# Subtitle vertical positions dictionary
SUBTITLE_POSITIONS = {
    "bottom": {
        "alignment": 2,          # Bottom-center in ASS
        "margin_v_ratio": 0.12,  # 12% from bottom (~230px)
    },
    "center": {
        "alignment": 5,          # Middle-center in ASS
        "margin_v_ratio": 0.0,
    },
    "top": {
        "alignment": 8,          # Top-center in ASS
        "margin_v_ratio": 0.12,  # 12% from top (~230px)
    }
}


def format_ass_time(seconds: float) -> str:
    """Formats float seconds into ASS timestamp string H:MM:SS.cs."""
    s = max(0.0, seconds)
    hrs = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    cs = int(round((s - int(s)) * 100))
    if cs >= 100:
        secs += 1
        cs = 0
    return f"{hrs:01d}:{mins:02d}:{secs:02d}.{cs:02d}"


def create_styled_ass(
    speech_segments: List[Dict[str, Any]],
    reel_start_sec: float,
    reel_end_sec: float,
    output_ass_path: str,
    video_w: int = 1080,
    video_h: int = 1920,
    fallback_transcript: Optional[str] = None,
    style_name: str = "bold",
    position_name: str = "bottom",
    enable_emojis: bool = False
) -> bool:
    """
    Generates a SubStation Alpha (.ass) subtitle file formatted for 9:16 vertical video.
    Supports selectable styling (clean, bold, minimal), positions (bottom, center, top),
    and emoji enrichment.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_ass_path)), exist_ok=True)

    # Apply emoji enrichment if requested
    active_segments = enhance_segments_with_emojis(speech_segments or [], enabled=enable_emojis)

    valid_segments = []
    if active_segments:
        for seg in active_segments:
            s = seg.get("start", 0.0)
            e = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
            if e > reel_start_sec and s < reel_end_sec and text:
                valid_segments.append({"start": s, "end": e, "text": text})

    # Synthetic fallback if no timed segments overlap but transcript exists
    if not valid_segments and fallback_transcript and fallback_transcript.strip():
        words = fallback_transcript.strip().split()
        if words:
            reel_dur = max(1.0, reel_end_sec - reel_start_sec)
            chunk_size = 5
            chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
            time_per_chunk = reel_dur / len(chunks)
            for idx, chk in enumerate(chunks):
                abs_s = reel_start_sec + idx * time_per_chunk
                abs_e = abs_s + time_per_chunk
                valid_segments.append({"start": abs_s, "end": abs_e, "text": chk})

    if not valid_segments:
        return False

    formatted_dialogues = []
    for seg in valid_segments:
        s = seg["start"]
        e = seg["end"]
        text = seg["text"].strip()
        rel_s = max(0.0, s - reel_start_sec)
        rel_e = min(reel_end_sec - reel_start_sec, e - reel_start_sec)
        dur = rel_e - rel_s
        if dur <= 0.1:
            continue

        words = text.split()
        if not words:
            continue

        # Short punchy lines (max 6 words per dialogue slice)
        max_words = 6
        if len(words) <= max_words:
            chunks = [" ".join(words)]
        else:
            num_chunks = (len(words) + max_words - 1) // max_words
            chunk_size = (len(words) + num_chunks - 1) // num_chunks
            chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

        chunk_dur = dur / len(chunks)
        for idx, chk in enumerate(chunks):
            c_s = rel_s + idx * chunk_dur
            c_e = c_s + chunk_dur

            # Balance into 2 lines if longer than 22 characters
            chk_words = chk.split()
            if len(chk_words) > 4 and len(chk) > 22:
                mid = len(chk_words) // 2
                line1 = " ".join(chk_words[:mid])
                line2 = " ".join(chk_words[mid:])
                styled_text = f"{line1}\\N{line2}"
            else:
                styled_text = chk

            formatted_dialogues.append((c_s, c_e, styled_text))

    if not formatted_dialogues:
        return False

    # Retrieve style & position parameters
    style_cfg = SUBTITLE_STYLES.get(style_name.lower(), SUBTITLE_STYLES["bold"])
    pos_cfg = SUBTITLE_POSITIONS.get(position_name.lower(), SUBTITLE_POSITIONS["bottom"])

    font_name = style_cfg["font_name"]
    font_size = max(24, int(video_h * style_cfg["font_size_ratio"]))
    primary_color = style_cfg["primary_color"]
    outline_color = style_cfg["outline_color"]
    back_color = style_cfg["back_color"]
    is_bold = style_cfg["bold"]
    outline = style_cfg["outline_width"]
    shadow = style_cfg["shadow_depth"]

    alignment = pos_cfg["alignment"]
    margin_v = int(video_h * pos_cfg["margin_v_ratio"])
    margin_h = int(video_w * 0.10)  # 10% horizontal margin

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},{back_color},{is_bold},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        for c_s, c_e, text in formatted_dialogues:
            t_start = format_ass_time(c_s)
            t_end = format_ass_time(c_e)
            f.write(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{text}\n")

    return True
