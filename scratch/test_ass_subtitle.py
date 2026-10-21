import os
import subprocess
import json
import imageio_ffmpeg

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print("Using ffmpeg binary:", ffmpeg_exe)

video_width = 1080
video_height = 1920

def create_styled_ass(
    speech_segments: list,
    reel_start_sec: float,
    reel_end_sec: float,
    output_ass_path: str,
    video_w: int = 1080,
    video_h: int = 1920,
    font_name: str = "Arial"
) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(output_ass_path)), exist_ok=True)

    valid_segments = []
    for seg in speech_segments:
        s = seg["start"]
        e = seg["end"]
        text = seg["text"].strip()
        if e > reel_start_sec and s < reel_end_sec and text:
            valid_segments.append({"start": s, "end": e, "text": text})

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

        max_words_per_chunk = 6
        if len(words) <= max_words_per_chunk:
            chunks = [" ".join(words)]
        else:
            num_chunks = (len(words) + max_words_per_chunk - 1) // max_words_per_chunk
            chunk_size = (len(words) + num_chunks - 1) // num_chunks
            chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

        chunk_dur = dur / len(chunks)
        for idx, chk in enumerate(chunks):
            c_s = rel_s + idx * chunk_dur
            c_e = c_s + chunk_dur
            
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

    font_size = int(video_h * 0.024)  # ~46px for 1080x1920
    margin_v = int(video_h * 0.12)     # ~230px from bottom
    margin_h = int(video_w * 0.10)     # ~108px from left/right

    def format_ass_time(seconds: float) -> str:
        s = max(0.0, seconds)
        hrs = int(s // 3600)
        mins = int((s % 3600) // 60)
        secs = int(s % 60)
        cs = int(round((s - int(s)) * 100))
        if cs >= 100:
            secs += 1
            cs = 0
        return f"{hrs:01d}:{mins:02d}:{secs:02d}.{cs:02d}"

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H70000000,1,0,0,0,100,100,0,0,1,3.0,1.5,2,{margin_h},{margin_h},{margin_v},1

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

sample_video = "data/youtube_downloads/test_final/8xTyh6Y_3Ww.mp4"
if not os.path.exists(sample_video):
    sample_video = "data/youtube_downloads/test_pytube/8xTyh6Y_3Ww.mp4"

print("Sample video path:", sample_video, "| Exists:", os.path.exists(sample_video))

output_reel = "scratch/test_styled_reel.mp4"
ass_path = "scratch/test_styled_reel.ass"

sample_segments = [
    {"start": 1.0, "end": 6.5, "text": "Take 5 minutes charge and you will get the video playback time of 5 to 6 hours."},
    {"start": 7.0, "end": 12.0, "text": "The camera performance on the new iPhone 18 Pro Max is absolutely insane."},
    {"start": 12.5, "end": 17.5, "text": "Let us compare this directly with the previous generation."}
]

res_ass = create_styled_ass(
    speech_segments=sample_segments,
    reel_start_sec=0.0,
    reel_end_sec=20.0,
    output_ass_path=ass_path,
    video_w=1080,
    video_h=1920,
    font_name="Arial"
)
print("ASS file created:", res_ass)

escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
vf_arg = f"crop=trunc(ih*9/16/2)*2:ih:(iw-trunc(ih*9/16/2)*2)/2:0,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,subtitles='{escaped_ass}'"

cmd = [
    ffmpeg_exe, "-y",
    "-ss", "0",
    "-i", sample_video,
    "-t", "15",
    "-vf", vf_arg,
    "-c:v", "libx264", "-preset", "fast", "-crf", "22",
    "-c:a", "aac", "-b:a", "128k",
    output_reel
]

print("\nRunning FFmpeg subtitle render test...")
res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
if res.returncode == 0:
    print("SUCCESS! Rendered video size:", os.path.getsize(output_reel), "bytes")
else:
    print("FAILED! FFmpeg error:", res.stderr[:500])
