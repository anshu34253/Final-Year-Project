"""
highlight_generator.py
----------------------
Generates a short summary highlight video using MoviePy and FFmpeg.

This module processes predicted importance scores, identifies top-scoring video segments,
slices subclips from the original raw video using MoviePy/FFmpeg, and stitches them
together into a final summary highlight video.
"""

import os
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips


def select_highlight_segments(
    scores: np.ndarray,
    top_k_percent: float = 0.20,
    segment_duration_sec: float = 1.0
) -> list[tuple[float, float]]:
    """
    Identifies top-scoring contiguous video time segments for highlights.

    Parameters:
        scores (np.ndarray): 1D array of importance scores (one per segment).
        top_k_percent (float): Percentage of total video duration to include in highlight (e.g., 0.20 for top 20%).
        segment_duration_sec (float): Duration in seconds represented by each score index.

    Returns:
        list[tuple[float, float]]: List of (start_sec, end_sec) time intervals to cut.
    """
    total_segments = len(scores)
    if total_segments == 0:
        return []

    # Calculate target number of segments to keep
    num_to_select = max(1, int(total_segments * top_k_percent))

    # Find score threshold value for top K segments
    threshold = np.partition(scores, -num_to_select)[-num_to_select]

    # Mask binary indicators for selected segments
    selected_mask = scores >= threshold

    # Group contiguous selected segments into continuous subclip intervals (start, end)
    intervals = []
    in_highlight = False
    start_time = 0.0

    for idx, is_selected in enumerate(selected_mask):
        current_time = idx * segment_duration_sec

        if is_selected and not in_highlight:
            # Segment start transition
            in_highlight = True
            start_time = current_time

        elif not is_selected and in_highlight:
            # Segment end transition
            in_highlight = False
            end_time = current_time
            intervals.append((start_time, end_time))

    # Handle case where video ends while still inside a highlight clip
    if in_highlight:
        intervals.append((start_time, total_segments * segment_duration_sec))

    print(f"[HighlightGen] Selected {len(intervals)} highlight clips covering top {top_k_percent*100:.0f}% moments.")
    return intervals


def generate_highlight_video(
    video_path: str,
    scores: np.ndarray,
    output_path: str = "outputs/highlights/highlight_output.mp4",
    top_k_percent: float = 0.20,
    sample_fps: float = 1.0
) -> str:
    """
    Cuts selected top-scoring clips from the input video and stitches them into a highlight video.

    Parameters:
        video_path (str): Path to original input video (.mp4).
        scores (np.ndarray): Predicted segment importance scores.
        output_path (str): File destination path for the rendered summary video.
        top_k_percent (float): Target fraction of video length to retain (e.g. 0.20 = 20%).
        sample_fps (float): Frames per second sampling rate used during feature extraction.

    Returns:
        str: Absolute path to rendered output highlight video file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[HighlightGen] Input video not found: {video_path}")

    # Each score index corresponds to (1 / sample_fps) seconds
    segment_duration_sec = 1.0 / sample_fps if sample_fps > 0 else 1.0

    # Determine time intervals for highlight clips
    intervals = select_highlight_segments(
        scores=scores,
        top_k_percent=top_k_percent,
        segment_duration_sec=segment_duration_sec
    )

    if not intervals:
        print("[HighlightGen] Warning: No highlight segments selected.")
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[HighlightGen] Loading original video via MoviePy: {video_path}")
    original_video = VideoFileClip(video_path)
    video_duration = original_video.duration

    subclips = []
    for start_sec, end_sec in intervals:
        # Clamp start and end times to valid video duration
        start_sec = max(0.0, start_sec)
        end_sec = min(video_duration, end_sec)

        if end_sec > start_sec:
            clip = original_video.subclip(start_sec, end_sec)
            subclips.append(clip)

    if subclips:
        print(f"[HighlightGen] Concatenating {len(subclips)} clips into highlight video...")
        final_highlight = concatenate_videoclips(subclips)
        
        # Write final video file to output folder using FFmpeg codec
        final_highlight.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            logger=None  # Set logger='bar' for progress bar
        )
        print(f"[HighlightGen] Rendered highlight video successfully saved to: {output_path}")

        # Close video readers to free memory resources
        final_highlight.close()
        original_video.close()

        return output_path

    original_video.close()
    return ""


if __name__ == "__main__":
    print("Highlight generator module ready.")
