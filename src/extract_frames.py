"""
extract_frames.py
-----------------
Extracts individual image frames from raw video files using OpenCV.

Video frame extraction is the first step in the pipeline. Instead of saving
every single frame (which consumes too much memory), we downsample the video by
extracting 1 frame per second (or every N-th frame).
"""

import os
import cv2
import numpy as np


def extract_frames_from_video(video_path: str, sample_fps: int = 1, save_dir: str = None) -> list[np.ndarray]:
    """
    Extracts frames from a video at a specified frame rate (sample_fps).

    Parameters:
        video_path (str): Path to the input .mp4 / video file.
        sample_fps (int): Target number of frames to extract per second of video.
        save_dir (str, optional): Directory to save extracted frame images (.jpg).

    Returns:
        list[np.ndarray]: List of extracted frame images in RGB format (NumPy arrays).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[ExtractFrames] Video file not found: {video_path}")

    # Open video capture stream using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"[ExtractFrames] Could not open video file: {video_path}")

    # Retrieve video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / original_fps if original_fps > 0 else 0

    print(f"[ExtractFrames] Processing: {os.path.basename(video_path)}")
    print(f"                FPS: {original_fps:.2f} | Total Frames: {total_frames} | Duration: {duration_sec:.2f}s")

    # Calculate frame step size (how many raw frames to skip for each sampled frame)
    frame_step = max(1, int(original_fps / sample_fps))

    extracted_frames = []
    frame_count = 0
    saved_count = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video stream

        # Sample frame every `frame_step` frames
        if frame_count % frame_step == 0:
            # OpenCV reads images in BGR format; convert to RGB standard for PyTorch/ResNet
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extracted_frames.append(frame_rgb)

            # Optionally save frame image to disk
            if save_dir:
                frame_filename = os.path.join(save_dir, f"frame_{saved_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)  # OpenCV saves in BGR format
                
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[ExtractFrames] Successfully extracted {len(extracted_frames)} frames.")
    return extracted_frames


if __name__ == "__main__":
    # Example standalone test execution
    print("Testing extract_frames module...")
