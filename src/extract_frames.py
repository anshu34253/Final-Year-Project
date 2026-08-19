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


def extract_frames_from_video(
    video_path: str,
    sample_fps: float = 1.0,
    save_dir: str = None
) -> dict:
    """
    Extracts frames from a video at a specified sampling frame rate (sample_fps).

    Parameters:
        video_path (str): Path to the input .mp4 video file.
        sample_fps (float): Target number of frames to extract per second (default: 1.0 = 1 frame/sec).
        save_dir (str, optional): Directory path to save extracted frame images (.jpg).

    Returns:
        dict: Summary metadata and list of extracted frame arrays:
            - "video_path": str
            - "video_id": str
            - "fps": float (original video FPS)
            - "duration": float (duration in seconds)
            - "total_original_frames": int
            - "num_sampled_frames": int
            - "timestamps": list[int] (timestamp in seconds for each extracted frame)
            - "output_dir": str
            - "frames": list[np.ndarray] (RGB numpy image arrays)
            - "saved_paths": list[str] (file paths of saved frames)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[ExtractFrames] Video file not found: {video_path}")

    # Open video capture stream using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"[ExtractFrames] Could not open video file: {video_path}")

    # Retrieve video properties from OpenCV
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_original_frames / original_fps if original_fps > 0 else 0.0

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    if save_dir is None:
        save_dir = os.path.join("outputs", "frames", video_id)

    os.makedirs(save_dir, exist_ok=True)

    # Frame step size calculation (how many raw video frames to skip per sampled frame)
    frame_step = max(1, int(round(original_fps / sample_fps)))

    extracted_frames = []
    timestamps = []
    saved_paths = []

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video stream

        # Sample frame every `frame_step` frames
        if frame_count % frame_step == 0:
            # OpenCV reads in BGR format; convert to RGB standard for visualization & PyTorch
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extracted_frames.append(frame_rgb)

            # Calculate timestamp in seconds
            sec_timestamp = int(round(saved_count / sample_fps))
            timestamps.append(sec_timestamp)

            # Save frame image file with index and timestamp, e.g., frame_000000_0s.jpg
            frame_filename = f"frame_{saved_count:06d}_{sec_timestamp}s.jpg"
            frame_filepath = os.path.join(save_dir, frame_filename)
            
            cv2.imwrite(frame_filepath, frame)  # Save OpenCV BGR frame to JPEG
            saved_paths.append(frame_filepath)

            saved_count += 1

        frame_count += 1

    cap.release()

    result = {
        "video_path": video_path,
        "video_id": video_id,
        "fps": original_fps,
        "duration": duration_sec,
        "total_original_frames": total_original_frames,
        "num_sampled_frames": len(extracted_frames),
        "timestamps": timestamps,
        "output_dir": save_dir,
        "frames": extracted_frames,
        "saved_paths": saved_paths
    }

    return result


if __name__ == "__main__":
    print("Testing extract_frames module...")
