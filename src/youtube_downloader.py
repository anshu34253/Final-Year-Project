"""
src/youtube_downloader.py
-------------------------
Safe acquisition of YouTube videos using yt-dlp with cookie support & fallback strategies.

Features:
- Regex validation for YouTube URLs (standard, shortened, and shorts).
- Safe download execution with yt-dlp into isolated directories.
- Automatic cookiefile detection (`cookies.txt` in project folder).
- Browser cookie fallback and client spoofing.
- Live percentage progress updates.
- Clear, actionable error messaging for YouTube bot detection challenges.
"""

import os
import sys
import re
import logging
from typing import Dict, Any, Optional
import yt_dlp

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("YouTubeDownloader")

# Valid YouTube URL patterns
YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|embed/|v/|shorts/)|youtu\.be/)([\w-]{11})([^\s]*)$"
)


def is_valid_youtube_url(url: str) -> bool:
    """
    Validates whether the provided string is a legitimate YouTube URL.
    
    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    """
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    return bool(YOUTUBE_REGEX.match(url))


def extract_video_id(url: str) -> Optional[str]:
    """Extracts the 11-character YouTube video ID."""
    if not url:
        return None
    match = YOUTUBE_REGEX.match(url.strip())
    if match:
        return match.group(5)
    return None


def find_cookie_file() -> Optional[str]:
    """Scans project directories for a user-provided cookies.txt file."""
    candidate_paths = [
        "cookies.txt",
        "data/cookies.txt",
        "FinalYearProject/cookies.txt",
        "src/cookies.txt",
        os.path.join(os.path.dirname(__file__), "..", "cookies.txt")
    ]
    for cp in candidate_paths:
        abs_p = os.path.abspath(cp)
        if os.path.isfile(abs_p) and os.path.getsize(abs_p) > 0:
            logger.info(f"Found YouTube cookie file: {abs_p}")
            return abs_p
    return None


def download_youtube_video(
    youtube_url: str,
    output_dir: str,
    max_duration_seconds: int = 1800,  # 30 min max for highlight inference
    max_filesize_mb: int = 400,
    stage_callback: callable = None
) -> Dict[str, Any]:
    """
    Downloads a YouTube video to the specified directory using yt-dlp.

    Args:
        youtube_url: Validated YouTube URL
        output_dir: Target directory for storing video
        max_duration_seconds: Maximum allowed video length
        max_filesize_mb: Maximum allowed file size
        stage_callback: Optional progress update callback

    Returns:
        Dict containing video metadata and local MP4 path
    """
    setup_utf8_environment()
    import shutil
    ffmpeg_exe = shutil.which("ffmpeg")

    youtube_url = youtube_url.strip()
    if not is_valid_youtube_url(youtube_url):
        raise ValueError(f"Invalid YouTube URL: {youtube_url}. Please provide a valid YouTube link.")

    video_id = extract_video_id(youtube_url)
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    def progress_hook(d):
        if stage_callback and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            if total > 0:
                pct = min(100, max(0, int((downloaded / total) * 100)))
                stage_callback(f"Getting YouTube video ({pct}% downloaded)...")

    ydl_opts = {
        "format": "18/22/bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 2,
        "http_chunk_size": 10485760,
        "max_filesize": max_filesize_mb * 1024 * 1024,
        "progress_hooks": [progress_hook]
    }
    if ffmpeg_exe:
        ydl_opts["ffmpeg_location"] = ffmpeg_exe

    cookie_path = find_cookie_file()
    if cookie_path:
        ydl_opts["cookiefile"] = cookie_path

    if stage_callback:
        stage_callback("Extracting YouTube video metadata...")

    logger.info(f"Extracting YouTube metadata for URL: {youtube_url}")
    info = None
    download_success = False
    last_error_msg = ""

    # Strategy 1: Primary yt-dlp download (with cookiefile if available)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            duration = info.get("duration", 0)
            if duration and duration > max_duration_seconds:
                raise ValueError(
                    f"Video is too long ({duration/60:.1f} minutes). "
                    f"Maximum supported duration is {max_duration_seconds/60:.0f} minutes."
                )

            title = info.get("title", f"YouTube Video {video_id}")
            uploader = info.get("uploader") or info.get("channel") or "Unknown Creator"

            if stage_callback:
                stage_callback(f"Getting YouTube video: '{title[:30]}...' (0% downloaded)")

            logger.info(f"Downloading YouTube video: '{title}' ({duration}s)...")
            ydl.download([youtube_url])
            download_success = True
    except Exception as primary_err:
        last_error_msg = str(primary_err)
        logger.warning(f"Primary yt-dlp download attempt failed ({primary_err}). Trying fallback strategies...")

    # Strategy 2: Try browser cookies fallback if primary failed
    if not download_success:
        for browser_name in ["firefox", "chrome", "edge", "opera", "brave"]:
            try:
                cb_opts = dict(ydl_opts)
                cb_opts["cookiesfrombrowser"] = (browser_name,)
                with yt_dlp.YoutubeDL(cb_opts) as ydl_cb:
                    info = ydl_cb.extract_info(youtube_url, download=False)
                    title = info.get("title", f"YouTube Video {video_id}")
                    uploader = info.get("uploader") or info.get("channel") or "Unknown Creator"
                    duration = info.get("duration", 0)
                    ydl_cb.download([youtube_url])
                    download_success = True
                    logger.info(f"Successfully downloaded via browser cookies ({browser_name})!")
                    break
            except Exception as cb_err:
                logger.debug(f"Cookies fallback for {browser_name} failed: {cb_err}")

    # Strategy 3: pytubefix Fallback Engine if yt-dlp was blocked
    if not download_success:
        try:
            logger.info("Executing pytubefix fallback downloader...")
            if stage_callback:
                stage_callback("Getting YouTube video (fallback engine)...")
            from pytubefix import YouTube
            yt = YouTube(youtube_url)
            title = yt.title or f"YouTube Video {video_id}"
            duration = float(yt.length or 0)
            uploader = yt.author or "Unknown Creator"
            stream = (
                yt.streams.filter(progressive=True, file_extension="mp4").first()
                or yt.streams.filter(file_extension="mp4").first()
            )
            if stream:
                dl_file = stream.download(output_path=output_dir, filename=f"{video_id}.mp4")
                download_success = True
                logger.info(f"Pytubefix fallback successfully downloaded '{title}' to {dl_file}")
        except Exception as py_err:
            logger.error(f"Pytubefix fallback failed: {py_err}")

    # Locate the downloaded video file
    expected_mp4 = os.path.join(output_dir, f"{video_id}.mp4")
    if not os.path.isfile(expected_mp4):
        candidates = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith(video_id)]
        if candidates:
            expected_mp4 = candidates[0]
        else:
            if "Sign in to confirm you’re not a bot" in last_error_msg or "Sign in to confirm" in last_error_msg or "bot" in last_error_msg.lower():
                raise RuntimeError(
                    "YouTube requested bot verification for this video. "
                    "To fix: Export YouTube cookies from your browser into a 'cookies.txt' file in your project folder, "
                    "or close your open Chrome/Edge browser windows so yt-dlp can import your browser session automatically."
                )
            raise FileNotFoundError(f"Downloaded video file not found in {output_dir}. Error: {last_error_msg}")

    return {
        "video_path": os.path.abspath(expected_mp4),
        "title": title if info or 'title' in locals() else f"YouTube Video {video_id}",
        "duration": float(duration) if 'duration' in locals() and duration else 0.0,
        "uploader": uploader if 'uploader' in locals() else "Unknown Creator",
        "video_id": video_id,
        "webpage_url": f"https://www.youtube.com/watch?v={video_id}"
    }
