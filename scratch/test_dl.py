import time
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.youtube_downloader import download_youtube_video

t0 = time.time()
print('Testing YouTube video download speed...')
res = download_youtube_video('https://www.youtube.com/watch?v=akI8YFjEmUw', 'data/youtube_downloads/test_fast')
print(f"Downloaded in {time.time() - t0:.2f} seconds! Path: {res['video_path']}")
