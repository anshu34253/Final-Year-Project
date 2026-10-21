import os
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

from main import run_multi_reel_pipeline

video_path = os.path.join(BASE_DIR, "data", "youtube_downloads", "test_fast", "akI8YFjEmUw.mp4")
out_dir = os.path.join(BASE_DIR, "web_outputs", "test_pipeline_run")

print(f"Running end-to-end multi-reel pipeline on: {video_path}")
t0 = time.time()

def stage_cb(msg):
    print(f"[STAGE CALLBACK] {msg}")

res = run_multi_reel_pipeline(
    video_path=video_path,
    youtube_url=None,
    model_mode="multimodal",
    num_reels="3",
    reel_length="30-60",
    aspect_ratio="9:16",
    captions=True,
    output_dir=out_dir,
    stage_callback=stage_cb
)

print(f"PIPELINE SUCCESS in {time.time() - t0:.2f} seconds!")
print(f"Num Reels Generated: {res['num_reels']}")
for r in res['reels']:
    print(f" - Reel {r['reel_num']}: {r['start_formatted']} -> {r['end_formatted']} ({r['file_size_mb']} MB, filename: {r['filename']})")
