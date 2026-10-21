import time
import os
import sys
import yt_dlp

url = "https://www.youtube.com/watch?v=akI8YFjEmUw"
out_dir = r"c:\Users\hp\OneDrive\Desktop\viral_shorts1\scratch\speed_tests"
os.makedirs(out_dir, exist_ok=True)

configs = [
    {
        "name": "Config A: android client",
        "opts": {
            "format": "bestvideo[height<=720]+bestaudio/best",
            "outtmpl": os.path.join(out_dir, "test_a.%(ext)s"),
            "quiet": True,
            "nocheckcertificate": True,
            "extractor_args": {"youtube": {"player_client": ["android"]}}
        }
    },
    {
        "name": "Config B: ios client",
        "opts": {
            "format": "bestvideo[height<=720]+bestaudio/best",
            "outtmpl": os.path.join(out_dir, "test_b.%(ext)s"),
            "quiet": True,
            "nocheckcertificate": True,
            "extractor_args": {"youtube": {"player_client": ["ios"]}}
        }
    },
    {
        "name": "Config C: format 18/22 (single stream mp4)",
        "opts": {
            "format": "18/22/best",
            "outtmpl": os.path.join(out_dir, "test_c.%(ext)s"),
            "quiet": True,
            "nocheckcertificate": True,
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}}
        }
    },
    {
        "name": "Config D: mweb client",
        "opts": {
            "format": "bestvideo[height<=720]+bestaudio/best",
            "outtmpl": os.path.join(out_dir, "test_d.%(ext)s"),
            "quiet": True,
            "nocheckcertificate": True,
            "extractor_args": {"youtube": {"player_client": ["mweb"]}}
        }
    }
]

for cfg in configs:
    print(f"\n--- Testing {cfg['name']} ---")
    t0 = time.time()
    try:
        with yt_dlp.YoutubeDL(cfg["opts"]) as ydl:
            ydl.download([url])
        dur = time.time() - t0
        print(f"--> {cfg['name']} DOWNLOAD SUCCESS in {dur:.2f} seconds!")
    except Exception as e:
        print(f"--> {cfg['name']} FAILED: {e}")
