import yt_dlp
import os

url = "https://youtu.be/8xTyh6Y_3Ww"
out_dir = "data/youtube_downloads/test_oauth"
os.makedirs(out_dir, exist_ok=True)

print("--- TESTING YT-DLP TV / EMBEDDED EXTRACTOR ARGS ---")
try:
    ydl_opts = {
        "format": "18/22/best[height<=480][ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "tv_test.%(ext)s"),
        "extractor_args": {
            "youtube": {
                "player_client": ["tv", "web_embedded"]
            }
        },
        "quiet": False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        print("TV CLIENT SUCCESS! Title:", info.get("title"))
except Exception as e:
    print("TV CLIENT FAILED:", e)

print("\n--- TESTING YT-DLP WEB_EMBEDDED CLIENT ---")
try:
    ydl_opts = {
        "format": "18/22/best[height<=480][ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "web_embedded_test.%(ext)s"),
        "extractor_args": {
            "youtube": {
                "player_client": ["web_embedded"]
            }
        },
        "quiet": False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        print("WEB_EMBEDDED SUCCESS! Title:", info.get("title"))
except Exception as e:
    print("WEB_EMBEDDED FAILED:", e)
