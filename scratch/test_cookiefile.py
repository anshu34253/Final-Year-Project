import yt_dlp
import os

url = "https://youtu.be/8xTyh6Y_3Ww"
out_dir = "data/youtube_downloads/test_cookiefile"
os.makedirs(out_dir, exist_ok=True)

cookie_file = "cookies.txt"
print("Does cookies.txt exist:", os.path.exists(cookie_file))

ydl_opts = {
    "format": "18/22/best[height<=480][ext=mp4]/best",
    "outtmpl": os.path.join(out_dir, "cookie_test.%(ext)s"),
    "quiet": False
}

if os.path.exists(cookie_file):
    ydl_opts["cookiefile"] = cookie_file

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        print("SUCCESS! Title:", info.get("title"))
except Exception as e:
    print("FAILED:", e)
