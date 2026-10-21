import sys
import os
import time

url = "https://youtu.be/8xTyh6Y_3Ww"
out_dir = "data/youtube_downloads/test_yt"
os.makedirs(out_dir, exist_ok=True)

print("--- TESTING YT-DLP CLIENT SPOOFING ---")
import yt_dlp

clients_to_test = [
    ["mweb", "android"],
    ["android", "ios"],
    ["web_creator", "mweb"],
    ["tv", "android"]
]

success = False
for clients in clients_to_test:
    print(f"\nTrying extractor_args player_client: {clients}")
    ydl_opts = {
        "format": "18/22/best[height<=480][ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "yt_test.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "extractor_args": {
            "youtube": {
                "player_client": clients
            }
        }
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            print("SUCCESS with client:", clients, "| Title:", info.get("title"))
            success = True
            break
    except Exception as e:
        print(f"FAILED with client {clients}: {e}")

if not success:
    print("\n--- TESTING PYTUBEFIX ---")
    try:
        from pytubefix import YouTube
        yt = YouTube(url)
        print("Pytubefix title:", yt.title)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
        if stream:
            f = stream.download(output_path=out_dir, filename="pytube_test.mp4")
            print("Pytubefix success! File:", f)
            success = True
    except Exception as py_err:
        print("Pytubefix failed:", py_err)

if success:
    print("\nOVERALL TEST RESULT: PASSED")
else:
    print("\nOVERALL TEST RESULT: FAILED")
