import os
import shutil
import tempfile
import yt_dlp

user_profile = os.environ.get('USERPROFILE', '')

chrome_cookie_path = os.path.join(user_profile, 'AppData', 'Local', 'Google', 'Chrome', 'User Data', 'Default', 'Network', 'Cookies')
edge_cookie_path = os.path.join(user_profile, 'AppData', 'Local', 'Microsoft', 'Edge', 'User Data', 'Default', 'Network', 'Cookies')

print("Chrome cookie path exists:", os.path.exists(chrome_cookie_path))
print("Edge cookie path exists:", os.path.exists(edge_cookie_path))

# Test copying Chrome/Edge cookies to temp directory to bypass browser file locks
temp_dir = tempfile.mkdtemp()
print("Temp dir created:", temp_dir)

copied_path = None
if os.path.exists(edge_cookie_path):
    try:
        dst = os.path.join(temp_dir, "Edge_Cookies")
        # Use powershell Copy-Item which handles open file sharing on Windows
        os.system(f'powershell -Command "Copy-Item -Path \'{edge_cookie_path}\' -Destination \'{dst}\' -Force"')
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print("Edge cookies copied successfully! Size:", os.path.getsize(dst))
            copied_path = dst
    except Exception as e:
        print("Edge copy failed:", e)

if not copied_path and os.path.exists(chrome_cookie_path):
    try:
        dst = os.path.join(temp_dir, "Chrome_Cookies")
        os.system(f'powershell -Command "Copy-Item -Path \'{chrome_cookie_path}\' -Destination \'{dst}\' -Force"')
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print("Chrome cookies copied successfully! Size:", os.path.getsize(dst))
            copied_path = dst
    except Exception as e:
        print("Chrome copy failed:", e)

# Test yt-dlp with copied cookie database or client flags
url = "https://youtu.be/8xTyh6Y_3Ww"
out_dir = "data/youtube_downloads/test_copy_cookies"
os.makedirs(out_dir, exist_ok=True)

# Test iOS client spoofing (iOS client bypasses bot check on most YouTube URLs)
print("\n--- TESTING IOS CLIENT SPOOFING ---")
try:
    ydl_opts = {
        "format": "18/22/best[height<=480][ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "ios_test.%(ext)s"),
        "extractor_args": {
            "youtube": {
                "player_client": ["ios"]
            }
        },
        "quiet": False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        print("IOS CLIENT SUCCESS! Title:", info.get("title"))
except Exception as e:
    print("IOS CLIENT FAILED:", e)

# Test android_vr client spoofing
print("\n--- TESTING ANDROID_VR CLIENT SPOOFING ---")
try:
    ydl_opts = {
        "format": "18/22/best[height<=480][ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "android_vr_test.%(ext)s"),
        "extractor_args": {
            "youtube": {
                "player_client": ["android_vr"]
            }
        },
        "quiet": False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        print("ANDROID_VR CLIENT SUCCESS! Title:", info.get("title"))
except Exception as e:
    print("ANDROID_VR CLIENT FAILED:", e)

# Clean up temp dir
shutil.rmtree(temp_dir, ignore_errors=True)
