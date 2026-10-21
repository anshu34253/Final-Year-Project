import urllib.request
import urllib.parse
import json
import time

base_url = "http://127.0.0.1:5000"
yt_url = "https://youtu.be/8xTyh6Y_3Ww"

print(f"Submitting URL {yt_url} to {base_url}/upload ...")
data = urllib.parse.urlencode({
    "upload_type": "youtube",
    "youtube_url": yt_url,
    "mode": "multimodal",
    "output_mode": "reels",
    "num_reels": "auto",
    "reel_length": "30-60",
    "aspect_ratio": "9:16"
}).encode("utf-8")

req = urllib.request.Request(f"{base_url}/upload", data=data, method="POST")
try:
    with urllib.request.urlopen(req) as resp:
        redirect_url = resp.geturl()
        print("Response URL:", redirect_url)
        if "/processing/" in redirect_url:
            job_id = redirect_url.split("/processing/")[-1]
            print("Job ID:", job_id)
            
            # Poll status API
            start_time = time.time()
            while True:
                time.sleep(3)
                status_req = urllib.request.Request(f"{base_url}/api/status/{job_id}")
                with urllib.request.urlopen(status_req) as s_resp:
                    status_data = json.loads(s_resp.read().decode("utf-8"))
                    elapsed = round(time.time() - start_time, 1)
                    print(f"[{elapsed}s] [{status_data.get('status')}] {status_data.get('stage')}")
                    if status_data.get("status") in ["complete", "error", "failed"]:
                        print("FINAL RESULT DATA:", json.dumps(status_data, indent=2))
                        break
        else:
            print("Unexpected response redirect:", redirect_url)
except Exception as e:
    print("Request failed:", e)
