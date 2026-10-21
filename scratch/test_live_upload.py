import urllib.request
import urllib.parse
import json
import time
import os

base_url = "http://127.0.0.1:5000"

# Find a sample MP4 video in the workspace or data directory
test_mp4 = None
for root, dirs, files in os.walk("."):
    for f in files:
        if f.endswith(".mp4") and not f.startswith("short") and not f.startswith("reel"):
            test_mp4 = os.path.abspath(os.path.join(root, f))
            break
    if test_mp4:
        break

print("Found sample test MP4 video:", test_mp4)

if not test_mp4:
    print("No sample MP4 found in workspace.")
else:
    # Submit multipart upload
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    with open(test_mp4, "rb") as f:
        file_bytes = f.read()

    body = []
    body.append(f"--{boundary}".encode())
    body.append(f'Content-Disposition: form-data; name="video"; filename="{os.path.basename(test_mp4)}"'.encode())
    body.append(b"Content-Type: video/mp4")
    body.append(b"")
    body.append(file_bytes)
    
    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="upload_type"')
    body.append(b"")
    body.append(b"file")

    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="mode"')
    body.append(b"")
    body.append(b"multimodal")

    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="output_mode"')
    body.append(b"")
    body.append(b"reels")

    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="num_reels"')
    body.append(b"")
    body.append(b"auto")

    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="reel_length"')
    body.append(b"")
    body.append(b"30-60")

    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="aspect_ratio"')
    body.append(b"")
    body.append(b"9:16")

    body.append(f"--{boundary}--".encode())
    body.append(b"")

    payload = b"\r\n".join(body)

    req = urllib.request.Request(f"{base_url}/upload", data=payload, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

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
                    time.sleep(2)
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
