import urllib.request
import json
import time

url = 'http://127.0.0.1:5000/api/upload'
payload = {'upload_type': 'youtube', 'youtube_url': 'https://www.youtube.com/watch?v=jNQXAC9IVRw', 'num_reels': '2'}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})

res = urllib.request.urlopen(req)
resp_data = json.loads(res.read().decode('utf-8'))
print('UPLOAD RESPONSE:', resp_data)

job_id = resp_data['job_id']
status_url = f'http://127.0.0.1:5000/api/status/{job_id}'

for i in range(12):
    time.sleep(1)
    s_res = urllib.request.urlopen(status_url)
    s_data = json.loads(s_res.read().decode('utf-8'))
    print(f'[POLL {i+1}]: stage = "{s_data.get("stage")}", status = "{s_data.get("status")}"')
