import requests
import sys
import glob
import os

url = "http://localhost:8000/api/deepfake/analyze"
videos = glob.glob(r"C:\Users\gamin\Downloads\videos_fake\*.mp4")
if not videos:
    print("No fake videos found in Downloads")
    sys.exit(1)

test_video = videos[0]
print(f"Uploading {os.path.basename(test_video)} to {url}...")
try:
    with open(test_video, "rb") as f:
        files = {"file": (os.path.basename(test_video), f, "video/mp4")}
        res = requests.post(url, files=files, timeout=120)
    print(f"Status Code: {res.status_code}")
    print("Response:", res.text)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
