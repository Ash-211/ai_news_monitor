import os
import sys
import glob
sys.path.append(".")
from src.intelligence.deepfake_detector import detect_deepfake_video

def test_batch(folder, label):
    videos = glob.glob(os.path.join(folder, "*.mp4"))[:3]
    print(f"\n--- Testing {len(videos)} videos from {folder} (Expected: {label}) ---")
    for v in videos:
        res = detect_deepfake_video(v, sample_rate=20) # use 20 to make it faster for testing
        print(f"File: {os.path.basename(v)} -> is_fake: {res.get('is_fake')}, label: {res.get('label')}, conf: {res.get('confidence')}")

if __name__ == "__main__":
    test_batch(r"C:\Users\gamin\Downloads\videos_fake", "Fake")
    test_batch(r"C:\Users\gamin\Downloads\videos_real", "Real")
