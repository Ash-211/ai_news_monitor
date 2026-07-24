import sys
sys.path.insert(0, ".")
from src.intelligence.deepfake_detector import detect_deepfake_video

video_path = r"C:\Users\gamin\Downloads\videos_fake\vs1.mp4"
try:
    print(f"Testing {video_path}")
    res = detect_deepfake_video(video_path)
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
