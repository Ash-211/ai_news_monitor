"""Quick validation: test the production deepfake_detector.py on a few files."""
import os, sys, glob
sys.path.insert(0, ".")
from src.intelligence.deepfake_detector import detect_deepfake_video

fake_dir = r"C:\Users\gamin\Downloads\videos_fake"
real_dir = r"C:\Users\gamin\Downloads\videos_real"

fake_videos = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))[:5]
real_videos = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))[:5]

print("--- FAKE (expect Fake) ---")
for v in fake_videos:
    r = detect_deepfake_video(v, sample_rate=10)
    ok = "OK" if r["is_fake"] else "XX"
    print(f"  {ok} {os.path.basename(v)}: {r['label']} conf={r['confidence']:.3f} band={r['confidence_band']} ens={r['raw_scores']['Fake']:.3f}")

print("\n--- REAL (expect Real) ---")
for v in real_videos:
    r = detect_deepfake_video(v, sample_rate=10)
    ok = "OK" if not r["is_fake"] else "XX"
    print(f"  {ok} {os.path.basename(v)}: {r['label']} conf={r['confidence']:.3f} band={r['confidence_band']} ens={r['raw_scores']['Fake']:.3f}")
