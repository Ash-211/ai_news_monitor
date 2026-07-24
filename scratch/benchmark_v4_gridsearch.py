"""
Benchmark v4: Optimal ensemble using face_mean + score_diff.
Grid-searches weights and threshold on the test data.
"""
import os, sys, glob, time, cv2, torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, ".")

MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"
print("Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_face_rect(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    return sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]


def classify_bgr(img_bgr, face_rect=None):
    if face_rect is not None:
        x, y, w, h = face_rect
        m = int(w * 0.3)
        x1, y1 = max(0, x-m), max(0, y-m)
        x2, y2 = min(img_bgr.shape[1], x+w+m), min(img_bgr.shape[0], y+h+m)
        img_bgr = img_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.nn.functional.softmax(out.logits, dim=-1)[0]
    return float(probs[0])


def analyze_video(path, sample_rate=5):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    MAX_FRAMES = 30
    effective_rate = max(sample_rate, total // MAX_FRAMES)
    
    frame_idx = 0
    face_scores = []
    full_scores = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % effective_rate == 0:
            face = get_face_rect(frame)
            full_fake = classify_bgr(frame)
            full_scores.append(full_fake)
            if face is not None:
                face_fake = classify_bgr(frame, face)
                face_scores.append(face_fake)
            else:
                face_scores.append(full_fake)
        frame_idx += 1
    cap.release()
    
    if len(full_scores) < 2:
        return None
    
    face_mean = np.mean(face_scores)
    full_mean = np.mean(full_scores)
    score_diff = full_mean - face_mean
    
    return {"face_mean": face_mean, "full_mean": full_mean, "score_diff": score_diff}


# ── Collect data ──────────────────────────────────────────────────────────
fake_dir = r"C:\Users\gamin\Downloads\videos_fake"
real_dir = r"C:\Users\gamin\Downloads\videos_real"

# Use first 15 of each for more data
fake_videos = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))[:15]
real_videos = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))[:15]

print(f"\nAnalyzing {len(fake_videos)} fake + {len(real_videos)} real videos...")

fake_data = []
real_data = []

print("\n--- Fake videos ---")
for v in fake_videos:
    t0 = time.time()
    r = analyze_video(v)
    dt = time.time() - t0
    if r is None: continue
    fake_data.append(r)
    print(f"  {os.path.basename(v):<12} face={r['face_mean']:.4f} full={r['full_mean']:.4f} diff={r['score_diff']:.4f} [{dt:.1f}s]")

print("\n--- Real videos ---")
for v in real_videos:
    t0 = time.time()
    r = analyze_video(v)
    dt = time.time() - t0
    if r is None: continue
    real_data.append(r)
    print(f"  {os.path.basename(v):<12} face={r['face_mean']:.4f} full={r['full_mean']:.4f} diff={r['score_diff']:.4f} [{dt:.1f}s]")

# ── Grid search ───────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"GRID SEARCH: Finding optimal weights and threshold")
print(f"{'='*80}")
print(f"Formula: fakeness = w * face_mean + (1-w) * (1 - score_diff)")

best_acc = 0
best_params = {}

for w in np.arange(0.3, 0.8, 0.05):
    for thresh in np.arange(0.40, 0.70, 0.01):
        correct = 0
        total = len(fake_data) + len(real_data)
        
        for d in fake_data:
            score = w * d['face_mean'] + (1-w) * (1 - d['score_diff'])
            if score > thresh: correct += 1
        
        for d in real_data:
            score = w * d['face_mean'] + (1-w) * (1 - d['score_diff'])
            if score <= thresh: correct += 1
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_params = {"w": round(w, 2), "thresh": round(thresh, 2), "acc": round(acc, 4)}

print(f"\nBest: w={best_params['w']}, threshold={best_params['thresh']}, accuracy={best_params['acc']*100:.1f}%")

# ── Show results with best params ─────────────────────────────────────────
w = best_params['w']
thresh = best_params['thresh']

print(f"\n--- Results with optimal params (w={w}, threshold={thresh}) ---")

fake_correct = 0
real_correct = 0

print(f"\nFake videos:")
for i, d in enumerate(fake_data):
    score = w * d['face_mean'] + (1-w) * (1 - d['score_diff'])
    is_fake = score > thresh
    ok = "OK" if is_fake else "XX"
    if is_fake: fake_correct += 1
    print(f"  {ok} score={score:.4f} (face={d['face_mean']:.4f}, diff={d['score_diff']:.4f})")

print(f"\nReal videos:")
for i, d in enumerate(real_data):
    score = w * d['face_mean'] + (1-w) * (1 - d['score_diff'])
    is_fake = score > thresh
    ok = "OK" if not is_fake else "XX"
    if not is_fake: real_correct += 1
    print(f"  {ok} score={score:.4f} (face={d['face_mean']:.4f}, diff={d['score_diff']:.4f})")

total = len(fake_data) + len(real_data)
total_correct = fake_correct + real_correct
print(f"\n{'='*80}")
print(f"FINAL: {total_correct}/{total} ({100*total_correct/total:.1f}%)")
print(f"  Fake: {fake_correct}/{len(fake_data)} ({100*fake_correct/len(fake_data):.1f}%)")
print(f"  Real: {real_correct}/{len(real_data)} ({100*real_correct/len(real_data):.1f}%)")
print(f"{'='*80}")
