"""
Benchmark v3: Temporal consistency analysis.
Deepfakes have MORE frame-to-frame variance in model scores and face position.
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
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    return faces[0]  # (x, y, w, h)


def classify_bgr(img_bgr, face_rect=None):
    """Classify an image region. Returns fake_prob."""
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
    return float(probs[0])  # fake prob


def analyze_video_temporal(path, sample_rate=5):
    """Extract per-frame scores and face positions for temporal analysis."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Cap frames analyzed
    MAX_FRAMES = 30
    effective_rate = max(sample_rate, total // MAX_FRAMES)
    
    frame_idx = 0
    face_scores = []     # face-crop model scores per frame
    full_scores = []     # full-frame model scores per frame
    face_positions = []  # (cx, cy, w, h) per frame
    face_found_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % effective_rate == 0:
            face = get_face_rect(frame)
            
            # Full frame score
            full_fake = classify_bgr(frame)
            full_scores.append(full_fake)
            
            if face is not None:
                # Face-crop score
                face_fake = classify_bgr(frame, face)
                face_scores.append(face_fake)
                
                x, y, w, h = face
                cx, cy = x + w/2, y + h/2
                face_positions.append((cx, cy, w, h))
                face_found_count += 1
            else:
                face_scores.append(full_fake)  # fallback
        
        frame_idx += 1
    cap.release()
    
    if len(full_scores) < 3:
        return None
    
    full_arr = np.array(full_scores)
    face_arr = np.array(face_scores)
    
    # ── Temporal signals ──────────────────────────────────────────────
    
    # 1. Score variance (deepfakes have more variable per-frame scores)
    face_score_std = face_arr.std()
    full_score_std = full_arr.std()
    
    # 2. Face position jitter (deepfakes have jittery face tracking)
    face_jitter = 0
    if len(face_positions) >= 3:
        pos_arr = np.array(face_positions)
        # Compute frame-to-frame position deltas normalized by face size
        deltas = np.diff(pos_arr[:, :2], axis=0)  # cx, cy changes
        face_sizes = pos_arr[:-1, 2]  # w of face
        normalized_deltas = np.sqrt((deltas**2).sum(axis=1)) / (face_sizes + 1e-8)
        face_jitter = normalized_deltas.std()  # std of normalized movement
    
    # 3. Face size consistency (deepfakes have variable face sizes)
    face_size_var = 0
    if len(face_positions) >= 3:
        sizes = np.array([p[2] * p[3] for p in face_positions])
        face_size_var = sizes.std() / (sizes.mean() + 1e-8)
    
    # 4. Score difference pattern: how much does face-crop differ from full-frame?
    diff = full_arr - face_arr
    avg_diff = diff.mean()  # reals tend to have HIGHER diff (face looks more real than full frame)
    
    # 5. Mean scores
    face_mean = face_arr.mean()
    full_mean = full_arr.mean()
    
    return {
        "face_mean": round(face_mean, 4),
        "full_mean": round(full_mean, 4),
        "face_std": round(face_score_std, 4),
        "full_std": round(full_score_std, 4),
        "face_jitter": round(face_jitter, 4),
        "face_size_var": round(face_size_var, 4),
        "score_diff": round(avg_diff, 4),
        "frames": len(full_scores),
        "faces_found": face_found_count,
    }


# ── Benchmark ─────────────────────────────────────────────────────────────
fake_dir = r"C:\Users\gamin\Downloads\videos_fake"
real_dir = r"C:\Users\gamin\Downloads\videos_real"

fake_videos = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))[:10]
real_videos = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))[:10]

print(f"\n{'='*100}")
print(f"TEMPORAL ANALYSIS: {len(fake_videos)} fake + {len(real_videos)} real videos")
print(f"{'='*100}")

header = f"  {'File':<12} {'FaceMn':>6} {'FullMn':>6} {'FaceSD':>6} {'FullSD':>6} {'Jitter':>6} {'SzVar':>6} {'Diff':>6}"

print(f"\n--- FAKE VIDEOS ---")
print(header)
fake_results = []
for v in fake_videos:
    t0 = time.time()
    r = analyze_video_temporal(v)
    dt = time.time() - t0
    if r is None: continue
    fake_results.append(r)
    print(f"  {os.path.basename(v):<12} {r['face_mean']:.4f} {r['full_mean']:.4f} {r['face_std']:.4f} {r['full_std']:.4f} {r['face_jitter']:.4f} {r['face_size_var']:.4f} {r['score_diff']:.4f}  [{dt:.1f}s]")

print(f"\n--- REAL VIDEOS ---")
print(header)
real_results = []
for v in real_videos:
    t0 = time.time()
    r = analyze_video_temporal(v)
    dt = time.time() - t0
    if r is None: continue
    real_results.append(r)
    print(f"  {os.path.basename(v):<12} {r['face_mean']:.4f} {r['full_mean']:.4f} {r['face_std']:.4f} {r['full_std']:.4f} {r['face_jitter']:.4f} {r['face_size_var']:.4f} {r['score_diff']:.4f}  [{dt:.1f}s]")

# ── Statistical comparison ────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"SIGNAL ANALYSIS (mean +/- std for each group)")
print(f"{'='*100}")

for signal in ['face_mean', 'full_mean', 'face_std', 'full_std', 'face_jitter', 'face_size_var', 'score_diff']:
    fake_vals = [r[signal] for r in fake_results]
    real_vals = [r[signal] for r in real_results]
    f_mean, f_std = np.mean(fake_vals), np.std(fake_vals)
    r_mean, r_std = np.mean(real_vals), np.std(real_vals)
    sep = abs(f_mean - r_mean) / (max(f_std, r_std) + 1e-8)
    direction = "FAKE>REAL" if f_mean > r_mean else "REAL>FAKE"
    quality = "***" if sep > 1.0 else "**" if sep > 0.5 else "*" if sep > 0.3 else ""
    print(f"  {signal:<14}: Fake={f_mean:.4f}+/-{f_std:.4f}  Real={r_mean:.4f}+/-{r_std:.4f}  Sep={sep:.2f} {direction} {quality}")
