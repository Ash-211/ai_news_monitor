"""
Comprehensive benchmark of the ensemble deepfake detection approach.
Tests original model with and without face cropping, frequency analysis,
and combined ensemble on the user's test dataset.
"""
import os
import sys
import glob
import time
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, ".")

MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"

print("Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()
id2label = model.config.id2label
print(f"Labels: {id2label}")


def get_face_crop(img_bgr):
    """Crop to largest face. Returns cropped BGR image or None if no face found."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    margin = int(w * 0.3)  # slightly larger margin
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img_bgr.shape[1], x + w + margin), min(img_bgr.shape[0], y + h + margin)
    return img_bgr[y1:y2, x1:x2]


def classify_pil(pil_image):
    """Run the model on a PIL image, return (fake_prob, real_prob)."""
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.nn.functional.softmax(out.logits, dim=-1)[0]
    # id2label: {0: 'Fake', 1: 'Real'}
    return float(probs[0]), float(probs[1])


def frequency_analysis(img_bgr):
    """
    Analyze frequency-domain artifacts that indicate AI generation or face blending.
    Returns a 'fakeness' score from 0-1.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. FFT analysis - deepfakes often have unusual frequency patterns
    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log(1 + np.abs(f_shift))
    
    # Normalize
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    # High-frequency energy ratio (deepfakes often have less high-freq detail)
    r_mid = min(h, w) // 4
    r_high = min(h, w) // 2
    
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    mid_energy = magnitude[(dist >= r_mid) & (dist < r_high)].mean()
    low_energy = magnitude[dist < r_mid].mean()
    
    # Ratio of mid-to-low frequency energy
    freq_ratio = mid_energy / (low_energy + 1e-8)
    
    # 2. Laplacian variance (blur detection - deepfakes are often slightly blurrier)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: very sharp images have high variance
    blur_score = 1.0 / (1.0 + laplacian_var / 500.0)
    
    # 3. Color channel consistency check
    b, g, r = cv2.split(img_bgr)
    # Cross-channel correlation - deepfakes sometimes have inconsistent color channels
    rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
    rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
    gb_corr = np.corrcoef(g.flatten(), b.flatten())[0, 1]
    
    # High correlation between all channels is more natural
    color_consistency = (rg_corr + rb_corr + gb_corr) / 3.0
    color_anomaly = max(0, 1.0 - color_consistency)
    
    # Combine signals
    freq_score = max(0, min(1, 1.0 - freq_ratio * 2))  # Lower ratio = more fake
    
    # Weighted combination
    fakeness = 0.4 * freq_score + 0.35 * blur_score + 0.25 * color_anomaly
    return round(fakeness, 4)


def analyze_frame(img_bgr):
    """
    Run ensemble analysis on a single BGR frame.
    Returns dict with individual and combined scores.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_full = Image.fromarray(rgb)
    
    # 1. Full-frame model prediction
    fake_full, real_full = classify_pil(pil_full)
    
    # 2. Face-cropped model prediction
    face = get_face_crop(img_bgr)
    if face is not None:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)
        fake_face, real_face = classify_pil(pil_face)
        has_face = True
    else:
        fake_face, real_face = fake_full, real_full  # fallback
        has_face = False
    
    # 3. Frequency analysis
    freq_fakeness = frequency_analysis(face if face is not None else img_bgr)
    
    # 4. Ensemble: combine signals
    # Full-frame model is good at catching fakes but flags too many reals as fake
    # Face-crop model is good at confirming reals but misses some fakes
    # Use full-frame as primary, face-crop as correction, freq as tiebreaker
    
    if has_face:
        # Weighted ensemble with both signals
        ensemble_fake = (
            0.45 * fake_full +      # full-frame catches fakes well
            0.35 * fake_face +      # face-crop is more conservative
            0.20 * freq_fakeness    # frequency as supporting signal
        )
    else:
        # No face detected - rely more on full frame + frequency
        ensemble_fake = (
            0.60 * fake_full +
            0.40 * freq_fakeness
        )
    
    ensemble_real = 1.0 - ensemble_fake
    
    return {
        "full_frame": {"fake": fake_full, "real": real_full},
        "face_crop": {"fake": fake_face, "real": real_face, "has_face": has_face},
        "frequency": freq_fakeness,
        "ensemble": {"fake": round(ensemble_fake, 4), "real": round(ensemble_real, 4)},
    }


def test_video(path, sample_rate=20):
    """Analyze a video and return the ensemble verdict."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            r = analyze_frame(frame)
            results.append(r)
        frame_idx += 1
    cap.release()
    
    if not results:
        return None
    
    avg_fake = np.mean([r["ensemble"]["fake"] for r in results])
    avg_real = np.mean([r["ensemble"]["real"] for r in results])
    avg_full_fake = np.mean([r["full_frame"]["fake"] for r in results])
    avg_face_fake = np.mean([r["face_crop"]["fake"] for r in results])
    avg_freq = np.mean([r["frequency"] for r in results])
    
    return {
        "is_fake": avg_fake > 0.5,
        "ensemble_fake": round(avg_fake, 4),
        "ensemble_real": round(avg_real, 4),
        "full_frame_fake": round(avg_full_fake, 4),
        "face_crop_fake": round(avg_face_fake, 4),
        "freq_score": round(avg_freq, 4),
        "frames_analyzed": len(results),
    }


# ── Run benchmark ─────────────────────────────────────────────────────────
fake_dir = r"C:\Users\gamin\Downloads\videos_fake"
real_dir = r"C:\Users\gamin\Downloads\videos_real"

fake_videos = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))[:10]
real_videos = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))[:10]

print(f"\n{'='*80}")
print(f"BENCHMARK: Testing {len(fake_videos)} fake + {len(real_videos)} real videos")
print(f"{'='*80}")

correct_fake = 0
correct_real = 0

print(f"\n--- FAKE VIDEOS (should detect as Fake) ---")
for v in fake_videos:
    t0 = time.time()
    r = test_video(v)
    dt = time.time() - t0
    if r is None:
        print(f"  {os.path.basename(v)}: SKIPPED")
        continue
    verdict = "FAKE" if r["is_fake"] else "REAL"
    correct = "OK" if r["is_fake"] else "XX"
    if r["is_fake"]:
        correct_fake += 1
    print(f"  {correct} {os.path.basename(v)}: {verdict} (ens={r['ensemble_fake']:.3f}, full={r['full_frame_fake']:.3f}, face={r['face_crop_fake']:.3f}, freq={r['freq_score']:.3f}) [{dt:.1f}s]")

print(f"\n--- REAL VIDEOS (should detect as Real) ---")
for v in real_videos:
    t0 = time.time()
    r = test_video(v)
    dt = time.time() - t0
    if r is None:
        print(f"  {os.path.basename(v)}: SKIPPED")
        continue
    verdict = "FAKE" if r["is_fake"] else "REAL"
    correct = "OK" if not r["is_fake"] else "XX"
    if not r["is_fake"]:
        correct_real += 1
    print(f"  {correct} {os.path.basename(v)}: {verdict} (ens={r['ensemble_fake']:.3f}, full={r['full_frame_fake']:.3f}, face={r['face_crop_fake']:.3f}, freq={r['freq_score']:.3f}) [{dt:.1f}s]")

total = len(fake_videos) + len(real_videos)
correct_total = correct_fake + correct_real
print(f"\n{'='*80}")
print(f"RESULTS: {correct_total}/{total} correct ({100*correct_total/total:.1f}%)")
print(f"  Fake accuracy:  {correct_fake}/{len(fake_videos)} ({100*correct_fake/len(fake_videos):.1f}%)")
print(f"  Real accuracy:  {correct_real}/{len(real_videos)} ({100*correct_real/len(real_videos):.1f}%)")
print(f"{'='*80}")
