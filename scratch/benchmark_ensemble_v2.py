"""
Improved ensemble benchmark v2 - adds forensic face-boundary and noise
analysis, and properly calibrates using the face-crop signal.
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


def get_face_region(img_bgr):
    """Returns (face_crop, face_mask, face_rect) or (None, None, None)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None, None, None
    
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    margin = int(w * 0.3)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img_bgr.shape[1], x + w + margin), min(img_bgr.shape[0], y + h + margin)
    
    face_crop = img_bgr[y1:y2, x1:x2]
    
    # Create binary mask for face region (elliptical)
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cx, cy = x + w // 2, y + h // 2
    cv2.ellipse(mask, (cx, cy), (w // 2, int(h * 0.6)), 0, 0, 360, 255, -1)
    
    return face_crop, mask, (x, y, w, h)


def classify_pil(pil_image):
    """Run model -> (fake_prob, real_prob)."""
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.nn.functional.softmax(out.logits, dim=-1)[0]
    return float(probs[0]), float(probs[1])


def face_boundary_analysis(img_bgr, face_mask):
    """
    Analyze the boundary where a swapped face meets the original image.
    Deepfakes have smoother/blurrier boundaries due to blending.
    Returns fakeness score 0-1.
    """
    if face_mask is None:
        return 0.5  # neutral
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Create boundary strip: dilated mask - eroded mask
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(face_mask, kernel, iterations=3)
    eroded = cv2.erode(face_mask, kernel, iterations=3)
    boundary = cv2.subtract(dilated, eroded)
    
    if boundary.sum() == 0:
        return 0.5
    
    # Compute gradient magnitude along the boundary
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Average gradient at boundary vs. inside face
    boundary_grad = grad_mag[boundary > 0].mean() if (boundary > 0).any() else 0
    inside_grad = grad_mag[eroded > 0].mean() if (eroded > 0).any() else 1
    
    # In deepfakes, boundary gradients are often LOWER (smoother blending)
    # relative to interior detail. Genuine faces have natural transitions.
    ratio = boundary_grad / (inside_grad + 1e-8)
    
    # High ratio = sharp boundary (more natural), Low ratio = smooth blend (more fake)
    # Typical range: 0.5 to 3.0
    # Map to fakeness: lower ratio = more likely fake
    fakeness = max(0, min(1, 1.0 - (ratio - 0.5) / 2.5))
    
    return round(fakeness, 4)


def noise_inconsistency(img_bgr, face_mask):
    """
    Check if the noise pattern in the face region differs from the background.
    Face-swapped images have different noise characteristics in swapped regions.
    Returns fakeness score 0-1.
    """
    if face_mask is None:
        return 0.5
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    # High-pass filter to isolate noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blurred
    
    # Noise statistics inside vs outside face
    face_pixels = noise[face_mask > 0]
    bg_mask = cv2.bitwise_not(face_mask)
    bg_pixels = noise[bg_mask > 0]
    
    if len(face_pixels) < 100 or len(bg_pixels) < 100:
        return 0.5
    
    face_std = face_pixels.std()
    bg_std = bg_pixels.std()
    face_mean = abs(face_pixels.mean())
    bg_mean = abs(bg_pixels.mean())
    
    # Noise level difference - in genuine images, noise is more uniform
    std_diff = abs(face_std - bg_std) / (max(face_std, bg_std) + 1e-8)
    mean_diff = abs(face_mean - bg_mean) / (max(face_mean, bg_mean) + 1e-8)
    
    # Higher difference = more likely manipulated
    fakeness = min(1.0, (std_diff * 0.6 + mean_diff * 0.4) * 2.0)
    
    return round(fakeness, 4)


def error_level_analysis(img_bgr, face_mask):
    """
    Error Level Analysis (ELA): Re-save at known JPEG quality, compare
    error patterns. Manipulated regions show different error levels.
    Returns fakeness score 0-1.
    """
    if face_mask is None:
        return 0.5
    
    # Encode as JPEG at quality 90, then decode
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, encoded = cv2.imencode('.jpg', img_bgr, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    # ELA = absolute difference between original and re-saved
    ela = cv2.absdiff(img_bgr, decoded).astype(np.float64)
    ela_gray = cv2.cvtColor(ela.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    # Compare ELA levels inside face vs background
    face_ela = ela_gray[face_mask > 0]
    bg_mask = cv2.bitwise_not(face_mask)
    bg_ela = ela_gray[bg_mask > 0]
    
    if len(face_ela) < 100 or len(bg_ela) < 100:
        return 0.5
    
    face_ela_mean = face_ela.mean()
    bg_ela_mean = bg_ela.mean()
    
    # Different ELA levels between face and background indicate manipulation
    ela_diff = abs(face_ela_mean - bg_ela_mean) / (max(face_ela_mean, bg_ela_mean) + 1e-8)
    
    fakeness = min(1.0, ela_diff * 3.0)
    return round(fakeness, 4)


def analyze_frame(img_bgr):
    """Ensemble analysis on a single BGR frame."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_full = Image.fromarray(rgb)
    
    face_crop, face_mask, face_rect = get_face_region(img_bgr)
    
    # 1. Full-frame model
    fake_full, real_full = classify_pil(pil_full)
    
    # 2. Face-crop model
    if face_crop is not None:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)
        fake_face, real_face = classify_pil(pil_face)
        has_face = True
    else:
        fake_face, real_face = fake_full, real_full
        has_face = False
    
    # 3. Forensic signals
    boundary_score = face_boundary_analysis(img_bgr, face_mask)
    noise_score = noise_inconsistency(img_bgr, face_mask)
    ela_score = error_level_analysis(img_bgr, face_mask)
    
    # 4. Calibrated ensemble
    # Key insight from v1 benchmark: face_crop is the most discriminating signal.
    # full_frame is biased toward Fake but useful as a secondary signal.
    # Forensic signals add independent information.
    if has_face:
        ensemble_fake = (
            0.15 * fake_full +         # full-frame (biased, downweight it)
            0.40 * fake_face +         # face-crop (most discriminating)
            0.15 * boundary_score +    # face boundary artifacts
            0.15 * noise_score +       # noise inconsistency
            0.15 * ela_score           # error level analysis
        )
    else:
        ensemble_fake = (
            0.40 * fake_full +
            0.20 * boundary_score +
            0.20 * noise_score +
            0.20 * ela_score
        )
    
    return {
        "full_fake": fake_full,
        "face_fake": fake_face,
        "has_face": has_face,
        "boundary": boundary_score,
        "noise": noise_score,
        "ela": ela_score,
        "ensemble_fake": round(ensemble_fake, 4),
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
    
    avg = lambda key: round(np.mean([r[key] for r in results]), 4)
    
    return {
        "is_fake": avg("ensemble_fake") > 0.5,
        "ens": avg("ensemble_fake"),
        "full": avg("full_fake"),
        "face": avg("face_fake"),
        "bnd": avg("boundary"),
        "noi": avg("noise"),
        "ela": avg("ela"),
        "frames": len(results),
    }


# ── Benchmark ─────────────────────────────────────────────────────────────
fake_dir = r"C:\Users\gamin\Downloads\videos_fake"
real_dir = r"C:\Users\gamin\Downloads\videos_real"

fake_videos = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))[:10]
real_videos = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))[:10]

print(f"\n{'='*90}")
print(f"BENCHMARK v2: {len(fake_videos)} fake + {len(real_videos)} real videos")
print(f"{'='*90}")

correct_fake = 0
correct_real = 0

print(f"\n--- FAKE VIDEOS (expect Fake) ---")
print(f"  {'File':<12} {'Result':<6} {'Ens':>5} {'Full':>5} {'Face':>5} {'Bnd':>5} {'Noi':>5} {'ELA':>5}")
for v in fake_videos:
    t0 = time.time()
    r = test_video(v)
    dt = time.time() - t0
    if r is None: continue
    ok = "OK" if r["is_fake"] else "XX"
    if r["is_fake"]: correct_fake += 1
    print(f"  {ok} {os.path.basename(v):<10} {'FAKE' if r['is_fake'] else 'REAL':<6} {r['ens']:.3f} {r['full']:.3f} {r['face']:.3f} {r['bnd']:.3f} {r['noi']:.3f} {r['ela']:.3f}  [{dt:.1f}s]")

print(f"\n--- REAL VIDEOS (expect Real) ---")
print(f"  {'File':<12} {'Result':<6} {'Ens':>5} {'Full':>5} {'Face':>5} {'Bnd':>5} {'Noi':>5} {'ELA':>5}")
for v in real_videos:
    t0 = time.time()
    r = test_video(v)
    dt = time.time() - t0
    if r is None: continue
    ok = "OK" if not r["is_fake"] else "XX"
    if not r["is_fake"]: correct_real += 1
    print(f"  {ok} {os.path.basename(v):<10} {'FAKE' if r['is_fake'] else 'REAL':<6} {r['ens']:.3f} {r['full']:.3f} {r['face']:.3f} {r['bnd']:.3f} {r['noi']:.3f} {r['ela']:.3f}  [{dt:.1f}s]")

total = len(fake_videos) + len(real_videos)
correct_total = correct_fake + correct_real
print(f"\n{'='*90}")
print(f"RESULTS: {correct_total}/{total} correct ({100*correct_total/total:.1f}%)")
print(f"  Fake accuracy:  {correct_fake}/{len(fake_videos)} ({100*correct_fake/len(fake_videos):.1f}%)")
print(f"  Real accuracy:  {correct_real}/{len(real_videos)} ({100*correct_real/len(real_videos):.1f}%)")
print(f"{'='*90}")
