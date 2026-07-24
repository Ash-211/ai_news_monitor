"""
Deepfake Detection Module — Ensemble Approach
==============================================
Uses a pre-trained SigLIP-based Vision Transformer combined with a
dual-pass ensemble (full-frame + face-crop) to classify images and
video frames as "Real" or "Fake" (AI-generated / deepfake).

Model: prithivMLmods/deepfake-detector-model-v1
Architecture: google/siglip-base-patch16-512 (fine-tuned)

Ensemble Strategy:
  The SigLIP model has a strong bias toward "Fake" on video frames due to
  compression artifacts. To counteract this, we run TWO passes:
    1. Full-frame  → captures overall synthetic patterns (biased toward Fake)
    2. Face-crop   → focuses on the face region (more discriminating)
  The final score is:
    fakeness = 0.5 * face_fake + 0.5 * (1 - score_diff)
  where score_diff = full_fake - face_fake. A higher gap between full-frame
  and face-crop indicates the face looks genuine (full-frame is biased by
  compression but the face itself is real).

Pipeline:
  1. Image → Full-frame + Face-crop inference → Ensemble → Verdict
  2. Video → Sample every Nth frame → Run (1) on each → Aggregate
"""

import os
import logging
import tempfile
import requests as hf_requests
import io
import base64
from typing import Optional

logger = logging.getLogger(__name__)

# ── Global model cache (lazy-loaded once, reused across requests) ─────────
_face_cascade = None

# Model identifier on HuggingFace
DEEPFAKE_MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"

# HuggingFace Inference API URL
HF_API_URL = f"https://api-inference.huggingface.co/models/{DEEPFAKE_MODEL_ID}"

# Ensemble parameters (calibrated via grid search on 30 test videos)
ENSEMBLE_WEIGHT = 0.5      # Weight for face_mean vs (1 - score_diff)
ENSEMBLE_THRESHOLD = 0.56  # Scores above this are classified as Fake


def _ensure_face_cascade():
    """Lazy-load the Haar Cascade face detector (cached globally)."""
    global _face_cascade
    if _face_cascade is None:
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _get_face_rect(img_bgr):
    """
    Detect the largest face in a BGR image.
    Returns (x, y, w, h) tuple or None if no face found.
    """
    import cv2
    cascade = _ensure_face_cascade()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Return the largest face by area
    return max(faces, key=lambda f: f[2] * f[3])


def _crop_face(img_bgr, face_rect, margin_ratio=0.3):
    """
    Crop the face region from a BGR image with a margin.
    Returns cropped BGR image.
    """
    x, y, w, h = face_rect
    margin = int(w * margin_ratio)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img_bgr.shape[1], x + w + margin), min(img_bgr.shape[0], y + h + margin)
    return img_bgr[y1:y2, x1:x2]


def _classify_bgr(img_bgr):
    """
    Run the SigLIP model on a BGR image via HuggingFace Inference API.
    Returns fake_prob (float, 0-1).
    """
    import cv2

    hf_token = os.getenv("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    # Encode the BGR image as JPEG bytes for the API
    success, img_encoded = cv2.imencode('.jpg', img_bgr)
    if not success:
        logger.warning("Failed to encode image for API call")
        return 0.5  # Neutral score on failure

    img_bytes = img_encoded.tobytes()

    try:
        response = hf_requests.post(HF_API_URL, headers=headers, data=img_bytes, timeout=30)

        if response.status_code == 200:
            results = response.json()
            # Results format: [{"label": "Fake", "score": 0.8}, {"label": "Real", "score": 0.2}]
            for item in results:
                if item.get("label", "").lower() == "fake":
                    return float(item["score"])
            # If "Fake" label not found, return 1 - Real score
            for item in results:
                if item.get("label", "").lower() == "real":
                    return 1.0 - float(item["score"])
            return 0.5  # Fallback
        else:
            logger.warning("HuggingFace deepfake API returned status %d: %s", response.status_code, response.text[:200])
            return 0.5  # Neutral score on API error
    except Exception as e:
        logger.error("HuggingFace deepfake API call failed: %s", e)
        return 0.5  # Neutral score on failure


def _compute_ensemble_score(full_fake, face_fake):
    """
    Compute the calibrated ensemble fakeness score.
    
    Uses the insight that the gap between full-frame and face-crop scores
    is a strong indicator: real faces have a LARGER gap (full frame biased
    by compression, but face itself looks genuine), while deepfakes have
    a SMALLER gap (both full frame and face look synthetic).
    
    Returns: float (0-1), where higher = more likely fake.
    """
    score_diff = full_fake - face_fake
    # Combine face_mean (direct signal) with inverted score_diff (gap signal)
    fakeness = ENSEMBLE_WEIGHT * face_fake + (1 - ENSEMBLE_WEIGHT) * (1.0 - score_diff)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, fakeness))


def _classify_confidence_band(ensemble_score):
    """
    Map ensemble score to a human-readable confidence band.
    Calibrated against benchmark results to be honest about uncertainty.
    """
    if ensemble_score >= 0.80:
        return "high"
    elif ensemble_score >= 0.65:
        return "moderate"
    elif ensemble_score >= ENSEMBLE_THRESHOLD:
        return "low"
    elif ensemble_score >= (1.0 - 0.65):  # mirror for "real"
        return "low"
    elif ensemble_score >= (1.0 - 0.80):
        return "moderate"
    else:
        return "high"


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def detect_deepfake_image(image_path: str) -> dict:
    """
    Analyze a single image for deepfake / AI-generation indicators using
    the dual-pass ensemble approach.

    Args:
        image_path: Absolute path to the image file (jpg, png, webp).

    Returns:
        dict with keys:
            - is_fake (bool): True if the ensemble classifies as deepfake.
            - confidence (float): 0.0-1.0 ensemble confidence.
            - label (str): Human-readable label ("Real" or "Fake").
            - confidence_band (str): "high", "moderate", or "low".
            - raw_scores (dict): Breakdown of individual signals.
            - explanation (str): XAI-style human-readable reasoning.
    """
    import cv2
    from PIL import Image as PILImage

    # Load the image as BGR for OpenCV processing
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        # Fallback: try loading with PIL and converting
        pil_img = PILImage.open(image_path).convert("RGB")
        import numpy as np
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ── Pass 1: Full-frame classification ─────────────────────────────
    full_fake = _classify_bgr(img_bgr)

    # ── Pass 2: Face-crop classification ──────────────────────────────
    face_rect = _get_face_rect(img_bgr)
    if face_rect is not None:
        face_crop = _crop_face(img_bgr, face_rect)
        face_fake = _classify_bgr(face_crop)
        has_face = True
    else:
        face_fake = full_fake  # No face found → fallback to full-frame
        has_face = False

    # ── Ensemble scoring ──────────────────────────────────────────────
    ensemble_score = _compute_ensemble_score(full_fake, face_fake)
    is_fake = ensemble_score > ENSEMBLE_THRESHOLD

    # Compute a "confidence in the verdict" (distance from threshold)
    if is_fake:
        confidence = min(1.0, 0.5 + (ensemble_score - ENSEMBLE_THRESHOLD) / (1.0 - ENSEMBLE_THRESHOLD) * 0.5)
    else:
        confidence = min(1.0, 0.5 + (ENSEMBLE_THRESHOLD - ensemble_score) / ENSEMBLE_THRESHOLD * 0.5)

    label = "Fake" if is_fake else "Real"
    band = _classify_confidence_band(ensemble_score)

    # Build raw scores for transparency
    raw_scores = {
        "Fake": round(ensemble_score, 4),
        "Real": round(1.0 - ensemble_score, 4),
        "full_frame_fake": round(full_fake, 4),
        "face_crop_fake": round(face_fake, 4),
        "score_diff": round(full_fake - face_fake, 4),
        "face_detected": has_face,
    }

    h, w = img_bgr.shape[:2]
    explanation = _generate_image_explanation(label, confidence, band, raw_scores, (w, h))

    return {
        "is_fake": is_fake,
        "confidence": round(confidence, 4),
        "label": label,
        "confidence_band": band,
        "raw_scores": raw_scores,
        "explanation": explanation,
    }


def _generate_image_explanation(label: str, confidence: float, band: str,
                                 raw_scores: dict, image_size: tuple) -> str:
    """
    Produce a rich, human-readable explanation of the deepfake analysis result.
    Honest about uncertainty levels.
    """
    pct = round(confidence * 100, 1)
    w, h = image_size
    is_fake = label == "Fake"
    face_detected = raw_scores.get("face_detected", False)

    # Confidence-calibrated verdicts
    if is_fake:
        if band == "high":
            verdict = f"This image shows strong indicators of AI generation or manipulation ({pct}% confidence)."
            detail = "Multiple analysis passes detected consistent synthetic patterns in both the full image and facial region."
        elif band == "moderate":
            verdict = f"This image shows moderate indicators of possible manipulation ({pct}% confidence)."
            detail = "Some synthetic patterns were detected. This could indicate AI generation, heavy filtering, or face-swap manipulation."
        else:
            verdict = f"This image shows mild indicators of possible manipulation ({pct}% confidence)."
            detail = "The analysis detected borderline signals. The result is uncertain -- manual review is recommended."
    else:
        if band == "high":
            verdict = f"This image appears authentic ({pct}% confidence)."
            detail = "The image exhibits natural patterns consistent with real photography across all analysis passes."
        elif band == "moderate":
            verdict = f"This image appears likely authentic ({pct}% confidence)."
            detail = "The image shows predominantly natural characteristics, with some minor ambiguous elements."
        else:
            verdict = f"This image shows uncertain results ({pct}% confidence)."
            detail = "The analysis produced borderline scores. The image may be authentic or subtly manipulated. Manual review is recommended."

    # Add context notes
    notes = []
    resolution_note = f"Image resolution: {w}x{h}px."
    if w < 256 or h < 256:
        notes.append("Low resolution may reduce detection accuracy.")
    if not face_detected:
        notes.append("No face was detected -- analysis was performed on the full image only.")

    context = resolution_note
    if notes:
        context += " " + " ".join(notes)

    return f"{verdict}\n\n{detail}\n\n{context}"


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def detect_deepfake_video(video_path: str, sample_rate: int = 10) -> dict:
    """
    Analyze a video for deepfake indicators by sampling every Nth frame
    and running the dual-pass ensemble on each.

    Args:
        video_path: Absolute path to the video file (mp4, avi, mov).
        sample_rate: Analyze every Nth frame (default: every 10th frame).

    Returns:
        dict with keys:
            - is_fake (bool): Overall verdict based on ensemble scoring.
            - confidence (float): Ensemble confidence in the verdict.
            - label (str): "Real" or "Fake" overall verdict.
            - confidence_band (str): "high", "moderate", or "low".
            - total_frames (int): Total frames in the video.
            - analyzed_frames (int): Frames actually analyzed.
            - fps (float): Video frames per second.
            - duration_seconds (float): Video duration in seconds.
            - frame_results (list): Per-frame results for timeline.
            - raw_scores (dict): Aggregated signal breakdown.
            - explanation (str): XAI-style human-readable reasoning.
    """
    import cv2
    import numpy as np

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Failed to open video file. The format may not be supported."}

    # Extract video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0

    # Cap the maximum number of frames to analyze
    MAX_FRAMES_TO_ANALYZE = 30
    effective_sample_rate = max(sample_rate, total_frames // MAX_FRAMES_TO_ANALYZE) \
        if total_frames > MAX_FRAMES_TO_ANALYZE * sample_rate else sample_rate

    frame_results = []
    full_scores = []
    face_scores = []
    frame_idx = 0

    logger.info(
        "Analyzing video: %d total frames, %.1f fps, %.1f sec, sampling every %d frames",
        total_frames, fps, duration, effective_sample_rate
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % effective_sample_rate == 0:
            # ── Pass 1: Full-frame ────────────────────────────────────
            full_fake = _classify_bgr(frame)
            full_scores.append(full_fake)

            # ── Pass 2: Face-crop ─────────────────────────────────────
            face_rect = _get_face_rect(frame)
            if face_rect is not None:
                face_crop = _crop_face(frame, face_rect)
                face_fake = _classify_bgr(face_crop)
            else:
                face_fake = full_fake  # fallback
            face_scores.append(face_fake)

            # ── Per-frame ensemble score ──────────────────────────────
            frame_ensemble = _compute_ensemble_score(full_fake, face_fake)
            is_frame_fake = frame_ensemble > ENSEMBLE_THRESHOLD

            frame_results.append({
                "frame": frame_idx,
                "timestamp": round(frame_idx / fps, 2),
                "label": "Fake" if is_frame_fake else "Real",
                "confidence": round(frame_ensemble, 4),
                "is_fake": is_frame_fake,
            })

        frame_idx += 1

    cap.release()

    if not frame_results:
        return {"error": "No frames could be extracted from the video."}

    # ── Aggregate across all frames using the ensemble ────────────────
    face_mean = float(np.mean(face_scores)) if face_scores else 0.0
    full_mean = float(np.mean(full_scores)) if full_scores else 0.0
    score_diff = full_mean - face_mean

    # Overall ensemble score (aggregated, not per-frame average)
    overall_ensemble = _compute_ensemble_score(full_mean, face_mean)
    overall_is_fake = overall_ensemble > ENSEMBLE_THRESHOLD

    # Confidence in the verdict
    if overall_is_fake:
        confidence = min(1.0, 0.5 + (overall_ensemble - ENSEMBLE_THRESHOLD) / (1.0 - ENSEMBLE_THRESHOLD) * 0.5)
    else:
        confidence = min(1.0, 0.5 + (ENSEMBLE_THRESHOLD - overall_ensemble) / ENSEMBLE_THRESHOLD * 0.5)

    label = "Fake" if overall_is_fake else "Real"
    band = _classify_confidence_band(overall_ensemble)

    # Frame-level stats for timeline
    fake_count = sum(1 for f in frame_results if f["is_fake"])
    real_count = len(frame_results) - fake_count
    fake_ratio = fake_count / len(frame_results)

    raw_scores = {
        "Fake": round(overall_ensemble, 4),
        "Real": round(1.0 - overall_ensemble, 4),
        "full_frame_fake": round(full_mean, 4),
        "face_crop_fake": round(face_mean, 4),
        "score_diff": round(score_diff, 4),
        "fake_frame_ratio": round(fake_ratio, 4),
    }

    explanation = _generate_video_explanation(
        label, confidence, band, fake_count, real_count,
        len(frame_results), total_frames, duration, raw_scores
    )

    return {
        "is_fake": overall_is_fake,
        "confidence": round(confidence, 4),
        "label": label,
        "confidence_band": band,
        "total_frames": total_frames,
        "analyzed_frames": len(frame_results),
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 2),
        "fake_frame_ratio": round(fake_ratio, 4),
        "frame_results": frame_results,
        "raw_scores": raw_scores,
        "explanation": explanation,
    }


def _generate_video_explanation(
    label: str, confidence: float, band: str,
    fake_count: int, real_count: int,
    analyzed: int, total: int, duration: float,
    raw_scores: dict
) -> str:
    """
    Produce a rich, honest explanation for video deepfake analysis.
    """
    pct = round(confidence * 100, 1)
    is_fake = label == "Fake"
    ratio_pct = round((fake_count / analyzed) * 100, 1) if analyzed > 0 else 0

    if is_fake:
        if band == "high":
            verdict = f"This video shows strong indicators of deepfake manipulation ({pct}% confidence)."
            detail = (
                f"Across {analyzed} sampled frames (from {total} total, {round(duration, 1)}s), "
                f"the dual-pass analysis consistently detected synthetic facial patterns."
            )
        elif band == "moderate":
            verdict = f"This video shows moderate indicators of possible manipulation ({pct}% confidence)."
            detail = (
                f"Across {analyzed} sampled frames, the analysis detected mixed signals "
                f"with {fake_count} frames ({ratio_pct}%) flagged as potentially manipulated."
            )
        else:
            verdict = f"This video shows mild indicators of possible manipulation ({pct}% confidence)."
            detail = (
                f"The analysis produced borderline scores across {analyzed} frames. "
                f"The result is uncertain -- manual review is recommended."
            )
    else:
        if band == "high":
            verdict = f"This video appears authentic ({pct}% confidence)."
            detail = (
                f"Across {analyzed} sampled frames (from {total} total, {round(duration, 1)}s), "
                f"the analysis found consistent natural patterns in both full-frame and facial regions."
            )
        elif band == "moderate":
            verdict = f"This video appears likely authentic ({pct}% confidence)."
            detail = (
                f"Across {analyzed} sampled frames, the analysis found predominantly "
                f"natural characteristics with some minor ambiguous elements."
            )
        else:
            verdict = f"This video shows uncertain results ({pct}% confidence)."
            detail = (
                f"The analysis produced borderline scores across {analyzed} frames. "
                f"The video may be authentic or subtly manipulated. Manual review is recommended."
            )

    # Technical note about the ensemble approach
    tech_note = (
        f"Analysis method: Dual-pass ensemble (full-frame + face-crop) with "
        f"calibrated scoring. Full-frame signal: {raw_scores.get('full_frame_fake', 0):.2f}, "
        f"Face-crop signal: {raw_scores.get('face_crop_fake', 0):.2f}."
    )

    return f"{verdict}\n\n{detail}\n\n{tech_note}"
