import cv2
import torch
from PIL import Image

def get_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return Image.open(image_path).convert("RGB") # Fallback to full image
    
    # Get largest face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    
    # Add margin
    margin = int(w * 0.2)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)
    
    face_img = img[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)
