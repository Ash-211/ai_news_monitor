import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os, glob

model_id = "dima806/deepfake_vs_real_image_detection"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)
model.eval()

def test_file(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        cap = cv2.VideoCapture(path)
        ret, img = cap.read()
        if not ret: return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        margin = int(w * 0.2)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)
        img = img[y1:y2, x1:x2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.nn.functional.softmax(out.logits, dim=-1)[0]
    idx = probs.argmax().item()
    print(f"{os.path.basename(path)} -> Label: {model.config.id2label[idx]}, conf: {probs[idx]:.4f}")

test_file(r"C:\Users\gamin\Downloads\videos_fake\vs1.mp4")
test_file(r"C:\Users\gamin\Downloads\videos_fake\vs10.mp4")
test_file(r"C:\Users\gamin\Downloads\videos_real\v1.mp4")
test_file(r"C:\Users\gamin\Downloads\videos_real\v10.mp4")
