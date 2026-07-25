"""
AI News Worker — Hugging Face Space
Loads the fine-tuned DistilBERT fake news detection model and serves
predictions via a simple FastAPI endpoint.

Render backend sends article text here → this Space runs it through
the neural network → returns the credibility score back to Render.
"""

import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Load Model at Startup ─────────────────────────────────────────────────────
MODEL_ID = "vinitsingare/distilbert_fake_news"

print(f"Loading model '{MODEL_ID}' from Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded successfully on {device}!")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI News Worker", description="DistilBERT Fake News Detection API")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    real_probability: float
    fake_probability: float
    label: str


@app.get("/")
def health():
    """Health check endpoint so HF Spaces doesn't show 404."""
    return {"status": "running", "model": MODEL_ID}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accepts article text, runs it through the DistilBERT model,
    and returns the real/fake probabilities.
    """
    text = request.text.strip()
    if not text or len(text) < 10:
        return PredictResponse(real_probability=0.5, fake_probability=0.5, label="UNKNOWN")

    # Tokenize and run inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Index 0 = REAL, Index 1 = FAKE
        real_prob = float(probabilities[0][0].item())
        fake_prob = float(probabilities[0][1].item())

    label = "REAL" if real_prob >= 0.5 else "FAKE"

    return PredictResponse(
        real_probability=round(real_prob, 4),
        fake_probability=round(fake_prob, 4),
        label=label
    )
