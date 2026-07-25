"""
AI News Worker — Hugging Face Space (Gradio SDK)
Loads the fine-tuned DistilBERT fake news detection model and serves
predictions via Gradio's built-in API.

Render backend sends article text here → this Space runs it through
the neural network → returns the credibility score back to Render.
"""

try:
    import spaces  # Must be imported before torch on HF Spaces
except ImportError:
    pass

import torch
import gradio as gr
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


def predict(text: str) -> str:
    """
    Accepts article text, runs it through the DistilBERT model,
    and returns a JSON string with real/fake probabilities.
    """
    import json

    text = (text or "").strip()
    if not text or len(text) < 10:
        return json.dumps({
            "real_probability": 0.5,
            "fake_probability": 0.5,
            "label": "UNKNOWN"
        })

    # Tokenize and run inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Index 0 = REAL, Index 1 = FAKE
        real_prob = round(float(probabilities[0][0].item()), 4)
        fake_prob = round(float(probabilities[0][1].item()), 4)

    label = "REAL" if real_prob >= 0.5 else "FAKE"

    return json.dumps({
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "label": label
    })


# ── Gradio Interface ──────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Article Text", lines=5, placeholder="Paste article text here..."),
    outputs=gr.Textbox(label="Prediction (JSON)"),
    title="🔍 AI News Worker — DistilBERT Fake News Detector",
    description="Paste any news article text and this model will predict whether it is REAL or FAKE.",
    examples=[
        ["The United Nations released its annual report on climate change impacts across developing nations."],
        ["BREAKING: Scientists discover that drinking bleach cures all diseases instantly!"],
    ],
    api_name="predict"
)

if __name__ == "__main__":
    demo.launch()
