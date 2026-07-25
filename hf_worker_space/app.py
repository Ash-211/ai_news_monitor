"""
AI News Worker — Hugging Face Space (Gradio SDK)
Loads the fine-tuned DistilBERT fake news detection model and serves
predictions via Gradio's built-in API.
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Load Model at Startup ─────────────────────────────────────────────────────
MODEL_ID = "vinitsingare/distilbert_fake_news"

print(f"Loading model '{MODEL_ID}' from Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded successfully on CPU!")


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

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        real_prob = round(float(probabilities[0][0].item()), 4)
        fake_prob = round(float(probabilities[0][1].item()), 4)

    label = "REAL" if real_prob >= 0.5 else "FAKE"

    return json.dumps({
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "label": label
    })


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Article Text", lines=5, placeholder="Paste article text here..."),
    outputs=gr.Textbox(label="Prediction (JSON)"),
    title="AI News Worker - DistilBERT Fake News Detector",
    description="Paste any news article text and this model will predict whether it is REAL or FAKE.",
    api_name="predict"
)

if __name__ == "__main__":
    demo.launch()
