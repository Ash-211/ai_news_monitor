"""
AI News Worker — Hugging Face Space (Gradio SDK + ZeroGPU)
"""

import spaces
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "vinitsingare/distilbert_fake_news"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

@spaces.GPU
def predict(text: str) -> str:
    import json
    text = (text or "").strip()
    if not text or len(text) < 10:
        return json.dumps({"real_probability": 0.5, "fake_probability": 0.5, "label": "UNKNOWN"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        real_prob = round(float(probabilities[0][0].item()), 4)
        fake_prob = round(float(probabilities[0][1].item()), 4)

    label = "REAL" if real_prob >= 0.5 else "FAKE"
    return json.dumps({"real_probability": real_prob, "fake_probability": fake_prob, "label": label})

demo = gr.Interface(fn=predict, inputs=gr.Textbox(lines=5), outputs=gr.Textbox())
if __name__ == "__main__":
    demo.launch()
