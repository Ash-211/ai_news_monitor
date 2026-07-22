from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

text = "Delhi restaurant fire LIVE: At least 21 people killed, several foreigners among those dead. Afire broke out at a hotel in Delhi’s Malviya Nagar on Wednesday morning (June 3, 2026), killing at least 21 people and leading to the rescue of more than 40 others."

print("Testing mrm8488/distilroberta-finetuned-fake-news...")
try:
    model_name = "mrm8488/distilroberta-finetuned-fake-news"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Check what index 0 and 1 mean for this specific model (usually 0 is reliable, 1 is fake, or vice versa)
    print(f"Probabilities: {probs}")
    print(f"Predicted class: {model.config.id2label[probs.argmax().item()]}")
except Exception as e:
    print(f"Error with distilroberta: {e}")

