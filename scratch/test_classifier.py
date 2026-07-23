import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.intelligence.classifier import load_classifier, classify_article

model = load_classifier()

texts = [
    ("Random Ass Thoughts", "I was thinking about trimming my beard today, but then I realized it's too much work. Life is crazy right now."),
    ("New Swiffer Lemon Duster", "Introducing the new Swiffer Lemon Duster. Get the fresh scent of clean with no spray required. Buy now at your local store."),
    ("Parakram Gate 2026 Batch", "Join the Parakram GATE 2026 Batch for Computer Science. Enroll today to secure your future!"),
    ("Global Markets Fall Amid Tech Selloff", "Global stock markets tumbled on Thursday following a massive tech selloff on Wall Street. Investors remain cautious. The Dow Jones Industrial Average dropped 500 points."),
]

for title, text in texts:
    full_text = f"{title}. {text}"
    cat, conf = classify_article(full_text, model=model)
    print(f"\n--- {title} ---")
    print(f"Category: {cat}, Confidence: {conf:.4f}")
