import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.intelligence.classifier import load_classifier, classify_article

model = load_classifier()

texts = [
    ("Local Firefighters Rescue Cat", "A local firefighter team in downtown Seattle successfully rescued a cat stuck in a 40-foot oak tree this morning. The cat, named Whiskers, was returned safely to its owner."),
    ("Scientists Discover New Exoplanet", "Astronomers using the James Webb Space Telescope have discovered a new exoplanet with potential signs of water in its atmosphere, located 40 light-years away."),
    ("President Signs Historic Climate Bill", "The President signed a historic climate bill into law today, allocating $300 billion for renewable energy initiatives over the next decade."),
    ("Nadal Wins French Open", "Rafael Nadal secured his 15th French Open title on Sunday after defeating his opponent in straight sets. The match lasted only two hours."),
]

for title, text in texts:
    full_text = f"{title}. {text}"
    cat, conf = classify_article(full_text, model=model)
    print(f"\n--- {title} ---")
    print(f"Category: {cat}, Confidence: {conf:.4f}")
