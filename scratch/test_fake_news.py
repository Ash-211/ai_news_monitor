import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.intelligence.fake_news import detect_fake_news, load_fake_news_detector

title = "Random Ass Thoughts"
content = "I was thinking about trimming my beard today, but then I realized it's too much work. Life is crazy right now."

model, tokenizer = load_fake_news_detector()
is_fake, score, breakdown = detect_fake_news(title, content, model=model, tokenizer=tokenizer)

print("Score:", score)
print("Breakdown:", breakdown)
