from transformers import pipeline

print("Loading flan-t5-base...")
classifier = pipeline("text2text-generation", model="google/flan-t5-base")

texts = [
    ("Random Ass Thoughts", "I was thinking about trimming my beard today, but then I realized it's too much work. Life is crazy right now."),
    ("New Swiffer Lemon Duster", "Introducing the new Swiffer Lemon Duster. Get the fresh scent of clean with no spray required. Buy now at your local store."),
    ("Parakram Gate 2026 Batch", "Join the Parakram GATE 2026 Batch for Computer Science. Enroll today to secure your future!"),
    ("Global Markets Fall Amid Tech Selloff", "Global stock markets tumbled on Thursday following a massive tech selloff on Wall Street. Investors remain cautious."),
]

for title, text in texts:
    full_text = f"Title: {title}\nText: {text}\n\nIs this text a professional journalistic news article? Answer yes or no."
    result = classifier(full_text, max_new_tokens=5)
    print(f"\n--- {title} ---")
    print("Result:", result[0]['generated_text'])
