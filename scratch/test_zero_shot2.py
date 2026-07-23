from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
candidate_labels = ["professional news reporting", "personal blog post or corporate advertisement"]

texts = [
    ("Random Ass Thoughts", "I was thinking about trimming my beard today, but then I realized it's too much work. Life is crazy right now."),
    ("New Swiffer Lemon Duster", "Introducing the new Swiffer Lemon Duster. Get the fresh scent of clean with no spray required. Buy now at your local store."),
    ("Parakram Gate 2026 Batch", "Join the Parakram GATE 2026 Batch for Computer Science. Enroll today to secure your future!"),
    ("Global Markets Fall Amid Tech Selloff", "Global stock markets tumbled on Thursday following a massive tech selloff on Wall Street. Investors remain cautious."),
]

for title, text in texts:
    full_text = f"{title}. {text}"
    result = classifier(full_text, candidate_labels)
    print(f"\n--- {title} ---")
    for i in range(len(result['labels'])):
        print(f"{result['labels'][i]}: {result['scores'][i]:.4f}")
