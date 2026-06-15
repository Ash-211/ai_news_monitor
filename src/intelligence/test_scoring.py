import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.intelligence.fake_news import load_fake_news_detector, detect_fake_news

def test_samples():
    model, tokenizer = load_fake_news_detector()
    if not model:
        print("Model not found. Please train it first.")
        return

    # test cases
    test_cases = [
        {
            "label": "TRUE (Trusted Source)",
            "title": "Kerala student who went missing in Karnataka Chandradrona hills found dead",
            "content": "The body of a 22-year-old student from Kerala, who went missing in the Chandradrona hill ranges of Chikkamagaluru district on Wednesday, was found on Thursday.",
            "source": "Times of India",
        },
        {
            "label": "TRUE (Regular Source)",
            "title": "A new park was inaugurated in the city center",
            "content": "A new park was inaugurated in the city center yesterday by the local municipal corporation to provide more green space for residents.",
            "source": "Local News",
        },
        {
            "label": "FAKE (Sensationalist)",
            "title": "SHOCKING: Secret documents from the moon reveal gravity is a hoax",
            "content": "SHOCKING: Secret documents from the moon reveal that gravity is actually a controlled experiment by billionaire tech giants! SHARE THIS NOW!",
            "source": "ViralTruth.net",
        },
        {
            "label": "FAKE (Medical Misinfo)",
            "title": "Doctors are hiding this 10-second habit that cures cancer",
            "content": "Doctors are hiding this 10-second habit that cures cancer completely overnight. No more chemotherapy needed, just eat this one fruit!",
            "source": "HealthyTipsDaily",
        }
    ]

    print("\n" + "="*80)
    print(f"{'TEST CASE':<30} | {'PREDICTION':<10} | {'SCORE':<10}")
    print("-" * 80)

    for case in test_cases:
        is_fake, score, breakdown = detect_fake_news(case['title'], case['content'], model=model, tokenizer=tokenizer, source=case['source'])
        status = "FAKE" if is_fake else "REAL"
        
        print(f"{case['label']:<30} | {status:<10} | {score*100:>6.1f}%")
        print(f"  Explanation: {breakdown.get('explanation_text', '')}")
        print("-" * 80)

if __name__ == "__main__":
    test_samples()
