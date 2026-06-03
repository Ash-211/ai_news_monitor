
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.intelligence.fake_news import load_fake_news_detector, detect_fake_news, TRUSTED_SOURCES

def test_samples():
    model = load_fake_news_detector()
    if not model:
        print("Model not found. Please train it first.")
        return

    # test cases: (text_preview, source, expected_real, corroboration)
    test_cases = [
        {
            "label": "TRUE (Trusted Source)",
            "text": "Kerala student who went missing in Karnataka Chandradrona hills found dead",
            "source": "Times of India",
            "corr": 1
        },
        {
            "label": "TRUE (Regular Source)",
            "text": "A new park was inaugurated in the city center yesterday by the local municipal corporation to provide more green space for residents.",
            "source": "Local News",
            "corr": 0
        },
        {
            "label": "FAKE (Sensationalist)",
            "text": "SHOCKING: Secret documents from the moon reveal that gravity is actually a controlled experiment by billionaire tech giants! SHARE THIS NOW!",
            "source": "ViralTruth.net",
            "corr": 0
        },
        {
            "label": "FAKE (Medical Misinfo)",
            "text": "Doctors are hiding this 10-second habit that cures cancer completely overnight. No more chemotherapy needed, just eat this one fruit!",
            "source": "HealthyTipsDaily",
            "corr": 0
        },
        {
            "label": "CORROBORATED (Potential Fake but multiple reporting)",
            "text": "Breaking news: Reports of a major chemical spill near the industrial zone. Emergency services are arriving on site.",
            "source": "Independent Blogger",
            "corr": 2
        }
    ]

    print("\n" + "="*80)
    print(f"{'TEST CASE':<30} | {'PREDICTION':<8} | {'SCORE':<10} | {'BASIS'}")
    print("-" * 80)

    for case in test_cases:
        is_fake, score = detect_fake_news(case['text'], model=model, source=case['source'], corroboration_count=case['corr'])
        status = "FAKE" if is_fake else "REAL"
        
        # Derive basis
        from sklearn.feature_extraction.text import TfidfVectorizer
        probs = model.predict_proba([case['text']])[0]
        base_ml = probs[0]
        
        basis_str = f"ML: {base_ml:.2f}"
        if any(ts.lower() in case['source'].lower() for ts in TRUSTED_SOURCES):
            basis_str += " + 0.15 (Trusted)"
        if case['corr'] >= 1:
            basis_str += " + 0.10 (Corr)"
        if not any(ts.lower() in case['source'].lower() for ts in TRUSTED_SOURCES) and case['corr'] == 0:
            basis_str += " - 0.15 (Lone)"

        print(f"{case['label']:<30} | {status:<10} | {score*100:>6.1f}% | {basis_str}")

if __name__ == "__main__":
    test_samples()
