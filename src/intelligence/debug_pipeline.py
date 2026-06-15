
import sys
import os
sys.path.append(os.getcwd())

from src.ingestion.database import get_session, Article
from src.intelligence.classifier import load_classifier, classify_batch
from src.intelligence.fake_news import load_fake_news_detector, detect_batch

def debug_db():
    session = get_session()
    if not session:
        print("Database session failed.")
        return

    articles = session.query(Article).limit(5).all()
    print(f"--- Database Check (Top {len(articles)}) ---")
    for a in articles:
        rc_len = len(a.raw_content) if a.raw_content else 0
        cc_len = len(a.clean_content) if a.clean_content else 0
        print(f"ID: {a.id} | Title: {a.title[:40]}... | Raw: {rc_len} | Clean: {cc_len}")
        print(f"  Category: {a.category} | Fake: {a.is_fake} | Score: {a.credibility_score}")

    # Check model loading
    clf = load_classifier()
    detector, tokenizer = load_fake_news_detector()
    
    if articles:
        texts = [a.title + " " + (a.clean_content or "") for a in articles]
        print("\n--- Model Check on these articles ---")
        if clf:
            cats = classify_batch(texts, clf)
            print(f"Categories: {cats}")
        if detector and tokenizer:
            titles = [a.title for a in articles]
            contents = [a.clean_content or "" for a in articles]
            fakes = detect_batch(titles, contents, model=detector, tokenizer=tokenizer, sources=[a.source for a in articles])
            print(f"Detections: {fakes}")

    session.close()

if __name__ == "__main__":
    debug_db()
