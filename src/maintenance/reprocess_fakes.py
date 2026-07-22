import os
import sys

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ingestion.database import get_session, Article
from src.intelligence.fake_news import detect_fake_news, load_fake_news_detector

def reprocess_fakes():
    """
    Finds articles currently marked as fake by the old model
    and re-runs the detection logic with the new DistilBERT model.
    """
    session = get_session()
    model, tokenizer = load_fake_news_detector()
    
    if not model or not tokenizer:
        print("Error: Fake news detector model not found.")
        return

    print("\n--- Reprocessing Old Fakes with DistilBERT ---")
    
    # Query for articles where is_fake is True
    query = session.query(Article).filter(Article.is_fake == True)
    
    all_fakes = query.all()
    
    print(f"Found {len(all_fakes)} total articles marked as fake.")
    if not all_fakes:
        return

    print("=" * 60)

    updated_count = 0
    for a in all_fakes:
        old_score = a.credibility_score
        
        # Build text (consistent with pipeline.py)
        title = a.title or ''
        content = a.clean_content or a.raw_content or ''
            
        # Re-detect
        is_fake, new_score, breakdown = detect_fake_news(title, content, model=model, tokenizer=tokenizer, source=a.source)
        
        # If the old score was None, treat it as 0 to avoid TypeError in abs()
        old_score_val = old_score if old_score is not None else 0.0
        
        if is_fake != a.is_fake or abs(new_score - old_score_val) > 0.01 or not a.score_details:
            status_change = "FAKE -> REAL" if not is_fake else "STILL FAKE"
            safe_title = (a.title or "").encode('ascii', 'ignore').decode()
            print(f"ID {a.id}: {status_change} | Score: {old_score_val:.2f} -> {new_score:.2f} | {safe_title[:50]}...")
            
            a.is_fake = is_fake
            a.credibility_score = new_score
            import json
            a.score_details = json.dumps(breakdown)
            updated_count += 1
            
    if updated_count > 0:
        session.commit()
        print("\n" + "=" * 60)
        print(f"Successfully updated {updated_count} articles.")
    else:
        print("\nNo status changes detected after re-analysis.")
        
    session.close()

if __name__ == "__main__":
    reprocess_fakes()
