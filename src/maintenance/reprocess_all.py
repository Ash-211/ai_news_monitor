import os
import sys
import json

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ingestion.database import get_session, Article
from src.intelligence.fake_news import detect_fake_news, load_fake_news_detector

def reprocess_all():
    """
    Re-runs the detection logic with the new DistilBERT model and XAI explanation 
    for ALL articles in the database.
    """
    session = get_session()
    model, tokenizer = load_fake_news_detector()
    
    if not model or not tokenizer:
        print("Error: Fake news detector model not found.")
        return

    print("\n--- Reprocessing ALL Articles with New XAI Explanations ---")
    
    all_articles = session.query(Article).limit(5).all()
    print(f"Found {len(all_articles)} total articles.")
    if not all_articles:
        return

    print("=" * 60)

    updated_count = 0
    for a in all_articles:
        # Build text (consistent with pipeline.py)
        title = a.title or ''
        content = a.clean_content or a.raw_content or ''
        
        # Load fact check data if available in old details
        verification_result = None
        if a.score_details:
            try:
                old_details = json.loads(a.score_details)
                verification_result = old_details.get('fact_check')
            except:
                pass
            
        # Re-detect
        is_fake, new_score, breakdown = detect_fake_news(
            title, 
            content, 
            model=model, 
            tokenizer=tokenizer, 
            source=a.source,
            verification_result={"fact_check": verification_result} if verification_result else None
        )
        
        # Preserve existing fact_check data if it existed
        if verification_result:
            breakdown['fact_check'] = verification_result

        # Update the article
        a.is_fake = is_fake
        a.credibility_score = new_score
        a.score_details = json.dumps(breakdown)
        
        updated_count += 1
        safe_title = (a.title or "").encode('ascii', 'ignore').decode()
        print(f"[{updated_count}/{len(all_articles)}] Updated ID {a.id} | Score: {new_score:.2f} | {safe_title[:40]}...")
            
    if updated_count > 0:
        session.commit()
        print("\n" + "=" * 60)
        print(f"Successfully updated explanations for {updated_count} articles.")
        
    session.close()

if __name__ == "__main__":
    reprocess_all()
