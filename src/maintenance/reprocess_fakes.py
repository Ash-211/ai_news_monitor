import os
import sys
import sqlite3

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ingestion.database import get_session, Article
from src.intelligence.fake_news import detect_fake_news, load_fake_news_detector, TRUSTED_SOURCES

def reprocess_trusted_fakes():
    """
    Finds articles from trusted sources currently marked as fake
    and re-runs the detection logic with updated heuristics.
    """
    session = get_session()
    model = load_fake_news_detector()
    
    if not model:
        print("Error: Fake news detector model not found.")
        return

    print("\n--- Reprocessing Trusted Source Fakes ---")
    
    # Query for articles from trusted sources where is_fake is True
    # We use a broad LIKE filter first for performance
    search_terms = ['ndtv', 'hindu', 'express', 'times', 'bbc', 'reuters']
    
    query = session.query(Article).filter(Article.is_fake == True)
    
    articles_to_fix = []
    all_fakes = query.all()
    
    print(f"Found {len(all_fakes)} total articles marked as fake.")
    
    for a in all_fakes:
        if not a.source:
            continue
            
        is_trusted = False
        source_clean = a.source.lower()
        for ts in TRUSTED_SOURCES:
            if ts.lower() in source_clean:
                is_trusted = True
                break
        
        if is_trusted:
            articles_to_fix.append(a)
            
    if not articles_to_fix:
        print("No articles from trusted sources are currently marked as fake.")
        return

    print(f"Identified {len(articles_to_fix)} articles from trusted sources to re-process.")
    print("=" * 60)

    updated_count = 0
    for a in articles_to_fix:
        old_score = a.credibility_score
        
        # Build text (consistent with pipeline.py)
        title = a.title or ''
        content = a.clean_content or a.raw_content or ''
            
        # Re-detect
        # Note: We need corroboration_count if available, but for now we'll use 0 
        # as it's the safest default for a single-article re-run without full topic context.
        is_fake, new_score, _ = detect_fake_news(title, content, model=model, source=a.source)
        
        if is_fake != a.is_fake or abs(new_score - (old_score or 0)) > 0.01:
            status_change = "FAKE -> REAL" if not is_fake else "STILL FAKE"
            safe_title = (a.title or "").encode('ascii', 'ignore').decode()
            print(f"ID {a.id}: {status_change} | Score: {old_score:.2f} -> {new_score:.2f} | {safe_title[:50]}...")
            
            a.is_fake = is_fake
            a.credibility_score = new_score
            updated_count += 1
            
    if updated_count > 0:
        session.commit()
        print("\n" + "=" * 60)
        print(f"Successfully updated {updated_count} articles.")
    else:
        print("\nNo status changes detected after re-analysis.")
        
    session.close()

if __name__ == "__main__":
    reprocess_trusted_fakes()
