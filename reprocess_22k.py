import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.ingestion.database import get_session, Article, init_db
from src.intelligence.pipeline import run_intelligence_pipeline

# Configure how many latest articles to reprocess (set to 10 for quick testing)
TARGET_LIMIT = 10

def reprocess_latest():
    # Upgrade schema first (adds any missing columns like embedding)
    init_db()
    
    session = get_session()
    print("==================================================")
    print(f"  TARGETED REPROCESSING SCRIPT (Latest {TARGET_LIMIT} Articles)")
    print("==================================================")
    
    # 1. Fetch only the latest TARGET_LIMIT articles by date
    latest_rows = session.query(Article.id).order_by(
        Article.published_at.desc().nullslast(), 
        Article.id.desc()
    ).limit(TARGET_LIMIT).all()
    
    target_ids = [r[0] for r in latest_rows]
    print(f"Found {len(target_ids)} latest articles to reset for reprocessing...")

    if not target_ids:
        print("No articles found in database.")
        session.close()
        return

    # 2. Reset intelligence fields ONLY for these latest articles
    session.query(Article).filter(Article.id.in_(target_ids)).update({
        Article.is_fake: None, 
        Article.credibility_score: None, 
        Article.score_details: None,
        Article.image_status: 'pending' # Reset so Deepfake scanner reruns
    }, synchronize_session=False)
    session.commit()
    print(f"Successfully reset intelligence fields for the {len(target_ids)} latest articles.")

    # 3. Run the pipeline in chunks until target_limit is reached
    processed = 0
    total = len(target_ids)
    while processed < total:
        print(f"\n--- Processing next chunk ({processed} out of {total} done) ---")
        
        count = run_intelligence_pipeline(batch_size=TARGET_LIMIT)
        if count == 0:
            print("\nNo more articles pending processing!")
            break
            
        processed += count
        if processed >= total:
            print(f"\nReached target limit of {total} articles!")
            break

        print("Cooling down for 2 seconds...")
        time.sleep(2)
        
    session.close()
    print(f"\n==================================================")
    print(f"  Finished reprocessing {processed} latest articles successfully!")
    print(f"==================================================")
    
if __name__ == "__main__":
    reprocess_latest()
