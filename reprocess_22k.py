import time
from src.ingestion.database import get_session, Article, init_db
from src.intelligence.pipeline import run_intelligence_pipeline

def reprocess_all():
    # Upgrade schema first (adds the embedding column if it's missing)
    init_db()
    
    session = get_session()
    print("==================================================")
    print("  MASS REPROCESSING SCRIPT")
    print("==================================================")
    print("Resetting old articles so they get picked up by the new LLM pipeline...")
    
    # 1. Reset the intelligence fields for all articles
    # We set is_fake = None so the pipeline sees them as "unprocessed"
    session.query(Article).update({
        Article.is_fake: None, 
        Article.credibility_score: None, 
        Article.score_details: None,
        Article.image_status: 'pending' # Reset image status so Deepfake scanner reruns
    })
    session.commit()
    
    total = session.query(Article).count()
    print(f"Successfully reset {total} articles in the database.")
    
    # 2. Run the pipeline in safe chunks of 100
    processed = 0
    while True:
        print(f"\n--- Processing next chunk ({processed} out of {total} done) ---")
        
        # run_intelligence_pipeline() now safely limits itself to 100 articles
        count = run_intelligence_pipeline()
        
        if count == 0:
            print("\nNo more articles to process!")
            break
            
        processed += count
        
        # Give Hugging Face a 3-second cooldown to avoid Rate Limit (429) bans
        print("Cooling down for 3 seconds to respect Hugging Face Free API limits...")
        time.sleep(3)
        
    print(f"\n==================================================")
    print(f"  Finished reprocessing all {total} articles with new 50/25/25 logic!")
    print(f"==================================================")
    
if __name__ == "__main__":
    reprocess_all()
