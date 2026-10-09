import os
import time
from dotenv import load_dotenv
from src.ingestion.database import get_session, Article, init_db
from src.intelligence.pipeline import run_intelligence_pipeline

# Load .env variables (including HF_TOKEN)
load_dotenv()

# Ensure env vars are set for this run
os.environ["HF_SPACE_FACTCHECK_URL"] = "https://SRG4545-llm-setup.hf.space"
os.environ["DATABASE_URL"] = "postgresql://neondb_owner:npg_5cM6sbpAhfgj@ep-sparkling-sound-aqch9rll.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require"

def reprocess_100():
    init_db()
    session = get_session()
    print("==================================================")
    print("  REPROCESSING 100 ARTICLES SCRIPT")
    print("==================================================")
    
    # Get 100 articles to reprocess
    articles = session.query(Article).order_by(Article.published_at.desc().nulls_last()).limit(100).all()
    
    print(f"Resetting {len(articles)} recent articles to unprocessed state...")
    for a in articles:
        a.is_fake = None
        a.credibility_score = None
        a.score_details = None
        a.image_status = 'pending'
    
    session.commit()
    print("Reset complete. Running intelligence pipeline for these articles...")
    
    # run_intelligence_pipeline() processes up to 100 by default.
    count = run_intelligence_pipeline()
    
    print(f"\n==================================================")
    print(f"  Finished reprocessing {count} articles with the new pipeline!")
    print(f"==================================================")

if __name__ == "__main__":
    reprocess_100()
