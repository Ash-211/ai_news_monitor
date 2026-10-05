import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup environment
os.environ["DATABASE_URL"] = "postgresql://neondb_owner:npg_M0ulAbo1kgYO@ep-hidden-shape-b4gan4iu-pooler.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require"

from src.ingestion.database import Article
from src.intelligence.fake_news import detect_batch, load_fake_news_detector

def test_db_dataset():
    engine = create_engine(os.environ["DATABASE_URL"])
    Session = sessionmaker(bind=engine)
    session = Session()
    
    print("Checking database for articles...")
    articles = session.query(Article).limit(10).all()
    
    if not articles:
        print("❌ No articles found in the 'articles' table of the provided dataset!")
        return
        
    print(f"Found {len(articles)} articles. Running Hybrid Fake News Detection...")
    
    model, tokenizer = load_fake_news_detector()
    
    # Format for detect_batch
    items = []
    for art in articles:
        items.append({
            "title": art.title,
            "content": art.clean_content or art.raw_content or art.title,
            "source": art.source,
            "platform": "db"
        })
        
    # Fetch evidence and run batch
    from src.ingestion.trending_scraper import cross_validate_claim
    for i, item in enumerate(items):
        cv = cross_validate_claim(item["title"])
        item["verification"] = cv.get("evidence_string", "No major news outlets are reporting this.")
        
    print("\nRunning AI Ensemble...")
    analyzed_items = detect_batch(items, model, tokenizer)
    
    print("\n============================================================")
    print("  RESULTS ON YOUR DATASET")
    print("============================================================\n")
    for item in analyzed_items:
        score = item.get("analysis", {}).get("credibility_score", 0)
        verdict = item.get("analysis", {}).get("verdict", "Unknown")
        explanation = item.get("analysis", {}).get("explanation", "")
        print(f"[Score: {int(score*100)}%] [{verdict}]")
        print(f"Title: {item['title']}")
        print(f"Explanation: {explanation}\n")
        
if __name__ == "__main__":
    test_db_dataset()
