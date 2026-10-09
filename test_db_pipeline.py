import os
import psycopg2
import json
from psycopg2.extras import RealDictCursor

# Set environment variables for the test
os.environ["HF_SPACE_FACTCHECK_URL"] = "https://SRG4545-llm-setup.hf.space"
os.environ["DATABASE_URL"] = "postgresql://neondb_owner:npg_5cM6sbpAhfgj@ep-sparkling-sound-aqch9rll.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require"

from src.ingestion.trending_scraper import analyze_trending_item

def run_test():
    print("Connecting to Neon database...")
    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT * FROM articles LIMIT 2;")
        articles = cur.fetchall()
        
        if not articles:
            print("No articles found in the database!")
            return
            
        print(f"Found {len(articles)} articles. Running pipeline without saving to DB...")
        
        for art in articles:
            print("\n" + "="*80)
            print(f"Testing Pipeline on DB Article: {art.get('title')}")
            print(f"Source: {art.get('source')} | Snippet: {str(art.get('snippet'))[:50]}...")
            print("="*80)
            
            # Convert to dictionary (like what the scraper outputs)
            item = dict(art)
            
            # The model arguments can be None, the function will load them lazily
            print("\nExecuting Pipeline (RAG Verification -> DistilBERT -> Qwen 1.5B)...")
            result = analyze_trending_item(item)
            
            print("\n--- Pipeline Analysis Result ---")
            print(json.dumps(result.get('analysis'), indent=2, default=str))
            
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        if 'cur' in locals() and cur: cur.close()
        if 'conn' in locals() and conn: conn.close()
        print("\nTest complete. Database was not modified.")

if __name__ == "__main__":
    run_test()
