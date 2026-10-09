import time
import schedule
import requests
from src.ingestion.fetcher import run_ingestion
from src.ingestion.database import init_db
from src.maintenance.reprocess_fakes import reprocess_fakes

def sync_fts():
    """Tell the API to refresh its FTS index and bust its cache."""
    try:
        requests.post("http://localhost:8000/api/refresh-fts", timeout=5)
        print("  [FTS] Search index synced.")
    except Exception:
        pass  # API might not be running yet

from src.intelligence.pipeline import run_intelligence_pipeline

def run_trending_ingestion():
    from src.ingestion.trending_scraper import scan_all_platforms
    print("  [Scheduler] Running scheduled trending ingestion...")
    result = scan_all_platforms(reddit_limit=5, gnews_limit=10)
    import json
    import os
    os.makedirs("data", exist_ok=True)
    with open("data/latest_trending.json", "w") as f:
        json.dump(result, f)
    print("  [Scheduler] Trending ingestion complete and saved.")

def run_jobs():
    run_ingestion()
    run_intelligence_pipeline()
    run_trending_ingestion()
    reprocess_fakes()
    sync_fts()

def main():
    print("Initializing Database...")
    init_db()
    
    print("Setting up scheduler...")
    # Schedule the ingestion job to run every hour
    schedule.every(1).hours.do(run_jobs)
    schedule.every(60).seconds.do(reprocess_fakes)
    # Run once immediately at startup
    run_jobs()
    
    print("Scheduler is now running. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60) # check every minute
    except KeyboardInterrupt:
        print("Scheduler stopped manually.")

if __name__ == "__main__":
    main()
