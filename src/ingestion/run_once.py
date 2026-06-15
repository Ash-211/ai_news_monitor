import time
from src.ingestion.fetcher import run_ingestion
from src.ingestion.database import init_db
from src.maintenance.reprocess_fakes import reprocess_fakes
from src.intelligence.pipeline import run_intelligence_pipeline
import requests

def sync_fts():
    """Tell the API to refresh its FTS index and bust its cache."""
    try:
        requests.post("http://localhost:8000/api/refresh-fts", timeout=5)
        print("  [FTS] Search index synced.")
    except Exception:
        pass

def run_once():
    print("Initializing Database...")
    init_db()
    
    print("Starting daily ingestion and intelligence pipeline...")
    start_time = time.time()
    
    run_ingestion()
    run_intelligence_pipeline()
    reprocess_fakes()
    sync_fts()
    
    elapsed = time.time() - start_time
    print(f"Daily pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_once()
