import time
from src.ingestion.fetcher import run_ingestion
from src.ingestion.database import init_db
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
    
    # Step 1: Fetch new articles from RSS & NewsAPI
    run_ingestion()
    
    # Step 2: Ensure ML models are available (download from HuggingFace if needed)
    try:
        from src.intelligence.download_models import download_models
        models_available = download_models()
    except Exception as e:
        print(f"[WARN] Model download check failed: {e}")
        models_available = False
    
    # Step 3: Run intelligence pipeline (ML models)
    if models_available:
        try:
            from src.intelligence.pipeline import run_intelligence_pipeline
            run_intelligence_pipeline()
            print("[OK] Intelligence pipeline completed.")
        except Exception as e:
            print(f"[WARN] Intelligence pipeline failed: {e}")
            print("       Articles were saved but ML processing was not applied.")
    else:
        print("[WARN] ML models not available. Skipping intelligence pipeline.")
        print("       Articles were saved. Run 'python -m src.intelligence.pipeline' locally.")
    
    # Step 4: Reprocess fake news scores
    if models_available:
        try:
            from src.maintenance.reprocess_fakes import reprocess_fakes
            reprocess_fakes()
        except Exception as e:
            print(f"[WARN] Reprocess fakes skipped: {e}")
    
    # Step 5: Sync full-text search index
    sync_fts()
    
    elapsed = time.time() - start_time
    print(f"\nDaily pipeline completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_once()
