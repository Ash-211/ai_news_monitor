import time
import schedule
from src.ingestion.fetcher import run_ingestion
from src.ingestion.database import init_db

def main():
    print("Initializing Database...")
    init_db()
    
    print("Setting up scheduler...")
    # Schedule the ingestion job to run every 15 minutes (for demo purposes)
    schedule.every(15).minutes.do(run_ingestion)
    
    # Run once immediately at startup
    run_ingestion()
    
    print("Scheduler is now running. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60) # check every minute
    except KeyboardInterrupt:
        print("Scheduler stopped manually.")

if __name__ == "__main__":
    main()
