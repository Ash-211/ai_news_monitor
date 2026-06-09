import os
import sys
from dotenv import load_dotenv
from sqlalchemy import text

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.database import get_session, Article, get_engine

load_dotenv()

def test_connection():
    db_url = os.getenv('DATABASE_URL', '')
    print(f"DATABASE_URL starts with: {db_url[:20]}..." if db_url else "DATABASE_URL is not set!")
    
    try:
        engine = get_engine()
        print(f"Engine built successfully: {engine.url}")
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("Successfully connected and executed query: SELECT 1")
            
            # Check tables
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = conn.execute(tables_query).fetchall()
            print("Tables in database:", [t[0] for t in tables])
            
            # If articles table exists, get some stats
            if 'articles' in [t[0] for t in tables]:
                count = conn.execute(text("SELECT COUNT(*) FROM articles")).scalar()
                print(f"Total articles in 'articles' table: {count}")
                
                recent = conn.execute(text("SELECT title, published_at, source FROM articles ORDER BY published_at DESC LIMIT 5")).fetchall()
                print("\nMost recent 5 articles:")
                for r in recent:
                    print(f"- {r[0]} | Pub: {r[1]} | Source: {r[2]}")
            else:
                print("WARNING: 'articles' table does not exist in public schema.")
                
    except Exception as e:
        print(f"ERROR: Database connection or query failed: {e}")

if __name__ == '__main__':
    test_connection()
