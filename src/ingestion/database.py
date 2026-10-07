import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Define base class for SQLAlchemy models
Base = declarative_base()

class Article(Base):
    """
    Representation of a news article in the database.
    Stores raw ingestion data as well as placeholders for downstream analytics.
    """
    __tablename__ = 'articles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    url = Column(String, unique=True, nullable=False)
    source = Column(String, nullable=True)
    author = Column(String, nullable=True)
    published_at = Column(DateTime, default=datetime.utcnow)
    
    # Text Content
    raw_content = Column(Text, nullable=True)
    clean_content = Column(Text, nullable=True)
    
    # Layer 2: Image Provenance (Deepfake Pipeline)
    image_url = Column(String, nullable=True)              # Hero image URL (og:image / top_image)
    image_status = Column(String, default='pending')       # pending | real | deepfake | discarded | no_image
    deepfake_score = Column(Float, nullable=True)          # Sightengine/C2PA deepfake probability (0.0-1.0)
    
    # Layer 3: Pipeline Results (Intelligence)
    category = Column(String, nullable=True)           # e.g., Tech, Finance, Politics
    is_fake = Column(Boolean, nullable=True)           # Fake news flag
    topic_cluster = Column(Integer, nullable=True)     # LDA cluster mapping
    credibility_score = Column(Float, nullable=True)    # Fake news confidence (0.0-1.0)
    score_details = Column(Text, nullable=True)         # JSON string for XAI explanation
    keywords = Column(Text, nullable=True)              # Comma-separated TF-IDF keywords
    embedding = Column(Vector(384), nullable=True)
    
    # Layer 4: Summarization
    summary_extractive = Column(Text, nullable=True)
    summary_abstractive = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Article(title='{self.title[:30]}...', source='{self.source}')>"


class SocialMediaPost(Base):
    """
    Representation of a social media post (Reddit, Twitter, Facebook) in the database.
    Stores raw ingestion data, deepfake image provenance, and fact-check scores.
    """
    __tablename__ = 'social_media_posts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    url = Column(String, unique=True, nullable=False)
    source = Column(String, nullable=True)             # e.g., Reddit, Twitter
    author = Column(String, nullable=True)
    published_at = Column(DateTime, default=datetime.utcnow)
    
    # Text Content
    raw_content = Column(Text, nullable=True)
    clean_content = Column(Text, nullable=True)
    
    # Layer 2: Image Provenance (Deepfake Pipeline)
    image_url = Column(String, nullable=True)
    image_status = Column(String, default='pending')
    deepfake_score = Column(Float, nullable=True)
    
    # Layer 3: Pipeline Results (Intelligence)
    category = Column(String, nullable=True)
    is_fake = Column(Boolean, nullable=True)
    topic_cluster = Column(Integer, nullable=True)
    credibility_score = Column(Float, nullable=True)
    score_details = Column(Text, nullable=True)
    keywords = Column(Text, nullable=True)
    
    # Semantic Linking (To connect to News Articles)
    embedding = Column(Vector(384), nullable=True)    # all-MiniLM-L6-v2 embeddings are 384-dimensional

    def __repr__(self):
        return f"<SocialMediaPost(title='{self.title[:30]}...', source='{self.source}')>"


class DiscordSubscription(Base):
    """
    Tracks which Discord channels are subscribed to receive
    the automated daily news drops from the bot.
    """
    __tablename__ = 'discord_subscriptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    server_id = Column(String, nullable=False)          # Discord Guild ID
    channel_id = Column(String, unique=True, nullable=False)  # Discord Channel ID
    category = Column(String, nullable=True)            # Optional: filter by category (e.g., "Sci/Tech")
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<DiscordSubscription(server={self.server_id}, channel={self.channel_id})>"


def _is_postgres():
    """Check if the configured database is PostgreSQL."""
    db_url = os.getenv('DATABASE_URL', '')
    return db_url.startswith('postgresql')


def get_engine():
    """
    Initializes and returns the database engine.
    Uses DATABASE_URL from .env if available (Neon PostgreSQL),
    otherwise falls back to local SQLite.
    """
    db_url = os.getenv('DATABASE_URL', '').strip()

    if db_url and db_url.startswith('postgresql'):
        # Cloud PostgreSQL (Neon) with auto-reconnect and keepalives to prevent SSL timeout drops
        engine = create_engine(
            db_url, 
            echo=False, 
            pool_pre_ping=True, 
            pool_recycle=300,
            connect_args={
                'keepalives': 1,
                'keepalives_idle': 30,
                'keepalives_interval': 10,
                'keepalives_count': 5
            }
        )
        return engine

    # Fallback: Local SQLite
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    db_path = f"sqlite:///{os.path.join(data_dir, 'database.sqlite')}"
    engine = create_engine(db_path, echo=False, connect_args={'timeout': 15})
    return engine

def init_db():
    """
    Creates all tables in the database based on defined models.
    Works for both SQLite and PostgreSQL.
    """
    engine = get_engine()
    
    # Auto-enable pgvector extension on NeonDB
    if _is_postgres():
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Ensure embedding columns exist (in case tables were created before pgvector was enabled)
            conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'articles' AND column_name = 'embedding'
                    ) THEN
                        ALTER TABLE articles ADD COLUMN embedding VECTOR(384);
                    END IF;
                END $$;
            """))

    Base.metadata.create_all(engine)
    print("Database initialized successfully.")

def get_session():
    """
    Returns a new database session instance.
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

if __name__ == "__main__":
    # Test initialization
    init_db()
