import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker

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
    
    # Layer 3: Pipeline Results (Intelligence)
    category = Column(String, nullable=True)           # e.g., Tech, Finance, Politics
    is_fake = Column(Boolean, nullable=True)           # Fake news flag
    topic_cluster = Column(Integer, nullable=True)     # LDA cluster mapping
    credibility_score = Column(Float, nullable=True)    # Fake news confidence (0.0-1.0)
    keywords = Column(Text, nullable=True)              # Comma-separated TF-IDF keywords
    
    # Layer 4: Summarization
    summary_extractive = Column(Text, nullable=True)
    summary_abstractive = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Article(title='{self.title[:30]}...', source='{self.source}')>"

class DiscordSubscription(Base):
    """
    Represents a Discord channel subscription to daily automated news drops.
    """
    __tablename__ = 'discord_subscriptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    server_id = Column(String, nullable=False)
    channel_id = Column(String, nullable=False)
    category = Column(String, default="all")

    def __repr__(self):
        return f"<DiscordSubscription(server_id='{self.server_id}', channel_id='{self.channel_id}', category='{self.category}')>"

def get_engine():
    """
    Initializes and returns the database engine.
    Supports DATABASE_URL env var for PostgreSQL integration, falling back to local SQLite.
    """
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url, echo=False)

    # Fallback to local SQLite
    os.makedirs('data', exist_ok=True)
    db_path = 'sqlite:///data/database.sqlite'
    return create_engine(db_path, echo=False)

def init_db():
    """
    Creates all tables in the database based on defined models.
    """
    engine = get_engine()
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

