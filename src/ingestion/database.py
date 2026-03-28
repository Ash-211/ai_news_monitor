import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
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
    
    # Layer 4: Summarization
    summary_extractive = Column(Text, nullable=True)
    summary_abstractive = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Article(title='{self.title[:30]}...', source='{self.source}')>"

def get_engine():
    """
    Initializes and returns the database engine.
    For this project, we are using a local SQLite database in the data/ folder.
    """
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    db_path = 'sqlite:///data/database.sqlite'
    engine = create_engine(db_path, echo=False)
    return engine

def init_db():
    """
    Creates all tables in the SQLite database based on defined models.
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
