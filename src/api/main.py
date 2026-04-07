from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.ingestion.database import Article
import os
from pydantic import BaseModel

app = FastAPI(title="AI News API", description="API serving intelligence-processed news articles.")

# Enable CORS for React frontend (Vite defaults to 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_session():
    db_path = 'sqlite:///data/database.sqlite'
    if not os.path.exists('data/database.sqlite'):
        return None
    engine = create_engine(db_path, echo=False)
    Session = sessionmaker(bind=engine)
    return Session()

@app.get("/api/stats")
def get_stats():
    session = get_db_session()
    if not session:
         return {"error": "Database not found"}
    try:
        total = session.query(Article).count()
        fake_count = session.query(Article).filter(Article.is_fake == True).count()
        real_count = session.query(Article).filter(Article.is_fake == False).count()
        categories = {}
        for category in session.query(Article.category).distinct():
            if category[0]:
                count = session.query(Article).filter(Article.category == category[0]).count()
                categories[category[0]] = count
                
        return {
            "total_articles": total,
            "fake_articles": fake_count,
            "real_articles": real_count,
            "categories": categories
        }
    finally:
        session.close()

@app.get("/api/articles")
def get_articles(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    category: str = None,
    is_fake: bool = None,
    search: str = None
):
    session = get_db_session()
    if not session:
         return {"items": [], "total": 0, "page": page, "limit": limit}
    
    try:
        query = session.query(Article)
        
        if category:
            query = query.filter(Article.category == category)
        if is_fake is not None:
            query = query.filter(Article.is_fake == is_fake)
        if search:
            query = query.filter(Article.title.ilike(f"%{search}%"))
            
        total = query.count()
        articles = query.order_by(Article.published_at.desc()).offset((page - 1) * limit).limit(limit).all()
        
        items = []
        for a in articles:
            items.append({
                "id": a.id,
                "title": a.title,
                "url": a.url,
                "source": a.source,
                "author": a.author,
                "published_at": a.published_at,
                "category": a.category,
                "is_fake": a.is_fake,
                "credibility_score": a.credibility_score,
                "topic_cluster": a.topic_cluster,
                "keywords": a.keywords,
                "summary": a.clean_content[:200] + "..." if a.clean_content else (a.raw_content[:200] + "..." if a.raw_content else "")
            })
            
        return {
            "items": items,
            "total": total,
            "page": page,
            "pages": (total + limit - 1) // limit
        }
    finally:
        session.close()
