from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.ingestion.database import Article
import os
from src.intelligence.fake_news import (
    TRUSTED_SOURCES, load_fake_news_detector, explain_prediction, analyze_linguistic_style, detect_fake_news
)
from src.intelligence.keyword_extractor import extract_keywords
from src.intelligence.fact_checker import verify_article


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

    # Load machine learning model for explanations
    model = load_fake_news_detector()
    
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
            # Call `detect_fake_news` with dual-scoring to retrieve the transparent breakdown
            # We don't overwrite `credibility_score` from DB, we just grab the dictionary.
            content_to_analyze = a.clean_content[:500] if a.clean_content else (a.raw_content[:500] if a.raw_content else "")
            
            # Check corroboration
            corr_count = 0
            if a.topic_cluster is not None:
                corr_count = session.query(Article).filter(
                    Article.topic_cluster == a.topic_cluster,
                    Article.id != a.id,
                    Article.source.in_(TRUSTED_SOURCES)
                ).count()
                
            is_fake_calc, final_score, breakdown = detect_fake_news(a.title, content_to_analyze, model=model, source=a.source, corroboration_count=corr_count)
            
            # Run external fact-check for "unsure" articles (saves API quota)
            verification_data = None
            recalc_score = breakdown.get("base_combined", 0.5)
            if 0.3 <= recalc_score <= 0.6:
                try:
                    verification_data = verify_article(a.title or "")
                    if verification_data.get("verification_score", 0.5) != 0.5:
                        is_fake_calc, final_score, breakdown = detect_fake_news(
                            a.title, content_to_analyze, model=model, 
                            source=a.source, corroboration_count=corr_count,
                            verification_result=verification_data
                        )
                except Exception:
                    pass  # Graceful fallback if API fails
            
            is_trusted = breakdown.get("source_boost", 0) > 0
            
            # Get AI Reasoning (Top Keywords and Style)
            ai_explanation = {"trust_terms": [], "risk_terms": []}
            linguistic_style = {"sensationalism": "Normal"}
            if model:
                ai_explanation = explain_prediction(content_to_analyze, model=model, top_n=3)
            linguistic_style = analyze_linguistic_style(a.title or "")

            items.append({
                "id": a.id,
                "title": a.title,
                "url": a.url,
                "source": a.source,
                "author": a.author,
                "published_at": a.published_at,
                "category": a.category,
                "is_fake": a.is_fake if a.is_fake is not None else is_fake_calc,
                "credibility_score": a.credibility_score if a.credibility_score is not None else final_score,
                "topic_cluster": a.topic_cluster,
                "score_details": {
                    "base_ml": breakdown.get("base_combined", 0.5),
                    "headline_score": breakdown.get("headline_score", 0.5),
                    "content_score": breakdown.get("content_score", 0.5),
                    "is_trusted": is_trusted,
                    "corroboration_count": corr_count,
                    "corroboration_boost": breakdown.get("corroboration_boost", 0),
                    "bonus_applied": breakdown.get("source_boost", 0) + breakdown.get("corroboration_boost", 0),
                    "verification_boost": breakdown.get("verification_boost", 0),
                    "penalty_applied": breakdown.get("penalty", 0),
                    "penalty_reasons": breakdown.get("penalty_reasons", []),
                    "fact_check": breakdown.get("fact_check"),
                    "ai_logic": {
                        "trust_keywords": ai_explanation["trust_terms"],
                        "risk_keywords": ai_explanation["risk_terms"],
                        "sensationalism_score": linguistic_style["sensationalism_score"],
                        "objectivity_score": linguistic_style["objectivity_score"]
                    }
                },
                "keywords": a.keywords,
                "summary": a.clean_content[:200] + "..." if a.clean_content else (a.raw_content[:200] + "..." if a.raw_content else ""),
                "full_content": a.clean_content or a.raw_content or "Content not available for this article."
            })
            
        return {
            "items": items,
            "total": total,
            "page": page,
            "pages": (total + limit - 1) // limit
        }
    finally:
        session.close()


# ─── Intelligence Pipeline Trigger (Optional Internal) ─────────────
