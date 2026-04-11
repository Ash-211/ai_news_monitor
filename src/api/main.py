from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.ingestion.database import Article
import os
from typing import Optional
from pydantic import BaseModel
from src.intelligence.fake_news import (
    TRUSTED_SOURCES, FAKE_THRESHOLD,
    load_fake_news_detector, detect_fake_news,
    explain_prediction, analyze_linguistic_style
)
from src.intelligence.keyword_extractor import extract_keywords


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
            # Calculate breakdown on the fly
            is_trusted = False
            if a.source:
                for ts in TRUSTED_SOURCES:
                    if ts.lower() in a.source.lower():
                        is_trusted = True
                        break
            
            # Check for corroboration (other articles in same topic cluster from trusted sources)
            corr_count = 0
            if a.topic_cluster is not None:
                corr_count = session.query(Article).filter(
                    Article.topic_cluster == a.topic_cluster,
                    Article.id != a.id,
                    Article.source.in_(TRUSTED_SOURCES)
                ).count()

            # Reverse engineer the base score
            # Score = ML + (0.15 if trusted else 0) + (0.10 if corr >= 1 else 0) - (0.15 if not trusted and corr==0 else 0)
            bonus = 0
            if is_trusted: bonus += 0.15
            if corr_count >= 1: bonus += 0.10
            if not is_trusted and corr_count == 0: bonus -= 0.15
            
            base_score = (a.credibility_score or 0.5) - bonus
            base_score = max(0.0, min(1.0, base_score))

            # New: Get AI Reasoning (Top Keywords and Style)
            ai_explanation = {"trust_terms": [], "risk_terms": []}
            linguistic_style = {"sensationalism": "Normal"}
            
            content_to_analyze = a.title + " " + (a.clean_content[:500] if a.clean_content else "")
            if model:
                ai_explanation = explain_prediction(content_to_analyze, model=model, top_n=3)
            linguistic_style = analyze_linguistic_style(content_to_analyze)

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
                "score_details": {
                    "base_ml": round(base_score, 4),
                    "is_trusted": is_trusted,
                    "corroboration_count": corr_count,
                    "bonus_applied": round(bonus, 4),
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


# ─── Live Article Analysis Endpoint ───────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    source: Optional[str] = None


@app.post("/api/analyze")
def analyze_article(req: AnalyzeRequest):
    """
    Accepts any article text (and optional source name) and returns 
    a full fake-news analysis with AI reasoning breakdown.
    """
    text = (req.text or "").strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Article text must be at least 20 characters.")

    model = load_fake_news_detector()
    if model is None:
        raise HTTPException(status_code=503, detail="Fake news model not loaded. Train it first.")

    source = req.source or "Unknown Source"

    # 1. Core detection
    is_fake, credibility_score = detect_fake_news(text, model=model, source=source, corroboration_count=0)

    # 2. AI explanation (top keywords)
    explanation = explain_prediction(text, model=model, top_n=4)

    # 3. Linguistic style analysis
    style = analyze_linguistic_style(text)

    # 4. Keyword extraction
    try:
        kw_list = extract_keywords(text, top_n=8)
    except Exception:
        kw_list = []

    # 5. Source trust check
    is_trusted = False
    if source:
        for ts in TRUSTED_SOURCES:
            if ts.lower() in source.lower():
                is_trusted = True
                break

    # 6. Compute heuristic breakdown for transparency
    probabilities = model.predict_proba([text])[0]
    base_ml_score = float(probabilities[0])  # raw ML confidence of being real

    source_bonus = 0.25 if is_trusted else 0.0
    corr_bonus = 0.0  # no corroboration for manually submitted articles
    isolation_penalty = -0.15 if (not is_trusted and True) else 0.0  # always applies for manual

    return {
        "verdict": "Potentially Misleading" if is_fake else "Authentic",
        "is_fake": is_fake,
        "credibility_score": round(credibility_score, 4),
        "credibility_percent": round(credibility_score * 100, 1),
        "threshold": FAKE_THRESHOLD,
        "source": source,
        "is_trusted_source": is_trusted,
        "score_breakdown": {
            "base_ml_score": round(base_ml_score, 4),
            "base_ml_percent": round(base_ml_score * 100, 1),
            "source_bonus": source_bonus,
            "corroboration_bonus": corr_bonus,
            "isolation_penalty": isolation_penalty,
        },
        "ai_reasoning": {
            "trust_keywords": explanation.get("trust_terms", []),
            "risk_keywords": explanation.get("risk_terms", []),
            "sensationalism_score": style.get("sensationalism_score", 0),
            "objectivity_score": style.get("objectivity_score", 100),
            "caps_ratio": style.get("caps_ratio", 0),
            "punctuation_count": style.get("punc_count", 0),
        },
        "keywords": kw_list,
    }
