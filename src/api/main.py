from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from src.ingestion.database import Article
import os, time, hashlib, json

# ── Singleton engine & session factory (created ONCE at import time) ──────────
def _build_engine():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    db_file = os.path.join(base_dir, 'data', 'database.sqlite')
    if not os.path.exists(db_file):
        return None, None
    engine = create_engine(
        f"sqlite:///{db_file}",
        echo=False,
        connect_args={"timeout": 15, "check_same_thread": False},
    )
    # Create indexes + FTS5 table once at startup
    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_category      ON articles (category)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_is_fake       ON articles (is_fake)"))
        # FTS5 virtual table for blazing-fast full-text search
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                title, keywords, source,
                content='articles',
                content_rowid='id'
            )
        """))
        # Populate FTS index from existing data (only inserts missing rows)
        conn.execute(text("""
            INSERT OR IGNORE INTO articles_fts(rowid, title, keywords, source)
            SELECT id, COALESCE(title,''), COALESCE(keywords,''), COALESCE(source,'')
            FROM articles
            WHERE id NOT IN (SELECT rowid FROM articles_fts)
        """))
        conn.commit()
    return engine, sessionmaker(bind=engine)

_engine, _SessionFactory = _build_engine()

# ── Simple in-memory cache with TTL ──────────────────────────────────────────
_cache = {}
_CACHE_TTL = 30  # seconds

def _cache_key(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        return entry["data"]
    return None

def _cache_set(key: str, data):
    # Evict old entries if cache gets too big (keep last 200)
    if len(_cache) > 200:
        oldest = sorted(_cache, key=lambda k: _cache[k]["ts"])[:100]
        for k in oldest:
            del _cache[k]
    _cache[key] = {"data": data, "ts": time.time()}

@contextmanager
def get_session():
    if _SessionFactory is None:
        yield None
        return
    session: Session = _SessionFactory()
    try:
        yield session
    finally:
        session.close()

# ── Keep FTS in sync: call after ingestion inserts new articles ───────────────
def refresh_fts():
    """Sync the FTS5 index with any newly inserted articles."""
    if _engine is None:
        return
    with _engine.connect() as conn:
        conn.execute(text("""
            INSERT OR IGNORE INTO articles_fts(rowid, title, keywords, source)
            SELECT id, COALESCE(title,''), COALESCE(keywords,''), COALESCE(source,'')
            FROM articles
            WHERE id NOT IN (SELECT rowid FROM articles_fts)
        """))
        conn.commit()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI News API", description="API serving intelligence-processed news articles.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Stats ─────────────────────────────────────────────────────────────────────
@app.get("/api/stats")
def get_stats():
    ck = _cache_key({"endpoint": "stats"})
    cached = _cache_get(ck)
    if cached:
        return cached

    with get_session() as session:
        if session is None:
            return {"error": "Database not found"}
        total = session.query(Article).count()
        fake_count  = session.query(Article).filter(Article.is_fake == True).count()
        real_count  = session.query(Article).filter(Article.is_fake == False).count()
        categories  = {}
        for (cat,) in session.query(Article.category).distinct():
            if cat:
                categories[cat] = session.query(Article).filter(Article.category == cat).count()
        result = {
            "total_articles": total,
            "fake_articles":  fake_count,
            "real_articles":  real_count,
            "categories":     categories,
        }
        _cache_set(ck, result)
        return result

# ── Articles ──────────────────────────────────────────────────────────────────
def _serialize_article(a):
    score = a.credibility_score if a.credibility_score is not None else 0.5
    return {
        "id":               a.id,
        "title":            a.title,
        "url":              a.url,
        "source":           a.source,
        "author":           a.author,
        "published_at":     str(a.published_at) if a.published_at else None,
        "category":         a.category,
        "is_fake":          a.is_fake,
        "credibility_score": score,
        "topic_cluster":    a.topic_cluster,
        "score_details": {
            "base_ml":            score,
            "headline_score":     score,
            "content_score":      score,
            "is_trusted":         False,
            "corroboration_count": 0,
            "bonus_applied":      0,
            "verification_boost": 0,
            "penalty_applied":    0,
            "fact_check":         None,
            "ai_logic": {
                "trust_keywords":      [],
                "risk_keywords":       [],
                "sensationalism_score": 0,
                "objectivity_score":   1,
            },
        },
        "keywords": a.keywords,
        "summary":  (a.clean_content[:200] + "...") if a.clean_content
                    else ((a.raw_content[:200] + "...") if a.raw_content else ""),
        "full_content": a.clean_content or a.raw_content or "Content not available for this article.",
    }

@app.get("/api/articles")
def get_articles(
    page:     int  = Query(1,  ge=1),
    limit:    int  = Query(20, ge=1, le=100),
    category: str  = None,
    is_fake:  bool = None,
    search:   str  = None,
):
    # Check cache first
    params = {"p": page, "l": limit, "c": category, "f": is_fake, "s": search}
    ck = _cache_key(params)
    cached = _cache_get(ck)
    if cached:
        return cached

    with get_session() as session:
        if session is None:
            return {"items": [], "total": 0, "page": page, "pages": 0}

        # Use FTS5 for search queries (orders of magnitude faster than LIKE)
        if search and search.strip():
            fts_term = search.strip().replace('"', '""')
            # FTS5 match — search title, keywords, and source
            fts_sql = text("""
                SELECT rowid FROM articles_fts
                WHERE articles_fts MATCH :term
                ORDER BY rank
            """)
            try:
                fts_rows = session.execute(fts_sql, {"term": f'"{fts_term}"'}).fetchall()
                matched_ids = [r[0] for r in fts_rows]
            except Exception:
                # Fallback: if FTS fails (e.g. special chars), use LIKE
                matched_ids = None

            if matched_ids is not None:
                if not matched_ids:
                    result = {"items": [], "total": 0, "page": page, "pages": 0}
                    _cache_set(ck, result)
                    return result
                q = session.query(Article).filter(Article.id.in_(matched_ids))
            else:
                # Fallback to LIKE
                q = session.query(Article).filter(Article.title.ilike(f"%{search}%"))
        else:
            q = session.query(Article)

        if category:
            q = q.filter(Article.category == category)
        if is_fake is not None:
            q = q.filter(Article.is_fake == is_fake)

        total    = q.count()
        articles = (
            q.order_by(Article.published_at.desc())
             .offset((page - 1) * limit)
             .limit(limit)
             .all()
        )

        items = [_serialize_article(a) for a in articles]

        result = {
            "items":  items,
            "total":  total,
            "page":   page,
            "pages":  (total + limit - 1) // limit,
        }
        _cache_set(ck, result)
        return result

# ── FTS Sync endpoint (called by scheduler after ingestion) ───────────────────
@app.post("/api/refresh-fts")
def api_refresh_fts():
    refresh_fts()
    # Also bust the cache since new articles arrived
    _cache.clear()
    return {"status": "ok"}

# ── Intelligence Pipeline Trigger (Optional Internal) ─────────────────────────
