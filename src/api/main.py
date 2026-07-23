from fastapi import FastAPI, Query, Request, Response, UploadFile, File
import asyncio
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from src.ingestion.database import Article, _is_postgres, Base
import os, time, hashlib, json, threading, logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Self-Ping Keep-Alive (replaces external uptime bot) ───────────────────────
# Pings this service's own Render URL every 4 minutes to prevent spin-down,
# but ONLY during 10 AM – 9 PM IST. Outside those hours, it does nothing
# and lets Render spin down naturally to save free-tier hours.

IST = timezone(timedelta(hours=5, minutes=30))
ACTIVE_START_HOUR = 10  # 10 AM IST
ACTIVE_END_HOUR = 21    # 9 PM IST
SELF_PING_INTERVAL = 240  # 4 minutes in seconds

_selfping_logger = logging.getLogger("selfping")

def _is_active_hours() -> bool:
    """Check if current IST time is within active window (10 AM - 9 PM)."""
    now_ist = datetime.now(IST)
    return ACTIVE_START_HOUR <= now_ist.hour < ACTIVE_END_HOUR

def _self_ping_loop():
    """
    Background thread: pings this service's own URL every 4 minutes
    during active hours to prevent Render from spinning it down.
    """
    import urllib.request
    service_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
    if not service_url:
        _selfping_logger.warning("RENDER_EXTERNAL_URL not set — self-ping disabled.")
        return

    ping_url = f"{service_url.rstrip('/')}/health"
    _selfping_logger.info(f"Self-ping thread started. Target: {ping_url}")

    while True:
        try:
            if _is_active_hours():
                urllib.request.urlopen(ping_url, timeout=15)
                now_ist = datetime.now(IST).strftime("%I:%M %p IST")
                _selfping_logger.info(f"✅ Self-ping OK at {now_ist}")
            else:
                now_ist = datetime.now(IST).strftime("%I:%M %p IST")
                _selfping_logger.info(f"😴 Outside active hours ({now_ist}). Letting Render sleep.")
        except Exception as e:
            _selfping_logger.warning(f"⚠️ Self-ping failed: {e}")
        time.sleep(SELF_PING_INTERVAL)

# ── Singleton engine & session factory (created ONCE at import time) ──────────
def _build_engine():
    db_url = os.getenv('DATABASE_URL', '').strip()

    if db_url and db_url.startswith('postgresql'):
        # ── Cloud PostgreSQL (Neon) ──
        engine = create_engine(db_url, echo=False, pool_pre_ping=True)
        # Ensure tables are created in the database first
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            # Create standard B-tree indexes (same purpose as before)
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at DESC)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_category      ON articles (category)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_articles_is_fake       ON articles (is_fake)"))

            # PostgreSQL Full-Text Search: add a tsvector column + GIN index
            # Step 1: Add the column if it doesn't exist
            conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'articles' AND column_name = 'search_vector'
                    ) THEN
                        ALTER TABLE articles ADD COLUMN search_vector tsvector;
                    END IF;
                END $$;
            """))
            # Step 2: Create GIN index on the tsvector column
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_articles_fts ON articles USING GIN (search_vector)
            """))
            # Step 3: Populate the search_vector for existing rows that are NULL
            conn.execute(text("""
                UPDATE articles
                SET search_vector = to_tsvector('english',
                    COALESCE(title, '') || ' ' || COALESCE(keywords, '') || ' ' || COALESCE(source, '')
                )
                WHERE search_vector IS NULL
            """))
            conn.commit()
        return engine, sessionmaker(bind=engine)

    # ── Fallback: Local SQLite ──
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    db_file = os.path.join(base_dir, 'data', 'database.sqlite')
    if not os.path.exists(db_file):
        return None, None
    engine = create_engine(
        f"sqlite:///{db_file}",
        echo=False,
        connect_args={"timeout": 15, "check_same_thread": False},
    )
    # Create indexes + FTS5 table once at startup (SQLite only)
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

# ── Keep search index in sync: call after ingestion inserts new articles ──────
def refresh_fts():
    """Sync the search index with any newly inserted articles."""
    if _engine is None:
        return
    with _engine.connect() as conn:
        if _is_postgres():
            # PostgreSQL: update tsvector for rows where it's NULL
            conn.execute(text("""
                UPDATE articles
                SET search_vector = to_tsvector('english',
                    COALESCE(title, '') || ' ' || COALESCE(keywords, '') || ' ' || COALESCE(source, '')
                )
                WHERE search_vector IS NULL
            """))
        else:
            # SQLite: sync FTS5 virtual table
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

# ── Health check endpoint (used by self-ping and GitHub Actions wake-up) ──────
@app.get("/health")
def health_check():
    now_ist = datetime.now(IST).strftime("%I:%M %p IST")
    return {"status": "alive", "time_ist": now_ist, "active": _is_active_hours()}

# ── Start self-ping thread on app startup ─────────────────────────────────────
@app.on_event("startup")
def start_self_ping():
    thread = threading.Thread(target=_self_ping_loop, daemon=True)
    thread.start()
    _selfping_logger.info("🚀 Self-ping background thread launched.")

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
    
    details = {}
    if hasattr(a, 'score_details') and a.score_details:
        try:
            details = json.loads(a.score_details)
        except Exception:
            pass
            
    # Provide a fallback if explanation_text is missing
    if "explanation_text" not in details:
        details["explanation_text"] = "No AI reasoning available for this article."

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
        "score_details":    details,
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

        # Use full-text search for search queries
        if search and search.strip():
            if _is_postgres():
                # PostgreSQL: use tsvector/tsquery
                fts_term = search.strip().replace("'", "''")
                # plainto_tsquery handles multi-word input safely
                fts_sql = text("""
                    SELECT id FROM articles
                    WHERE search_vector @@ plainto_tsquery('english', :term)
                """)
                try:
                    fts_rows = session.execute(fts_sql, {"term": fts_term}).fetchall()
                    matched_ids = [r[0] for r in fts_rows]
                except Exception:
                    matched_ids = None
            else:
                # SQLite: use FTS5 virtual table
                fts_term = search.strip().replace('"', '""')
                fts_sql = text("""
                    SELECT rowid FROM articles_fts
                    WHERE articles_fts MATCH :term
                    ORDER BY rank
                """)
                try:
                    fts_rows = session.execute(fts_sql, {"term": f'"{fts_term}"'}).fetchall()
                    matched_ids = [r[0] for r in fts_rows]
                except Exception:
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

# ── Verify URL (on-demand single-article pipeline) ────────────────────────────
class VerifyRequest(BaseModel):
    url: str

@app.post("/api/verify")
def api_verify_url(payload: VerifyRequest):
    """
    Accepts a news article URL, scrapes it, and runs the full intelligence
    pipeline (classify, fake-news detect, keyword extract, fact-check).
    Returns the analysis results without saving to the database.
    """
    url = payload.url.strip()
    if not url:
        return {"error": "No URL provided."}

    try:
        from src.intelligence.url_verifier import verify_url
        result = verify_url(url)
        return result
    except Exception as e:
        logging.exception("Verify URL endpoint error")
        return {"error": f"Verification failed: {str(e)}"}

# ── WhatsApp Webhook (Meta Cloud API) ─────────────────────────────────────────
# GET  /whatsapp-webhook  → Verification handshake (Meta confirms our server)
# POST /whatsapp-webhook  → Incoming messages from users

WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "ai-news-bot-verify-token")

@app.get("/whatsapp-webhook")
def whatsapp_verify(request: Request):
    """
    Meta sends a GET request with hub.mode, hub.verify_token, and hub.challenge
    when you register the webhook URL in the Meta Developer dashboard.
    We must echo back hub.challenge if the token matches.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        logging.info("✅ WhatsApp webhook verified successfully.")
        return Response(content=challenge, media_type="text/plain")
    else:
        logging.warning("⚠️ WhatsApp webhook verification failed. Token mismatch.")
        return Response(content="Forbidden", status_code=403)


@app.post("/whatsapp-webhook")
async def whatsapp_incoming(request: Request):
    """
    Receives incoming WhatsApp messages from Meta's Cloud API.
    Extracts the sender's phone number and message text, then
    delegates to the WhatsApp handler for processing.
    """
    body = await request.json()

    try:
        # Navigate Meta's nested webhook payload structure
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        for msg in messages:
            from_number = msg.get("from", "")
            message_text = ""
            
            if msg.get("type") == "text":
                message_text = msg.get("text", {}).get("body", "")
            elif msg.get("type") == "interactive":
                interactive = msg.get("interactive", {})
                if interactive.get("type") == "button_reply":
                    message_text = interactive.get("button_reply", {}).get("id", "")
                elif interactive.get("type") == "list_reply":
                    message_text = interactive.get("list_reply", {}).get("id", "")
            
            if from_number and message_text:
                # Process in background so we return 200 quickly
                # (Meta expects a fast response to avoid retries)
                from src.whatsapp_bot.handler import handle_incoming_message
                asyncio.create_task(handle_incoming_message(from_number, message_text))

    except (IndexError, KeyError, TypeError) as e:
        logging.warning("⚠️ Could not parse WhatsApp webhook payload: %s", e)

    # Always return 200 to acknowledge receipt (Meta retries on non-200)
    return {"status": "ok"}

@app.get("/api/whatsapp-info")
def get_whatsapp_info():
    bot_number = os.getenv("WHATSAPP_BOT_NUMBER", "")
    return {"bot_number": bot_number, "available": bool(bot_number)}

# ── Deepfake Detection ────────────────────────────────────────────────────────

# Allowed file types and size limits for deepfake analysis
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo", "video/webm"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024   # 50 MB

@app.post("/api/deepfake/analyze")
async def analyze_deepfake(file: UploadFile = File(...)):
    """
    Accepts an uploaded image or video file, runs it through the
    deepfake detection AI model, and returns a verdict with confidence
    scores and a human-readable explanation.

    Supported formats:
      Images: jpg, png, webp (max 10 MB)
      Videos: mp4, avi, mov, webm (max 50 MB)
    """
    import tempfile

    content_type = file.content_type or ""
    is_image = content_type in ALLOWED_IMAGE_TYPES
    is_video = content_type in ALLOWED_VIDEO_TYPES

    # ── Validate file type ────────────────────────────────────────────
    if not is_image and not is_video:
        return {
            "error": f"Unsupported file type: {content_type}. "
                     f"Please upload an image (jpg, png, webp) or video (mp4, avi, mov, webm)."
        }

    # ── Validate file size ────────────────────────────────────────────
    contents = await file.read()
    max_size = MAX_VIDEO_SIZE if is_video else MAX_IMAGE_SIZE
    if len(contents) > max_size:
        limit_mb = max_size // (1024 * 1024)
        return {"error": f"File too large. Maximum size is {limit_mb} MB."}

    # ── Save to temp file and analyze ─────────────────────────────────
    suffix = os.path.splitext(file.filename or "upload")[1] or (".png" if is_image else ".mp4")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(contents)
        tmp.close()

        if is_image:
            from src.intelligence.deepfake_detector import detect_deepfake_image
            result = detect_deepfake_image(tmp.name)
        else:
            from src.intelligence.deepfake_detector import detect_deepfake_video
            result = detect_deepfake_video(tmp.name)

        # Tag the result with the media type for the frontend
        result["media_type"] = "image" if is_image else "video"
        result["filename"] = file.filename
        return result

    except Exception as e:
        logging.error("Deepfake analysis failed: %s", e)
        return {"error": f"Analysis failed: {str(e)}"}

    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

# ── Intelligence Pipeline Trigger (Optional Internal) ─────────────────────────
