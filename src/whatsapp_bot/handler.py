"""
handler.py — WhatsApp message handler for the AI News Bot.

Parses incoming user messages, queries the database using the same
patterns as the Discord bot's news_commands cog, and formats results
for WhatsApp's plain-text + light-markup format.

Supported commands:
    "news"  / "latest"           → Latest 5 articles
    "search <topic>"             → Search articles by keyword
    "categories"                 → List available categories
    "category <name>"            → Articles in a specific category
    "verify <url>"               → Run credibility check on a URL
    "help"                       → Show available commands
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional

from sqlalchemy import func

from src.ingestion.database import Article, get_session
from src.whatsapp_bot.sender import send_whatsapp_message, send_interactive_buttons, send_interactive_list

logger = logging.getLogger("whatsapp_bot.handler")

MAX_ARTICLES = 5
MAX_SUMMARY_LEN = 150

# In-memory session tracking for pagination
# Keyed by phone number, stores last command context
user_sessions = {}  # { "919876543210": {"command": "news", "page": 1, "query": None, "category": None, "target_date": None} }


# ════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ════════════════════════════════════════════════════════════════════════════

async def handle_incoming_message(from_number: str, message_text: str) -> None:
    """
    Process an incoming WhatsApp message and send an appropriate response.

    Args:
        from_number:  Sender's phone number (international format).
        message_text: The raw text of the message.
    """
    text = message_text.strip().lower()
    logger.info("📩 Message from %s: %s", from_number, text[:100])

    try:
        response = None
        if text in ("hi", "hello", "hey", "start"):
            response = _welcome_message()

        elif text in ("help", "menu", "commands", "?", "show_help"):
            response = _help_message()

        elif text in ("news", "latest", "today", "daily"):
            response, target_date = _get_latest_news(page=1)
            user_sessions[from_number] = {"command": "news", "page": 1, "target_date": target_date}

        elif text in ("categories", "show_categories"):
            response = await _get_categories(from_number)

        elif text.startswith("category "):
            category_name = message_text.strip()[9:].strip()
            response = _get_news_by_category(category_name, page=1)
            user_sessions[from_number] = {"command": "category", "page": 1, "category": category_name}
            
        elif text.startswith("cat_"):
            category_name = text[4:].replace("_", " ")
            response = _get_news_by_category(category_name, page=1)
            user_sessions[from_number] = {"command": "category", "page": 1, "category": category_name}

        elif text.startswith("search "):
            query = message_text.strip()[7:].strip()
            response = _search_news(query, page=1)
            user_sessions[from_number] = {"command": "search", "page": 1, "query": query}
            
        elif text in ("more", "more_news"):
            session = user_sessions.get(from_number)
            if session:
                session["page"] += 1
                if session["command"] == "news":
                    response, target_date = _get_latest_news(page=session["page"], target_date=session.get("target_date"))
                    session["target_date"] = target_date
                elif session["command"] == "search":
                    response = _search_news(session["query"], page=session["page"])
                elif session["command"] == "category":
                    response = _get_news_by_category(session["category"], page=session["page"])
                else:
                    response, target_date = _get_latest_news(page=1)
                    user_sessions[from_number] = {"command": "news", "page": 1, "target_date": target_date}
            else:
                response, target_date = _get_latest_news(page=1)
                user_sessions[from_number] = {"command": "news", "page": 1, "target_date": target_date}

        elif text.startswith("verify "):
            url = message_text.strip()[7:].strip()
            response = await _verify_url(url)

        else:
            response = _unknown_command()

        if response is not None:
            await send_whatsapp_message(from_number, response)
            
            # Send interactive buttons after text response
            buttons = [
                {"id": "more_news", "title": "📰 More News"},
                {"id": "show_categories", "title": "📂 Categories"},
                {"id": "show_help", "title": "❓ Help"}
            ]
            await send_interactive_buttons(from_number, "What would you like to do next?", buttons)

    except Exception as e:
        logger.exception("❌ Error handling message from %s: %s", from_number, e)
        await send_whatsapp_message(
            from_number,
            "⚠️ Something went wrong while processing your request. Please try again later."
        )


# ════════════════════════════════════════════════════════════════════════════
#  Command Handlers
# ════════════════════════════════════════════════════════════════════════════

def _welcome_message() -> str:
    return (
        "👋 *Welcome to the AI News Bot!*\n\n"
        "I can help you stay updated with the latest AI-curated news.\n\n"
        "Type *help* to see available commands."
    )


def _help_message() -> str:
    return (
        "📋 *Available Commands:*\n\n"
        "📰 *news* — Get the latest articles\n"
        "🔍 *search <topic>* — Search news by keyword\n"
        "   _Example: search artificial intelligence_\n\n"
        "📂 *categories* — List all categories\n"
        "📂 *category <name>* — News in a specific category\n"
        "   _Example: category Sci/Tech_\n\n"
        "🛡️ *verify <url>* — Check a news article's credibility\n"
        "   _Example: verify https://example.com/article_\n\n"
        "💡 _Tip: Just type *news* to get started!_"
    )


def _unknown_command() -> str:
    return (
        "🤔 I didn't understand that.\n\n"
        "Type *help* to see the list of commands I support."
    )


# ── Latest News ────────────────────────────────────────────────────────────

def _get_latest_news(page: int = 1, target_date: Optional[date] = None) -> tuple[str, Optional[date]]:
    """Fetch the latest articles (today or most recent date)."""
    session = get_session()
    try:
        offset = (page - 1) * MAX_ARTICLES
        
        if not target_date:
            target_date = date.today()
            articles = _fetch_articles(session, target_date=target_date, limit=MAX_ARTICLES, offset=offset)

            if not articles and page == 1:
                # Fall back to the most recent date with articles
                latest = (
                    session.query(Article)
                    .order_by(Article.published_at.desc())
                    .first()
                )
                if latest and latest.published_at:
                    target_date = latest.published_at.date()
                    articles = _fetch_articles(session, target_date=target_date, limit=MAX_ARTICLES, offset=offset)
        else:
            articles = _fetch_articles(session, target_date=target_date, limit=MAX_ARTICLES, offset=offset)

        if not articles:
            return "📭 No news articles found in the database. Check back later!", None
            
        total_count = _count_articles(session, target_date=target_date)
        total_pages = (total_count + MAX_ARTICLES - 1) // MAX_ARTICLES
        page_info = f" (page {page}/{total_pages})" if total_pages > 1 else ""

        fallback_note = ""
        if target_date != date.today():
            fallback_note = f"\n_No news today. Showing latest from {target_date.strftime('%B %d, %Y')}._\n"

        header = f"📰 *Latest News*{page_info} ({total_count} articles){fallback_note}\n\n"
        body = "\n".join(_format_article(i + 1, a) for i, a in enumerate(articles))

        return header + body, target_date

    finally:
        session.close()


# ── Search ─────────────────────────────────────────────────────────────────

def _search_news(query: str, page: int = 1) -> str:
    """Search articles by keyword in title, keywords, or category."""
    if not query:
        return "⚠️ Please provide a search term.\n_Example: search AI_"

    session = get_session()
    try:
        offset = (page - 1) * MAX_ARTICLES
        articles = _fetch_articles(session, tag=query, limit=MAX_ARTICLES, offset=offset)

        if not articles:
            return f"📭 No results found for *{query}*.\nTry a different keyword."
            
        total_count = _count_articles(session, tag=query)
        total_pages = (total_count + MAX_ARTICLES - 1) // MAX_ARTICLES
        page_info = f" (page {page}/{total_pages})" if total_pages > 1 else ""

        header = f"🔍 *Search results for \"{query}\"*{page_info} ({total_count} found)\n\n"
        body = "\n".join(_format_article(i + 1, a) for i, a in enumerate(articles))
        return header + body

    finally:
        session.close()


# ── Categories ─────────────────────────────────────────────────────────────

async def _get_categories(to: str) -> Optional[str]:
    """List all news categories with article counts."""
    session = get_session()
    try:
        rows = (
            session.query(Article.category, func.count(Article.id))
            .filter(Article.category.isnot(None))
            .group_by(Article.category)
            .order_by(func.count(Article.id).desc())
            .all()
        )

        if not rows:
            return "📭 No categories found."

        sections = [{
            "title": "News Categories",
            "rows": []
        }]
        
        for cat, count in rows:
            cat_id = f"cat_{cat.replace(' ', '_').lower()}"
            sections[0]["rows"].append({
                "id": cat_id[:200],
                "title": cat[:24],
                "description": f"{count} articles"
            })
            if len(sections[0]["rows"]) == 10:
                break

        await send_interactive_list(
            to=to,
            body="Here are the available news categories:",
            button_text="View Categories",
            sections=sections,
            header="📂 News Categories"
        )
        return None

    finally:
        session.close()


def _get_news_by_category(category_name: str, page: int = 1) -> str:
    """Fetch articles in a specific category."""
    if not category_name:
        return "⚠️ Please specify a category.\n_Example: category Sci/Tech_"

    session = get_session()
    try:
        offset = (page - 1) * MAX_ARTICLES
        articles = _fetch_articles(session, category=category_name, limit=MAX_ARTICLES, offset=offset)

        if not articles:
            return f"📭 No articles found in category *{category_name}*.\nType *categories* to see all options."

        total_count = _count_articles(session, category=category_name)
        total_pages = (total_count + MAX_ARTICLES - 1) // MAX_ARTICLES
        page_info = f" (page {page}/{total_pages})" if total_pages > 1 else ""

        header = f"📂 *{category_name}*{page_info} — {total_count} article(s)\n\n"
        body = "\n".join(_format_article(i + 1, a) for i, a in enumerate(articles))
        return header + body

    finally:
        session.close()


# ── Verify URL ─────────────────────────────────────────────────────────────

async def _verify_url(url: str) -> str:
    """Run the credibility verification pipeline on a URL."""
    if not url or not url.startswith("http"):
        return "⚠️ Please provide a valid URL.\n_Example: verify https://example.com/article_"

    try:
        import asyncio
        from src.intelligence.url_verifier import verify_url
        data = await asyncio.to_thread(verify_url, url)
    except Exception as e:
        logger.exception("Verify URL error: %s", e)
        return f"❌ Verification failed: {str(e)}"

    if data.get("error"):
        return f"❌ {data['error']}"

    # Format the result
    title = data.get("title", "Untitled")
    score = data.get("credibility_score")
    is_fake = data.get("is_fake")
    category = data.get("category", "N/A")
    explanation = data.get("explanation", "")

    # Credibility badge
    if score is not None:
        if score >= 0.6:
            badge = f"✅ Credible ({score:.0%})"
        elif score >= 0.4:
            badge = f"⚠️ Uncertain ({score:.0%})"
        else:
            badge = f"🚩 Flagged ({score:.0%})"
    else:
        badge = "N/A"

    # Verdict
    if is_fake is None:
        verdict = "⬜ Unknown"
    elif is_fake:
        verdict = "🚩 Potentially Misleading"
    else:
        verdict = "✅ Authentic"

    lines = [
        f"🔍 *Verification Result*\n",
        f"📰 *{title}*\n",
        f"🛡️ Credibility: {badge}",
        f"📋 Verdict: {verdict}",
        f"📂 Category: {category}",
    ]

    if explanation:
        if len(explanation) > 300:
            explanation = explanation[:300] + "…"
        lines.append(f"\n🤖 *AI Analysis:*\n_{explanation}_")

    # Fact-check data
    fact_check = data.get("fact_check")
    if fact_check and isinstance(fact_check, dict):
        v_score = fact_check.get("verification_score")
        if v_score is not None:
            lines.append(f"\n🔎 External Verification Score: {v_score:.0%}")

    lines.append(f"\n🔗 {url}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  Shared Helpers (mirror Discord bot's fetch_articles pattern)
# ════════════════════════════════════════════════════════════════════════════

def _fetch_articles(
    session,
    category: Optional[str] = None,
    target_date: Optional[date] = None,
    tag: Optional[str] = None,
    limit: int = 5,
    offset: int = 0,
) -> list:
    """
    Query the database for articles, matching the Discord bot's
    fetch_articles() pattern from news_commands.py.
    """
    query = session.query(Article)

    if target_date:
        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
        query = query.filter(Article.published_at >= start, Article.published_at < end)

    if category and category.lower() != "all":
        query = query.filter(func.lower(Article.category) == category.lower())

    if tag:
        tag_term = tag.strip()
        query = query.filter(
            (Article.title.ilike(f"%{tag_term}%"))
            | (Article.keywords.ilike(f"%{tag_term}%"))
            | (Article.category.ilike(f"%{tag_term}%"))
        )

    return query.order_by(Article.published_at.desc(), Article.id.desc()).offset(offset).limit(limit).all()


def _count_articles(
    session,
    category: Optional[str] = None,
    target_date: Optional[date] = None,
    tag: Optional[str] = None,
) -> int:
    query = session.query(func.count(Article.id))

    if target_date:
        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
        query = query.filter(Article.published_at >= start, Article.published_at < end)

    if category and category.lower() != "all":
        query = query.filter(func.lower(Article.category) == category.lower())

    if tag:
        tag_term = tag.strip()
        query = query.filter(
            (Article.title.ilike(f"%{tag_term}%"))
            | (Article.keywords.ilike(f"%{tag_term}%"))
            | (Article.category.ilike(f"%{tag_term}%"))
        )

    return query.scalar()


def _format_article(index: int, article: Article) -> str:
    """
    Format a single article for WhatsApp.
    Uses WhatsApp's supported formatting: *bold*, _italic_, ~strikethrough~.
    """
    # Get the best available summary
    summary = (
        article.summary_abstractive
        or article.summary_extractive
        or article.clean_content
        or article.raw_content
        or "No summary available."
    )
    if len(summary) > MAX_SUMMARY_LEN:
        summary = summary[:MAX_SUMMARY_LEN].rsplit(" ", 1)[0] + "…"

    # Credibility indicator
    score = article.credibility_score
    if score is not None:
        if score >= 0.6:
            cred = f"✅ {score:.0%}"
        elif score >= 0.4:
            cred = f"⚠️ {score:.0%}"
        else:
            cred = f"🚩 {score:.0%}"
    else:
        cred = "—"

    source = article.source or "Unknown"
    category = article.category or "Uncategorised"

    # Published date
    pub = ""
    if article.published_at:
        pub = article.published_at.strftime("%b %d, %Y")

    number_emojis = {1: '①', 2: '②', 3: '③', 4: '④', 5: '⑤'}
    emoji = number_emojis.get((index - 1) % 5 + 1, str(index))
    
    lines = [
        f"{emoji} *{article.title}*",
        f"_{summary}_",
        f"📂 {category}  |  🛡️ {cred}  |  📡 {source}",
    ]
    if pub:
        lines[2] += f"  |  📅 {pub}"

    lines.append(f"🔗 {article.url}")
    lines.append("━━━━━━")  # Horizontal line separator

    return "\n".join(lines)
