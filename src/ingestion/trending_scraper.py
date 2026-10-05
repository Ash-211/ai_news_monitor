"""
Trending Social Media Scraper for Fake News Detection

Scrapes trending/popular content from platforms where fake news spreads most:
  1. Reddit (via public JSON endpoint — no API key needed)
  2. Google News / GNews (via gnews package or RSS)
  3. Twitter/X (via Nitter RSS bridges or public RSS mirrors)
  4. Facebook (via public page RSS feeds)

Each scraped item is run through the fake news detection pipeline
(DistilBERT + linguistic analysis + fact-checking).

Research shows fake news spreads most on:
  Facebook > X/Twitter > TikTok > Reddit > WhatsApp > Instagram
  (Source: 2025 Reuters Institute Digital News Report, WEF Global Risks 2025)
"""

import os
import json
import time
import hashlib
import requests
import feedparser
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# ── Platform Definitions ─────────────────────────────────────────────────────

PLATFORMS = {
    "reddit": {
        "name": "Reddit",
        "icon": "🟠",
        "color": "#FF4500",
        "risk_level": "high",
        "description": "Popular posts from trending subreddits"
    },
    "gnews": {
        "name": "Google News",
        "icon": "📰",
        "color": "#4285F4",
        "risk_level": "medium",
        "description": "Trending headlines via Google News"
    },
    "twitter": {
        "name": "X / Twitter",
        "icon": "𝕏",
        "color": "#000000",
        "risk_level": "critical",
        "description": "Trending topics via public RSS mirrors"
    },
    "facebook": {
        "name": "Facebook",
        "icon": "📘",
        "color": "#1877F2",
        "risk_level": "critical",
        "description": "Public page posts via RSS feeds"
    },
}

# Subreddits known strictly for formal news reporting, focusing on India
REDDIT_SUBREDDITS = [
    "india", "indianews", "unitedstatesofindia", "worldnews", "geopolitics"
]

# Facebook public pages that are often sources/subjects of misinformation
FACEBOOK_RSS_FEEDS = [
    # These use RSS bridge services for public Facebook pages
    "https://rss.app/feeds/_placeholder_fb_news.xml",  # placeholder — user can add real RSS bridge URLs
]

# Twitter/X RSS bridges (Nitter instances or RSS bridge services)
TWITTER_RSS_SOURCES = [
    # Public RSS feeds that mirror trending tweets
    "https://rss.app/feeds/_placeholder_twitter_trending.xml",  # placeholder
]

# ── Scraper Functions ─────────────────────────────────────────────────────────

def _hash_url(url: str) -> str:
    """Generate a short hash for deduplication."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def scrape_reddit(subreddits: List[str] = None, limit: int = 10) -> List[Dict]:
    """
    Scrapes trending posts from Reddit without ANY accounts or API keys.
    Uses the free public api.rss2json.com to proxy Reddit's RSS feeds,
    bypassing Reddit's strict datacenter IP blocks (403 Forbidden).
    """
    subreddits = subreddits or REDDIT_SUBREDDITS
    results = []
    
    for sub in subreddits:
        try:
            # Construct Reddit RSS URL
            rss_url = f"https://www.reddit.com/r/{sub}/hot/.rss"
            # Use rss2json proxy to bypass Reddit's 403 blocks
            api_url = f"https://api.rss2json.com/v1/api.json?rss_url={rss_url}&api_key="
            
            resp = requests.get(api_url, timeout=15)
            
            if resp.status_code != 200:
                print(f"  [Reddit] rss2json proxy failed for r/{sub} (Status: {resp.status_code})")
                continue
                
            data = resp.json()
            items = data.get("items", [])
            
            for item in items[:limit]:
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                url = item.get("link", "")
                author = item.get("author", "").replace("/u/", "")
                published = item.get("pubDate", "")
                
                if not title or not url:
                    continue
                    
                results.append({
                    "platform": "reddit",
                    "title": title,
                    "content": content,
                    "url": url,
                    "external_url": None,
                    "source": f"r/{sub}",
                    "author": author,
                    "engagement": {}, # RSS doesn't give us upvotes
                    "published_at": published if published else datetime.now(timezone.utc).isoformat(),
                    "image_url": item.get("thumbnail") if item.get("thumbnail") else None,
                    "hash": _hash_url(url),
                })
                
            # Be respectful to the free proxy
            time.sleep(1)
            
        except Exception as e:
            print(f"  [Reddit] Error scraping r/{sub} via proxy: {e}")
            
    print(f"  [Reddit] Scraped {len(results)} trending posts from {len(subreddits)} subreddits via RSS Proxy")
    return results


def scrape_gnews(max_results: int = 20) -> List[Dict]:
    """
    Scrapes trending news headlines via the gnews Python package 
    (wraps Google News RSS). Falls back to direct Google News RSS.
    No API key required.
    """
    results = []
    
    # Method 1: Try the gnews package
    try:
        from gnews import GNews
        
        gn = GNews(language='en', country='IN', period='1d', max_results=max_results)
        articles = gn.get_top_news()
        
        for article in articles:
            title = article.get("title", "").strip()
            url = article.get("url", "")
            source = article.get("publisher", {})
            if isinstance(source, dict):
                source_name = source.get("title", "Unknown")
            else:
                source_name = str(source)
            
            # Try to get full article text
            content = article.get("description", title)
            
            results.append({
                "platform": "gnews",
                "title": title,
                "content": content,
                "url": url,
                "external_url": None,
                "source": source_name,
                "author": None,
                "engagement": {},
                "published_at": article.get("published date", datetime.now(timezone.utc).isoformat()),
                "image_url": None,
                "hash": _hash_url(url),
            })
        
        if results:
            print(f"  [GNews] Scraped {len(results)} trending headlines via gnews package")
            return results
            
    except ImportError:
        print("  [GNews] gnews package not installed, falling back to RSS")
    except Exception as e:
        print(f"  [GNews] gnews package error: {e}, falling back to RSS")
    
    # Method 2: Fallback to Google News RSS
    try:
        rss_url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:max_results]:
            title = entry.get("title", "").strip()
            url = entry.get("link", "")
            source = entry.get("source", {}).get("title", "Google News")
            
            results.append({
                "platform": "gnews",
                "title": title,
                "content": entry.get("summary", title),
                "url": url,
                "external_url": None,
                "source": source,
                "author": None,
                "engagement": {},
                "published_at": entry.get("published", datetime.now(timezone.utc).isoformat()),
                "image_url": None,
                "hash": _hash_url(url),
            })
        
        print(f"  [GNews] Scraped {len(results)} trending headlines via RSS")
        
    except Exception as e:
        print(f"  [GNews] RSS fallback failed: {e}")
    
    return results


def scrape_twitter_rss(feed_urls: List[str] = None) -> List[Dict]:
    """
    Scrapes trending tweets by querying Google News for recent Twitter indexing.
    This safely bypasses Twitter API limits without an account.
    """
    results = []
    try:
        import feedparser
        # Query GNews for site:twitter.com and breaking/news in India
        rss_url = "https://news.google.com/rss/search?q=site:twitter.com+india+news&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:15]:
            title = entry.get("title", "").strip()
            title = title.replace(" - twitter.com", "")
            url = entry.get("link", "")
            results.append({
                "platform": "twitter",
                "title": title[:280],
                "content": entry.get("summary", title),
                "url": url,
                "external_url": None,
                "source": "X / Twitter",
                "author": "",
                "engagement": {},
                "published_at": entry.get("published", datetime.now(timezone.utc).isoformat()),
                "image_url": None,
                "hash": _hash_url(url),
            })
    except Exception as e:
        print(f"  [Twitter] GNews proxy error: {e}")
    
    print(f"  [Twitter] Scraped {len(results)} trending posts via GNews proxy")
    return results


def scrape_facebook_rss(feed_urls: List[str] = None) -> List[Dict]:
    """
    Scrapes trending Facebook posts by querying Google News for recent FB indexing.
    This safely bypasses Meta blocks without an account.
    """
    results = []
    try:
        import feedparser
        # Query GNews for site:facebook.com and breaking/news in India
        rss_url = "https://news.google.com/rss/search?q=site:facebook.com+india+news&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:15]:
            title = entry.get("title", "").strip()
            title = title.replace(" - facebook.com", "")
            url = entry.get("link", "")
            results.append({
                "platform": "facebook",
                "title": title,
                "content": entry.get("summary", title),
                "url": url,
                "external_url": None,
                "source": "Facebook",
                "author": "",
                "engagement": {},
                "published_at": entry.get("published", datetime.now(timezone.utc).isoformat()),
                "image_url": None,
                "hash": _hash_url(url),
            })
    except Exception as e:
        print(f"  [Facebook] GNews proxy error: {e}")
    
    print(f"  [Facebook] Scraped {len(results)} posts via GNews proxy")
    return results


# ── Analysis Pipeline ─────────────────────────────────────────────────────────

def is_news_or_claim(title: str, content: str) -> bool:
    """
    STRICT FILTER: Identifies if a post is a formal news item or factual claim.
    Rejects personal opinions, questions, casual chat, and low-effort posts.
    """
    text = (title + " " + (content or "")).lower()
    
    # 1. Reject casual/personal posts immediately
    personal_pronouns = [" i ", " my ", " me ", " we ", " our ", " am i "]
    if any(p in f" {text} " for p in personal_pronouns):
        return False
        
    # 2. Reject questions (unless it's a formal headline question with news keywords)
    if "?" in title and not any(w in title.lower() for w in ["senate", "court", "president", "police", "government"]):
        return False
        
    # 3. Require at least one strong news/claim marker
    news_markers = [
        "says", "claims", "reports", "study", "research", "police", "government",
        "court", "judge", "senate", "congress", "president", "ceo", "company",
        "announces", "discovered", "investigation", "arrested", "billion", "million",
        "update", "breaking", "reveals", "states", "killed", "law", "minister", "market"
    ]
    has_marker = any(marker in text for marker in news_markers)
    
    # 4. Require named entities (Capitalized words indicating people, places, organizations)
    words = title.split()
    capitalized_words = [w for w in words[1:] if w.istitle() and len(w) > 1]
    has_entities = len(capitalized_words) >= 1
    
    # Strict rule: Must have BOTH a news marker and named entities
    if not (has_marker and has_entities):
        return False
        
    # Length check: Factual claims are usually descriptive
    if len(words) < 5:
        return False
        
    return True

def batch_llm_gatekeeper(items: List[Dict]) -> List[Dict]:
    """
    Uses Llama 3.2 via Hugging Face API to filter out memes and casual chat in a single batch call.
    Only allows posts that are presenting themselves as factual claims or news.
    """
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        return [item for item in items if is_news_or_claim(item.get("title", ""), item.get("content", ""))]
        
    try:
        from huggingface_hub import InferenceClient
        import json
        client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct", token=api_key)
        
        # Prepare batch input
        lines = []
        for i, item in enumerate(items):
            title = item.get("title", "").replace('\n', ' ')
            url = item.get("url", "")
            lines.append(f"[{i}] {title} | URL: {url}")
        
        batch_text = "\n".join(lines)
        
        prompt = f"""
You are a strict content moderator for a news monitoring platform.
Given a list of social media post titles, identify which ones present themselves as factual written news claims, written reports, or written mainstream articles.
You must REJECT:
- Memes, jokes, satire, casual chat
- Videos, Reels, Shorts, Clips, or live streams
- Image-only posts, posters, or infographics
- Any content that does not appear to be a text-based journalistic article

Only accept items that strongly appear to be written news.

Analyze these items:
{batch_text}

Return ONLY a JSON array of the integer indices of the items you ACCEPT. Do not include markdown formatting or explanation. Example: [0, 2, 5]
"""
        messages = [
            {"role": "system", "content": "You are a professional JSON filtering API. Output only the JSON array of accepted indices."},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat_completion(messages=messages, max_tokens=1024, temperature=0.1)
        text = response.choices[0].message.content.strip()
        
        # Parse the JSON array from response
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        accepted_indices = json.loads(text)
        accepted_indices = set(accepted_indices)
        
        filtered = [item for i, item in enumerate(items) if i in accepted_indices]
        return filtered
    except Exception as e:
        print(f"  [Gatekeeper] Batch Llama API error: {e}")
        return [item for item in items if is_news_or_claim(item.get("title", ""), item.get("content", ""))]
def cross_validate_claim(title: str) -> dict:
    """Uses Local DB and Google News search to fetch RAG evidence for claims."""
    evidence = []
    
    # 1. Search Local Database
    try:
        from src.ingestion.database import get_session, Article
        session = get_session()
        words = [w for w in title.lower().split() if len(w) > 4][:4]
        if words:
            query = session.query(Article)
            for word in words:
                query = query.filter(Article.title.ilike(f"%{word}%"))
            for match in query.limit(2).all():
                src_name = match.source or "Local DB"
                evidence.append(f"- {src_name}: {match.title}")
        session.close()
    except Exception as e:
        print(f"  [DB Cross-val] Error: {e}")

    # 2. Search Google News
    try:
        import urllib.parse
        import feedparser
        query = urllib.parse.quote_plus(title[:100])
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:3]:
            src_name = entry.get("source", {}).get("title", "News Outlet")
            headline = entry.get("title", "")
            evidence.append(f"- {src_name}: {headline}")
    except Exception:
        pass
        
    unique_evidence = list(set(evidence))
    if not unique_evidence:
        return {"verified": False, "evidence_string": "No major news outlets are reporting this."}
        
    return {"verified": True, "evidence_string": "\n".join(unique_evidence), "count": len(unique_evidence)}


def analyze_trending_item(item: Dict, model=None, tokenizer=None) -> Dict:
    """
    Runs a single scraped trending item through the fake news detection pipeline.
    Returns the item enriched with credibility analysis.
    """
    from src.intelligence.fake_news import detect_fake_news
    
    title = item.get("title", "")
    content = item.get("content", title)
    source = item.get("source", "")
    
    try:
        # Cross validate social media claims
        verification = None
        if item.get("platform") != "gnews":
            cv = cross_validate_claim(title)
            if cv["verified"]:
                verification = f"Cross-validated by {cv['count']} trusted sources including {', '.join(cv['sources'][:2])}."
            else:
                verification = "Claim could not be cross-validated by any major news outlet."
                
        # Strip emojis and weird unicode that crash the WordPiece tokenizer
        title_clean = title.encode('ascii', 'ignore').decode()
        content_clean = content.encode('ascii', 'ignore').decode()
        
        is_fake, credibility_score, breakdown = detect_fake_news(
            title=title_clean,
            content=content_clean,
            model=model,
            tokenizer=tokenizer,
            source=source,
            verification_result=verification,
        )
        
        item["analysis"] = {
            "is_fake": is_fake,
            "credibility_score": round(credibility_score, 4),
            "explanation": breakdown.get("explanation_text", ""),
            "verdict": "Potentially Misleading" if is_fake else "Likely Authentic",
        }
    except Exception as e:
        print(f"  [Analysis] Error analyzing '{title[:50]}...': {e}")
        item["analysis"] = {
            "is_fake": None,
            "credibility_score": None,
            "explanation": f"Analysis failed: {str(e)}",
            "verdict": "Error",
        }
    
    return item


def scan_all_platforms(
    platforms: List[str] = None,
    reddit_limit: int = 8,
    gnews_limit: int = 15,
) -> Dict:
    """
    Master function: scrapes all configured platforms and analyzes each item.
    
    Args:
        platforms: List of platform keys to scrape. None = all.
        reddit_limit: Posts per subreddit.
        gnews_limit: Max headlines from Google News.
    
    Returns:
        Dict with keys:
          - items: list of analyzed trending items
          - stats: per-platform counts and summary
          - scanned_at: ISO timestamp
    """
    print(f"\n{'='*60}")
    print(f"  TRENDING SOCIAL MEDIA SCANNER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    all_platforms = platforms or ["reddit", "gnews", "twitter", "facebook"]
    all_items = []
    platform_stats = {}
    
    # ── Scrape each platform ──────────────────────────────────────────
    if "reddit" in all_platforms:
        reddit_items = scrape_reddit(limit=reddit_limit)
        all_items.extend(reddit_items)
        platform_stats["reddit"] = len(reddit_items)
    
    if "gnews" in all_platforms:
        gnews_items = scrape_gnews(max_results=gnews_limit)
        all_items.extend(gnews_items)
        platform_stats["gnews"] = len(gnews_items)
    
    if "twitter" in all_platforms:
        twitter_items = scrape_twitter_rss()
        all_items.extend(twitter_items)
        platform_stats["twitter"] = len(twitter_items)
    
    if "facebook" in all_platforms:
        fb_items = scrape_facebook_rss()
        all_items.extend(fb_items)
        platform_stats["facebook"] = len(fb_items)
    
    # ── Deduplicate by hash ───────────────────────────────────────────
    seen = set()
    unique_items = []
    for item in all_items:
        h = item.get("hash", "")
        if h not in seen:
            seen.add(h)
            unique_items.append(item)
    
    deduped_count = len(all_items) - len(unique_items)
    if deduped_count > 0:
        print(f"\n  Removed {deduped_count} duplicate items")
        
    # ── Pre-filter: Keep only items that look like news or claims ─────
    print("\n  Running LLM Gatekeeper to filter out memes and noise...")
    gnews_items = [item for item in unique_items if item["platform"] == "gnews"]
    social_items = [item for item in unique_items if item["platform"] != "gnews"]
    
    if social_items:
        filtered_social = batch_llm_gatekeeper(social_items)
    else:
        filtered_social = []
        
    news_items = gnews_items + filtered_social
    skipped_count = len(unique_items) - len(news_items)
            
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} items (identified as memes/noise by Gatekeeper)")
    
    print(f"\n  Total news items to analyze: {len(news_items)}")
    print(f"  Running fake news detection pipeline...\n")
    
    # ── Load model once, analyze all items ────────────────────────────
    model, tokenizer = None, None
    try:
        from src.intelligence.fake_news import load_fake_news_detector
        model, tokenizer = load_fake_news_detector()
    except Exception as e:
        print(f"  Warning: Could not load local model: {e}")
        print(f"  Will use HF Worker API for detection.\n")
    
    # ── Fetch RAG Evidence & Batch Process ────────────────────────────
    print("  Fetching cross-validation evidence for all claims...")
    for i, item in enumerate(news_items):
        safe_title = item['title'][:60].encode('ascii', 'ignore').decode()
        print(f"  [{i+1}/{len(news_items)}] Fetching Evidence | {safe_title}...")
        
        if item.get("platform") != "gnews":
            cv = cross_validate_claim(item.get("title", ""))
            item["verification"] = cv.get("evidence_string", "No major news outlets are reporting this.")
        else:
            item["verification"] = "Verified GNews source."

    print(f"\n  Sending Batch RAG Ensemble request to Llama for {len(news_items)} items...")
    from src.intelligence.fake_news import detect_batch
    analyzed_items = detect_batch(news_items, model=model, tokenizer=tokenizer)
    
    fake_count = sum(1 for x in analyzed_items if x.get("analysis", {}).get("is_fake"))
    
    # ── Sort: flagged items first, then by credibility (ascending) ────
    analyzed_items.sort(
        key=lambda x: (
            not (x.get("analysis", {}).get("is_fake") is True),
            x.get("analysis", {}).get("credibility_score") if x.get("analysis", {}).get("credibility_score") is not None else 0.5,
        )
    )
    
    # ── Summary ───────────────────────────────────────────────────────
    total = len(analyzed_items)
    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE")
    print(f"  Total scanned:   {total}")
    print(f"  Flagged as fake: {fake_count}")
    print(f"  Authentic:       {total - fake_count}")
    print(f"{'='*60}\n")
    
    return {
        "items": analyzed_items,
        "stats": {
            "total_scanned": total,
            "flagged_fake": fake_count,
            "authentic": total - fake_count,
            "platforms": platform_stats,
            "duplicates_removed": deduped_count,
            "non_news_skipped": skipped_count,
        },
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = scan_all_platforms()
    
    print("\n--- Top Flagged Items ---")
    for item in result["items"][:10]:
        analysis = item.get("analysis", {})
        status = "🚩 FAKE" if analysis.get("is_fake") else "✅ REAL"
        score = analysis.get("credibility_score", "?")
        if isinstance(score, float):
            score = f"{score:.0%}"
        print(f"  {status} [{score}] [{item['platform'].upper()}] {item['title'][:70]}")
