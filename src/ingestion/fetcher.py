import os
import feedparser
from datetime import datetime
from dateutil import parser
from dotenv import load_dotenv
from newsapi import NewsApiClient
from sqlalchemy.exc import IntegrityError
from newspaper import Article as NewsArticle
from src.ingestion.database import get_session, Article

# Load environment variables (e.g., NEWSAPI_KEY)
load_dotenv()

class NewsAPIFetcher:
    def __init__(self):
        api_key = os.getenv('NEWSAPI_KEY')
        if not api_key:
            print("WARNING: NEWSAPI_KEY not found in .env file.")
        self.newsapi = NewsApiClient(api_key=api_key) if api_key else None

    def fetch_top_headlines(self, category='technology', language='en', page_size=20):
        if not self.newsapi:
            return []
            
        try:
            response = self.newsapi.get_top_headlines(
                category=category,
                language=language,
                page_size=page_size
            )
            
            if response['status'] == 'ok':
                return response['articles']
            return []
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return []

class RSSFetcher:
    def __init__(self, feed_urls):
        self.feed_urls = feed_urls

    def fetch_feeds(self):
        articles = []
        for url in self.feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    articles.append({
                        'title': entry.get('title', ''),
                        'url': entry.get('link', ''),
                        'source': feed.feed.get('title', 'RSS Feed'),
                        'author': entry.get('author', ''),
                        'publishedAt': entry.get('published', datetime.utcnow().isoformat()),
                        'content': entry.get('summary', '') or entry.get('description', '')
                    })
            except Exception as e:
                print(f"Error fetching RSS feed '{url}': {e}")
        return articles

def save_articles_to_db(raw_articles, source_type="NewsAPI"):
    """
    Takes raw articles from fetchers and saves them to the database.
    Prevents duplicates using URL uniqueness. Each article is committed
    individually so new data appears in the DB immediately.
    """
    session = get_session()
    saved_count = 0

    # Pre-import Config from newspaper for timeouts
    from newspaper import Config
    config = Config()
    config.request_timeout = 3 # 3 seconds timeout limit

    for item in raw_articles:
        # Standardize article payload
        title = item.get('title')
        url = item.get('url')
        
        # Skip if missing required fields
        if not title or not url:
            continue
            
        # Fast pre-check: skip if URL already exists in database
        if session.query(Article.id).filter(Article.url == url).first():
            continue

        raw_content = item.get('content') or item.get('description') or ''
        
        # Parse full text from live URL using newspaper3k
        if url:
            try:
                print(f"  Scraping new article: {title[:60]}...", flush=True)
                web_article = NewsArticle(url, config=config, keep_article_html=False)
                web_article.download()
                if web_article.download_state == 2:  # SUCCESS
                    web_article.parse()
                    if web_article.text and len(web_article.text.strip()) > len(raw_content):
                        raw_content = web_article.text
            except Exception:
                # Fallback to RSS snippet if scraping fails
                pass

        source = item.get('source', {}).get('name') if isinstance(item.get('source'), dict) else item.get('source', source_type)
        author = item.get('author')
        
        # Parse publication date
        pub_date_str = item.get('publishedAt')
        if pub_date_str:
            try:
                pub_date = parser.parse(pub_date_str).replace(tzinfo=None) # remove tz for sqlite
            except Exception:
                pub_date = datetime.utcnow()
        else:
            pub_date = datetime.utcnow()

        # Save each article immediately so it appears in the DB right away
        article = Article(
            title=title,
            url=url,
            source=source,
            author=author,
            published_at=pub_date,
            raw_content=raw_content
        )
        session.add(article)
        try:
            session.commit()
            saved_count += 1
        except IntegrityError:
            session.rollback()  # skip duplicate
        except Exception as e:
            print(f"Error saving article to DB: {e}")
            session.rollback()

    session.close()
    return saved_count

def run_ingestion():
    """
    Main entry point for running all fetchers and saving to the DB.
    """
    print(f"[{datetime.now()}] Starting data ingestion cycle...")
    
    # 1. Fetch from NewsAPI
    newsapi_fetcher = NewsAPIFetcher()
    # You can loop through multiple categories: 'business', 'technology', 'health', 'science', 'sports'
    api_articles = newsapi_fetcher.fetch_top_headlines(category='technology')
    api_saved = save_articles_to_db(api_articles, source_type="NewsAPI")
    print(f"Saved {api_saved} new articles from NewsAPI.")

    # 2. Fetch from RSS Feeds
    rss_urls = [
        # International
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://techcrunch.com/feed/",
        "http://feeds.reuters.com/reuters/topNews",
        # Indian
        "https://feeds.feedburner.com/ndtvnews-top-stories",
        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "https://indianexpress.com/feed/",
        "https://www.thehindu.com/news/national/feeder/default.rss"
    ]
    rss_fetcher = RSSFetcher(rss_urls)
    rss_articles = rss_fetcher.fetch_feeds()
    rss_saved = save_articles_to_db(rss_articles, source_type="RSS")
    print(f"Saved {rss_saved} new articles from RSS feeds.")

    print(f"[{datetime.now()}] Data ingestion cycle complete.")

if __name__ == "__main__":
    run_ingestion()
