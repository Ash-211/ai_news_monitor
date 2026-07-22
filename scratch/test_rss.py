import feedparser
from datetime import datetime
from dateutil import parser

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

def test_rss():
    print(f"Current UTC time: {datetime.utcnow()}")
    for url in rss_urls:
        try:
            print(f"\nChecking feed: {url}")
            feed = feedparser.parse(url)
            print(f"Feed title: {feed.feed.get('title', 'N/A')}")
            entries = feed.entries[:3]  # just check top 3
            if not entries:
                print("No entries found.")
            for entry in entries:
                pub_date_str = entry.get('published', '')
                try:
                    pub_date = parser.parse(pub_date_str)
                except:
                    pub_date = "unknown"
                print(f"- {entry.get('title')} | Published: {pub_date_str} (Parsed: {pub_date})")
        except Exception as e:
            print(f"Error checking feed {url}: {e}")

if __name__ == '__main__':
    test_rss()
