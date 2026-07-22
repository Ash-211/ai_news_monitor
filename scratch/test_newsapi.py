import os
import sys
from dotenv import load_dotenv
from newsapi import NewsApiClient

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

def test_api():
    api_key = os.getenv('NEWSAPI_KEY')
    print(f"NEWSAPI_KEY: {api_key[:10]}..." if api_key else "NEWSAPI_KEY is not set!")
    if not api_key:
        return
        
    try:
        newsapi = NewsApiClient(api_key=api_key)
        response = newsapi.get_top_headlines(category='technology', language='en', page_size=20)
        
        if response.get('status') == 'ok':
            articles = response.get('articles', [])
            print(f"Total articles returned: {len(articles)}")
            for idx, art in enumerate(articles, 1):
                print(f"{idx}. {art.get('title')} | Pub: {art.get('publishedAt')} | Source: {art.get('source', {}).get('name')}")
        else:
            print(f"NewsAPI returned error status: {response}")
    except Exception as e:
        print(f"ERROR: NewsAPI call failed: {e}")

if __name__ == '__main__':
    test_api()
