"""
External Fact-Check Verification Module
Uses NewsAPI for cross-referencing and Google Fact Check API for claim validation.
Provides a verification_score (0.0 - 1.0) that feeds into the credibility pipeline.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── API Configuration ─────────────────────────────────────────────
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_KEY")

NEWSAPI_SEARCH_URL = "https://newsapi.org/v2/everything"
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def _extract_search_keywords(title: str, max_words: int = 6) -> str:
    """
    Extracts the most meaningful keywords from a title for API search.
    Strips common filler words to improve search precision.
    """
    if not title:
        return ""
    
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'and', 'or', 'but', 'not', 'has',
        'have', 'had', 'be', 'been', 'being', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'as', 'also', 'than', 'after', 'before',
        'says', 'said', 'over', 'up', 'out', 'into', 'about', 'how', 'what',
        'when', 'where', 'who', 'which', 'why', 'new', 'all', 'more', 'most'
    }
    
    words = title.split()
    # Keep words that are meaningful (not stop words, not too short)
    keywords = [w.strip('.,!?:;"\'-()[]') for w in words 
                if w.lower().strip('.,!?:;"\'-()[]') not in stop_words and len(w) > 2]
    
    return " ".join(keywords[:max_words])


def cross_reference_news(title: str, api_key: str = None) -> dict:
    """
    Searches NewsAPI to see how many other outlets reported a similar story.
    
    Returns:
        {
            "score": float (0.0 - 1.0),
            "total_results": int,
            "matching_sources": list[str],
            "status": "ok" | "error" | "skipped"
        }
    """
    key = api_key or NEWSAPI_KEY
    if not key:
        return {"score": 0.5, "total_results": 0, "matching_sources": [], "status": "skipped"}
    
    query = _extract_search_keywords(title)
    if not query or len(query) < 5:
        return {"score": 0.5, "total_results": 0, "matching_sources": [], "status": "skipped"}
    
    try:
        response = requests.get(NEWSAPI_SEARCH_URL, params={
            "q": query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 10,
            "apiKey": key
        }, timeout=5)
        
        if response.status_code != 200:
            print(f"  [FactCheck] NewsAPI error: {response.status_code}")
            return {"score": 0.5, "total_results": 0, "matching_sources": [], "status": "error"}
        
        data = response.json()
        total = data.get("totalResults", 0)
        articles = data.get("articles", [])
        
        # Extract unique source names
        sources = list(set(
            a.get("source", {}).get("name", "Unknown") 
            for a in articles if a.get("source")
        ))
        
        # Scoring logic:
        # 0 results   → 0.2 (suspicious, nobody else reports this)
        # 1-2 results → 0.4 (weak corroboration)
        # 3-5 results → 0.6 (moderate corroboration)
        # 5-10 results→ 0.8 (strong corroboration)
        # 10+ results → 1.0 (widely reported)
        if total == 0:
            score = 0.2
        elif total <= 2:
            score = 0.4
        elif total <= 5:
            score = 0.6
        elif total <= 10:
            score = 0.8
        else:
            score = 1.0
        
        return {
            "score": score,
            "total_results": total,
            "matching_sources": sources[:5],  # Cap at 5 for payload size
            "status": "ok"
        }
    
    except requests.exceptions.Timeout:
        print("  [FactCheck] NewsAPI timeout")
        return {"score": 0.5, "total_results": 0, "matching_sources": [], "status": "error"}
    except Exception as e:
        print(f"  [FactCheck] NewsAPI exception: {e}")
        return {"score": 0.5, "total_results": 0, "matching_sources": [], "status": "error"}


def check_fact_claim(title: str, api_key: str = None) -> dict:
    """
    Queries Google Fact Check Tools API to see if professional fact-checkers
    have reviewed a claim matching this headline.
    
    Returns:
        {
            "score": float (0.0 - 1.0),
            "claims_found": int,
            "ratings": list[str],       # e.g. ["False", "Mostly True"]
            "fact_check_urls": list[str],
            "status": "ok" | "error" | "skipped"
        }
    """
    key = api_key or GOOGLE_FACTCHECK_KEY
    if not key:
        return {"score": 0.5, "claims_found": 0, "ratings": [], "fact_check_urls": [], "status": "skipped"}
    
    query = _extract_search_keywords(title, max_words=8)
    if not query or len(query) < 5:
        return {"score": 0.5, "claims_found": 0, "ratings": [], "fact_check_urls": [], "status": "skipped"}
    
    try:
        response = requests.get(GOOGLE_FACTCHECK_URL, params={
            "query": query,
            "key": key,
            "languageCode": "en"
        }, timeout=5)
        
        if response.status_code != 200:
            print(f"  [FactCheck] Google API error: {response.status_code}")
            return {"score": 0.5, "claims_found": 0, "ratings": [], "fact_check_urls": [], "status": "error"}
        
        data = response.json()
        claims = data.get("claims", [])
        
        if not claims:
            # No fact-check found — neutral score (we don't penalize for absence)
            return {"score": 0.5, "claims_found": 0, "ratings": [], "fact_check_urls": [], "status": "ok"}
        
        # Parse ratings from fact-checkers
        ratings = []
        urls = []
        for claim in claims[:5]:
            for review in claim.get("claimReview", []):
                rating_text = review.get("textualRating", "").lower()
                ratings.append(review.get("textualRating", "Unknown"))
                urls.append(review.get("url", ""))
        
        # Convert textual ratings to a numeric score
        # Positive ratings boost, negative ratings penalize
        positive_terms = ['true', 'correct', 'accurate', 'mostly true', 'partly true']
        negative_terms = ['false', 'fake', 'misleading', 'incorrect', 'pants on fire', 
                         'mostly false', 'partly false', 'fabricated', 'hoax', 'scam']
        
        positive_count = sum(1 for r in ratings if any(p in r.lower() for p in positive_terms))
        negative_count = sum(1 for r in ratings if any(n in r.lower() for n in negative_terms))
        total_rated = positive_count + negative_count
        
        if total_rated == 0:
            score = 0.5  # Neutral if ratings are ambiguous
        elif negative_count > positive_count:
            # More fact-checkers say it's false
            score = max(0.1, 0.5 - (negative_count / total_rated) * 0.4)
        else:
            # More fact-checkers say it's true
            score = min(1.0, 0.5 + (positive_count / total_rated) * 0.4)
        
        return {
            "score": round(score, 4),
            "claims_found": len(claims),
            "ratings": ratings[:3],
            "fact_check_urls": urls[:3],
            "status": "ok"
        }
    
    except requests.exceptions.Timeout:
        print("  [FactCheck] Google API timeout")
        return {"score": 0.5, "claims_found": 0, "ratings": [], "fact_check_urls": [], "status": "error"}
    except Exception as e:
        print(f"  [FactCheck] Google API exception: {e}")
        return {"score": 0.5, "claims_found": 0, "ratings": [], "fact_check_urls": [], "status": "error"}


def verify_article(title: str, newsapi_key: str = None, google_key: str = None) -> dict:
    """
    Orchestrator: Runs both NewsAPI cross-referencing and Google Fact Check,
    combines the results into a single verification_score.
    
    Returns:
        {
            "verification_score": float (0.0 - 1.0),
            "cross_reference": { ... },   # NewsAPI results
            "fact_check": { ... },         # Google results
        }
    """
    cross_ref = cross_reference_news(title, api_key=newsapi_key)
    fact_check = check_fact_claim(title, api_key=google_key)
    
    # Combine scores
    # If both APIs returned data, weight equally
    # If only one returned data, use that one entirely
    cross_ok = cross_ref["status"] == "ok"
    fact_ok = fact_check["status"] == "ok" and fact_check["claims_found"] > 0
    
    if cross_ok and fact_ok:
        verification_score = (0.5 * cross_ref["score"]) + (0.5 * fact_check["score"])
    elif cross_ok:
        verification_score = cross_ref["score"]
    elif fact_ok:
        verification_score = fact_check["score"]
    else:
        verification_score = 0.5  # Neutral fallback
    
    return {
        "verification_score": round(verification_score, 4),
        "cross_reference": cross_ref,
        "fact_check": fact_check
    }


# ─── Quick Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FACT CHECKER — Testing")
    print("=" * 60)
    
    test_headlines = [
        "Three held for running illegal e-cigarette racket, vapes worth 34 lakh seized",
        "UNESCO declares Indian National Anthem the best in the world",
        "India and Bangladesh sign new water-sharing agreement for Teesta river",
    ]
    
    for title in test_headlines:
        print(f"\n--- {title[:60]}... ---")
        result = verify_article(title)
        print(f"  Verification Score: {result['verification_score']}")
        print(f"  NewsAPI: {result['cross_reference']['total_results']} results, score={result['cross_reference']['score']}")
        print(f"  Google:  {result['fact_check']['claims_found']} claims, score={result['fact_check']['score']}")
