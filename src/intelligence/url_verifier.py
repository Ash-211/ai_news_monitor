"""
URL Verifier — On-Demand Single-Article Pipeline
Scrapes a user-provided URL and runs the full intelligence pipeline:
  1. Scrape article (newspaper3k)
  2. Clean text (NLP preprocessing)
  3. Classify category (TF-IDF + LogReg)
  4. Detect fake news (DistilBERT)
  5. Extract keywords (TF-IDF)
  6. External fact-check (NewsAPI + Google Fact Check)

Returns a plain dict with all results — does NOT persist to the database.
"""

import re
import os
import logging
import requests
from urllib.parse import urlparse

from newspaper import Article as NewsArticle, Config

from src.preprocessing.text_cleaner import clean_text
from src.intelligence.classifier import classify_article, load_classifier
from src.intelligence.fake_news import (
    detect_fake_news, load_fake_news_detector
)
from src.intelligence.keyword_extractor import extract_keywords
from src.intelligence.fact_checker import verify_article

logger = logging.getLogger("ai_news.url_verifier")

# ── Module-Level Model Cache ─────────────────────────────────────────────────
# Loaded once on first call, reused for every subsequent /verify invocation.
_classifier_model = None
_fake_news_model = None
_fake_news_tokenizer = None
_models_loaded = False


def _ensure_models_loaded():
    """Lazy-load all AI models once and cache them at module level."""
    global _classifier_model, _fake_news_model, _fake_news_tokenizer, _models_loaded

    if _models_loaded:
        return

    logger.info("Loading AI models for URL verification (first call)...")

    # News category classifier (TF-IDF + LogReg sklearn pipeline)
    try:
        _classifier_model = load_classifier()
    except Exception as e:
        logger.warning("Could not load classifier: %s", e)
        _classifier_model = None

    # Fake news detector (DistilBERT)
    try:
        from src.intelligence.fake_news import load_fake_news_detector
        _fake_news_model, _fake_news_tokenizer = load_fake_news_detector()
    except Exception as e:
        logger.warning("Could not load fake news detector: %s", e)
        _fake_news_model = None
        _fake_news_tokenizer = None

    _models_loaded = True
    logger.info("AI models loaded.")


# ── URL Validation ────────────────────────────────────────────────────────────

_URL_REGEX = re.compile(
    r'^https?://'                # http:// or https://
    r'(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}'  # domain
    r'(?:/[^\s]*)?$'             # optional path
)


def _is_valid_url(url: str) -> bool:
    """Basic check that the string looks like an HTTP(S) URL."""
    if not url or not isinstance(url, str):
        return False
    return bool(_URL_REGEX.match(url.strip()))


def _extract_domain(url: str) -> str:
    """Pull a human-readable source name from the URL's domain."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        # Strip 'www.' prefix for cleaner display
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "Unknown"


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def verify_url(url: str) -> dict:
    """
    Scrape a URL and run the full AI pipeline on the article content.

    Returns a dict with:
        url, title, source, raw_content, clean_content,
        category, category_confidence,
        is_fake, credibility_score, explanation,
        keywords,
        fact_check: { verification_score, cross_reference, fact_check },
        error   — None on success, error string on failure
    """
    result = {
        "url": url,
        "title": None,
        "source": None,
        "raw_content": None,
        "clean_content": None,
        "category": None,
        "category_confidence": None,
        "is_fake": None,
        "credibility_score": None,
        "explanation": None,
        "keywords": [],
        "fact_check": None,
        "error": None,
    }

    # ── Step 0: Validate URL ──────────────────────────────────────────
    url = url.strip()
    if not _is_valid_url(url):
        result["error"] = "Invalid URL. Please provide a valid HTTP or HTTPS link."
        return result

    result["source"] = _extract_domain(url)

    # ── Step 1: Scrape the article ────────────────────────────────────
    try:
        config = Config()
        config.request_timeout = 10
        config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

        article = NewsArticle(url, config=config, keep_article_html=False)
        article.download()

        if article.download_state != 2:  # 2 == SUCCESS
            result["error"] = (
                "Could not download this article. "
                "The site may be paywalled, geo-blocked, or temporarily down."
            )
            return result

        article.parse()
        result["title"] = article.title or "Untitled Article"
        result["raw_content"] = article.text or ""

        if not result["raw_content"] or len(result["raw_content"].strip()) < 50:
            result["error"] = (
                "The article's text content is too short or could not be extracted. "
                "This often happens with paywalled or JavaScript-heavy sites."
            )
            return result

    except Exception as e:
        logger.error("Scrape failed for %s: %s", url, e)
        result["error"] = f"Failed to scrape the article: {e}"
        return result

    # ── Step 1.5: Is it actually news? (AI Zero-Shot Check via HuggingFace API) ─
    try:
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token:
            # Test the first 500 characters
            test_text = f"{result['title']}. {result['raw_content'][:500]}"
            candidate_labels = [
                "news report", "personal opinion blog", "product advertisement", 
                "educational material", "corporate landing page", "software platform"
            ]

            api_url = "https://router.huggingface.co/hf-inference/models/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "inputs": test_text,
                "parameters": {"candidate_labels": candidate_labels}
            }

            hf_response = requests.post(api_url, headers=headers, json=payload, timeout=10)

            if hf_response.status_code == 200:
                zs_result = hf_response.json()
                if isinstance(zs_result, list) and len(zs_result) > 0:
                    zs_result = zs_result[0]
                if isinstance(zs_result, dict) and 'labels' in zs_result and 'scores' in zs_result:
                    top_label = zs_result['labels'][0]
                    top_score = zs_result['scores'][0]

                    logger.info(f"Zero-shot classification: {top_label} ({top_score:.2f})")

                    # If the AI strongly believes this is a blog, ad, landing page, etc (not news)
                    if top_label != "news report" and top_score > 0.40:
                        result["error"] = f"This URL was classified as a '{top_label}' rather than a journalistic news article. Please provide a standard news link."
                        return result
            else:
                logger.warning(f"HuggingFace API returned status {hf_response.status_code}: {hf_response.text[:200]}")
        else:
            logger.info("HF_TOKEN not set, skipping zero-shot content check.")

    except Exception as e:
        logger.error(f"Zero-shot check failed: {e}")
        # Non-fatal backup heuristic: If API fails, at least ensure it has enough text to be a news article
        if len(result["raw_content"].split()) < 150:
            result["error"] = "This URL does not appear to contain a full news article (the extracted text is too short). Please provide a standard news link."
            return result


    # ── Step 2: Clean text ────────────────────────────────────────────
    try:
        result["clean_content"] = clean_text(result["raw_content"])
    except Exception as e:
        logger.warning("Text cleaning failed: %s", e)
        result["clean_content"] = result["raw_content"]

    # ── Load AI models (lazy, cached) ─────────────────────────────────
    _ensure_models_loaded()

    # ── Step 3: Classify category ─────────────────────────────────────
    try:
        if _classifier_model is not None:
            # Classifier works better on raw text (training data wasn't lemmatised)
            category, confidence = classify_article(
                result["raw_content"], model=_classifier_model
            )
            result["category"] = category
            result["category_confidence"] = round(confidence, 4)
        else:
            result["category"] = "N/A (model not loaded)"
    except Exception as e:
        logger.warning("Classification failed: %s", e)
        result["category"] = "N/A"

    try:
        is_fake, credibility, breakdown = detect_fake_news(
            title=result["title"],
            content=result["raw_content"],
            model=_fake_news_model,
            tokenizer=_fake_news_tokenizer,
            source=result["source"],
            verification_result=result.get("fact_check")
        )
        result["is_fake"] = is_fake
        result["credibility_score"] = round(credibility, 4)
        result["explanation"] = breakdown.get("explanation_text", "")
    except Exception as e:
        logger.warning("Fake news detection failed: %s", e)
        result["explanation"] = f"Detection error: {e}"

    # ── Step 5: Keyword extraction ────────────────────────────────────
    try:
        text_for_keywords = result["clean_content"] or result["raw_content"] or ""
        result["keywords"] = extract_keywords(text_for_keywords, top_n=8)
    except Exception as e:
        logger.warning("Keyword extraction failed: %s", e)
        result["keywords"] = []

    # ── Step 6: External fact-check ───────────────────────────────────
    try:
        if result["title"]:
            fact_check_result = verify_article(result["title"])
            result["fact_check"] = fact_check_result

            # If credibility was borderline, let fact-check adjust it
            v_score = fact_check_result.get("verification_score", 0.5)
            if (
                result["credibility_score"] is not None
                and 0.3 <= result["credibility_score"] <= 0.6
                and v_score != 0.5
            ):
                # Re-run detection with verification context for a refined score
                try:
                    is_fake, new_score, new_breakdown = detect_fake_news(
                        title=result["title"],
                        content=result["raw_content"],
                        model=_fake_news_model,
                        tokenizer=_fake_news_tokenizer,
                        source=result["source"],
                        verification_result=fact_check_result,
                    )
                    result["is_fake"] = is_fake
                    result["credibility_score"] = round(new_score, 4)
                    result["explanation"] = new_breakdown.get("explanation_text", "")
                except Exception:
                    pass  # Keep original score on re-run failure
    except Exception as e:
        logger.warning("Fact-check failed: %s", e)
        result["fact_check"] = None

    return result


# ── Quick CLI Test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_url = "https://www.bbc.com/news/technology"
    print("=" * 60)
    print("  URL VERIFIER — Testing")
    print("=" * 60)
    print(f"\nVerifying: {test_url}\n")

    output = verify_url(test_url)
    print(json.dumps(output, indent=2, default=str))
