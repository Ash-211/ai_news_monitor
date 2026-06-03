"""
Intelligence Pipeline Orchestrator
Fetches unprocessed articles from the database and runs all Layer 3 modules:
  1. Multi-class news classification → category
  2. Fake news detection → is_fake + credibility_score
  3. Keyword extraction → keywords
  4. Topic modeling → topic_cluster

(Proposal Section 5.3 – 5.5)
"""

from src.ingestion.database import get_session, Article
from src.intelligence.classifier import classify_batch, load_classifier
from src.intelligence.fake_news import detect_batch, load_fake_news_detector
from src.intelligence.keyword_extractor import extract_keywords_batch
from src.intelligence.topic_modeling import (
    train_lda_model, get_topics_batch, load_lda_model, print_topics
)



def run_intelligence_pipeline():
    """
    Main entry point for the Intelligence Layer.
    Fetches articles missing intelligence fields and processes them in batch.
    """
    session = get_session()

    try:
        # Fetch articles that need processing
        # An article needs processing if ANY intelligence field is NULL
        articles = session.query(Article).filter(
            (Article.category == None) |
            (Article.is_fake == None) |
            (Article.keywords == None) |
            (Article.topic_cluster == None)
        ).all()

        if not articles:
            print("No articles pending intelligence processing.")
            return 0

        print(f"\nFound {len(articles)} articles to process through Intelligence Layer.")
        print("=" * 60)

        # Prepare texts — prefer clean_content, fall back to raw_content
        texts = []
        for article in articles:
            text = article.clean_content or article.raw_content or ""
            texts.append(text)

        # Also keep raw texts for classifier (raw text often works better 
        # for trained classifiers since the training data wasn't lemmatized)
        raw_texts = []
        short_content_count = 0
        for article in articles:
            text = article.raw_content or article.clean_content or ""
            raw_texts.append(text)
            if len(text.strip()) < 150:
                short_content_count += 1
        
        if short_content_count > len(articles) * 0.5:
            print(f"  [WARNING] High density of short articles ({short_content_count}/{len(articles)}).")
            print("            AI accuracy (Layer 3) will be significantly degraded.")

        # ─── Step 1: Classification ──────────────────────────────────
        print("\n[1/4] Running News Classification...")
        classifier_model = load_classifier()
        if classifier_model:
            classifications = classify_batch(raw_texts, classifier_model)
            for i, article in enumerate(articles):
                if article.category is None:
                    category, confidence = classifications[i]
                    article.category = category
            print(f"  [OK] Classified {len(articles)} articles.")
        else:
            print("  [SKIP] Classifier not trained yet. Skipping.")
            print("    Run: python -m src.intelligence.classifier")

        # ─── Step 2: Topic Modeling (LDA) ────────────────────────────
        print("\n[2/4] Running Topic Modeling (LDA)...")
        # Check if LDA model exists, train if not
        lda_model, vectorizer = load_lda_model()
        if lda_model is None:
            print("  No existing LDA model. Training on current batch...")
            valid_texts = [t for t in texts if t and len(t.strip()) > 10]
            if len(valid_texts) >= 5:
                lda_model, vectorizer = train_lda_model(valid_texts, num_topics=5)
                if lda_model:
                    print_topics(lda_model, vectorizer)
            else:
                print("  [SKIP] Not enough articles to train LDA. Need at least 5.")

        if lda_model and vectorizer:
            topic_ids = get_topics_batch(texts, lda_model, vectorizer)
            for i, article in enumerate(articles):
                if article.topic_cluster is None:
                    article.topic_cluster = topic_ids[i]
            print(f"  [OK] Assigned topic clusters to {len(articles)} articles.")
        else:
            print("  [SKIP] Topic modeling skipped (model unavailable).")
            
        session.commit()

        # ─── Step 3: Keyword Extraction ──────────────────────────────
        print("\n[3/4] Extracting Keywords (TF-IDF)...")
        all_keywords = extract_keywords_batch(texts, top_n=10)
        for i, article in enumerate(articles):
            if article.keywords is None and all_keywords[i]:
                article.keywords = ", ".join(all_keywords[i])
        print(f"  [OK] Extracted keywords for {len(articles)} articles.")

        # ─── Step 4: Fake News Detection ─────────────────────────────
        print("\n[4/4] Running Fake News Detection...")
        fake_news_model, fake_news_tokenizer = load_fake_news_detector()
        if fake_news_model and fake_news_tokenizer:
            # Build separate arrays for titles and contents for dual-scoring
            detection_titles = []
            detection_contents = []
            sources_list = []
            for article in articles:
                title = article.title or ''
                content = article.clean_content or article.raw_content or ''
                
                detection_titles.append(title)
                detection_contents.append(content)
                sources_list.append(article.source)
                
            detections = detect_batch(
                detection_titles, 
                detection_contents, 
                model=fake_news_model, 
                tokenizer=fake_news_tokenizer, 
                sources=sources_list
            )
            
            import json
            for i, article in enumerate(articles):
                if article.is_fake is None:
                    is_fake, credibility, breakdown = detections[i]
                    article.is_fake = is_fake
                    article.credibility_score = credibility
                    article.score_details = json.dumps(breakdown)
            print(f"  [OK] Analyzed {len(articles)} articles for credibility.")
            
            # ─── Step 4b: External Fact-Check for "unsure" articles ────
            try:
                from src.intelligence.fact_checker import verify_article
                from src.intelligence.fake_news import detect_fake_news
                
                unsure_articles = [
                    a for a in articles 
                    if a.credibility_score is not None and 0.3 <= a.credibility_score <= 0.6
                ]
                
                if unsure_articles:
                    print(f"\n  [4b] Running external fact-check on {len(unsure_articles)} 'unsure' articles...")
                    verified_count = 0
                    for article in unsure_articles:
                        title = article.title or ''
                        content = article.clean_content or article.raw_content or ''
                        
                        verification = verify_article(title)
                        if verification.get("verification_score", 0.5) != 0.5:
                            # Re-run detection with verification data
                            is_fake, new_score, new_breakdown = detect_fake_news(
                                title, content, 
                                model=fake_news_model,
                                tokenizer=fake_news_tokenizer,
                                source=article.source, 
                                verification_result=verification
                            )
                            article.is_fake = is_fake
                            article.credibility_score = new_score
                            article.score_details = json.dumps(new_breakdown)
                            verified_count += 1
                    print(f"  [OK] Externally verified {verified_count} articles.")
                else:
                    print("  [4b] No 'unsure' articles to fact-check.")
            except Exception as e:
                print(f"  [WARN] Fact-check step skipped: {e}")

        else:
            print("  [SKIP] Fake news detector not trained yet. Skipping.")
            print("    Run: python -m src.intelligence.fake_news")

        # ─── Commit all updates ──────────────────────────────────────
        session.commit()
        print("\n" + "=" * 60)
        print(f"Intelligence pipeline complete. Updated {len(articles)} articles.")
        print("=" * 60)

        # Print a sample
        print("\n--- Sample Results ---")
        for article in articles[:3]:
            print(f"\n  Title: {article.title[:60]}...")
            print(f"  Category: {article.category}")
            print(f"  Fake: {article.is_fake} | Credibility: {article.credibility_score}")
            print(f"  Keywords: {article.keywords[:80] if article.keywords else 'N/A'}...")
            print(f"  Topic Cluster: {article.topic_cluster}")

        return len(articles)

    except Exception as e:
        session.rollback()
        print(f"\nERROR in intelligence pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  INTELLIGENCE PIPELINE — Layer 3")
    print("=" * 60)
    run_intelligence_pipeline()
