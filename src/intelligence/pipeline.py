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
        for article in articles:
            text = article.raw_content or article.clean_content or ""
            raw_texts.append(text)

        # ─── Step 1: Classification ──────────────────────────────────
        print("\n[1/4] Running News Classification...")
        classifier_model = load_classifier()
        if classifier_model:
            classifications = classify_batch(raw_texts, classifier_model)
            for i, article in enumerate(articles):
                if article.category is None:
                    category, confidence = classifications[i]
                    article.category = category
            print(f"  ✓ Classified {len(articles)} articles.")
        else:
            print("  ✗ Classifier not trained yet. Skipping.")
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
                print("  ✗ Not enough articles to train LDA. Need at least 5.")

        if lda_model and vectorizer:
            topic_ids = get_topics_batch(texts, lda_model, vectorizer)
            for i, article in enumerate(articles):
                if article.topic_cluster is None:
                    article.topic_cluster = topic_ids[i]
            print(f"  ✓ Assigned topic clusters to {len(articles)} articles.")
        else:
            print("  ✗ Topic modeling skipped (model unavailable).")
            
        session.commit()

        # ─── Step 3: Keyword Extraction ──────────────────────────────
        print("\n[3/4] Extracting Keywords (TF-IDF)...")
        all_keywords = extract_keywords_batch(texts, top_n=10)
        for i, article in enumerate(articles):
            if article.keywords is None and all_keywords[i]:
                article.keywords = ", ".join(all_keywords[i])
        print(f"  ✓ Extracted keywords for {len(articles)} articles.")

        # ─── Step 4: Fake News Detection ─────────────────────────────
        print("\n[4/4] Running Fake News Detection...")
        fake_news_model = load_fake_news_detector()
        if fake_news_model:
            # Prepare sources and corroboration counts
            from src.intelligence.fake_news import TRUSTED_SOURCES
            sources_list = []
            corroboration_counts = []
            
            for article in articles:
                sources_list.append(article.source)
                corr_count = 0
                if article.topic_cluster is not None:
                    # Count other articles with the same topic cluster that are from trusted sources
                    corr_count = session.query(Article).filter(
                        Article.topic_cluster == article.topic_cluster,
                        Article.id != article.id,
                        Article.source.in_(TRUSTED_SOURCES)
                    ).count()
                corroboration_counts.append(corr_count)
                
            detections = detect_batch(raw_texts, fake_news_model, sources_list, corroboration_counts)
            for i, article in enumerate(articles):
                if article.is_fake is None:
                    is_fake, credibility = detections[i]
                    article.is_fake = is_fake
                    article.credibility_score = credibility
            print(f"  ✓ Analyzed {len(articles)} articles for credibility.")
        else:
            print("  ✗ Fake news detector not trained yet. Skipping.")
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
