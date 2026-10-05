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
        # LIMIT 100 to prevent crashing the Hugging Face API with massive context windows
        articles = session.query(Article).filter(
            (Article.category == None) |
            (Article.is_fake == None) |
            (Article.keywords == None) |
            (Article.topic_cluster == None)
        ).limit(100).all()

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

        # ─── Step 3b: Event Linker Embeddings ────────────────────────
        print("\n[3b/4] Generating Embeddings for Event Linking...")
        try:
            from src.intelligence.event_linker import get_embedding
            embeddings_generated = 0
            for article in articles:
                if article.embedding is None:
                    text_for_embed = f"{article.title} {article.clean_content or article.raw_content or ''}"[:500]
                    article.embedding = get_embedding(text_for_embed)
                    embeddings_generated += 1
            print(f"  [OK] Generated vector embeddings for {embeddings_generated} articles.")
        except Exception as e:
            print(f"  [WARN] Failed to generate embeddings: {e}")

        # ─── Step 4: Fake News Detection ─────────────────────────────
        print("\n[4/4] Running Fake News Detection...")
        fake_news_model, fake_news_tokenizer = load_fake_news_detector()
        if fake_news_model and fake_news_tokenizer:
            # Build list of dicts matching detect_batch(items: list[dict]) signature
            batch_items = []
            for article in articles:
                batch_items.append({
                    "title": article.title or '',
                    "content": article.raw_content or article.clean_content or '',
                    "source": article.source or '',
                })
                
            analyzed_items = detect_batch(
                batch_items,
                model=fake_news_model, 
                tokenizer=fake_news_tokenizer,
            )
            
            import json
            for i, article in enumerate(articles):
                if article.is_fake is None:
                    analysis = analyzed_items[i].get("analysis", {})
                    article.is_fake = analysis.get("is_fake", None)
                    article.credibility_score = analysis.get("credibility_score", None)
                    article.score_details = json.dumps({
                        "explanation_text": analysis.get("explanation", ""),
                        "verdict": analysis.get("verdict", ""),
                        "fact_score": analysis.get("fact_score", 0.5),
                        "model_score": analysis.get("model_score", 0.5)
                    })
            print(f"  [OK] Analyzed {len(articles)} articles for credibility.")
            
            # ─── Step 4b: External Fact-Check for "unsure" articles ────
            try:
                from src.intelligence.fact_checker import verify_article
                
                unsure_articles = [
                    a for a in articles 
                    if a.credibility_score is not None and 0.3 <= a.credibility_score <= 0.6
                ]
                
                if unsure_articles:
                    print(f"\n  [4b] Running external fact-check on {len(unsure_articles)} 'unsure' articles...")
                    verified_count = 0
                    for article in unsure_articles:
                        title = article.title or ''
                        
                        verification = verify_article(title)
                        v_score = verification.get("verification_score", 0.5)
                        
                        if v_score != 0.5:
                            # Merge external verification with LLM fact score
                            try:
                                details = json.loads(article.score_details) if article.score_details else {}
                            except:
                                details = {}
                                
                            llm_fact = details.get("fact_score", 0.5)
                            model_score = details.get("model_score", 0.5)
                            
                            # Average the two fact-checking sources
                            new_fact_score = (llm_fact + v_score) / 2.0
                            
                            # Recalculate combined score (2/3 fact, 1/3 model)
                            new_final_score = (new_fact_score * 0.667) + (model_score * 0.333)
                            
                            from src.intelligence.fake_news import FAKE_THRESHOLD
                            article.is_fake = bool(new_final_score < FAKE_THRESHOLD)
                            article.credibility_score = new_final_score
                            
                            details["fact_score"] = new_fact_score
                            article.score_details = json.dumps(details)
                            verified_count += 1
                    print(f"  [OK] Externally verified {verified_count} articles.")
                else:
                    print("  [4b] No 'unsure' articles to fact-check.")
            except Exception as e:
                print(f"  [WARN] Fact-check step skipped: {e}")

        else:
            print("  [SKIP] Fake news detector not trained yet. Skipping.")
            print("    Run: python -m src.intelligence.fake_news")

        # ─── Step 5: Automated Deepfake Detection ────────────────────
        print("\n[5] Running Automated Deepfake Detection on Article Images...")
        try:
            import os
            import tempfile
            import requests as img_requests
            from src.ingestion.image_filter import (
                check_trusted_source, check_exif_authenticity, check_ai_dimensions
            )

            pending_images = session.query(Article).filter(
                Article.image_status == 'pending',
                Article.image_url.isnot(None),
            ).all()

            if not pending_images:
                print("  No pending images to analyze.")
            else:
                print(f"  Found {len(pending_images)} images pending deepfake analysis.")
                api_calls_made = 0
                filter_skipped = 0

                for article in pending_images:
                    image_url = article.image_url
                    tmp_path = None

                    try:
                        # ── Filter 4: URL Deduplication ───────────────
                        existing = session.query(Article).filter(
                            Article.image_url == image_url,
                            Article.image_status.in_(['real', 'deepfake']),
                            Article.id != article.id,
                        ).first()

                        if existing:
                            article.image_status = existing.image_status
                            article.deepfake_score = existing.deepfake_score
                            filter_skipped += 1
                            print(f"    [Filter 4] URL dedup: reused #{existing.id} -> {image_url[:60]}")
                            continue

                        # ── Filter 5: Trusted Source Bypass ───────────
                        verdict = check_trusted_source(article.source)
                        if verdict:
                            article.image_status = verdict['image_status']
                            article.deepfake_score = verdict['deepfake_score']
                            filter_skipped += 1
                            print("    [Filter 5] Trusted source bypass applied.")
                            continue

                        # ── Download image for local analysis ─────────
                        with img_requests.get(
                            image_url, timeout=8, stream=True,
                            headers={'User-Agent': 'Mozilla/5.0 (NewsMonitor/1.0)'}
                        ) as resp:
                            if resp.status_code != 200:
                                article.image_status = 'discarded'
                                print(f"    [Download] Failed ({resp.status_code}): {image_url[:60]}")
                                continue

                            ct = resp.headers.get('Content-Type', '').lower()
                            if not ct.startswith('image/'):
                                article.image_status = 'discarded'
                                print(f"    [Download] Non-image response ({ct or 'missing'}): {image_url[:60]}")
                                continue

                            suffix = '.jpg'
                            if 'png' in ct:
                                suffix = '.png'
                            elif 'webp' in ct:
                                suffix = '.webp'

                            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                                for chunk in resp.iter_content(8192):
                                    if chunk:
                                        tmp.write(chunk)
                                tmp_path = tmp.name

                        # ── Filter 6: EXIF Camera Metadata ───────────
                        verdict = check_exif_authenticity(tmp_path)
                        if verdict:
                            article.image_status = verdict['image_status']
                            article.deepfake_score = verdict['deepfake_score']
                            filter_skipped += 1
                            print(f"    [Filter 6] {verdict['reason']}")
                            continue

                        # ── Filter 7: AI Dimension Fingerprinting ────
                        verdict = check_ai_dimensions(tmp_path)
                        if verdict and verdict.get('skip_api'):
                            article.image_status = verdict['image_status']
                            article.deepfake_score = verdict['deepfake_score']
                            filter_skipped += 1
                            print(f"    [Filter 7] {verdict['reason']}")
                            continue

                        # ── C2PA + Sightengine (existing detector) ────
                        from src.intelligence.deepfake_detector import detect_deepfake_image
                        result = detect_deepfake_image(tmp_path, article_context=article.title)

                        article.deepfake_score = result.get('raw_scores', {}).get('Overall Fake', 0.5)
                        article.image_status = 'deepfake' if result['is_fake'] else 'real'
                        api_calls_made += 1

                        print(f"    [Detector] {article.image_status.upper()} "
                              f"(score={article.deepfake_score:.2f}): {article.title[:50]}...")

                    except Exception as img_err:
                        print(f"    [Error] {article.title[:40]}...: {img_err}")
                        article.image_status = 'discarded'
                    finally:
                        if tmp_path:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

                # ── Apply 50% Fact / 25% Model / 25% Image score recalculation ─────
                deepfake_penalty_count = 0
                for article in pending_images:
                    if article.image_status in ['deepfake', 'real'] and article.credibility_score is not None:
                        df_score = article.deepfake_score or 0.5
                        
                        try:
                            details = json.loads(article.score_details) if article.score_details else {}
                        except Exception:
                            details = {}
                            
                        # Retrieve previous components
                        fact_score = details.get("fact_score", article.credibility_score)
                        model_score = details.get("model_score", article.credibility_score)
                        
                        # Calculate Image Authenticity
                        image_authenticity = 1.0 - df_score
                        
                        # Apply User's EXACT Requested Formula:
                        # 50% Fact Checking, 25% AI Model, 25% Image Detection
                        old_cred = article.credibility_score
                        new_credibility = (fact_score * 0.50) + (model_score * 0.25) + (image_authenticity * 0.25)
                        
                        article.credibility_score = max(0.01, new_credibility)
                        
                        # Re-evaluate is_fake flag with new score
                        from src.intelligence.fake_news import FAKE_THRESHOLD
                        article.is_fake = bool(article.credibility_score < FAKE_THRESHOLD)
                            
                        details['deepfake_score'] = df_score
                        details['image_authenticity'] = image_authenticity
                        article.score_details = json.dumps(details)

                        deepfake_penalty_count += 1
                        print(f"    [Image Validated] {old_cred:.2f} -> {article.credibility_score:.2f} "
                              f"(Fact: {fact_score:.2f}, Model: {model_score:.2f}, Image: {image_authenticity:.2f}): {article.title[:50]}...")

                print(f"\n  [5] Complete: {api_calls_made} API calls, {filter_skipped} skipped by pre-filters, "
                      f"{deepfake_penalty_count} deepfake penalties applied.")

        except Exception as step5_err:
            print(f"  [WARN] Deepfake detection step failed: {step5_err}")
            import traceback as tb5
            tb5.print_exc()

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
