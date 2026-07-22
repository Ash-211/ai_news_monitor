# Layer 3: Core Intelligence — Function-by-Function Explanation

> **Files covered:**
> - `src/intelligence/classifier.py` — Multi-class news classification
> - `src/intelligence/fake_news.py` — Fake news detection + credibility scoring
> - `src/intelligence/keyword_extractor.py` — TF-IDF keyword extraction
> - `src/intelligence/topic_modeling.py` — LDA topic clustering
> - `src/intelligence/fact_checker.py` — External API fact verification
> - `src/intelligence/pipeline.py` — Pipeline orchestrator

---

## File: `src/intelligence/classifier.py`

### Constants
- `MODELS_DIR` — path to `models/` directory
- `MODEL_PATH` — path to `models/news_classifier.pkl`
- `AG_NEWS_LABELS` — mapping: `{0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}`

---

### Function: `download_ag_news()`

**Purpose:** Downloads the AG News dataset from HuggingFace.

**What it does:**
1. Imports `load_dataset` from HuggingFace `datasets` library
2. `load_dataset("ag_news")` — downloads and caches the dataset
3. Extracts `train['text']`, `train['label']`, `test['text']`, `test['label']`
4. Returns all four lists as a tuple
5. Handles `ImportError` (if `datasets` not installed) and general exceptions

---

### Function: `train_classifier(max_train_samples=20000)`

**Purpose:** Trains a text classifier on AG News and saves the best model.

**Step-by-step:**
1. Calls `download_ag_news()` to get training/test data
2. **Subsampling:** If training data > `max_train_samples`, randomly selects 20,000 samples (seeded with `RandomState(42)` for reproducibility)
3. Defines two classifiers to evaluate:
   - Logistic Regression (`max_iter=1000, n_jobs=-1`)
   - Multinomial Naive Bayes (`alpha=0.1`)
4. For each classifier:
   a. Creates a `Pipeline([TfidfVectorizer, Classifier])`
   b. TF-IDF config: 50K features, (1,2)-grams, English stopwords, sublinear TF
   c. `pipeline.fit(train_texts, train_labels)`
   d. `pipeline.predict(test_texts)`
   e. Computes accuracy score
   f. Prints full classification report (precision, recall, F1 per class)
5. Selects the classifier with highest accuracy
6. `pickle.dump(best_model)` → saves to `models/news_classifier.pkl`
7. Returns the trained pipeline

---

### Function: `load_classifier()`

**Purpose:** Loads the pre-trained classifier from disk.

**What it does:**
1. Checks if `models/news_classifier.pkl` exists
2. `pickle.load()` → returns the trained sklearn Pipeline
3. Returns `None` if file not found

---

### Function: `classify_article(text, model=None)`

**Purpose:** Classifies a single article into a category.

**What it does:**
1. Loads model from disk if not provided
2. Returns `("Unknown", 0.0)` if text is empty or too short (< 5 chars)
3. `model.predict([text])` → gets the predicted class index
4. `model.predict_proba([text])` → gets probability distribution over 4 classes
5. Takes the maximum probability as the confidence score
6. If confidence < 0.35, returns `"General"` instead (low-confidence fallback)
7. Maps the numeric prediction to a label via `AG_NEWS_LABELS`
8. Returns `(category, confidence)` tuple

---

### Function: `classify_batch(texts, model=None)`

**Purpose:** Classifies a batch of articles at once (more efficient).

**What it does:**
1. Loads model from disk if not provided
2. Separates valid texts (≥ 5 chars) from invalid ones
3. Runs batch prediction on all valid texts at once
4. For each valid text, applies the 0.35 confidence threshold
5. Reconstructs results in original order (invalid texts get `("Unknown", 0.0)`)
6. Returns list of `(category, confidence)` tuples

---

## File: `src/intelligence/fake_news.py`

This is the **most complex module** in the project, containing 683 lines of code.

### Constants
- `TRUSTED_SOURCES` — list of 18 trusted news brands: BBC, Reuters, TechCrunch, NDTV, Times of India, Indian Express, etc.
- `FAKE_THRESHOLD = 0.40` — articles below this credibility score are labeled "Fake"
- `TRUSTED_SOURCE_FLOOR = 0.55` — minimum credibility for trusted sources
- `MIN_NEWS_WORDS = 50` — articles shorter than this get penalized
- `PROFANITY_LIST` — 19 profane words to detect
- `GLUE_WORDS` — 31 function/structure words used to check article quality

---

### Function: `download_fake_news_dataset()`

**Purpose:** Downloads training data for the fake news model.

**Three fallback options:**
1. **HuggingFace:** `load_dataset("GonzaloA/fake_news", split="train")` — filters samples with text > 50 chars
2. **Local CSV:** Reads `data/True.csv` and `data/Fake.csv` (Kaggle format), concatenates title + text
3. **Demo dataset:** Calls `_get_demo_dataset()` — returns 400 synthetic samples

---

### Function: `_get_indian_news_augmentation()`

**Purpose:** Returns curated real Indian/international news samples to prevent Indian vocabulary from being classified as fake.

**What it does:**
1. Defines 26 hand-written real Indian news samples covering Indian politics, diplomacy, economy, defense, ISRO, Supreme Court, etc.
2. Multiplies them by 8 = **208 samples**
3. Optionally reads up to 500 verified real articles from the database (articles where `is_fake == False`)
4. Combines and returns with label 0 (Real)

---

### Function: `_get_indian_fake_news_augmentation()`

**Purpose:** Returns curated fake Indian news samples to balance the augmented real data.

**What it does:**
1. Defines 30 hand-written fake news samples mimicking:
   - UNESCO hoaxes
   - WhatsApp forwards
   - Medical misinformation
   - Communal fear-mongering
   - Financial scams
   - Conspiracy theories
   - Political fabrications
2. Multiplies them by 60 = **1,800 samples**
3. Returns with label 1 (Fake)

---

### Function: `train_fake_news_detector(max_samples=20000)`

**Purpose:** Trains the binary fake news classifier with augmented data.

**Step-by-step:**
1. Downloads base dataset → subsamples to 20,000 if needed
2. Augments with Indian real news (208+ samples)
3. Augments with Indian fake news (1,800 samples)
4. Total: ~22,000-22,500 samples
5. `train_test_split(test_size=0.2, stratify=labels)` — stratified to maintain class balance
6. Trains Logistic Regression and Naive Bayes (same Pipeline approach as classifier)
7. Picks the best, saves to `models/fake_news_detector.pkl`

---

### Function: `load_fake_news_detector()`

**Purpose:** Loads the trained model from disk. Returns `None` if not found.

---

### Function: `_check_text_quality(text)`

**Purpose:** Analyzes text for quality issues that indicate low credibility.

**Returns a dict with flags:**
| Flag | Condition |
|------|-----------|
| `has_profanity` | Any word matches the PROFANITY_LIST |
| `is_repetitive` | Unique word ratio < 30% AND word count > 20 |
| `lacks_structure` | Function word ratio < 15% AND word count > 10 |
| `is_random` | Average word length < 3.0 or > 12.0 AND word count > 10 |
| `bad_capitalization` | Capitalized word ratio < 5% AND word count > 10 |
| `is_low_quality` | Any of the above is true |
| `word_count` | Total number of words |

---

### Function: `_apply_scoring_heuristics(title, content, content_score, source=None, corroboration_count=0, verification_result=None)`

**Purpose:** The scoring engine — takes the raw ML probability and applies bonuses/penalties to produce the final credibility score.

**This is the heart of the credibility system.**

**Step-by-step:**
1. **Headline Analysis:** Calls `analyze_linguistic_style(title)` → gets sensationalism and objectivity scores → computes headline_score
2. **Base Combined Score:** `(0.3 × headline_score) + (0.7 × content_score)` — headlines weigh 30%, content weighs 70%
3. **Quality Check:** Calls `_check_text_quality(content)`
4. **Trusted Source Detection:** Checks if the article source matches any brand in `TRUSTED_SOURCES` (case-insensitive substring match)
5. **Source Bonus:** If trusted AND base score > 0.30 → adds +15%
6. **Corroboration Bonus:** If corroboration_count ≥ 1 → adds +10%
7. **No Corroboration Penalty:** If NOT trusted AND zero corroborations → subtracts 15%
8. **Quality Penalties:**
   - Profanity detected → -70%
   - Random/nonsensical text → -55%
   - Bad capitalization → -55%
   - Repetitive text → -45%
   - Lacks structure → -45%
   - For trusted sources (non-severe): penalty halved
9. **Brevity Penalty:** If word count < 50 → penalty = missing_words × 0.015 (capped at -40%)
   - For trusted sources: step is 0.005 and cap is -20%
10. **External Verification:** If verification_result is provided:
    - Score ≥ 0.70 → +15% bonus
    - Score ≤ 0.30 → -15% penalty
11. **Clamp:** Final score is clamped to [0.01, 1.00]
12. **Trust Floor:** If trusted AND base > 0.30 AND no profanity/nonsense → floor at 55%
13. **Final Decision:** `is_fake = (final_score < 0.40)`
14. **Returns:** `(is_fake, final_score, breakdown_dict)` — breakdown contains every factor for transparency

---

### Function: `detect_fake_news(title, content, model=None, source=None, corroboration_count=0, verification_result=None)`

**Purpose:** Public API for single-article detection.

**What it does:**
1. Loads model from disk if not provided
2. Falls back to title as content if content is too short (< 10 chars)
3. `model.predict_proba([content])` → gets the probability that the article is real (index 0)
4. Passes the ML score to `_apply_scoring_heuristics()` with all metadata
5. Returns `(is_fake, final_score, breakdown_dict)`

---

### Function: `detect_batch(titles, contents, model=None, sources=None, corroboration_counts=None)`

**Purpose:** Batch version of `detect_fake_news()` — processes multiple articles at once.

**What it does:**
1. Validates each article (falls back to title if content is short)
2. Runs batch `predict_proba()` on all valid texts
3. Applies `_apply_scoring_heuristics()` individually for each article
4. Returns list of `(is_fake, final_score, breakdown_dict)` tuples

---

### Function: `explain_prediction(text, model=None, top_n=4)`

**Purpose:** Explainable AI — identifies which words most influenced the model's decision.

**How it works:**
1. Gets the TF-IDF feature vector for the text
2. Gets the Logistic Regression coefficients
3. Element-wise multiplies: `tfidf_weights × lr_coefficients`
4. Finds non-zero entries and sorts by absolute value
5. Splits into:
   - **Trust terms** (negative coefficients → push toward "Real")
   - **Risk terms** (positive coefficients → push toward "Fake")
6. Returns `{"trust_terms": [...], "risk_terms": [...]}` with impact percentages

---

### Function: `analyze_linguistic_style(text)`

**Purpose:** Mathematical analysis of headline style.

**Metrics computed:**
- **Sensationalism Score (0-100):** Weighted combination of:
  - 40%: ALL CAPS word ratio (words >2 chars that are fully capitalized)
  - 30%: Punctuation density (! and ? per 100 characters)
  - 30%: Clickbait term matches (shocking, exposed, unbelievable, etc.)
- **Objectivity Score (0-100):** Starts at 100, subtracts 15 per subjective marker found (amazing, terrible, worst, etc.), minimum 30
- Also returns: `caps_ratio`, `punc_count`

---

## File: `src/intelligence/keyword_extractor.py`

### Function: `extract_keywords(text, top_n=10)`

**Purpose:** Extracts the most important keywords from a single article.

**How it works:**
1. Splits the text into sentences (by period)
2. If only 1 sentence, returns unique words (simple fallback)
3. If ≥ 2 sentences, creates a mini-corpus where each sentence is a "document"
4. Fits a TF-IDF Vectorizer (500 features, unigrams + bigrams, English stopwords)
5. Sums TF-IDF scores across all sentences for each term
6. Returns the top N terms by summed score

---

### Function: `extract_keywords_batch(texts, top_n=10)`

**Purpose:** Extracts keywords for multiple articles simultaneously.

**Advantage over single extraction:** Uses the entire batch as the corpus for TF-IDF, so IDF scores reflect cross-article term frequency — words unique to one article score higher.

**How it works:**
1. Replaces empty/invalid texts with the string "empty"
2. Fits a TF-IDF Vectorizer on the full batch (1000 features)
3. For each document row, sorts features by TF-IDF score
4. Returns the top N non-zero features per document

---

## File: `src/intelligence/topic_modeling.py`

### Function: `train_lda_model(texts, num_topics=5)`

**Purpose:** Trains an LDA topic model on the article corpus.

**Step-by-step:**
1. Filters out texts shorter than 10 characters
2. Requires at least 2 valid documents
3. Creates a `CountVectorizer`:
   - 5,000 features
   - English stopwords removed
   - `max_df=0.85` (ignore terms in >85% of docs)
   - `min_df=2` (ignore terms in <2 docs)
4. Builds the Document-Term Matrix (DTM) — a sparse matrix
5. If the vectorizer fails (too few unique terms), retries with relaxed constraints
6. Creates `LatentDirichletAllocation(n_components=5, max_iter=20, learning_method='online')`
7. `lda_model.fit(dtm)` — runs the EM algorithm to discover topic distributions
8. Saves `lda_model.pkl` and `lda_vectorizer.pkl`
9. Returns `(lda_model, vectorizer)`

---

### Function: `load_lda_model()`

**Purpose:** Loads saved LDA model and vectorizer from disk. Returns `(None, None)` if not found.

---

### Function: `get_topic_for_document(text, lda_model=None, vectorizer=None)`

**Purpose:** Assigns a single article to its dominant topic cluster.

**What it does:**
1. Transforms text into a BoW vector via the saved vectorizer
2. `lda_model.transform(vector)` → probability distribution over 5 topics
3. `np.argmax()` → returns the topic ID with highest probability
4. Returns -1 on failure

---

### Function: `get_topics_batch(texts, lda_model=None, vectorizer=None)`

**Purpose:** Batch version — assigns topic IDs to multiple articles.

---

### Function: `print_topics(lda_model, vectorizer, num_words=8)`

**Purpose:** Prints the top 8 words for each discovered topic (for human interpretability).

---

## File: `src/intelligence/fact_checker.py`

### Function: `_extract_search_keywords(title, max_words=6)`

**Purpose:** Strips filler words from a headline to create a focused search query for APIs.

**What it does:**
1. Defines 50+ stopwords (the, a, an, is, are, for, etc.)
2. Splits the title into words
3. Filters out stopwords and words with ≤ 2 characters
4. Returns the first 6 meaningful words joined by spaces

---

### Function: `cross_reference_news(title, api_key=None)`

**Purpose:** Queries NewsAPI to check if other outlets reported the same story.

**Scoring logic:**
| Results Found | Score | Interpretation |
|---------------|-------|----------------|
| 0 | 0.2 | Suspicious — nobody else reports this |
| 1–2 | 0.4 | Weak corroboration |
| 3–5 | 0.6 | Moderate corroboration |
| 5–10 | 0.8 | Strong corroboration |
| 10+ | 1.0 | Widely reported |

**Returns:** `{score, total_results, matching_sources, status}`

---

### Function: `check_fact_claim(title, api_key=None)`

**Purpose:** Queries Google Fact Check Tools API to see if professional fact-checkers have reviewed the claim.

**What it does:**
1. Extracts up to 8 keywords from the title
2. Queries the Google Fact Check API
3. Parses `claimReview` ratings from fact-checkers (e.g., "False", "Mostly True")
4. Counts positive ratings (true, correct, accurate) vs. negative ratings (false, fake, misleading)
5. Computes a score:
   - More negative than positive → score < 0.5
   - More positive than negative → score > 0.5
   - All ambiguous → score = 0.5

**Returns:** `{score, claims_found, ratings, fact_check_urls, status}`

---

### Function: `verify_article(title, newsapi_key=None, google_key=None)`

**Purpose:** Orchestrator — runs both NewsAPI and Google Fact Check, combines results.

**Combination logic:**
- Both APIs returned data → 50/50 weighted average
- Only NewsAPI → use its score
- Only Google → use its score
- Neither → neutral fallback (0.5)

**Returns:** `{verification_score, cross_reference: {...}, fact_check: {...}}`

---

## File: `src/intelligence/pipeline.py`

### Function: `run_intelligence_pipeline()`

**Purpose:** The master orchestrator — runs all Layer 3 modules in sequence on unprocessed articles.

**Step-by-step:**

#### Pre-pass Cleanup
Resets stale scores from old model versions:
```sql
UPDATE articles SET is_fake = NULL, credibility_score = NULL
WHERE credibility_score < 0.08 AND is_fake = TRUE
```

#### Step 1/4: News Classification
1. Queries for articles with any NULL intelligence field
2. Loads the classifier model
3. Calls `classify_batch()` on raw texts (raw performs better than lemmatized for this model)
4. Updates `article.category` for articles with NULL category

#### Step 2/4: Topic Modeling (LDA)
1. Loads LDA model from disk
2. If no model exists, trains a new one on current articles (needs ≥ 5 articles)
3. Calls `get_topics_batch()` to assign topic cluster IDs
4. Updates `article.topic_cluster`
5. Commits changes (intermediate commit)

#### Step 3/4: Keyword Extraction (TF-IDF)
1. Calls `extract_keywords_batch()` on cleaned texts
2. Joins keywords with commas
3. Updates `article.keywords`

#### Step 4/4: Fake News Detection
1. Loads the fake news detector model
2. For each article, computes `corroboration_count`:
   - Counts other articles in the same topic cluster from trusted sources
3. Calls `detect_batch()` with titles, contents, sources, and corroboration counts
4. Updates `article.is_fake` and `article.credibility_score`

#### Step 4b: External Fact-Check for Uncertain Articles
1. Identifies "unsure" articles where 0.30 ≤ credibility_score ≤ 0.60
2. For each, calls `verify_article(title)` to query NewsAPI + Google Fact Check
3. If verification returned a non-neutral score, re-runs `detect_fake_news()` with the verification data
4. Updates is_fake and credibility_score with the refined values

#### Final
- Commits all changes
- Prints a sample of 3 processed articles for verification
- Returns total count

---

## Complete Intelligence Pipeline Flow

```
                    ┌─────────────────────────┐
                    │   SQLite Database        │
                    │  (articles with NULLs)   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Pre-pass: Reset stale   │
                    │  scores (< 8% + fake)    │
                    └────────────┬─────────────┘
                                 │
          ┌──────────────────────▼────────────────────────┐
          │ STEP 1: classify_batch() → article.category   │
          │ (AG News: World / Sports / Business / Sci/Tech)│
          └──────────────────────┬────────────────────────┘
                                 │
          ┌──────────────────────▼────────────────────────┐
          │ STEP 2: get_topics_batch() → topic_cluster    │
          │ (LDA unsupervised topic modeling)              │
          └──────────────────────┬────────────────────────┘
                                 │ COMMIT
          ┌──────────────────────▼────────────────────────┐
          │ STEP 3: extract_keywords_batch() → keywords   │
          │ (TF-IDF scoring across article corpus)        │
          └──────────────────────┬────────────────────────┘
                                 │
          ┌──────────────────────▼────────────────────────┐
          │ STEP 4: detect_batch() → is_fake, cred_score  │
          │ (ML + headline analysis + source bonuses      │
          │  + quality penalties + corroboration)           │
          └──────────────────────┬────────────────────────┘
                                 │
          ┌──────────────────────▼────────────────────────┐
          │ STEP 4b: verify_article() on unsure articles  │
          │ (NewsAPI cross-reference + Google Fact Check)  │
          └──────────────────────┬────────────────────────┘
                                 │ COMMIT
                    ┌────────────▼─────────────┐
                    │   SQLite Database         │
                    │ (all fields populated)    │
                    └──────────────────────────┘
```
