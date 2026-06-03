# Project Workflow — End-to-End Data Journey

> This document traces the complete lifecycle of a single news article from the moment it is fetched to the moment it appears on the user's dashboard with a credibility score.

---

## Phase 1: Article Discovery & Ingestion

**Trigger:** `scheduler.py` fires `run_ingestion()` every 15 minutes.

```
                 ┌──────────────┐
     ┌───────────│   NewsAPI    │─── GET top headlines (technology, 20 articles)
     │           └──────────────┘
     │
     │           ┌──────────────┐
     ├───────────│  BBC RSS     │─── Parse XML feed
     ├───────────│ Reuters RSS  │─── Parse XML feed
     ├───────────│ TechCrunch   │─── Parse XML feed
     ├───────────│  NDTV RSS    │─── Parse XML feed
     ├───────────│  TOI RSS     │─── Parse XML feed
     ├───────────│  IE RSS      │─── Parse XML feed
     └───────────│ The Hindu    │─── Parse XML feed
                 └──────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  For each article:  │
              │  1. Has title? URL? │──── No → Skip
              │  2. Download URL    │
              │     (newspaper3k)   │
              │  3. Parse body text │
              │  4. longer than     │
              │     API snippet?    │──── Yes → Use full text
              │  5. Parse pub date  │
              │  6. INSERT into DB  │──── Duplicate URL? → Rollback silently
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   SQLite Database   │
              │   articles table    │
              │                     │
              │  title     ✓        │
              │  url       ✓        │
              │  source    ✓        │
              │  raw_content ✓      │
              │  clean_content NULL  │ ← Not yet cleaned
              │  category    NULL   │ ← Not yet classified
              │  is_fake     NULL   │ ← Not yet scored
              │  credibility NULL   │
              │  keywords    NULL   │
              │  topic_cluster NULL │
              └─────────────────────┘
```

---

## Phase 2: NLP Preprocessing

**Trigger:** `pipeline.py` runs (typically right after ingestion).

`process_uncleaned_articles()` is called from the pipeline (or can be run standalone).

```
              ┌─────────────────────┐
              │  Query: articles    │
              │  WHERE clean_content│
              │  IS NULL            │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  For each article:  │
              │                     │
              │  raw_content:       │
              │  "Apple IS looking  │
              │  at buying U.K.     │
              │  startup for $1     │
              │  billion! 123"      │
              │         │           │
              │    ┌────▼────┐      │
              │    │Lowercase│      │
              │    └────┬────┘      │
              │    ┌────▼────────┐  │
              │    │Remove punct.│  │
              │    └────┬────────┘  │
              │    ┌────▼────────┐  │
              │    │Remove nums  │  │
              │    └────┬────────┘  │
              │    ┌────▼────────┐  │
              │    │Stop words   │  │
              │    └────┬────────┘  │
              │    ┌────▼────────┐  │
              │    │Lemmatize    │  │
              │    │(spaCy)      │  │
              │    └────┬────────┘  │
              │         │           │
              │  clean_content:     │
              │  "apple look buy    │
              │  uk startup billion"│
              └──────────┬──────────┘
                         │
                    UPDATE DB:
                  clean_content = result
```

---

## Phase 3: Intelligence Pipeline

**Trigger:** `python -m src.intelligence.pipeline`

### Step 3.1: Classification

```
┌───────────────────────────────────────┐
│  Input: raw_content (raw works better │
│         for classifier)               │
│                                       │
│  Model: TF-IDF + Logistic Regression  │
│  Dataset: AG News (120K articles)     │
│                                       │
│  Output:                              │
│    category = "Sci/Tech"              │
│    confidence = 0.87                  │
└───────────────────────────────────────┘
```

### Step 3.2: Topic Modeling

```
┌───────────────────────────────────────┐
│  Input: clean_content                 │
│                                       │
│  Model: CountVectorizer + LDA         │
│  Topics: 5 clusters (unsupervised)    │
│                                       │
│  Output:                              │
│    topic_cluster = 2                  │
│                                       │
│  (Enables cross-referencing between   │
│   articles about the same topic)      │
└───────────────────────────────────────┘
```

### Step 3.3: Keyword Extraction

```
┌───────────────────────────────────────┐
│  Input: clean_content (all articles   │
│         as a single TF-IDF corpus)    │
│                                       │
│  Method: TF-IDF Vectorizer            │
│  Features: 1000 (unigrams + bigrams)  │
│                                       │
│  Output:                              │
│    keywords = "ai, startup, apple,    │
│    technology, machine learning, ..." │
└───────────────────────────────────────┘
```

### Step 3.4: Fake News Detection (Multi-Factor)

```
┌───────────────────────────────────────────────────────┐
│                                                       │
│  Input: title + content + source + corroboration_count│
│                                                       │
│  ┌─────────────────────────────────────┐              │
│  │ ML MODEL (TF-IDF + LogReg)         │              │
│  │ predict_proba([content]) → 0.82    │              │
│  │ (82% chance of being REAL)         │              │
│  └────────────────┬────────────────────┘              │
│                   │                                   │
│  ┌────────────────▼────────────────────┐              │
│  │ HEADLINE ANALYSIS                   │              │
│  │ sensationalism: 5%  objectivity: 95%│              │
│  │ headline_score: 0.95               │              │
│  └────────────────┬────────────────────┘              │
│                   │                                   │
│  Base = (0.3 × 0.95) + (0.7 × 0.82) = 0.86          │
│                   │                                   │
│  ┌────────────────▼────────────────────┐              │
│  │ BONUSES                             │              │
│  │ Source = "BBC" ∈ TRUSTED → +0.15   │              │
│  │ Corroboration ≥ 1      → +0.10    │              │
│  └────────────────┬────────────────────┘              │
│                   │                                   │
│  ┌────────────────▼────────────────────┐              │
│  │ PENALTIES                           │              │
│  │ Quality check: OK (no flags)       │              │
│  │ Word count: 450 > 50 (OK)         │              │
│  └────────────────┬────────────────────┘              │
│                   │                                   │
│  Final = 0.86 + 0.15 + 0.10 = 1.11 → clamped to 1.0 │
│                   │                                   │
│  Decision: 1.0 > 0.40 threshold → is_fake = FALSE    │
│                                                       │
│  Output:                                              │
│    is_fake = False                                    │
│    credibility_score = 1.0                            │
│    breakdown = {headline: 0.95, content: 0.82, ...}  │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Step 3.4b: External Verification (for "Unsure" Articles Only)

```
┌───────────────────────────────────────────────────────┐
│  Condition: 0.30 ≤ credibility_score ≤ 0.60          │
│  (The AI is unsure, so we ask the internet)          │
│                                                       │
│  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │ NewsAPI              │  │ Google Fact Check    │   │
│  │ "How many outlets    │  │ "Have professional  │   │
│  │  report this story?" │  │  fact-checkers      │   │
│  │                      │  │  reviewed this?"    │   │
│  │ 8 results → 0.80    │  │ 2 claims: "False"  │   │
│  └──────────┬───────────┘  └──────────┬──────────┘   │
│             │                         │               │
│             └──────────┬──────────────┘               │
│                        │                              │
│           Weighted average: (0.5×0.8)+(0.5×0.1)=0.45│
│                        │                              │
│           Re-run detect_fake_news() with              │
│           verification_result = {score: 0.45}        │
│                        │                              │
│           Adjusts final score accordingly             │
└───────────────────────────────────────────────────────┘
```

---

## Phase 4: API Serving

```
┌───────────────────────────────────────────────────────┐
│  User opens http://localhost:5173                     │
│                                                       │
│  React App calls GET /api/articles?page=1&limit=40   │
│                        │                              │
│  FastAPI:                                             │
│  1. Query DB for articles (paginated, filtered)      │
│  2. FOR EACH article on the page:                    │
│     a. Re-run detect_fake_news() → live breakdown    │
│     b. Run explain_prediction() → trust/risk terms   │
│     c. Run analyze_linguistic_style() → headline     │
│     d. If score 0.3-0.6 → verify_article() too      │
│  3. Build JSON response with all transparency data   │
│  4. Return paginated response                        │
└───────────────────────────────────────────────────────┘
```

---

## Phase 5: Dashboard Rendering

```
┌───────────────────────────────────────────────────────┐
│  React receives JSON response                        │
│                                                       │
│  For each article object:                            │
│  ┌─ ArticleCard ─────────────────────────────────┐  │
│  │                                                │  │
│  │  Header: source | category      badge         │  │
│  │  Badge shows: ✓ Verified 87% (green)          │  │
│  │           or: ⚠️ Flagged Fake 22% (red)       │  │
│  │                                                │  │
│  │  Title: "Apple announces new AI features"     │  │
│  │  Summary: first 200 chars of cleaned content  │  │
│  │                                                │  │
│  │  [Click badge → Score Breakdown]              │  │
│  │  ├── ML Content Analysis:        72%          │  │
│  │  ├── Sensationalism:             5%  ████     │  │
│  │  ├── Objectivity:               100% █████████│  │
│  │  ├── Authoritative Source:      +15%          │  │
│  │  ├── Cross-Reference Bonus:     +10%          │  │
│  │  └── (No quality penalties)                   │  │
│  │                                                │  │
│  │  [Click Preview → Full Content]               │  │
│  │  ├── Scrollable container (max 250px)         │  │
│  │  └── Full cleaned article text                │  │
│  │                                                │  │
│  │  Keywords: [ai] [startup] [apple]             │  │
│  │  Date | Author            [Preview] [Full ↗]  │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  Sidebar allows filtering by:                        │
│  • All / Verified / Fake                             │
│  • By category (World, Sports, Business, Sci/Tech)   │
│  Search bar enables title-based searching             │
│  Pagination: ← 1 2 [3] 4 5 ... 12 →                 │
└───────────────────────────────────────────────────────┘
```

---

## Summary: One Article's Complete Journey

```
1. BBC RSS feed publishes: "Apple announces new AI chip for 2026 iPhones"
2. feedparser.parse() pulls the entry from the RSS XML
3. newspaper3k downloads the full article from the URL (800 words)
4. Saved to SQLite with raw_content, source="BBC", published_at
5. text_cleaner.py lowercases, removes stopwords, lemmatizes → clean_content
6. classifier.py: TF-IDF + LogReg → category = "Sci/Tech" (conf: 0.91)
7. topic_modeling.py: LDA → topic_cluster = 3
8. keyword_extractor.py: TF-IDF → keywords = "apple, chip, ai, iphone, ..."
9. fake_news.py:
   a. ML model: predict_proba → content_score = 0.85
   b. Headline analysis: sensationalism = 0%, objectivity = 100% → headline_score = 1.0
   c. Base combined: (0.3 × 1.0) + (0.7 × 0.85) = 0.895
   d. Source = "BBC" → trusted → +0.15 = 1.045
   e. Topic cluster 3 has Reuters article → corroboration +0.10 = 1.145
   f. Clamped to 1.0
   g. 1.0 > 0.40 → is_fake = False
10. Database updated with all fields
11. User opens dashboard → React calls /api/articles
12. FastAPI returns article with score_details breakdown
13. ArticleCard renders: "✓ Verified 100%" with green badge
14. User clicks badge → sees full scoring transparency
```
