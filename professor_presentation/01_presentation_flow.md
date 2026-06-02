# AI-Based News Monitoring & Automated Daily Briefing System — Presentation Guide

> **Purpose:** This document is a structured script for explaining the project to your professor. Follow this flow for a clear, confident, and complete walkthrough.

---

## 🎯 Opening Statement (30 seconds)

> "Good morning/afternoon Professor. The project I'm presenting is an **AI-Based News Monitoring and Automated Daily Briefing System**. It is an end-to-end pipeline that automatically ingests news from live sources, cleans them using NLP, runs AI-based classification and fake news detection, and presents everything on a real-time web dashboard where users can see which articles are credible and which are potentially misleading — along with full transparency into *why* the AI made each decision."

---

## 📋 Flow of the Explanation (Recommended Order)

### 1. Problem Statement (1 minute)

Explain the problem first:

- **Information overload:** Hundreds of articles are published every hour. Readers cannot verify all of them.
- **Fake news epidemic:** Misinformation spreads 6× faster than real news on social media (reference: MIT 2018 study).
- **No centralized verification:** No single tool ingests, classifies, checks credibility, and presents articles with transparent AI reasoning.
- **Our goal:** Build an automated system that does all of this — from data collection to an interactive dashboard — without any manual intervention.

---

### 2. System Architecture Overview (2 minutes)

Describe the **5-layer architecture** (top-down):

| Layer | Name                     | What It Does                                                        |
|-------|--------------------------|---------------------------------------------------------------------|
| 1     | **Data Ingestion**       | Fetches articles from NewsAPI and 7 RSS feeds, scrapes full text, stores in SQLite |
| 2     | **NLP Preprocessing**    | Lowercasing, punctuation removal, stopword removal, lemmatization (spaCy) |
| 3     | **Core Intelligence**    | Multi-class classification (4 categories), fake news detection, topic modeling (LDA), keyword extraction (TF-IDF) |
| 4     | **Briefing Generation**  | Extractive + abstractive summarization (BART transformer)            |
| 5     | **Presentation Layer**   | React + Vite dashboard served via FastAPI REST backend               |

> "Each layer is independent and modular. Layer 1 feeds Layer 2, which feeds Layer 3, and so on. This decoupled design means we can upgrade any single layer (e.g., swap the ML model) without touching the others."

---

### 3. Technology Stack (1–2 minutes)

Walk through the key technologies:

#### Backend (Python)
| Library         | Purpose                                    |
|-----------------|--------------------------------------------|
| `newsapi-python` | Fetching headlines from NewsAPI           |
| `feedparser`     | Parsing RSS feeds (BBC, Reuters, NDTV, etc.) |
| `newspaper3k`   | Scraping full article text from URLs       |
| `SQLAlchemy`     | ORM for SQLite database                   |
| `schedule`       | Recurring background ingestion (every 15 min) |
| `NLTK`           | Stopword removal, stemming                |
| `spaCy`          | Lemmatization (`en_core_web_sm`)          |
| `scikit-learn`   | TF-IDF vectorization, Logistic Regression, Naive Bayes, LDA |
| `transformers`   | BART model for abstractive summarization  |
| `FastAPI`        | REST API serving articles to the frontend |
| `Uvicorn`        | ASGI server running FastAPI               |

#### Frontend (JavaScript)
| Library  | Purpose                                  |
|----------|------------------------------------------|
| `React`  | Component-based UI framework             |
| `Vite`   | Fast build tool and dev server           |

#### Database
- **SQLite** — lightweight, file-based relational database (no external server needed)

---

### 4. Data Flow Walkthrough (3–4 minutes) — THE CORE OF THE DEMO

Walk through the pipeline step by step. This is the most important section:

#### Step 1: Data Ingestion
> "We fetch articles from two sources: the NewsAPI (top headlines in technology category) and 7 RSS feeds including BBC, Reuters, TechCrunch, NDTV, Times of India, Indian Express, and The Hindu. For each article, we also use the `newspaper3k` library to scrape the full body text from the actual URL, because API snippets are often truncated. All articles are saved to a local SQLite database. Duplicates are handled by a unique constraint on the URL column."

#### Step 2: NLP Preprocessing
> "Before any ML model sees the data, we clean it. The cleaning pipeline does: lowercasing → punctuation removal → number removal → whitespace normalization → stopword removal (NLTK) → lemmatization (spaCy). This converts something like 'Apple IS looking at buying U.K. startup for $1 billion!' into 'apple look buy uk startup billion'."

#### Step 3: Classification
> "We trained a Logistic Regression classifier on the AG News dataset (120,000 labeled articles) to categorize every article into one of four sectors: World, Sports, Business, or Sci/Tech. The model uses TF-IDF features (50,000 features, unigrams + bigrams) and achieved **90.2% accuracy** on the test set."

#### Step 4: Fake News Detection
> "This is the most complex module. We trained a separate Logistic Regression model on the HuggingFace `GonzaloA/fake_news` dataset, augmented with 200+ hand-curated Indian political news samples (both real and fake) to prevent the US-politics-biased dataset from misclassifying Indian news. The model achieved **97.5% accuracy**. But we don't just blindly trust the ML model — we apply a multi-factor scoring system:"

- **ML Content Score (70%)** — raw probability from the trained classifier
- **Headline Analysis Score (30%)** — sensationalism detection (ALL CAPS ratio, exclamation density, clickbait term matching), objectivity scoring
- **Trusted Source Bonus (+15%)** — articles from verified outlets (BBC, Reuters, NDTV, etc.) get a credibility boost
- **Cross-Reference Bonus (+10%)** — if other articles in the same topic cluster come from trusted sources
- **Quality Penalties** — deductions for profanity, nonsensical text, poor capitalization, very short articles, repetitive content
- **External Fact-Check** — for "unsure" articles (score 30–60%), we query the Google Fact Check API and NewsAPI to cross-reference with professional fact-checkers
- **Trust Floor** — trusted sources can never go below 55% credibility unless they contain profanity or nonsensical content

#### Step 5: Topic Modeling
> "We use Latent Dirichlet Allocation (LDA) from scikit-learn to discover hidden themes across all articles. The model finds 5 topic clusters and assigns each article to the most dominant one. This enables the cross-reference feature — if multiple articles are about the same topic, they corroborate each other."

#### Step 6: Keyword Extraction
> "For each article, we extract the top 10 keywords using TF-IDF scoring. In batch mode, the entire corpus acts as the IDF reference, so keywords unique to each article bubble up."

#### Step 7: API & Dashboard
> "The FastAPI backend exposes two endpoints: `/api/stats` (aggregate counts) and `/api/articles` (paginated, filterable article feed). The React frontend shows article cards with credibility badges. Users can click the badge to see a full transparency breakdown — exactly how the AI arrived at its score, including ML base score, bonuses, penalties, and even the specific penalty reasons."

---

### 5. Live Demo (2–3 minutes)

Run the system live:

1. Double-click `run.bat` — this opens 5 terminal windows automatically
2. Wait ~30 seconds for all services to initialize
3. Open browser at `http://localhost:5173`
4. Show the dashboard:
   - Point out the **stat cards** (Total Articles, Verified Real, Detected Fake)
   - Click on a **"✓ Verified 87%"** badge to show the transparency breakdown
   - Click on a **"⚠️ Flagged Fake"** badge to show why it's fake (quality penalties, etc.)
   - Use the **sidebar filters** (Verified Authentic, Flagged Fake, categories)
   - Use the **search bar** to find specific articles
   - Click **"Preview"** to see the full cleaned content
   - Show **pagination** for navigating large datasets
5. Show the terminal windows briefly — ingestion running, pipeline processing, API serving

---

### 6. Challenges Faced & Solutions (1 minute)

| Challenge | Solution |
|-----------|----------|
| Model pickle files broke across NumPy versions | Retrained all models locally; excluded `.pkl` from Git |
| Indian political news was misclassified as fake | Augmented training data with 200+ curated Indian news samples (real & fake) |
| "Verified 0%" showing for new articles | API now computes live fallback scores when DB values are NULL |
| Score breakdown didn't show penalties | Added penalty reason tracking and rendered it in the UI |
| Full article text was truncated | Used `newspaper3k` to scrape complete body text from article URLs |

---

### 7. Closing Summary (30 seconds)

> "To summarize, this project demonstrates a complete, production-style AI pipeline — from live data ingestion, through NLP preprocessing and multi-model intelligence analysis, to a polished interactive dashboard. The system doesn't just give a binary 'real or fake' label — it provides **full transparency** into every factor that influenced the AI's decision. This is important because trust in AI systems requires explainability."

---

## 🙋 Potential Professor Questions & Answers

| Question | Answer |
|----------|--------|
| "Why Logistic Regression instead of deep learning?" | Logistic Regression is interpretable, fast to train, and achieved 90%+ accuracy. For a monitoring system that needs to process articles in real-time, speed and interpretability matter more than squeezing out 1-2% more accuracy with a transformer model. |
| "What is TF-IDF?" | Term Frequency–Inverse Document Frequency. It converts text into numerical vectors by measuring how important each word is to a specific document relative to the whole corpus. Common words get low scores, unique/discriminative words get high scores. |
| "What is LDA?" | Latent Dirichlet Allocation — a probabilistic topic model that assumes each document is a mixture of topics, and each topic is a distribution over words. It discovers hidden thematic clusters without any labeled data (unsupervised learning). |
| "Is the fake news model biased?" | The base HuggingFace dataset is US-politics-heavy. We mitigated this by augmenting with curated Indian news samples and adding heuristic layers (trusted source lists, quality checks) that don't depend on the ML model's vocabulary bias. |
| "How do you handle scaling?" | Currently we use SQLite and run everything locally. For production, we would migrate to PostgreSQL, deploy on cloud (AWS/GCP), and use a message queue (RabbitMQ) for async pipeline stages. |
| "What datasets did you use?" | AG News (120k articles, 4 classes) for classification; HuggingFace GonzaloA/fake_news + augmented Indian samples for fake news detection. |
| "What if the API keys expire?" | The system gracefully degrades — if NewsAPI or Google Fact Check API keys are missing or invalid, those modules return neutral scores (0.5) and the system continues with ML-only scoring. |
