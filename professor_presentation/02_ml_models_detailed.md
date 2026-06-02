# ML Models — Complete Technical Details

> This document covers every ML model used in the project: what it does, whether it's pre-trained or custom-trained, the exact training procedure, datasets, hyperparameters, and performance metrics.

---

## Overview of All Models

| # | Model | Task | Type | Algorithm | Accuracy |
|---|-------|------|------|-----------|----------|
| 1 | News Classifier | Multi-class article categorization | **Custom-trained** from scratch | TF-IDF + Logistic Regression | **90.2%** |
| 2 | Fake News Detector | Binary authenticity classification | **Custom-trained** from scratch | TF-IDF + Logistic Regression | **97.5%** |
| 3 | LDA Topic Model | Unsupervised topic clustering | **Custom-trained** from scratch | Latent Dirichlet Allocation | N/A (unsupervised) |
| 4 | spaCy `en_core_web_sm` | Lemmatization | **Pre-trained** (downloaded) | CNN-based NER + POS tagger | N/A |
| 5 | BART (facebook/bart-large-cnn) | Abstractive summarization | **Pre-trained** (HuggingFace) | Transformer (Seq2Seq) | N/A |

> **Key distinction:** Models 1, 2, and 3 are **trained from scratch on specific datasets by our code**. Models 4 and 5 are **pre-trained models downloaded and used as-is** (transfer learning / zero-shot).

---

## Model 1: News Classifier

### Purpose
Classifies each ingested news article into one of **4 categories**: World, Sports, Business, Sci/Tech.

### File
`src/intelligence/classifier.py`

### Type
**Custom-trained from scratch** — we wrote the training code, downloaded the dataset, and trained the model locally.

### Dataset
- **Name:** AG News
- **Source:** HuggingFace `ag_news` dataset (downloaded programmatically via `datasets.load_dataset("ag_news")`)
- **Original Source:** The AG News dataset was originally collected from 2,000+ news sources by the academic community (Zhang, Zhao, LeCun 2015)
- **Size:** 120,000 training samples + 7,600 test samples
- **Labels:** 4 classes — World (0), Sports (1), Business (2), Sci/Tech (3)
- **Subsampling:** We use 20,000 training samples (randomly selected with `RandomState(42)`) for faster training

### Feature Extraction
- **Method:** TF-IDF (Term Frequency–Inverse Document Frequency) Vectorization
- **Library:** `sklearn.feature_extraction.text.TfidfVectorizer`
- **Hyperparameters:**
  - `max_features=50,000` — vocabulary size cap
  - `ngram_range=(1, 2)` — both unigrams and bigrams
  - `stop_words='english'` — removes common English stopwords
  - `sublinear_tf=True` — applies `1 + log(tf)` instead of raw term frequency, which dampens the effect of very frequent terms

### Classifiers Evaluated
We train and compare **two algorithms** and pick the best:

| Algorithm | Configuration | Role |
|-----------|--------------|------|
| **Logistic Regression** | `max_iter=1000, random_state=42, n_jobs=-1` | Primary candidate |
| **Multinomial Naive Bayes** | `alpha=0.1` (Laplace smoothing) | Comparison baseline |

### Training Pipeline
```
Raw Text → TF-IDF Vectorizer → Classifier → Predictions
```
Both are wrapped in an `sklearn.pipeline.Pipeline` so the vectorizer and classifier are always applied together.

### Training Procedure (Exact Steps)
1. Download AG News dataset via HuggingFace `datasets` library
2. Subsample to 20,000 training examples for speed
3. For each classifier (Logistic Regression, Naive Bayes):
   a. Build a `Pipeline([TfidfVectorizer, Classifier])`
   b. `pipeline.fit(train_texts, train_labels)`
   c. `pipeline.predict(test_texts)` → compute accuracy and classification report
4. Select the classifier with the highest test accuracy
5. Serialize the winning pipeline using `pickle.dump()` → saves to `models/news_classifier.pkl`

### Results
- **Winning Algorithm:** Logistic Regression
- **Test Accuracy:** 90.2%
- **Per-class metrics:** (generated via `sklearn.metrics.classification_report`)

### Model Size
- `models/news_classifier.pkl` — approximately **9 MB**

### How It's Used at Runtime
```python
from src.intelligence.classifier import classify_article
category, confidence = classify_article("Apple announces new iPhone with AI features")
# Returns: ("Sci/Tech", 0.87)
```
If confidence < 0.35, the system returns "General" instead, to avoid low-confidence misclassifications.

### Retraining Command
```bash
python -m src.intelligence.classifier
```

---

## Model 2: Fake News Detector

### Purpose
Binary classification — labels each article as **Authentic** (Real) or **Potentially Misleading** (Fake) and outputs a credibility confidence score (0.0 to 1.0).

### File
`src/intelligence/fake_news.py`

### Type
**Custom-trained from scratch** with **manual data augmentation** — we wrote the training code, downloaded the base dataset, and augmented it with hand-curated Indian/international news samples.

### Dataset (Multi-Source)

#### Primary Dataset
- **Name:** GonzaloA/fake_news
- **Source:** HuggingFace (`datasets.load_dataset("GonzaloA/fake_news", split="train")`)
- **Size:** ~40,000+ articles
- **Labels:** 0 = Real, 1 = Fake
- **Limitation:** This dataset is heavily biased toward US political news (2016–2020 era). Indian political news vocabulary (names like Modi, Jaishankar, Rajya Sabha, etc.) would be misclassified as "unusual" → flagged fake.

#### Fallback Dataset (if HuggingFace fails)
- **Name:** ISOT / Kaggle Fake News Dataset
- **Files:** `data/True.csv` and `data/Fake.csv`
- **Format:** CSV with columns `title`, `text`, `subject`, `date`
- **Processing:** Title + text are concatenated into `full_text`, entries shorter than 50 characters are discarded

#### Demo Dataset (last resort)
- 5 real news templates × 40 repetitions = 200 samples
- 5 fake news templates × 40 repetitions = 200 samples
- Used only if both HuggingFace and local CSV are unavailable

### Data Augmentation (Critical Differentiator)

#### Real Indian News Augmentation (`_get_indian_news_augmentation()`)
- **26 hand-written** real Indian/international political news samples
- Each sample is **repeated 8 times** for training weight = **208 augmented real samples**
- Additionally pulls up to **500 verified real articles** from the local SQLite database (articles previously marked as `is_fake = False`)
- **Purpose:** Teaches the model that Indian political vocabulary is legitimate news language

#### Fake Indian News Augmentation (`_get_indian_fake_news_augmentation()`)
- **30 hand-written** typical Indian fake news / WhatsApp forward patterns including:
  - UNESCO hoaxes ("UNESCO declares X the best in the world")
  - Medical misinformation ("Drink hot water to cure COVID")
  - Religious/communal fear-mongering
  - Financial scams ("Forward to 10 groups for free data")
  - Conspiracy theories ("Government monitoring all calls")
  - Political fabrications ("Secret MOU with China")
- Each sample is **repeated 60 times** for training weight = **1,800 augmented fake samples**
- **Purpose:** Prevents the model from treating "all Indian vocabulary = real" after adding Indian real samples

#### Total Training Data After Augmentation
```
Base dataset:        ~20,000 samples (subsampled from 40,000)
+ Indian real:       ~208 + up to 500 DB samples
+ Indian fake:       ~1,800 samples
= Total:             ~22,000–22,500 samples
```

### Feature Extraction
Identical to the News Classifier:
- **TF-IDF Vectorizer:** `max_features=50,000, ngram_range=(1,2), stop_words='english', sublinear_tf=True`

### Classifiers Evaluated
| Algorithm | Configuration |
|-----------|--------------|
| **Logistic Regression** | `max_iter=1000, random_state=42, n_jobs=-1` |
| **Multinomial Naive Bayes** | `alpha=0.1` |

### Training Procedure
1. Download base dataset from HuggingFace
2. Subsample to 20,000 if larger
3. Augment with Indian real news samples (208 + DB articles)
4. Augment with Indian fake news samples (1,800)
5. `train_test_split(test_size=0.2, stratify=labels)` — stratified split to maintain class balance
6. Train both classifiers in sklearn Pipeline
7. Compare accuracy, pick the best
8. Save with `pickle.dump()` → `models/fake_news_detector.pkl`

### Results
- **Winning Algorithm:** Logistic Regression
- **Test Accuracy:** 97.5%
- **Per-class metrics:** Precision, Recall, F1-Score for "Authentic" and "Potentially Misleading"

### Model Size
- `models/fake_news_detector.pkl` — approximately **36.8 MB** (larger due to bigger vocabulary from augmentation)

### Post-ML Heuristic Scoring System
The ML model's raw output is **NOT the final score**. It goes through a multi-layer heuristic pipeline:

```
Final Credibility = (0.3 × Headline Score) + (0.7 × ML Content Score)
                  + Source Bonus (if trusted)
                  + Corroboration Bonus (if topic has supporting articles)
                  - Quality Penalties (if low quality detected)
                  + External Verification (if available)
```

#### Detailed Scoring Factors:

| Factor | Value | Condition |
|--------|-------|-----------|
| Headline Score Weight | 30% | Always applied |
| ML Content Score Weight | 70% | Always applied |
| Trusted Source Bonus | +15% | Source is in TRUSTED_SOURCES list AND base score > 0.30 |
| Cross-Reference Bonus | +10% | Article's topic cluster has ≥ 1 other article from a trusted source |
| No Corroboration Penalty | -15% | Not from trusted source AND zero corroborating articles |
| External Verification Boost | +15% | Google Fact Check / NewsAPI verification score ≥ 0.70 |
| External Verification Penalty | -15% | Verification score ≤ 0.30 |
| Profanity Penalty | -70% | Text contains profane words |
| Random/Nonsense Penalty | -55% | Average word length < 3 or > 12 characters |
| Low Structure Penalty | -45% | Function word ratio (the, and, with...) < 15% |
| Repetitive Text Penalty | -45% | Unique word ratio < 30% |
| Short Article Penalty | up to -40% | Word count < 50 (penalty = missing_words × 0.015) |

#### Trust Floor
- **Trusted sources** can never go below **55% credibility** unless they contain profanity or nonsensical content
- Quality penalties for trusted sources are **halved** (except for profanity/nonsense)

#### Fake Threshold
- Any article with a final credibility score **below 40%** is labeled as **Fake**
- Articles **above 40%** are labeled as **Authentic**

### Headline Linguistic Analysis (`analyze_linguistic_style()`)

This sub-module analyzes headline quality to feed into the scoring:

| Metric | How It's Calculated |
|--------|-------------------|
| **Sensationalism Score** (0-100) | 40% × CAPS word ratio + 30% × punctuation density (!?) + 30% × clickbait term count |
| **Objectivity Score** (0-100) | 100 - (subjective marker count × 15), minimum 30 |

**Clickbait terms detected:** shocking, exposed, unbelievable, reveal, secret, won't believe, trick
**Subjective markers:** amazing, terrible, worst, incredible, best, clearly, obviously, actually

### AI Explainability (`explain_prediction()`)
The system provides **XAI (Explainable AI)** by:
1. Extracting the TF-IDF feature weights from the trained model
2. Multiplying them by the Logistic Regression coefficients
3. Sorting by absolute magnitude
4. Returning the top N **trust terms** (words that pushed toward "Real") and **risk terms** (words that pushed toward "Fake")

This allows the dashboard to show users *which specific words* influenced the AI's decision.

### Retraining Command
```bash
python -m src.intelligence.fake_news
```

---

## Model 3: LDA Topic Model

### Purpose
Discovers **hidden thematic clusters** across all articles using unsupervised learning. Assigns each article a `topic_cluster` ID (0–4).

### File
`src/intelligence/topic_modeling.py`

### Type
**Custom-trained from scratch** — trained on whatever articles are currently in the database. Retrains automatically on first pipeline run.

### Algorithm
- **Latent Dirichlet Allocation (LDA)** from `sklearn.decomposition.LatentDirichletAllocation`
- This is an **unsupervised** model — it does **not** need labeled training data

### Feature Extraction
- **CountVectorizer** (Bag of Words) — different from TF-IDF used in other models
- **Hyperparameters:**
  - `max_features=5,000`
  - `stop_words='english'`
  - `max_df=0.85` — ignore terms appearing in > 85% of documents (too common)
  - `min_df=2` — ignore terms appearing in < 2 documents (too rare)
  - `ngram_range=(1, 1)` — unigrams only

### LDA Hyperparameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_components` | 5 | Number of topics to discover |
| `random_state` | 42 | Reproducibility |
| `max_iter` | 20 | Maximum EM iterations |
| `learning_method` | 'online' | Online variational Bayes (faster than batch) |
| `n_jobs` | -1 | Use all CPU cores |

### Training Procedure
1. Collect all cleaned article texts from the database
2. Filter out texts shorter than 10 characters
3. Require at least 5 valid documents (minimum for meaningful topics)
4. Build Document-Term Matrix using CountVectorizer
5. `lda_model.fit(dtm)` — discover topic distributions
6. Save model: `models/lda_model.pkl`
7. Save vectorizer: `models/lda_vectorizer.pkl`

### How Topics Are Assigned
For each new article:
1. Transform the text into a BoW vector using the saved vectorizer
2. `lda_model.transform(vector)` → returns a probability distribution over 5 topics
3. The topic with the **highest probability** is assigned as the cluster ID

### Model Size
- `models/lda_model.pkl` — approximately **165 KB**
- `models/lda_vectorizer.pkl` — approximately **100 KB**

### Role in the Pipeline
- Enables **cross-referencing**: if multiple articles share the same topic cluster, they corroborate each other
- The fake news detector gives a **+10% bonus** to articles whose topic cluster contains at least one other article from a trusted source

### Retraining
The LDA model **automatically retrains** on the first pipeline run if no saved model exists. It can also be retrained manually via:
```bash
python -m src.intelligence.pipeline
```

---

## Model 4: spaCy `en_core_web_sm` (Pre-trained)

### Purpose
Used for **lemmatization** during the NLP preprocessing step.

### File
`src/preprocessing/text_cleaner.py`

### Type
**Pre-trained model, downloaded and used as-is** — we did NOT train this ourselves.

### Details
- **Model:** `en_core_web_sm` — a small English NLP pipeline by Explosion AI
- **Size:** ~12 MB (downloaded on first run)
- **What it does:** Tokenization, POS tagging, dependency parsing, NER, and crucially **lemmatization** (converting words to their base form: "running" → "run", "better" → "good")
- **Training:** This model was trained by the spaCy team on the **OntoNotes 5.0** corpus (1.7M words of English text from news, broadcast, web, etc.)

### How We Use It
```python
doc = nlp("Apple is looking at buying a UK startup")
lemmas = [token.lemma_ for token in doc]
# ['apple', 'be', 'look', 'at', 'buy', 'a', 'uk', 'startup']
```

We only use the **lemmatization** functionality. The NER, POS, and dependency features are loaded but not explicitly used.

---

## Model 5: BART Transformer (Pre-trained, Optional)

### Purpose
**Abstractive summarization** — generates human-like summaries of articles (not just extracting sentences, but rewriting).

### Type
**Pre-trained model from HuggingFace, used as-is** — we did NOT fine-tune this.

### Details
- **Model:** `facebook/bart-large-cnn`
- **Library:** HuggingFace `transformers`
- **Architecture:** BART (Bidirectional and Auto-Regressive Transformers) — a denoising autoencoder for sequence-to-sequence tasks
- **Training:** Pre-trained by Facebook AI on the CNN/DailyMail summarization dataset (300K news articles with reference summaries)
- **Size:** ~1.6 GB (downloaded on first use)

### Note
The BART summarization is listed in `requirements.txt` (`transformers==4.37.2`, `torch==2.1.2`) and mentioned in the README as Layer 4 (Briefing Generation Layer). The primary summarization shown in the dashboard uses the cleaned article content (extractive approach via truncation) rather than BART-generated summaries, due to the computational cost of running BART on every article in real-time.

---

## Summary Comparison Table

| Aspect | News Classifier | Fake News Detector | LDA Topic Model | spaCy | BART |
|--------|----------------|-------------------|-----------------|-------|------|
| **Trained by us?** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ Pre-trained | ❌ Pre-trained |
| **Learning type** | Supervised | Supervised | Unsupervised | Supervised (by spaCy team) | Supervised (by Facebook) |
| **Algorithm** | TF-IDF + Logistic Regression | TF-IDF + Logistic Regression | CountVectorizer + LDA | CNN pipeline | Transformer (Seq2Seq) |
| **Dataset** | AG News (120K) | GonzaloA/fake_news + Indian augmentation | Live article corpus | OntoNotes 5.0 | CNN/DailyMail |
| **Disk size** | ~9 MB | ~36.8 MB | ~265 KB | ~12 MB | ~1.6 GB |
| **Accuracy** | 90.2% | 97.5% | N/A | N/A | N/A |
| **Output** | Category + confidence | is_fake + credibility score + breakdown | Topic cluster ID | Lemmatized tokens | Summary text |
