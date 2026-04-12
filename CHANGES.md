# Change Log — `feat/scoring-fixes-and-ui-polish`

> Branch cut from local working directory on **2026-04-12**.  
> All changes relate to bugfixes, scoring transparency improvements, and developer-experience polish.

---

## 1. Startup & Developer Experience

### `run.bat` — Startup Orchestrator
- **Created** `run.bat` from scratch as a single-click startup script.
- Opens **five separate terminal windows**:
  1. `Data Ingestion` — runs `python -m src.ingestion.fetcher`
  2. `Intelligence Pipeline` — runs `python -m src.intelligence.pipeline`
  3. `API Backend` — runs `uvicorn src.api.main:app --reload --port 8000`
  4. `Frontend Dashboard` — runs `npm run dev` inside `dashboard/`
  5. `Background Scheduler` — runs `python -m src.ingestion.scheduler`
- Previously, the fetcher and pipeline were chained in **one** terminal window with `&&`, which made it impossible to monitor each independently.

### `src/ingestion/scheduler.py` — Ingestion Interval
- Changed interval from **every 1 hour** → **every 15 minutes** for demo purposes.

---

## 2. Credibility Scoring — Bug Fixes

### Problem: "Verified 0%" on new articles
**Root cause:** New articles fetched to the database have `NULL` credibility scores until the pipeline processes them. When the API returned these articles, the frontend defaulted to `0%`.

**Fix (`src/api/main.py`):**  
The API now captures the live-calculated `is_fake_calc` and `final_score` from `detect_fake_news()` and uses them as fallbacks when the database values are `NULL`:
```python
"is_fake": a.is_fake if a.is_fake is not None else is_fake_calc,
"credibility_score": a.credibility_score if a.credibility_score is not None else final_score,
```

---

### Problem: Score mismatch — e.g. "Verified 6%" but ML shows 54%
**Root cause (two separate issues stacked):**

**Issue A — Stale database scores from old model pkl:**  
After retraining the ML models to fix a `numpy._core.numeric` pickle incompatibility error, the pipeline was designed to only score articles with `NULL` fields. This meant previously-scored articles kept their old, broken scores permanently even after model replacement.

**Fix (`src/intelligence/pipeline.py`):**  
Added a pre-pass cleanup step at the start of every pipeline run:
```python
stale_count = session.query(Article).filter(
    Article.credibility_score < 0.08,
    Article.is_fake == True
).update({Article.is_fake: None, Article.credibility_score: None})
```
Any article with a suspiciously low sub-8% fake score gets its intelligence fields wiped and re-processed by the pipeline automatically.

**Issue B — Quality Penalty was hidden in the UI:**  
The backend was correctly computing a penalty (e.g. `-67%` for an article that was too short) and sending it to the frontend in the `penalty_applied` field. However, the `ArticleCard.jsx` component was **not rendering it at all**, making the math appear impossible (54% + 15% + 10% = 6% made no sense to users).

**Fix (`dashboard/src/components/ArticleCard.jsx`):**  
Added a `Quality Penalty` row that renders when `details.penalty_applied < 0`.

---

### Problem: Cross-Reference Bonus showed "Enabled" instead of a percentage
The cross-reference bonus was always `+10%` but the UI just showed the word "Enabled", making it invisible in the math breakdown.

**Fix (`dashboard/src/components/ArticleCard.jsx`):**  
Changed the row to display the actual bonus value:
```jsx
<span className="breakdown-value plus">
  +{Math.round((details.corroboration_boost || 0.10) * 100)}%
</span>
```
Also exposed `corroboration_boost` explicitly from the API so the UI doesn't have to hardcode `0.10`.

---

## 3. Scoring Transparency — Penalty Reasons

### Problem: Users couldn't tell *why* a penalty was applied
The `Quality Penalty -67%` row appeared but gave no context for why the penalty hit.

**Fix (`src/intelligence/fake_news.py`):**  
Added a `penalty_reasons` list to the `breakdown` dict returned by `_apply_scoring_heuristics()`. The list is populated with human-readable strings for each quality flag that triggered:

| Trigger | Reason String |
|---|---|
| `has_profanity` | "Contains profanity or offensive language" |
| `is_random` | "Text appears random or nonsensical" |
| `bad_capitalization` | "Poor or inconsistent capitalization" |
| `is_repetitive` | "Highly repetitive text detected" |
| `lacks_structure` | "Lacks article structure (low function word ratio)" |
| `word_count < MIN_NEWS_WORDS` | "Article too short (N words, min 50)" |
| No trusted source + no corroboration | "No corroborating sources found" |

**Fix (`src/api/main.py`):**  
`penalty_reasons` is now forwarded inside `score_details` in the API response.

**Fix (`dashboard/src/components/ArticleCard.jsx`):**  
Penalty reasons are rendered as a sub-list under the Quality Penalty row:
```
Quality Penalty              -67%
  ↳ Article too short (32 words, min 50)
  ↳ Lacks article structure (low function word ratio)
```

---

## 4. Frontend — Preview Text Consistency

### Problem: Article preview text had inconsistent font sizes and line gaps
Different articles rendered their content with different font sizes and paragraph spacings because there was no dedicated CSS for the preview section.

**Fix (`dashboard/src/index.css`):**  
Added dedicated CSS classes:
- `.article-summary-box` — light gray container with padding and rounded corners
- `.summary-scroll` — max-height 250px with a styled scrollbar
- `.summary-para` — enforces `font-size: 0.95rem`, `line-height: 1.6`, `color: #334155`, `margin-bottom: 0.8rem` on every paragraph

---

## 5. Model Compatibility Fix

### Problem: `ModuleNotFoundError: No module named 'numpy._core.numeric'`
Saved `.pkl` model files (classifier, fake news detector, LDA model) were trained under a different version of NumPy than the one installed. Python's `pickle` cannot deserialize these cross-version binaries.

**Fix:**  
Deleted all stale `.pkl` files and retrained all three models locally:
- `models/news_classifier.pkl` — Logistic Regression, **90.2% accuracy** (AG News dataset)
- `models/fake_news_detector.pkl` — Logistic Regression, **97.5% accuracy** (HuggingFace + augmented Indian news)
- `models/lda_model.pkl` + `lda_vectorizer.pkl` — LDA topic model, retrained on current article corpus

All models are excluded from version control via `.gitignore` and must be retrained locally by running:
```bash
python -m src.intelligence.classifier
python -m src.intelligence.fake_news
python -m src.intelligence.pipeline   # trains LDA on first run
```

---

## 6. `.gitignore` Updates
Added ignores for:
- `models/*.pkl` — large binary ML model files (~35MB each)
- `data/*.sqlite`, `data/*.db` — runtime database
- `dashboard/node_modules/` — npm dependencies
- `logs/` — runtime log output
- `scratch/` — local debug/test scripts
