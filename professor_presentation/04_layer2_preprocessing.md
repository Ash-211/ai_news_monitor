# Layer 2: NLP Preprocessing — Function-by-Function Explanation

> **File covered:** `src/preprocessing/text_cleaner.py`

---

## Module Setup (Top of File)

Before any functions run, the module sets up global NLP tools:

1. **NLTK Stopwords:** Downloads the English stopwords corpus if not already present (`nltk.data.find('corpora/stopwords')`)
2. **NLTK Punkt Tokenizer:** Downloads the sentence tokenizer if not present
3. **spaCy Model:** Loads `en_core_web_sm` — if it's not installed, automatically downloads it via `spacy.cli.download()`
4. **Global objects created:**
   - `stop_words` — a `set` of 179 English stopwords from NLTK
   - `stemmer` — a `PorterStemmer` instance (available but not used by default)
   - `nlp` — spaCy language model pipeline

---

## Function: `clean_text(text, apply_stemming=False, apply_lemmatization=True)`

**Purpose:** Takes a raw text string and returns a cleaned, normalized version suitable for ML models.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `text` | — | The raw input text to clean |
| `apply_stemming` | `False` | Whether to apply Porter stemming |
| `apply_lemmatization` | `True` | Whether to apply spaCy lemmatization |

**Processing pipeline (6 steps):**

### Step 1: Lowercase
```python
text = text.lower()
```
Converts "Apple IS Looking" → "apple is looking"

### Step 2: Punctuation Removal
```python
text = text.translate(str.maketrans('', '', string.punctuation))
```
Removes all punctuation characters. "hello, world!" → "hello world"

### Step 3: Noise Removal
```python
text = re.sub(r'\d+', '', text)        # Remove all digits
text = re.sub(r'\s+', ' ', text).strip()  # Collapse multiple spaces
```
"apple 123 billion" → "apple  billion" → "apple billion"

### Step 4: Tokenization & Stopword Removal
```python
tokens = [word for word in text.split() if word not in stop_words]
```
Splits into words and removes common English words like "the", "is", "at", "which", etc.
"apple is looking at buying" → ["apple", "looking", "buying"]

### Step 5: Stemming (Optional, OFF by default)
```python
if apply_stemming and not apply_lemmatization:
    tokens = [stemmer.stem(word) for word in tokens]
```
Porter stemming aggressively chops word endings: "running" → "run", "happiness" → "happi"
Only used if explicitly enabled AND lemmatization is disabled.

### Step 6: Lemmatization (ON by default)
```python
if apply_lemmatization:
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]
```
Uses spaCy's English model to find the dictionary base form:
"running" → "run", "better" → "good", "mice" → "mouse"
More linguistically accurate than stemming.

**Complete example:**
```
Input:  "Apple IS looking at buying U.K. startup for $1 billion! 123"
Step 1: "apple is looking at buying u.k. startup for $1 billion! 123"
Step 2: "apple is looking at buying uk startup for 1 billion 123"
Step 3: "apple is looking at buying uk startup for  billion "
Step 4: ["apple", "looking", "buying", "uk", "startup", "billion"]
Step 6: ["apple", "look", "buy", "uk", "startup", "billion"]
Output: "apple look buy uk startup billion"
```

---

## Function: `process_uncleaned_articles()`

**Purpose:** Batch processor that fetches all articles from the database that haven't been cleaned yet and applies the cleaning pipeline.

**What it does:**
1. Opens a database session
2. Queries for all articles where `clean_content` is `NULL` or empty string:
   ```python
   session.query(Article).filter(
       (Article.clean_content == None) | (Article.clean_content == "")
   ).all()
   ```
3. For each uncleaned article:
   - If `raw_content` exists, passes it to `clean_text()`
   - Saves the result to `article.clean_content`
4. Commits all changes in a single transaction
5. Returns the count of processed articles
6. Rolls back on any error

---

## Data Flow

```
Database (raw_content)
    │
    ▼
process_uncleaned_articles()
    │
    ├── Filter: clean_content IS NULL
    │
    ▼
clean_text(raw_content)
    │
    ├── 1. Lowercase
    ├── 2. Remove punctuation
    ├── 3. Remove numbers + normalize whitespace
    ├── 4. Remove stopwords (NLTK)
    └── 5. Lemmatize (spaCy en_core_web_sm)
    │
    ▼
Database (clean_content = cleaned text)
```
