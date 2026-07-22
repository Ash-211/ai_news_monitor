# Layer 1: Data Ingestion — Function-by-Function Explanation

> **Files covered:**
> - `src/ingestion/database.py` — Database schema and connection
> - `src/ingestion/fetcher.py` — Article fetching from APIs and RSS
> - `src/ingestion/scheduler.py` — Background scheduling

---

## File: `src/ingestion/database.py`

This file defines the database schema and connection utilities using SQLAlchemy ORM.

### Class: `Article(Base)`

**Purpose:** Defines the `articles` table schema in SQLite. Every news article in the system is stored as a row in this table.

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-incrementing primary key |
| `title` | String | Article headline (required) |
| `url` | String (UNIQUE) | Article URL — uniqueness prevents duplicates |
| `source` | String | Publisher name (e.g., "BBC News") |
| `author` | String | Author name (optional) |
| `published_at` | DateTime | Publication date (defaults to current UTC time) |
| `raw_content` | Text | Original unprocessed article body |
| `clean_content` | Text | NLP-cleaned version (filled by Layer 2) |
| `category` | String | e.g., "Sci/Tech", "World" (filled by Layer 3) |
| `is_fake` | Boolean | True = flagged fake (filled by Layer 3) |
| `topic_cluster` | Integer | LDA topic ID (filled by Layer 3) |
| `credibility_score` | Float | 0.0–1.0 credibility confidence (filled by Layer 3) |
| `keywords` | Text | Comma-separated TF-IDF keywords (filled by Layer 3) |
| `summary_extractive` | Text | Extractive summary (filled by Layer 4) |
| `summary_abstractive` | Text | BART-generated summary (filled by Layer 4) |

---

### Function: `get_engine()`

**Purpose:** Creates and returns a SQLAlchemy database engine pointing to `data/database.sqlite`.

**What it does:**
1. Creates the `data/` directory if it doesn't exist
2. Builds the SQLite connection string: `sqlite:///data/database.sqlite`
3. Returns a SQLAlchemy `Engine` object with `echo=False` (no SQL logging)

---

### Function: `init_db()`

**Purpose:** Creates all tables in the database based on the defined ORM models.

**What it does:**
1. Calls `get_engine()` to get a database connection
2. `Base.metadata.create_all(engine)` — inspects all classes inheriting from `Base` and creates their corresponding tables if they don't exist
3. Prints confirmation message

---

### Function: `get_session()`

**Purpose:** Returns a new SQLAlchemy session for database operations.

**What it does:**
1. Gets the engine via `get_engine()`
2. Creates a `sessionmaker` bound to that engine
3. Returns a new `Session()` instance

---

## File: `src/ingestion/fetcher.py`

This file handles fetching articles from external sources and saving them to the database.

### Class: `NewsAPIFetcher`

**Purpose:** Fetches top headlines from the NewsAPI service.

#### `__init__(self)`
- Loads the `NEWSAPI_KEY` from environment variables (`.env` file)
- Creates a `NewsApiClient` instance if the key exists
- Prints a warning if the key is missing

#### `fetch_top_headlines(self, category='technology', language='en', page_size=20)`
- Calls `self.newsapi.get_top_headlines()` with the given parameters
- Returns a list of article dictionaries if `response['status'] == 'ok'`
- Returns an empty list on failure
- **Default behavior:** Fetches 20 technology headlines in English

---

### Class: `RSSFetcher`

**Purpose:** Fetches articles from a list of RSS feed URLs.

#### `__init__(self, feed_urls)`
- Stores the list of RSS feed URLs

#### `fetch_feeds(self)`
- Iterates through each feed URL
- Uses `feedparser.parse(url)` to parse the RSS XML
- For each entry in the feed, creates a standardized article dictionary with keys: `title`, `url`, `source`, `author`, `publishedAt`, `content`
- Returns the combined list of all articles from all feeds
- Handles errors per-feed gracefully (one bad feed doesn't stop others)

---

### Function: `save_articles_to_db(raw_articles, source_type="NewsAPI")`

**Purpose:** Takes raw article dictionaries from any fetcher and persists them to the SQLite database.

**What it does (step by step):**
1. Opens a database session
2. For each article in the input list:
   a. Extracts `title` and `url` — skips if either is missing
   b. Gets initial `raw_content` from the API response
   c. **Full-text scraping:** Uses `newspaper3k` to download and parse the actual article URL:
      - Creates a `NewsArticle(url)` object
      - Downloads the page HTML
      - Parses the article body
      - If the scraped text is longer than the API snippet, replaces `raw_content` with the full text
      - Falls back to the API snippet if scraping fails
   d. Standardizes the `source` field (handles both dict and string formats from different APIs)
   e. Parses `publishedAt` with `dateutil.parser.parse()` and strips timezone info for SQLite compatibility
   f. Creates an `Article` ORM instance
   g. `session.add(article)` + `session.commit()`
   h. If a `IntegrityError` occurs (duplicate URL), rolls back silently
3. Returns the count of newly saved articles

---

### Function: `run_ingestion()`

**Purpose:** Main entry point — orchestrates a complete ingestion cycle.

**What it does:**
1. Prints timestamp for logging
2. Creates a `NewsAPIFetcher` and fetches technology headlines
3. Saves them to the database → prints count
4. Creates an `RSSFetcher` with 7 feed URLs:
   - BBC News World
   - TechCrunch
   - Reuters Top News
   - NDTV Top Stories
   - Times of India
   - Indian Express
   - The Hindu National
5. Fetches all RSS articles
6. Saves them to the database → prints count
7. Prints completion timestamp

---

## File: `src/ingestion/scheduler.py`

### Function: `main()`

**Purpose:** Runs the ingestion cycle on a recurring schedule.

**What it does:**
1. Calls `init_db()` to ensure the database tables exist
2. Registers `run_ingestion()` to run **every 15 minutes** using the `schedule` library
3. Immediately runs one ingestion cycle at startup
4. Enters an infinite loop, checking every 60 seconds for pending tasks
5. Handles `KeyboardInterrupt` for graceful shutdown

---

## Data Flow Summary

```
NewsAPI (technology headlines)  ──────┐
                                      ├──→ save_articles_to_db() ──→ SQLite Database
RSS Feeds (7 sources)  ───────────────┘         │
         │                                      │
         └── For each article:                  │
             1. Extract title, URL, source      │
             2. Scrape full text (newspaper3k)   │
             3. Parse publication date           │
             4. INSERT with duplicate check      │
                                                │
scheduler.py runs this every 15 minutes ────────┘
```
