# Layer 5: API Backend & Presentation Layer — Function-by-Function Explanation

> **Files covered:**
> - `src/api/main.py` — FastAPI REST backend
> - `dashboard/src/App.jsx` — Main React application
> - `dashboard/src/components/Sidebar.jsx` — Navigation sidebar
> - `dashboard/src/components/ArticleCard.jsx` — Article display card
> - `dashboard/src/index.css` — Complete design system

---

## File: `src/api/main.py`

This file is the **FastAPI REST API** that connects the Python backend to the React frontend.

### App Initialization
```python
app = FastAPI(title="AI News API", description="API serving intelligence-processed news articles.")
```
- Adds CORS middleware with `allow_origins=["*"]` to allow the React frontend (running on port 5173) to make API calls

### Function: `get_db_session()`

**Purpose:** Opens a SQLAlchemy session to the SQLite database.

**What it does:**
1. Checks if `data/database.sqlite` exists
2. If not, returns `None` (API will return empty data instead of crashing)
3. Creates engine and returns a session

---

### Endpoint: `GET /api/stats`

**Purpose:** Returns aggregate statistics for the dashboard header cards.

**Response:**
```json
{
    "total_articles": 450,
    "fake_articles": 32,
    "real_articles": 418,
    "categories": {
        "Sci/Tech": 120,
        "World": 95,
        "Business": 130,
        "Sports": 105
    }
}
```

**How it works:**
1. `session.query(Article).count()` → total
2. `filter(Article.is_fake == True).count()` → fake
3. `filter(Article.is_fake == False).count()` → real
4. Groups by distinct categories and counts each

---

### Endpoint: `GET /api/articles`

**Purpose:** Returns paginated, filterable articles with full intelligence data and transparency breakdowns.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number (≥ 1) |
| `limit` | int | 20 | Items per page (1–100) |
| `category` | str | None | Filter by category name |
| `is_fake` | bool | None | Filter by fake/real status |
| `search` | str | None | Title search (ILIKE match) |

**How it works (step by step):**

1. Opens database session
2. Loads the fake news detector ML model
3. Builds a SQLAlchemy query with optional filters (category, is_fake, title search)
4. Counts total results
5. Applies pagination: `.offset((page - 1) * limit).limit(limit)`
6. Orders by `published_at DESC` (newest first)

**For each article in the page:**

7. **Live credibility analysis:** Calls `detect_fake_news()` with:
   - Article title and content (truncated to 500 chars for speed)
   - Source name
   - Corroboration count (query: how many other articles in the same topic cluster are from trusted sources)
8. **External fact-check for uncertain articles:** If the article's base ML score is between 30-60%, calls `verify_article()` to query NewsAPI and Google Fact Check, then re-runs detection with the verification data
9. **AI explainability:** Calls `explain_prediction()` → gets trust terms and risk terms
10. **Headline analysis:** Calls `analyze_linguistic_style()` → sensationalism and objectivity scores
11. **Constructs response object** with all data including `score_details` containing:
    - `base_ml`, `headline_score`, `content_score`
    - `is_trusted`, `corroboration_count`, `corroboration_boost`
    - `bonus_applied`, `verification_boost`
    - `penalty_applied`, `penalty_reasons`
    - `fact_check` (if applicable)
    - `ai_logic` (trust/risk keywords, sensationalism, objectivity)

**Response:**
```json
{
    "items": [ ... ],
    "total": 450,
    "page": 1,
    "pages": 12
}
```

**Fallback behavior:** If the database values for `is_fake` or `credibility_score` are NULL (article not yet processed by pipeline), the live-calculated values are used instead.

---

## File: `dashboard/src/App.jsx`

### Component: `App`

**Purpose:** The root React component — manages state, fetches data, and renders the layout.

**State variables:**
| State | Type | Purpose |
|-------|------|---------|
| `articles` | Array | Current page of article objects |
| `stats` | Object | Aggregate statistics from `/api/stats` |
| `loading` | Boolean | Loading spinner control |
| `currentFilter` | String | Active filter: 'all', 'real', 'fake', or a category name |
| `search` | String | Search input text |
| `page` | Integer | Current pagination page |
| `totalPages` | Integer | Total number of pages |

---

### Function: `fetchStats()`

**What it does:** Calls `GET /api/stats` and updates the `stats` state. Runs once on component mount.

---

### Function: `fetchArticles(targetPage = 1)`

**What it does:**
1. Sets `loading = true`
2. Constructs URL: `http://localhost:8000/api/articles?page=N&limit=40`
3. Appends filter params:
   - `currentFilter === 'real'` → `is_fake=false`
   - `currentFilter === 'fake'` → `is_fake=true`
   - Other values → `category=<value>`
4. Appends search param if search text exists
5. Fetches the data
6. Updates `articles`, `page`, `totalPages`
7. Smooth-scrolls to top of page
8. Sets `loading = false`

**Trigger:** Runs whenever `currentFilter` or `search` changes, with a **300ms debounce** to avoid API flooding while typing.

---

### Function: `handlePageChange(newPage)`

**What it does:** Validates page bounds and calls `fetchArticles(newPage)`.

---

### Function: `renderPagination()`

**What it does:** Renders a pagination bar with:
- ← Previous / Next → buttons (disabled at boundaries)
- Window of 5 page numbers centered around current page
- "..." dots when pages are skipped
- First/last page buttons when appropriate

---

### Rendered Layout

```
┌──────────────────────────────────────────────────┐
│ ┌──────────┐  ┌──────────────────────────────┐  │
│ │ Sidebar  │  │  Stats Grid (3 cards)        │  │
│ │          │  │  - Total Articles Analyzed    │  │
│ │ Filters: │  │  - Verified Real News         │  │
│ │ • Global │  │  - Detected Fake News         │  │
│ │ • Real   │  ├──────────────────────────────┤  │
│ │ • Fake   │  │  Intelligence Feed    [🔍]    │  │
│ │          │  ├──────────────────────────────┤  │
│ │ Sectors: │  │  ┌──────────┐ ┌──────────┐  │  │
│ │ • World  │  │  │ Article  │ │ Article  │  │  │
│ │ • Sports │  │  │ Card     │ │ Card     │  │  │
│ │ • Biz    │  │  └──────────┘ └──────────┘  │  │
│ │ • Sci    │  │  ┌──────────┐ ┌──────────┐  │  │
│ │          │  │  │ Article  │ │ Article  │  │  │
│ └──────────┘  │  │ Card     │ │ Card     │  │  │
│               │  └──────────┘ └──────────┘  │  │
│               │  ← 1 2 [3] 4 5 ... 12 →    │  │
│               └──────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

---

## File: `dashboard/src/components/Sidebar.jsx`

### Component: `Sidebar`

**Props:**
| Prop | Type | Description |
|------|------|-------------|
| `stats` | Object | Contains `total_articles`, `real_articles`, `fake_articles`, `categories` |
| `currentFilter` | String | Currently active filter |
| `setCurrentFilter` | Function | Callback to change the active filter |

**Renders:**
1. **Logo:** Gradient icon "N" + brand name "DailyNewsAI"
2. **Main Intelligence section:** Three filter buttons:
   - Global Feed (shows total count)
   - Verified Authentic (shows real count)
   - Flagged Fake (shows fake count)
3. **Sector Analysis section:** Dynamically renders a button for each category returned by the API, showing the category name and article count

**Active state:** The currently selected filter button gets the `.active` CSS class (blue highlight).

---

## File: `dashboard/src/components/ArticleCard.jsx`

### Component: `ArticleCard`

**Props:**
| Prop | Type | Description |
|------|------|-------------|
| `article` | Object | Full article data including `score_details`, `keywords`, `full_content` |

**State:**
| State | Type | Purpose |
|-------|------|---------|
| `showDetails` | Boolean | Whether the score breakdown panel is visible |
| `showSummary` | Boolean | Whether the full content preview is visible |

**Interaction logic:**
- Clicking the credibility badge toggles the score details panel (and closes summary)
- Clicking "Preview" toggles the full content view (and closes details)
- Only one panel can be open at a time

---

### Rendered Card Structure

```
┌─────────────────────────────────────────┐
│  BBC | Sci/Tech          ✓ Verified 87%  │  ← Header (source, category, badge)
│                                          │
│  Article Headline Goes Here              │  ← Title
│  First 3 lines of cleaned content...     │  ← Summary (clamped)
│                                          │
│  ┌─ Score Breakdown (if expanded) ────┐  │
│  │ ML Content Analysis         72%    │  │
│  │ Sensationalism:             12%    │  │  ← Progress bar
│  │ Objectivity:                100%   │  │  ← Progress bar
│  │ External Verification       +15%   │  │
│  │ Authoritative Source        +15%   │  │
│  │ Quality Penalty             -67%   │  │
│  │   ↳ Article too short (32 words)   │  │  ← Penalty reasons
│  │   ↳ Lacks article structure        │  │
│  │ Cross-Reference Bonus       +10%   │  │
│  └────────────────────────────────────┘  │
│                                          │
│  [keyword1] [keyword2] [keyword3]        │  ← Keywords (when panels closed)
│                                          │
│  Apr 12, 2026 • Author    [Preview][Full]│  ← Footer
└─────────────────────────────────────────┘
```

**Conditional rendering:**
- **Score breakdown:** Only appears when the credibility badge is clicked
- **External Verification row:** Only appears when `verification_boost ≠ 0`
- **Authoritative Source row:** Only appears when `is_trusted = true`
- **Quality Penalty row:** Only appears when `penalty_applied < 0`
  - Penalty reasons sub-list appears when `penalty_reasons` array has items
- **Cross-Reference Bonus:** Only appears when corroboration count > 0
- **Fact Check row:** Only appears when professional fact-checkers have reviewed the claim
- **Keywords:** Only appear when score details and summary are both hidden

---

## File: `dashboard/src/index.css`

### Design System Overview

**Typography:**
- Body: `Inter` (Google Font) — weights 300–700
- Headings: `Outfit` (Google Font) — weights 400, 600, 800

**Color Palette:**
| Variable | Value | Usage |
|----------|-------|-------|
| `--bg-body` | `#f1f5f9` | Page background (light gray) |
| `--bg-sidebar` | `#ffffff` | Sidebar background |
| `--bg-card` | `#ffffff` | Card background |
| `--text-main` | `#1e293b` | Primary text (dark navy) |
| `--text-muted` | `#64748b` | Secondary text (gray) |
| `--accent-primary` | `#2563eb` | Blue accent (links, active states) |
| `--accent-secondary` | `#7c3aed` | Purple accent (logo gradient) |
| `--success` | `#059669` | Green (verified/real) |
| `--danger` | `#dc2626` | Red (fake/warning) |
| `--warning` | `#d97706` | Orange/amber |

**Key Design Features:**
- Cards with 20px border radius and subtle shadows
- Hover animations (`translateY(-6px)`)
- Gradient logo icon (blue → purple)
- Clean pagination with active state highlighting
- Scrollable content preview with custom scrollbar styling
- Responsive layout: sidebar collapses on screens < 1024px

---

## File: `run.bat`

### Startup Orchestrator

**Purpose:** Single-click startup script that launches all 5 system processes.

**What it does:**
1. Creates directories: `data/`, `models/`, `logs/` (if they don't exist)
2. Launches 5 separate terminal windows:

| Window Title | Command | Purpose |
|-------------|---------|---------|
| Data Ingestion | `python -m src.ingestion.fetcher` | One-time article fetch |
| Intelligence Pipeline | `python -m src.intelligence.pipeline` | Process unscored articles |
| API Backend | `uvicorn src.api.main:app --reload --port 8000` | REST API on port 8000 |
| Frontend Dashboard | `cd dashboard && npm run dev` | Vite dev server on port 5173 |
| Background Scheduler | `python -m src.ingestion.scheduler` | Recurring fetch every 15 min |

3. Prints startup confirmation with URLs
4. Pauses for user acknowledgment

---

## File: `src/maintenance/reprocess_fakes.py`

### Function: `reprocess_trusted_fakes()`

**Purpose:** Maintenance script to re-evaluate articles from trusted sources that are currently marked as fake.

**Use case:** After updating the scoring heuristics or retraining models, old articles may have stale scores. This script re-runs detection only on articles from trusted sources that are incorrectly flagged.

**What it does:**
1. Queries all articles where `is_fake == True`
2. For each, checks if the source matches any brand in `TRUSTED_SOURCES`
3. Re-runs `detect_fake_news()` with updated heuristics
4. If the status changed, prints the transition and updates the database

---

## Complete System Workflow

```
User double-clicks run.bat
         │
         ├──→ [Window 1] fetcher.py → NewsAPI + RSS → SQLite
         │
         ├──→ [Window 2] pipeline.py → Classification + Fake Detection + Topics + Keywords → SQLite
         │
         ├──→ [Window 3] uvicorn → FastAPI on :8000
         │                              │
         ├──→ [Window 4] npm run dev → Vite on :5173
         │                              │
         │                React App ←── │ ──→ GET /api/articles ──→ FastAPI ──→ SQLite
         │                    │                                        │
         │                    └── Render ArticleCards ←─── JSON Response
         │                         │
         │                         ├── Credibility badge with live scoring
         │                         ├── Score breakdown (ML + bonuses + penalties)
         │                         ├── AI reasoning (trust/risk keywords)
         │                         └── External fact-check data
         │
         └──→ [Window 5] scheduler.py → Re-runs fetcher every 15 minutes
```
