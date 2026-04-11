"""
Fake News Detection Module
Binary classifier to label articles as 'Authentic' or 'Potentially Misleading'.
Outputs both a boolean label and a credibility confidence score (0.0 – 1.0).
Uses TF-IDF vectorization + Logistic Regression trained on the ISOT / Kaggle
Fake News dataset.
(Proposal Section 5.4)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


# Model save paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_detector.pkl')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Label mapping
LABEL_MAP = {
    0: False,   # Real / Authentic
    1: True     # Fake / Potentially Misleading
}

TRUSTED_SOURCES = [
    'BBC News', 'BBC', 'Reuters', 'TechCrunch', 
    'The Hindu', 'NDTV News', 'NDTV', 'Times of India', 'Indian Express',
    'Hindustan Times', 'The Wire', 'Scroll.in', 'Mint', 'Economic Times',
    'Associated Press', 'AP News', 'The Guardian', 'Al Jazeera'
]

# Threshold below which an article is considered fake
FAKE_THRESHOLD = 0.40

# Minimum credibility floor for trusted sources (they can never be flagged fake)
TRUSTED_SOURCE_FLOOR = 0.55


def download_fake_news_dataset():
    """
    Downloads a fake news dataset for training.
    Tries the HuggingFace 'GonzaloA/fake_news' dataset first, 
    then falls back to local CSV files if available.
    
    Returns:
        Tuple of (texts, labels) or (None, None) on failure.
    """
    # Option 1: Try HuggingFace datasets library
    try:
        from datasets import load_dataset
        print("Downloading fake news dataset from HuggingFace...")
        dataset = load_dataset("GonzaloA/fake_news", split="train")
        
        texts = []
        labels = []
        for item in dataset:
            text = item.get('text', '') or ''
            label = item.get('label', 0)
            if len(text.strip()) > 50:
                texts.append(text)
                labels.append(label)
        
        print(f"Loaded {len(texts)} samples from HuggingFace fake news dataset.")
        if len(texts) > 100:
            return texts, labels
            
    except Exception as e:
        print(f"HuggingFace download failed: {e}")

    # Option 2: Check for local CSV files (Kaggle-style True.csv / Fake.csv)
    true_path = os.path.join(DATA_DIR, 'True.csv')
    fake_path = os.path.join(DATA_DIR, 'Fake.csv')

    if os.path.exists(true_path) and os.path.exists(fake_path):
        print("Loading local True.csv and Fake.csv files...")
        try:
            true_df = pd.read_csv(true_path)
            fake_df = pd.read_csv(fake_path)

            true_df['label'] = 0  # Authentic
            fake_df['label'] = 1  # Fake

            # Combine title and text for richer features
            true_df['full_text'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
            fake_df['full_text'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')

            combined = pd.concat([true_df, fake_df], ignore_index=True)
            combined = combined[combined['full_text'].str.len() > 50]

            texts = combined['full_text'].tolist()
            labels = combined['label'].tolist()
            print(f"Loaded {len(texts)} samples from local CSV files.")
            return texts, labels
        except Exception as e:
            print(f"Error reading local CSV files: {e}")

    # Option 3: Generate a small synthetic dataset for development
    print("\nWARNING: No fake news dataset found.")
    print("Please either:")
    print("  1. Install 'datasets' library: pip install datasets")
    print("  2. Download True.csv and Fake.csv from Kaggle and place in data/")
    print("\nUsing a small built-in demo dataset for now...\n")
    
    return _get_demo_dataset()


def _get_indian_news_augmentation():
    """
    Returns additional Indian/international political news samples labeled as REAL (0).
    These augment the training data so the model doesn't misclassify Indian political
    news vocabulary as fake. Also pulls verified articles from the local database.
    """
    # Hand-curated real Indian political news samples
    indian_real_samples = [
        "Prime Minister Narendra Modi inaugurated the new parliament building in New Delhi, marking a historic moment for Indian democracy.",
        "Chief Minister Chandrababu Naidu announced a new industrial corridor project in Andhra Pradesh to boost economic development in the region.",
        "Mamata Banerjee held a rally in Kolkata demanding action against rising fuel prices and calling for national-level policy reforms.",
        "External Affairs Minister S. Jaishankar met with his counterparts from Bangladesh and Pakistan to discuss bilateral trade agreements.",
        "The Indian Supreme Court delivered a landmark verdict on the right to privacy, ruling it a fundamental right under the Constitution.",
        "Rahul Gandhi addressed the Lok Sabha on the issue of unemployment among youth and proposed a national employment guarantee scheme.",
        "Foreign Secretary Vikram Misri held discussions with US Secretary of State on strengthening defence and trade cooperation between India and the United States.",
        "Nitish Kumar was sworn in as a member of the Rajya Sabha in a ceremony attended by senior leaders from the ruling coalition.",
        "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent, citing stable inflation and strong GDP growth projections.",
        "India and Bangladesh signed a new water-sharing agreement for the Teesta river following high-level diplomatic negotiations.",
        "Home Minister Amit Shah reviewed security arrangements in Jammu and Kashmir ahead of the upcoming assembly elections.",
        "Andhra Pradesh Chief Minister launched a fleet of new fire service vehicles and emergency response equipment in Amaravati.",
        "The Election Commission of India announced the schedule for state assembly elections in five states across northern and eastern India.",
        "Defence Minister Rajnath Singh commissioned the INS Vikrant aircraft carrier at Cochin Shipyard in a major milestone for Indian naval capability.",
        "Pakistan's Foreign Minister held talks with his Indian counterpart on the sidelines of the United Nations General Assembly.",
        "The Indian government announced new tariffs on imports from China and the European Union as part of its trade rebalancing strategy.",
        "South Korea deployed advanced thermal imaging cameras to track escaped animals from the Seoul metropolitan zoo after a containment breach.",
        "Ireland's coalition government reached a deal on fuel subsidies for rural households following weeks of pressure from farming communities.",
        "US-Iran nuclear negotiations entered a critical phase as diplomats discussed sanctions relief and enrichment limits in Geneva.",
        "The United Kingdom announced a post-Brexit trade agreement with Australia covering agricultural products and digital services.",
        "Sri Lanka's central bank raised interest rates to combat inflation as the island nation works to stabilize its economy after the debt crisis.",
        "Nepal and China agreed to extend the railway line from Lhasa to Kathmandu as part of the Belt and Road Initiative infrastructure plan.",
        "The BRICS summit in Johannesburg discussed expansion of membership and creation of a common trade settlement currency.",
        "Indian Space Research Organisation launched the Chandrayaan mission from Sriharikota, marking India's next step in lunar exploration.",
        "The World Health Organization praised India's vaccination campaign for achieving high coverage rates across rural and urban districts.",
    ]

    # Try to pull verified real articles from local DB to augment training
    db_samples = []
    try:
        from src.ingestion.database import get_session, Article
        session = get_session()
        # Get articles from trusted sources that were previously marked real
        real_articles = session.query(Article).filter(
            Article.is_fake == False
        ).limit(500).all()
        for a in real_articles:
            text = (a.title or '') + ' ' + (a.raw_content or a.clean_content or '')
            if len(text.strip()) > 50:
                db_samples.append(text[:1000])  # Cap length
        session.close()
        print(f"  Augmented with {len(db_samples)} verified real articles from database.")
    except Exception as e:
        print(f"  Could not augment from DB: {e}")

    all_real = indian_real_samples * 8 + db_samples  # Repeat curated samples for balance
    labels = [0] * len(all_real)  # All labeled as REAL
    
    print(f"  Indian/international augmentation: {len(all_real)} real samples added.")
    return all_real, labels


def _get_demo_dataset():
    """
    Returns a small synthetic dataset for development/testing purposes.
    NOT suitable for production use — just ensures the pipeline doesn't crash.
    """
    real_samples = [
        "The Federal Reserve announced a quarter-point interest rate increase today, citing continued economic growth and stable employment figures across major sectors.",
        "Scientists at MIT have developed a new battery technology that could extend electric vehicle range by 40 percent, according to a peer-reviewed study published in Nature.",
        "The World Health Organization reported a 15 percent decline in global malaria cases over the past five years, attributing the decrease to improved prevention measures.",
        "SpaceX successfully launched its latest Falcon 9 rocket carrying 60 Starlink satellites into orbit from Cape Canaveral on Friday morning.",
        "The European Union passed comprehensive data privacy regulations that will affect how technology companies collect and process user information.",
    ] * 40  # Repeat to get enough samples

    fake_samples = [
        "BREAKING: Secret government documents reveal that the moon landing was staged in a Hollywood studio with actors and special effects!!!",
        "EXPOSED: Doctors DON'T want you to know this ONE WEIRD TRICK that cures all diseases overnight! Big pharma is TERRIFIED!",
        "SHOCKING: Celebrities caught in underground conspiracy to control world governments through mind control technology!",
        "URGENT: Scientists CONFIRM that drinking bleach can cure all viruses - mainstream media is HIDING this from you!",
        "BREAKING: Aliens have been living among us for decades according to leaked classified documents from Area 51!",
    ] * 40

    texts = real_samples + fake_samples
    labels = [0] * len(real_samples) + [1] * len(fake_samples)
    
    print(f"Demo dataset: {len(texts)} samples (for development only)")
    return texts, labels


def train_fake_news_detector(max_samples: int = 20000):
    """
    Trains a binary fake news classifier.
    Evaluates Logistic Regression and Naive Bayes, picks the best.
    Augments training data with Indian/international news samples so the
    model doesn't misclassify regional political vocabulary as fake.
    
    Args:
        max_samples: Max number of samples to use for training.
        
    Returns:
        The trained sklearn Pipeline, or None on failure.
    """
    texts, labels = download_fake_news_dataset()
    if texts is None:
        return None

    # Subsample if needed
    if len(texts) > max_samples:
        indices = np.random.RandomState(42).choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    # ─── Augment with Indian/international news samples ───
    print("\nAugmenting training data with Indian/international news...")
    aug_texts, aug_labels = _get_indian_news_augmentation()
    if aug_texts:
        texts.extend(aug_texts)
        labels.extend(aug_labels)
        print(f"Total training samples after augmentation: {len(texts)}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "Naive Bayes": MultinomialNB(alpha=0.1),
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, clf in classifiers.items():
        print(f"\n--- Training {name} for Fake News Detection ---")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                stop_words='english',
                sublinear_tf=True
            )),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n{name} — Accuracy: {accuracy:.4f}")
        print(classification_report(
            y_test, predictions,
            target_names=["Authentic", "Potentially Misleading"]
        ))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_name = name

    # Save the best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest model: {best_name} (Accuracy: {best_accuracy:.4f})")
    print(f"Model saved to: {MODEL_PATH}")
    return best_model


def load_fake_news_detector():
    """
    Loads the trained fake news detector from disk.
    
    Returns:
        The trained pipeline, or None if not found.
    """
    if not os.path.exists(MODEL_PATH):
        print("No saved fake news model found. Please train it first.")
        print("Run: python -m src.intelligence.fake_news")
        return None

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Fake news detector loaded successfully.")
    return model


def detect_fake_news(text: str, model=None, source: str = None, corroboration_count: int = 0) -> tuple:
    """
    Checks if a single article is fake news.
    
    Args:
        text: The article text.
        model: The trained pipeline.
        source: The news source name.
        corroboration_count: Number of other trusted articles covering this topic.
        
    Returns:
        Tuple of (is_fake: bool, credibility_score: float)
        credibility_score is between 0.0 (definitely fake) and 1.0 (definitely real).
    """
    if model is None:
        model = load_fake_news_detector()
        if model is None:
            return False, 0.5  # Default: uncertain

    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return False, 0.5

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    # credibility_score = probability of being authentic (class 0)
    base_score = float(probabilities[0])
    
    # ─── Heuristics Application ───
    final_score = base_score
    
    # 1. Source Reputation (boosted from +0.15 to +0.25)
    is_trusted = False
    if source:
        for ts in TRUSTED_SOURCES:
            if ts.lower() in source.lower():
                is_trusted = True
                break

    if is_trusted:
        final_score += 0.25
        
    # 2. Corroboration Count
    if corroboration_count >= 1:
        final_score += 0.10
        
    # 3. Penalty for unverified lone claims
    if not is_trusted and corroboration_count == 0:
        final_score -= 0.15

    # Clamp the final score
    final_score = max(0.0, min(1.0, final_score))
    
    # 4. Trusted source floor — trusted outlets never get flagged as fake
    if is_trusted:
        final_score = max(final_score, TRUSTED_SOURCE_FLOOR)

    is_fake = bool(final_score < FAKE_THRESHOLD)

    return is_fake, final_score


def detect_batch(texts: list, model=None, sources: list = None, corroboration_counts: list = None) -> list:
    """
    Runs fake news detection on a batch of articles with heuristics.
    
    Args:
        texts: List of article texts.
        model: The trained pipeline.
        sources: List of news sources (strings).
        corroboration_counts: List of integers denoting corroboration.
        
    Returns:
        List of tuples (is_fake: bool, credibility_score: float)
    """
    if model is None:
        model = load_fake_news_detector()
        if model is None:
            return [(False, 0.5)] * len(texts)

    results = []
    valid_indices = []
    valid_texts = []

    for i, text in enumerate(texts):
        if text and isinstance(text, str) and len(text.strip()) >= 10:
            valid_texts.append(text)
            valid_indices.append(i)

    if valid_texts:
        predictions = model.predict(valid_texts)
        probabilities = model.predict_proba(valid_texts)

        result_map = {}
        for j, idx in enumerate(valid_indices):
            base_score = float(probabilities[j][0])
            
            source = sources[idx] if sources and idx < len(sources) else None
            corr_count = corroboration_counts[idx] if corroboration_counts and idx < len(corroboration_counts) else 0

            final_score = base_score
            is_trusted = False
            if source:
                for ts in TRUSTED_SOURCES:
                    if ts.lower() in source.lower():
                        is_trusted = True
                        break

            if is_trusted:
                final_score += 0.25
                
            if corr_count >= 1:
                final_score += 0.10
                
            if not is_trusted and corr_count == 0:
                final_score -= 0.15

            final_score = max(0.0, min(1.0, final_score))
            
            # Trusted source floor
            if is_trusted:
                final_score = max(final_score, TRUSTED_SOURCE_FLOOR)

            is_fake = bool(final_score < FAKE_THRESHOLD)

            result_map[idx] = (is_fake, final_score)

        for i in range(len(texts)):
            if i in result_map:
                results.append(result_map[i])
            else:
                results.append((False, 0.5))
    else:
        results = [(False, 0.5)] * len(texts)

    return results


def explain_prediction(text: str, model=None, top_n: int = 4) -> dict:
    """
    Analyzes which words most influenced the AI's decision, 
    returning their percentage contribution to the top feature set.
    """
    if model is None:
        model = load_fake_news_detector()
        if model is None:
            return {"trust_terms": [], "risk_terms": []}

    try:
        tfidf = model.named_steps['tfidf']
        clf = model.named_steps['classifier']
        X_vec = tfidf.transform([text])
        feature_names = tfidf.get_feature_names_out()
        
        # Calculate impacts (Coefficient * TF-IDF weight)
        weights = X_vec.toarray()[0] * clf.coef_[0]
        
        # Find indices of non-zero weights
        non_zero_indices = np.where(abs(weights) > 1e-5)[0]
        if len(non_zero_indices) == 0:
            return {"trust_terms": [], "risk_terms": []}

        # Filter to top N terms total
        top_indices = non_zero_indices[np.argsort(abs(weights[non_zero_indices]))][-top_n*2:]
        
        # Normalize weights of top items to sum to 100% impact
        total_impact = np.sum(abs(weights[top_indices]))
        
        trust_terms = []
        risk_terms = []
        
        for i in top_indices:
            impact_pct = int(round((abs(weights[i]) / total_impact) * 100))
            if weights[i] < 0: # Trust
                trust_terms.append({"word": feature_names[i], "impact": impact_pct})
            else: # Risk
                risk_terms.append({"word": feature_names[i], "impact": impact_pct})
                
        # Sort by impact
        trust_terms.sort(key=lambda x: x['impact'], reverse=True)
        risk_terms.sort(key=lambda x: x['impact'], reverse=True)
                
        return {
            "trust_terms": trust_terms[:top_n],
            "risk_terms": risk_terms[:top_n]
        }
    except Exception as e:
        print(f"Error explaining prediction: {e}")
        return {"trust_terms": [], "risk_terms": []}


def analyze_linguistic_style(text: str) -> dict:
    """
    Returns mathematical scores (0-100) for Sensationalism and Objectivity.
    """
    if not text:
        return {"sensationalism_score": 0, "objectivity_score": 100}

    words = text.split()
    if not words:
        return {"sensationalism_score": 0, "objectivity_score": 100}

    # 1. Sensationalism Index Component: ALL CAPS
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    caps_ratio = len(caps_words) / len(words)
    caps_score = min(100, caps_ratio * 400) # 25% caps = 100 pts
    
    # 2. Sensationalism Index Component: Punctuation Density (!!!, ???)
    excl_count = text.count('!')
    ques_count = text.count('?')
    punc_density = (excl_count + ques_count) / (len(text) / 100) # per 100 chars
    punc_score = min(100, punc_density * 20) # 5 marks/100 chars = 100 pts
    
    # 3. Sensationalism Index Component: Clickbait Lexicon
    CLICKBAIT_TERMS = ['shocking', 'exposed', 'unbelievable', 'reveal', 'secret', 'won\'t believe', 'trick']
    cb_matches = sum(1 for term in CLICKBAIT_TERMS if term in text.lower())
    cb_score = min(100, cb_matches * 25)

    # Final Sensationalism Index (Weighted Average)
    s_index = int((caps_score * 0.4) + (punc_score * 0.3) + (cb_score * 0.3))
    
    # 4. Objectivity Score (Lexicon of Subjectivity)
    # Simple heuristic: excessive adjectives/adverbs often reduce objectivity
    SUBJECTIVE_MARKERS = ['amazing', 'terrible', 'worst', 'incredible', 'best', 'clearly', 'obviously', 'actually']
    sub_matches = sum(1 for term in SUBJECTIVE_MARKERS if term in text.lower())
    # Objectivity starts at 100 and drops per subjective marker found
    objectivity = max(30, 100 - (sub_matches * 15))

    return {
        "sensationalism_score": min(100, s_index),
        "objectivity_score": int(objectivity),
        "caps_ratio": round(caps_ratio, 2),
        "punc_count": excl_count + ques_count
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  FAKE NEWS DETECTOR — Training")
    print("=" * 60)
    model = train_fake_news_detector(max_samples=20000)

    if model:
        print("\n--- Sanity Check ---")
        test_samples = [
            "The United Nations released its annual report on climate change, citing increased global temperatures based on satellite data from multiple agencies.",
            "SHOCKING REVELATION: Government secretly implanting microchips in vaccines to track and control the population!!!",
            "Researchers at Stanford University published findings on a new cancer treatment in the Journal of Medicine.",
        ]
        for text in test_samples:
            is_fake, score = detect_fake_news(text, model)
            status = "FAKE" if is_fake else "REAL"
            print(f"  [{status} | Credibility: {score:.2f}] {text[:65]}...")
