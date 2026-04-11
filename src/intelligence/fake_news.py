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

# Quality Heuristics Constants
MIN_NEWS_WORDS = 50  # Increased from 40
PROFANITY_LIST = [
    'fuck', 'shit', 'bitch', 'asshole', 'dick', 'pussy', 'bastard', 'cunt',
    'damn', 'hell', 'stupid', 'idiot', 'moron', 'retard', 'faggot', 'nigger',
    'slant', 'chink', 'spic'
]
GLUE_WORDS = [
    'the', 'and', 'with', 'from', 'this', 'that', 'which', 'their', 'they', 
    'have', 'been', 'for', 'was', 'were', 'not', 'but', 'are', 'has', 'had',
    'in', 'is', 'it', 'to', 'of', 'at', 'on', 'by', 'an', 'as', 'be'
]


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
        "The United Nations released its annual report on climate change, citing increased global temperatures based on satellite data from multiple agencies."
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


def _get_indian_fake_news_augmentation():
    """
    Returns additional Indian political, WhatsApp forwards, and communal fake news
    labeled as FAKE (1). Balances the model against the curated real news to 
    prevent the model from treating all Indian political names as "Real".
    """
    indian_fake_samples = [
        "UNESCO has declared the Indian National Anthem as the best in the world following an international vote at the UN headquarters.",
        "The new ₹2000 notes issued by RBI contain a nano-GPS chip that can be tracked by satellites even 120 meters underground, allowing the government to recover black money.",
        "BREAKING: Secret documents leaked online reveal opposition party leaders met with foreign spies to manipulate EVM polling machines on election day.",
        "UNESCO declares Prime Minister Narendra Modi the best Prime Minister in the world.",
        "Forward this message to 10 groups, and WhatsApp will change its logo color to blue. Mukesh Ambani has promised 50GB free Jio data if you do it within 24 hours.",
        "SHOCKING: Police expose underground plot by minority communities to poison the water supply of major cities ahead of the upcoming legislative assembly elections.",
        "A rare venomous spider from South America has arrived in India via banana shipments. If it bites you, death is certain within 5 minutes. Forward to warn your family!",
        "Election Commission to cancel votes of those who do not link their Aadhaar card to their Voter ID by tomorrow evening. Strict orders from the Supreme Court.",
        "Famous Bollywood superstar caught on camera insulting the Indian army and demanding the division of the country. Viral video proves sedition!",
        "Drink hot water with crushed garlic and lemon three times a day to cure the coronavirus instantly. This secret remedy is being hidden by big pharma companies.",
        "Major Indian political leader arrested in secret overnight raid for embezzling billions into Swiss bank accounts. Mainstream media is totally silent!",
        "WARNING: Do not drink any cold drinks from local brands for the next few months. A worker at the factory deliberately injected HIV infected blood into the bottling line.",
        "Muslim population to overtake Hindu population in India within the next 10 years, according to a secret UN demographic intelligence report.",
        "CCTV footage clearly shows members of the ruling BJP distributing alcohol and cash outside polling booths to buy votes in broad daylight.",
        "Congress party signs secret MOU with China to hand over border territories in exchange for massive election funding, top intelligence sources claim.",
        "NASA satellite images taken during Diwali show India completely illuminated from space, proving the massive scale of the ancient Hindu festival.",
        "Eating onions and placing them in your socks while sleeping absorbs all the toxins from your body and cures all fevers. Proven Ayurvedic miracle!",
        "Government announces complete nationwide lockdown starting midnight tonight to deploy military forces against violent protests. Stock up on rations!",
        "The Supreme Court of India has ordered that starting next month, all citizens must declare their religion on their official social media profiles.",
        "Video shows a massive ghost floating across the highway near the haunted village in Rajasthan! Unbelievable paranormal evidence caught on tape.",
        "If you receive a phone call from the number starting with 777, DO NOT answer. It is ISIS hackers who will immediately steal all money from your bank account through the call.",
        "A young girl in a village gave birth to a snake after committing a sin against the temple deity. Thousands are gathering to witness the curse.",
        "The historical Taj Mahal was actually an ancient Hindu temple called Tejo Mahalaya that was forcefully taken over and converted.",
        "Amit Shah secretly admitted during a closed-door meeting that the party knows it will lose the upcoming elections in the southern states.",
        "Ratan Tata announces he will give his entire wealth to Pakistan if India loses the upcoming cricket world cup match.",
        "An enormous 50-foot snake was found by construction workers digging the new metro line in Bangalore. Pictures inside!",
        "Government has started recording all your phone calls and monitoring your WhatsApp messages under the new IT regulations. Beware of what you post!",
        "A highly contagious new virus called 'Nipah-X' that turns people into flesh-eating zombies has been discovered in a remote Indian village.",
        "Opposition leaders caught offering millions of dollars to global news outlets (BBC, NYT) to publish fake stories ruining India's international image.",
        "Scientists confirm the Earth will experience three days of total darkness starting next Monday due to a rare solar alignment not seen in 10,000 years."
    ]
    
    # We multiply these heavily (e.g. 50 times) to ensure they have enough weight 
    # to combat the 40,000 item US-election biased base dataset.
    all_fakes = indian_fake_samples * 60
    labels = [1] * len(all_fakes)
    
    print(f"  Indian fake news augmentation: {len(all_fakes)} fake samples added.")
    return all_fakes, labels


def _get_demo_dataset():
    """
    Returns a small synthetic dataset for development/testing purposes.
    """
    real_samples = [
        "The Federal Reserve announced a quarter-point interest rate increase today, citing continued economic growth and stable employment figures across major sectors.",
        "Scientists at MIT have developed a new battery technology that could extend electric vehicle range by 40 percent, according to a peer-reviewed study published in Nature.",
        "The World Health Organization reported a 15 percent decline in global malaria cases over the past five years, attributing the decrease to improved prevention measures.",
        "SpaceX successfully launched its latest Falcon 9 rocket carrying 60 Starlink satellites into orbit from Cape Canaveral on Friday morning.",
        "The European Union passed comprehensive data privacy regulations that will affect how technology companies collect and process user information.",
    ] * 40

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
    """
    texts, labels = download_fake_news_dataset()
    if texts is None:
        return None

    if len(texts) > max_samples:
        indices = np.random.RandomState(42).choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    print("\nAugmenting training data with Indian/international news...")
    # Add real Indian news
    aug_texts, aug_labels = _get_indian_news_augmentation()
    if aug_texts:
        texts.extend(aug_texts)
        labels.extend(aug_labels)
        
    # Add fake Indian news to balance!
    fake_aug_texts, fake_aug_labels = _get_indian_fake_news_augmentation()
    if fake_aug_texts:
        texts.extend(fake_aug_texts)
        labels.extend(fake_aug_labels)
        
    print(f"Total training samples after augmentation: {len(texts)}")

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

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest model: {best_name} (Accuracy: {best_accuracy:.4f})")
    print(f"Model saved to: {MODEL_PATH}")
    return best_model


def load_fake_news_detector():
    """
    Loads the trained fake news detector from disk.
    """
    if not os.path.exists(MODEL_PATH):
        print("No saved fake news model found. Please train it first.")
        print("Run: python -m src.intelligence.fake_news")
        return None

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Fake news detector loaded successfully.")
    return model


def _check_text_quality(text: str) -> dict:
    """
    Checks for profanity, nonsense, and quality metrics.
    """
    words = [w.lower().strip('.,!?;:"()') for w in text.split()]
    if not words:
        return {"is_low_quality": True, "reason": "empty"}

    found_profanity = [w for w in words if any(p in w for p in PROFANITY_LIST)]
    has_profanity = len(found_profanity) > 0

    unique_ratio = len(set(words)) / len(words)
    is_repetitive = unique_ratio < 0.3 and len(words) > 20
    
    glue_count = sum(1 for w in words if w in GLUE_WORDS)
    glue_density = glue_count / len(words)
    lacks_structure = glue_density < 0.15 and len(words) > 10 
    
    original_words = text.split()
    cap_count = sum(1 for w in original_words if len(w) > 0 and w[0].isupper())
    cap_ratio = cap_count / (len(original_words) or 1)
    bad_capitalization = cap_ratio < 0.05 and len(words) > 10

    avg_len = sum(len(w) for w in words) / (len(words) or 1)
    is_random = (avg_len < 3.0 or avg_len > 12.0) and len(words) > 10

    is_low_quality = has_profanity or is_repetitive or lacks_structure or is_random or bad_capitalization

    return {
        "is_low_quality": is_low_quality,
        "has_profanity": has_profanity,
        "is_repetitive": is_repetitive,
        "lacks_structure": lacks_structure,
        "is_random": is_random,
        "bad_capitalization": bad_capitalization,
        "word_count": len(words)
    }


def _apply_scoring_heuristics(title, content, content_score, source=None, corroboration_count=0, verification_result=None):
    """
    Shared logic for applying heuristical bonuses and penalties.
    Returns: (is_fake, final_score, breakdown_dict)
    """
    # 1. Headline Score
    style = analyze_linguistic_style(title)
    headline_score = ((100 - style["sensationalism_score"]) + style["objectivity_score"]) / 200.0
    
    # 2. Combined ML Base Score
    base_combined = (0.3 * headline_score) + (0.7 * content_score)
    final_score = base_combined
    
    quality = _check_text_quality(content)

    breakdown = {
        "headline_score": round(headline_score, 4),
        "content_score": round(content_score, 4),
        "base_combined": round(base_combined, 4),
        "source_boost": 0.0,
        "corroboration_boost": 0.0,
        "verification_boost": 0.0,
        "penalty": 0.0,
        "fact_check": None
    }
    
    # 3. Trusted Source Detection (Robust Brand Matching)
    is_trusted = False
    if source:
        source_clean = source.lower()
        for ts in TRUSTED_SOURCES:
            if ts.lower() in source_clean:
                is_trusted = True
                break
    
    # 4. Heuristic Bonuses (Conditional)
    if is_trusted and base_combined > 0.3:
        breakdown["source_boost"] = 0.15
        final_score += 0.15
        
    if corroboration_count >= 1:
        breakdown["corroboration_boost"] = 0.10
        final_score += 0.10
        
    if not is_trusted and corroboration_count == 0:
        breakdown["penalty"] -= 0.15
        final_score -= 0.15

    # 5. Quality Penalties
    if quality["is_low_quality"]:
        # Minor flags like 'repetitive' or 'lacks_structure' are less severe for trusted sources
        is_severe = quality["has_profanity"] or quality["is_random"]
        
        penalty_amount = 0.45
        if quality["has_profanity"]: penalty_amount = 0.7
        if quality["is_random"] or quality["bad_capitalization"]: penalty_amount = 0.55
        
        # Halve the penalty for trusted sources unless it's severe (profanity/nonsense)
        if is_trusted and not is_severe:
            penalty_amount *= 0.5
            
        breakdown["penalty"] -= penalty_amount
        final_score -= penalty_amount

    # 4. Brevity Penalty (Word Count)
    if quality["word_count"] < MIN_NEWS_WORDS:
        diff = MIN_NEWS_WORDS - quality["word_count"]
        # Reduce penalty step-size for trusted sources (from 0.015 to 0.005)
        step = 0.005 if is_trusted else 0.015
        word_penalty = diff * step
        applied_word_penalty = min(0.4 if not is_trusted else 0.2, word_penalty)
        
        breakdown["penalty"] -= applied_word_penalty
        final_score -= applied_word_penalty

    # 7. External Verification Boost/Penalty (NewsAPI + Google Fact Check)
    if verification_result and isinstance(verification_result, dict):
        v_score = verification_result.get("verification_score", 0.5)
        breakdown["fact_check"] = verification_result
        
        if v_score >= 0.7:
            # Well-corroborated by external sources → bonus
            v_boost = 0.15
            breakdown["verification_boost"] = v_boost
            final_score += v_boost
        elif v_score <= 0.3:
            # Flagged or uncorroborated → penalty
            v_penalty = 0.15
            breakdown["verification_boost"] = -v_penalty
            final_score -= v_penalty
        # 0.3 < v_score < 0.7 → neutral, no adjustment

    final_score = max(0.01, min(1.0, final_score))
    
    # 6. Apply "Trust Floor"
    # Even if an article is short or has minor formatting issues, 
    # if it's from a trusted brand and passes basic safety AND ML wasn't completely terrible, it stays REAL.
    if is_trusted and base_combined > 0.3:
        safety_pass = not quality["has_profanity"] and not quality["is_random"]
        if safety_pass:
            if final_score < TRUSTED_SOURCE_FLOOR:
                diff_to_floor = TRUSTED_SOURCE_FLOOR - final_score
                breakdown["source_boost"] += diff_to_floor
                final_score = TRUSTED_SOURCE_FLOOR

    # Clean up breakdown payload
    breakdown["source_boost"] = round(breakdown["source_boost"], 4)
    breakdown["corroboration_boost"] = round(breakdown["corroboration_boost"], 4)
    breakdown["verification_boost"] = round(breakdown["verification_boost"], 4)
    breakdown["penalty"] = round(breakdown["penalty"], 4)

    is_fake = bool(final_score < FAKE_THRESHOLD)
    return is_fake, final_score, breakdown


def detect_fake_news(title: str, content: str, model=None, source: str = None, corroboration_count: int = 0, verification_result: dict = None) -> tuple:
    """
    Checks if a single article is fake news with dual-scoring heuristics.
    Returns: (is_fake, final_score, breakdown_dict)
    """
    if model is None:
        model = load_fake_news_detector()
        if model is None:
            return False, 0.5, {}

    title = title or ""
    content = content or ""
    
    # If the scraper failed to get body content, use the title as the content
    # so it still goes through the ML model and heuristic pipeline!
    if not content or len(content.strip()) < 10:
        content = title

    probabilities = model.predict_proba([content])[0]
    content_score = float(probabilities[0])
    
    return _apply_scoring_heuristics(title, content, content_score, source, corroboration_count, verification_result)


def detect_batch(titles: list, contents: list, model=None, sources: list = None, corroboration_counts: list = None) -> list:
    """
    Runs fake news detection on a batch of articles with heuristics.
    Returns list of tuples: [(is_fake, final_score, breakdown_dict), ...]
    """
    if model is None:
        model = load_fake_news_detector()
        if model is None:
            return [(False, 0.5, {})] * len(texts)

    results = []
    valid_indices = []
    valid_texts = []

    for i, content in enumerate(contents):
        title = titles[i] if titles and i < len(titles) else ""
        content = content or ""
        
        # Fallback to title if content is missing
        if len(content.strip()) < 10:
            content = title
            
        if len(content.strip()) >= 10:
            valid_texts.append(content)
            valid_indices.append(i)

    if valid_texts:
        probabilities = model.predict_proba(valid_texts)

        result_map = {}
        for j, idx in enumerate(valid_indices):
            content_score = float(probabilities[j][0])
            source = sources[idx] if sources and idx < len(sources) else None
            corr_count = corroboration_counts[idx] if corroboration_counts and idx < len(corroboration_counts) else 0
            
            title = titles[idx] if titles and idx < len(titles) else ""
            is_fake, final_score, breakdown = _apply_scoring_heuristics(title, valid_texts[j], content_score, source, corr_count)
            result_map[idx] = (is_fake, final_score, breakdown)

        for i in range(len(contents)):
            if i in result_map:
                results.append(result_map[i])
            else:
                results.append((True, 0.1, {"penalty": -0.4, "headline_score": 0.5, "content_score": 0.1, "base_combined": 0.1}))
    else:
        results = [(True, 0.1, {"penalty": -0.4, "headline_score": 0.5, "content_score": 0.1, "base_combined": 0.1})] * len(contents)

    return results


def explain_prediction(text: str, model=None, top_n: int = 4) -> dict:
    """
    Analyzes which words most influenced the AI's decision.
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
        weights = X_vec.toarray()[0] * clf.coef_[0]
        non_zero_indices = np.where(abs(weights) > 1e-5)[0]
        if len(non_zero_indices) == 0:
            return {"trust_terms": [], "risk_terms": []}
        top_indices = non_zero_indices[np.argsort(abs(weights[non_zero_indices]))][-top_n*2:]
        total_impact = np.sum(abs(weights[top_indices]))
        trust_terms = []
        risk_terms = []
        for i in top_indices:
            impact_pct = int(round((abs(weights[i]) / total_impact) * 100))
            if weights[i] < 0: # Trust
                trust_terms.append({"word": feature_names[i], "impact": impact_pct})
            else: # Risk
                risk_terms.append({"word": feature_names[i], "impact": impact_pct})
        trust_terms.sort(key=lambda x: x['impact'], reverse=True)
        risk_terms.sort(key=lambda x: x['impact'], reverse=True)
        return {"trust_terms": trust_terms[:top_n], "risk_terms": risk_terms[:top_n]}
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

    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    caps_ratio = len(caps_words) / len(words)
    caps_score = min(100, caps_ratio * 400) 
    excl_count = text.count('!')
    ques_count = text.count('?')
    punc_density = (excl_count + ques_count) / (len(text) / 100)
    punc_score = min(100, punc_density * 20) 
    CLICKBAIT_TERMS = ['shocking', 'exposed', 'unbelievable', 'reveal', 'secret', 'won\'t believe', 'trick']
    cb_matches = sum(1 for term in CLICKBAIT_TERMS if term in text.lower())
    cb_score = min(100, cb_matches * 25)
    s_index = int((caps_score * 0.4) + (punc_score * 0.3) + (cb_score * 0.3))
    SUBJECTIVE_MARKERS = ['amazing', 'terrible', 'worst', 'incredible', 'best', 'clearly', 'obviously', 'actually']
    sub_matches = sum(1 for term in SUBJECTIVE_MARKERS if term in text.lower())
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
            ("UN Report", "The United Nations released its annual report on climate change."),
            ("SHOCKING REVELATION!!!", "Government secretly implanting microchips!!!"),
            ("Stanford Med Research", "Researchers at Stanford University published findings on a new cancer treatment.")
        ]
        for title, content in test_samples:
            is_fake, score, _ = detect_fake_news(title, content, model=model)
            status = "FAKE" if is_fake else "REAL"
            print(f"  [{status} | Credibility: {score:.2f}] {title} - {content[:45]}...")
