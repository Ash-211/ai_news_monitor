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


def detect_fake_news(text: str, model=None) -> tuple:
    """
    Checks if a single article is fake news.
    
    Args:
        text: The article text.
        model: The trained pipeline.
        
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

    is_fake = bool(prediction == 1)
    # credibility_score = probability of being authentic (class 0)
    credibility_score = float(probabilities[0])

    return is_fake, credibility_score


def detect_batch(texts: list, model=None) -> list:
    """
    Runs fake news detection on a batch of articles.
    
    Args:
        texts: List of article texts.
        model: The trained pipeline.
        
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
            is_fake = bool(predictions[j] == 1)
            credibility_score = float(probabilities[j][0])
            result_map[idx] = (is_fake, credibility_score)

        for i in range(len(texts)):
            if i in result_map:
                results.append(result_map[i])
            else:
                results.append((False, 0.5))
    else:
        results = [(False, 0.5)] * len(texts)

    return results


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
