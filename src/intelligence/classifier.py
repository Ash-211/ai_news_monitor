"""
Multi-Class News Classification Module
Trains a text classifier on the AG News dataset to categorize articles
into sectors: World, Sports, Business, Sci/Tech.
Uses TF-IDF vectorization + Logistic Regression / Naive Bayes.
(Proposal Section 5.3)
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Model save directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'news_classifier.pkl')

# AG News label mapping
AG_NEWS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


def download_ag_news():
    """
    Downloads the AG News dataset using the HuggingFace datasets library.
    
    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
    """
    try:
        from datasets import load_dataset
        print("Downloading AG News dataset from HuggingFace...")
        dataset = load_dataset("ag_news")

        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']

        print(f"AG News loaded: {len(train_texts)} train, {len(test_texts)} test samples.")
        return train_texts, train_labels, test_texts, test_labels

    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        return None, None, None, None
    except Exception as e:
        print(f"Error downloading AG News: {e}")
        return None, None, None, None


def train_classifier(max_train_samples: int = 20000):
    """
    Trains a TF-IDF + Logistic Regression pipeline on AG News.
    Evaluates multiple classifiers and saves the best one.
    
    Args:
        max_train_samples: Max number of training samples to use (for speed).
        
    Returns:
        The trained sklearn Pipeline, or None on failure.
    """
    train_texts, train_labels, test_texts, test_labels = download_ag_news()
    if train_texts is None:
        return None

    # Subsample for faster training if needed
    if max_train_samples and len(train_texts) > max_train_samples:
        indices = np.random.RandomState(42).choice(
            len(train_texts), max_train_samples, replace=False
        )
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        print(f"Subsampled to {max_train_samples} training examples for speed.")

    # Define classifiers to evaluate
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "Naive Bayes": MultinomialNB(alpha=0.1),
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, clf in classifiers.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                stop_words='english',
                sublinear_tf=True
            )),
            ('classifier', clf)
        ])

        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(test_texts)
        accuracy = accuracy_score(test_labels, predictions)
        
        label_names = [AG_NEWS_LABELS[i] for i in sorted(AG_NEWS_LABELS.keys())]
        print(f"\n{name} — Accuracy: {accuracy:.4f}")
        print(classification_report(
            test_labels, predictions,
            target_names=label_names
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


def load_classifier():
    """
    Loads a previously trained classifier from disk.
    
    Returns:
        The trained sklearn Pipeline, or None if not found.
    """
    if not os.path.exists(MODEL_PATH):
        print("No saved classifier found. Please train the model first.")
        print("Run: python -m src.intelligence.classifier")
        return None

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("News classifier loaded successfully.")
    return model


def classify_article(text: str, model=None) -> tuple:
    """
    Classifies a single article into a category.
    
    Args:
        text: The article text (cleaned or raw).
        model: The trained classifier pipeline.
        
    Returns:
        Tuple of (category_label, confidence_score)
    """
    if model is None:
        model = load_classifier()
        if model is None:
            return "Unknown", 0.0

    if not text or not isinstance(text, str) or len(text.strip()) < 5:
        return "Unknown", 0.0

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))
    
    # If confidence is very low, it's safer to say 'General'
    if confidence < 0.35:
        return "General", confidence
        
    category = AG_NEWS_LABELS.get(prediction, "Unknown")

    return category, confidence


def classify_batch(texts: list, model=None) -> list:
    """
    Classifies a batch of articles.
    
    Args:
        texts: List of article texts.
        model: The trained classifier pipeline.
        
    Returns:
        List of tuples (category_label, confidence_score)
    """
    if model is None:
        model = load_classifier()
        if model is None:
            return [("Unknown", 0.0)] * len(texts)

    results = []
    valid_indices = []
    valid_texts = []

    for i, text in enumerate(texts):
        if text and isinstance(text, str) and len(text.strip()) >= 5:
            valid_texts.append(text)
            valid_indices.append(i)
        else:
            results.append(("Unknown", 0.0))

    if valid_texts:
        predictions = model.predict(valid_texts)
        probabilities = model.predict_proba(valid_texts)

        result_map = {}
        for j, idx in enumerate(valid_indices):
            confidence = float(max(probabilities[j]))
            if confidence < 0.35:
                category = "General"
            else:
                category = AG_NEWS_LABELS.get(predictions[j], "Unknown")
            result_map[idx] = (category, confidence)

        # Rebuild results in original order
        final_results = []
        valid_ptr = 0
        for i in range(len(texts)):
            if i in result_map:
                final_results.append(result_map[i])
            else:
                final_results.append(("Unknown", 0.0))
        return final_results

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  NEWS CLASSIFIER — Training on AG News Dataset")
    print("=" * 60)
    model = train_classifier(max_train_samples=20000)
    
    if model:
        # Quick sanity test
        test_samples = [
            "Apple announces new iPhone with AI features and improved camera technology",
            "Stock market crashes as inflation data exceeds expectations",
            "India wins cricket world cup after thrilling final match",
            "President signs new healthcare reform bill into law",
        ]
        print("\n--- Sanity Check ---")
        for text in test_samples:
            cat, conf = classify_article(text, model)
            print(f"  [{cat} ({conf:.2f})] {text[:60]}...")
