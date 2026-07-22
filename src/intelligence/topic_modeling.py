"""
Topic Modeling Module
Performs Latent Dirichlet Allocation (LDA) on the article corpus
to identify trending themes and assign topic clusters.
Uses scikit-learn's LDA implementation for maximum compatibility.
(Proposal Section 5.5)
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Path to save/load the trained LDA model
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
LDA_MODEL_PATH = os.path.join(MODELS_DIR, 'lda_model.pkl')
LDA_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'lda_vectorizer.pkl')


def train_lda_model(texts: list, num_topics: int = 5):
    """
    Trains an LDA model on the given texts using scikit-learn.
    
    Args:
        texts: List of cleaned article texts.
        num_topics: Number of topics to extract.
        
    Returns:
        Tuple of (lda_model, vectorizer) or (None, None) on failure.
    """
    # Filter out empty / very short texts
    valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.strip()) > 10]

    if len(valid_texts) < 2:
        print("Not enough documents to train LDA model.")
        return None, None

    print(f"Training LDA model with {num_topics} topics on {len(valid_texts)} documents...")

    # Create document-term matrix using CountVectorizer (bag of words)
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words='english',
        max_df=0.85,        # Ignore terms in >85% of docs
        min_df=2,           # Ignore terms in fewer than 2 docs
        ngram_range=(1, 1)
    )

    try:
        dtm = vectorizer.fit_transform(valid_texts)
    except ValueError as e:
        print(f"Vectorizer error (likely too few unique terms): {e}")
        # Relax constraints
        vectorizer = CountVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 1)
        )
        dtm = vectorizer.fit_transform(valid_texts)

    # Train the LDA model
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=20,
        learning_method='online',
        n_jobs=-1
    )
    lda_model.fit(dtm)

    # Save model artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LDA_MODEL_PATH, 'wb') as f:
        pickle.dump(lda_model, f)
    with open(LDA_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print("LDA model saved to models/ directory.")

    return lda_model, vectorizer


def load_lda_model():
    """
    Loads a previously trained LDA model and vectorizer from disk.
    
    Returns:
        Tuple of (lda_model, vectorizer), or (None, None) if not found.
    """
    if not os.path.exists(LDA_MODEL_PATH) or not os.path.exists(LDA_VECTORIZER_PATH):
        print("No saved LDA model found. Please train the model first.")
        return None, None

    with open(LDA_MODEL_PATH, 'rb') as f:
        lda_model = pickle.load(f)
    with open(LDA_VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("LDA model loaded successfully.")
    return lda_model, vectorizer


def get_topic_for_document(text: str, lda_model=None, vectorizer=None) -> int:
    """
    Assigns a topic cluster ID to a single document.
    
    Args:
        text: Cleaned article text.
        lda_model: Trained LDA model (loaded from disk if None).
        vectorizer: CountVectorizer (loaded from disk if None).
        
    Returns:
        Integer topic cluster ID (or -1 if assignment fails).
    """
    if lda_model is None or vectorizer is None:
        lda_model, vectorizer = load_lda_model()
        if lda_model is None:
            return -1

    if not text or not isinstance(text, str) or len(text.strip()) < 5:
        return -1

    try:
        dtm = vectorizer.transform([text])
        topic_distribution = lda_model.transform(dtm)
        dominant_topic = int(np.argmax(topic_distribution[0]))
        return dominant_topic
    except Exception as e:
        print(f"Error assigning topic: {e}")
        return -1


def get_topics_batch(texts: list, lda_model=None, vectorizer=None) -> list:
    """
    Assigns topic cluster IDs to a batch of documents.
    
    Args:
        texts: List of cleaned article texts.
        lda_model: Trained LDA model.
        vectorizer: CountVectorizer.
        
    Returns:
        List of integer topic cluster IDs.
    """
    if lda_model is None or vectorizer is None:
        lda_model, vectorizer = load_lda_model()
        if lda_model is None:
            return [-1] * len(texts)

    results = []
    for text in texts:
        topic_id = get_topic_for_document(text, lda_model, vectorizer)
        results.append(topic_id)
    return results


def print_topics(lda_model, vectorizer, num_words: int = 8):
    """
    Prints the top words for each topic for interpretability.
    """
    if lda_model is None or vectorizer is None:
        print("No model to display topics from.")
        return

    feature_names = vectorizer.get_feature_names_out()

    print("\n===== Discovered Topics =====")
    for topic_idx, topic in enumerate(lda_model.components_):
        top_word_indices = topic.argsort()[-num_words:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        print(f"  Topic {topic_idx}: {', '.join(top_words)}")
    print("=============================\n")


if __name__ == "__main__":
    # Test with sample documents
    sample_docs = [
        "stock market trading shares investors financial quarter earnings report",
        "football soccer league championship goal score match tournament",
        "artificial intelligence machine learning deep neural network data",
        "government election president congress policy reform legislation",
        "health medical vaccine hospital patient treatment clinical trial",
        "technology software startup silicon valley innovation digital platform",
        "economy inflation interest rate federal reserve monetary policy",
        "basketball nba playoffs season team draft pick player",
        "cancer research clinical study drug trial patient treatment therapy",
        "cybersecurity data breach hacking malware ransomware attack security",
    ]
    
    model, vectorizer = train_lda_model(sample_docs, num_topics=4)
    if model:
        print_topics(model, vectorizer)
        for i, doc in enumerate(sample_docs):
            topic = get_topic_for_document(doc, model, vectorizer)
            print(f"  Doc {i} ('{doc[:40]}...') -> Topic {topic}")
