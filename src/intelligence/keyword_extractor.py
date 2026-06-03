"""
Keyword Extraction Module
Extracts top-N keywords from article text using TF-IDF scoring.
(Proposal Section 5.5)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Extracts top-N keywords from a single document using TF-IDF.
    Since TF-IDF requires a corpus, we split the document into sentences
    and treat each sentence as a "document" to compute term importance.
    
    Args:
        text: The cleaned article text.
        top_n: Number of top keywords to extract.
        
    Returns:
        A list of top-N keywords sorted by TF-IDF score.
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return []

    # Split into sentences to build a mini-corpus for TF-IDF
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    if len(sentences) < 2:
        # If there's only one sentence, split by spaces and return unique words
        words = text.split()
        return list(set(words))[:top_n]

    try:
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)   # unigrams and bigrams
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores across all sentences for each term
        scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]

        keywords = [feature_names[i] for i in top_indices]
        return keywords

    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []


def extract_keywords_batch(texts: list, top_n: int = 10) -> list:
    """
    Extracts keywords for a batch of documents.
    Uses the full corpus for better TF-IDF scoring.
    
    Args:
        texts: List of cleaned article texts.
        top_n: Number of keywords per article.
        
    Returns:
        A list of lists, where each inner list contains keywords for one article.
    """
    if not texts:
        return []

    valid_texts = [t if (t and isinstance(t, str) and len(t.strip()) > 10) else "empty" for t in texts]

    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()

        all_keywords = []
        for i in range(tfidf_matrix.shape[0]):
            row = np.asarray(tfidf_matrix[i].todense()).flatten()
            top_indices = row.argsort()[-top_n:][::-1]
            keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            all_keywords.append(keywords)

        return all_keywords

    except Exception as e:
        print(f"Error in batch keyword extraction: {e}")
        return [[] for _ in texts]


if __name__ == "__main__":
    sample = (
        "Apple is reportedly looking at acquiring a UK-based artificial intelligence startup. "
        "The technology giant has been investing heavily in machine learning and deep learning research. "
        "This acquisition would strengthen Apple's position in the AI market against competitors like Google and Microsoft. "
        "The startup specializes in natural language processing and computer vision applications."
    )
    print("Sample Keywords:", extract_keywords(sample, top_n=8))
