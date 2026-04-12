import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.ingestion.database import get_session, Article

# Download required NLTK components
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load or download the spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize global tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text: str, apply_stemming: bool = False, apply_lemmatization: bool = True) -> str:
    """
    Cleans raw text data using a standard NLP pipeline.
    Includes lowercasing, punctuation removal, stopword removal, stemming, and lemmatization.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()
    
    # 2. Punctuation removal
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Noise removal (numbers and extra whitespace)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Tokenization & Stopword removal (NLTK)
    tokens = [word for word in text.split() if word not in stop_words]
    
    # 5. Stemming (Porter)
    if apply_stemming and not apply_lemmatization:
        tokens = [stemmer.stem(word) for word in tokens]
        return " ".join(tokens)
        
    # 6. Lemmatization (spaCy)
    if apply_lemmatization:
        doc = nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc]
        return " ".join(lemmas)
        
    return " ".join(tokens)

def process_uncleaned_articles():
    """
    Fetches raw articles from the database, applies the NLP cleaning pipeline,
    and saves the cleaned text back to the database.
    """
    session = get_session()
    
    try:
        # Fetch articles that haven't been cleaned yet
        uncleaned_articles = session.query(Article).filter(
            (Article.clean_content == None) | (Article.clean_content == "")
        ).all()
        
        print(f"Found {len(uncleaned_articles)} articles pending NLP preprocessing...")
        
        processed_count = 0
        for article in uncleaned_articles:
            if article.raw_content:
                cleaned = clean_text(article.raw_content)
                article.clean_content = cleaned
                processed_count += 1
                
        session.commit()
        print(f"Successfully cleaned and updated {processed_count} articles.")
        return processed_count
        
    except Exception as e:
        session.rollback()
        print(f"Error during preprocessing: {e}")
        return 0
    finally:
        session.close()

if __name__ == "__main__":
    # Test on a dummy string
    sample = "Apple IS looking at buying U.K. startup for $1 billion! 123"
    print("Original:", sample)
    print("Cleaned:", clean_text(sample))
    
    # Run bulk processing
    print("\nRunning database preprocessing...")
    process_uncleaned_articles()
