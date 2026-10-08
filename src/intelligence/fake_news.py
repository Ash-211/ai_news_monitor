"""
Fake News Detection Module
Binary classifier to label articles as 'Authentic' or 'Potentially Misleading'.
Outputs both a boolean label and a credibility confidence score (0.0 – 1.0).
Uses DistilBERT fine-tuned on fake news datasets.
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Model save paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'distilbert_fake_news')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Label mapping
LABEL_MAP = {
    0: False,   # Real / Authentic
    1: True     # Fake / Potentially Misleading
}

# Threshold below which an article is considered fake
FAKE_THRESHOLD = 0.40

_HF_SPACE_CLIENT = None

def call_local_fact_checker(prompt: str, json_mode: bool = True) -> str:
    """Passes the RAG prompt to the custom Hugging Face Space API and returns its response."""
    global _HF_SPACE_CLIENT
    import os
    
    hf_space_url = os.environ.get("HF_SPACE_FACTCHECK_URL")
    if not hf_space_url:
        print("[Fact-Checker] HF_SPACE_FACTCHECK_URL not found in environment! Please add it.")
        return ""
        
    try:
        from gradio_client import Client
        
        if _HF_SPACE_CLIENT is None:
            print("  [Fact-Checker] Connecting to custom Hugging Face Space LLM...")
            _HF_SPACE_CLIENT = Client(hf_space_url)
        
        response = _HF_SPACE_CLIENT.predict(
            prompt=prompt,
            api_name="/generate_text_internal"
        )
        return response.strip()
    except ImportError:
        print("[Fact-Checker] gradio-client not installed. Please run: pip install gradio-client")
        return ""
    except Exception as e:
        print(f"[HF Space API] Error generating response: {e}")
        # Reset client on error so next attempt reconnects cleanly
        _HF_SPACE_CLIENT = None
        return ""

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
    # Comprehensive Real News Scenarios (Disasters, Crime, Geopolitics, Politics, etc.)
    indian_real_samples = [
        # Politics & Economy (Original)
        "Prime Minister Narendra Modi inaugurated the new parliament building in New Delhi, marking a historic moment for Indian democracy.",
        "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent, citing stable inflation and strong GDP growth projections.",
        
        # Disasters & Tragedies (To fix the bias)
        "Delhi restaurant fire LIVE: At least 21 people killed, several foreigners among those dead in the devastating blaze.",
        "A massive magnitude 7.2 earthquake struck the northern region, causing widespread destruction and leaving hundreds dead.",
        "Floods in Assam have displaced over 50,000 residents, with the military deployed for rescue operations.",
        "Tragic train derailment in Odisha results in over 200 fatalities and 900 injured passengers. Investigation underway.",
        "Landslide in Himachal Pradesh blocks major highway, trapping tourist vehicles and causing three casualties.",
        "Tsunami warning issued for coastal areas following a massive undersea tremor in the Pacific.",
        "Building collapse in Mumbai leaves 14 dead; rescue workers are still searching through the rubble.",
        
        # Crime & Accidents
        "Three held for running illegal e-cigarette racket, vapes worth 34 lakh seized by local police.",
        "Police arrest a notorious gang leader involved in multiple bank robberies across three states.",
        "A horrific bus crash on the expressway claimed 12 lives after the driver fell asleep at the wheel.",
        "CBI raids multiple locations in connection with a multi-crore telecom scam involving senior officials.",
        "Shooting at a local mall leaves two critically injured; suspect apprehended by law enforcement.",
        "Customs officials seize 50 kg of smuggled gold at the international airport hidden in cargo shipments.",
        
        # Geopolitics & Conflict
        "Border skirmish results in casualties on both sides as military leaders agree to emergency talks.",
        "United Nations passes resolution condemning the military coup and demanding the release of political prisoners.",
        "Air strikes hit the capital city overnight, destroying key infrastructure and leaving dozens dead.",
        "Naval forces intercept a hijacked cargo ship in the Arabian sea, rescuing the entire crew safely.",
        
        # Health & Science
        "New variant of the virus detected in several cities, prompting the health ministry to issue an alert.",
        "Hospital reports a sudden spike in dengue cases, with 5 patients succumbing to the illness this week.",
        "Scientists announce a major breakthrough in nuclear fusion, potentially paving the way for clean energy."
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

    all_real = indian_real_samples * 2 + db_samples  # Repeat curated samples for balance
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
    
    all_fakes = indian_fake_samples * 2
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


class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_fake_news_detector(max_samples: int = 20000, epochs: int = 3, batch_size: int = 16):
    """
    Trains a binary fake news classifier using DistilBERT.
    """
    texts, labels = download_fake_news_dataset()
    if texts is None:
        return None, None

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

    print("\nLoading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print("Tokenizing datasets...")
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer)
    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    print(f"\n--- Training DistilBERT (Device: {device}) ---")
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_tensor = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        print(f"Average training loss: {total_loss / len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_tensor = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            correct += (predictions == labels_tensor).sum().item()
            total += labels_tensor.size(0)

    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.4f}")

    print(f"Saving model to {MODEL_PATH}")
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    return model, tokenizer


def load_fake_news_detector():
    """
    Loads the trained fake news detector (DistilBERT) directly from Hugging Face Hub.
    """
    model_name_or_path = "vinitsingare/distilbert_fake_news"
    
    print(f"Loading Model from {model_name_or_path}...")
    try:
        # Load the base tokenizer (bypasses corrupted tokenizer.json on the hub)
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        # Load the trained weights
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def generate_explanation(score: float, title: str = "", content: str = "",
                         source: str = None, verification_result: dict = None) -> str:
    """
    Generates a detailed, multi-factor natural language explanation
    for the credibility score. Analyses linguistic signals in the
    title and content to explain WHY the model scored it this way.
    """
    signals = []
    risk_factors = []
    trust_factors = []

    # ── Analyse title signals ─────────────────────────────────────────
    if title:
        title_upper_ratio = sum(1 for c in title if c.isupper()) / max(len(title), 1)
        exclamation_count = title.count('!')
        question_marks = title.count('?')
        has_all_caps_words = any(
            w.isupper() and len(w) > 2
            for w in title.split()
        )

        clickbait_phrases = [
            'you won\'t believe', 'shocking', 'breaking', 'exposed',
            'secret', 'they don\'t want you', 'one weird trick',
            'urgent', 'bombshell', 'gone wrong', 'mind blowing',
            'jaw dropping', 'must see', 'what happened next'
        ]
        title_lower = title.lower()
        found_clickbait = [p for p in clickbait_phrases if p in title_lower]

        if has_all_caps_words or title_upper_ratio > 0.5:
            risk_factors.append("excessive capitalisation in the headline (a common sensationalism tactic)")
        if exclamation_count >= 2:
            risk_factors.append(f"multiple exclamation marks ({exclamation_count}×) suggesting emotional manipulation")
        elif exclamation_count == 1:
            risk_factors.append("use of exclamation marks in the headline")
        if found_clickbait:
            risk_factors.append(f"clickbait language detected (\"{found_clickbait[0]}\")")
        if question_marks >= 2:
            risk_factors.append("heavy use of rhetorical questions (often used to imply unverified claims)")

        # Trust signals in title
        if not has_all_caps_words and exclamation_count == 0 and not found_clickbait:
            trust_factors.append("the headline uses measured, factual language consistent with professional journalism")

    # ── Analyse content signals ───────────────────────────────────────
    if content and len(content.strip()) > 50:
        word_count = len(content.split())
        avg_word_len = sum(len(w) for w in content.split()) / max(word_count, 1)
        sentence_count = max(content.count('.') + content.count('!') + content.count('?'), 1)
        avg_sentence_len = word_count / sentence_count

        # Content length assessment
        if word_count >= 300:
            trust_factors.append(f"substantial article length ({word_count} words) typical of in-depth reporting")
        elif word_count < 80:
            risk_factors.append(f"very short content ({word_count} words) — legitimate news articles are typically more detailed")

        # Vocabulary complexity
        if avg_word_len >= 5.0:
            trust_factors.append("sophisticated vocabulary usage indicating domain expertise")

        # Sentence structure
        if 15 <= avg_sentence_len <= 30:
            trust_factors.append("well-structured sentences of appropriate length for news reporting")
        elif avg_sentence_len < 8:
            risk_factors.append("unusually short, fragmented sentences often seen in viral misinformation")

        # Attribution signals
        attribution_words = ['according to', 'reported', 'said', 'stated', 'announced',
                             'confirmed', 'officials', 'spokesperson', 'study', 'research',
                             'published', 'peer-reviewed', 'data shows']
        found_attributions = [a for a in attribution_words if a in content.lower()]
        if len(found_attributions) >= 2:
            trust_factors.append(f"proper source attribution detected ({', '.join(found_attributions[:3])})")
        elif len(found_attributions) == 0 and word_count > 100:
            risk_factors.append("no source attribution or citations found in the article body")

        # Emotional language density
        emotional_words = ['horrifying', 'terrifying', 'unbelievable', 'outrageous',
                           'disgusting', 'insane', 'destroyed', 'slammed', 'blasted',
                           'fury', 'rage', 'chaos', 'panic', 'nightmare']
        content_lower = content.lower()
        emotional_count = sum(1 for w in emotional_words if w in content_lower)
        if emotional_count >= 3:
            risk_factors.append(f"high density of emotionally charged language ({emotional_count} markers)")
        elif emotional_count == 0 and word_count > 100:
            trust_factors.append("neutral, objective tone throughout the article")

    # ── Source reputation ─────────────────────────────────────────────
    if source:
        reputable_domains = [
            'bbc', 'reuters', 'ap news', 'associated press', 'nytimes',
            'washington post', 'guardian', 'ndtv', 'hindu', 'times of india',
            'indian express', 'techcrunch', 'nature', 'science', 'bbc.com',
            'reuters.com', 'nytimes.com', 'theguardian.com'
        ]
        source_lower = source.lower()
        if any(rep in source_lower for rep in reputable_domains):
            trust_factors.append(f"published by {source}, a recognised and established news outlet")

    # ── External verification ─────────────────────────────────────────
    if verification_result and isinstance(verification_result, dict):
        v_score = verification_result.get("verification_score", 0.5)
        cross_ref = verification_result.get("cross_reference", {})
        fact_check = verification_result.get("fact_check", {})

        total_outlets = cross_ref.get("total_results", 0)
        claims_found = fact_check.get("claims_found", 0)

        if v_score >= 0.7:
            if total_outlets > 5:
                trust_factors.append(f"widely corroborated — {total_outlets} other outlets are reporting the same story")
            elif total_outlets > 0:
                trust_factors.append(f"corroborated by {total_outlets} other news source(s)")
        elif v_score <= 0.3:
            if claims_found > 0:
                ratings = fact_check.get("ratings", [])
                if ratings:
                    risk_factors.append(f"professional fact-checkers have rated related claims as: {', '.join(ratings[:2])}")
                else:
                    risk_factors.append("external fact-checkers have flagged related claims")
            elif total_outlets == 0:
                risk_factors.append("no other major outlets are reporting this story, raising exclusivity concerns")



    # ── Build the explanation prompt ──────────────────────────────────
    prompt = f"Article Title: {title}\n"
    if source: prompt += f"Source: {source}\n"
    prompt += f"Transformer Grammar Score: {int(score*100)}%\n"
    if isinstance(verification_result, str):
        prompt += f"RAG Evidence: {verification_result}\n"
    
    prompt += "\nWrite a detailed, dynamic explanation of exactly what the AI models think about this article's credibility. Act as an ensemble fact-checker merging the grammar score and the fetched evidence. Your explanation must be between 4 and 6 sentences long."

    full_prompt = (
        "You are a professional AI news verification assistant. You provide detailed, analytical reasoning for credibility scores.\n\n"
        f"{prompt}"
    )
    
    explanation = call_local_fact_checker(full_prompt, json_mode=False)
    
    # Try to parse out the text if it still gave us JSON (just in case)
    try:
        import json
        parsed = json.loads(explanation)
        if isinstance(parsed, list) and len(parsed) > 0:
            item0 = parsed[0]
            if 'explanation' in item0:
                explanation = item0['explanation']
            elif 'evidence' in item0:
                ev = item0['evidence']
                explanation = " ".join(ev) if isinstance(ev, list) else str(ev)
        elif isinstance(parsed, dict):
            if 'explanation' in parsed:
                explanation = parsed['explanation']
            elif 'evidence' in parsed:
                ev = parsed['evidence']
                explanation = " ".join(ev) if isinstance(ev, list) else str(ev)
    except:
        pass
        
    # Strip markdown brackets
    import re
    explanation = re.sub(r'\[|\]|\{|\}', '', explanation)
    
    if explanation and len(explanation) >= 15:
        return explanation
        
    # Rich dynamic fallback explanation if AI model APIs are unavailable
    lines = []
    lines.append(f"This article received a credibility score of {int(score * 100)}%.")
    if trust_factors:
        lines.append(f"Key credibility indicators include: {'; '.join(trust_factors)}.")
    if risk_factors:
        lines.append(f"Potential concerns identified: {'; '.join(risk_factors)}.")
    if not trust_factors and not risk_factors:
        if score >= 0.70:
            lines.append("The content exhibits typical formatting and stylistic traits consistent with reliable news reporting.")
        elif score <= 0.40:
            lines.append("The article exhibits patterns commonly associated with sensationalized or unverified claims.")
        else:
            lines.append("The content shows a neutral mix of indicators requiring further cross-source verification.")
    
    lines.append("This assessment combines automated linguistic pattern analysis with real-time news source verification.")
    return " ".join(lines)


def detect_fake_news_local_only(title: str, content: str, model=None, tokenizer=None) -> float:
    """Calculates purely the DistilBERT linguistic grammar score locally."""
    import os
    title = title or ""
    content = content or ""
    if not content or len(content.strip()) < 10:
        content = title

    real_probability = 0.5 
    if model and tokenizer:
        import torch
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        full_text = f"{title} {content}".strip()
        
        import re
        full_text = re.sub(r'[\n\t\r]+', ' ', full_text)
        full_text = re.sub(r'(?i)Published - .*?IST', '', full_text)
        full_text = re.sub(r'(?i)Written by .*?(?=\s)', '', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            id2label = getattr(model.config, 'id2label', {})
            prob_dict = {str(v).upper(): probabilities[k].item() for k, v in id2label.items()}
            
            if "REAL" in prob_dict:
                real_probability = prob_dict["REAL"]
            elif "TRUE" in prob_dict:
                real_probability = prob_dict["TRUE"]
            elif "LABEL_0" in prob_dict:
                # 0 is Authentic / Real in distilbert_fake_news
                real_probability = prob_dict["LABEL_0"]
            elif "FAKE" in prob_dict:
                real_probability = 1.0 - prob_dict["FAKE"]
            elif "LABEL_1" in prob_dict:
                real_probability = 1.0 - prob_dict["LABEL_1"]
            else:
                real_probability = probabilities[0].item() if len(probabilities) > 0 else 0.5
                
    return max(0.01, min(1.0, real_probability))

def detect_fake_news(title: str, content: str, model=None, tokenizer=None, source: str = None, verification_result = None) -> tuple:
    """
    Single-item fake news detection: gets local score, then calls RAG ensemble API.
    """
    local_score = detect_fake_news_local_only(title, content, model, tokenizer)
    
    # We use Llama as an ensemble to merge the local score and RAG evidence
    explanation = generate_explanation(local_score, title=title, content=content, source=source, verification_result=verification_result)
    
    # Extract fact score from verification result if available
    fact_score = 0.5
    if verification_result:
        if isinstance(verification_result, dict):
            fact_score = verification_result.get("verification_score", 0.5)
        elif isinstance(verification_result, str):
            if "Cross-validated by" in verification_result:
                fact_score = 0.8
            elif "No major news outlets" in verification_result or "could not be cross-validated" in verification_result:
                fact_score = 0.2
            
    # Combine Model (33.3%) and Fact-Check (66.7%) for text-only phase
    # (This becomes 25% Model, 50% Fact-Check, 25% Image later in the pipeline)
    final_score = (local_score * 0.333) + (fact_score * 0.667)
    is_fake = bool(final_score < FAKE_THRESHOLD)
    
    breakdown = {
        "explanation_text": explanation,
        "fact_score": fact_score,
        "model_score": local_score
    }
    
    return is_fake, final_score, breakdown


def _process_batch_chunk(items: list, model=None, tokenizer=None) -> list:
    """Processes a mini-batch chunk of up to 10 articles with the Space LLM."""
    if not items:
        return items
        
    prompt_lines = []
    for i, item in enumerate(items):
        title = item.get("title", "")
        content = item.get("content", title)
        source = item.get("source", "")
        local_score = detect_fake_news_local_only(title, content, model, tokenizer)
        item["_local_score"] = local_score
        
        snippet = (content[:250] + '...') if len(content) > 250 else content
        prompt_lines.append(f"--- Article [{i}] ---")
        prompt_lines.append(f"Title: {title}")
        if source:
            prompt_lines.append(f"Source: {source}")
        if snippet:
            prompt_lines.append(f"Content: {snippet}\n")

    batch_text = "\n".join(prompt_lines)
    prompt = f"""You are an expert journalistic news credibility analyst. I am providing you with {len(items)} news articles.
Analyze each article's title, publication source, and content snippet for factual credibility, reporting tone, and plausibility.

TASK:
For each article, generate a 'fact_score' (0.0 to 1.0) and a UNIQUE 1-2 sentence explanation tailored specifically to that article:
- 0.70 to 1.00: Credible, standard news reporting from established sources, plausible events.
- 0.40 to 0.69: Neutral, unverified, or developing news.
- 0.00 to 0.39: Sensationalist, conspiracy, demonstrably false, or fabricated claims.

CRITICAL REQUIREMENT:
Each explanation MUST explicitly mention the specific entities, subject matter, and publication source of that individual article. Do NOT repeat identical boilerplate phrases across articles.

You MUST return a valid JSON array containing EXACTLY {len(items)} objects in the identical order as the input.
Format:
[
  {{"fact_score": 0.85, "explanation": "Specific explanation discussing the article's subject and source."}}
]
Do not return any markdown wrappers, just the raw JSON array.

ARTICLES:
{batch_text}
"""
    response_text = call_local_fact_checker(prompt)
    try:
        import json, re
        text = response_text.strip()
        match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if match:
            text = match.group(0)
        if text.startswith("```json"): text = text[7:-3].strip()
        if text.startswith("```"): text = text[3:-3].strip()
        
        parsed_results = json.loads(text)
        if isinstance(parsed_results, list) and len(parsed_results) == len(items):
            for i, res in enumerate(parsed_results):
                fact_score = float(res.get("fact_score", res.get("score", 0.5)))
                local_score = items[i].get("_local_score", 0.5)
                final_score = (local_score * 0.333) + (fact_score * 0.667)
                explanation = res.get("explanation")
                if not explanation and "evidence" in res:
                    ev = res["evidence"]
                    explanation = " ".join(ev) if isinstance(ev, list) else str(ev)
                if not explanation:
                    explanation = "Verified by ensemble AI."

                items[i]["analysis"] = {
                    "is_fake": bool(final_score < FAKE_THRESHOLD),
                    "credibility_score": round(final_score, 4),
                    "explanation": explanation,
                    "verdict": "Potentially Misleading" if final_score < FAKE_THRESHOLD else "Likely Authentic",
                    "fact_score": fact_score,
                    "model_score": local_score
                }
                if "_local_score" in items[i]:
                    del items[i]["_local_score"]
            return items
    except Exception as e:
        print(f"[Batch Ensemble Chunk] Chunk JSON error: {e}. Falling back to iterative processing for this chunk.")

    # Fallback for this chunk only
    for item in items:
        title = item.get("title", "")
        content = item.get("content", title)
        source = item.get("source", "")
        verification = item.get("verification", "")
        
        is_fake, final_score, breakdown = detect_fake_news(
            title=title, content=content, model=model, tokenizer=tokenizer, 
            source=source, verification_result=verification
        )
        
        item["analysis"] = {
            "is_fake": is_fake,
            "credibility_score": round(final_score, 4),
            "explanation": breakdown.get("explanation_text", ""),
            "verdict": "Potentially Misleading" if is_fake else "Likely Authentic",
            "fact_score": breakdown.get("fact_score", 0.5),
            "model_score": breakdown.get("model_score", 0.5)
        }
        if "_local_score" in item:
            del item["_local_score"]
            
    return items


def detect_batch(items: list, model=None, tokenizer=None, chunk_size: int = 10) -> list:
    """
    Batch RAG Processing: Runs linguistic scoring locally for all items, 
    then uses chunked Llama LLM calls (10 items per call) to avoid JSON truncation and speed up processing.
    Returns: List of modified items with ["analysis"] attached.
    """
    if not items:
        return []
        
    for start_idx in range(0, len(items), chunk_size):
        sub_items = items[start_idx:start_idx + chunk_size]
        _process_batch_chunk(sub_items, model=model, tokenizer=tokenizer)
        
    return items


if __name__ == "__main__":
    print("=" * 60)
    print("  FAKE NEWS DETECTOR — Training")
    print("=" * 60)
    # Use very few samples for lightning fast retraining on CPU
    model, tokenizer = train_fake_news_detector(max_samples=200, epochs=1)
    if model:
        print("\n--- Sanity Check ---")
        test_samples = [
            ("UN Report", "The United Nations released its annual report on climate change."),
            ("Delhi restaurant fire LIVE", "At least 21 people killed, several foreigners among those dead in the devastating blaze. Multiple fire tenders were rushed to the hospital.")
        ]
        for title, content in test_samples:
            is_fake, score, _ = detect_fake_news(title, content, model=model, tokenizer=tokenizer)
            status = "FAKE" if is_fake else "REAL"
            print(f"  [{status} | Credibility: {score:.2f}] {title} - {content[:45]}...")
