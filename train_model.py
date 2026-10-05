"""
=============================================================
  DistilBERT Fine-Tuning Script — Neon DB Edition
=============================================================
  Connects to the Neon PostgreSQL database, fetches all labeled
  articles from the 'articles' table, and fine-tunes a DistilBERT
  model for fake news detection.

  Usage:
      python train_model.py

  Output:
      models/distilbert_fake_news/   (the trained model folder)

  Requirements:
      pip install transformers torch pandas psycopg2-binary scikit-learn tqdm numpy
=============================================================
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


# ==========================================
#  NEON DB CONNECTION
# ==========================================
NEON_DB_URL = (
    "postgresql://neondb_owner:npg_9otjxS2DWdsn@"
    "ep-falling-butterfly-axujl4ft-pooler.c-4.us-east-2.aws.neon.tech/"
    "neondb?sslmode=require"
)

# ==========================================
#  TRAINING CONFIG
# ==========================================
EPOCHS = 4                # 4 passes (approx 12-15 mins total)
BATCH_SIZE = 16           # Back to 16 since we reduced sequence length
MAX_SEQ_LENGTH = 512      # Increased to 512 for full article context
LEARNING_RATE = 2e-5      # Adjusted for RoBERTa architecture
TEST_SPLIT = 0.2          # 80% train, 20% test

# Output path — same folder the project already loads from
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "distilbert_fake_news")


# ==========================================
#  STEP 1: FETCH DATA FROM NEON DB
# ==========================================
def fetch_dataset():
    """
    Connects to Neon DB and fetches all labeled articles.
    Only articles where 'is_fake' is NOT NULL are used (i.e., labeled data).
    
    Uses RAW content (not cleaned) so the model can see punctuation,
    capitalization, and numbers — critical signals for fake news detection.
    
    Applies stratified downsampling to balance real/fake classes.
    
    Returns:
        texts (list[str]), labels (list[int])
        label 0 = Real/Authentic
        label 1 = Fake/Misleading
    """
    print("=" * 60)
    print("  STEP 1: Fetching dataset from Neon DB")
    print("=" * 60)

    try:
        engine = create_engine(NEON_DB_URL)
        print("  Connected to Neon DB successfully!\n")

        # FIX: Use raw_content FIRST (not clean_content) so the model
        # can see punctuation, caps, numbers — key fake news signals
        query = """
            SELECT title, 
                   COALESCE(raw_content, clean_content, '') AS content,
                   is_fake,
                   COALESCE(category, 'General') AS category
            FROM articles
            WHERE is_fake IS NOT NULL
        """

        df = pd.read_sql_query(query, engine)
        engine.dispose()

        if df.empty:
            print("  No labeled articles found in the database!")
            print("    Make sure your articles have 'is_fake' set to TRUE or FALSE.")
            return None, None

        # Combine title + content for richer features
        df["full_text"] = df["title"].fillna("") + " " + df["content"].fillna("")

        # Remove rows with very little text
        df = df[df["full_text"].str.strip().str.len() > 20]

        print("  Applying Anti-Bias Text Normalization (Removing Ollama formatting)...")
        import re
        def normalize_text(text):
            if not isinstance(text, str):
                return ""
            # Remove all newlines and tabs (destroys Ollama \n\n formatting bias)
            text = re.sub(r'[\n\t\r]+', ' ', text)
            # Remove standard news metadata tags that might leak reality
            text = re.sub(r'(?i)Published - .*?IST', '', text)
            text = re.sub(r'(?i)Written by .*?(?=\s)', '', text)
            # Normalize spaces
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        df["full_text"] = df["full_text"].apply(normalize_text)

        # Convert boolean is_fake to integer labels: False->0 (Real), True->1 (Fake)
        df["label"] = df["is_fake"].astype(int)

        real_count = (df["label"] == 0).sum()
        fake_count = (df["label"] == 1).sum()

        print(f"  Total labeled articles fetched: {len(df)}")
        print(f"    -> Real (is_fake=false):  {real_count}")
        print(f"    -> Fake (is_fake=true):   {fake_count}")

        if real_count == 0 or fake_count == 0:
            print("\n  WARNING: You need BOTH real and fake articles to train!")
            print("    The model requires examples of both classes.")
            return None, None

        # ── Compute Class Weights for Loss Function ───────────────
        real_weight = len(df) / (2.0 * real_count) if real_count > 0 else 1.0
        fake_weight = len(df) / (2.0 * fake_count) if fake_count > 0 else 1.0
        weights = [real_weight, fake_weight]
        print(f"\n  Computed Class Weights (to handle imbalance):")
        print(f"    -> Real Weight: {real_weight:.4f}")
        print(f"    -> Fake Weight: {fake_weight:.4f}")

        # Combine balanced dataset (Shuffle only)
        df_balanced = df.sample(frac=1, random_state=42)

        texts = df_balanced["full_text"].tolist()
        labels = df_balanced["label"].tolist()

        return texts, labels, weights

    except Exception as e:
        print(f"  Database connection failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==========================================
#  STEP 2: PYTORCH DATASET CLASS
# ==========================================
class FakeNewsDataset(Dataset):
    """Wraps tokenized texts + labels into a PyTorch Dataset."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQ_LENGTH):
        print(f"    Tokenizing {len(texts)} samples (max_length={max_length})...")
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ==========================================
#  STEP 3: TRAINING
# ==========================================
def train():
    # --- Fetch Data ---
    fetch_result = fetch_dataset()
    if fetch_result == (None, None):
        print("\nTraining aborted. Fix the issues above and try again.")
        sys.exit(1)
    texts, labels, class_weights = fetch_result

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels
    )

    print(f"\n  Train set: {len(X_train)} articles")
    print(f"  Test set:  {len(X_test)} articles")

    # --- Load Base Model ---
    print("\n" + "=" * 60)
    print("  STEP 2: Loading DistilRoBERTa base model")
    print("=" * 60)
    tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "distilroberta-base", num_labels=2
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"  Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  TIP: Training on CPU will be slower. A GPU is recommended for 8000+ articles.")
    model.to(device)

    # --- Tokenize ---
    print("\n  Preparing datasets...")
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer)
    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- Learning Rate Scheduler ---
    total_steps = len(train_loader) * EPOCHS
    # Warm up for the first 10% of training steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # Apply class weights to handle imbalance
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    print(f"\n  Using weighted CrossEntropyLoss (Weights: {class_weights})")

    # --- Training Loop with Early Stopping ---
    print("\n" + "=" * 60)
    print("  STEP 3: Fine-tuning RoBERTa with Early Stopping")
    print("=" * 60)

    best_val_accuracy = 0
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print(f"\n  Epoch {epoch + 1}/{EPOCHS}")

        for batch in tqdm(train_loader, desc="  Training", ncols=80):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels_tensor)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

        avg_loss = total_loss / len(train_loader)
        
        # Validation step
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_tensor = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels_tensor.cpu().numpy())
                
        val_accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        print(f"  Average loss: {avg_loss:.4f} | Val Accuracy: {val_accuracy*100:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            print(f"  [+] Validation accuracy improved ({best_val_accuracy*100:.2f}% -> {val_accuracy*100:.2f}%). Saving checkpoint...")
            best_val_accuracy = val_accuracy
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # --- Final Evaluation on Best Model ---
    print("\n" + "=" * 60)
    print("  STEP 4: Final Evaluation on Best Model")
    print("=" * 60)
    
    model = RobertaForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Evaluating", ncols=80):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())

    final_accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n  [SUCCESS] Best Test Accuracy: {final_accuracy * 100:.2f}%\n")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Real (Authentic)", "Fake (Misleading)"]
    ))

    print(f"\n  [SUCCESS] Best model saved to: {MODEL_SAVE_PATH}")
    for f in os.listdir(MODEL_SAVE_PATH):
        size = os.path.getsize(os.path.join(MODEL_SAVE_PATH, f))
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
        print(f"      {f} ({size_str})")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("  Your project will now automatically use this new model.")
    print("=" * 60)

    # --- Quick Sanity Check ---
    print("\n  Running sanity check on sample articles...\n")
    test_articles = [
        ("UN Climate Report", "The United Nations released its annual report on climate change, highlighting rising global temperatures."),
        ("SHOCKING CURE", "Doctors don't want you to know this ONE WEIRD TRICK that cures all diseases overnight! Big pharma is TERRIFIED!"),
        ("Delhi Fire", "At least 21 people killed in a devastating restaurant fire in Delhi. Multiple fire tenders were rushed to the scene."),
    ]

    model.eval()
    for title, content in test_articles:
        inputs = tokenizer(
            title + " " + content,
            return_tensors="pt", truncation=True, padding=True, max_length=MAX_SEQ_LENGTH
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            real_prob = probs[0][0].item()
            label = "REAL" if real_prob >= 0.5 else "FAKE"
            print(f"  [{label} | Credibility: {real_prob:.2f}] {title}")

    print()


if __name__ == "__main__":
    train()
