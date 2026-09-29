"""
Evaluate DistilBERT Fake News Detector — Overall Accuracy & Metrics
===================================================================
Loads the saved DistilBERT model from models/distilbert_fake_news/ and
evaluates it on a held-out test split using the same dataset pipeline
that was used for training.

Reports: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
"""

import os
import sys
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Add project root to path so we can import src modules ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.intelligence.fake_news import (
    download_fake_news_dataset,
    _get_indian_news_augmentation,
    _get_indian_fake_news_augmentation,
    FakeNewsDataset,
    load_fake_news_detector,
    MODEL_PATH,
    FAKE_THRESHOLD,
)


def build_test_set(max_samples: int = 20000):
    """
    Reproduces the exact same train/test split used during training
    so that the test set has NOT been seen by the model.
    """
    texts, labels = download_fake_news_dataset()
    if texts is None:
        print("ERROR: Could not load any dataset.")
        sys.exit(1)

    # Sub-sample if needed (same seed as training)
    if len(texts) > max_samples:
        indices = np.random.RandomState(42).choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    # Augment with Indian real + fake news (mirrors training pipeline)
    aug_texts, aug_labels = _get_indian_news_augmentation()
    if aug_texts:
        texts.extend(aug_texts)
        labels.extend(aug_labels)

    fake_aug_texts, fake_aug_labels = _get_indian_fake_news_augmentation()
    if fake_aug_texts:
        texts.extend(fake_aug_texts)
        labels.extend(fake_aug_labels)

    # Same split parameters as train_fake_news_detector()
    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nTest set size: {len(X_test)} samples")
    print(f"  Class 0 (Real):  {y_test.count(0)}")
    print(f"  Class 1 (Fake):  {y_test.count(1)}")
    return X_test, y_test


def evaluate():
    """Main evaluation routine."""
    print("=" * 60)
    print("  DistilBERT Fake News Detector — Evaluation")
    print("=" * 60)

    # ── 1. Load saved model ──
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ No saved model found at: {MODEL_PATH}")
        print("   Train the model first:  python -m src.intelligence.fake_news")
        sys.exit(1)

    model, tokenizer = load_fake_news_detector()
    if model is None:
        sys.exit(1)

    device = next(model.parameters()).device

    # ── 2. Build test set (same split as training) ──
    X_test, y_test = build_test_set()

    # ── 3. Tokenize & create DataLoader ──
    print("\nTokenizing test data...")
    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ── 4. Run inference ──
    print("Running inference on test set...\n")
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── 5. Compute metrics ──
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    # -- 6. Print results --
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  Overall Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Weighted Precision: {prec:.4f}")
    print(f"  Weighted Recall   : {rec:.4f}")
    print(f"  Weighted F1-Score : {f1:.4f}")

    print(f"\n  Fake Threshold    : {FAKE_THRESHOLD}")
    print(f"  Device            : {device}")
    print(f"  Test Samples      : {len(all_labels)}")

    print("\n-- Confusion Matrix --")
    print(f"  (rows = actual, cols = predicted)")
    print(f"               Pred REAL   Pred FAKE")
    print(f"  Actual REAL    {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"  Actual FAKE    {cm[1][0]:>6}      {cm[1][1]:>6}")

    print("\n-- Per-Class Classification Report --")
    target_names = ["Real (0)", "Fake (1)"]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    print("=" * 60)
    print("  Evaluation complete.")
    print("=" * 60)

    return acc


if __name__ == "__main__":
    evaluate()
