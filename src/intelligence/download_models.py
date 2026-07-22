"""
Downloads ML model files from HuggingFace Hub if they are missing or corrupted locally.
This is used by the GitHub Actions cron job to get the models without relying on Git LFS.
"""
import os
import pickle

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# HuggingFace Space where models are already deployed
HF_REPO = "vinitsingare/ai-news-api"
HF_REPO_TYPE = "space"  # It's a Space, not a model repo

MODEL_FILES = [
    "news_classifier.pkl",
    "lda_model.pkl",
    "lda_vectorizer.pkl",
    "distilbert_fake_news/config.json",
    "distilbert_fake_news/model.safetensors",
    "distilbert_fake_news/special_tokens_map.json",
    "distilbert_fake_news/tokenizer_config.json",
    "distilbert_fake_news/vocab.txt"
]

def is_valid_file(filepath):
    """Check if a file is real (not a Git LFS pointer) and not corrupted."""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, 'rb') as f:
            first_bytes = f.read(20)
            # Git LFS pointer files start with "version https://git-lfs"
            if first_bytes.startswith(b'version'):
                return False
        # Try actually loading it if it's a pickle file
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                pickle.load(f)
        return True
    except Exception:
        return False


def download_models():
    """Download model files from HuggingFace Hub if they are missing or corrupted."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "distilbert_fake_news"), exist_ok=True)
    
    needs_download = False
    for fname in MODEL_FILES:
        fpath = os.path.join(MODELS_DIR, fname)
        if not is_valid_file(fpath):
            needs_download = True
            print(f"  [MODEL] {fname} is missing or corrupted (LFS pointer). Will download.")
        else:
            print(f"  [MODEL] {fname} is valid.")
    
    if not needs_download:
        print("[OK] All models are valid. No download needed.")
        return True
    
    try:
        from huggingface_hub import hf_hub_download
        
        for fname in MODEL_FILES:
            fpath = os.path.join(MODELS_DIR, fname)
            if not is_valid_file(fpath):
                print(f"  [DOWNLOAD] Downloading {fname} from HuggingFace ({HF_REPO})...")
                hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type=HF_REPO_TYPE,
                    filename=f"models/{fname}",
                    local_dir=os.path.join(MODELS_DIR, ".."),
                    local_dir_use_symlinks=False,
                )
                print(f"  [OK] {fname} downloaded successfully.")
        
        print("[OK] All models downloaded from HuggingFace Hub.")
        return True
        
    except ImportError:
        print("[WARN] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[WARN] Failed to download models from HuggingFace: {e}")
        return False


if __name__ == "__main__":
    download_models()
