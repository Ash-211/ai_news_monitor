import os
import sys
from huggingface_hub import HfApi, create_repo

def upload():
    token = os.getenv("HF_TOKEN")
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()

    if not token:
        print("Error: HF_TOKEN is required to upload.")
        print("Usage: python upload_model.py <YOUR_HF_WRITE_TOKEN>")
        return

    repo_id = "vinitsingare/distilbert_fake_news"
    print(f"Creating repository '{repo_id}' on Hugging Face...")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note/Warning when creating repo: {e}")

    print("Uploading local model files (models/distilbert_fake_news) to Hugging Face...")
    api = HfApi()
    api.upload_folder(
        folder_path="models/distilbert_fake_news",
        repo_id=repo_id,
        token=token,
        repo_type="model"
    )
    print("\n" + "="*60)
    print("SUCCESS! Your trained DistilBERT model is now live on Hugging Face at:")
    print(f"https://huggingface.co/{repo_id}")
    print("="*60 + "\n")

if __name__ == "__main__":
    upload()
