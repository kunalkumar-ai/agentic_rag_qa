"""One-time script to upload index files to Hugging Face Hub.

Run this once from your local machine:
    python3 upload_index.py
"""

from huggingface_hub import HfApi

REPO_ID = "kunalkumar-ai/rag-index"

api = HfApi()

print("Creating private dataset repo…")
api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)

print("Uploading chroma_db/ …")
api.upload_folder(
    folder_path="chroma_db",
    path_in_repo="chroma_db",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print("Uploading bm25_index.pkl …")
api.upload_file(
    path_or_fileobj="bm25_index.pkl",
    path_in_repo="bm25_index.pkl",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print("Uploading parents.pkl …")
api.upload_file(
    path_or_fileobj="parents.pkl",
    path_in_repo="parents.pkl",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print("Done. Index files are on Hugging Face Hub.")
