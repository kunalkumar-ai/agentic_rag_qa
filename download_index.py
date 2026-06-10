"""Download index files from Hugging Face Hub at app startup.

Called automatically by app.py. Skips download if files already exist.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "kunalkumar-ai/rag-index"


def download_index():
    chroma_exists = Path("chroma_db").exists() and any(Path("chroma_db").iterdir())
    pkls_exist = Path("bm25_index.pkl").exists() and Path("parents.pkl").exists()

    if chroma_exists and pkls_exist:
        print("Index files already present — skipping download.")
        return

    token = os.environ.get("HF_TOKEN")
    print("Downloading index files from Hugging Face Hub…")

    for fname in ["bm25_index.pkl", "parents.pkl"]:
        if not Path(fname).exists():
            print(f"  {fname}…")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                repo_type="dataset",
                token=token,
                local_dir=".",
            )

    if not chroma_exists:
        print("  chroma_db/…")
        Path("chroma_db").mkdir(exist_ok=True)
        for file_path in list_repo_files(REPO_ID, repo_type="dataset", token=token):
            if file_path.startswith("chroma_db/"):
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file_path,
                    repo_type="dataset",
                    token=token,
                    local_dir=".",
                )

    print("Download complete.")


if __name__ == "__main__":
    download_index()
